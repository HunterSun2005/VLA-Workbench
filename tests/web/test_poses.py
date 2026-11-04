from dataclasses import replace
import json
from types import SimpleNamespace as NS
import threading
import time
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vla.web.app import create_app
from vla.web.config import SessionConfig
from vla.web.demo import DemoBackend
from vla.web.poses import PoseStore, validate_pose
from vla.web.pose_control import (PoseActions, FrankyPoseDriver, ROSPoseDriver,
                                  pose_value, compatible, check_bounds)
from vla.web.runtime import Runtime
from tests.rlt.test_gamepad_teleop import rig


def sample(arm='left'):
    return pose_value(arm, 'robot', 'base/EE', [.4,0,.3], np.eye(3), [0,0,.1], np.eye(3))


def test_library_persistence_validation_atomicity_and_api(tmp_path, monkeypatch):
    store = PoseStore(tmp_path)
    r = store.add('  桌面起点  ', sample())
    assert PoseStore(tmp_path).get(r['id'])['note'] == '桌面起点'
    assert PoseStore(tmp_path, demo=True).list() == []
    with pytest.raises(ValueError):
        validate_pose({**sample(), 'gripper': 0})
    with pytest.raises(ValueError):
        store.add('bad', {**sample(), 'rotation':np.diag([1,1,-1]).tolist()})
    with pytest.raises(ValueError):
        store.add('bad', {**sample(), 'position':[float('nan'),0,0]})
    original = store.path.read_bytes()
    with monkeypatch.context() as m:
        m.setattr('vla.web.poses.os.replace', lambda *_: (_ for _ in ()).throw(OSError('disk failure')))
        with pytest.raises(OSError):
            store.add('failed', sample())
    assert store.path.read_bytes() == original
    with TestClient(create_app(tmp_path), base_url='http://127.0.0.1') as client:
        path = '/api/poses/'+r['id']
        assert client.post(path, json={'operation':'delete'}).status_code == 403
        headers = {'x-vla-token':client.get('/api/bootstrap').json()['token']}
        assert client.post(path, json={'operation':'rename','note':'新位置'}, headers=headers).status_code == 200
        assert client.get('/api/poses').json()['records'][0]['note'] == '新位置'
        assert client.post(path, json={'operation':'delete'}, headers=headers).status_code == 200
        assert client.get('/api/poses').json()['records'] == []
    store.path.write_text('{bad json')
    with pytest.raises(ValueError):
        store.add('never overwrite corruption', sample())
    assert store.path.read_text() == '{bad json'


@pytest.mark.parametrize('arms,source', [('single_left','gamepad'),('single_left','gello'),('dual','gamepad'),('dual','gello')])
def test_demo_persist_restore_pause_dedup_and_recording_guard(tmp_path, arms, source):
    config = SessionConfig(arms=arms,source=source)
    backend = DemoBackend(config,tmp_path/'session1',tmp_path)
    runtime = Runtime('session1',config,tmp_path,backend)
    selected = 'right' if arms == 'dual' else 'left'
    def send(operation, args=None):
        msg = dict(kind='command', session_id='session1', command_id=str(uuid.uuid4()), revision=runtime.revision,
                   operation=operation, args=args or {})
        runtime.rpc(msg)
        runtime.poll_command()
        runtime.rpc(msg)  # Retries never repeat a capture or start.
        assert runtime.results[msg['command_id']]['status'] == 'completed'
    send('pose_capture', {'arm':selected,'note':'带备注起点'})
    records = backend.poses.store.list()
    assert len(records) == 1 and records[0]['pose']['arm'] == selected
    assert 'gripper' not in json.dumps(records)
    r = records[0]
    backend.command('enable',{})
    with pytest.raises(RuntimeError, match='待命'):
        backend.command('pose_restore',{'pose_id':r['id']})
    backend.command('record',{})
    with pytest.raises(RuntimeError):
        backend.command('pose_capture',{'arm':selected,'note':'no'})
    backend.command('discard',{})
    send('pose_restore', {'pose_id':r['id']})
    assert runtime.pending_completion is None  # Pause stays available during motion.
    assert backend.sample()['actual_source'] == 'pose_return'
    with pytest.raises(RuntimeError):
        backend.command('record',{})
    send('pause')
    assert backend.poses.status['state'] == 'cancelled'
    second = DemoBackend(config,tmp_path/'session2',tmp_path)
    second.command('pose_restore',{'pose_id':r['id']})
    second.poses.driver.begin -= 1
    second.tick()
    assert second.poses.status['state'] == 'completed'
    assert second.sample()['state'] == 'standby'
    assert second.poses.driver.read(selected) == r['pose']
    assert not second.recording


def test_identity_tool_workspace_rejection():
    current = sample()
    for changed in ({'device':'other'}, {'arm':'right'}, {'frame':'other'},
                    {'tool':{'position':[.1,0,0],'rotation':np.eye(3).tolist()}}):
        with pytest.raises(ValueError):
            compatible({**current,**changed}, current)
    with pytest.raises(ValueError, match='工作空间'):
        check_bounds(current, [0,0,0], [.2,.2,.2])


def test_franky_restores_exact_rotation_no_gripper_and_cancels_before_input(rig, tmp_path):
    from scipy.spatial.transform import Rotation
    rig.arm.state.F_T_EE = np.eye(4)
    rig.advance()
    config = SessionConfig()
    backend = NS(config=config, arms=['left'], session=NS(controller=rig.app))
    backend.sample = lambda: {'state':rig.app.mode,'enabled':rig.app.mode=='active'}
    actions = PoseActions(backend,tmp_path)
    actions.command('pose_capture',{'arm':'left','note':'初始位置'})
    saved = actions.store.list()[0]
    rig.arm.state.O_T_EE[:3,3] += [.04,0,0]
    rig.arm.state.O_T_EE[:3,:3] = Rotation.from_euler('z',30,degrees=True).as_matrix()
    rig.advance()
    actions.command('pose_restore',{'pose_id':saved['id']})
    np.testing.assert_allclose(rig.app._pose_return.rotation, saved['pose']['rotation'])
    actions.tick()
    rig.advance()
    debug = rig.robot.command_debug_snapshot()
    assert np.linalg.norm(debug['commanded_linear_velocity']) <= .030001
    assert np.linalg.norm(debug['commanded_angular_velocity']) <= .150001
    assert rig.gripper.target is None  # No gripper motion was requested during the return.
    rig.receiver.packet = replace(rig.receiver.packet, axes=(0,0,-1,0,0,1))
    actions.tick()
    assert actions.status['state'] == 'cancelled' and rig.app.mode == 'standby'
    assert rig.gripper.target == .75
    rig.receiver.packet = replace(rig.receiver.packet, axes=(0,0,1,0,0,1))
    rig.advance()
    actions.command('pose_restore',{'pose_id':saved['id']})
    rig.arm.state.O_T_EE[:3,3] = saved['pose']['position']
    rig.arm.state.O_T_EE[:3,:3] = saved['pose']['rotation']
    actions.tick()
    assert actions.status['state'] == 'completed' and rig.app.mode == 'standby'
    rig.advance()
    actions.command('pose_restore',{'pose_id':saved['id']})
    rig.receiver.error = TimeoutError('stale Joy')
    actions.tick()
    assert actions.status['state'] == 'failed' and rig.app.mode == 'standby'


def ros_backend(source, arm='left'):
    from vla.datacollection.franka_robotiq_rollout.safety import FrankaHealthMonitor
    config = SessionConfig(arms='dual',source=source)
    b = NS(config=config, arms=['left','right'], directory=None, collector=None, enabled=False,
           hardware={'LEFT_ROBOT_IP':'robot','RIGHT_ROBOT_IP':'robot2'}, _health_stamps={}, health=FrankaHealthMonitor(.9,3))
    from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
    rollout = load_rollout_config(config.default_config())
    b.rollout_config = lambda: rollout
    def stamped(x,y,z):
        return NS(pose=NS(position=NS(x=x,y=y,z=z),orientation=NS(x=0,y=0,z=0,w=1)))
    state = NS(o_t_ee=stamped(.4,0,.3),f_t_ee=stamped(0,0,.1),robot_mode=1,
               ROBOT_MODE_IDLE=1,ROBOT_MODE_MOVE=2,current_errors=NS(),control_command_success_rate=1.)
    now = time.monotonic()
    latest = {f'{arm}.robot_state':(int(now*1e9),state),
              f'{arm}.measured_joint':(int(now*1e9),NS(position=np.zeros(7)))}
    node = NS(_lock=threading.Lock(), _latest=latest)
    model = NS(joint_names=[f'q{i}' for i in range(7)],lower_limits=np.full(7,-3),upper_limits=np.full(7,3))
    messages = []
    if source == 'gamepad':
        node.config = rollout
        node._measured_joints = lambda a,m:np.array(m.position)
        node._faulted = False
        node._neutral_gamepads = lambda:True
        node.servos = {arm:NS(model=model)}
        node._publish_arm_command = lambda a,q,g,**kw:messages.append((a.name,q.copy(),kw))
        b.collector = node
    else:
        b.recorder = node
        arbiter = NS(managed_fault=None, measured_joints=np.zeros(7), measured_joint_time=now,source_timeout=.5,
                     model=model,joint_publisher=NS(publish=messages.append),get_clock=lambda:NS(now=lambda:NS(to_msg=lambda:None)))
        b.arbiters = [arbiter,arbiter]
    return b, latest, messages


@pytest.mark.parametrize('source,arm',[('gamepad','left'),('gamepad','right'),('gello','left'),('gello','right')])
def test_ros_real_adapter_freshness_solver_joint_only_and_hold(monkeypatch, source, arm):
    import sys
    b, latest, messages = ros_backend(source,arm)
    class Servo:
        def __init__(self,model,config):
            self.model,self.config = model,config
        def step(self,q,p,r,tp,tr,dt):
            assert self.config.max_joint_speed <= .15 and self.config.max_linear_velocity <= .03
            return q + dt*.05
        def forward_tcp(self,q,tp,tr):
            return np.array([.4,0,.3]),np.eye(3)
    monkeypatch.setitem(sys.modules,'vla.datacollection.franka_robotiq_rollout.collect_ros',NS(CartesianServo=Servo))
    monkeypatch.setitem(sys.modules,'sensor_msgs.msg',NS(JointState=lambda:NS(header=NS())))
    driver = ROSPoseDriver(b)
    pose = driver.read(arm)
    target = {**pose,'position':[.45,0,.3]}
    driver.start(target)
    driver.last -= .015
    driver.tick(target)
    assert len(messages) == 1
    if source == 'gamepad':
        assert messages[0][0] == arm and messages[0][2] == {'publish_gripper':False}
    else:
        assert len(messages[0].position) == 7
    driver.stop()
    assert len(messages) == 2
    stale = latest[f'{arm}.robot_state']
    latest[f'{arm}.robot_state'] = (stale[0]-int(1e9),stale[1])
    with pytest.raises(RuntimeError,match='过期'):
        driver.read(arm)


def test_ros_reflex_and_timeout_rejected(monkeypatch):
    b,latest,_ = ros_backend('gello')
    driver = ROSPoseDriver(b)
    latest['left.robot_state'][1].robot_mode = 4
    with pytest.raises(RuntimeError,match='IDLE/MOVE'):
        driver.read('left')
    driver.last = time.monotonic()-1
    with pytest.raises(RuntimeError,match='超时'):
        driver.tick(sample())


def test_saved_return_cancels_on_packet_arriving_after_adapter_check(rig):
    from vla.datacollection.start_teleop.gamepad_teleop import PoseReturn
    rig.advance()
    rig.app.enable()
    rig.app._pose_return = PoseReturn('saved pose', np.array([.5,-.2,.3]), np.eye(3), rig.clock[0]+60)
    moves = len(rig.arm.moves)
    rig.receiver.packet = replace(rig.receiver.packet, axes=(0,0,-1,0,0,1))
    rig.advance()  # No adapter precheck: a newly arrived trigger must still cancel.
    assert rig.app.mode == 'standby' and rig.app._pose_return is None
    assert len(rig.arm.moves) == moves
    assert rig.gripper.target == .75
