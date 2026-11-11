"""Production input integration and coordinator, independent real-wheel fake-Robot process."""
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
from vla.datacollection.franka_robotiq_rollout.collect_ros import RolloutCollector, GamepadTarget
from vla.datacollection.franka_robotiq_rollout.gamepad import decode_cartesian_gamepad_values, gamepad_is_neutral
from vla.datacollection.franka_robotiq_rollout.gamepad_layout import axis_buttons, node_for_layout
from vla.control.ros import ControlCoordinator, ArmMeasurement
from tests.control.test_franky_process import process_worker, request

ROOT = Path(__file__).resolve().parents[2]


def rig(profile):
    c = RolloutCollector.__new__(RolloutCollector)
    c.config = load_rollout_config(ROOT/f'src/vla/datacollection/franka_robotiq_rollout/configs/{profile}.yaml')
    c._lock = threading.RLock()
    c._gamepad_targets, c._previous_buttons, c._initial_pose = {}, {}, {}
    c._high_speed, c._speed_scale = defaultdict(bool), defaultdict(lambda:1.)
    c._recording = False
    c._latest, c._input_diagnostics = {}, {}
    c._update_gamepad_gripper = lambda a,t,g,m: setattr(t,'gripper', np.clip(t.gripper+m*.01,0,1))
    c._publish_gripper_speed = lambda *args: None
    c.coordinator = ControlCoordinator(c.config.action_spec, {},
        {a.name:(np.array(a.workspace_min),np.array(a.workspace_max)) for a in c.config.arms},
        lambda *a: None, lambda *a: None, backend='franky')
    for a in c.config.arms:
        measured = Rotation.from_euler('xyz',[.3,-.4,.8]).as_matrix()
        ref = Rotation.from_euler('xyz',[.35,-.4,.8]).as_matrix()
        p = np.array([.4,0,.3])
        c._gamepad_targets[a.name] = GamepadTarget(p.copy(),ref.copy(),.5,np.zeros(7),np.zeros(3),np.eye(3))
        c._initial_pose[a.name] = (p.copy(),ref.copy())
        robot = NS(o_t_ee=c._worker_pose([*p, *Rotation.from_matrix(measured).as_quat()]))
        now = time.monotonic_ns()
        for key,value in [('robot_state',robot),('gamepad',NS(axes=[0.,0.,0.,0.,0.,0.],buttons=[0]*21)),
                          ('gripper_state',NS(name=[a.gripper_joint_name],position=[.4])),
                          ('measured_joint',NS(name=[f'fr3_joint{i}' for i in range(1,8)],position=[0.]*7))]:
            c._latest[f'{a.name}.{key}']=(now,value)
        c.coordinator.observe(a.name, ArmMeasurement(np.zeros(7), .5, np.zeros(3),np.eye(3), now))
    return c


@pytest.mark.parametrize('profile',['single_left','dual'])
@pytest.mark.parametrize('axis',[0,1,2,3])
@pytest.mark.parametrize('sign',[-1,1])
def test_each_stick_direction_and_actual_tool_rotation(profile,axis,sign):
    c=rig(profile)
    values={k:v[1] for k,v in c._latest.items()}
    arm=c.config.arms[0]; target=c._gamepad_targets[arm.name]
    p,r=target.position.copy(),target.rotation.copy()
    values[f'{arm.name}.gamepad'].axes[axis]=sign
    assert c._publish_gamepad(values,.01)
    if axis<2:
        delta=target.position-p
        assert delta[1-axis]*sign>0
        np.testing.assert_allclose(np.delete(delta,1-axis),0,atol=1e-12)
        np.testing.assert_allclose(target.rotation,r,atol=1e-12)
    else:
        decoded=decode_cartesian_gamepad_values(values[f'{arm.name}.gamepad'].axes,[0]*21,arm.gamepad,c.config)
        measured=Rotation.from_euler('xyz',[.3,-.4,.8]).as_matrix()
        expected=Rotation.from_rotvec(measured@decoded.angular_velocity_tool*.01).as_matrix()@r
        np.testing.assert_allclose(target.rotation,expected,atol=1e-12)
        assert Rotation.from_matrix(target.rotation@r.T).magnitude()>0
        np.testing.assert_allclose(target.position,p,atol=1e-12)
    held_p,held_r=target.position.copy(),target.rotation.copy()
    values[f'{arm.name}.gamepad'].axes[axis]=0
    c._publish_gamepad(values,.01)
    np.testing.assert_allclose(target.position,held_p)
    np.testing.assert_allclose(target.rotation,held_r)


def test_standard_buttons_raw_profile_and_workspace():
    c=rig('single_left'); a=c.config.arms[0]; m=a.gamepad
    values={k:v[1] for k,v in c._latest.items()}; joy=values['left.gamepad']
    assert node_for_layout(m.input_layout)=='game_controller_node'
    assert gamepad_is_neutral(joy.axes,joy.buttons,m,c.config)
    raw=replace(m, input_layout='raw_joy', **axis_buttons('raw_joy'))
    assert node_for_layout(raw.input_layout)=='joy_node'
    assert gamepad_is_neutral([0,0,1,0,0,1],[0]*11,raw,c.config)
    # Logical shoulder controls no longer overlap SDL Back/Start buttons.
    joy.buttons[m.speed_button]=1;c._publish_gamepad(values,.01)
    assert c._high_speed['left']
    joy.buttons[m.speed_button]=0;joy.buttons[m.upright_button]=1
    c._publish_gamepad(values,.01)
    np.testing.assert_allclose(c._gamepad_targets['left'].rotation[:,2],[0,0,-1],atol=1e-12)
    joy.buttons=[0]*21;joy.buttons[m.reset_pose_button]=1;c._publish_gamepad(values,.01)
    np.testing.assert_allclose(c._gamepad_targets['left'].rotation,c._initial_pose['left'][1])
    joy.buttons=[0]*21;joy.axes[1]=1
    c._gamepad_targets['left'].position[0]=a.workspace_max[0]
    c._publish_gamepad(values,.01)
    assert c._gamepad_targets['left'].position[0]==a.workspace_max[0]
    assert c.input_diagnostics()['left']['workspace_clipped'][0]
    joy.axes[1]=0;joy.axes[4]=-1
    c._publish_gamepad(values,.01)
    assert c._gamepad_targets['left'].gripper>.5


@pytest.mark.parametrize('profile',['single_left','dual'])
@pytest.mark.parametrize('source',['gamepad','gello','policy'])
def test_input_through_coordinator_to_real_tracker_fake_process(profile,source):
    c=rig(profile); names=tuple(a.name for a in c.config.arms)
    with process_worker(names) as (root,child,argv):
        (root/'capture-targets').touch()
        c.coordinator.request_reset=lambda gen: request(root,'reset',gen)
        gen=c.coordinator.transition(source)
        pending={};sequence=0
        def publish(name,pose,gripper,generation,command_id):
            nonlocal sequence
            pending[name]={'pose':pose.tolist()}
            if set(pending)==set(names):
                sequence+=1
                response=request(root,'target',generation,sequence=sequence,timestamp_ns=time.monotonic_ns(),arms=dict(pending))
                assert not response.get('fault'),response
                pending.clear()
        c.coordinator.publish=publish
        for a in c.config.arms:
            now=time.monotonic_ns()
            c.coordinator.observe(a.name,ArmMeasurement(np.zeros(7),.5,np.zeros(3),np.eye(3),now))
            assert c.coordinator.acknowledge(a.name,gen,0,np.zeros(7),now)
        values={k:v[1] for k,v in c._latest.items()}
        values['left.gamepad'].axes[1]=1
        c._publish_gamepad(values,.01)
        action=c._input_action
        if source=='gello':
            c.coordinator.servos={name:NS(reset_tracking=lambda:None, forward_tcp=lambda q,tp,tr:(np.array([.41,.02,.3]),np.eye(3))) for name in names}
            action=c.coordinator.gello_target({n:np.zeros(7) for n in names},{n:.5 for n in names})
        for slot in range(32):
            c.coordinator.accept(action,source=source,generation=gen,slot_id=slot)
            c.coordinator.tick(1/30)
        captures=[json.loads(line) for line in (root/'targets.jsonl').read_text().splitlines()]
        assert len(captures)==32*len(names)
        for index in range(len(names)):
            expected=action[index*c.config.action_spec.width:][:3]
            np.testing.assert_allclose(captures[-len(names)+index]['translation'],expected,atol=1e-6)
        stopped=request(root,'stop',gen+1)
        assert not any(s['tracker_running'] for s in stopped['states'])
        stale=request(root,'target',gen,sequence=999,timestamp_ns=time.monotonic_ns(),arms={n:{'pose':[.4,0,.3,0,0,0,1]} for n in names})
        assert stale['request_status']=='ignored'


def test_standard_rejects_raw_trigger_shape_and_all_button_roles():
    c=rig('single_left');a=c.config.arms[0];m=a.gamepad
    with pytest.raises(ValueError,match='standard_sdl'):
        gamepad_is_neutral([0,0,1,0,0,1],[0]*21,m,c.config)
    with pytest.raises(ValueError,match='invalid gamepad'):
        decode_cartesian_gamepad_values([0,1.1,0,0,0,0],[0]*21,m,c.config)
    for button,linear_axis,sign in [(m.a_button,2,-1),(m.y_button,2,1)]:
        buttons=[0]*21;buttons[button]=1
        decoded=decode_cartesian_gamepad_values([0]*6,buttons,m,c.config)
        assert decoded.linear_velocity_base[linear_axis]*sign>0
        np.testing.assert_allclose(decoded.angular_velocity_tool,0)
    for button,sign in [(m.x_button,-1),(m.b_button,1)]:
        buttons=[0]*21;buttons[button]=1
        decoded=decode_cartesian_gamepad_values([0]*6,buttons,m,c.config)
        assert decoded.angular_velocity_tool[2]*sign>0
        np.testing.assert_allclose(decoded.linear_velocity_base,0)
    values={k:v[1] for k,v in c._latest.items()}
    c._recording=True
    toggles=[];c.toggle_phase=lambda:toggles.append(True)
    values['left.gamepad'].buttons[m.phase_button]=1
    c._publish_gamepad(values,.01);c._publish_gamepad(values,.01)
    assert toggles==[True]
