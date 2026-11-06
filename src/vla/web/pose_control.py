"""Pre-recording pose operations, executed exclusively by the runner control loop."""
from __future__ import annotations
from dataclasses import replace
import time
import numpy as np
from .poses import PoseStore, note_text, validate_pose


def pose_value(arm, device, frame, position, rotation, tool_position, tool_rotation):
    return validate_pose({'arm':arm, 'device':device, 'frame':frame,
                         'position':np.asarray(position).tolist(), 'rotation':np.asarray(rotation).tolist(),
                         'tool':{'position':np.asarray(tool_position).tolist(), 'rotation':np.asarray(tool_rotation).tolist()}})


def check_bounds(pose, minimum, maximum):
    xyz = np.asarray(pose['position'])
    if np.any(xyz < minimum) or np.any(xyz > maximum):
        raise ValueError(f"{pose['arm']} TCP XYZ={xyz.round(4).tolist()} 超出工作空间 {list(minimum)} → {list(maximum)}")


def compatible(saved, current):
    for key in ('arm', 'device', 'frame'):
        if saved[key] != current[key]:
            raise ValueError('记录与当前机械臂设备／坐标系不匹配')
    for key in ('position', 'rotation'):
        if not np.allclose(saved['tool'][key], current['tool'][key], atol=1e-5, rtol=0):
            raise ValueError('TCP 工具定义已变化，请重新记录姿态')


def reached(saved, current, position_tolerance=.003, orientation_tolerance_deg=2.):
    distance = np.linalg.norm(np.asarray(saved['position'])-current['position'])
    rotation = np.asarray(saved['rotation']) @ np.asarray(current['rotation']).T
    angle = np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))
    return distance <= position_tolerance and angle <= np.deg2rad(orientation_tolerance_deg)


class PoseActions:
    def __init__(self, backend, root, *, demo=False):
        self.backend = backend
        self.store = PoseStore(root, demo=demo)
        self.driver = None
        self.status = {'state':'idle'}
        self.target = None
        self.demo = demo

    @property
    def moving(self):
        return self.status['state'] == 'moving'

    def decorate(self, status):
        status['pose_operation'] = self.status
        if self.backend.config.workflow == 'collect':
            status['available_commands'] = list(status.get('available_commands', [])) + ['pose_capture', 'pose_restore']
        if self.moving:
            status.update(state='active', enabled=True, actual_source='pose_return', available_commands=['pause', 'stop'])
            status['arms'] = {arm: {**value, **({'source':'pose_return'} if arm == self.target['arm'] else {})}
                              for arm, value in status.get('arms', {}).items()}
        return status

    def command(self, operation, args):
        if operation not in ('pose_capture', 'pose_restore'):
            if self.moving:
                if operation not in ('pause', 'stop'):
                    raise RuntimeError('正在恢复姿态；请等待完成或先软件暂停')
                self.finish('cancelled', '操作员停止恢复')
            return False
        if self.backend.config.workflow != 'collect':
            raise ValueError('姿态库仅用于数据采集前定位')
        status = self.backend.sample()
        if self.moving or status.get('state') != 'standby' or status.get('enabled') or status.get('recording') or status.get('saving') or status.get('error'):
            raise RuntimeError('请先结束／保存当前回合并软件暂停；姿态操作要求无故障待命状态')
        if self.driver is None:
            self.driver = DemoPoseDriver(self.backend) if self.demo else ROSPoseDriver(self.backend)
        if operation == 'pose_capture':
            note_text(args.get('note'))
            arm = args.get('arm')
            if arm not in self.backend.arms:
                raise ValueError('请选择本会话中的机械臂')
            record = self.store.add(args['note'], self.driver.read(arm))
            self.status = {'state':'saved', 'record_id':record['id'], 'note':record['note']}
        else:
            record = self.store.get(args.get('pose_id'))
            self.target = record['pose']
            current = self.driver.read(self.target['arm'])
            compatible(self.target, current)
            try:
                self.driver.start(self.target)
            except Exception:
                self.driver.stop()
                raise
            self.status = {'state':'moving', 'record_id':record['id'], 'note':record['note'], 'arm':self.target['arm']}
            self.deadline = time.monotonic()+60
        return True

    def finish(self, state, reason):
        try:
            self.driver.stop()
        except Exception as error:
            state, reason = 'failed', f'{reason}; 停止失败: {error}'
        self.status = {**self.status, 'state':state, 'message':reason}

    def tick(self):
        if not self.moving:
            return
        try:
            if time.monotonic() > self.deadline:
                raise TimeoutError('姿态恢复超时，已停止；请检查可达性与现场障碍物')
            state = self.driver.tick(self.target)
            if state:
                self.finish(state, '已恢复，回到待命；夹爪未恢复' if state == 'completed' else '输入接管或控制暂停，已取消恢复')
        except Exception as error:
            self.finish('failed', str(error))


class ROSPoseDriver:
    def __init__(self, backend):
        self.b = backend
        self.config = backend.collector.config if backend.collector else backend.rollout_config()
        self.arms = {a.name:a for a in self.config.arms}
        self.servo = None
        self.last_command = None
        self.arm = None

    def _state(self, arm):
        from vla.datacollection.franka_robotiq_single_left import schema
        if arm not in self.arms:
            raise ValueError('本会话没有该机械臂')
        node = self.b.collector or self.b.recorder
        with node._lock:
            latest = dict(node._latest)
        def fresh(key):
            entry = latest.get(f'{arm}.{key}') or (latest.get(key) if arm == 'left' else None)
            if entry is None or not 0 <= time.monotonic()-entry[0]/1e9 <= .5:
                raise RuntimeError(f'{arm} {key} 数据缺失／过期')
            return entry
        stamp, state = fresh('robot_state')
        if stamp != self.b._health_stamps.get(arm):
            self.b.health.check(arm, state)
            self.b._health_stamps[arm] = stamp
        if self.b.collector:
            joints = self.b.collector._measured_joints(self.arms[arm], fresh('measured_joint')[1])
            if self.b.collector._faulted:
                raise RuntimeError(self.b.collector._fault_reason)
        else:
            node = self.b.arbiters[self.b.arms.index(arm)]
            if node.managed_fault:
                raise RuntimeError(node.managed_fault)
            if node.measured_joints is None or time.monotonic()-node.measured_joint_time > node.source_timeout:
                raise RuntimeError('关节状态过期')
            joints = node.measured_joints.copy()
        position, rotation = schema.measured_pose_from_robot_state(state)
        tool_p, tool_r = schema.pose_from_stamped(state.f_t_ee)
        pose = pose_value(arm, self.b.hardware[f'{arm.upper()}_ROBOT_IP'], self.arms[arm].base_link+'/EE', position, rotation, tool_p, tool_r)
        check_bounds(pose, self.arms[arm].workspace_min, self.arms[arm].workspace_max)
        if np.asarray(joints).shape != (7,) or not np.isfinite(joints).all():
            raise ValueError('无效关节状态')
        return pose, joints

    def read(self, arm):
        return self._state(arm)[0]

    def start(self, target):
        self.arm = target['arm']
        arm = self.arms[self.arm]
        check_bounds(target, arm.workspace_min, arm.workspace_max)
        self.read(self.arm)
        c = self.b.collector
        if not c._neutral_gamepads():
            raise RuntimeError('请松开手柄并确认输入在线')
        c._control_boundary('pose', reason='saved pose restore')
        self.last = time.monotonic()

    def tick(self, target):
        from vla.actions import matrix_to_rot6d
        c = self.b.collector
        if c._faulted or c.coordinator.mode == 'fault':
            raise RuntimeError(c._fault_reason or c.coordinator.reason)
        if not c._neutral_gamepads():
            return 'cancelled'
        current = self.read(self.arm)
        compatible(target, current)
        if reached(target, current):
            return 'completed'
        if c.coordinator.mode != 'active':
            return
        now = time.monotonic()
        dt = now-self.last
        if dt < .01:
            return
        self.last = now
        if dt > .1:
            raise RuntimeError('恢复控制循环超时')
        c._observe_control()
        tcp = {}
        for arm in c.config.arms:
            pose = target if arm.name == self.arm else self.read(arm.name)
            m = c.coordinator._measurements[arm.name]
            tcp[arm.name] = (np.asarray(pose['position']), np.asarray(pose['rotation']), m.gripper)
        action = c.coordinator.gamepad_target(tcp, 1/c.config.fps)
        c.coordinator.accept(action, source='pose', generation=c.coordinator.generation, slot_id=time.monotonic_ns()*30//1_000_000_000)
        c._track_control(min(dt, .05))

    def stop(self):
        self.b.collector._control_boundary(reason='pose restore stopped')
        self.b.enabled = False


class DemoPoseDriver:
    def __init__(self, backend):
        self.b = backend

    def read(self, arm):
        if arm not in self.b.arms:
            raise ValueError('本会话没有该机械臂')
        if arm in self.b._demo_poses:
            return self.b._demo_poses[arm]
        values = self.b.sample()['values']
        from scipy.spatial.transform import Rotation
        return pose_value(arm, 'demo', 'demo:base/EE', [values[f'{arm}.tcp.{x}'] for x in 'xyz'],
                          Rotation.from_rotvec([values[f'{arm}.orientation.r{x}'] for x in 'xyz']).as_matrix(), np.zeros(3), np.eye(3))

    def start(self, target):
        self.begin = time.monotonic()

    def tick(self, target):
        if time.monotonic()-self.begin >= .5:
            self.b._demo_poses[target['arm']] = target
            return 'completed'

    def stop(self):
        self.b.enabled = False
