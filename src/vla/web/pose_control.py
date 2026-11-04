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
            self.driver = DemoPoseDriver(self.backend) if self.demo else ROSPoseDriver(self.backend) if self.backend.config.ros else FrankyPoseDriver(self.backend)
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


class FrankyPoseDriver:
    def __init__(self, backend):
        self.b = backend
        self.c = backend.session.controller

    def read(self, arm):
        if arm != 'left':
            raise ValueError('本会话没有该机械臂')
        from vla.datacollection.start_teleop.gamepad_teleop import pose_matrix
        state = self.c._read_state()
        pose, tool = pose_matrix(state.O_T_EE), pose_matrix(state.F_T_EE)
        return pose_value(arm, self.c.config.robot.robot_ip, 'franky:base/EE', pose[:3,3], pose[:3,:3], tool[:3,3], tool[:3,:3])

    def start(self, target):
        from vla.datacollection.start_teleop.gamepad_teleop import PoseReturn
        check_bounds(target, self.c.config.robot.workspace_min, self.c.config.robot.workspace_max)
        if not (self.c.config.robot.human_linear_velocity_control and self.c.config.robot.human_angular_velocity_control):
            raise ValueError('恢复姿态需要直接线速度和角速度控制配置')
        self.c.enable()
        if self.c.mode != 'active':
            raise RuntimeError('无法启用恢复：检查机械臂状态，并松开手柄所有按键和摇杆')
        self.c._pose_return = PoseReturn('saved pose', np.array(target['position']), np.array(target['rotation']), self.c.clock()+60)
        self.c._pose_return_path = []

    def tick(self, target):
        if self.c.mode == 'fault':
            raise RuntimeError(self.c.reason)
        if self.c.mode != 'active':
            return 'cancelled'
        # Any input cancels BEFORE controller.tick, so gripper/shortcuts cannot
        # run for one frame during cancellation. No implicit manual takeover.
        if not self.c._neutral(self.c._packet()):
            return 'cancelled'
        current = self.read(target['arm'])
        if reached(target, current, self.c.config.pose_shortcuts.position_tolerance_m,
                   self.c.config.pose_shortcuts.orientation_tolerance_deg):
            return 'completed'
        if self.c._pose_return is None:
            raise RuntimeError('恢复已中止，尚未达到保存姿态')

    def stop(self):
        self.c.pause('saved pose positioning stopped')


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
        from vla.datacollection.franka_robotiq_rollout.collect_ros import CartesianServo
        self.arm = target['arm']
        arm = self.arms[self.arm]
        check_bounds(target, arm.workspace_min, arm.workspace_max)
        current, joints = self._state(self.arm)
        if self.b.collector:
            if not self.b.collector._neutral_gamepads():
                raise RuntimeError('请松开双手柄并确认输入在线')
            model = self.b.collector.servos[self.arm].model
        else:
            model = self.b.arbiters[self.b.arms.index(self.arm)].model
        if np.any(joints < model.lower_limits) or np.any(joints > model.upper_limits):
            raise ValueError('当前关节超出配置范围，拒绝自动拉回')
        # Same bounded Cartesian solver as ROS rollout; separate tracking state
        # and conservative speeds for unrecorded positioning.
        self.servo = CartesianServo(model, replace(self.config,
            max_linear_velocity=min(.03,self.config.max_linear_velocity),
            max_angular_velocity=min(.15,self.config.max_angular_velocity),
            max_joint_speed=min(.15,self.config.max_joint_speed)))
        xyz, rotation = self.servo.forward_tcp(joints, np.array(current['tool']['position']), np.array(current['tool']['rotation']))
        if not reached({**current, 'position':xyz, 'rotation':rotation}, current):
            raise ValueError('运动学模型与实测 TCP 不一致，请检查基座、法兰及工具坐标系')
        self.last_command = joints.copy()
        self.last = time.monotonic()

    def _publish(self, joints):
        if self.b.collector:
            self.b.collector._publish_arm_command(self.arms[self.arm], joints, 0., publish_gripper=False)
        else:
            from sensor_msgs.msg import JointState
            node = self.b.arbiters[self.b.arms.index(self.arm)]
            message = JointState()
            message.header.stamp = node.get_clock().now().to_msg()
            message.header.frame_id = self.arms[self.arm].base_link
            message.name = list(node.model.joint_names)
            message.position = np.asarray(joints).tolist()
            node.joint_publisher.publish(message)

    def tick(self, target):
        now = time.monotonic()
        elapsed = now-self.last
        if elapsed < .01:
            return
        self.last = now
        if elapsed > .1:
            raise RuntimeError('恢复控制循环超时，已停止')
        if self.b.collector and not self.b.collector._neutral_gamepads():
            return 'cancelled'
        current, joints = self._state(self.arm)
        compatible(target, current)
        if reached(target, current):
            return 'completed'
        if np.max(np.abs(joints-self.last_command)) > .12:
            raise RuntimeError('关节跟踪误差过大，已停止')
        tool_p, tool_r = np.array(current['tool']['position']), np.array(current['tool']['rotation'])
        dt = min(elapsed, .02)
        command = self.servo.step(self.last_command, np.array(target['position']), np.array(target['rotation']), tool_p, tool_r, dt)
        model = self.servo.model
        if not np.isfinite(command).all() or np.any(command < model.lower_limits) or np.any(command > model.upper_limits):
            raise RuntimeError('关节目标无效／超限')
        xyz, rot = self.servo.forward_tcp(command, tool_p, tool_r)
        arm = self.arms[self.arm]
        check_bounds({'arm':self.arm,'position':xyz}, arm.workspace_min, arm.workspace_max)
        self._publish(command)
        self.last_command = command.copy()

    def stop(self):
        if self.arm is not None and self.last_command is not None:
            # Hold fresh measured joints; if disconnected retain the last small
            # bounded target. Never publish a stored or computed gripper target.
            try:
                _, joints = self._state(self.arm)
            except Exception:
                joints = self.last_command
            self._publish(joints)
        self.last_command = None
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
