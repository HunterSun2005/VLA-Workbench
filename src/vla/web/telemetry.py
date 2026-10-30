"""Convert cached measured state to labelled channels; unavailable values are absent."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .demo import descriptors


def vector(values, arm, group, names, array):
    if array is None:
        return
    array = np.asarray(array, dtype=float).reshape(-1)
    for name, number in zip(names, array):
        if np.isfinite(number):
            values[f'{arm}.{group}.{name}'] = float(number)


def pose_values(values, arm, position, rotation):
    vector(values, arm, 'tcp', ('x','y','z'), position)
    vector(values, arm, 'orientation', ('rx','ry','rz'), Rotation.from_matrix(rotation).as_rotvec())


def franky_values(state, width=None):
    from vla.robots.franka_robotiq.robot import pose_matrix
    values = {}
    if state is None:
        return values
    vector(values, 'left', 'health', ['success_rate'], getattr(state,'control_command_success_rate',None))
    pose = pose_matrix(state.O_T_EE)
    pose_values(values, 'left', pose[:3,3], pose[:3,:3])
    for attr, group, prefix in [('q','joint','q'),('dq','velocity','dq'),('tau_J','torque','tau')]:
        vector(values, 'left', group, [f'{prefix}{i}' for i in range(1,8)], getattr(state, attr, None))
    vector(values, 'left','gripper', ['width'], [width] if width is not None else None)
    return values


def ros_values(latest, arms):
    from vla.datacollection.franka_robotiq_single_left import schema
    values = {}
    import time
    for arm in arms:
        entry = latest.get(f'{arm}.robot_state') or (latest.get('robot_state') if arm == 'left' else None)
        if not entry or time.monotonic() - entry[0]/1e9 > .5:
            continue
        state = entry[1]
        vector(values, arm, 'health', ['success_rate'], getattr(state,'control_command_success_rate',None))
        pos, rot = schema.measured_pose_from_robot_state(state)
        pose_values(values, arm, pos, rot)
        vector(values, arm, 'joint', [f'q{i}' for i in range(1,8)], schema.measured_joint_positions(state))
        measured = getattr(state, 'measured_joint_state', None)
        if measured is not None:
            vector(values, arm, 'velocity', [f'dq{i}' for i in range(1,8)], measured.velocity)
            vector(values, arm, 'torque', [f'tau{i}' for i in range(1,8)], measured.effort)
        grip = latest.get(f'{arm}.gripper_state') or (latest.get('gripper') if arm == 'left' else None)
        if grip and time.monotonic()-grip[0]/1e9 <= .5:
            vector(values, arm, 'gripper', ['width'], [schema.gripper_width(grip[1])])
    return values


def extra_signals(arms):
    return [dict(key=f'{arm}.{group}.{name}',arm=arm,group=group,label=name,unit=unit,
                 source='measured' if group=='health' else 'target',default=False)
            for arm in arms for group,names,unit in (
                ('health',['success_rate'],'ratio'), ('target_tcp',['x','y','z'],'m'),
                ('target_joint',[f'q{i}' for i in range(1,8)],'rad'),
                ('target_linear',['x','y','z'],'m/s'), ('target_angular',['x','y','z'],'rad/s'))
            for name in names]
