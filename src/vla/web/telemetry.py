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


def wrench_values(values, arm, wrench):
    """External wrench at K, expressed in base O; never substitute K-frame data."""
    if wrench is None:
        return
    wrench = np.asarray(wrench, dtype=float).reshape(-1)
    if wrench.shape != (6,):
        return
    vector(values, arm, 'tcp_force', ('x', 'y', 'z'), wrench[:3])
    vector(values, arm, 'tcp_torque', ('x', 'y', 'z'), wrench[3:])


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
        wrench = getattr(getattr(state, 'o_f_ext_hat_k', None), 'wrench', None)
        if wrench is not None:
            wrench_values(values, arm, [wrench.force.x, wrench.force.y, wrench.force.z,
                                       wrench.torque.x, wrench.torque.y, wrench.torque.z])
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
