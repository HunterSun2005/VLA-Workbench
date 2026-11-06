"""Legacy 31D field layout for offline data inspection and extraction.

Live collection uses schema.STATE_FIELDS (50D per arm).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vla.control.geometry import state_vector


FORCE_STATE_FIELDS = tuple(
    field
    for prefix in ("joint_torque", "external_joint_torque", "joint_velocity")
    for field in (f"{prefix}_{index}" for index in range(1, 8))
)

STATE_31D_FIELDS = (
    "tcp_position_x",
    "tcp_position_y",
    "tcp_position_z",
    "tcp_rot6d_col0_x",
    "tcp_rot6d_col0_y",
    "tcp_rot6d_col0_z",
    "tcp_rot6d_col1_x",
    "tcp_rot6d_col1_y",
    "tcp_rot6d_col1_z",
    "gripper_position",
) + FORCE_STATE_FIELDS


def _finite_joint_vector(value: Any, field_name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError(f"{field_name} must be a finite 7-vector")
    return result


def state_vector_31d(robot_state: Any, gripper_position: float) -> np.ndarray:
    """Return 10D policy state plus tau_J, external torque, and dq."""
    state_10d = state_vector(robot_state, gripper_position)
    measured_torque = _finite_joint_vector(robot_state.tau_J, "tau_J")
    external_torque = _finite_joint_vector(
        robot_state.tau_ext_hat_filtered,
        "tau_ext_hat_filtered",
    )
    joint_velocity = _finite_joint_vector(robot_state.dq, "dq")
    result = np.concatenate(
        (state_10d, measured_torque, external_torque, joint_velocity)
    ).astype(np.float32)
    if result.shape != (31,) or not np.isfinite(result).all():
        raise ValueError("invalid Franka/Robotiq 31D state")
    return result


def state_dict_31d(robot_state: Any, gripper_position: float) -> dict[str, float]:
    """Return named 31D values for an independent recorder."""
    values = state_vector_31d(robot_state, gripper_position)
    return dict(zip(STATE_31D_FIELDS, values.tolist(), strict=True))
