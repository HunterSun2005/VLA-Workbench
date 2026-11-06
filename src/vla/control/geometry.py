"""Shared geometry and read-only conversion of historical state records."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

STATE_FIELDS = (
    "tcp_position_x", "tcp_position_y", "tcp_position_z",
    "tcp_rot6d_col0_x", "tcp_rot6d_col0_y", "tcp_rot6d_col0_z",
    "tcp_rot6d_col1_x", "tcp_rot6d_col1_y", "tcp_rot6d_col1_z",
    "gripper_position",
)
ACTION_FIELDS = (
    "desired_tcp_position_x", "desired_tcp_position_y", "desired_tcp_position_z",
    "desired_tcp_rot6d_col0_x", "desired_tcp_rot6d_col0_y", "desired_tcp_rot6d_col0_z",
    "desired_tcp_rot6d_col1_x", "desired_tcp_rot6d_col1_y", "desired_tcp_rot6d_col1_z",
    "gripper",
)
TARGET_SMOOTHING_ALPHA = 0.25
GRIPPER_OPEN_WIDTH_M = 0.085
FR3_JOINT_LOWER = np.array(
    [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159],
    dtype=np.float64,
)
FR3_JOINT_UPPER = np.array(
    [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159],
    dtype=np.float64,
)


@dataclass(frozen=True)
class FrankaRolloutSnapshot:
    observed_at_ns: int
    pose: np.ndarray
    joint_positions: np.ndarray
    base_wrench: np.ndarray
    joint_torques: np.ndarray
    external_joint_torques: np.ndarray
    joint_velocities: np.ndarray
    gripper_width: float
    gripper_open_fraction: float
    images: dict[str, np.ndarray]
    image_timestamps: dict[str, float]

    @property
    def policy_state(self) -> np.ndarray:
        return np.concatenate(
            (
                self.pose[:3, 3],
                matrix_to_rot6d(self.pose[:3, :3]),
                [self.gripper_width],
            )
        ).astype(np.float32)


def _clamp_vector_norm(value: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm == 0.0 or norm <= maximum:
        return value
    return value * (maximum / norm)


def solve_human_joint_velocity(
    jacobian: np.ndarray,
    cartesian_velocity: np.ndarray,
    *,
    joint_positions: np.ndarray,
    joint7_velocity: float,
    joint7_active: bool,
    damping: float,
    max_joint_speed: float,
    joint_limit_margin: float,
    dt: float,
) -> np.ndarray:
    """Map a Cartesian command and optional wrist command to bounded FR3 joint velocity."""
    jacobian = np.asarray(jacobian, dtype=np.float64)
    cartesian_velocity = np.asarray(cartesian_velocity, dtype=np.float64)
    joint_positions = np.asarray(joint_positions, dtype=np.float64)
    if jacobian.shape != (6, 7) or cartesian_velocity.shape != (6,):
        raise ValueError("human Jacobian/velocity must have shapes (6, 7) and (6,)")
    if joint_positions.shape != (7,):
        raise ValueError("human joint positions must have shape (7,)")
    scalars = (joint7_velocity, damping, max_joint_speed, joint_limit_margin, dt)
    if not all(np.isfinite(value) for value in scalars):
        raise ValueError("human control scalars must be finite")
    if not np.isfinite(jacobian).all() or not np.isfinite(cartesian_velocity).all():
        raise ValueError("human control input must be finite")
    if damping <= 0.0 or max_joint_speed <= 0.0 or dt <= 0.0:
        raise ValueError("human damping, max speed, and dt must be positive")

    if joint7_active:
        joint7_row = np.zeros((1, 7), dtype=np.float64)
        joint7_row[0, 6] = 1.0
        objective = np.vstack((jacobian[:3], joint7_row))
        target = np.concatenate((cartesian_velocity[:3], [joint7_velocity]))
    else:
        objective = jacobian
        target = cartesian_velocity
    regularized = objective @ objective.T + damping**2 * np.eye(len(objective))
    velocity = objective.T @ np.linalg.solve(regularized, target)

    lower = FR3_JOINT_LOWER + joint_limit_margin
    upper = FR3_JOINT_UPPER - joint_limit_margin
    minimum_velocity = (lower - joint_positions) / dt
    maximum_velocity = (upper - joint_positions) / dt
    return np.clip(
        velocity,
        np.maximum(-max_joint_speed, minimum_velocity),
        np.minimum(max_joint_speed, maximum_velocity),
    )


def integrate_human_tool_rotation(
    current_rotation: np.ndarray,
    target_rotation: np.ndarray,
    angular_velocity_tool: np.ndarray | None,
    *,
    joint7_velocity: float,
    dt: float,
    max_angular_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine tool-axis rates before one SO(3) increment; never rotate XYZ."""
    angular = np.zeros(3) if angular_velocity_tool is None else np.asarray(angular_velocity_tool, dtype=float)
    if angular.shape != (3,) or not np.isfinite(angular).all():
        raise ValueError("human tool angular velocity must be a finite 3-vector")
    if not np.isfinite([joint7_velocity, dt, max_angular_speed]).all() or dt <= 0 or max_angular_speed <= 0:
        raise ValueError("invalid human rotation limits")
    if joint7_velocity != 0 and np.any(angular != 0):
        raise ValueError("cannot combine joint-7 control with tool angular velocity")
    angular_base = current_rotation @ _clamp_vector_norm(angular, max_angular_speed)
    speed = float(np.linalg.norm(angular_base))
    rotation = target_rotation.copy()
    if speed > 1e-8:
        rotation = _axis_angle_matrix(angular_base / speed, speed * dt) @ rotation
    return rotation, angular_base


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation matrix must be finite with shape (3, 3)")
    return np.concatenate((matrix[:, 0], matrix[:, 1]))


def rot6d_to_matrix(rot6d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    value = np.asarray(rot6d, dtype=np.float64)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("rot6d must be a finite 6-vector")
    first = value[:3]
    norm = np.linalg.norm(first)
    if norm < eps:
        raise ValueError("rot6d first column has zero norm")
    first = first / norm
    second = value[3:] - np.dot(first, value[3:]) * first
    norm = np.linalg.norm(second)
    if norm < eps:
        raise ValueError("rot6d columns are collinear")
    second = second / norm
    return np.column_stack((first, second, np.cross(first, second)))


def pose_matrix(value: Any) -> np.ndarray:
    if hasattr(value, "matrix"):
        value = value.matrix
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size == 16:
        matrix = matrix.reshape(4, 4, order="F") if matrix.shape != (4, 4) else matrix
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"O_T_EE must be a finite 4x4 pose, got {matrix.shape}")
    return matrix.copy()


def state_vector(robot_state: Any, gripper_position: float) -> np.ndarray:
    matrix = pose_matrix(robot_state.O_T_EE)
    value = np.concatenate((matrix[:3, 3], matrix_to_rot6d(matrix[:3, :3]), [gripper_position]))
    if value.shape != (10,) or not np.isfinite(value).all():
        raise ValueError("invalid Franka/Robotiq state")
    return value.astype(np.float32)


def action_vector(action: dict[str, Any] | np.ndarray) -> np.ndarray:
    if isinstance(action, dict):
        try:
            value = np.array([float(np.asarray(action[name]).reshape(-1)[0]) for name in ACTION_FIELDS])
        except KeyError as error:
            raise ValueError(f"action is missing field {error.args[0]!r}") from error
    else:
        value = np.asarray(action, dtype=np.float64)
    if value.shape != (10,) or not np.isfinite(value).all():
        raise ValueError(f"action must be a finite 10-vector, got {value.shape}")
    return value


def _rotation_angle(rotation: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))


def _rotation_axis(rotation: np.ndarray, angle: float) -> np.ndarray:
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0])
    axis = np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]
    )
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        values, vectors = np.linalg.eigh(rotation)
        axis = vectors[:, np.argmin(np.abs(values - 1.0))]
        return axis / np.linalg.norm(axis)
    return axis / norm


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def limit_pose(
    current: np.ndarray,
    target: np.ndarray,
    *,
    workspace_min: tuple[float, float, float],
    workspace_max: tuple[float, float, float],
    max_linear_step: float,
    max_angular_step: float,
) -> np.ndarray:
    current, target = pose_matrix(current), pose_matrix(target)
    result = target.copy()
    requested = np.clip(target[:3, 3], workspace_min, workspace_max)
    delta = requested - current[:3, 3]
    distance = np.linalg.norm(delta)
    if distance > max_linear_step:
        delta *= max_linear_step / distance
    result[:3, 3] = current[:3, 3] + delta

    relative = target[:3, :3] @ current[:3, :3].T
    angle = _rotation_angle(relative)
    if angle > max_angular_step:
        relative = _axis_angle_matrix(_rotation_axis(relative, angle), max_angular_step)
        result[:3, :3] = relative @ current[:3, :3]
    return result


def smooth_pose_target(
    previous: np.ndarray,
    current: np.ndarray,
    alpha: float = TARGET_SMOOTHING_ALPHA,
) -> np.ndarray:
    """Smooth consecutive absolute pose targets in position and SO(3)."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError("target smoothing alpha must be in [0, 1)")
    previous, current = pose_matrix(previous), pose_matrix(current)
    result = current.copy()
    result[:3, 3] = alpha * previous[:3, 3] + (1.0 - alpha) * current[:3, 3]

    relative = current[:3, :3] @ previous[:3, :3].T
    angle = _rotation_angle(relative)
    if angle < 1e-8:
        result[:3, :3] = previous[:3, :3]
    else:
        step = _axis_angle_matrix(_rotation_axis(relative, angle), (1.0 - alpha) * angle)
        result[:3, :3] = step @ previous[:3, :3]
    return result


def pose_step_to_twist(
    current: np.ndarray,
    limited: np.ndarray,
    tracking_tau_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a limited pose error into a base-frame tracking twist."""
    if tracking_tau_s <= 0:
        raise ValueError("tracking time constant must be positive")
    current, limited = pose_matrix(current), pose_matrix(limited)
    linear = (limited[:3, 3] - current[:3, 3]) / tracking_tau_s
    relative = limited[:3, :3] @ current[:3, :3].T
    angle = _rotation_angle(relative)
    angular = (
        _rotation_axis(relative, angle) * (angle / tracking_tau_s)
        if angle >= 1e-8
        else np.zeros(3)
    )
    return linear, angular



def upright_tcp_rotation(rotation: np.ndarray) -> np.ndarray:
    """Point tool Z down while retaining the current horizontal tool heading."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("TCP rotation must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1):
        raise ValueError("TCP rotation must be in SO(3)")
    z_axis = np.array([0.0, 0.0, -1.0])
    x_axis = rotation[:, 0].copy()
    x_axis[2] = 0.0
    norm = float(np.linalg.norm(x_axis))
    if norm < 1e-6:
        # When tool X is vertical, retain the heading implied by tool Y.
        y_axis = rotation[:, 1].copy()
        y_axis[2] = 0.0
        y_norm = float(np.linalg.norm(y_axis))
        x_axis = np.cross(y_axis / y_norm, z_axis) if y_norm >= 1e-6 else np.array([1.0, 0.0, 0.0])
    else:
        x_axis /= norm
    y_axis = np.cross(z_axis, x_axis)
    result = np.column_stack((x_axis, y_axis, z_axis))
    if not np.allclose(result.T @ result, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(result), 1):
        raise RuntimeError("failed to construct an upright TCP rotation")
    return result


def axis_aligned_upright_rotation(rotation: np.ndarray) -> np.ndarray:
    """Choose the base-axis-aligned downward pose nearest the reference heading."""
    upright = upright_tcp_rotation(rotation)
    headings = np.array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.]])
    x_axis = headings[np.argmax(headings @ upright[:, 0])]
    z_axis = np.array([0., 0., -1.])
    return np.column_stack((x_axis, np.cross(z_axis, x_axis), z_axis))


def reset_tcp_heading(rotation: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rotate only about base Z to the reference heading, retaining tilt."""
    current_heading = upright_tcp_rotation(rotation)
    reference_heading = upright_tcp_rotation(reference)
    # The downward-Z upright frames cancel to a pure base-Z rotation.
    return reference_heading @ current_heading.T @ rotation
