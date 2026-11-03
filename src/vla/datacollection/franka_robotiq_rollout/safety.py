from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from vla.datacollection.franka_robotiq_single_left import schema


# Continuous VLA outputs can overshoot the normalized endpoint slightly after
# action denormalization. Saturate small endpoint overshoots while retaining a
# guard against a broken action processor or checkpoint contract.
GRIPPER_ACTION_TOLERANCE = 0.05


def compose_pose(
    parent_position: np.ndarray,
    parent_rotation: np.ndarray,
    child_position: np.ndarray,
    child_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose parent_T_child from position/rotation components."""
    parent_position = np.asarray(parent_position, dtype=np.float64)
    parent_rotation = np.asarray(parent_rotation, dtype=np.float64)
    child_position = np.asarray(child_position, dtype=np.float64)
    child_rotation = np.asarray(child_rotation, dtype=np.float64)
    if parent_position.shape != (3,) or child_position.shape != (3,):
        raise ValueError("pose positions must be 3-vectors")
    if parent_rotation.shape != (3, 3) or child_rotation.shape != (3, 3):
        raise ValueError("pose rotations must be 3x3 matrices")
    if not all(
        np.isfinite(value).all()
        for value in (parent_position, parent_rotation, child_position, child_rotation)
    ):
        raise ValueError("pose components must be finite")
    return (
        parent_position + parent_rotation @ child_position,
        parent_rotation @ child_rotation,
    )


def flange_pose_from_tcp(
    tcp_position: np.ndarray,
    tcp_rotation: np.ndarray,
    flange_to_tcp_position: np.ndarray,
    flange_to_tcp_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert base_T_tcp into base_T_flange using flange_T_tcp."""
    tcp_position = np.asarray(tcp_position, dtype=np.float64)
    tcp_rotation = np.asarray(tcp_rotation, dtype=np.float64)
    flange_to_tcp_position = np.asarray(flange_to_tcp_position, dtype=np.float64)
    flange_to_tcp_rotation = np.asarray(flange_to_tcp_rotation, dtype=np.float64)
    flange_rotation = tcp_rotation @ flange_to_tcp_rotation.T
    flange_position = tcp_position - flange_rotation @ flange_to_tcp_position
    return flange_position, flange_rotation


def active_boolean_fields(message: object) -> list[str]:
    """Return true ROS-message boolean fields without depending on ROS imports."""
    get_fields = getattr(message, "get_fields_and_field_types", None)
    if callable(get_fields):
        names = get_fields().keys()
    elif hasattr(message, "__dict__"):
        names = vars(message).keys()
    else:
        names = ()
    return [name for name in names if not name.startswith("_") and getattr(message, name, False) is True]


class FrankaHealthMonitor:
    def __init__(self, minimum_success_rate: float, low_rate_samples: int) -> None:
        if not 0.0 < minimum_success_rate <= 1.0:
            raise ValueError("minimum_success_rate must be in (0, 1]")
        if low_rate_samples < 1:
            raise ValueError("low_rate_samples must be positive")
        self.minimum_success_rate = float(minimum_success_rate)
        # Match the gamepad monitor: accept the threshold's float32
        # representation (or adjacent float64 rounding), not the next lower
        # float32 value. Do not apply a general tolerance to low rates.
        self._success_rate_boundary = min(
            float(np.float32(self.minimum_success_rate)),
            np.nextafter(self.minimum_success_rate, -np.inf),
        )
        self.low_rate_samples = int(low_rate_samples)
        self._low_rate_counts: dict[str, int] = {}

    def check(self, arm_name: str, robot_state: object) -> None:
        errors = active_boolean_fields(robot_state.current_errors)
        if errors:
            raise RuntimeError(f"{arm_name} Franka current_errors={errors}")

        mode = int(robot_state.robot_mode)
        allowed_modes = {
            int(robot_state.ROBOT_MODE_IDLE),
            int(robot_state.ROBOT_MODE_MOVE),
        }
        if mode not in allowed_modes:
            raise RuntimeError(f"{arm_name} Franka robot_mode={mode} is not IDLE/MOVE")

        success_rate = float(robot_state.control_command_success_rate)
        if not np.isfinite(success_rate):
            raise RuntimeError(f"{arm_name} Franka control_command_success_rate is non-finite")
        count = self._low_rate_counts.get(arm_name, 0)
        count = count + 1 if success_rate < self._success_rate_boundary else 0
        self._low_rate_counts[arm_name] = count
        if count >= self.low_rate_samples:
            raise RuntimeError(
                f"{arm_name} Franka control_command_success_rate={success_rate:.17g} "
                f"below {self.minimum_success_rate:.3f} for {count} samples "
                f"(comparison boundary={self._success_rate_boundary:.17g})"
            )


def rot6d_to_matrix(rot6d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    value = np.asarray(rot6d, dtype=np.float64)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("rot6d must be a finite 6-vector")
    first = value[:3]
    first_norm = float(np.linalg.norm(first))
    if first_norm < eps:
        raise ValueError("rot6d first column has zero norm")
    first = first / first_norm
    second = value[3:] - np.dot(first, value[3:]) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < eps:
        raise ValueError("rot6d columns are collinear")
    second = second / second_norm
    return np.column_stack((first, second, np.cross(first, second)))


def validate_absolute_action(
    action: np.ndarray,
    *,
    arm_count: int,
    workspace_bounds: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (10 * arm_count,) or not np.isfinite(value).all():
        raise ValueError(f"action must be finite with shape {(10 * arm_count,)}, got {value.shape}")
    decoded = []
    for arm_index in range(arm_count):
        arm = value[10 * arm_index : 10 * (arm_index + 1)]
        position = arm[:3]
        lower, upper = workspace_bounds[arm_index]
        if np.any(position < lower) or np.any(position > upper):
            raise ValueError(f"arm {arm_index} target is outside workspace: {position.tolist()}")
        gripper = float(arm[9])
        if not 0.0 <= gripper <= 1.0:
            raise ValueError(f"arm {arm_index} gripper is outside [0, 1]: {gripper}")
        decoded.append((position.copy(), rot6d_to_matrix(arm[3:9]), gripper))
    return decoded


def clamp_gripper_action_roundoff(
    action: np.ndarray,
    *,
    arm_count: int,
    tolerance: float = GRIPPER_ACTION_TOLERANCE,
) -> np.ndarray:
    """Clamp small continuous-policy overshoots at bounded gripper outputs."""
    value = np.asarray(action)
    if value.shape != (10 * arm_count,) or not np.isfinite(value).all():
        raise ValueError(f"action must be finite with shape {(10 * arm_count,)}, got {value.shape}")
    if tolerance < 0.0:
        raise ValueError("gripper tolerance must be non-negative")
    result = value.copy()
    for arm_index in range(arm_count):
        gripper_index = 10 * arm_index + 9
        gripper = float(result[gripper_index])
        if gripper < -tolerance or gripper > 1.0 + tolerance:
            raise ValueError(
                f"arm {arm_index} gripper is outside [0, 1] beyond "
                f"tolerance {tolerance}: {gripper}"
            )
        result[gripper_index] = np.clip(gripper, 0.0, 1.0)
    return result


def validate_rollout_episode(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    arm_count: int,
    workspace_bounds: list[tuple[np.ndarray, np.ndarray]],
    maximum_gripper_width: float,
) -> None:
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    state_dim_per_arm = len(schema.STATE_FIELDS)
    if states.ndim != 2 or states.shape[1] != state_dim_per_arm * arm_count:
        raise ValueError(f"rollout state array has invalid shape: {states.shape}")
    if actions.shape != (len(states), 10 * arm_count):
        raise ValueError(f"rollout action array has invalid shape: {actions.shape}")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("rollout episode contains NaN or infinity")
    for arm_index in range(arm_count):
        gripper = states[:, arm_index * state_dim_per_arm + 9]
        if np.any(gripper < 0.0) or np.any(gripper > maximum_gripper_width + 1e-6):
            raise ValueError(f"arm {arm_index} measured gripper width is outside its physical range")
    for action in actions:
        validate_absolute_action(
            action,
            arm_count=arm_count,
            workspace_bounds=workspace_bounds,
        )


def policy_state_from_raw(raw_state: np.ndarray, arm_count: int) -> np.ndarray:
    """Select each arm's 10D training state from the expanded raw state."""
    value = np.asarray(raw_state)
    expected = len(schema.STATE_FIELDS) * arm_count
    if value.shape != (expected,) or not np.isfinite(value).all():
        raise ValueError(f"raw state must be finite with shape {(expected,)}, got {value.shape}")
    return np.concatenate(
        [
            value[index * len(schema.STATE_FIELDS) : index * len(schema.STATE_FIELDS) + 10]
            for index in range(arm_count)
        ]
    )


def action_from_joint_command(
    joints: np.ndarray,
    gripper: float,
    robot_state,
    forward_kinematics,
) -> np.ndarray:
    """Reconstruct the TCP action from the exact joint command sent to a controller."""
    flange_position, flange_rotation = forward_kinematics(joints)
    tcp_position, tcp_rotation = schema.tcp_pose_from_flange(
        flange_position,
        flange_rotation,
        robot_state,
    )
    value = np.concatenate(
        (tcp_position, schema.matrix_to_rot6d(tcp_rotation), [float(gripper)])
    ).astype(np.float32)
    if value.shape != (10,) or not np.isfinite(value).all():
        raise ValueError(f"invalid action reconstructed from joint command: {value.shape}")
    return value


def clamp_pose_step(
    current_position: np.ndarray,
    current_rotation: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    *,
    max_linear_step: float,
    max_angular_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(target_position) - np.asarray(current_position)
    distance = float(np.linalg.norm(delta))
    if distance > max_linear_step:
        delta *= max_linear_step / distance
    position = np.asarray(current_position) + delta
    rotation_delta = Rotation.from_matrix(target_rotation @ current_rotation.T).as_rotvec()
    angle = float(np.linalg.norm(rotation_delta))
    if angle > max_angular_step:
        rotation_delta *= max_angular_step / angle
    rotation = Rotation.from_rotvec(rotation_delta).as_matrix() @ current_rotation
    return position, rotation


def smooth_pose_target(
    previous_position: np.ndarray,
    previous_rotation: np.ndarray,
    current_position: np.ndarray,
    current_rotation: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the RTC target EMA in translation and on SO(3)."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError("target smoothing alpha must be in [0, 1)")
    previous_position = np.asarray(previous_position, dtype=np.float64)
    current_position = np.asarray(current_position, dtype=np.float64)
    previous_rotation = np.asarray(previous_rotation, dtype=np.float64)
    current_rotation = np.asarray(current_rotation, dtype=np.float64)
    if previous_position.shape != (3,) or current_position.shape != (3,):
        raise ValueError("pose positions must be 3-vectors")
    if previous_rotation.shape != (3, 3) or current_rotation.shape != (3, 3):
        raise ValueError("pose rotations must be 3x3 matrices")
    if not all(
        np.isfinite(value).all()
        for value in (
            previous_position,
            current_position,
            previous_rotation,
            current_rotation,
        )
    ):
        raise ValueError("pose components must be finite")

    position = alpha * previous_position + (1.0 - alpha) * current_position
    rotation_delta = Rotation.from_matrix(current_rotation @ previous_rotation.T).as_rotvec()
    rotation = (
        Rotation.from_rotvec((1.0 - alpha) * rotation_delta).as_matrix()
        @ previous_rotation
    )
    return position, rotation


def pose_step_to_twist(
    current_position: np.ndarray,
    current_rotation: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    *,
    tracking_tau_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a bounded pose target into the RTC base-frame tracking twist."""
    if tracking_tau_s <= 0.0:
        raise ValueError("tracking time constant must be positive")
    current_position = np.asarray(current_position, dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)
    current_rotation = np.asarray(current_rotation, dtype=np.float64)
    target_rotation = np.asarray(target_rotation, dtype=np.float64)
    if current_position.shape != (3,) or target_position.shape != (3,):
        raise ValueError("pose positions must be 3-vectors")
    if current_rotation.shape != (3, 3) or target_rotation.shape != (3, 3):
        raise ValueError("pose rotations must be 3x3 matrices")
    if not all(
        np.isfinite(value).all()
        for value in (current_position, target_position, current_rotation, target_rotation)
    ):
        raise ValueError("pose components must be finite")
    linear = (target_position - current_position) / tracking_tau_s
    angular = (
        Rotation.from_matrix(target_rotation @ current_rotation.T).as_rotvec()
        / tracking_tau_s
    )
    return linear, angular


def gamepad_tracking_twist(
    current_position: np.ndarray,
    current_rotation: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    target_velocity: np.ndarray,
    *,
    position_gain: float,
    orientation_gain: float,
    max_linear_speed: float,
    max_angular_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the proportional-plus-feedforward twist used by fine teleop."""
    if min(position_gain, orientation_gain, max_linear_speed, max_angular_speed) <= 0.0:
        raise ValueError("gamepad tracking gains and speed limits must be positive")
    target_velocity = np.asarray(target_velocity, dtype=np.float64)
    if target_velocity.shape != (3,) or not np.isfinite(target_velocity).all():
        raise ValueError("gamepad target velocity must be a finite 3-vector")
    linear_error, angular_error = pose_step_to_twist(
        current_position,
        current_rotation,
        target_position,
        target_rotation,
        tracking_tau_s=1.0,
    )
    linear = position_gain * linear_error + target_velocity
    angular = orientation_gain * angular_error
    linear_norm = float(np.linalg.norm(linear))
    if linear_norm > max_linear_speed:
        linear *= max_linear_speed / linear_norm
    angular_norm = float(np.linalg.norm(angular))
    if angular_norm > max_angular_speed:
        angular *= max_angular_speed / angular_norm
    return linear, angular
