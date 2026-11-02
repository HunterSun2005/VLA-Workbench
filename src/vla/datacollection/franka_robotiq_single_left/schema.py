"""Absolute Franka TCP schema shared by the recorder and offline tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
from vla.paths import workspace
REPO_ROOT = workspace()
OUTPUT_ROOT = Path(
    os.environ.get("LEFT_FRANKA_DATASET_ROOT", REPO_ROOT / "datasets" / "lerobot_dataset_v2")
).expanduser()
TASK = os.environ.get("LEFT_FRANKA_TASK", "Place the blue block on the white paper.")
FPS = 30
MAX_SYNC_DELTA_NS = 100_000_000

TOPIC_ROBOT_STATE = os.environ.get(
    "LEFT_FRANKA_STATE_TOPIC", "/left/franka_robot_state_broadcaster/robot_state"
)
TOPIC_GRIPPER = os.environ.get(
    "LEFT_GRIPPER_STATE_TOPIC", "/left/gripper/joint_state_broadcaster_robotiq/joint_states"
)
TOPIC_ARM_COMMAND = os.environ.get(
    "LEFT_ARM_COMMAND_TOPIC", "/left/teleop/joint_states"
)
DEFAULT_GRIPPER_COMMAND_TOPIC = "/left/teleop/gripper_action_width_percent"
TOPIC_GRIPPER_COMMAND = os.environ.get(
    "LEFT_GRIPPER_COMMAND_TOPIC", DEFAULT_GRIPPER_COMMAND_TOPIC
)

ROBOTIQ_JOINT_NAME = "robotiq_85_left_knuckle_joint"
ROBOTIQ_OPEN_POSITION_RAD = 0.0
ROBOTIQ_CLOSED_POSITION_RAD = 0.7929
ROBOTIQ_MAX_OPENING_WIDTH_M = 0.085

TCP_STATE_FIELDS = [
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
]
EXTENDED_STATE_FIELDS = [
    "base_tcp_force_x",
    "base_tcp_force_y",
    "base_tcp_force_z",
    "base_tcp_torque_x",
    "base_tcp_torque_y",
    "base_tcp_torque_z",
    "base_tcp_linear_velocity_x",
    "base_tcp_linear_velocity_y",
    "base_tcp_linear_velocity_z",
    "base_tcp_angular_velocity_x",
    "base_tcp_angular_velocity_y",
    "base_tcp_angular_velocity_z",
]
JOINT_STATE_FIELDS = [f"joint_position_{index}" for index in range(1, 8)]
LEGACY_STATE_FIELDS = TCP_STATE_FIELDS + EXTENDED_STATE_FIELDS + JOINT_STATE_FIELDS
JOINT_TORQUE_FIELDS = [f"joint_torque_{index}" for index in range(1, 8)]
EXTERNAL_JOINT_TORQUE_FIELDS = [f"external_joint_torque_{index}" for index in range(1, 8)]
JOINT_VELOCITY_FIELDS = [f"joint_velocity_{index}" for index in range(1, 8)]
STATE_FIELDS = (LEGACY_STATE_FIELDS + JOINT_TORQUE_FIELDS
                + EXTERNAL_JOINT_TORQUE_FIELDS + JOINT_VELOCITY_FIELDS)
ACTION_FIELDS = [
    "desired_tcp_position_x",
    "desired_tcp_position_y",
    "desired_tcp_position_z",
    "desired_tcp_rot6d_col0_x",
    "desired_tcp_rot6d_col0_y",
    "desired_tcp_rot6d_col0_z",
    "desired_tcp_rot6d_col1_x",
    "desired_tcp_rot6d_col1_y",
    "desired_tcp_rot6d_col1_z",
    "gripper",
]
# A stationary controller target still jitters at sub-millimetre scale. Require
# deliberate episode motion so noise cannot make a semantically empty action
# stream look valid.
ACTION_MIN_TRANSLATION_SPAN_M = 5e-3
ACTION_MIN_ROTATION_SPAN_RAD = 1e-2


def normalize_joint_name(name: str) -> str:
    return name.strip("/").split("/")[-1]


def canonical_franka_joint_name(name: str) -> str:
    clean = normalize_joint_name(name)
    for prefix in ("left_", "right_"):
        if clean.startswith(prefix):
            clean = clean.removeprefix(prefix)
            break
    return clean


def measured_joint_positions(robot_state_message) -> np.ndarray:
    """Return measured J1-J7 positions from a Franka robot-state message."""
    return measured_joint_values(robot_state_message.measured_joint_state, "position")


def measured_joint_values(joint_state, field: str) -> np.ndarray:
    """Read a finite J1-J7 vector, reordering named ROS position/velocity/effort."""
    positions = np.asarray(getattr(joint_state, field), dtype=np.float64)
    if positions.shape != (7,) or not np.isfinite(positions).all():
        raise ValueError(f"measured Franka joint {field} must be a finite 7-vector")

    names = list(joint_state.name)
    if names:
        if len(names) != 7:
            raise ValueError("measured Franka joint state must contain seven joint names")
        by_name = {
            canonical_franka_joint_name(name): position
            for name, position in zip(names, positions)
        }
        expected = tuple(f"fr3_joint{index}" for index in range(1, 8))
        if len(by_name) != 7 or any(name not in by_name for name in expected):
            raise ValueError(
                "measured Franka joint names must identify fr3_joint1 through fr3_joint7"
            )
        positions = np.asarray([by_name[name] for name in expected], dtype=np.float64)
    return positions


def gripper_width(message, joint_name: str = ROBOTIQ_JOINT_NAME) -> float:
    """Return the measured Robotiq opening width in metres."""
    expected = normalize_joint_name(joint_name)
    for name, position in zip(message.name, message.position):
        clean_name = normalize_joint_name(name)
        if clean_name == expected or clean_name.endswith(expected):
            position = float(position)
            if not np.isfinite(position):
                raise ValueError("non-finite measured gripper position")
            tolerance = 0.05
            if not (
                ROBOTIQ_OPEN_POSITION_RAD - tolerance
                <= position
                <= ROBOTIQ_CLOSED_POSITION_RAD + tolerance
            ):
                raise ValueError(f"measured gripper joint is outside its physical range: {position}")
            closed_fraction = np.clip(
                (position - ROBOTIQ_OPEN_POSITION_RAD)
                / (ROBOTIQ_CLOSED_POSITION_RAD - ROBOTIQ_OPEN_POSITION_RAD),
                0.0,
                1.0,
            )
            return float(ROBOTIQ_MAX_OPENING_WIDTH_M * (1.0 - closed_fraction))
    raise KeyError(joint_name)


def gripper_action(message) -> float:
    """Return the synchronized teleop command (0=closed, 1=open)."""
    value = float(message.data)
    if not np.isfinite(value):
        raise ValueError("non-finite gripper command")
    if value < -1e-3 or value > 1.0 + 1e-3:
        raise ValueError(f"gripper command is outside [0, 1]: {value}")
    return float(np.clip(value, 0.0, 1.0))


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion must be a finite 4-vector in xyzw order")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-9:
        raise ValueError("zero-norm quaternion")
    return Rotation.from_quat(quaternion / norm).as_matrix()


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation matrix must be finite with shape (3, 3)")
    return np.concatenate((matrix[:, 0], matrix[:, 1]))


def quaternion_to_rot6d(quaternion: np.ndarray) -> np.ndarray:
    return matrix_to_rot6d(quaternion_to_matrix(quaternion))


def rot6d_to_matrix(rot6d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Decode two rotation columns with the standard Gram-Schmidt projection."""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    if rot6d.shape != (6,) or not np.isfinite(rot6d).all():
        raise ValueError("rot6d must be a finite 6-vector")
    first = rot6d[:3]
    first_norm = np.linalg.norm(first)
    if first_norm < eps:
        raise ValueError("rot6d first column has zero norm")
    first = first / first_norm
    second = rot6d[3:] - np.dot(first, rot6d[3:]) * first
    second_norm = np.linalg.norm(second)
    if second_norm < eps:
        raise ValueError("rot6d columns are collinear")
    second = second / second_norm
    third = np.cross(first, second)
    return np.column_stack((first, second, third))


def pose_from_stamped(pose_stamped) -> tuple[np.ndarray, np.ndarray]:
    pose = pose_stamped.pose
    position = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
    quaternion = np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        dtype=np.float64,
    )
    if not np.isfinite(position).all():
        raise ValueError("non-finite TCP position")
    matrix = quaternion_to_matrix(quaternion)
    return position, matrix


def measured_pose_from_robot_state(message) -> tuple[np.ndarray, np.ndarray]:
    return pose_from_stamped(message.o_t_ee)


def desired_pose_from_robot_state(message) -> tuple[np.ndarray, np.ndarray]:
    return pose_from_stamped(message.o_t_ee_d)


def tcp_pose_from_flange(
    flange_position: np.ndarray,
    flange_rotation: np.ndarray,
    robot_state_message,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Franka's configured F_T_EE to an FK-computed flange pose."""
    flange_position = np.asarray(flange_position, dtype=np.float64)
    flange_rotation = np.asarray(flange_rotation, dtype=np.float64)
    if flange_position.shape != (3,) or flange_rotation.shape != (3, 3):
        raise ValueError("flange pose must have shapes (3,) and (3, 3)")
    ee_position, ee_rotation = pose_from_stamped(robot_state_message.f_t_ee)
    return (
        flange_position + flange_rotation @ ee_position,
        flange_rotation @ ee_rotation,
    )


def commanded_joint_positions(message, expected_names: tuple[str, ...]) -> np.ndarray:
    """Read the final controller input in kinematic-chain order."""
    if len(message.name) < len(expected_names) or len(message.position) < len(expected_names):
        raise ValueError(
            f"arm command must contain {len(expected_names)} named positions, "
            f"got names={len(message.name)}, positions={len(message.position)}"
        )
    received = {
        canonical_franka_joint_name(name): float(value)
        for name, value in zip(message.name, message.position)
    }
    if len(received) != len(message.name):
        raise ValueError("arm command contains duplicate joint names")
    values = []
    for expected in expected_names:
        clean_expected = canonical_franka_joint_name(expected)
        if clean_expected not in received:
            raise ValueError(
                f"arm command does not contain expected joint {expected!r}; "
                f"received={list(message.name)!r}"
            )
        values.append(received[clean_expected])
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(expected_names),) or not np.isfinite(result).all():
        raise ValueError("arm command positions must be finite")
    return result


def base_tcp_velocity(
    previous_position: np.ndarray | None,
    previous_rotation: np.ndarray | None,
    previous_time_ns: int | None,
    position: np.ndarray,
    rotation: np.ndarray,
    time_ns: int,
) -> np.ndarray:
    if previous_position is None or previous_rotation is None or previous_time_ns is None:
        return np.zeros(6, dtype=np.float64)
    if time_ns <= previous_time_ns:
        return np.zeros(6, dtype=np.float64)
    dt = (time_ns - previous_time_ns) / 1_000_000_000.0
    linear = (position - previous_position) / dt
    angular = Rotation.from_matrix(rotation @ previous_rotation.T).as_rotvec() / dt
    return np.concatenate((linear, angular))


def state_vector(
    robot_state_message,
    gripper_message,
    velocity: np.ndarray,
    position: np.ndarray | None = None,
    rotation: np.ndarray | None = None,
    gripper_joint_name: str = ROBOTIQ_JOINT_NAME,
    joint_positions: np.ndarray | None = None,
) -> np.ndarray:
    if position is None or rotation is None:
        position, rotation = measured_pose_from_robot_state(robot_state_message)
    # STATE_FIELDS explicitly declare a base-frame wrench. k_f_ext_hat_k is
    # expressed in the stiffness frame and must not be stored under these names.
    wrench = robot_state_message.o_f_ext_hat_k.wrench
    force = np.array([wrench.force.x, wrench.force.y, wrench.force.z])
    torque = np.array([wrench.torque.x, wrench.torque.y, wrench.torque.z])
    if joint_positions is None:
        joint_positions = measured_joint_positions(robot_state_message)
    joint_positions = np.asarray(joint_positions, dtype=np.float64)
    if joint_positions.shape != (7,) or not np.isfinite(joint_positions).all():
        raise ValueError("joint positions must be a finite 7-vector")
    value = np.concatenate(
        (
            position,
            matrix_to_rot6d(rotation),
            [gripper_width(gripper_message, gripper_joint_name)],
            force,
            torque,
            velocity,
            joint_positions,
            measured_joint_values(robot_state_message.measured_joint_state, "effort"),
            measured_joint_values(robot_state_message.tau_ext_hat_filtered, "effort"),
            measured_joint_values(robot_state_message.measured_joint_state, "velocity"),
        )
    ).astype(np.float32)
    if value.shape != (len(STATE_FIELDS),) or not np.isfinite(value).all():
        raise ValueError(f"invalid absolute state: shape={value.shape}")
    return value


def action_vector_from_pose(
    position: np.ndarray,
    rotation: np.ndarray,
    gripper_command_message,
) -> np.ndarray:
    """Build an absolute action from FK of the controller's final joint target."""
    position = np.asarray(position, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("commanded TCP position must be a finite 3-vector")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("commanded TCP rotation must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError("commanded TCP rotation is not a proper rotation matrix")
    value = np.concatenate(
        (position, matrix_to_rot6d(rotation), [gripper_action(gripper_command_message)])
    ).astype(np.float32)
    if value.shape != (10,) or not np.isfinite(value).all():
        raise ValueError(f"invalid absolute action: shape={value.shape}")
    return value


def action_vector(robot_state_message, gripper_command_message) -> np.ndarray:
    """Build an action for legacy Cartesian controllers that update o_t_ee_d."""
    position, rotation = desired_pose_from_robot_state(robot_state_message)
    return action_vector_from_pose(position, rotation, gripper_command_message)


def validate_episode_features(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    require_action_motion: bool = True,
) -> dict[str, np.ndarray | float]:
    """Reject corrupt or semantically degenerate episodes before serialization."""
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != len(STATE_FIELDS):
        raise ValueError(f"invalid state episode shape: {states.shape}")
    if actions.shape != (len(states), len(ACTION_FIELDS)):
        raise ValueError(f"invalid action episode shape: {actions.shape}")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("state/action episode contains NaN or infinity")
    if np.any((actions[:, 9] < 0.0) | (actions[:, 9] > 1.0)):
        raise ValueError("gripper action is outside [0, 1]")
    if np.any((states[:, 9] < 0.0) | (states[:, 9] > ROBOTIQ_MAX_OPENING_WIDTH_M + 1e-6)):
        raise ValueError("measured gripper width is outside the Robotiq range")

    action_rotations = np.stack([rot6d_to_matrix(value) for value in actions[:, 3:9]])
    translation_span = float(np.linalg.norm(np.ptp(actions[:, :3], axis=0)))
    reference = action_rotations[0]
    rotation_span = max(
        float(Rotation.from_matrix(rotation @ reference.T).magnitude())
        for rotation in action_rotations
    )
    if require_action_motion and (
        translation_span < ACTION_MIN_TRANSLATION_SPAN_M
        and rotation_span < ACTION_MIN_ROTATION_SPAN_RAD
    ):
        raise ValueError(
            "commanded TCP action is constant: "
            f"translation_span={translation_span:.6g}m, "
            f"rotation_span={rotation_span:.6g}rad; "
            "discarding episode to prevent a degenerate checkpoint"
        )
    return {
        "state_range": np.ptp(states, axis=0),
        "action_range": np.ptp(actions, axis=0),
        "translation_span_m": translation_span,
        "rotation_span_rad": rotation_span,
    }
