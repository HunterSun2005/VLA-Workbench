"""Absolute dual-Franka TCP schema for LeRobot v3.0 SFT recording."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vla.datacollection.franka_robotiq_single_left import schema as single_schema


SCRIPT_DIR = Path(__file__).resolve().parent
from vla.paths import workspace
REPO_ROOT = workspace()
OUTPUT_ROOT = Path(
    os.environ.get("DUAL_FRANKA_DATASET_ROOT", REPO_ROOT / "datasets" / "dual_lerobot_dataset")
).expanduser()
TASK = os.environ.get("DUAL_FRANKA_TASK", "Complete the bimanual manipulation task.")
FPS = single_schema.FPS
MAX_SYNC_DELTA_NS = single_schema.MAX_SYNC_DELTA_NS


@dataclass(frozen=True)
class ArmSchema:
    name: str
    robot_state_topic: str
    command_joint_topic: str
    gripper_state_topic: str
    gripper_command_topic: str
    robot_state_publisher: str
    base_link: str
    tip_link: str
    gripper_joint_name: str

    @property
    def robot_state_key(self) -> str:
        return f"{self.name}.robot_state"

    @property
    def gripper_state_key(self) -> str:
        return f"{self.name}.gripper_state"

    @property
    def command_joint_key(self) -> str:
        return f"{self.name}.command_joint"

    @property
    def gripper_command_key(self) -> str:
        return f"{self.name}.gripper_command"


ARMS = (
    ArmSchema(
        name="left",
        robot_state_topic=os.environ.get(
            "LEFT_FRANKA_STATE_TOPIC",
            "/left/franka_robot_state_broadcaster/robot_state",
        ),
        command_joint_topic=os.environ.get(
            "LEFT_ARM_COMMAND_TOPIC",
            "/left/teleop/joint_states",
        ),
        gripper_state_topic=os.environ.get(
            "LEFT_GRIPPER_STATE_TOPIC",
            "/left/gripper/joint_state_broadcaster_robotiq/joint_states",
        ),
        gripper_command_topic=os.environ.get(
            "LEFT_GRIPPER_COMMAND_TOPIC",
            "/left/teleop/gripper_action_width_percent",
        ),
        robot_state_publisher=os.environ.get(
            "LEFT_ROBOT_STATE_PUBLISHER",
            "/left/robot_state_publisher",
        ),
        base_link=os.environ.get("LEFT_FRANKA_BASE_LINK", "left_fr3_link0"),
        tip_link=os.environ.get("LEFT_FRANKA_TIP_LINK", "left_fr3_link8"),
        gripper_joint_name=os.environ.get(
            "LEFT_ROBOTIQ_JOINT_NAME",
            "robotiq_85_left_knuckle_joint",
        ),
    ),
    ArmSchema(
        name="right",
        robot_state_topic=os.environ.get(
            "RIGHT_FRANKA_STATE_TOPIC",
            "/right/franka_robot_state_broadcaster/robot_state",
        ),
        command_joint_topic=os.environ.get(
            "RIGHT_ARM_COMMAND_TOPIC",
            "/right/teleop/joint_states",
        ),
        gripper_state_topic=os.environ.get(
            "RIGHT_GRIPPER_STATE_TOPIC",
            "/right/gripper/joint_state_broadcaster_robotiq/joint_states",
        ),
        gripper_command_topic=os.environ.get(
            "RIGHT_GRIPPER_COMMAND_TOPIC",
            "/right/gello/gripper_width_percent",
        ),
        robot_state_publisher=os.environ.get(
            "RIGHT_ROBOT_STATE_PUBLISHER",
            "/right/robot_state_publisher",
        ),
        base_link=os.environ.get("RIGHT_FRANKA_BASE_LINK", "right_fr3_link0"),
        tip_link=os.environ.get("RIGHT_FRANKA_TIP_LINK", "right_fr3_link8"),
        gripper_joint_name=os.environ.get(
            "RIGHT_ROBOTIQ_JOINT_NAME",
            # Both namespaced Robotiq drivers publish this URDF joint name.
            "robotiq_85_left_knuckle_joint",
        ),
    ),
)

CAMERA_SERIALS = {
    "base": os.environ.get("BASE_CAMERA_SERIAL", "260722303785"),
    "left_wrist": os.environ.get("LEFT_WRIST_CAMERA_SERIAL", "346222073620"),
    "right_wrist": os.environ.get("RIGHT_WRIST_CAMERA_SERIAL", "346122070471"),
}
CAMERA_FEATURES = {
    "observation.images.base": "base",
    "observation.images.left_wrist": "left_wrist",
    "observation.images.right_wrist": "right_wrist",
}


def _prefixed_fields(fields: list[str]) -> tuple[str, ...]:
    return tuple(f"{arm.name}_{field}" for arm in ARMS for field in fields)


STATE_FIELDS = _prefixed_fields(single_schema.STATE_FIELDS)
ACTION_FIELDS = _prefixed_fields(single_schema.ACTION_FIELDS)


def arm_state_vector(
    arm: ArmSchema,
    robot_state_message,
    gripper_message,
    velocity: np.ndarray,
    position: np.ndarray | None = None,
    rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Build one arm's 29D absolute measured state."""
    return single_schema.state_vector(
        robot_state_message,
        gripper_message,
        velocity,
        position,
        rotation,
        gripper_joint_name=arm.gripper_joint_name,
    )


def arm_action_vector(
    position: np.ndarray,
    rotation: np.ndarray,
    gripper_command_message,
) -> np.ndarray:
    """Build one arm's 10D action from the controller-consumed joint target."""
    return single_schema.action_vector_from_pose(
        position,
        rotation,
        gripper_command_message,
    )


def dual_state_vector(parts: dict[str, np.ndarray]) -> np.ndarray:
    value = np.concatenate([np.asarray(parts[arm.name], dtype=np.float32) for arm in ARMS])
    if value.shape != (len(STATE_FIELDS),) or not np.isfinite(value).all():
        raise ValueError(f"invalid dual-arm absolute state: shape={value.shape}")
    return value


def dual_action_vector(parts: dict[str, np.ndarray]) -> np.ndarray:
    value = np.concatenate([np.asarray(parts[arm.name], dtype=np.float32) for arm in ARMS])
    if value.shape != (len(ACTION_FIELDS),) or not np.isfinite(value).all():
        raise ValueError(f"invalid dual-arm absolute action: shape={value.shape}")
    return value


def validate_episode_features(
    states: np.ndarray,
    actions: np.ndarray,
) -> dict[str, dict[str, np.ndarray | float]]:
    """Validate a dual episode and reject a fully stationary TCP command stream."""
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != len(STATE_FIELDS):
        raise ValueError(f"invalid dual state episode shape: {states.shape}")
    if actions.shape != (len(states), len(ACTION_FIELDS)):
        raise ValueError(f"invalid dual action episode shape: {actions.shape}")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("dual state/action episode contains NaN or infinity")

    reports = {}
    stationary_arms = []
    state_width = len(single_schema.STATE_FIELDS)
    action_width = len(single_schema.ACTION_FIELDS)
    for index, arm in enumerate(ARMS):
        report = single_schema.validate_episode_features(
            states[:, index * state_width : (index + 1) * state_width],
            actions[:, index * action_width : (index + 1) * action_width],
            require_action_motion=False,
        )
        reports[arm.name] = report
        if (
            report["translation_span_m"] < single_schema.ACTION_MIN_TRANSLATION_SPAN_M
            and report["rotation_span_rad"] < single_schema.ACTION_MIN_ROTATION_SPAN_RAD
        ):
            stationary_arms.append(arm.name)

    if len(stationary_arms) == len(ARMS):
        spans = ", ".join(
            f"{arm.name}=(translation={reports[arm.name]['translation_span_m']:.6g}m, "
            f"rotation={reports[arm.name]['rotation_span_rad']:.6g}rad)"
            for arm in ARMS
        )
        raise ValueError(
            f"both commanded TCP actions are constant: {spans}; "
            "discarding episode to prevent a degenerate checkpoint"
        )
    return reports
