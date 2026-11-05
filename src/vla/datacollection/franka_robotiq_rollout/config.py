from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GamepadConfig:
    joy_topic: str
    left_stick_x_axis: int
    left_stick_y_axis: int
    left_trigger_axis: int
    right_trigger_axis: int
    a_button: int
    b_button: int
    x_button: int
    y_button: int
    joint7_button_sign: float
    left_stick_x_sign: float
    left_stick_y_sign: float
    right_stick_x_axis: int = 3
    right_stick_y_axis: int = 4
    right_stick_x_sign: float = -1.0
    right_stick_y_sign: float = 1.0
    tool_roll_button_sign: float = -1.0
    right_stick_rotation_deg: float = 0.0

    def __post_init__(self):
        angle = self.right_stick_rotation_deg
        if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not math.isfinite(angle):
            raise ValueError("right_stick_rotation_deg must be a finite angle")


@dataclass(frozen=True)
class ArmConfig:
    name: str
    robot_state_topic: str
    measured_joint_topic: str
    gripper_state_topic: str
    gamepad: GamepadConfig
    command_joint_topic: str
    command_gripper_topic: str
    gripper_action_topic: str
    gripper_speed_topic: str
    gripper_open_position: float
    gripper_closed_position: float
    gripper_max_effort: float
    robot_state_publisher: str
    base_link: str
    tip_link: str
    gripper_joint_name: str
    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]

    @property
    def gripper_drive_topic(self) -> str:
        from vla.datacollection.gripper_contract import gripper_drive_topic
        return gripper_drive_topic(self.command_gripper_topic)


@dataclass(frozen=True)
class CameraConfig:
    name: str
    serial: str

    @property
    def feature_key(self) -> str:
        return f"observation.images.{self.name}"


@dataclass(frozen=True)
class RolloutConfig:
    backend: str
    dataset_root: Path
    task: str
    fps: int
    image_size: tuple[int, int]
    socket_path: Path
    joy_socket_path: Path
    arms: tuple[ArmConfig, ...]
    cameras: tuple[CameraConfig, ...]
    sync_tolerance_ms: float
    policy_timeout_s: float
    control_discontinuity_timeout_s: float
    rtc_deployment_config: Path | None
    max_linear_velocity: float
    max_angular_velocity: float
    velocity_tracking_tau_s: float
    target_smoothing_alpha: float
    velocity_smoothing: float
    smooth_angular_velocity: bool
    damping: float
    max_joint_speed: float
    joint_limit_margin: float
    gamepad_control_rate: float
    gamepad_joy_timeout_s: float
    gamepad_state_timeout_s: float
    gamepad_deadzone: float
    gamepad_xy_speed: float
    gamepad_z_speed: float
    gamepad_joint7_speed: float
    gamepad_gripper_speed: float
    policy_gripper_speed: float
    gamepad_gripper_trigger_deadzone: float
    gamepad_position_gain: float
    gamepad_orientation_gain: float
    gamepad_max_angular_speed: float
    min_control_command_success_rate: float
    low_success_rate_samples: int
    gamepad_tilt_speed: float = 0.15
    gamepad_roll_speed: float = 0.30
    gamepad_profile: str | None = None
    subtasks: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()

    @property
    def policy_state_dim(self) -> int:
        return 10 * len(self.arms)

    @property
    def action_dim(self) -> int:
        return 10 * len(self.arms)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _vector3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain three values")
    return tuple(float(item) for item in value)


def _resolve_reference(path: str, *, relative_to: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        from_cwd = candidate.resolve()
        from_config = (relative_to / candidate).resolve()
        resolved = from_cwd if from_cwd.is_file() else from_config
    if not resolved.is_file():
        raise FileNotFoundError(f"referenced RTC deployment config is missing: {resolved}")
    return resolved


def _rtc_control_values(
    control: dict[str, Any],
    *,
    config_path: Path,
    fps: int,
    arms: tuple[ArmConfig, ...],
) -> tuple[Path | None, dict[str, float | bool]]:
    profile_value = control.get("rtc_deployment_config")
    if profile_value is None:
        # Keep standalone profiles usable, while accepting the old speed key names.
        return None, {
            "max_linear_velocity": float(
                control.get("max_linear_velocity", control.get("max_linear_speed", 0.04))
            ),
            "max_angular_velocity": float(
                control.get("max_angular_velocity", control.get("max_angular_speed", 0.30))
            ),
            "velocity_tracking_tau_s": float(control.get("velocity_tracking_tau_s", 0.15)),
            "target_smoothing_alpha": float(control.get("target_smoothing_alpha", 0.0)),
            "velocity_smoothing": float(control.get("velocity_smoothing", 0.0)),
            "smooth_angular_velocity": bool(control.get("smooth_angular_velocity", False)),
            "control_discontinuity_timeout_s": float(
                control.get("control_discontinuity_timeout_ms", 100.0)
            )
            / 1000.0,
        }

    profile_path = _resolve_reference(str(profile_value), relative_to=config_path.parent)
    with profile_path.open(encoding="utf-8") as stream:
        profile = _mapping(yaml.safe_load(stream), "RTC deployment config")
    robot = _mapping(profile.get("robot"), "RTC deployment robot config")
    deployment = _mapping(profile.get("deployment"), "RTC deployment config.deployment")
    required = (
        "control_frequency",
        "max_linear_velocity",
        "max_angular_velocity",
        "velocity_tracking_tau_s",
        "target_smoothing_alpha",
        "velocity_smoothing",
        "smooth_angular_velocity",
    )
    missing = [key for key in required if key not in robot]
    if missing:
        raise ValueError(f"RTC deployment robot config is missing: {', '.join(missing)}")
    if float(robot["control_frequency"]) != float(fps):
        raise ValueError(
            "rollout dataset FPS must match RTC robot.control_frequency: "
            f"{fps} != {robot['control_frequency']}"
        )

    workspace = _mapping(robot.get("workspace"), "RTC deployment robot.workspace")
    profile_minimum = _vector3(workspace.get("min"), "RTC deployment workspace.min")
    profile_maximum = _vector3(workspace.get("max"), "RTC deployment workspace.max")
    for arm in arms:
        if any(value < bound for value, bound in zip(arm.workspace_min, profile_minimum)) or any(
            value > bound for value, bound in zip(arm.workspace_max, profile_maximum)
        ):
            raise ValueError(
                f"rollout workspace for {arm.name!r} must stay within the RTC deployment workspace"
            )

    overridden = sorted(
        key
        for key in (
            "max_linear_velocity",
            "max_linear_speed",
            "max_angular_velocity",
            "max_angular_speed",
            "velocity_tracking_tau_s",
            "target_smoothing_alpha",
            "velocity_smoothing",
            "smooth_angular_velocity",
        )
        if key in control
    )
    if overridden:
        raise ValueError(
            "RTC-owned control values cannot be overridden in rollout config: "
            + ", ".join(overridden)
        )
    return profile_path, {
        "max_linear_velocity": float(robot["max_linear_velocity"]),
        "max_angular_velocity": float(robot["max_angular_velocity"]),
        "velocity_tracking_tau_s": float(robot["velocity_tracking_tau_s"]),
        "target_smoothing_alpha": float(robot["target_smoothing_alpha"]),
        "velocity_smoothing": float(robot["velocity_smoothing"]),
        "smooth_angular_velocity": bool(robot["smooth_angular_velocity"]),
        "control_discontinuity_timeout_s": float(
            deployment.get(
                "control_discontinuity_timeout_ms",
                robot.get("velocity_motion_duration_ms", 100),
            )
        )
        / 1000.0,
    }


def _arm(value: Any) -> ArmConfig:
    raw = _mapping(value, "arm")
    gamepad_raw = _mapping(raw.get("gamepad"), f"{raw.get('name', 'arm')}.gamepad")
    required = (
        "name",
        "robot_state_topic",
        "measured_joint_topic",
        "gripper_state_topic",
        "command_joint_topic",
        "command_gripper_topic",
        "gripper_action_topic",
        "gripper_speed_topic",
        "robot_state_publisher",
        "base_link",
        "tip_link",
        "gripper_joint_name",
        "workspace_min",
        "workspace_max",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"arm config is missing: {', '.join(missing)}")
    minimum = _vector3(raw["workspace_min"], f"{raw['name']}.workspace_min")
    maximum = _vector3(raw["workspace_max"], f"{raw['name']}.workspace_max")
    if any(lower >= upper for lower, upper in zip(minimum, maximum)):
        raise ValueError(f"invalid workspace bounds for arm {raw['name']!r}")
    gripper_open_position = float(raw.get("gripper_open_position", 0.0))
    gripper_closed_position = float(raw.get("gripper_closed_position", 0.8))
    gripper_max_effort = float(raw.get("gripper_max_effort", 50.0))
    if abs(gripper_open_position - gripper_closed_position) < 1e-9:
        raise ValueError(f"{raw['name']}.gripper open and closed positions must differ")
    if gripper_max_effort < 0.0:
        raise ValueError(f"{raw['name']}.gripper_max_effort must be non-negative")
    return ArmConfig(
        name=str(raw["name"]),
        robot_state_topic=str(raw["robot_state_topic"]),
        measured_joint_topic=str(raw["measured_joint_topic"]),
        gripper_state_topic=str(raw["gripper_state_topic"]),
        gamepad=GamepadConfig(
            joy_topic=str(gamepad_raw["joy_topic"]),
            left_stick_x_axis=int(gamepad_raw.get("left_stick_x_axis", 0)),
            left_stick_y_axis=int(gamepad_raw.get("left_stick_y_axis", 1)),
            left_trigger_axis=int(gamepad_raw.get("left_trigger_axis", 2)),
            right_trigger_axis=int(gamepad_raw.get("right_trigger_axis", 5)),
            a_button=int(gamepad_raw.get("a_button", 0)),
            b_button=int(gamepad_raw.get("b_button", 1)),
            x_button=int(gamepad_raw.get("x_button", 2)),
            y_button=int(gamepad_raw.get("y_button", 3)),
            joint7_button_sign=float(gamepad_raw.get("joint7_button_sign", -1.0)),
            left_stick_x_sign=float(gamepad_raw.get("left_stick_x_sign", 1.0)),
            left_stick_y_sign=float(gamepad_raw.get("left_stick_y_sign", 1.0)),
            right_stick_x_axis=int(gamepad_raw.get("right_stick_x_axis", 3)),
            right_stick_y_axis=int(gamepad_raw.get("right_stick_y_axis", 4)),
            right_stick_x_sign=float(gamepad_raw.get("right_stick_x_sign", -1.0)),
            right_stick_y_sign=float(gamepad_raw.get("right_stick_y_sign", 1.0)),
            right_stick_rotation_deg=gamepad_raw.get("right_stick_rotation_deg", 0.0),
            tool_roll_button_sign=float(gamepad_raw.get(
                "tool_roll_button_sign", gamepad_raw.get("joint7_button_sign", -1.0)
            )),
        ),
        command_joint_topic=str(raw["command_joint_topic"]),
        command_gripper_topic=str(raw["command_gripper_topic"]),
        gripper_action_topic=str(raw["gripper_action_topic"]),
        gripper_speed_topic=str(raw["gripper_speed_topic"]),
        gripper_open_position=gripper_open_position,
        gripper_closed_position=gripper_closed_position,
        gripper_max_effort=gripper_max_effort,
        robot_state_publisher=str(raw["robot_state_publisher"]),
        base_link=str(raw["base_link"]),
        tip_link=str(raw["tip_link"]),
        gripper_joint_name=str(raw["gripper_joint_name"]),
        workspace_min=minimum,
        workspace_max=maximum,
    )


def load_rollout_config(path: str | Path) -> RolloutConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        raw = _mapping(yaml.safe_load(stream), "rollout config")
    if raw.get("version") != 1:
        raise ValueError("rollout config version must be 1")
    backend = str(raw.get("backend", "ros2"))
    if backend not in {"franky", "ros2"}:
        raise ValueError("rollout backend must be 'franky' or 'ros2'")
    dataset = _mapping(raw.get("dataset"), "dataset")
    task = str(dataset.get("task", "")).strip()
    tasks_raw = dataset.get("tasks", [])
    if tasks_raw is None:
        tasks_raw = []
    if (not isinstance(tasks_raw, list)
            or any(not isinstance(item, str) or not item.strip() for item in tasks_raw)):
        raise ValueError("dataset.tasks must be a list of nonempty task prompts")
    tasks = tuple(dict.fromkeys(item.strip() for item in tasks_raw))
    if task and task not in tasks:
        tasks = (task, *tasks)
    if not tasks:
        raise ValueError("dataset.task must be nonempty")
    subtasks = dataset.get("subtasks", [])
    if (not isinstance(subtasks, list)
            or any(not isinstance(name, str) or not name.strip() for name in subtasks)
            or len(set(subtasks)) != len(subtasks)):
        raise ValueError("dataset.subtasks must be a list of unique nonempty stage labels")
    if subtasks and backend != "ros2":
        raise ValueError("subtask annotations currently require the ROS2 collector")
    transport = _mapping(raw.get("transport"), "transport")
    control = _mapping(raw.get("control"), "control")
    gamepad_profile = control.get("gamepad_profile")
    if gamepad_profile not in (None, "sft_record"):
        raise ValueError("gamepad_profile must be sft_record or omitted")
    if gamepad_profile is not None and backend != "franky":
        raise ValueError("sft_record gamepad profile requires the Franky backend")
    arms_raw = raw.get("arms")
    cameras_raw = raw.get("cameras")
    if not isinstance(arms_raw, list) or len(arms_raw) not in {1, 2}:
        raise ValueError("arms must contain one or two arm configurations")
    if not isinstance(cameras_raw, list) or not cameras_raw:
        raise ValueError("cameras must be a non-empty list")
    arms = tuple(_arm(item) for item in arms_raw)
    if len({arm.name for arm in arms}) != len(arms):
        raise ValueError("arm names must be unique")
    cameras = tuple(
        CameraConfig(name=str(_mapping(item, "camera")["name"]), serial=str(item["serial"]))
        for item in cameras_raw
    )
    if len({camera.name for camera in cameras}) != len(cameras):
        raise ValueError("camera names must be unique")
    fps = int(dataset.get("fps", 30))
    image_size_raw = dataset.get("image_size", [128, 128])
    if not isinstance(image_size_raw, list) or len(image_size_raw) != 2:
        raise ValueError("dataset.image_size must be [height, width]")
    image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
    if any(value <= 0 for value in image_size):
        raise ValueError("dataset.image_size values must be positive")
    rtc_deployment_config, rtc_control = _rtc_control_values(
        control,
        config_path=config_path,
        fps=fps,
        arms=arms,
    )
    positive = {
        "fps": fps,
        "sync_tolerance_ms": float(control.get("sync_tolerance_ms", 100.0)),
        "policy_timeout_s": float(control.get("policy_timeout_s", 2.0)),
        "control_discontinuity_timeout_s": float(
            rtc_control["control_discontinuity_timeout_s"]
        ),
        "damping": float(control.get("damping", 0.05)),
        "max_linear_velocity": float(rtc_control["max_linear_velocity"]),
        "max_angular_velocity": float(rtc_control["max_angular_velocity"]),
        "velocity_tracking_tau_s": float(rtc_control["velocity_tracking_tau_s"]),
        "max_joint_speed": float(control.get("max_joint_speed", 0.30)),
        "joint_limit_margin": float(control.get("joint_limit_margin", 0.05)),
        "gamepad_control_rate": float(control.get("gamepad_control_rate", 100.0)),
        "gamepad_joy_timeout_s": float(control.get("gamepad_joy_timeout_s", 0.2)),
        "gamepad_state_timeout_s": float(control.get("gamepad_state_timeout_s", 0.5)),
        "gamepad_deadzone": float(control.get("gamepad_deadzone", 0.08)),
        "gamepad_xy_speed": float(control.get("gamepad_xy_speed", 0.02)),
        "gamepad_z_speed": float(control.get("gamepad_z_speed", 0.01)),
        "gamepad_joint7_speed": float(control.get("gamepad_joint7_speed", 0.30)),
        "gamepad_tilt_speed": float(control.get("gamepad_tilt_speed", 0.15)),
        "gamepad_roll_speed": float(control.get("gamepad_roll_speed", control.get("gamepad_joint7_speed", 0.30))),
        "gamepad_gripper_speed": float(control.get("gamepad_gripper_speed", 0.40)),
        "policy_gripper_speed": float(control.get("policy_gripper_speed", 1.0)),
        "gamepad_position_gain": float(control.get("gamepad_position_gain", 8.0)),
        "gamepad_orientation_gain": float(control.get("gamepad_orientation_gain", 3.0)),
        "gamepad_max_angular_speed": float(
            control.get("gamepad_max_angular_speed", 0.30)
        ),
    }
    invalid = [name for name, value in positive.items() if not math.isfinite(value) or value <= 0]
    if invalid:
        raise ValueError("configuration values must be positive: " + ", ".join(invalid))
    if positive["control_discontinuity_timeout_s"] <= 1.0 / fps:
        raise ValueError("control discontinuity timeout must exceed one rollout period")
    if positive["gamepad_deadzone"] >= 1.0:
        raise ValueError("gamepad_deadzone must be less than 1.0")
    if positive["gamepad_gripper_speed"] > 1.0:
        raise ValueError("gamepad_gripper_speed must be at most 1.0")
    if positive["policy_gripper_speed"] > 1.0:
        raise ValueError("policy_gripper_speed must be at most 1.0")
    gamepad_gripper_trigger_deadzone = float(
        control.get("gamepad_gripper_trigger_deadzone", 0.05)
    )
    if not 0.0 <= gamepad_gripper_trigger_deadzone < 1.0:
        raise ValueError("gamepad_gripper_trigger_deadzone must be in [0, 1)")
    target_smoothing_alpha = float(rtc_control["target_smoothing_alpha"])
    velocity_smoothing = float(rtc_control["velocity_smoothing"])
    if not 0.0 <= target_smoothing_alpha < 1.0:
        raise ValueError("target_smoothing_alpha must be in [0, 1)")
    if not 0.0 <= velocity_smoothing < 1.0:
        raise ValueError("velocity_smoothing must be in [0, 1)")
    min_success_rate = float(control.get("min_control_command_success_rate", 0.90))
    low_success_rate_samples = int(control.get("low_success_rate_samples", 3))
    if not 0.0 < min_success_rate <= 1.0:
        raise ValueError("min_control_command_success_rate must be in (0, 1]")
    if low_success_rate_samples < 1:
        raise ValueError("low_success_rate_samples must be positive")
    joy_topics = [arm.gamepad.joy_topic for arm in arms]
    if len(set(joy_topics)) != len(joy_topics):
        raise ValueError("each arm must use a distinct gamepad joy_topic")
    for arm in arms:
        indices = (
            arm.gamepad.left_stick_x_axis,
            arm.gamepad.left_stick_y_axis,
            arm.gamepad.left_trigger_axis,
            arm.gamepad.right_trigger_axis,
            arm.gamepad.right_stick_x_axis,
            arm.gamepad.right_stick_y_axis,
            arm.gamepad.a_button,
            arm.gamepad.b_button,
            arm.gamepad.x_button,
            arm.gamepad.y_button,
        )
        if any(index < 0 for index in indices):
            raise ValueError(f"{arm.name}.gamepad axis/button indices must be non-negative")
        if backend == "franky" and len(set(indices[:6])) != 6:
            raise ValueError(f"{arm.name}.gamepad stick and trigger axes must be distinct")
        if arm.gamepad.joint7_button_sign not in {-1.0, 1.0}:
            raise ValueError(
                f"{arm.name}.gamepad.joint7_button_sign must be -1.0 or 1.0"
            )
        for label, value in (
            ("left_stick_x_sign", arm.gamepad.left_stick_x_sign),
            ("left_stick_y_sign", arm.gamepad.left_stick_y_sign),
            ("right_stick_x_sign", arm.gamepad.right_stick_x_sign),
            ("right_stick_y_sign", arm.gamepad.right_stick_y_sign),
            ("tool_roll_button_sign", arm.gamepad.tool_roll_button_sign),
        ):
            if value not in {-1.0, 1.0}:
                raise ValueError(f"{arm.name}.gamepad.{label} must be -1.0 or 1.0")
    return RolloutConfig(
        backend=backend,
        dataset_root=Path(str(dataset["root"])).expanduser().resolve(),
        task=tasks[0],
        fps=fps,
        image_size=image_size,
        socket_path=Path(str(transport["socket_path"])).expanduser().resolve(),
        joy_socket_path=Path(
            str(transport.get("joy_socket_path", "/tmp/vla_rollout_joy.sock"))
        ).expanduser().resolve(),
        arms=arms,
        cameras=cameras,
        rtc_deployment_config=rtc_deployment_config,
        gamepad_profile=gamepad_profile,
        subtasks=tuple(subtasks),
        tasks=tasks,
        target_smoothing_alpha=target_smoothing_alpha,
        velocity_smoothing=velocity_smoothing,
        smooth_angular_velocity=bool(rtc_control["smooth_angular_velocity"]),
        gamepad_gripper_trigger_deadzone=gamepad_gripper_trigger_deadzone,
        min_control_command_success_rate=min_success_rate,
        low_success_rate_samples=low_success_rate_samples,
        **{name: float(value) for name, value in positive.items() if name != "fps"},
    )
