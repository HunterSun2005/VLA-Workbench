from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


FRANKY_COMPLIANCE_DEFAULTS = {
    "translational_stiffness": 1100.0,
    "rotational_stiffness": 65.0,
    "nullspace_stiffness": 20.0,
    "translational_clip": 0.008,
    "rotational_clip": 0.04,
    "max_step": 0.03,
    "max_step_rad": 0.10,
    "max_delta_tau": 0.2,
    "gains_time_constant": 0.1,
}


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
    right_stick_rotation_deg: float = -45.0
    input_layout: str = "raw_joy"
    speed_button: int = 4
    phase_button: int = 5
    upright_button: int = 7
    reset_pose_button: int = 6
    yaw_reset_button: int = 10

    def __post_init__(self):
        if self.input_layout not in ("standard_sdl", "raw_joy"):
            raise ValueError("gamepad input_layout must be standard_sdl or raw_joy")
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
    worker_socket_path: Path
    worker_python: Path | None
    franky_compliance: dict[str, float]
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
    max_linear_velocity: float
    max_angular_velocity: float
    joint_limit_margin: float
    gamepad_joy_timeout_s: float
    gamepad_state_timeout_s: float
    gamepad_deadzone: float
    gamepad_xy_speed: float
    gamepad_z_speed: float
    gamepad_joint7_speed: float
    gamepad_gripper_speed: float
    policy_gripper_speed: float
    gamepad_gripper_trigger_deadzone: float
    gamepad_max_angular_speed: float
    min_control_command_success_rate: float
    low_success_rate_samples: int
    gamepad_hold_timeout_s: float = 0.1
    gello_sample_timeout_s: float = 0.2
    gamepad_tilt_speed: float = 0.225
    gamepad_roll_speed: float = 0.45
    action_space: str = "tcp"
    input_source: str = "gamepad"
    gamepad_profile: str | None = None
    subtasks: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    inference_slots: int = 4
    installation_guard_slots: int = 1
    workspace_limits_enabled: bool = True

    @property
    def policy_state_dim(self) -> int:
        return 10 * len(self.arms)

    @property
    def action_dim(self) -> int:
        return self.action_spec.dim

    @property
    def action_spec(self):
        from vla.actions import ActionSpec
        return ActionSpec(self.action_space, tuple(arm.name for arm in self.arms))


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _vector3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain three values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f'{label} must be finite')
    return result


def _arm(value: Any) -> ArmConfig:
    raw = _mapping(value, "arm")
    if "command_joint_topic" in raw:
        raise ValueError("removed ROS arm command_joint_topic; Franky accepts TCP through IPC")
    gamepad_raw = _mapping(raw.get("gamepad"), f"{raw.get('name', 'arm')}.gamepad")
    required = (
        "name",
        "robot_state_topic",
        "measured_joint_topic",
        "gripper_state_topic",
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
    layout = gamepad_raw.get("input_layout", "raw_joy")
    from .gamepad_layout import axis_buttons, validate_layout
    validate_layout(gamepad_raw)
    defaults = axis_buttons(layout)
    return ArmConfig(
        name=str(raw["name"]),
        robot_state_topic=str(raw["robot_state_topic"]),
        measured_joint_topic=str(raw["measured_joint_topic"]),
        gripper_state_topic=str(raw["gripper_state_topic"]),
        gamepad=GamepadConfig(
            joy_topic=str(gamepad_raw["joy_topic"]),
            input_layout=layout,
            **{k: int(gamepad_raw.get(k, defaults[k])) for k in ("speed_button", "phase_button", "upright_button", "reset_pose_button", "yaw_reset_button")},
            left_stick_x_axis=int(gamepad_raw.get("left_stick_x_axis", defaults["left_stick_x_axis"])),
            left_stick_y_axis=int(gamepad_raw.get("left_stick_y_axis", defaults["left_stick_y_axis"])),
            left_trigger_axis=int(gamepad_raw.get("left_trigger_axis", defaults["left_trigger_axis"])),
            right_trigger_axis=int(gamepad_raw.get("right_trigger_axis", defaults["right_trigger_axis"])),
            a_button=int(gamepad_raw.get("a_button", defaults["a_button"])),
            b_button=int(gamepad_raw.get("b_button", defaults["b_button"])),
            x_button=int(gamepad_raw.get("x_button", defaults["x_button"])),
            y_button=int(gamepad_raw.get("y_button", defaults["y_button"])),
            joint7_button_sign=float(gamepad_raw.get("joint7_button_sign", -1.0)),
            left_stick_x_sign=float(gamepad_raw.get("left_stick_x_sign", 1.0)),
            left_stick_y_sign=float(gamepad_raw.get("left_stick_y_sign", 1.0)),
            right_stick_x_axis=int(gamepad_raw.get("right_stick_x_axis", defaults["right_stick_x_axis"])),
            right_stick_y_axis=int(gamepad_raw.get("right_stick_y_axis", defaults["right_stick_y_axis"])),
            right_stick_x_sign=float(gamepad_raw.get("right_stick_x_sign", -1.0)),
            right_stick_y_sign=float(gamepad_raw.get("right_stick_y_sign", 1.0)),
            right_stick_rotation_deg=gamepad_raw.get("right_stick_rotation_deg", -45.0),
            tool_roll_button_sign=float(gamepad_raw.get(
                "tool_roll_button_sign", gamepad_raw.get("joint7_button_sign", -1.0)
            )),
        ),
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
    if "robot" in raw:
        raise ValueError("Legacy Franky configuration requires explicit migration to the Cartesian worker")
    if raw.get("version") != 1:
        raise ValueError("rollout config version must be 1")
    backend = str(raw.get("backend", ""))
    if backend != "franky" or raw.get("action_space", "tcp") != "tcp":
        raise ValueError("new sessions require backend: franky and action_space: tcp")
    from vla.actions import ActionSpec
    ActionSpec(space=str(raw.get("action_space", "tcp")))
    if raw.get("input_source", "gamepad") not in ("gamepad", "gello"):
        raise ValueError("input_source must be gamepad or gello")
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
    if backend == "franky" and str(raw.get("action_space", "tcp")) != "tcp":
        raise ValueError("Franky Cartesian impedance requires action_space: tcp")
    transport = _mapping(raw.get("transport"), "transport")
    control = _mapping(raw.get("control"), "control")
    external_settings = (set(raw) | set(control)) & {"model_backend", "policy_endpoint"}
    if external_settings:
        raise ValueError("external model settings are not supported: " + ", ".join(sorted(external_settings)))
    workspace_limits_enabled = control.get('workspace_limits_enabled', True)
    if type(workspace_limits_enabled) is not bool:
        raise ValueError('control.workspace_limits_enabled must be boolean')
    compliance_raw = _mapping(control.get("compliance", {}), "control.compliance")
    compliance_keys = {
        "translational_stiffness", "rotational_stiffness", "nullspace_stiffness",
        "translational_clip", "rotational_clip", "max_step", "max_step_rad",
        "max_delta_tau", "gains_time_constant",
    }
    unknown_compliance = set(compliance_raw) - compliance_keys
    if unknown_compliance:
        raise ValueError("unknown Franky compliance parameters: " + ", ".join(sorted(unknown_compliance)))
    franky_compliance = {**FRANKY_COMPLIANCE_DEFAULTS,
                         **{key: float(value) for key, value in compliance_raw.items()}}
    if not (0 < franky_compliance['translational_stiffness'] <= 1200):
        raise ValueError('control.compliance.translational_stiffness must be in (0, 1200]')
    if not (0 < franky_compliance['rotational_stiffness'] <= 80):
        raise ValueError('control.compliance.rotational_stiffness must be in (0, 80]')
    if franky_compliance['nullspace_stiffness'] < 0:
        raise ValueError('control.compliance.nullspace_stiffness must be non-negative')
    if franky_compliance['translational_clip'] < .005:
        raise ValueError('control.compliance.translational_clip must be at least 0.005')
    if franky_compliance['rotational_clip'] < .02:
        raise ValueError('control.compliance.rotational_clip must be at least 0.02')
    if any(franky_compliance[key] <= 0 for key in ('max_step', 'max_step_rad', 'max_delta_tau', 'gains_time_constant')):
        raise ValueError('control.compliance step and tracker parameters must be positive')
    gamepad_profile = control.get("gamepad_profile")
    if gamepad_profile not in (None, "sft_record"):
        raise ValueError("gamepad_profile must be sft_record or omitted")
    if gamepad_profile is not None:
        raise ValueError("Legacy gamepad profile: migrate to the ROS2 control configuration")
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
    obsolete = set(control) & set(['damping', 'gamepad_control_rate', 'gamepad_orientation_gain', 'gamepad_position_gain', 'max_joint_speed', 'rtc_deployment_config', 'smooth_angular_velocity', 'target_smoothing_alpha', 'velocity_smoothing', 'velocity_tracking_tau_s'])
    if obsolete:
        raise ValueError('removed ROS tracking settings: ' + ', '.join(sorted(obsolete)))
    positive = {
        "fps": fps,
        "sync_tolerance_ms": float(control.get("sync_tolerance_ms", 100.0)),
        "policy_timeout_s": float(control.get("policy_timeout_s", 2.0)),
        "control_discontinuity_timeout_s": float(control.get("control_discontinuity_timeout_ms", 100.0)) / 1000.0,
        "max_linear_velocity": float(control.get("max_linear_velocity", 0.16)),
        "max_angular_velocity": float(control.get("max_angular_velocity", 0.45)),
        "joint_limit_margin": float(control.get("joint_limit_margin", 0.05)),
        "gamepad_joy_timeout_s": float(control.get("gamepad_joy_timeout_s", 0.3)),
        "gamepad_hold_timeout_s": float(control.get("gamepad_hold_timeout_s", 0.1)),
        "gello_sample_timeout_s": float(control.get("gello_sample_timeout_s", 0.2)),
        "gamepad_state_timeout_s": float(control.get("gamepad_state_timeout_s", 0.5)),
        "gamepad_deadzone": float(control.get("gamepad_deadzone", 0.08)),
        "gamepad_xy_speed": float(control.get("gamepad_xy_speed", 0.03)),
        "gamepad_z_speed": float(control.get("gamepad_z_speed", 0.015)),
        "gamepad_joint7_speed": float(control.get("gamepad_joint7_speed", 0.45)),
        "gamepad_tilt_speed": float(control.get("gamepad_tilt_speed", 0.225)),
        "gamepad_roll_speed": float(control.get("gamepad_roll_speed", control.get("gamepad_joint7_speed", 0.45))),
        "gamepad_gripper_speed": float(control.get("gamepad_gripper_speed", 0.40)),
        "policy_gripper_speed": float(control.get("policy_gripper_speed", 1.0)),
        "gamepad_max_angular_speed": float(
            control.get("gamepad_max_angular_speed", 0.45)
        ),
    }
    invalid = [name for name, value in positive.items() if not math.isfinite(value) or value <= 0]
    if invalid:
        raise ValueError("configuration values must be positive: " + ", ".join(invalid))
    if positive["control_discontinuity_timeout_s"] <= 1.0 / fps:
        raise ValueError("control discontinuity timeout must exceed one rollout period")
    if not 1.0 / fps < positive["gamepad_hold_timeout_s"] < positive["gamepad_joy_timeout_s"]:
        raise ValueError("require control period < gamepad_hold_timeout_s < gamepad_joy_timeout_s")
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
            arm.gamepad.speed_button, arm.gamepad.phase_button,
            arm.gamepad.upright_button, arm.gamepad.reset_pose_button, arm.gamepad.yaw_reset_button,
        )
        if any(index < 0 for index in indices):
            raise ValueError(f"{arm.name}.gamepad axis/button indices must be non-negative")
        if len(set(indices[6:])) != len(indices[6:]):
            raise ValueError(f"{arm.name}.gamepad buttons must be distinct")
        if len(set(indices[:6])) != 6:
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
    inference_slots = control.get('inference_slots', 4)
    guard_slots = control.get('installation_guard_slots', 1)
    if any(type(v) is not int or v < 1 for v in (inference_slots, guard_slots)):
        raise ValueError('inference and installation budgets must be positive control-slot counts')
    if fps != 30:
        raise ValueError('execution contract requires one 30 Hz action/control loop')
    return RolloutConfig(
        inference_slots=inference_slots, installation_guard_slots=guard_slots,
        backend=backend,
        worker_socket_path=Path(str(transport.get("worker_socket_path", "/tmp/vla_franky_worker.sock"))).expanduser().resolve(),
        worker_python=(Path(str(transport["worker_python"])).expanduser().absolute()
                       if transport.get("worker_python") else None),
        franky_compliance=franky_compliance,
        workspace_limits_enabled=workspace_limits_enabled,
        action_space=str(raw.get("action_space", "tcp")),
        input_source=str(raw.get("input_source", "gamepad")),
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
        gamepad_profile=gamepad_profile,
        subtasks=tuple(subtasks),
        tasks=tasks,
        gamepad_gripper_trigger_deadzone=gamepad_gripper_trigger_deadzone,
        min_control_command_success_rate=min_success_rate,
        low_success_rate_samples=low_success_rate_samples,
        **{name: float(value) for name, value in positive.items() if name != "fps"},
    )
