"""Configuration for camera-free, single-left Franky teleoperation."""

from __future__ import annotations

from dataclasses import dataclass, fields
import ipaddress
import math
from pathlib import Path

import yaml

from vla.datacollection.franka_robotiq_rollout.config import GamepadConfig
from vla.robots.franka_robotiq.config import (
    FrankaRobotiqConfig, franka_robotiq_config_from_mapping,
)


@dataclass(frozen=True)
class TeleopParameters:
    control_rate: float = 100.0
    joy_timeout_s: float = 0.20
    state_timeout_s: float = 0.50
    discontinuity_timeout_s: float = 0.10
    gamepad_deadzone: float = 0.08
    gamepad_xy_speed: float = 0.02
    gamepad_z_speed: float = 0.01
    gamepad_tilt_speed: float = 0.15
    gamepad_roll_speed: float = 0.30
    gamepad_max_angular_speed: float = 0.30
    gamepad_gripper_trigger_deadzone: float = 0.05
    gripper_speed_scale: float = 0.40
    position_gain: float = 8.0
    orientation_gain: float = 3.0
    max_linear_speed: float = 0.04
    damping: float = 0.05
    max_joint_speed: float = 0.30
    joint_limit_margin: float = 0.05
    minimum_success_rate: float = 0.90
    low_rate_samples: int = 3

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"control.{field.name} must be finite and positive")
        for name in ("gamepad_deadzone", "gamepad_gripper_trigger_deadzone"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"control.{name} must be in (0, 1)")
        if self.gripper_speed_scale > 1 or self.minimum_success_rate > 1:
            raise ValueError("gripper_speed_scale and minimum_success_rate must be <= 1")
        if type(self.low_rate_samples) is not int:
            raise ValueError("low_rate_samples must be an integer")
        if not 1 / self.control_rate < self.discontinuity_timeout_s <= self.joy_timeout_s:
            raise ValueError("require control period < discontinuity timeout <= Joy timeout")

    def motion_limits(self):
        return dict(
            damping=self.damping, max_joint_speed=self.max_joint_speed,
            joint_limit_margin=self.joint_limit_margin, position_gain=self.position_gain,
            orientation_gain=self.orientation_gain, max_linear_speed=self.max_linear_speed,
            max_angular_speed=self.gamepad_max_angular_speed,
            gripper_speed_scale=self.gripper_speed_scale,
        )


@dataclass(frozen=True)
class SpeedModes:
    toggle_button: int = 4
    high_multiplier: float = 5.0
    transition_time_s: float = 0.3

    def __post_init__(self):
        if type(self.toggle_button) is not int or self.toggle_button < 0:
            raise ValueError("speed_modes.toggle_button must be a non-negative integer")
        if isinstance(self.high_multiplier, bool) or not isinstance(self.high_multiplier, (int, float)) or not math.isfinite(self.high_multiplier) or self.high_multiplier <= 1:
            raise ValueError("speed_modes.high_multiplier must be finite and > 1")
        if isinstance(self.transition_time_s, bool) or not isinstance(self.transition_time_s, (int, float)) or not math.isfinite(self.transition_time_s) or self.transition_time_s <= 0:
            raise ValueError("speed_modes.transition_time_s must be finite and positive")


@dataclass(frozen=True)
class PoseShortcuts:
    view_button: int = 6
    menu_button: int = 7
    yaw_reset_button: int | None = None
    position_tolerance_m: float = 0.0005
    slowdown_distance_m: float = 0.08
    final_approach_distance_m: float = 0.01
    final_approach_speed_m_s: float = 0.005
    orientation_tolerance_deg: float = 2.0
    timeout_s: float = 30.0

    def __post_init__(self):
        if self.yaw_reset_button is not None and (type(self.yaw_reset_button) is not int or self.yaw_reset_button < 0):
            raise ValueError("pose_shortcuts.yaw_reset_button must be a non-negative integer or null")
        for name in ("view_button", "menu_button"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"pose_shortcuts.{name} must be a non-negative integer")
        if self.view_button == self.menu_button:
            raise ValueError("pose shortcut buttons must be distinct")
        for name in ("position_tolerance_m", "orientation_tolerance_deg", "timeout_s",
                     "slowdown_distance_m", "final_approach_distance_m", "final_approach_speed_m_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"pose_shortcuts.{name} must be finite and positive")
        if self.orientation_tolerance_deg >= 180:
            raise ValueError("pose_shortcuts.orientation_tolerance_deg must be < 180")
        if not self.position_tolerance_m < self.final_approach_distance_m < self.slowdown_distance_m:
            raise ValueError("require position tolerance < final approach distance < slowdown distance")


@dataclass(frozen=True)
class TeleopConfig:
    robot: FrankaRobotiqConfig
    mapping: GamepadConfig
    control: TeleopParameters
    joy_device_id: int
    speed_modes: SpeedModes = SpeedModes()
    pose_shortcuts: PoseShortcuts = PoseShortcuts()


def load_config(path: Path, *, recording: bool = False) -> TeleopConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    allowed = {"robot", "mapping", "control", "joy_device_id", "speed_modes", "pose_shortcuts"}
    if recording:
        allowed.add("recording")
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError("expected robot, mapping, control, joy_device_id and speed_modes configuration")
    robot = franka_robotiq_config_from_mapping(raw)
    if robot.cameras_enabled != recording or robot.automatic_motion_recovery:
        raise ValueError(f"standalone teleop requires cameras_enabled={recording} and automatic_motion_recovery=false")
    ipaddress.ip_address(robot.robot_ip)
    for values in (robot.workspace_min, robot.workspace_max):
        if len(values) != 3 or not all(math.isfinite(v) for v in values):
            raise ValueError("workspace bounds must be finite 3-vectors")
    control = TeleopParameters(**raw.get("control", {}))
    if robot.velocity_motion_duration_ms / 1000 > control.discontinuity_timeout_s:
        raise ValueError("motion duration must not exceed the discontinuity timeout")
    if robot.gripper_command_timeout_s is None or robot.gripper_command_timeout_s > control.discontinuity_timeout_s:
        raise ValueError("standalone gripper requires a command timeout <= the discontinuity timeout")
    if control.max_linear_speed > robot.max_linear_velocity or control.gamepad_max_angular_speed > robot.max_angular_velocity:
        raise ValueError("human command speed limits must stay within robot velocity limits")
    defaults = dict(
        joy_topic="/left/gamepad_teleop/joy", left_stick_x_axis=0, left_stick_y_axis=1,
        left_trigger_axis=2, right_trigger_axis=5,
        a_button=0, b_button=1, x_button=2, y_button=3,
        joint7_button_sign=-1.0, left_stick_x_sign=1.0, left_stick_y_sign=1.0,
    )
    defaults.update(raw.get("mapping", {}))
    mapping = GamepadConfig(**defaults)
    for suffix in ("_axis", "_button"):
        indices = [getattr(mapping, f.name) for f in fields(mapping) if f.name.endswith(suffix)]
        if any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
            raise ValueError("gamepad axis/button indices must be non-negative, distinct integers")
    for field in fields(mapping):
        if field.name.endswith("_sign") and getattr(mapping, field.name) not in (-1, 1):
            raise ValueError("gamepad signs must be -1 or 1")
    if not mapping.joy_topic.startswith("/left/gamepad_teleop/"):
        raise ValueError("standalone Joy topic must be under /left/gamepad_teleop/")
    device = raw.get("joy_device_id", 0)
    if type(device) is not int or device < 0:
        raise ValueError("joy_device_id must be a non-negative integer")
    speed_modes = SpeedModes(**raw.get("speed_modes", {}))
    if speed_modes.transition_time_s < 2 / control.control_rate:
        raise ValueError("speed transition must cover at least two control periods")
    pose_shortcuts = PoseShortcuts(**raw.get("pose_shortcuts", {}))
    action_buttons = (mapping.a_button, mapping.b_button, mapping.x_button, mapping.y_button)
    occupied = (*action_buttons, speed_modes.toggle_button,
                pose_shortcuts.view_button, pose_shortcuts.menu_button)
    if pose_shortcuts.yaw_reset_button is not None:
        occupied += (pose_shortcuts.yaw_reset_button,)
    if len(set(occupied)) != len(occupied):
        raise ValueError("motion, speed and pose shortcut buttons must be distinct")
    return TeleopConfig(robot, mapping, control, device, speed_modes, pose_shortcuts)
