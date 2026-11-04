from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.robots import RobotConfig
import yaml


@RobotConfig.register_subclass("franka_robotiq")
@dataclass(kw_only=True)
class FrankaRobotiqConfig(RobotConfig):
    robot_ip: str = "172.16.0.2"
    cameras_enabled: bool = True
    automatic_motion_recovery: bool = True
    # "default" leaves Franky's initialization unchanged; gamepad profiles
    # explicitly select GELLO's existing full collision thresholds.
    collision_behavior_profile: str = "default"
    human_angular_velocity_control: bool = False
    human_linear_velocity_control: bool = False
    human_gripper_immediate_hold: bool = False
    human_max_vertical_velocity: float | None = None
    camera_serials: dict[str, str] = field(
        default_factory=lambda: {
            "base": "260722303785",
            "left_wrist": "346222073620",
        }
    )
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    image_size: int = 224
    camera_timeout_s: float = 1.0
    camera_max_pair_skew_s: float = 0.025
    observation_max_image_age_s: float = 0.075

    gripper_port: str = "/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DABC0KC0-if00-port0"
    gripper_baudrate: int = 115200
    gripper_bytesize: int = 8
    gripper_parity: str = "N"
    gripper_stopbits: int = 1
    gripper_device_id: int = 9
    gripper_timeout_s: float = 0.2
    gripper_state_timeout_s: float = 0.5
    gripper_command_timeout_s: float | None = None
    gripper_deadband: float = 0.01
    gripper_max_delta_per_s: float = 2.0
    gripper_speed: int = 128
    gripper_force: int = 128
    gripper_action_threshold: float = 0.75
    gripper_close_trigger: float = 0.75
    gripper_open_trigger: float = 0.875
    gripper_post_close_lock_s: float = 2.0
    gripper_startup_open_fraction: float = 0.95
    gripper_startup_open_timeout_s: float = 5.0

    workspace_min: tuple[float, float, float] = (0.20, -0.45, 0.05)
    workspace_max: tuple[float, float, float] = (0.75, 0.45, 0.75)
    max_linear_velocity: float = 0.04
    max_angular_velocity: float = 0.30
    control_frequency: float = 30.0
    dynamics_factor: float = 0.05
    target_smoothing_alpha: float = 0.25
    velocity_smoothing: float = 0.8
    smooth_angular_velocity: bool = False
    velocity_tracking_tau_s: float = 0.15
    velocity_motion_duration_ms: int = 100
    min_command_position_delta: float = 0.0002
    min_command_rotation_delta: float = 0.005
    max_consecutive_motion_failures: int = 5
    motion_cleanup_timeout_s: float = 1.0
    motion_recovery_timeout_s: float = 1.0
    motion_recovery_poll_interval_s: float = 0.02
    realtime_cpu: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.collision_behavior_profile not in ("default", "gello"):
            raise ValueError("collision_behavior_profile must be 'default' or 'gello'")
        if type(self.human_angular_velocity_control) is not bool:
            raise ValueError("human_angular_velocity_control must be a boolean")
        if type(self.human_linear_velocity_control) is not bool:
            raise ValueError("human_linear_velocity_control must be a boolean")
        if type(self.human_gripper_immediate_hold) is not bool:
            raise ValueError("human_gripper_immediate_hold must be a boolean")
        if self.human_max_vertical_velocity is not None and (
            isinstance(self.human_max_vertical_velocity, bool)
            or not isinstance(self.human_max_vertical_velocity, (int, float))
            or not math.isfinite(self.human_max_vertical_velocity) or self.human_max_vertical_velocity <= 0
        ):
            raise ValueError("human_max_vertical_velocity must be finite and positive")
        if self.cameras_enabled:
            if set(self.camera_serials) not in ({"base", "left_wrist"}, {"base", "left_wrist", "side"}):
                raise ValueError("camera_serials require base and left_wrist, with optional side")
            if any(not isinstance(v, str) or not v for v in self.camera_serials.values()) or len(set(self.camera_serials.values())) != len(self.camera_serials):
                raise ValueError("camera serials must be nonempty and distinct")
        if self.control_frequency <= 0 or self.camera_fps <= 0:
            raise ValueError("control and camera frequencies must be positive")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.camera_max_pair_skew_s <= 0 or self.observation_max_image_age_s <= 0:
            raise ValueError("camera synchronization thresholds must be positive")
        if self.camera_max_pair_skew_s >= self.observation_max_image_age_s:
            raise ValueError("camera pair skew must be smaller than observation image age")
        if any(lo >= hi for lo, hi in zip(self.workspace_min, self.workspace_max)):
            raise ValueError("each workspace_min value must be smaller than workspace_max")
        if not 0 < self.dynamics_factor <= 1:
            raise ValueError("dynamics_factor must be in (0, 1]")
        if not 0 <= self.target_smoothing_alpha < 1:
            raise ValueError("target_smoothing_alpha must be in [0, 1)")
        if not 0 <= self.velocity_smoothing < 1:
            raise ValueError("velocity_smoothing must be in [0, 1)")
        if self.velocity_tracking_tau_s <= 0:
            raise ValueError("velocity_tracking_tau_s must be positive")
        if self.velocity_motion_duration_ms <= 0:
            raise ValueError("velocity_motion_duration_ms must be positive")
        if self.min_command_position_delta < 0 or self.min_command_rotation_delta < 0:
            raise ValueError("minimum command deltas must be non-negative")
        gate_enabled = self.min_command_position_delta > 0 or self.min_command_rotation_delta > 0
        minimum_gate_hold_ms = math.ceil(2000.0 / self.control_frequency)
        if gate_enabled and self.velocity_motion_duration_ms < minimum_gate_hold_ms:
            raise ValueError(
                "velocity_motion_duration_ms must cover at least two control periods when gating is enabled"
            )
        if self.max_consecutive_motion_failures <= 0:
            raise ValueError("max_consecutive_motion_failures must be positive")
        if self.motion_cleanup_timeout_s <= 0 or self.motion_recovery_timeout_s <= 0:
            raise ValueError("motion cleanup and recovery timeouts must be positive")
        if not 0 < self.motion_recovery_poll_interval_s <= self.motion_recovery_timeout_s:
            raise ValueError("motion recovery poll interval must be in (0, recovery timeout]")
        if self.realtime_cpu is not None and self.realtime_cpu < 0:
            raise ValueError("realtime_cpu must be a non-negative CPU index")
        if self.gripper_command_timeout_s is not None and (
            not math.isfinite(self.gripper_command_timeout_s) or self.gripper_command_timeout_s <= 0
        ):
            raise ValueError("gripper_command_timeout_s must be finite and positive")
        if not 0 <= self.gripper_action_threshold <= 1:
            raise ValueError("gripper_action_threshold must be in [0, 1]")
        if not 0 <= self.gripper_close_trigger < self.gripper_open_trigger <= 1:
            raise ValueError("gripper triggers must satisfy 0 <= close < open <= 1")
        if self.gripper_post_close_lock_s < 0:
            raise ValueError("gripper_post_close_lock_s must be non-negative")
        if not 0 < self.gripper_startup_open_fraction <= 1:
            raise ValueError("gripper_startup_open_fraction must be in (0, 1]")
        if self.gripper_startup_open_timeout_s <= 0:
            raise ValueError("gripper_startup_open_timeout_s must be positive")


def franka_robotiq_config_from_mapping(values: dict[str, Any]) -> FrankaRobotiqConfig:
    """Build the robot config shared by deployment and rollout entrypoints."""
    robot = dict(values.get("robot", {}))
    cameras = dict(robot.pop("cameras", {}))
    gripper = dict(robot.pop("gripper", {}))
    workspace = dict(robot.pop("workspace", {}))
    robot.update(
        camera_serials={key: value["serial"] for key, value in cameras.items()},
        gripper_port=gripper.pop("port"),
        workspace_min=tuple(workspace.pop("min")),
        workspace_max=tuple(workspace.pop("max")),
    )
    if workspace:
        raise ValueError(f"unknown robot.workspace fields: {sorted(workspace)}")
    robot.update({f"gripper_{key}": value for key, value in gripper.items()})
    return FrankaRobotiqConfig(**robot)


def load_franka_robotiq_config(path: str | Path) -> FrankaRobotiqConfig:
    config_path = Path(path).expanduser().resolve()
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"deployment config must be a YAML mapping: {config_path}")
    return franka_robotiq_config_from_mapping(values)
