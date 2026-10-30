"""ROS-independent Xbox input decoding for rollout intervention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .config import GamepadConfig


class CartesianGamepadParameters(Protocol):
    gamepad_deadzone: float
    gamepad_xy_speed: float
    gamepad_z_speed: float
    gamepad_tilt_speed: float
    gamepad_roll_speed: float
    gamepad_max_angular_speed: float
    gamepad_gripper_trigger_deadzone: float


@dataclass(frozen=True)
class CartesianGamepadCommand:
    linear_velocity_base: np.ndarray
    angular_velocity_tool: np.ndarray
    gripper_motion: float


def decode_cartesian_gamepad_values(
    axes: Sequence[float],
    buttons: Sequence[int],
    mapping: GamepadConfig,
    config: CartesianGamepadParameters,
) -> CartesianGamepadCommand:
    if not np.isfinite(np.asarray(axes, dtype=float)).all():
        raise ValueError("gamepad axes must be finite")
    if not np.isin(np.asarray(buttons), [0, 1]).all():
        raise ValueError("gamepad buttons must be binary")
    linear, x_button, b_button, gripper = decode_gamepad_values(axes, buttons, mapping, config)
    try:
        horizontal = _deadzone(float(axes[mapping.right_stick_x_axis]), config.gamepad_deadzone)
        vertical = _deadzone(float(axes[mapping.right_stick_y_axis]), config.gamepad_deadzone)
    except IndexError as error:
        raise ValueError(f"gamepad message on {mapping.joy_topic} is missing right-stick axes") from error
    angular = np.array([
        mapping.right_stick_y_sign * vertical * config.gamepad_tilt_speed,
        mapping.right_stick_x_sign * horizontal * config.gamepad_tilt_speed,
        mapping.tool_roll_button_sign * (x_button - b_button) * config.gamepad_roll_speed,
    ], dtype=float)
    # Rotate only the tilt vector in tool XY. Positive angles are counterclockwise
    # in this plane (tool +X toward +Y); tool-Z roll and base-frame XYZ are unchanged.
    angle = math.radians(mapping.right_stick_rotation_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    angular[:2] = np.array([[cosine, -sine], [sine, cosine]]) @ angular[:2]
    norm = float(np.linalg.norm(angular))
    if norm > config.gamepad_max_angular_speed:
        angular *= config.gamepad_max_angular_speed / norm
    return CartesianGamepadCommand(linear, angular, gripper)


def _deadzone(value: float, deadzone: float) -> float:
    if abs(value) <= deadzone:
        return 0.0
    magnitude = min(1.0, (abs(value) - deadzone) / (1.0 - deadzone))
    return math.copysign(magnitude, value)


def gamepad_is_neutral(axes, buttons, mapping, config) -> bool:
    """Gate start/takeover on released inputs, including opposing triggers."""
    required_axes = (mapping.left_stick_x_axis, mapping.left_stick_y_axis,
                     mapping.right_stick_x_axis, mapping.right_stick_y_axis,
                     mapping.left_trigger_axis, mapping.right_trigger_axis)
    required_buttons = (mapping.a_button, mapping.b_button, mapping.x_button, mapping.y_button)
    if (len(axes) <= max(required_axes) or len(buttons) <= max(required_buttons)
            or not np.isfinite(np.asarray(axes, dtype=float)).all()
            or not np.isin(np.asarray(buttons), [0, 1]).all()):
        raise ValueError(f"invalid gamepad input on {mapping.joy_topic}")
    return (not any(buttons)
            and all(abs(axes[index]) <= config.gamepad_deadzone for index in required_axes[:4])
            and all(_trigger(axes[index]) <= config.gamepad_gripper_trigger_deadzone
                    for index in required_axes[4:]))


def _trigger(raw_value: float) -> float:
    return float(np.clip((1.0 - raw_value) * 0.5, 0.0, 1.0))


def _signed_deadzone(value: float, deadzone: float) -> float:
    if abs(value) <= deadzone:
        return 0.0
    return math.copysign(
        (abs(value) - deadzone) / (1.0 - deadzone),
        value,
    )


def decode_gamepad_values(
    axes: Sequence[float],
    buttons: Sequence[int],
    mapping: GamepadConfig,
    config: CartesianGamepadParameters,
) -> tuple[np.ndarray, float, float, float]:
    try:
        stick_x = _deadzone(
            float(axes[mapping.left_stick_x_axis]), config.gamepad_deadzone
        )
        stick_y = _deadzone(
            float(axes[mapping.left_stick_y_axis]), config.gamepad_deadzone
        )
        left_trigger = _trigger(float(axes[mapping.left_trigger_axis]))
        right_trigger = _trigger(float(axes[mapping.right_trigger_axis]))
        a_button = float(buttons[mapping.a_button])
        b_button = float(buttons[mapping.b_button])
        x_button = float(buttons[mapping.x_button])
        y_button = float(buttons[mapping.y_button])
    except IndexError as error:
        raise ValueError(
            f"gamepad message on {mapping.joy_topic} does not match configured mapping"
        ) from error

    velocity = np.asarray(
        [
            mapping.left_stick_y_sign * stick_y * config.gamepad_xy_speed,
            mapping.left_stick_x_sign * stick_x * config.gamepad_xy_speed,
            (y_button - a_button) * config.gamepad_z_speed,
        ],
        dtype=float,
    )
    gripper_motion = _signed_deadzone(
        left_trigger - right_trigger,
        config.gamepad_gripper_trigger_deadzone,
    )
    return velocity, x_button, b_button, gripper_motion


def gripper_direction_transition(
    previous_direction: int,
    motion: float,
    measured_open_fraction: float,
) -> tuple[int, float | None]:
    """Return the new direction and a goal only when that direction changes."""
    if previous_direction not in {-1, 0, 1}:
        raise ValueError("previous gripper direction must be -1, 0, or 1")
    if not np.isfinite(motion) or not np.isfinite(measured_open_fraction):
        raise ValueError("gripper motion and measured position must be finite")
    direction = 0 if motion == 0.0 else (1 if motion > 0.0 else -1)
    if direction == previous_direction:
        return direction, None
    if direction == 0:
        goal = float(np.clip(measured_open_fraction, 0.0, 1.0))
    else:
        goal = 1.0 if direction > 0 else 0.0
    return direction, goal
