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
    validate_gamepad_values(axes, buttons, mapping)
    linear, x_button, b_button, gripper = decode_gamepad_values(axes, buttons, mapping, config)
    try:
        horizontal = _deadzone(float(axes[mapping.right_stick_x_axis]), config.gamepad_deadzone)
        vertical = _deadzone(float(axes[mapping.right_stick_y_axis]), config.gamepad_deadzone)
    except IndexError as error:
        raise ValueError(f"gamepad message on {mapping.joy_topic} is missing right-stick axes") from error
    # Preserve the original single-arm mapping BEFORE tool-plane calibration.
    # The physical tilt directions are defined by this mapping plus the -45
    # degree calibration, not by relabeling the tool X/Y axes as pitch/roll.
    angular = np.array([
        mapping.right_stick_y_sign * vertical * config.gamepad_tilt_speed,
        mapping.right_stick_x_sign * horizontal * config.gamepad_tilt_speed,
        mapping.tool_roll_button_sign * (x_button - b_button) * config.gamepad_roll_speed,
    ], dtype=float)
    # Rotate only the tilt vector in tool XY. Positive angles are counterclockwise
    # in this plane (tool +X toward +Y); tool-Z roll and base-frame XYZ are unchanged.
    angular = gamepad_rotation_basis(mapping) @ angular
    norm = float(np.linalg.norm(angular))
    if norm > config.gamepad_max_angular_speed:
        angular *= config.gamepad_max_angular_speed / norm
    return CartesianGamepadCommand(linear, angular, gripper)


def gamepad_rotation_basis(mapping: GamepadConfig) -> np.ndarray:
    """Columns are the calibrated vertical, horizontal and twist control axes."""
    angle = math.radians(mapping.right_stick_rotation_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])


def release_rotation_reference(measured, reference, basis_tool, released):
    """Remove only released-channel error, in the current measured control basis.

    Project a single rotation vector, rather than subtracting Euler angles.
    The other channels keep their reference error; subsequent neutral samples
    do not re-anchor the pose or turn impedance hold into gravity compensation.
    """
    if not np.any(released):
        return reference
    from scipy.spatial.transform import Rotation
    basis_base = measured @ basis_tool
    error = Rotation.from_matrix(reference @ measured.T).as_rotvec()
    components = basis_base.T @ error
    components[released] = 0.
    return Rotation.from_rotvec(basis_base @ components).as_matrix() @ measured


def _deadzone(value: float, deadzone: float) -> float:
    if abs(value) <= deadzone:
        return 0.0
    magnitude = min(1.0, (abs(value) - deadzone) / (1.0 - deadzone))
    return math.copysign(magnitude, value)


def validate_gamepad_values(axes, buttons, mapping):
    required_axes = (mapping.left_stick_x_axis, mapping.left_stick_y_axis,
                     mapping.right_stick_x_axis, mapping.right_stick_y_axis,
                     mapping.left_trigger_axis, mapping.right_trigger_axis)
    required_buttons = (mapping.a_button, mapping.b_button, mapping.x_button, mapping.y_button,
                        mapping.speed_button, mapping.phase_button,
                        mapping.upright_button, mapping.reset_pose_button, mapping.yaw_reset_button)
    if (len(axes) <= max(required_axes) or len(buttons) <= max(required_buttons)
            or not np.isfinite(np.asarray(axes, dtype=float)).all()
            or np.any(np.abs(np.asarray(axes, dtype=float)) > 1.0001)
            or not np.isin(np.asarray(buttons), [0, 1]).all()):
        raise ValueError(f"invalid gamepad input on {mapping.joy_topic} ({mapping.input_layout}); "
                         "check controller recognition and axis/button layout")
    if mapping.input_layout == 'standard_sdl':
        if len(axes) != 6 or any(axes[i] > 1e-4 for i in required_axes[4:]):
            raise ValueError('standard_sdl requires six game_controller_node axes; triggers are 0 released, -1 pressed')


def gamepad_is_neutral(axes, buttons, mapping, config) -> bool:
    """Gate start/takeover on released inputs, including opposing triggers."""
    validate_gamepad_values(axes, buttons, mapping)
    required_axes = (mapping.left_stick_x_axis, mapping.left_stick_y_axis,
                     mapping.right_stick_x_axis, mapping.right_stick_y_axis,
                     mapping.left_trigger_axis, mapping.right_trigger_axis)
    return (not any(buttons)
            and all(abs(axes[index]) <= config.gamepad_deadzone for index in required_axes[:4])
            and all(_trigger(axes[index], mapping.input_layout) <= config.gamepad_gripper_trigger_deadzone
                    for index in required_axes[4:]))


def gamepad_has_motion(axes, buttons, mapping, config) -> bool:
    """Individual held motion controls cancel shortcuts, even if they cancel out."""
    return (any(abs(axes[i]) > config.gamepad_deadzone for i in (
                mapping.left_stick_x_axis, mapping.left_stick_y_axis,
                mapping.right_stick_x_axis, mapping.right_stick_y_axis))
            or any(buttons[i] for i in (mapping.a_button, mapping.b_button, mapping.x_button, mapping.y_button))
            or any(_trigger(axes[i], mapping.input_layout) > config.gamepad_gripper_trigger_deadzone
                   for i in (mapping.left_trigger_axis, mapping.right_trigger_axis)))


@dataclass
class GamepadPoseReturn:
    """A shortcut goal advanced only by the existing 30 Hz Gamepad callback."""
    label: str
    position: np.ndarray
    rotation: np.ndarray
    deadline: float

    def step(self, position, rotation, measured_position, measured_rotation, *,
             now, dt, max_linear_speed, max_angular_speed):
        from scipy.spatial.transform import Rotation
        if now > self.deadline:
            raise RuntimeError(f'Gamepad {self.label} return timed out')
        delta = self.position - position
        distance = float(np.linalg.norm(delta))
        # Match the old return's approach profile, applied to the absolute
        # reference. This is not an extra control clock or pose-to-velocity SDK.
        approach = min(.005, max_linear_speed)
        if distance <= .01:
            speed = approach * distance / .01
        else:
            speed = approach + min(1., (distance-.01)/.07) * (max_linear_speed-approach)
        linear_step = min(distance, speed*dt)
        next_position = position + delta * (linear_step/distance) if distance > 0 else position.copy()
        vector = Rotation.from_matrix(self.rotation @ rotation.T).as_rotvec()
        angle = float(np.linalg.norm(vector))
        angular_step = min(angle, min(max_angular_speed, 3.*angle)*dt)
        next_rotation = (Rotation.from_rotvec(vector*(angular_step/angle)).as_matrix() @ rotation
                         if angle > 1e-12 else rotation.copy())
        position_error = float(np.linalg.norm(self.position-measured_position))
        rotation_error = Rotation.from_matrix(self.rotation @ measured_rotation.T).magnitude()
        reached = (position_error <= .0005 and rotation_error <= math.radians(2.)
                   and distance <= .0005 and angle <= math.radians(2.))
        # Keep the final bounded reference on completion, avoiding a tolerance-
        # sized final jump. It remains fixed until explicit new manual input.
        return next_position, next_rotation, reached


def _trigger(raw_value: float, input_layout: str = 'raw_joy') -> float:
    # ROS game_controller_node negates SDL's [0, 32767] trigger range.
    if input_layout == 'standard_sdl':
        return float(np.clip(-raw_value, 0.0, 1.0))
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
        left_trigger = _trigger(float(axes[mapping.left_trigger_axis]), mapping.input_layout)
        right_trigger = _trigger(float(axes[mapping.right_trigger_axis]), mapping.input_layout)
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
