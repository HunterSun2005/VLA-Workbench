"""Reuse demonstration gamepad controls at online observation boundaries."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from vla.datacollection.franka_robotiq_rollout.gamepad import decode_cartesian_gamepad_values
from vla.datacollection.start_teleop.gamepad_recording import load_recording_config
from vla.datacollection.start_teleop.gamepad_teleop import GamepadTeleop, upright_tcp_rotation
from vla.robots.franka_robotiq.robot import pose_matrix


PROFILE = Path(__file__).parent / "configs/single_left_gamepad_record.yaml"


class RolloutGamepadControls(GamepadTeleop):
    """Input-only adapter: never call the standalone enable/tick/motion methods."""

    def __init__(self, robot, receiver, *, report=print):
        config, recording = load_recording_config(PROFILE)
        super().__init__(config, robot, receiver, report=report)
        self.critical_button = recording.critical_button
        self.profile_metadata = json.dumps(asdict(config), default=str, sort_keys=True)
        self._dataset_home_position = None
        self._dataset_home_rotation = None
        self.reset_episode()

    def configure_dataset_home_position(self, position, rotation=None):
        """Set the Menu shortcut's XYZ target from the deployment home pose."""
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("dataset home position must be a finite 3-vector")
        self._dataset_home_position = position.copy()
        if rotation is not None:
            from vla.datacollection.start_teleop.gamepad_teleop import axis_aligned_upright_rotation
            self._dataset_home_rotation = axis_aligned_upright_rotation(rotation)
        else:
            self._dataset_home_rotation = None

    def capture_initial_pose(self, state):
        if self._initial_position is None:
            pose = pose_matrix(state.O_T_EE)
            self._initial_position = pose[:3, 3].copy()
            self._initial_upright_rotation = upright_tcp_rotation(pose[:3, :3])

    def reset_episode(self):
        self._critical_ready = False
        self.reset_human()

    def reset_human(self):
        self.mode = "standby"
        self._reset_speed()
        self._shortcut_ready = {self.config.pose_shortcuts.view_button: False,
                                self.config.pose_shortcuts.menu_button: False}
        self._shortcut_packet = None
        self._pose_return = None
        self._pose_return_path = []

    def request_dataset_home_return(self, position, rotation, clearance_z, position_tolerance_m):
        self.mode = "active"
        super().request_dataset_home_return(position, rotation, clearance_z, position_tolerance_m)

    def observe(self, packet, *, active, human):
        if not active:
            self.reset_episode()
            return False
        required = (self.critical_button, self.config.speed_modes.toggle_button,
                    *self._shortcut_ready)
        if max(required) >= len(packet.buttons):
            raise ValueError("gamepad message is missing RB/LB/View/Menu buttons")
        decode_cartesian_gamepad_values(packet.axes, packet.buttons, self.config.mapping, self.config.control)
        pressed = bool(packet.buttons[self.critical_button])
        critical_rising = pressed and self._critical_ready
        self._critical_ready = not pressed
        if not human:
            self.reset_human()
            return critical_rising
        self.mode = "active"
        self._update_speed(packet)
        manual = self._has_manual_motion(packet)
        shortcut_rising = False
        for button, ready in self._shortcut_ready.items():
            shortcut_rising |= bool(packet.buttons[button]) and ready
            self._shortcut_ready[button] = not bool(packet.buttons[button])
        if shortcut_rising:
            if manual:
                self.report("Pose shortcut ignored: release motion and gripper controls first")
            else:
                self._shortcut_packet = packet
        if manual:
            self._shortcut_packet = None
            if self._pose_return is not None:
                self.report(f"Cancelled {self._pose_return.label} return")
                self._pose_return = None
                self._pose_return_path = []
        return critical_rising

    def make_command(self, packet, state, dt):
        if self.mode != "active":
            raise RuntimeError("SFT gamepad controls require human intervention")
        self.capture_initial_pose(state)
        self._advance_speed(dt)
        params = self.control
        manual = self._has_manual_motion(packet)
        if self._shortcut_packet is not None:
            pending, self._shortcut_packet = self._shortcut_packet, None
            self._view_button_ready = self._menu_button_ready = True
            self._update_pose_shortcuts(
                pending, state, manual_motion=manual,
                menu_position=self._dataset_home_position,
                menu_rotation=self._dataset_home_rotation,
            )
        if self._pose_return is not None and manual:
            self._pose_return = None
            self._pose_return_path = []
        if self._pose_return is not None:
            fast_return = self._pose_return.label == "initial pose" or self._pose_return.label.startswith("home ")
            scale = self.config.speed_modes.high_multiplier if fast_return else 1.0
            params = self._scaled_control(scale)
            linear, angular = self._pose_return_command(state, params)
            gripper = 0.0
        else:
            command = decode_cartesian_gamepad_values(packet.axes, packet.buttons, self.config.mapping, params)
            linear, angular, gripper = command.linear_velocity_base, command.angular_velocity_tool, command.gripper_motion
        return linear, angular, gripper, params.motion_limits()
