"""Standalone Franky gamepad control; no cameras, policy client or dataset writer."""

from __future__ import annotations

from vla.terminal import (console_print as print, console_status, console_debug, terminal_session,
                          TerminalConfirmation, diagnostic_path, current_reporter)

import argparse
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
import importlib.util
import os
from pathlib import Path
import queue
import select
import signal
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty

import numpy as np

from vla.datacollection.franka_robotiq_rollout.gamepad import decode_cartesian_gamepad_values
from vla.datacollection.franka_robotiq_rollout.joy_transport import JoyReceiver
from vla.datacollection.start_teleop.gamepad_config import TeleopConfig, load_config
from vla.robots.franka_robotiq.resource_lock import HardwareLease
from vla.robots.franka_robotiq.robot import (
    FrankaRobotiqRobot, _rotation_angle, _rotation_axis, pose_matrix,
)


from vla.paths import workspace
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "single_left_gamepad.yaml"


@dataclass
class PoseReturn:
    label: str
    position: np.ndarray
    rotation: np.ndarray
    deadline: float


def upright_tcp_rotation(rotation: np.ndarray) -> np.ndarray:
    """Point tool Z down while retaining the current horizontal tool heading."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("TCP rotation must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1):
        raise ValueError("TCP rotation must be in SO(3)")
    z_axis = np.array([0.0, 0.0, -1.0])
    x_axis = rotation[:, 0].copy()
    x_axis[2] = 0.0
    norm = float(np.linalg.norm(x_axis))
    if norm < 1e-6:
        # When tool X is vertical, retain the heading implied by tool Y.
        y_axis = rotation[:, 1].copy()
        y_axis[2] = 0.0
        y_norm = float(np.linalg.norm(y_axis))
        x_axis = np.cross(y_axis / y_norm, z_axis) if y_norm >= 1e-6 else np.array([1.0, 0.0, 0.0])
    else:
        x_axis /= norm
    y_axis = np.cross(z_axis, x_axis)
    result = np.column_stack((x_axis, y_axis, z_axis))
    if not np.allclose(result.T @ result, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(result), 1):
        raise RuntimeError("failed to construct an upright TCP rotation")
    return result


def axis_aligned_upright_rotation(rotation: np.ndarray) -> np.ndarray:
    """Choose the base-axis-aligned downward pose nearest the reference heading."""
    upright = upright_tcp_rotation(rotation)
    headings = np.array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.]])
    x_axis = headings[np.argmax(headings @ upright[:, 0])]
    z_axis = np.array([0., 0., -1.])
    return np.column_stack((x_axis, np.cross(z_axis, x_axis), z_axis))


def reset_tcp_heading(rotation: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rotate only about base Z to the reference heading, retaining tilt."""
    current_heading = upright_tcp_rotation(rotation)
    reference_heading = upright_tcp_rotation(reference)
    # The downward-Z upright frames cancel to a pure base-Z rotation.
    return reference_heading @ current_heading.T @ rotation


class AsyncReporter:
    """Best-effort console output must never wait on the control thread."""

    def __init__(self, output=print, *, capacity=16):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("reporter capacity must be a positive integer")
        self.output = output
        self._queue = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="gamepad-console", daemon=True)
        self._thread.start()

    def __call__(self, message):
        if self._stop.is_set():
            return
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            pass

    def _worker(self):
        while not self._stop.is_set() or not self._queue.empty():
            try:
                message = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self.output(message)
            except Exception:
                # A closed terminal must not terminate or stall robot control.
                pass

    def close(self):
        self._stop.set()
        self._thread.join(timeout=0.2)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class GamepadTeleop:
    def __init__(self, config: TeleopConfig, robot, receiver, *, clock=time.monotonic, report=print, on_command=None):
        self.config, self.robot, self.receiver = config, robot, receiver
        self.clock, self.report = clock, report
        self.on_command = on_command
        self.mode = "standby"
        self.reason = ""
        self._last_tick = clock()
        self._paused_at = self._last_tick
        self._state_time = None
        self._state_changed_at = self._last_tick
        self._state_advanced = False
        self._low_rates = 0
        self._rate_history = deque(maxlen=16)
        self.fault_snapshot = None
        self._success_rate_monitor_started_at = None
        self._initial_position = None
        self._initial_upright_rotation = None
        self._pose_return = None
        self._pose_return_path = []
        self._home_position_tolerance_m = None
        self._view_button_ready = False
        self._menu_button_ready = False
        self._yaw_button_ready = False
        self._reset_speed()

    def _reset_speed(self):
        self.high_speed = False
        self.speed_scale = 1.0
        self._speed_button_ready = False
        self._control = self.config.control

    def _scaled_control(self, scale):
        params, robot = self.config.control, self.config.robot
        if scale == 1.0:
            return params
        return replace(
            params,
            gamepad_xy_speed=params.gamepad_xy_speed * scale,
            gamepad_z_speed=params.gamepad_z_speed * scale,
            max_linear_speed=min(params.max_linear_speed * scale, robot.max_linear_velocity),
        )

    @property
    def control(self):
        return self._control

    def _neutral(self, packet):
        mapping, params = self.config.mapping, self.config.control
        axes = (mapping.left_stick_x_axis, mapping.left_stick_y_axis,
                mapping.right_stick_x_axis, mapping.right_stick_y_axis)
        triggers = (mapping.left_trigger_axis, mapping.right_trigger_axis)
        # Test individual controls: opposing held buttons/triggers are not neutral.
        return (
            all(abs(packet.axes[i]) <= params.gamepad_deadzone for i in axes)
            and all((1 - packet.axes[i]) / 2 <= params.gamepad_gripper_trigger_deadzone for i in triggers)
            and not any(packet.buttons)
        )

    def _update_speed(self, packet):
        pressed = bool(packet.buttons[self.config.speed_modes.toggle_button])
        rising = pressed and self._speed_button_ready
        self._speed_button_ready = not pressed
        if self.mode != "active" or not rising:
            return
        self.high_speed = not self.high_speed
        self.report(f"XYZ speed target={'HIGH' if self.high_speed else 'LOW'}; effective={self.speed_scale:.3f}x (rotation/gripper unchanged)")

    def _advance_speed(self, dt):
        modes = self.config.speed_modes
        target = modes.high_multiplier if self.high_speed else 1.0
        delta = target - self.speed_scale
        if delta == 0:
            return
        step = (modes.high_multiplier - 1.0) * dt / modes.transition_time_s
        if abs(delta) <= step + 1e-12:
            self.speed_scale = target
        else:
            self.speed_scale += float(np.clip(delta, -step, step))
        self._control = self._scaled_control(self.speed_scale)
        if self.speed_scale != target:
            return
        params = self.control
        self.report(
            f"Speed={'HIGH' if self.high_speed else 'LOW'} reached (XYZ only): "
            f"XY={params.gamepad_xy_speed:.3f} Z={params.gamepad_z_speed:.3f} m/s; "
            f"tilt={params.gamepad_tilt_speed:.2f} roll={params.gamepad_roll_speed:.2f} rad/s; "
            f"limits={params.max_linear_speed:.3f} m/s, {params.gamepad_max_angular_speed:.2f} rad/s"
        )

    def _read_state(self):
        state = self.robot.get_control_state()
        now = self.clock()
        stamp = FrankaRobotiqRobot._state_time_s(state)
        if stamp is None or not np.isfinite(stamp):
            raise ValueError("Franka state has no valid time")
        if self._state_time is not None and stamp < self._state_time:
            raise ValueError("Franka state time moved backwards")
        fresh_state = self._state_time is None or stamp > self._state_time
        if fresh_state:
            self._state_advanced = self._state_time is not None
            self._state_time, self._state_changed_at = stamp, now
        if now - self._state_changed_at > self.config.control.state_timeout_s:
            raise TimeoutError("Franka state is stale")
        if int(state.robot_mode) not in (1, 2):
            raise RuntimeError(f"Franka mode is not Idle/Move: {state.robot_mode}")
        rate = float(state.control_command_success_rate)
        if not np.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("invalid Franka command success rate")
        threshold = self.config.control.minimum_success_rate
        startup_grace_s = max(
            self.config.robot.velocity_motion_duration_ms / 1000.0,
            2.0 / self.config.control.control_rate,
        )
        monitoring = (
            self._success_rate_monitor_started_at is not None
            and now - self._success_rate_monitor_started_at >= startup_grace_s
        )
        if self.mode == "active" and fresh_state and monitoring:
            # Robot telemetry can carry a float32-rounded ratio in a Python
            # float. Admit that representation of the threshold, but not the
            # next lower float32 value or genuinely lower rates such as .8999.
            boundary = min(float(np.float32(threshold)),
                           np.nextafter(float(threshold), -np.inf))
            low = rate < boundary
            self._low_rates = self._low_rates + 1 if low else 0
        elif self.mode == "active" and fresh_state:
            self._low_rates = 0
        if fresh_state:
            self._rate_history.append({
                "host_monotonic_s": now, "robot_time_s": stamp,
                "success_rate": rate, "consecutive_low_samples": self._low_rates,
                "success_rate_monitoring": monitoring,
                "robot_mode": int(state.robot_mode),
                "current_errors": str(getattr(state, "current_errors", "unavailable")),
                "last_motion_errors": str(getattr(state, "last_motion_errors", "unavailable")),
            })
        if self.mode == "active" and self._low_rates >= self.config.control.low_rate_samples:
            raise RuntimeError(
                f"Franka command success rate is low: rate={rate:.17g}; "
                f"threshold={threshold:.17g}; fresh_low_samples={self._low_rates}/"
                f"{self.config.control.low_rate_samples}; robot_time_s={stamp:.9f}"
            )
        pose = pose_matrix(state.O_T_EE)
        joints = np.asarray(state.q, dtype=float)
        if not np.isfinite(pose).all() or joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError("invalid Franka pose/joints")
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1):
            raise ValueError("invalid Franka rotation")
        if np.any(pose[:3, 3] < self.config.robot.workspace_min) or np.any(pose[:3, 3] > self.config.robot.workspace_max):
            raise ValueError(
                "TCP is outside the configured workspace; refusing to pull it into bounds; "
                f"TCP XYZ={pose[:3, 3].round(4).tolist()} m, "
                f"min={list(self.config.robot.workspace_min)}, max={list(self.config.robot.workspace_max)}"
            )
        return state

    def _packet(self):
        # The receiver samples monotonic time AFTER draining the socket. A time
        # sampled here can precede a newly arrived packet and falsely report a
        # negative age. Keep its strict stale/future timestamp checks intact.
        packet = self.receiver.require_fresh(self.config.control.joy_timeout_s)
        if not np.isfinite(packet.axes).all() or any(abs(axis) > 1.0001 for axis in packet.axes):
            raise ValueError("gamepad axes must be finite and within [-1, 1]")
        if not np.isin(packet.buttons, [0, 1]).all():
            raise ValueError("gamepad buttons must be binary")
        required_buttons = (
            self.config.speed_modes.toggle_button,
            self.config.pose_shortcuts.view_button,
            self.config.pose_shortcuts.menu_button,
        )
        if self.config.pose_shortcuts.yaw_reset_button is not None:
            required_buttons += (self.config.pose_shortcuts.yaw_reset_button,)
        if max(required_buttons) >= len(packet.buttons):
            raise ValueError("gamepad message is missing a speed or pose shortcut button")
        mapping = self.config.mapping
        if max(mapping.left_stick_x_axis, mapping.left_stick_y_axis,
               mapping.right_stick_x_axis, mapping.right_stick_y_axis,
               mapping.left_trigger_axis, mapping.right_trigger_axis) >= len(packet.axes):
            raise ValueError("gamepad message is missing mapped axes")
        if max(mapping.a_button, mapping.b_button, mapping.x_button, mapping.y_button) >= len(packet.buttons):
            raise ValueError("gamepad message is missing mapped action buttons")
        return packet

    def enable(self):
        if self.mode == "fault":
            self.report(f"Control fault latched: {self.reason}. 排查现场后执行故障恢复。")
            return
        if self.mode == "active":
            return
        try:
            state = self._read_state()
            packet = self._packet()
            neutral = self._neutral(packet)
            if not self._state_advanced or packet.monotonic_ns / 1e9 <= self._paused_at or not neutral:
                self.report("Not enabled: wait for fresh state and release both sticks, triggers and buttons.")
                return
        except (OSError, RuntimeError, ValueError) as error:
            self.report(f"Not enabled: {error}")
            return
        try:
            self.robot.begin_human_control()
            # Capture after the stop/reseed operation, which can take time.
            state = self._read_state()
            pose = pose_matrix(state.O_T_EE)
            initial_rotation = upright_tcp_rotation(pose[:3, :3])
        except Exception as error:
            self.pause(f"control initialization failed: {error}", fault=True)
            return
        self.mode, self.reason = "active", ""
        self._last_tick = self.clock()
        self._low_rates = 0
        self._rate_history.clear()
        self.fault_snapshot = None
        self._success_rate_monitor_started_at = None
        self._reset_speed()
        self._speed_button_ready = True
        self._initial_position = pose[:3, 3].copy()
        self._initial_upright_rotation = initial_rotation
        self._pose_return = None
        self._pose_return_path = []
        self._view_button_ready = self._menu_button_ready = True
        self._yaw_button_ready = True
        shortcuts = self.config.pose_shortcuts
        self.report(
            f"Gamepad control active; speed=LOW. Joy button {self.config.speed_modes.toggle_button}=continuous XYZ LOW/HIGH "
            f"({self.config.speed_modes.transition_time_s:.2f}s full-range ramp; rotation/gripper unchanged), "
            f"View {shortcuts.view_button}=initial XYZ+upright, Menu {shortcuts.menu_button}=current XYZ+upright+initial heading, "
            f"yaw reset={shortcuts.yaw_reset_button} (heading captured at s), "
            "d=pause, c/Esc/Ctrl+C=exit"
        )

    def _has_manual_motion(self, packet):
        mapping, params = self.config.mapping, self.config.control
        # Test controls before summation: opposing inputs also request takeover.
        return (
            any(abs(packet.axes[i]) > params.gamepad_deadzone for i in (
                mapping.left_stick_x_axis, mapping.left_stick_y_axis,
                mapping.right_stick_x_axis, mapping.right_stick_y_axis,
            ))
            or any(packet.buttons[i] for i in (
                mapping.a_button, mapping.b_button, mapping.x_button, mapping.y_button,
            ))
            or any((1 - packet.axes[i]) / 2 > params.gamepad_gripper_trigger_deadzone
                   for i in (mapping.left_trigger_axis, mapping.right_trigger_axis))
        )

    def _update_pose_shortcuts(self, packet, state, *, manual_motion, menu_position=None, menu_rotation=None):
        shortcuts = self.config.pose_shortcuts
        view_pressed = bool(packet.buttons[shortcuts.view_button])
        menu_pressed = bool(packet.buttons[shortcuts.menu_button])
        yaw_pressed = shortcuts.yaw_reset_button is not None and bool(packet.buttons[shortcuts.yaw_reset_button])
        yaw_rising = yaw_pressed and self._yaw_button_ready
        self._yaw_button_ready = not yaw_pressed
        view_rising = view_pressed and self._view_button_ready
        menu_rising = menu_pressed and self._menu_button_ready
        self._view_button_ready = not view_pressed
        self._menu_button_ready = not menu_pressed
        if not (view_rising or menu_rising or yaw_rising):
            return
        if manual_motion:
            self.report("Pose shortcut ignored: release motion and gripper controls first")
            return
        if not (self.config.robot.human_linear_velocity_control
                and self.config.robot.human_angular_velocity_control):
            self.report("Pose shortcut unavailable: direct linear and angular velocity control required")
            return
        pose = pose_matrix(state.O_T_EE)
        if yaw_rising:
            if self._initial_upright_rotation is None:
                raise RuntimeError("initial teleop heading was not captured")
            label = "initial heading (base Z only)"
            position = pose[:3, 3].copy()
            rotation = reset_tcp_heading(pose[:3, :3], self._initial_upright_rotation)
        elif view_rising:
            if self._initial_position is None or self._initial_upright_rotation is None:
                raise RuntimeError("initial teleop pose was not captured")
            label = "initial pose"
            position = self._initial_position.copy()
            rotation = self._initial_upright_rotation.copy()
            if menu_rising:
                self.report("View and Menu pressed together: returning to initial pose")
        else:
            label = "upright"
            # Online rollout Menu keeps the original direct upright return
            # path, but may use the deployment's configured home XYZ.
            position = (np.asarray(menu_position, dtype=np.float64).copy()
                        if menu_position is not None else pose[:3, 3].copy())
            # Restore the captured heading too, undoing X/B tool-Z rotation.
            # An explicit deployment home orientation still takes precedence.
            reference_rotation = (menu_rotation if menu_rotation is not None
                                  else self._initial_upright_rotation)
            if reference_rotation is None:
                raise RuntimeError("initial teleop heading was not captured")
            rotation = reference_rotation.copy()
        self._pose_return = PoseReturn(
            label, position, rotation, self.clock() + shortcuts.timeout_s,
        )
        self._pose_return_path = []
        self.report(f"Returning to {label}; move a manual control to cancel")

    def request_initial_pose_return(self):
        """Request the captured dataset-start pose from an external control key."""
        if self._initial_position is None or self._initial_upright_rotation is None:
            raise RuntimeError("initial teleop pose was not captured")
        if not (self.config.robot.human_linear_velocity_control
                and self.config.robot.human_angular_velocity_control):
            raise RuntimeError("initial pose return requires direct velocity control")
        shortcuts = self.config.pose_shortcuts
        self._pose_return = PoseReturn(
            "initial pose", self._initial_position.copy(),
            self._initial_upright_rotation.copy(),
            self.clock() + shortcuts.timeout_s,
        )
        self._pose_return_path = []
        self.report("Returning to initial pose; move a manual control to cancel")

    def request_dataset_home_return(self, position, rotation, clearance_z, position_tolerance_m):
        """Return to a fixed demonstration pose through a lifted Cartesian route."""
        if self.mode != "active":
            raise RuntimeError("enable teleop before returning home")
        if not (self.config.robot.human_linear_velocity_control
                and self.config.robot.human_angular_velocity_control):
            raise RuntimeError("home return requires direct velocity control")
        state = self._read_state()
        pose = pose_matrix(state.O_T_EE)
        position = np.asarray(position, dtype=np.float64)
        rotation = np.asarray(rotation, dtype=np.float64)
        position_tolerance_m = float(position_tolerance_m)
        bounds = self.config.robot
        lift_z = max(float(pose[2, 3]), float(position[2]), float(clearance_z))
        if (position.shape != (3,) or rotation.shape != (3, 3)
                or not np.isfinite(position).all() or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1.0)
                or np.any(position < bounds.workspace_min)
                or np.any(position > bounds.workspace_max)
                or lift_z > bounds.workspace_max[2]
                or not np.isfinite(position_tolerance_m)
                or not 0 < position_tolerance_m < self.config.pose_shortcuts.final_approach_distance_m):
            raise ValueError("dataset home pose or clearance is outside the valid workspace")
        waypoints = [
            ("home lift", np.array([*pose[:2, 3], lift_z]), pose[:3, :3].copy()),
            ("home transit", np.array([*position[:2], lift_z]), rotation.copy()),
            ("home settle", position.copy(), rotation.copy()),
        ]
        self._pose_return_path = waypoints[1:]
        self._home_position_tolerance_m = position_tolerance_m
        label, xyz, orientation = waypoints[0]
        self._pose_return = PoseReturn(label, xyz, orientation, self.clock() + self.config.pose_shortcuts.timeout_s)
        self.report("Returning to dataset home; move a manual control to cancel")

    def _pose_return_command(self, state, params):
        target = self._pose_return
        if target is None:
            raise RuntimeError("pose return is not active")
        if self.clock() > target.deadline:
            raise TimeoutError(f"{target.label} return timed out")
        pose = pose_matrix(state.O_T_EE)
        position_error = target.position - pose[:3, 3]
        distance = float(np.linalg.norm(position_error))
        shortcuts = self.config.pose_shortcuts
        relative = target.rotation @ pose[:3, :3].T
        angle = _rotation_angle(relative)
        position_tolerance = (
            self._home_position_tolerance_m
            if target.label.startswith("home ") else shortcuts.position_tolerance_m
        )
        position_done = distance <= position_tolerance
        rotation_done = np.degrees(angle) <= self.config.pose_shortcuts.orientation_tolerance_deg
        if position_done and rotation_done:
            label = target.label
            if self._pose_return_path:
                next_label, next_position, next_rotation = self._pose_return_path.pop(0)
                self._pose_return = PoseReturn(
                    next_label, next_position, next_rotation,
                    self.clock() + shortcuts.timeout_s,
                )
            else:
                self._pose_return = None
            self.report(f"Reached {label}; position_error={distance * 1000:.3f} mm; orientation_error={np.degrees(angle):.3f} deg")
            return np.zeros(3), np.zeros(3)
        linear_base = params.position_gain * position_error
        linear_norm = float(np.linalg.norm(linear_base))
        # Continuous distance-based cap: brake early, then creep toward the
        # target. Keep direction and send through the normal recorded velocity path.
        approach_speed = min(shortcuts.final_approach_speed_m_s, params.max_linear_speed)
        if distance <= shortcuts.final_approach_distance_m:
            speed_cap = approach_speed * distance / shortcuts.final_approach_distance_m
        else:
            fraction = min(1.0, (distance - shortcuts.final_approach_distance_m)
                           / (shortcuts.slowdown_distance_m - shortcuts.final_approach_distance_m))
            speed_cap = approach_speed + fraction * (params.max_linear_speed - approach_speed)
        if target.label == "saved pose":
            speed_cap = min(speed_cap, .03)
        if linear_norm > speed_cap:
            linear_base *= speed_cap / linear_norm
        angular_base = (
            _rotation_axis(relative, angle) * angle * params.orientation_gain
            if not rotation_done else np.zeros(3)
        )
        angular_norm = float(np.linalg.norm(angular_base))
        angular_cap = min(params.gamepad_max_angular_speed, .15) if target.label == "saved pose" else params.gamepad_max_angular_speed
        if angular_norm > angular_cap:
            angular_base *= angular_cap / angular_norm
        # send_human_command accepts angular velocity in the measured tool frame.
        return linear_base, pose[:3, :3].T @ angular_base

    def pause(self, reason="operator pause", *, fault=False):
        was_faulted = self.mode == "fault"
        # Freeze bounded diagnostic data before stopping/resetting the controller.
        snapshot = {
            "reason": reason, "host_monotonic_s": self.clock(),
            "minimum_success_rate": self.config.control.minimum_success_rate,
            "required_low_samples": self.config.control.low_rate_samples,
            "consecutive_low_samples": self._low_rates,
            "success_rate_monitor_started_at": self._success_rate_monitor_started_at,
            "speed_scale": self.speed_scale, "recent_states": list(self._rate_history),
        }
        self.mode = "fault" if fault or was_faulted else "standby"
        self.reason = reason
        self._paused_at = self.clock()
        self._pose_return = None
        self._pose_return_path = []
        self._view_button_ready = self._menu_button_ready = False
        self._reset_speed()
        try:
            self.robot.end_human_control()
        except Exception as error:
            self.mode = "fault"
            self.reason = f"{reason}; stop/hold failed: {error}"
        if self.mode == "fault" and not was_faulted:
            snapshot["reason"] = self.reason
            self.fault_snapshot = snapshot
        self.report(f"{self.mode.upper()}: {self.reason}; speed=LOW")

    def recover(self):
        if self.mode != "fault":
            self.report("当前没有待恢复故障，可启用或暂停控制。")
            return
        try:
            self.robot.recover_motion_fault()
            self._state_time, self._state_advanced, self._low_rates = None, False, 0
            self._read_state()
        except Exception as error:
            self.reason = str(error)
            self.report(f"Recovery failed; control stays disabled: {error}")
            return
        self.mode, self.reason = "standby", ""
        self._paused_at = self.clock()
        self._reset_speed()
        self._pose_return = None
        self._pose_return_path = []
        self._view_button_ready = self._menu_button_ready = False
        self.report("已恢复到待命；松开手柄输入后重新启用控制。")

    def tick(self):
        now = self.clock()
        elapsed, self._last_tick = now - self._last_tick, now
        if self.mode == "fault":
            return
        if self.mode == "active" and (elapsed <= 0 or elapsed > self.config.control.discontinuity_timeout_s):
            self.pause("control loop discontinuity; no catch-up motion")
            return
        try:
            state = self._read_state()
        except Exception as error:
            self.pause(str(error), fault=True)
            return
        try:
            packet = self._packet()
        except (OSError, RuntimeError, ValueError) as error:
            if self.mode == "active":
                self.pause(str(error))
            return
        if self.mode != "active":
            return
        dt = min(elapsed, 2 / self.config.control.control_rate)
        # Ramp and pose integration share the same bounded control time step.
        try:
            # Recheck the packet read by THIS tick: the managed adapter may have
            # checked an earlier packet. Cancellation must not apply one frame
            # of manual/gripper input or replace the saved target with a shortcut.
            if self._pose_return is not None and self._pose_return.label == "saved pose" and not self._neutral(packet):
                self.pause("saved pose return cancelled by gamepad input")
                return
            self._update_speed(packet)
            self._advance_speed(dt)
            params = self.control
            command = decode_cartesian_gamepad_values(
                packet.axes, packet.buttons, self.config.mapping, params,
            )
            manual_motion = self._has_manual_motion(packet)
            self._update_pose_shortcuts(packet, state, manual_motion=manual_motion)
            if self._pose_return is not None and manual_motion:
                self.report(f"Cancelled {self._pose_return.label} return")
                self._pose_return = None
                self._pose_return_path = []
            if self._pose_return is not None:
                # View and dataset home use the bounded HIGH translation limit;
                # Menu retains LOW drift correction.
                fast_return = self._pose_return.label == "initial pose" or self._pose_return.label.startswith("home ")
                scale = self.config.speed_modes.high_multiplier if fast_return else 1.0
                params = self._scaled_control(scale)
                linear_velocity, angular_velocity_tool = self._pose_return_command(state, params)
                gripper_motion = 0.0
            else:
                linear_velocity = command.linear_velocity_base
                angular_velocity_tool = command.angular_velocity_tool
                gripper_motion = command.gripper_motion
            self.robot.send_human_command(
                linear_velocity, 0.0, gripper_motion,
                dt,
                angular_velocity_tool=angular_velocity_tool,
                **params.motion_limits(),
            )
            if self._success_rate_monitor_started_at is None:
                self._success_rate_monitor_started_at = self.clock()
            if self.on_command is not None:
                self.on_command(self.robot.command_debug_snapshot())
        except Exception as error:
            self.pause(str(error), fault=True)


def ros_environment():
    keep = ("HOME", "USER", "LOGNAME", "LANG", "TERM", "ROS_DOMAIN_ID", "ROS_LOCALHOST_ONLY", "RMW_IMPLEMENTATION")
    result = {key: os.environ[key] for key in keep if key in os.environ}
    result.update(PATH="/usr/bin:/bin:/usr/sbin:/sbin", PYTHONNOUSERSITE="1",
                  PYTHONPATH=os.pathsep.join(str(workspace() / path) for path in ("ros_ws/python", "ros_ws/python_deps", "src")))
    result.update(ROS_HOME=str(workspace() / "runtime/ros"), ROS_LOG_DIR=str(workspace() / "runtime/ros/log"))
    return result


def ros_command(*args):
    setup = os.environ.get("ROS_SETUP", "/opt/ros/humble/setup.bash")
    if not Path(setup).is_file():
        raise FileNotFoundError(f"Missing ROS setup: {setup}")
    return ["/bin/bash", "-c", 'set -e; set +u; source "$1"; shift; exec "$@"',
            "vla-gamepad-ros", setup, *args]


def check_process_conflicts(port: str):
    target = Path(port).resolve()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if "ros2_control_node" in command and "/left" in command:
                raise RuntimeError(f"Left ROS controller is already running (pid={proc.name})")
            for fd in (proc / "fd").iterdir():
                try:
                    if fd.resolve() == target:
                        raise RuntimeError(f"Robotiq serial port is already open (pid={proc.name})")
                except OSError:
                    continue
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue


def preflight(config: TeleopConfig):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Franky teleop requires the vla Python 3.12 environment")
    for module in ("franky", "pymodbus", "serial"):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(f"Missing module in {sys.executable}: {module}")
    for path, mode in ((Path(f"/dev/input/js{config.joy_device_id}"), os.R_OK),
                       (Path(config.robot.gripper_port), os.R_OK | os.W_OK)):
        if not path.exists() or not os.access(path, mode):
            raise PermissionError(f"Device missing or inaccessible: {path}")
    cpu = config.robot.realtime_cpu
    if cpu is not None and (cpu not in os.sched_getaffinity(0) or len(os.sched_getaffinity(0)) < 2):
        raise RuntimeError(f"Realtime CPU {cpu} and at least one housekeeping CPU must be available")
    check_process_conflicts(config.robot.gripper_port)
    with HardwareLease(config.robot.robot_ip, config.robot.gripper_port):
        pass
    result = subprocess.run(
        ros_command("ros2", "node", "list", "--no-daemon", "--spin-time", "1"),
        env=ros_environment(), text=True, capture_output=True, timeout=15, check=True,
    )
    if any(line.strip().startswith("/left/") and "controller_manager" in line for line in result.stdout.splitlines()):
        raise RuntimeError("Left ROS controller manager is active; stop its owning launcher first")
    subprocess.run(
        ros_command("/usr/bin/python3", "-c", "import rclpy; from sensor_msgs.msg import Joy"),
        env=ros_environment(), check=True, timeout=10,
    )
    result = subprocess.run(ros_command("ros2", "pkg", "executables", "joy"),
                            env=ros_environment(), text=True, capture_output=True, check=True, timeout=10)
    if "joy joy_node" not in result.stdout:
        raise RuntimeError("ROS joy_node is not installed")


class JoySidecars:
    def __init__(self, config, socket_path: Path, runtime: Path):
        self.config, self.socket_path, self.runtime = config, socket_path, runtime
        self.children = []

    def __enter__(self):
        mapping = self.config.mapping
        commands = [
            ("joy", ros_command("ros2", "run", "joy", "joy_node", "--ros-args",
             "-r", "__node:=left_gamepad_teleop_joy", "-r", f"joy:={mapping.joy_topic}",
             "-p", f"device_id:={self.config.joy_device_id}", "-p", "deadzone:=0.0",
             "-p", f"autorepeat_rate:={self.config.control.control_rate}")),
            ("bridge", ros_command("/usr/bin/python3", "-m", "vla.datacollection.franka_robotiq_rollout.joy_bridge",
             "--topic", mapping.joy_topic, "--socket-path", str(self.socket_path),
             "--node-name", "left_gamepad_teleop_bridge")),
        ]
        try:
            for name, command in commands:
                log = diagnostic_path(f"{name}.log", self.runtime / f"{name}.log")
                with log.open("wb") as stream:
                    child = subprocess.Popen(command, env=ros_environment(), start_new_session=True,
                                             stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                self.children.append((child, log))
        except BaseException:
            self.close()
            raise
        return self

    def check(self):
        for child, log in self.children:
            if child.poll() is not None:
                raise RuntimeError(f"Joy sidecar exited ({child.returncode}):\n{log.read_text(errors='replace')[-4000:]}")

    def close(self):
        for child, _ in reversed(self.children):
            # Signal owned groups, including descendants of the ros2 wrapper.
            for sig, timeout in ((signal.SIGINT, 3), (signal.SIGTERM, 2), (signal.SIGKILL, 1)):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    break
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    child.poll()
                    try:
                        os.killpg(child.pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.05)
                else:
                    continue
                break
            child.wait(timeout=1)
        self.children.clear()

    def __exit__(self, *_):
        self.close()


@contextmanager
def keyboard():
    fd = sys.stdin.fileno()
    settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield fd
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, settings)


def run(config):
    if not sys.stdin.isatty():
        raise PermissionError("--execute requires an interactive terminal")
    print(f"Left Franka {config.robot.robot_ip}; gripper {config.robot.gripper_port}")
    print("Check the workspace and emergency stop. Gripper activation may be required; no home/open target is sent.")
    if config.robot.realtime_cpu is not None:
        os.sched_setaffinity(0, os.sched_getaffinity(0) - {config.robot.realtime_cpu})
    with tempfile.TemporaryDirectory(prefix="vla-gamepad-") as directory:
        runtime = Path(directory)
        receiver = JoyReceiver(runtime / "joy.sock")
        robot = None
        try:
            robot = FrankaRobotiqRobot(replace(config.robot, calibration_dir=runtime / "calibration"))
            with JoySidecars(config, receiver.path, runtime) as sidecars, (nullcontext(print) if current_reporter() else AsyncReporter()) as reporter:
                robot.connect()
                controller = GamepadTeleop(config, robot, receiver, report=reporter)
                print("STANDBY: s=enable, d=pause, r=recover, c/Esc/Ctrl+C=exit", flush=True)
                try:
                    with keyboard() as fd:
                        period = 1 / config.control.control_rate
                        next_status = 0.0
                        while True:
                            started = time.monotonic()
                            sidecars.check()
                            controller.tick()
                            if started >= next_status:
                                console_status(f"{controller.mode.upper()} source=GAMEPAD speed={'Fast' if controller.high_speed else 'Normal'}")
                                next_status = started + 1.0
                            if select.select([fd], [], [], 0)[0]:
                                keys = os.read(fd, 32).decode(errors="ignore").lower()
                                if not keys or any(key in keys for key in ("c", "\x1b", "\x04")):
                                    break
                                for key in keys:
                                    if key == "s":
                                        controller.enable()
                                    elif key == "d":
                                        controller.pause()
                                    elif key == "r":
                                        controller.recover()
                            time.sleep(max(0, period - (time.monotonic() - started)))
                finally:
                    controller.pause("shutdown")
                    robot.disconnect()
        finally:
            try:
                if robot is not None and robot.is_connected:
                    robot.disconnect()
            finally:
                receiver.close()


@terminal_session("gamepad-teleop")
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        config = load_config(args.config.expanduser().resolve())
        preflight(config)
        if not args.execute:
            print("Preflight passed; no robot connection or motion was requested.")
            return
        previous = signal.signal(signal.SIGTERM, _interrupt)
        try:
            run(config)
        finally:
            signal.signal(signal.SIGTERM, previous)
    except KeyboardInterrupt:
        print("Gamepad teleop stopped.", flush=True)
    except Exception as error:
        print(f"Gamepad teleop failed: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error


def _interrupt(*_):
    raise KeyboardInterrupt


if __name__ == "__main__":
    main()
