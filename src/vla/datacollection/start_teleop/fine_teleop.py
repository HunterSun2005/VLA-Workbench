#!/usr/bin/env python3
"""Arbitrate left GELLO and Xbox fine Cartesian teleoperation commands."""

from __future__ import annotations

import argparse
import math
import signal
import time
from enum import Enum
from typing import Sequence

import numpy as np
import PyKDL
import rclpy
from control_msgs.action import GripperCommand
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import parameter_value_to_python
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import Float32, Float64MultiArray
from std_srvs.srv import SetBool

try:
    from .franka_kinematics import build_kinematic_model, joint7_step
except ImportError:  # Direct execution by start_teleop.sh.
    from franka_kinematics import build_kinematic_model, joint7_step


class Mode(Enum):
    GELLO = "gello"
    FINE = "fine_gamepad"
    RETURNING = "returning_to_gello"


def apply_deadzone(value: float, deadzone: float) -> float:
    """Apply a continuous, rescaled symmetric deadzone."""
    if abs(value) <= deadzone:
        return 0.0
    magnitude = min(1.0, (abs(value) - deadzone) / (1.0 - deadzone))
    return math.copysign(magnitude, value)


def normalize_trigger(raw_value: float) -> float:
    """Convert the xpad trigger range (+1 released, -1 pressed) to [0, 1]."""
    return float(np.clip((1.0 - raw_value) * 0.5, 0.0, 1.0))


def width_percent_to_joint_position(
    width_percent: float, open_position: float, closed_position: float
) -> float:
    width_percent = float(np.clip(width_percent, 0.0, 1.0))
    return closed_position + width_percent * (open_position - closed_position)


def joint_position_to_width_percent(
    joint_position: float, open_position: float, closed_position: float
) -> float:
    span = open_position - closed_position
    if abs(span) < 1e-9:
        raise ValueError("gripper open_position and closed_position must differ")
    return float(np.clip((joint_position - closed_position) / span, 0.0, 1.0))


def move_towards(current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    delta = np.clip(target - current, -max_step, max_step)
    return current + delta


def _vector(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _clamp_norm(values: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm <= maximum or norm == 0.0:
        return values
    return values * (maximum / norm)


class FineTeleopNode(Node):
    def __init__(self, *, managed=False, **node_options) -> None:
        super().__init__("left_fine_teleop", **node_options)
        self.managed = managed
        self.managed_enabled = not managed
        self.managed_fault = None

        self._declare_parameters()
        self.control_rate = float(self.get_parameter("control_rate").value)
        self.joy_timeout = float(self.get_parameter("joy_timeout").value)
        self.source_timeout = float(self.get_parameter("source_timeout").value)
        self.deadzone = float(self.get_parameter("deadzone").value)
        self.max_xy_speed = float(self.get_parameter("max_xy_speed").value)
        self.max_z_speed = float(self.get_parameter("max_z_speed").value)
        self.max_joint_speed = float(self.get_parameter("max_joint_speed").value)
        self.joint7_speed = float(self.get_parameter("joint7_speed").value)
        self.return_joint_speed = float(self.get_parameter("return_joint_speed").value)
        self.gripper_speed = float(self.get_parameter("gripper_speed").value)
        self.gripper_return_speed = float(self.get_parameter("gripper_return_speed").value)
        self.gripper_trigger_deadzone = float(
            self.get_parameter("gripper_trigger_deadzone").value
        )
        self.gripper_open_position = float(
            self.get_parameter("gripper_open_position").value
        )
        self.gripper_closed_position = float(
            self.get_parameter("gripper_closed_position").value
        )
        self.gripper_max_effort = float(
            self.get_parameter("gripper_max_effort").value
        )
        self.position_gain = float(self.get_parameter("position_gain").value)
        self.orientation_gain = float(self.get_parameter("orientation_gain").value)
        self.max_orientation_speed = float(self.get_parameter("max_orientation_speed").value)
        self.damping = float(self.get_parameter("damping").value)
        self.return_joint_tolerance = float(self.get_parameter("return_joint_tolerance").value)
        self.return_gripper_tolerance = float(
            self.get_parameter("return_gripper_tolerance").value
        )
        self.left_stick_x_axis = int(self.get_parameter("left_stick_x_axis").value)
        self.left_stick_y_axis = int(self.get_parameter("left_stick_y_axis").value)
        self.left_trigger_axis = int(self.get_parameter("left_trigger_axis").value)
        self.right_trigger_axis = int(self.get_parameter("right_trigger_axis").value)
        self.a_button = int(self.get_parameter("a_button").value)
        self.b_button = int(self.get_parameter("b_button").value)
        self.x_button = int(self.get_parameter("x_button").value)
        self.y_button = int(self.get_parameter("y_button").value)
        self.joint7_button_sign = float(
            self.get_parameter("joint7_button_sign").value
        )
        self.left_stick_x_sign = float(self.get_parameter("left_stick_x_sign").value)
        self.left_stick_y_sign = float(self.get_parameter("left_stick_y_sign").value)

        self._validate_parameters()
        self.model = self._load_kinematic_model()
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(self.model.chain)
        self.jacobian_solver = PyKDL.ChainJntToJacSolver(self.model.chain)

        self.mode = Mode.GELLO
        self.raw_gello_joints: np.ndarray | None = None
        self.raw_gello_names: list[str] = []
        self.raw_gello_time = 0.0
        self.raw_gripper: float | None = None
        self.raw_gripper_time = 0.0
        self.measured_joints: np.ndarray | None = None
        self.measured_joint_time = 0.0
        self.measured_gripper: float | None = None
        self.measured_gripper_time = 0.0
        self.joy: Joy | None = None
        self.joy_time = 0.0
        self.command_joints: np.ndarray | None = None
        self.command_gripper: float | None = None
        self.target_position: np.ndarray | None = None
        self.target_rotation: PyKDL.Rotation | None = None
        self.gripper_direction = 0
        self.return_gripper_goal: float | None = None
        self._gripper_goal_futures: set[rclpy.task.Future] = set()
        self._pending_mode_request: bool | None = None
        self.last_tick = time.monotonic()
        self.joy_stale_reported = False

        raw_joint_topic = str(self.get_parameter("raw_joint_topic").value)
        raw_gripper_topic = str(self.get_parameter("raw_gripper_topic").value)
        measured_joint_topic = str(self.get_parameter("measured_joint_topic").value)
        measured_gripper_topic = str(self.get_parameter("measured_gripper_topic").value)
        joy_topic = str(self.get_parameter("joy_topic").value)
        command_joint_topic = str(self.get_parameter("command_joint_topic").value)
        command_gripper_topic = str(self.get_parameter("command_gripper_topic").value)
        action_gripper_command_topic = str(
            self.get_parameter("action_gripper_command_topic").value
        )
        drive_gripper_command_topic = str(self.get_parameter("drive_gripper_command_topic").value)
        if drive_gripper_command_topic in (action_gripper_command_topic, command_gripper_topic):
            raise ValueError("gripper drive and recording topics must be distinct")
        gripper_speed_topic = str(self.get_parameter("gripper_speed_topic").value)
        gripper_action_topic = str(self.get_parameter("gripper_action_topic").value)

        self.joint_publisher = self.create_publisher(JointState, command_joint_topic, 10)
        self.gripper_publisher = self.create_publisher(Float32, command_gripper_topic, 10)
        self.gripper_action_command_publisher = self.create_publisher(
            Float32, action_gripper_command_topic, 10
        )
        self.gripper_drive_publisher = self.create_publisher(Float32, drive_gripper_command_topic, 10)
        self.gripper_speed_publisher = self.create_publisher(
            Float64MultiArray, gripper_speed_topic, 10
        )
        self.create_subscription(JointState, raw_joint_topic, self._raw_joint_callback, 10)
        self.create_subscription(Float32, raw_gripper_topic, self._raw_gripper_callback, 10)
        self.create_subscription(JointState, measured_joint_topic, self._measured_joint_callback, 10)
        self.create_subscription(
            JointState, measured_gripper_topic, self._measured_gripper_callback, 10
        )
        self.create_subscription(Joy, joy_topic, self._joy_callback, 10)
        self.gripper_action_client = ActionClient(
            self, GripperCommand, gripper_action_topic
        )
        action_timeout = float(self.get_parameter("gripper_action_timeout").value)
        if not self.gripper_action_client.wait_for_server(timeout_sec=action_timeout):
            raise RuntimeError(
                f"Robotiq action server {gripper_action_topic!r} is unavailable"
            )
        if not managed:
            signal.signal(signal.SIGUSR1, self._request_fine_from_signal)
            signal.signal(signal.SIGUSR2, self._request_gello_from_signal)
        self.create_service(SetBool, "~/set_fine_mode", self._set_fine_mode)
        self.create_timer(1.0 / self.control_rate, self._control_tick)
        self._publish_gripper_speed(self.gripper_return_speed)

        self.get_logger().info(
            "Left command arbiter ready: s=true enters fine mode; d=false returns to GELLO"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "control_rate": 100.0,
            "joy_timeout": 0.2,
            "source_timeout": 0.5,
            "deadzone": 0.08,
            "max_xy_speed": 0.02,
            "max_z_speed": 0.01,
            "max_joint_speed": 0.30,
            "joint7_speed": 0.30,
            "return_joint_speed": 0.20,
            "gripper_speed": 0.40,
            "gripper_return_speed": 1.0,
            "gripper_trigger_deadzone": 0.05,
            "gripper_open_position": 0.0,
            "gripper_closed_position": 0.8,
            "gripper_max_effort": 50.0,
            "gripper_action_timeout": 30.0,
            "position_gain": 8.0,
            "orientation_gain": 3.0,
            "max_orientation_speed": 0.30,
            "damping": 0.05,
            "joint_limit_margin": 0.05,
            "return_joint_tolerance": 0.01,
            "return_gripper_tolerance": 0.02,
            "left_stick_x_axis": 0,
            "left_stick_y_axis": 1,
            "left_trigger_axis": 2,
            "right_trigger_axis": 5,
            "a_button": 0,
            "b_button": 1,
            "x_button": 2,
            "y_button": 3,
            "joint7_button_sign": -1.0,
            "left_stick_x_sign": 1.0,
            "left_stick_y_sign": 1.0,
            "base_link": "left_fr3_link0",
            "tip_link": "left_fr3_link8",
            "robot_state_publisher": "/left/robot_state_publisher",
            "raw_joint_topic": "/left/gello/joint_states",
            "raw_gripper_topic": "/left/gello/gripper_width_percent",
            "measured_joint_topic": (
                "/left/franka_robot_state_broadcaster/measured_joint_states"
            ),
            "measured_gripper_topic": (
                "/left/gripper/joint_state_broadcaster_robotiq/joint_states"
            ),
            "joy_topic": "/gamepad/joy",
            "command_joint_topic": "/left/teleop/joint_states",
            "command_gripper_topic": "/left/teleop/gripper_width_percent",
            "action_gripper_command_topic": (
                "/left/teleop/gripper_action_width_percent"
            ),
            "drive_gripper_command_topic": "/left/teleop/gripper_drive_width_percent",
            "gripper_speed_topic": (
                "/left/gripper/left_gripper_speed_controller/commands"
            ),
            "gripper_action_topic": (
                "/left/gripper/robotiq_gripper_controller/gripper_cmd"
            ),
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _validate_parameters(self) -> None:
        positive = {
            "control_rate": self.control_rate,
            "joy_timeout": self.joy_timeout,
            "source_timeout": self.source_timeout,
            "max_xy_speed": self.max_xy_speed,
            "max_z_speed": self.max_z_speed,
            "max_joint_speed": self.max_joint_speed,
            "joint7_speed": self.joint7_speed,
            "return_joint_speed": self.return_joint_speed,
            "gripper_speed": self.gripper_speed,
            "gripper_return_speed": self.gripper_return_speed,
            "position_gain": self.position_gain,
            "orientation_gain": self.orientation_gain,
            "max_orientation_speed": self.max_orientation_speed,
            "damping": self.damping,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError("Parameters must be positive: " + ", ".join(invalid))
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError("deadzone must be in [0, 1)")
        if not 0.0 <= self.gripper_trigger_deadzone < 1.0:
            raise ValueError("gripper_trigger_deadzone must be in [0, 1)")
        if not 0.0 < self.gripper_speed <= 1.0:
            raise ValueError("gripper_speed must be in (0, 1]")
        if not 0.0 < self.gripper_return_speed <= 1.0:
            raise ValueError("gripper_return_speed must be in (0, 1]")
        if abs(self.gripper_open_position - self.gripper_closed_position) < 1e-9:
            raise ValueError("gripper open and closed positions must differ")
        if self.gripper_max_effort < 0.0:
            raise ValueError("gripper_max_effort must be non-negative")
        if self.joint7_button_sign not in {-1.0, 1.0}:
            raise ValueError("joint7_button_sign must be -1.0 or 1.0")

    def _load_kinematic_model(self) -> KinematicModel:
        remote_node = str(self.get_parameter("robot_state_publisher").value)
        service_name = f"{remote_node.rstrip('/')}/get_parameters"
        client = self.create_client(GetParameters, service_name)
        if not client.wait_for_service(timeout_sec=30.0):
            raise RuntimeError(f"Parameter service for {remote_node} is unavailable")
        request = GetParameters.Request()
        request.names = ["robot_description"]
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        response = future.result()
        if response is None or not response.values:
            raise RuntimeError(f"Failed to read robot_description from {remote_node}")
        robot_description = parameter_value_to_python(response.values[0])
        if not isinstance(robot_description, str) or not robot_description:
            raise RuntimeError(f"robot_description from {remote_node} is empty")
        return build_kinematic_model(
            robot_description,
            str(self.get_parameter("base_link").value),
            str(self.get_parameter("tip_link").value),
            float(self.get_parameter("joint_limit_margin").value),
        )

    def _raw_joint_callback(self, msg: JointState) -> None:
        if len(msg.position) < 7:
            self.get_logger().warning("Ignoring GELLO message with fewer than 7 joints")
            return
        self.raw_gello_joints = _vector(msg.position[:7])
        self.raw_gello_names = list(msg.name[:7])
        self.raw_gello_time = time.monotonic()

    def _raw_gripper_callback(self, msg: Float32) -> None:
        self.raw_gripper = float(np.clip(msg.data, 0.0, 1.0))
        self.raw_gripper_time = time.monotonic()

    def _measured_joint_callback(self, msg: JointState) -> None:
        if msg.name:
            positions = dict(zip(msg.name, msg.position))
            if not all(name in positions for name in self.model.joint_names):
                return
            self.measured_joints = _vector([positions[name] for name in self.model.joint_names])
        elif len(msg.position) >= 7:
            self.measured_joints = _vector(msg.position[:7])
        else:
            return
        self.measured_joint_time = time.monotonic()

    def _measured_gripper_callback(self, msg: JointState) -> None:
        joint_name = "robotiq_85_left_knuckle_joint"
        if msg.name:
            positions = dict(zip(msg.name, msg.position))
            if joint_name not in positions:
                return
            position = float(positions[joint_name])
        elif msg.position:
            position = float(msg.position[0])
        else:
            return
        self.measured_gripper = joint_position_to_width_percent(
            position, self.gripper_open_position, self.gripper_closed_position
        )
        self.measured_gripper_time = time.monotonic()

    def _joy_callback(self, msg: Joy) -> None:
        self.joy = msg
        self.joy_time = time.monotonic()

    def _set_fine_mode(self, request: SetBool.Request, response: SetBool.Response):
        response.success, response.message = self._request_mode(bool(request.data))
        return response

    def _request_fine_from_signal(self, _signum, _frame) -> None:
        self._pending_mode_request = True

    def _request_gello_from_signal(self, _signum, _frame) -> None:
        self._pending_mode_request = False

    def _request_mode(self, fine_enabled: bool) -> tuple[bool, str]:
        if getattr(self, "managed", False) and not self.managed_enabled:
            return False, "Enable managed teleoperation first"
        if fine_enabled:
            if self.mode == Mode.FINE:
                return True, "Fine mode is already active"
            if self.mode == Mode.RETURNING:
                return False, "Wait for return-to-GELLO to finish"
            error = self._start_fine_mode()
            return error is None, error or "Fine gamepad mode enabled"

        if self.mode == Mode.GELLO:
            return True, "GELLO mode is already active"
        now = time.monotonic()
        if (
            self.raw_gripper is None
            or now - self.raw_gripper_time > self.source_timeout
            or self.measured_gripper is None
            or now - self.measured_gripper_time > self.source_timeout
        ):
            self._stop_fine_gripper()
            return False, "Cannot return: gripper command/state is missing or stale"
        self.mode = Mode.RETURNING
        self.gripper_direction = 0
        self._publish_gripper_speed(self.gripper_return_speed)
        self._send_gripper_goal(self.raw_gripper)
        self.return_gripper_goal = self.raw_gripper
        message = "Returning arm and gripper to current GELLO targets"
        self.get_logger().info(message)
        return True, message

    def _start_fine_mode(self) -> str | None:
        now = time.monotonic()
        if self.joy is None or now - self.joy_time > self.joy_timeout:
            return "Cannot enter fine mode: /gamepad/joy is missing or stale"
        if self.measured_joints is None or now - self.measured_joint_time > self.source_timeout:
            return "Cannot enter fine mode: measured left joint state is missing or stale"
        if self.raw_gripper is None or now - self.raw_gripper_time > self.source_timeout:
            return "Cannot enter fine mode: left GELLO gripper state is missing or stale"
        if (
            self.measured_gripper is None
            or now - self.measured_gripper_time > self.source_timeout
        ):
            return "Cannot enter fine mode: measured left gripper state is missing or stale"

        self.command_joints = self.measured_joints.copy()
        frame = self._forward_kinematics(self.command_joints)
        self.target_position = _vector([frame.p.x(), frame.p.y(), frame.p.z()])
        self.target_rotation = frame.M
        self.command_gripper = self.measured_gripper
        self.gripper_direction = 0
        self.return_gripper_goal = None
        self._publish_gripper_speed(0.0)
        self._send_gripper_goal(self.command_gripper)
        self.mode = Mode.FINE
        self.last_tick = time.monotonic()
        self.get_logger().info(
            "Fine mode enabled: left stick=XY, Y/A=Z, X/B=joint 7, "
            "LT=open, RT=close"
        )
        return None

    def managed_enable(self):
        now = time.monotonic()
        if self.managed_fault:
            raise RuntimeError(self.managed_fault)
        if any(value is None or now - stamp > self.source_timeout for value, stamp in (
            (self.measured_joints, self.measured_joint_time), (self.raw_gello_joints, self.raw_gello_time),
            (self.measured_gripper, self.measured_gripper_time), (self.raw_gripper, self.raw_gripper_time),
        )):
            raise RuntimeError("GELLO or measured state is unavailable/stale")
        self.command_joints = self.measured_joints.copy()
        self.command_gripper = self.measured_gripper
        self.managed_enabled = True
        self.mode = Mode.FINE
        ok, reason = self._request_mode(False)
        if not ok:
            self.managed_enabled = False
            raise RuntimeError(reason)

    def managed_pause(self):
        self.managed_enabled = False
        self._pending_mode_request = None
        self._stop_fine_gripper()
        self.command_joints = None if self.measured_joints is None else self.measured_joints.copy()
        self.command_gripper = self.measured_gripper
        self.mode = Mode.GELLO

    def _control_tick(self) -> None:
        if getattr(self, "managed", False):
            if self.managed_enabled:
                now = time.monotonic()
                if (now - self.measured_joint_time > self.source_timeout
                        or now - self.measured_gripper_time > self.source_timeout
                        or (self.mode != Mode.FINE and (now - self.raw_gello_time > self.source_timeout
                                                      or now - self.raw_gripper_time > self.source_timeout))):
                    self.managed_fault = "Managed teleoperation input/state became stale"
                    self.managed_pause()
            if not self.managed_enabled:
                return
        pending_mode_request = self._pending_mode_request
        if pending_mode_request is not None:
            self._pending_mode_request = None
            success, message = self._request_mode(pending_mode_request)
            if not success:
                self.get_logger().warning(f"Keyboard mode request rejected: {message}")

        now = time.monotonic()
        dt = min(max(now - self.last_tick, 1.0 / (2.0 * self.control_rate)), 0.05)
        self.last_tick = now

        if self.mode == Mode.GELLO:
            self._update_gello_mode(now)
        elif self.mode == Mode.FINE:
            self._update_fine_mode(now, dt)
        else:
            self._update_return_mode(now, dt)

        self._publish_commands(now)

    def _update_gello_mode(self, now: float) -> None:
        if self.raw_gello_joints is not None and now - self.raw_gello_time <= self.source_timeout:
            self.command_joints = self.raw_gello_joints.copy()
        if self.raw_gripper is not None and now - self.raw_gripper_time <= self.source_timeout:
            self.command_gripper = self.raw_gripper

    def _update_fine_mode(self, now: float, dt: float) -> None:
        if self.command_joints is None or self.target_position is None or self.target_rotation is None:
            return

        joy_is_fresh = self.joy is not None and now - self.joy_time <= self.joy_timeout
        measured_is_fresh = (
            self.measured_joints is not None
            and now - self.measured_joint_time <= self.source_timeout
            and self.measured_gripper is not None
            and now - self.measured_gripper_time <= self.source_timeout
        )
        if not joy_is_fresh or not measured_is_fresh:
            if not self.joy_stale_reported:
                self.get_logger().warning(
                    "Fine input/state stale; holding the Cartesian and gripper targets"
                )
                self._stop_fine_gripper()
                self.joy_stale_reported = True
            return
        self.joy_stale_reported = False

        assert self.joy is not None
        try:
            stick_x = apply_deadzone(self.joy.axes[self.left_stick_x_axis], self.deadzone)
            stick_y = apply_deadzone(self.joy.axes[self.left_stick_y_axis], self.deadzone)
            left_trigger = normalize_trigger(self.joy.axes[self.left_trigger_axis])
            right_trigger = normalize_trigger(self.joy.axes[self.right_trigger_axis])
            a_button = float(self.joy.buttons[self.a_button])
            b_button = float(self.joy.buttons[self.b_button])
            x_button = float(self.joy.buttons[self.x_button])
            y_button = float(self.joy.buttons[self.y_button])
        except IndexError:
            self.get_logger().error(
                "Configured Xbox axis/button index is absent from /gamepad/joy"
            )
            self._stop_fine_gripper()
            return

        velocity = _vector(
            [
                self.left_stick_y_sign * stick_y * self.max_xy_speed,
                self.left_stick_x_sign * stick_x * self.max_xy_speed,
                (y_button - a_button) * self.max_z_speed,
            ]
        )
        self.target_position += velocity * dt
        joint7_target = joint7_step(
            self.command_joints[6],
            x_button,
            b_button,
            self.joint7_speed,
            self.joint7_button_sign,
            dt,
            self.model.lower_limits[6],
            self.model.upper_limits[6],
        )
        command_joints = self._cartesian_step(
            self.command_joints,
            self.target_position,
            self.target_rotation,
            dt,
            velocity,
        )
        if x_button != b_button:
            command_joints[6] = joint7_target
            self.target_rotation = self._forward_kinematics(command_joints).M
        self.command_joints = command_joints
        trigger_motion = left_trigger - right_trigger
        if abs(trigger_motion) <= self.gripper_trigger_deadzone:
            trigger_motion = 0.0
        else:
            trigger_motion = math.copysign(
                (abs(trigger_motion) - self.gripper_trigger_deadzone)
                / (1.0 - self.gripper_trigger_deadzone),
                trigger_motion,
            )
        self._update_fine_gripper(trigger_motion)

    def _update_return_mode(self, now: float, dt: float) -> None:
        if (
            self.raw_gello_joints is None
            or self.raw_gripper is None
            or now - self.raw_gello_time > self.source_timeout
            or now - self.raw_gripper_time > self.source_timeout
        ):
            return
        if self.command_joints is None:
            self.command_joints = self.raw_gello_joints.copy()
        self.command_joints = move_towards(
            self.command_joints, self.raw_gello_joints, self.return_joint_speed * dt
        )
        if (
            self.return_gripper_goal is None
            or abs(self.raw_gripper - self.return_gripper_goal)
            >= self.return_gripper_tolerance
        ):
            self._send_gripper_goal(self.raw_gripper)
            self.return_gripper_goal = self.raw_gripper

        arm_error = float(np.max(np.abs(self.command_joints - self.raw_gello_joints)))
        gripper_is_fresh = (
            self.measured_gripper is not None
            and now - self.measured_gripper_time <= self.source_timeout
        )
        gripper_error = (
            abs(self.measured_gripper - self.raw_gripper)
            if gripper_is_fresh and self.measured_gripper is not None
            else float("inf")
        )
        if (
            arm_error <= self.return_joint_tolerance
            and gripper_error <= self.return_gripper_tolerance
        ):
            self.command_joints = self.raw_gello_joints.copy()
            self.command_gripper = self.raw_gripper
            self.mode = Mode.GELLO
            self.return_gripper_goal = None
            self.get_logger().info("Return complete; GELLO control restored")

    def _update_fine_gripper(self, trigger_motion: float) -> None:
        direction = 0 if trigger_motion == 0.0 else (1 if trigger_motion > 0.0 else -1)
        self._publish_gripper_speed(abs(trigger_motion) * self.gripper_speed)
        if direction == self.gripper_direction:
            return

        if direction == 0:
            self._send_gripper_goal(self._current_gripper_width())
        else:
            self._send_gripper_goal(1.0 if direction > 0 else 0.0)
        self.gripper_direction = direction

    def _stop_fine_gripper(self) -> None:
        self._publish_gripper_speed(0.0)
        if self.gripper_direction != 0:
            self._send_gripper_goal(self._current_gripper_width())
            self.gripper_direction = 0

    def _current_gripper_width(self) -> float:
        if self.measured_gripper is not None:
            return self.measured_gripper
        if self.command_gripper is not None:
            return self.command_gripper
        if self.raw_gripper is not None:
            return self.raw_gripper
        return 1.0

    def _publish_gripper_speed(self, speed: float) -> None:
        msg = Float64MultiArray()
        msg.data = [float(np.clip(speed, 0.0, 1.0))]
        self.gripper_speed_publisher.publish(msg)

    def _send_gripper_goal(self, width_percent: float) -> None:
        if not np.isfinite(width_percent):
            raise ValueError("gripper target must be finite")
        width_percent = float(np.clip(width_percent, 0.0, 1.0))
        goal = GripperCommand.Goal()
        goal.command.position = width_percent_to_joint_position(
            width_percent, self.gripper_open_position, self.gripper_closed_position
        )
        goal.command.max_effort = self.gripper_max_effort
        future = self.gripper_action_client.send_goal_async(goal)
        self.command_gripper = width_percent
        self._gripper_goal_futures.add(future)
        future.add_done_callback(self._gripper_goal_response)

    def _gripper_goal_response(self, future: rclpy.task.Future) -> None:
        self._gripper_goal_futures.discard(future)
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"Failed to send fine Robotiq goal: {exc}")
            return
        if not goal_handle.accepted:
            self.get_logger().error("Fine Robotiq goal was rejected")
            return
        result_future = goal_handle.get_result_async()
        self._gripper_goal_futures.add(result_future)
        result_future.add_done_callback(self._gripper_goal_result)

    def _gripper_goal_result(self, future: rclpy.task.Future) -> None:
        self._gripper_goal_futures.discard(future)
        try:
            future.result()
        except Exception as exc:
            self.get_logger().error(f"Fine Robotiq action failed: {exc}")

    def prepare_shutdown(self) -> None:
        self._publish_gripper_speed(0.0)
        if self.mode != Mode.GELLO:
            self._send_gripper_goal(self._current_gripper_width())

    def _cartesian_step(
        self,
        joints: np.ndarray,
        target_position: np.ndarray,
        target_rotation: PyKDL.Rotation,
        dt: float,
        target_velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        current = self._forward_kinematics(joints)
        target = PyKDL.Frame(
            target_rotation,
            PyKDL.Vector(*(float(value) for value in target_position)),
        )
        error = PyKDL.diff(current, target)
        linear = self.position_gain * _vector(
            [error.vel.x(), error.vel.y(), error.vel.z()]
        )
        if target_velocity is not None:
            linear += target_velocity
        linear = _clamp_norm(linear, 2.0 * self.max_xy_speed)
        angular = _clamp_norm(
            self.orientation_gain * _vector([error.rot.x(), error.rot.y(), error.rot.z()]),
            self.max_orientation_speed,
        )
        twist = np.concatenate((linear, angular))

        jacobian = self._jacobian(joints)
        regularized = jacobian @ jacobian.T + (self.damping**2) * np.eye(6)
        try:
            joint_velocity = jacobian.T @ np.linalg.solve(regularized, twist)
        except np.linalg.LinAlgError:
            self.get_logger().warning("Jacobian solve failed; holding fine target")
            return joints
        joint_velocity = np.clip(joint_velocity, -self.max_joint_speed, self.max_joint_speed)
        command = joints + joint_velocity * dt
        return np.clip(command, self.model.lower_limits, self.model.upper_limits)

    def _forward_kinematics(self, joints: np.ndarray) -> PyKDL.Frame:
        values = PyKDL.JntArray(7)
        for index, value in enumerate(joints):
            values[index] = float(value)
        frame = PyKDL.Frame()
        if self.fk_solver.JntToCart(values, frame) < 0:
            raise RuntimeError("KDL forward kinematics failed")
        return frame

    def _jacobian(self, joints: np.ndarray) -> np.ndarray:
        values = PyKDL.JntArray(7)
        for index, value in enumerate(joints):
            values[index] = float(value)
        output = PyKDL.Jacobian(7)
        if self.jacobian_solver.JntToJac(values, output) < 0:
            raise RuntimeError("KDL Jacobian calculation failed")
        return np.asarray(
            [[output[row, column] for column in range(7)] for row in range(6)],
            dtype=float,
        )

    def _publish_commands(self, now: float) -> None:
        if getattr(self, "managed", False) and (not self.managed_enabled or self.managed_fault):
            return
        if self.command_joints is not None:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "left_fr3_link0"
            msg.name = self.raw_gello_names or [f"fr3_joint{index}" for index in range(1, 8)]
            msg.position = self.command_joints.tolist()
            self.joint_publisher.publish(msg)
        gripper_fresh = self.measured_gripper is not None and now - self.measured_gripper_time <= self.source_timeout
        if self.mode != Mode.FINE:
            gripper_fresh = gripper_fresh and now - self.raw_gripper_time <= self.source_timeout
        if self.command_gripper is not None and gripper_fresh:
            msg = Float32()
            msg.data = float(self.command_gripper)
            self.gripper_publisher.publish(msg)
            self.gripper_action_command_publisher.publish(msg)
            if self.mode == Mode.GELLO:
                self.gripper_drive_publisher.publish(msg)


def _self_test() -> None:
    assert apply_deadzone(0.037, 0.08) == 0.0
    assert math.isclose(apply_deadzone(1.0, 0.08), 1.0)
    assert normalize_trigger(1.0) == 0.0
    assert normalize_trigger(-1.0) == 1.0
    assert width_percent_to_joint_position(1.0, 0.0, 0.8) == 0.0
    assert width_percent_to_joint_position(0.0, 0.0, 0.8) == 0.8
    assert joint_position_to_width_percent(0.4, 0.0, 0.8) == 0.5
    current = np.asarray([0.0, 1.0])
    target = np.asarray([1.0, 0.0])
    assert np.allclose(move_towards(current, target, 0.1), [0.1, 0.9])
    assert joint7_step(0.0, 1.0, 0.0, 0.3, 1.0, 0.1, -1.0, 1.0) > 0.0
    assert joint7_step(0.0, 0.0, 1.0, 0.3, 1.0, 0.1, -1.0, 1.0) < 0.0
    assert joint7_step(0.99, 1.0, 0.0, 0.3, 1.0, 0.1, -1.0, 1.0) == 1.0

    fake = object.__new__(FineTeleopNode)
    fake.gripper_speed = 0.4
    fake.gripper_direction = 0
    speeds: list[float] = []
    goals: list[float] = []
    fake._publish_gripper_speed = speeds.append
    fake._send_gripper_goal = goals.append
    fake._current_gripper_width = lambda: 0.55
    FineTeleopNode._update_fine_gripper(fake, -0.5)
    FineTeleopNode._update_fine_gripper(fake, -0.75)
    FineTeleopNode._update_fine_gripper(fake, 0.0)
    assert np.allclose(speeds, [0.2, 0.3, 0.0])
    assert goals == [0.0, 0.55]
    print("fine_teleop self-test passed")


def main() -> None:
    raise RuntimeError("Direct fine teleop ownership removed; use vla.cli.ros_session --source gello")

def _retired_main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--self-test", action="store_true")
    known, remaining = parser.parse_known_args()
    if known.self_test:
        _self_test()
        return

    rclpy.init(args=remaining, signal_handler_options=SignalHandlerOptions.NO)
    node: FineTeleopNode | None = None
    try:
        node = FineTeleopNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.prepare_shutdown()
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
