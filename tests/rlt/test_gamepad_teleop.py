from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import os
import socket
import subprocess
import sys
import threading

import numpy as np
import pytest
import yaml

from vla.datacollection.start_teleop.gamepad_config import load_config
from vla.datacollection.start_teleop.gamepad_teleop import (
    DEFAULT_CONFIG, AsyncReporter, GamepadTeleop, JoySidecars, ros_environment,
    upright_tcp_rotation,
    reset_tcp_heading,
)
from vla.datacollection.franka_robotiq_rollout.joy_transport import JoyPacket, JoyReceiver, encode_joy_packet
from vla.datacollection.franka_robotiq_rollout.gamepad import decode_cartesian_gamepad_values
from vla.robots.franka_robotiq.gripper import RobotiqModbus, parse_status_registers
from vla.robots.franka_robotiq.resource_lock import HardwareLease
from vla.robots.franka_robotiq.robot import FrankaRobotiqRobot, STATE_FIELDS
from tests.rlt.test_franka_pi05_robot import FakeArm, FakeGripper, FakeModbusClient


class Receiver:
    def __init__(self, clock):
        self.clock = clock
        self.error = None
        self.packet = JoyPacket(0, int(clock[0] * 1e9), (0, 0, 1, 0, 0, 1), (0,) * 11)

    def require_fresh(self, timeout, *, now_ns=None):
        if self.error:
            raise self.error
        if now_ns is None:
            now_ns = int(self.clock[0] * 1e9)
        age = (now_ns - self.packet.monotonic_ns) / 1e9
        if age < 0 or age > timeout:
            raise TimeoutError("stale Joy")
        return self.packet


@pytest.fixture
def rig(tmp_path, monkeypatch):
    config = load_config(DEFAULT_CONFIG)
    config = replace(config, robot=replace(config.robot, realtime_cpu=None, calibration_dir=tmp_path))
    monkeypatch.setattr("vla.robots.franka_robotiq.robot.RealSenseCapture", lambda *a, **k: pytest.fail("camera constructed"))
    arm, gripper = FakeArm("fake"), FakeGripper()
    arm.state.O_T_EE[:3, 3] = [.4, -.2, .3]
    arm.state.time = 10.
    arm.state.robot_mode = 1
    arm.state.control_command_success_rate = 1.
    clock = [10.]
    receiver = Receiver(clock)
    robot = FrankaRobotiqRobot(config.robot, arm_factory=lambda _: arm, gripper=gripper)
    robot.connect()
    reports = []
    app = GamepadTeleop(config, robot, receiver, clock=lambda: clock[0], report=reports.append)

    def advance(dt=.01, *, refresh=True, state=True):
        clock[0] += dt
        if refresh:
            receiver.packet = replace(receiver.packet, monotonic_ns=int(clock[0] * 1e9))
        if state:
            arm.state.time = clock[0]
        app.tick()

    app.tick()
    result = SimpleNamespace(config=config, arm=arm, gripper=gripper, clock=clock,
                             receiver=receiver, robot=robot, app=app, advance=advance, reports=reports)
    try:
        yield result
    finally:
        arm.motion_error = None
        robot.disconnect()


def enable(rig):
    rig.advance()
    rig.app.enable()
    assert rig.app.mode == "active"


def test_camera_free_startup_does_not_move_or_open(rig):
    assert rig.app.mode == "standby"
    assert not rig.arm.moves and rig.gripper.target is None
    assert tuple(rig.robot.observation_features) == STATE_FIELDS
    for method in (rig.robot.get_observation, rig.robot.get_rollout_snapshot):
        with pytest.raises(RuntimeError, match="disabled"):
            method()
    rig.app.enable()
    assert rig.app.mode == "standby"  # A state timestamp must actually advance first.


@pytest.mark.parametrize("axes,buttons", [
    ((.5, 0, 1, 0, 0, 1), (0,) * 11),
    ((0, 0, -1, 0, 0, -1), (0,) * 11),
    ((0, 0, 1, 0, 0, 1), (1, 0, 0, 1) + (0,) * 7),
])
def test_held_or_cancelling_inputs_cannot_arm(rig, axes, buttons):
    rig.receiver.packet = replace(rig.receiver.packet, axes=axes, buttons=buttons)
    rig.advance()
    rig.app.enable()
    assert rig.app.mode == "standby" and not rig.arm.moves


def test_combined_command_and_reseed_after_pause(rig):
    enable(rig)
    rig.receiver.packet = replace(rig.receiver.packet, axes=(1, 1, -1, -1, 1, 1), buttons=(0, 0, 1, 1) + (0,) * 7)
    rig.advance()
    assert len(rig.arm.moves) == 1
    debug = rig.robot.command_debug_snapshot()
    assert not debug["joint7_active"]
    assert np.linalg.norm(debug["commanded_linear_velocity"]) > 0
    assert np.linalg.norm(debug["commanded_angular_velocity"]) > 0
    assert rig.gripper.target == 1.
    rig.app.pause()
    assert rig.gripper.target == .75 and rig.robot._human_target_position is None
    rig.arm.state.O_T_EE[:3, 3] = [.45, -.1, .35]
    rig.receiver.packet = replace(rig.receiver.packet, axes=(0, 0, 1, 0, 0, 1), buttons=(0,) * 11)
    enable(rig)
    np.testing.assert_allclose(rig.robot._human_target_position, [.45, -.1, .35])


def test_rotation_only_keeps_xyz(rig):
    enable(rig)
    before = rig.robot._human_target_position.copy()
    rig.receiver.packet = replace(rig.receiver.packet, axes=(0, 0, 1, -.5, .5, 1))
    rig.advance()
    np.testing.assert_array_equal(rig.robot._human_target_position, before)
    np.testing.assert_allclose(rig.robot.command_debug_snapshot()["commanded_linear_velocity"], 0)
    assert len(rig.arm.moves) == 1


@pytest.mark.parametrize("axes,expected", [
    ((0, 0, 1, 1, 0, 1), (-1, -1)),
    ((0, 0, 1, 0, 1, 1), (1, -1)),
])
def test_tilt_mapping_rotates_clockwise_45_degrees(rig, axes, expected):
    assert rig.config.mapping.right_stick_rotation_deg == -45.
    command = decode_cartesian_gamepad_values(axes, (0,) * 11, rig.config.mapping, rig.config.control)
    np.testing.assert_allclose(command.angular_velocity_tool[:2], np.array(expected) * .15 / np.sqrt(2))
    assert command.angular_velocity_tool[2] == 0.
    np.testing.assert_array_equal(command.linear_velocity_base, np.zeros(3))
    assert command.gripper_motion == 0.


@pytest.mark.parametrize("axes,buttons", [
    ((0, 0, 1, .5, .5, 1), (0,) * 11),
    ((0, 0, 1, 0, 0, 1), (0, 0, 1, 0) + (0,) * 7),
    ((0, 0, 1, 0, 0, 1), (0, 1, 0, 0) + (0,) * 7),
])
def test_angular_release_does_not_chase_an_accumulated_target(rig, axes, buttons):
    enable(rig)
    rig.receiver.packet = replace(rig.receiver.packet, axes=axes, buttons=buttons)
    for _ in range(20):
        rig.advance()
    assert np.linalg.norm(rig.robot.command_debug_snapshot()["commanded_angular_velocity"]) > 0
    rig.receiver.packet = replace(rig.receiver.packet, axes=(0, 0, 1, 0, 0, 1), buttons=(0,) * 11)
    for _ in range(5):
        rig.advance()
        debug = rig.robot.command_debug_snapshot()
        np.testing.assert_array_equal(debug["commanded_angular_velocity"], np.zeros(3))
        np.testing.assert_array_equal(rig.robot._human_target_rotation, rig.arm.state.O_T_EE[:3, :3])
        assert debug["human_angular_velocity_control"]
    assert rig.app.mode == "active"
    assert all(type(motion).__name__ == "CartesianVelocityMotion" for motion, _ in rig.arm.moves)


def test_angular_partial_release_uses_only_current_tool_input(rig):
    enable(rig)
    rig.arm.state.O_T_EE[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    rig.receiver.packet = replace(rig.receiver.packet, axes=(0, 0, 1, .5, .5, 1), buttons=(0, 0, 1, 0) + (0,) * 7)
    for _ in range(10):
        rig.advance()
    # Release roll, retaining tilt. Neither roll error nor the earlier tool
    # orientation may leak into the newly requested base-frame angular velocity.
    rig.receiver.packet = replace(rig.receiver.packet, buttons=(0,) * 11)
    rig.advance()
    decoded = decode_cartesian_gamepad_values(
        rig.receiver.packet.axes, rig.receiver.packet.buttons, rig.config.mapping, rig.config.control,
    )
    expected = rig.arm.state.O_T_EE[:3, :3] @ decoded.angular_velocity_tool
    np.testing.assert_allclose(rig.robot.command_debug_snapshot()["commanded_angular_velocity"], expected)
    assert expected[2] == 0.
    # Release tilt, retaining roll: the 45-degree calibration does not affect X/B.
    rig.receiver.packet = replace(rig.receiver.packet, axes=(0, 0, 1, 0, 0, 1), buttons=(0, 0, 1, 0) + (0,) * 7)
    rig.advance()
    np.testing.assert_allclose(rig.robot.command_debug_snapshot()["commanded_angular_velocity"], [0, 0, -.3])


@pytest.mark.parametrize("button,direction", [(3, 1.), (0, -1.)])
def test_vertical_velocity_ignores_old_target_and_follows_speed_ramp(rig, button, direction):
    enable(rig)
    assert rig.config.robot.human_max_vertical_velocity is None
    rig.robot._human_target_position[2] += .1 * direction
    buttons = [0] * 11
    buttons[button] = 1
    rig.receiver.packet = replace(rig.receiver.packet, buttons=tuple(buttons))
    speed_button(rig)
    for _ in range(35):
        rig.advance()
        velocity = rig.robot.command_debug_snapshot()["commanded_linear_velocity"]
        assert velocity[2] == pytest.approx(direction * .01 * rig.app.speed_scale)
    assert rig.app.speed_scale == 5.
    speed_button(rig, False)
    speed_button(rig)
    for _ in range(35):
        rig.advance()
        velocity = rig.robot.command_debug_snapshot()["commanded_linear_velocity"]
        assert velocity[2] == pytest.approx(direction * .01 * rig.app.speed_scale)
    assert rig.app.speed_scale == 1.


@pytest.mark.parametrize("high", [False, True])
def test_all_axis_release_has_no_target_catchup_even_during_ramp(rig, high):
    enable(rig)
    rig.receiver.packet = replace(rig.receiver.packet, axes=(1, 1, -1, -1, 1, 1),
                                  buttons=(0, 0, 1, 1) + (0,) * 7)
    if high:
        speed_button(rig)
    for _ in range(15):
        rig.advance()
    rig.receiver.packet = replace(rig.receiver.packet, axes=(0, 0, 1, 0, 0, 1), buttons=(0,) * 11)
    for _ in range(5):
        # Simulate deceleration/drift: old positions must not cause catch-up.
        rig.arm.state.O_T_EE[:3, 3] += .0001
        rig.advance()
        debug = rig.robot.command_debug_snapshot()
        np.testing.assert_array_equal(debug["commanded_linear_velocity"], np.zeros(3))
        np.testing.assert_array_equal(debug["commanded_angular_velocity"], np.zeros(3))
        np.testing.assert_array_equal(rig.robot._human_target_position, rig.arm.state.O_T_EE[:3, 3])
        assert rig.gripper.target == rig.gripper.position()


def test_partial_linear_release_and_reversal_ignore_old_targets(rig):
    enable(rig)
    rig.receiver.packet = replace(rig.receiver.packet, axes=(1, 1, 1, 0, 0, 1),
                                  buttons=(0, 0, 0, 1) + (0,) * 7)
    for _ in range(20):
        rig.advance()
    rig.receiver.packet = replace(rig.receiver.packet, axes=(-1, 0, 1, 0, 0, 1), buttons=(0,) * 11)
    rig.advance()
    np.testing.assert_allclose(rig.robot.command_debug_snapshot()["commanded_linear_velocity"], [0, -.02, 0])


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sign", [-1, 1])
def test_direct_linear_workspace_guard(rig, axis, sign):
    enable(rig)
    boundary = rig.config.robot.workspace_max if sign > 0 else rig.config.robot.workspace_min
    rig.arm.state.O_T_EE[axis, 3] = boundary[axis] - sign * .00001
    velocity = np.zeros(3)
    velocity[axis] = sign * .02
    limits = rig.config.control.motion_limits()
    rig.robot.send_human_command(velocity, 0., 0., .01, angular_velocity_tool=np.zeros(3), **limits)
    assert rig.robot._human_target_position[axis] == pytest.approx(boundary[axis])
    assert rig.robot.command_debug_snapshot()["commanded_linear_velocity"][axis] == pytest.approx(sign * .001)
    # Outside the workspace: neutral remains zero; outward is blocked;
    # deliberate inward input remains available.
    rig.arm.state.O_T_EE[axis, 3] = boundary[axis] + sign * .001
    for requested, expected in [(velocity, 0.), (np.zeros(3), 0.), (-velocity, -sign * .02)]:
        rig.robot.send_human_command(requested, 0., 0., .01, angular_velocity_tool=np.zeros(3), **limits)
        assert rig.robot.command_debug_snapshot()["commanded_linear_velocity"][axis] == pytest.approx(expected)


def test_legacy_linear_tracking_is_unchanged(rig):
    rig.robot.config = replace(rig.robot.config, human_linear_velocity_control=False)
    enable(rig)
    rig.receiver.packet = replace(rig.receiver.packet, axes=(1, 0, 1, 0, 0, 1))
    rig.advance()
    rig.receiver.packet = replace(rig.receiver.packet, axes=(0, 0, 1, 0, 0, 1))
    rig.advance()
    assert rig.robot.command_debug_snapshot()["commanded_linear_velocity"][1] > 0


def test_legacy_orientation_tracking_remains_opt_in_compatible(rig):
    # Other collectors keep their original target-tracking semantics by default.
    rig.robot.config = replace(rig.robot.config, human_angular_velocity_control=False)
    enable(rig)
    rig.receiver.packet = replace(rig.receiver.packet, axes=(0, 0, 1, .5, 0, 1))
    for _ in range(10):
        rig.advance()
    rig.receiver.packet = replace(rig.receiver.packet, axes=(0, 0, 1, 0, 0, 1))
    rig.advance()
    assert np.linalg.norm(rig.robot.command_debug_snapshot()["commanded_angular_velocity"]) > 0
    assert not rig.robot.command_debug_snapshot()["human_angular_velocity_control"]


def speed_button(rig, pressed=True):
    buttons = list(rig.receiver.packet.buttons)
    buttons[rig.config.speed_modes.toggle_button] = int(pressed)
    rig.receiver.packet = replace(rig.receiver.packet, buttons=tuple(buttons))
    rig.advance()


def pose_button(rig, name, pressed=True):
    buttons = list(rig.receiver.packet.buttons)
    buttons[getattr(rig.config.pose_shortcuts, f"{name}_button")] = int(pressed)
    rig.receiver.packet = replace(rig.receiver.packet, buttons=tuple(buttons))
    rig.advance()


@pytest.mark.parametrize("current,reference", [(40, 0), (-40, 0), (179, -179), (-179, 179)])
def test_reset_heading_preserves_tilt_and_uses_short_rotation(current, reference):
    from scipy.spatial.transform import Rotation
    tilt = Rotation.from_euler("x", 155, degrees=True).as_matrix()
    rz = lambda degrees: Rotation.from_euler("z", degrees, degrees=True).as_matrix()
    rotation = rz(current) @ tilt
    target = reset_tcp_heading(rotation, rz(reference) @ tilt)
    np.testing.assert_allclose(target, rz(reference) @ tilt, atol=1e-12)
    np.testing.assert_allclose(target[2], rotation[2], atol=1e-12)
    delta = Rotation.from_matrix(target @ rotation.T).as_rotvec()
    np.testing.assert_allclose(delta[:2], 0, atol=1e-12)
    assert np.degrees(np.linalg.norm(delta)) == pytest.approx(abs((reference-current+180) % 360-180))


def test_yaw_shortcut_fixed_xyz_edge_trigger_and_manual_cancel(rig):
    from scipy.spatial.transform import Rotation
    rig.config = replace(rig.config, pose_shortcuts=replace(rig.config.pose_shortcuts, yaw_reset_button=10))
    rig.app.config = rig.config
    rig.arm.state.O_T_EE[:3, :3] = Rotation.from_euler("x", 180, degrees=True).as_matrix()
    enable(rig)
    rig.arm.state.O_T_EE[:3, :3] = Rotation.from_euler("zx", [35, 180], degrees=True).as_matrix()
    rig.arm.state.O_T_EE[:3, 3] = [.45, -.1, .35]
    pose_button(rig, "yaw_reset")
    target = rig.app._pose_return
    assert target is not None and "heading" in target.label
    np.testing.assert_allclose(target.position, [.45, -.1, .35])
    np.testing.assert_allclose(rig.robot.command_debug_snapshot()["commanded_linear_velocity"], 0)
    rig.advance()
    assert rig.app._pose_return is target
    rig.receiver.packet = replace(rig.receiver.packet, axes=(1, 0, 1, 0, 0, 1))
    rig.advance()
    assert rig.app._pose_return is None


def test_pose_shortcut_mapping_and_upright_rotation(rig):
    assert rig.config.pose_shortcuts.view_button == 6
    assert rig.config.pose_shortcuts.menu_button == 7
    angle = np.radians(35)
    rotation = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [np.sin(angle), 0, -np.cos(angle)],
        [0, 1, 0],
    ])
    upright = upright_tcp_rotation(rotation)
    np.testing.assert_allclose(upright.T @ upright, np.eye(3), atol=1e-12)
    assert np.linalg.det(upright) == pytest.approx(1)
    np.testing.assert_allclose(upright[:, 2], [0, 0, -1])
    np.testing.assert_allclose(upright[:2, 0], rotation[:2, 0] / np.linalg.norm(rotation[:2, 0]))


def test_menu_returns_upright_at_press_xyz_and_is_edge_triggered(rig):
    enable(rig)
    pressed_at = rig.arm.state.O_T_EE[:3, 3].copy()
    pose_button(rig, "menu")
    target = rig.app._pose_return
    assert target is not None and target.label == "upright"
    np.testing.assert_array_equal(target.position, pressed_at)
    np.testing.assert_allclose(target.rotation[:, 2], [0, 0, -1])
    np.testing.assert_array_equal(rig.robot.command_debug_snapshot()["commanded_linear_velocity"], np.zeros(3))
    assert np.linalg.norm(rig.robot.command_debug_snapshot()["commanded_angular_velocity"]) > 0

    rig.arm.state.O_T_EE[:3, 3] += [.02, 0, 0]
    pose_button(rig, "menu")  # Held: target is not recaptured.
    np.testing.assert_array_equal(rig.app._pose_return.position, pressed_at)
    assert rig.robot.command_debug_snapshot()["commanded_linear_velocity"][0] < 0
    pose_button(rig, "menu", False)
    pose_button(rig, "menu")
    np.testing.assert_array_equal(rig.app._pose_return.position, rig.arm.state.O_T_EE[:3, 3])


@pytest.mark.parametrize("roll_degrees", [-65, 65])
@pytest.mark.parametrize("tilt_degrees", [0, 20])
def test_menu_restores_initial_heading_after_tool_roll_and_tilt(rig, roll_degrees, tilt_degrees):
    from scipy.spatial.transform import Rotation

    initial = Rotation.from_euler("xyz", [180, 0, 35], degrees=True).as_matrix()
    rig.arm.state.O_T_EE[:3, :3] = initial
    enable(rig)
    # X/B rotate about tool Z; the right stick may additionally tilt the tool.
    rig.arm.state.O_T_EE[:3, :3] = initial @ Rotation.from_euler(
        "zx", [roll_degrees, tilt_degrees], degrees=True,
    ).as_matrix()
    rig.arm.state.O_T_EE[:3, 3] = [.45, -.1, .35]
    pose_button(rig, "menu")

    target = rig.app._pose_return
    assert target is not None
    np.testing.assert_allclose(target.rotation, initial, atol=1e-12)
    np.testing.assert_allclose(target.position, [.45, -.1, .35])
    debug = rig.robot.command_debug_snapshot()
    np.testing.assert_allclose(debug["commanded_linear_velocity"], 0)
    assert np.linalg.norm(debug["commanded_angular_velocity"]) > 0


def test_view_returns_to_position_captured_at_each_enable(rig):
    initial = rig.arm.state.O_T_EE[:3, 3].copy()
    enable(rig)
    rig.arm.state.O_T_EE[:3, 3] = [.55, -.1, .4]
    pose_button(rig, "view")
    target = rig.app._pose_return
    assert target is not None and target.label == "initial pose"
    np.testing.assert_array_equal(target.position, initial)
    np.testing.assert_allclose(target.rotation[:, 2], [0, 0, -1])
    assert rig.robot.command_debug_snapshot()["commanded_linear_velocity"][0] < 0

    rig.app.pause()
    rig.receiver.packet = replace(rig.receiver.packet, buttons=(0,) * 11)
    rig.arm.state.O_T_EE[:3, 3] = [.45, -.15, .35]
    enable(rig)
    np.testing.assert_array_equal(rig.app._initial_position, [.45, -.15, .35])


def test_view_uses_high_cap_and_slows_near_target_while_menu_stays_low(rig):
    enable(rig)
    initial = rig.arm.state.O_T_EE[:3, 3].copy()
    rig.arm.state.O_T_EE[0, 3] += .1
    pose_button(rig, "view")
    debug = rig.robot.command_debug_snapshot()
    np.testing.assert_allclose(debug["commanded_linear_velocity"], [-.16, 0, 0])
    assert np.linalg.norm(debug["commanded_angular_velocity"]) <= .3 + 1e-12
    assert not rig.app.high_speed  # Shortcut does not change the manual gear.
    rig.arm.state.O_T_EE[:3, 3] = initial + [.01, 0, 0]
    rig.advance()
    np.testing.assert_allclose(rig.robot.command_debug_snapshot()["commanded_linear_velocity"], [-.005, 0, 0])
    pose_button(rig, "menu")
    rig.arm.state.O_T_EE[0, 3] += .1
    rig.advance()
    np.testing.assert_allclose(rig.robot.command_debug_snapshot()["commanded_linear_velocity"], [-.04, 0, 0])


def test_pose_return_completes_with_zero_command_and_manual_input_cancels(rig):
    enable(rig)
    pose_button(rig, "menu")
    target = rig.app._pose_return
    pose_button(rig, "menu", False)
    rig.arm.state.O_T_EE[:3, 3] = target.position
    rig.arm.state.O_T_EE[:3, :3] = target.rotation
    rig.advance()
    assert rig.app._pose_return is None
    np.testing.assert_array_equal(rig.robot.command_debug_snapshot()["commanded_linear_velocity"], np.zeros(3))
    np.testing.assert_array_equal(rig.robot.command_debug_snapshot()["commanded_angular_velocity"], np.zeros(3))
    assert any("Reached upright" in message for message in rig.reports)

    rig.arm.state.O_T_EE[:3, 3] += [.05, 0, 0]
    pose_button(rig, "view")
    assert rig.app._pose_return is not None
    pose_button(rig, "view", False)
    rig.receiver.packet = replace(rig.receiver.packet, axes=(1, 0, 1, 0, 0, 1))
    rig.advance()
    assert rig.app._pose_return is None
    assert rig.robot.command_debug_snapshot()["commanded_linear_velocity"][1] > 0
    assert any("Cancelled initial pose return" in message for message in rig.reports)


@pytest.mark.parametrize("name", ["view", "menu"])
def test_return_brakes_continuously_and_uses_submillimetre_tolerance(rig, name):
    enable(rig)
    pose_button(rig, name)
    target = rig.app._pose_return
    cap = .16 if name == "view" else .04
    direction = np.array([1., 2., -2.]) / 3
    speeds = []
    distances = [.10, .080001, .08, .079999, .04, .010001, .01, .009999, .003, .001, .00051]
    for distance in distances:
        rig.arm.state.O_T_EE[:3, 3] = target.position + distance * direction
        rig.arm.state.O_T_EE[:3, :3] = target.rotation
        rig.advance()
        assert rig.app._pose_return is target
        velocity = np.asarray(rig.robot.command_debug_snapshot()["commanded_linear_velocity"])
        speed = np.linalg.norm(velocity)
        speeds.append(speed)
        assert speed <= cap + 1e-12
        np.testing.assert_allclose(velocity / speed, -direction, atol=1e-10)
    assert np.all(np.diff(speeds) <= 1e-12)
    assert max(abs(speeds[i] - speeds[j]) for i, j in [(1, 2), (2, 3), (5, 6), (6, 7)]) < 3e-6
    assert speeds[6] == pytest.approx(.005)
    assert speeds[-2] == pytest.approx(.0005)
    rig.arm.state.O_T_EE[:3, 3] = target.position + .00049 * direction
    rig.advance()
    assert rig.app._pose_return is None
    np.testing.assert_allclose(rig.robot.command_debug_snapshot()["commanded_linear_velocity"], 0)
    assert any("position_error=0.490 mm" in message for message in rig.reports)


@pytest.mark.parametrize("name", ["view", "menu"])
def test_return_converges_with_simulated_actuator_lag(rig, name):
    enable(rig)
    pose_button(rig, name)
    target = rig.app._pose_return
    rig.arm.state.O_T_EE[:3, :3] = target.rotation
    direction = np.array([1., -1., 1.]) / np.sqrt(3)
    rig.arm.state.O_T_EE[:3, 3] = target.position + .12 * direction
    velocity = np.zeros(3)
    # 150 ms first-order velocity response, including residual braking after
    # Reached. This is a synthetic plant, not a measured model of Franka.
    for _ in range(2800):
        rig.advance()
        requested = np.asarray(rig.robot.command_debug_snapshot()["commanded_linear_velocity"])
        velocity += (1 - np.exp(-.01 / .15)) * (requested - velocity)
        rig.arm.state.O_T_EE[:3, 3] += velocity * .01
        assert rig.app.mode == "active", rig.app.reason
        if rig.app._pose_return is None:
            break
    assert rig.app._pose_return is None
    for _ in range(100):
        rig.advance()
        velocity *= np.exp(-.01 / .15)
        rig.arm.state.O_T_EE[:3, 3] += velocity * .01
    assert np.linalg.norm(rig.arm.state.O_T_EE[:3, 3] - target.position) <= .0005


def test_pose_shortcut_commands_reach_recording_callback_and_view_wins(rig):
    commands = []
    rig.app.on_command = commands.append
    enable(rig)
    buttons = list(rig.receiver.packet.buttons)
    buttons[rig.config.pose_shortcuts.view_button] = 1
    buttons[rig.config.pose_shortcuts.menu_button] = 1
    rig.receiver.packet = replace(rig.receiver.packet, buttons=tuple(buttons))
    rig.advance()
    assert rig.app._pose_return.label == "initial pose"
    assert len(commands) == 1
    assert commands[0]["motion_command_sent"]
    assert np.linalg.norm(commands[0]["commanded_angular_velocity"]) > 0
    assert any("View and Menu pressed together" in message for message in rig.reports)


def test_pose_return_timeout_faults_and_pause_clears_target(rig):
    enable(rig)
    pose_button(rig, "view")
    rig.app._pose_return.deadline = rig.clock[0] - .001
    rig.advance()
    assert rig.app.mode == "fault" and "return timed out" in rig.app.reason
    assert rig.app._pose_return is None


@pytest.mark.parametrize("axes,buttons", [
    ((0, 0, 1, 0, 0, 1), (0, 1, 1, 0) + (0,) * 7),
    ((0, 0, 1, 0, 0, 1), (1, 0, 0, 1) + (0,) * 7),
    ((0, 0, -1, 0, 0, -1), (0,) * 11),
])
def test_opposing_manual_inputs_cancel_return_and_block_new_shortcut(rig, axes, buttons):
    enable(rig)
    pose_button(rig, "menu")
    assert rig.app._pose_return is not None
    rig.receiver.packet = replace(rig.receiver.packet, axes=axes, buttons=buttons)
    rig.advance()
    assert rig.app._pose_return is None
    debug = rig.robot.command_debug_snapshot()
    np.testing.assert_allclose(debug["commanded_linear_velocity"], 0)
    np.testing.assert_allclose(debug["commanded_angular_velocity"], 0)
    pose_button(rig, "menu")
    assert rig.app._pose_return is None
    assert any("release motion" in message for message in rig.reports)


@pytest.mark.parametrize("field", ["human_linear_velocity_control", "human_angular_velocity_control"])
def test_pose_shortcuts_refuse_legacy_target_tracking(rig, field):
    config = replace(rig.config, robot=replace(rig.config.robot, **{field: False}))
    rig.app.config, rig.robot.config = config, config.robot
    enable(rig)
    pose_button(rig, "menu")
    assert rig.app.mode == "active" and rig.app._pose_return is None
    assert any("direct linear and angular" in message for message in rig.reports)


def test_enable_captures_post_stop_pose_and_latches_initialization_errors(rig, monkeypatch):
    begin = rig.robot.begin_human_control

    def settle():
        begin()
        rig.arm.state.O_T_EE[0, 3] = .46

    monkeypatch.setattr(rig.robot, "begin_human_control", settle)
    enable(rig)
    assert rig.app._initial_position[0] == pytest.approx(.46)
    rig.app.pause()
    rig.advance()

    def fail(_):
        raise ValueError("invalid upright target")

    monkeypatch.setattr("vla.datacollection.start_teleop.gamepad_teleop.upright_tcp_rotation", fail)
    rig.app.enable()
    assert rig.app.mode == "fault"
    assert "control initialization failed" in rig.app.reason
    assert rig.robot._human_target_position is None


def test_menu_converges_in_fake_velocity_plant_and_holds_xyz(rig):
    from vla.robots.franka_robotiq.robot import _axis_angle_matrix

    enable(rig)
    initial = rig.arm.state.O_T_EE[:3, 3].copy()
    pose_button(rig, "menu")
    for _ in range(1400):
        debug = rig.robot.command_debug_snapshot()
        angular = np.asarray(debug["commanded_angular_velocity"])
        speed = np.linalg.norm(angular)
        assert speed <= .3 + 1e-12
        if speed:
            rig.arm.state.O_T_EE[:3, :3] = (
                _axis_angle_matrix(angular / speed, speed * .01)
                @ rig.arm.state.O_T_EE[:3, :3]
            )
        rig.arm.state.O_T_EE[:3, 3] += np.asarray(debug["commanded_linear_velocity"]) * .01
        rig.advance()
        if rig.app._pose_return is None:
            break
    assert rig.app.mode == "active" and rig.app._pose_return is None
    np.testing.assert_allclose(rig.arm.state.O_T_EE[:3, 3], initial, atol=1e-12)
    assert np.dot(rig.arm.state.O_T_EE[:3, 2], [0, 0, -1]) >= np.cos(np.radians(2))


def test_speed_toggle_is_edge_triggered_and_uses_both_profiles(rig):
    enable(rig)
    assert not rig.app.high_speed
    speed_button(rig)
    assert rig.app.high_speed
    assert rig.app.speed_scale == pytest.approx(1 + 4 / 30)
    for _ in range(29):
        rig.advance()
    assert rig.app.speed_scale == 5.
    assert rig.app.control.gamepad_xy_speed == .10
    assert rig.app.control.gamepad_z_speed == .05
    assert rig.app.control.gamepad_tilt_speed == .15
    assert rig.app.control.gamepad_roll_speed == .30
    assert rig.app.control.max_linear_speed == .16
    assert rig.app.control.gamepad_max_angular_speed == .30
    for _ in range(5):
        rig.advance()
    assert rig.app.high_speed
    assert sum("Speed=HIGH" in message for message in rig.reports) == 1
    speed_button(rig, False)
    speed_button(rig)
    assert not rig.app.high_speed
    assert rig.app.speed_scale == pytest.approx(5 - 4 / 30)
    for _ in range(29):
        rig.advance()
    assert rig.app.control is rig.config.control


@pytest.mark.parametrize("axes,buttons", [
    ((1, 0, 1, 0, 0, 1), (0,) * 11),
    ((0, 0, -1, 0, 0, -1), (0,) * 11),
    ((0, 0, 1, 0, 0, 1), (1, 0, 0, 1) + (0,) * 7),
])
def test_speed_change_works_with_held_inputs_and_needs_a_new_press(rig, axes, buttons):
    enable(rig)
    rig.receiver.packet = replace(rig.receiver.packet, axes=axes, buttons=buttons)
    speed_button(rig)
    assert rig.app.high_speed
    rig.receiver.packet = replace(rig.receiver.packet, axes=(0, 0, 1, 0, 0, 1), buttons=(0, 0, 0, 0, 1) + (0,) * 6)
    rig.advance()
    assert rig.app.high_speed  # Centering while LB remains held does not toggle.
    speed_button(rig, False)
    speed_button(rig)
    assert not rig.app.high_speed


def test_high_speed_command_is_combined_once_with_unchanged_gripper(rig):
    enable(rig)
    speed_button(rig)
    for _ in range(29):
        rig.advance()
    calls = []
    rig.robot.send_human_command = lambda *args, **kwargs: calls.append((args, kwargs))
    rig.receiver.packet = replace(rig.receiver.packet, axes=(1, 1, -1, -1, 1, 1), buttons=(0, 0, 1, 1) + (0,) * 7)
    rig.advance()
    assert len(calls) == 1
    args, limits = calls[0]
    np.testing.assert_allclose(args[0], [.10, .10, .05])
    assert args[1:3] == (0., 1.)
    assert np.linalg.norm(limits["angular_velocity_tool"]) == pytest.approx(.3)
    assert limits["max_angular_speed"] == .3
    assert limits["max_linear_speed"] == .16
    assert limits["gripper_speed_scale"] == .4
    assert rig.robot.config.dynamics_factor == .1


@pytest.mark.parametrize("failure", ["pause", "fault", "stale", "stall"])
def test_speed_returns_to_low_after_pause_or_fault(rig, failure):
    enable(rig)
    speed_button(rig)
    if failure == "pause":
        rig.app.pause()
    elif failure == "fault":
        rig.arm.has_errors = True
        rig.advance()
    elif failure == "stall":
        rig.advance(.11)
    else:
        rig.receiver.error = TimeoutError("stale Joy")
        rig.advance()
    assert not rig.app.high_speed
    assert rig.app.speed_scale == 1.
    assert rig.app.control is rig.config.control
    rig.advance()
    assert not rig.app.high_speed


def test_high_speed_respects_custom_robot_caps(rig):
    config = replace(rig.config, robot=replace(rig.config.robot, max_linear_velocity=.04, max_angular_velocity=.3))
    app = GamepadTeleop(config, rig.robot, rig.receiver, clock=lambda: rig.clock[0])
    app.high_speed = True
    for _ in range(30):
        app._advance_speed(.01)
        assert app.control.motion_limits()["max_linear_speed"] <= .04
        assert app.control.motion_limits()["max_angular_speed"] <= .3
    assert app.speed_scale == 5.
    assert app.control.motion_limits()["max_linear_speed"] == .04
    assert app.control.motion_limits()["max_angular_speed"] == .3


def test_missing_speed_button_pauses_control(rig):
    enable(rig)
    rig.receiver.packet = replace(rig.receiver.packet, buttons=(0,) * 4)
    rig.advance()
    assert rig.app.mode == "standby"


def test_ramp_keeps_commands_and_targets_continuous(rig, monkeypatch):
    enable(rig)
    position = rig.robot._human_target_position.copy()
    rig.receiver.packet = replace(rig.receiver.packet, axes=(1, 1, -1, -1, 1, 1), buttons=(0, 0, 1, 1) + (0,) * 7)
    captured = []
    send = rig.robot.send_human_command
    def capture(*args, **kwargs):
        captured.append((args, kwargs))
        result = send(*args, **kwargs)
        # Ideal measured tracking, rather than relying on target accumulation.
        rig.arm.state.O_T_EE[:3, 3] = rig.robot._human_target_position
        return result
    monkeypatch.setattr(rig.robot, "send_human_command", capture)
    monkeypatch.setattr(rig.robot, "begin_human_control", lambda: pytest.fail("ramp reseeded control"))
    monkeypatch.setattr(rig.robot, "end_human_control", lambda: pytest.fail("ramp stopped control"))
    # Include a release sample before requesting the reverse ramp.
    speed_button(rig)
    for _ in range(28):
        rig.advance()
    speed_button(rig, False)
    assert rig.app.speed_scale == 5.
    speed_button(rig)
    for _ in range(29):
        rig.advance()
    assert rig.app.speed_scale == 1.
    assert len(captured) == len(rig.arm.moves) == 60
    assert all(type(motion).__name__ == "CartesianVelocityMotion" for motion, _ in rig.arm.moves)
    values = np.array([args[0][0] / .02 for args, _ in captured])
    np.testing.assert_allclose(values[:30], np.linspace(1 + 4 / 30, 5, 30), atol=1e-12)
    np.testing.assert_allclose(values[30:], np.linspace(5 - 4 / 30, 1, 30), atol=1e-12)
    for args, limits in captured:
        scale = args[0][0] / .02
        assert np.linalg.norm(args[0]) > 0
        assert np.linalg.norm(limits["angular_velocity_tool"]) == pytest.approx(.3)
        assert limits["max_linear_speed"] == pytest.approx(min(.04 * scale, .16))
        assert limits["max_angular_speed"] == pytest.approx(.3)
        assert args[0][2] == pytest.approx(.01 * scale)
        assert args[2] == 1. and limits["gripper_speed_scale"] == .4
    effective = []
    for args, limits in captured:
        velocity = np.asarray(args[0], dtype=float)
        norm = np.linalg.norm(velocity)
        if norm > limits["max_linear_speed"]:
            velocity = velocity * limits["max_linear_speed"] / norm
        effective.append(velocity * args[3])
    delta = sum(effective, np.zeros(3))
    np.testing.assert_allclose(rig.robot._human_target_position, position + delta, atol=1e-12)


def test_ramp_reversal_starts_at_current_multiplier(rig):
    enable(rig)
    speed_button(rig)
    for _ in range(8):
        rig.advance()
    speed_button(rig, False)
    assert rig.app.speed_scale == pytest.approx(1 + 40 / 30)
    speed_button(rig)
    assert rig.app.speed_scale == pytest.approx(1 + 36 / 30)
    speed_button(rig, False)
    assert rig.app.speed_scale == pytest.approx(1 + 32 / 30)
    speed_button(rig)
    assert rig.app.speed_scale == pytest.approx(1 + 36 / 30)
    assert rig.app.mode == "active"


def test_ramp_uses_same_bounded_dt_as_pose_integration(rig):
    enable(rig)
    speed_button(rig)
    before = rig.app.speed_scale
    calls = []
    rig.robot.send_human_command = lambda *a, **k: calls.append((a, k))
    rig.advance(.09)
    assert rig.app.mode == "active"
    assert calls[0][0][3] == .02
    assert rig.app.speed_scale == pytest.approx(before + 4 * .02 / .3)


@pytest.mark.parametrize("axes,buttons", [
    ((float("nan"), 0, 1, 0, 0, 1), (0,) * 11),
    ((0, 0, 1), (0,) * 11),
    ((0, 0, 1, 0, 0, 1), (0, 0, 0, 0, 1.5) + (0,) * 6),
])
def test_invalid_packet_does_not_advance_speed(rig, axes, buttons):
    enable(rig)
    speed_button(rig)
    before = len(rig.arm.moves)
    rig.receiver.packet = replace(rig.receiver.packet, axes=axes, buttons=buttons)
    rig.advance()
    assert rig.app.mode == "standby" and rig.app.speed_scale == 1.
    assert len(rig.arm.moves) == before + 1  # Stop motion only.


def test_blocked_console_and_full_queue_do_not_stall_control(rig):
    enable(rig)
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    def output(message):
        entered.set()
        release.wait(timeout=5)
    reporter = AsyncReporter(output, capacity=2)
    rig.app.report = reporter
    reporter("block console")
    assert entered.wait(timeout=1)
    errors = []
    def control():
        try:
            for _ in range(20):
                speed_button(rig)
                speed_button(rig, False)
        except BaseException as error:
            errors.append(error)
        finally:
            completed.set()
    worker = threading.Thread(target=control)
    try:
        worker.start()
        assert completed.wait(timeout=2), "control waited for terminal output"
        assert not errors
        assert rig.app.mode == "active" and len(rig.arm.moves) == 40
        assert reporter._queue.qsize() <= 2
    finally:
        release.set()
        worker.join(timeout=3)
        reporter.close()
    assert not reporter._thread.is_alive()


def test_console_close_is_bounded_even_when_output_is_blocked():
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    def output(message):
        entered.set()
        release.wait(timeout=5)
    reporter = AsyncReporter(output)
    reporter("block")
    assert entered.wait(timeout=1)
    worker = threading.Thread(target=lambda: (reporter.close(), closed.set()))
    try:
        worker.start()
        assert closed.wait(timeout=1)
    finally:
        release.set()
        worker.join(timeout=2)
        reporter._thread.join(timeout=2)
    assert not reporter._thread.is_alive()


def test_old_config_defaults_to_point_three_second_ramp(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["speed_modes"].pop("transition_time_s")
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path).speed_modes.transition_time_s == .3


def test_old_pure_teleop_config_gets_default_pose_shortcuts(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw.pop("pose_shortcuts")
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    assert config.pose_shortcuts.view_button == 6
    assert config.pose_shortcuts.menu_button == 7


@pytest.mark.parametrize("error", [TimeoutError("Joy lost"), ValueError("invalid input")])
def test_input_failure_pauses_without_auto_resume(rig, error):
    enable(rig)
    rig.receiver.error = error
    rig.advance()
    assert rig.app.mode == "standby"
    rig.receiver.error = None
    rig.advance()
    assert rig.app.mode == "standby" and not rig.arm.moves
    rig.app.enable()
    assert rig.app.mode == "active"


@pytest.mark.parametrize("timestamp_offset_ns,expected_mode", [
    (0, "active"),
    (1_000_000, "standby"),
    (-210_000_000, "standby"),
])
def test_joy_age_is_measured_after_poll_and_still_rejects_bad_timestamps(
    rig, monkeypatch, timestamp_offset_ns, expected_mode,
):
    enable(rig)
    # Use the real freshness checker without opening a socket or real hardware.
    receiver = object.__new__(JoyReceiver)

    def poll():
        # A valid sample arrives 20 us after the control tick begins: before the
        # fix this produced exactly "Joy input is stale: age=-0.000s".
        rig.clock[0] += .00002
        return replace(
            rig.receiver.packet,
            monotonic_ns=int(rig.clock[0] * 1e9) + timestamp_offset_ns,
        )

    receiver.poll_latest = poll
    rig.app.receiver = receiver
    monkeypatch.setattr(
        "vla.datacollection.franka_robotiq_rollout.joy_transport.time.monotonic_ns",
        lambda: int(rig.clock[0] * 1e9),
    )
    rig.advance()
    assert rig.app.mode == expected_mode
    if expected_mode == "active":
        assert len(rig.arm.moves) == 1
        assert type(rig.arm.moves[0][0]).__name__ == "CartesianVelocityMotion"
    else:
        assert not rig.arm.moves
        assert not rig.app.high_speed and rig.app.speed_scale == 1.


def test_stall_drops_elapsed_motion(rig):
    enable(rig)
    rig.receiver.packet = replace(rig.receiver.packet, axes=(1, 0, 1, 0, 0, 1))
    rig.advance(.11)
    assert rig.app.mode == "standby" and not rig.arm.moves


def test_fault_is_manual_and_recovery_does_not_resume(rig):
    enable(rig)
    rig.arm.motion_error = RuntimeError("cartesian_reflex")
    rig.advance()
    assert rig.app.mode == "fault" and rig.arm.recover_calls == 0
    rig.app.enable()
    assert rig.app.mode == "fault"
    # Recovery requires a new Franky timestamp; the fake has no background state stream.
    rig.arm.state.time = None
    rig.app.recover()
    assert rig.arm.recover_calls == 1 and rig.app.mode == "fault"
    rig.arm.state.time = rig.clock[0]
    rig.robot.recover_motion_fault = lambda: None
    rig.app.recover()
    assert rig.app.mode == "standby"
    rig.advance()
    assert rig.app.mode == "standby"


def test_stale_state_and_bad_workspace_fault(rig):
    for _ in range(51):
        rig.advance(state=False)
    assert rig.app.mode == "fault"
    rig.app.mode = "standby"
    rig.arm.state.O_T_EE[0, 3] = 2.
    rig.advance()
    assert rig.app.mode == "fault" and not rig.arm.moves


def test_low_success_rate_latches_after_threshold(rig):
    enable(rig)
    rig.arm.state.control_command_success_rate = .5
    for _ in range(12):
        rig.advance()
    assert rig.app.mode == "active"
    for _ in range(3):
        rig.advance()
    assert rig.app.mode == "fault"


def test_startup_success_rate_window_does_not_false_fault(rig):
    enable(rig)
    rig.arm.state.control_command_success_rate = .28125
    for _ in range(10):
        rig.advance()
    assert rig.app.mode == "active" and rig.app._low_rates == 0
    assert any(not state["success_rate_monitoring"] for state in rig.app._rate_history)
    rig.arm.state.control_command_success_rate = 1.
    for _ in range(5):
        rig.advance()
    assert rig.app.mode == "active" and rig.app._low_rates == 0


@pytest.mark.parametrize("rate", [.9, np.nextafter(.9, -np.inf), float(np.float32(.9))])
def test_success_rate_boundary_does_not_false_fault(rig, rate):
    enable(rig)
    rig.arm.state.control_command_success_rate = rate
    for _ in range(16):
        rig.advance()
    assert rig.app.mode == "active" and rig.app._low_rates == 0


@pytest.mark.parametrize("rate", [.8999, .89, float(np.nextafter(np.float32(.9), np.float32(0)))])
def test_low_success_rate_counts_only_new_states_and_resets(rig, rate):
    enable(rig)
    for _ in range(12):
        rig.advance()
    rig.arm.state.control_command_success_rate = rate
    rig.advance()
    for _ in range(5):
        rig.app._read_state()
        rig.advance(state=False)
    assert rig.app.mode == "active" and rig.app._low_rates == 1
    rig.arm.state.control_command_success_rate = 1.
    rig.advance()
    assert rig.app._low_rates == 0
    rig.arm.state.control_command_success_rate = rate
    for _ in range(3):
        rig.advance()
    assert rig.app.mode == "fault"
    assert f"rate={rate:.17g}" in rig.app.reason
    assert "fresh_low_samples=3/3" in rig.app.reason
    detail = rig.app.fault_snapshot
    assert detail["consecutive_low_samples"] == 3
    assert [s["success_rate"] for s in detail["recent_states"]][-3:] == [rate] * 3
    rig.advance()
    assert rig.app.fault_snapshot == detail


def test_repeated_low_state_still_times_out(rig):
    enable(rig)
    for _ in range(12):
        rig.advance()
    rig.arm.state.control_command_success_rate = .5
    rig.advance()
    for _ in range(60):
        rig.advance(state=False)
    assert rig.app.mode == "fault" and "stale" in rig.app.reason


def test_resource_leases_cover_ip_and_serial_and_release_partial_acquisition(tmp_path):
    with HardwareLease("172.16.0.2", "/dev/fake-left", directory=tmp_path):
        for ip, port in (("172.16.0.2", "/dev/fake-right"), ("172.16.0.3", "/dev/fake-left")):
            with pytest.raises(RuntimeError, match="already owned"):
                HardwareLease(ip, port, directory=tmp_path)
        with HardwareLease("172.16.0.3", "/dev/fake-right", directory=tmp_path):
            pass
    with HardwareLease("172.16.0.2", "/dev/fake-left", directory=tmp_path):
        pass


def test_connect_failure_releases_lease(tmp_path):
    config = load_config(DEFAULT_CONFIG).robot
    config = replace(config, calibration_dir=tmp_path)
    def fail(_):
        raise RuntimeError("FCI busy")
    factory = lambda ip, port: HardwareLease(ip, port, directory=tmp_path / "leases")
    robot = FrankaRobotiqRobot(config, arm_factory=fail, gripper=FakeGripper(), hardware_lease_factory=factory)
    with pytest.raises(RuntimeError, match="FCI busy"):
        robot.connect()
    with factory(config.robot_ip, config.gripper_port):
        pass


def test_real_backend_automatically_acquires_shared_lease(tmp_path, monkeypatch):
    import franky
    monkeypatch.setattr(franky, "Robot", FakeArm)
    monkeypatch.setattr("vla.robots.franka_robotiq.robot.HardwareLease",
                        lambda ip, port: HardwareLease(ip, port, directory=tmp_path / "leases"))
    config = replace(load_config(DEFAULT_CONFIG).robot, calibration_dir=tmp_path)
    first = FrankaRobotiqRobot(config, gripper=FakeGripper())
    second = FrankaRobotiqRobot(config, gripper=FakeGripper())
    try:
        first.connect()
        with pytest.raises(RuntimeError, match="already owned"):
            second.connect()
        first.disconnect()
        second.connect()
    finally:
        first.disconnect()
        second.disconnect()


def test_live_gripper_prevents_hardware_lease_release(tmp_path):
    config = replace(load_config(DEFAULT_CONFIG).robot, calibration_dir=tmp_path)
    factory = lambda ip, port: HardwareLease(ip, port, directory=tmp_path / "leases")
    gripper = FakeGripper()
    robot = FrankaRobotiqRobot(config, arm_factory=FakeArm, gripper=gripper, hardware_lease_factory=factory)
    robot.connect()
    close = gripper.close
    def fail():
        raise RuntimeError("worker still active")
    gripper.close = fail
    try:
        with pytest.raises(RuntimeError, match="worker still active"):
            robot.disconnect()
        with pytest.raises(RuntimeError, match="already owned"):
            factory(config.robot_ip, config.gripper_port)
    finally:
        gripper.close = close
        robot.disconnect()


@pytest.mark.parametrize("previous", [0., 1., 1 - 128 / 255 + .001])
def test_gripper_release_hold_bypasses_ramp_and_deadband(previous):
    gripper = RobotiqModbus(port="fake", clock=lambda: 1., max_delta_per_s=.01)
    gripper._client = FakeModbusClient()
    gripper._status = parse_status_registers([0xF900, 0, 0x8000])
    gripper._status_time = 1.
    gripper._command = previous
    gripper.set_target(1.)
    gripper.hold_position()
    assert not gripper._client.writes  # Serial IO stays off the arm control thread.
    class Once(threading.Event):
        def wait(self, timeout=None):
            self.set()
            return True
    gripper._stop = Once()
    gripper._worker()
    assert gripper._error is None
    assert gripper._command == pytest.approx(1 - 128 / 255)
    assert gripper._client.writes[0][1][1] == 128


def test_gripper_hold_rejects_stale_state():
    gripper = RobotiqModbus(port="fake", clock=lambda: 1.)
    with pytest.raises(TimeoutError, match="stale"):
        gripper.hold_position()
    assert not gripper._hold_requested


def test_trigger_release_queues_hold_without_stopping_other_axes(rig, monkeypatch):
    enable(rig)
    held = []
    monkeypatch.setattr(rig.gripper, "hold_position", lambda: held.append(True))
    rig.receiver.packet = replace(rig.receiver.packet, axes=(1, 0, -1, 0, 0, 1))
    rig.advance()
    assert not held
    rig.receiver.packet = replace(rig.receiver.packet, axes=(1, 0, 1, 0, 0, 1))
    rig.advance()
    assert held == [True]
    assert np.linalg.norm(rig.robot.command_debug_snapshot()["commanded_linear_velocity"]) > 0
    rig.advance()
    assert held == [True]  # Not continually retargeted while at rest.


def test_gripper_watchdog_holds_measured_position_without_new_commands():
    clock = [1.]
    gripper = RobotiqModbus(port="fake", clock=lambda: clock[0], command_timeout_s=.1)
    gripper._client = FakeModbusClient()
    gripper._status = parse_status_registers([0xF900, 0, 0x8000])
    gripper._status_time = 1.
    gripper.set_target(0.)
    clock[0] = 1.11
    class Once(threading.Event):
        def wait(self, timeout=None):
            self.set()
            return True
    gripper._stop = Once()
    gripper._worker()
    assert gripper._error is None
    assert gripper._target == pytest.approx(1 - 128 / 255)
    assert gripper._client.writes[0][1][1] == 128


@pytest.mark.parametrize("section,key,value", [
    ("robot", "cameras_enabled", True), ("robot", "automatic_motion_recovery", True),
    ("control", "gamepad_xy_speed", float("nan")), ("mapping", "right_stick_x_axis", 2),
    ("mapping", "right_stick_x_sign", 0), ("control", "low_rate_samples", 2.5),
    ("speed_modes", "toggle_button", 0), ("speed_modes", "toggle_button", -1),
    ("speed_modes", "toggle_button", True), ("speed_modes", "high_multiplier", float("inf")),
    ("speed_modes", "high_multiplier", 1),
    ("speed_modes", "transition_time_s", 0), ("speed_modes", "transition_time_s", -.1),
    ("speed_modes", "transition_time_s", float("nan")), ("speed_modes", "transition_time_s", float("inf")),
    ("speed_modes", "transition_time_s", True), ("speed_modes", "transition_time_s", .01),
    ("pose_shortcuts", "view_button", 7), ("pose_shortcuts", "menu_button", 4),
    ("pose_shortcuts", "menu_button", True), ("pose_shortcuts", "position_tolerance_m", 0),
    ("pose_shortcuts", "orientation_tolerance_deg", 180), ("pose_shortcuts", "timeout_s", float("nan")),
    ("pose_shortcuts", "slowdown_distance_m", .01),
    ("pose_shortcuts", "final_approach_distance_m", .0005),
    ("pose_shortcuts", "final_approach_speed_m_s", 0),
    ("pose_shortcuts", "final_approach_speed_m_s", float("nan")),
    ("pose_shortcuts", "slowdown_distance_m", True),
    ("mapping", "right_stick_rotation_deg", float("nan")),
    ("mapping", "right_stick_rotation_deg", True),
    ("robot", "human_angular_velocity_control", "true"),
    ("robot", "human_linear_velocity_control", "true"),
    ("robot", "human_gripper_immediate_hold", "true"),
    ("robot", "human_max_vertical_velocity", float("nan")),
    ("robot", "human_max_vertical_velocity", True),
])
def test_invalid_configuration_is_rejected(tmp_path, section, key, value):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw.setdefault(section, {})[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(path)


def test_ros_environment_drops_conda_and_inherited_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/bad/python3.12/site-packages")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/bad/conda/lib")
    monkeypatch.setenv("CONDA_PREFIX", "/bad/conda")
    env = ros_environment()
    assert "CONDA_PREFIX" not in env and "LD_LIBRARY_PATH" not in env
    assert "/bad" not in env["PYTHONPATH"]
    assert env["PATH"].startswith("/usr/bin:")


def test_sidecar_cleanup_only_stops_owned_process(tmp_path):
    owned = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    manager = JoySidecars(load_config(DEFAULT_CONFIG), tmp_path / "joy.sock", tmp_path)
    manager.children.append((owned, tmp_path / "joy.log"))
    try:
        manager.close()
        assert owned.poll() is not None and other.poll() is None
    finally:
        for process in (owned, other):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)


def test_sidecar_partial_start_failure_cleans_up(tmp_path, monkeypatch):
    process_factory = subprocess.Popen
    started = []
    def spawn(*args, **kwargs):
        if started:
            raise OSError("bridge spawn failed")
        child = process_factory([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        started.append(child)
        return child
    monkeypatch.setattr("vla.datacollection.start_teleop.gamepad_teleop.subprocess.Popen", spawn)
    manager = JoySidecars(load_config(DEFAULT_CONFIG), tmp_path / "joy.sock", tmp_path)
    with pytest.raises(OSError, match="bridge spawn failed"):
        manager.__enter__()
    assert started[0].poll() is not None


def test_check_only_does_not_construct_hardware_or_start_children(monkeypatch):
    from vla.datacollection.start_teleop import gamepad_teleop as module
    calls = []
    monkeypatch.setattr(sys, "argv", ["gamepad_teleop", "--check-only"])
    monkeypatch.setattr(module, "preflight", lambda config: calls.append(config))
    monkeypatch.setattr(module, "run", lambda *a: pytest.fail("hardware execution requested"))
    module.main()
    assert len(calls) == 1 and not calls[0].robot.cameras_enabled


def test_control_shutdown_precedes_owned_sidecar_cleanup(tmp_path, monkeypatch):
    from vla.datacollection.start_teleop import gamepad_teleop as module
    events = []
    class Robot:
        is_connected = False
        def __init__(self, config):
            pass
        def connect(self):
            self.is_connected = True
        def end_human_control(self):
            events.append("stop")
        def disconnect(self):
            events.append("disconnect")
            self.is_connected = False
    class Sidecars:
        def __init__(self, *args):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_):
            events.append("sidecars-close")
        def check(self):
            raise RuntimeError("bridge died")
    class Receiver:
        def __init__(self, path):
            self.path = path
        def close(self):
            events.append("receiver-close")
    from contextlib import contextmanager
    @contextmanager
    def keys():
        yield 0
    config = load_config(DEFAULT_CONFIG)
    config = replace(config, robot=replace(config.robot, realtime_cpu=None))
    monkeypatch.setattr(module, "FrankaRobotiqRobot", Robot)
    monkeypatch.setattr(module, "JoySidecars", Sidecars)
    monkeypatch.setattr(module, "JoyReceiver", Receiver)
    monkeypatch.setattr(module, "keyboard", keys)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("unexpected ARM prompt"))
    with pytest.raises(RuntimeError, match="bridge died"):
        module.run(config)
    assert events == ["stop", "disconnect", "sidecars-close", "receiver-close"]


@pytest.mark.parametrize("recording", [False, True])
def test_launch_connects_without_prompt_and_waits_for_s(rig, tmp_path, monkeypatch, recording):
    from contextlib import nullcontext
    from vla.datacollection.start_teleop import gamepad_record, gamepad_teleop

    module = gamepad_record if recording else gamepad_teleop
    events = []

    def check_standby():
        assert not rig.arm.moves
        assert rig.robot._human_target_position is None
        events.append("standby")

    receiver = SimpleNamespace(path=tmp_path / "fake.sock", close=lambda: events.append("receiver closed"),
                               require_fresh=rig.receiver.require_fresh)
    sidecars = SimpleNamespace(check=check_standby)
    monkeypatch.setattr(module, "JoyReceiver", lambda *_: receiver)
    monkeypatch.setattr(module, "JoySidecars", lambda *_: nullcontext(sidecars))
    monkeypatch.setattr(module, "FrankaRobotiqRobot", lambda *_: rig.robot)
    monkeypatch.setattr(rig.robot, "connect", lambda: events.append("connected"))
    monkeypatch.setattr(module, "keyboard", lambda: nullcontext(0))
    monkeypatch.setattr(module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("unexpected ARM prompt"))
    monkeypatch.setattr(module.select, "select", lambda *_: ([0], [], []))
    monkeypatch.setattr(module.os, "read", lambda *_: b"c")
    if recording:
        _, record_config = gamepad_record.load_recording_config(gamepad_record.DEFAULT_CONFIG)
        store = SimpleNamespace(root=tmp_path, close=lambda: events.append("store closed"))
        monkeypatch.setattr(module, "RecordingStore", lambda *a, **k: store)
        viewer = SimpleNamespace(start=lambda: None, close=lambda: None, update_status=lambda *_: None)
        monkeypatch.setattr(module, "CameraPreview", lambda *a, **k: viewer)
        module.run(rig.config, record_config, tmp_path, "test task", False, preview=False)
        assert "store closed" in events
    else:
        module.run(rig.config)
    assert events[:2] == ["connected", "standby"]
    assert "receiver closed" in events
    assert not rig.arm.moves and not rig.robot.is_connected


def test_private_joy_transport_round_trip_and_cleanup(tmp_path):
    path = tmp_path / "joy.sock"
    try:
        receiver = JoyReceiver(path)
    except PermissionError:
        pytest.skip("sandbox disallows AF_UNIX bind")
    try:
        packet = JoyPacket(1, 10_000_000_000, (0, 0, 1, 0, 0, 1), (0,) * 11)
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
            sender.sendto(encode_joy_packet(packet), str(path))
        assert receiver.require_fresh(.2, now_ns=10_010_000_000) == packet
    finally:
        receiver.close()
    assert not path.exists()


def test_launcher_syntax_and_help():
    launcher = DEFAULT_CONFIG.parents[1] / "start_gamepad_single_left.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    assert os.access(launcher, os.X_OK)
    result = subprocess.run(["bash", str(launcher), "--help"], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--check-only" in result.stdout and "--execute" in result.stdout
