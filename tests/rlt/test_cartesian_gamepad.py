from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
from vla.datacollection.franka_robotiq_rollout.gamepad import decode_cartesian_gamepad_values, decode_gamepad_values
from vla.datacollection.franka_robotiq_rollout.collect_franky import FrankyRolloutCollector
from vla.robots.franka_robotiq.config import FrankaRobotiqConfig
from vla.robots.franka_robotiq.robot import (
    FrankaRobotiqRobot, action_vector, integrate_human_tool_rotation, rot6d_to_matrix,
)
from tests.rlt.test_franka_pi05_robot import FakeArm, FakeCamera, FakeGripper


CONFIG = Path(__file__).parents[2] / "src/vla/datacollection/franka_robotiq_rollout/configs/single_left.yaml"
LIMITS = dict(damping=.05, max_joint_speed=.3, joint_limit_margin=.05,
    position_gain=8., orientation_gain=3., max_linear_speed=.04,
    max_angular_speed=.3, gripper_speed_scale=.4)


@pytest.fixture
def config():
    return load_rollout_config(CONFIG)


def decode(config, axes=None, buttons=None, mapping=None):
    return decode_cartesian_gamepad_values(
        [0, 0, 1, 0, 0, 1] if axes is None else axes,
        [0] * 11 if buttons is None else buttons,
        config.arms[0].gamepad if mapping is None else mapping, config,
    )


@pytest.mark.parametrize("axis,value,expected", [
    (3, -1, [0, .15, 0]), (3, 1, [0, -.15, 0]),
    (4, 1, [.15, 0, 0]), (4, -1, [-.15, 0, 0]),
])
def test_right_stick_directions_keep_translation_zero(config, axis, value, expected):
    axes = [0, 0, 1, 0, 0, 1]
    axes[axis] = value
    command = decode(config, axes)
    np.testing.assert_allclose(command.linear_velocity_base, 0)
    np.testing.assert_allclose(command.angular_velocity_tool, expected)


def test_combined_inputs_and_opposite_buttons(config):
    buttons = [0] * 11
    buttons[2] = buttons[3] = 1
    command = decode(config, [1, 1, -1, -1, 1, 1], buttons)
    np.testing.assert_allclose(command.linear_velocity_base, [.02, .02, .01])
    expected = np.array([.15, .15, -.3])
    expected *= .3 / np.linalg.norm(expected)
    np.testing.assert_allclose(command.angular_velocity_tool, expected)
    assert command.gripper_motion == 1
    buttons[0] = buttons[1] = 1
    command = decode(config, [0, 0, -1, 0, 0, -1], buttons)
    np.testing.assert_allclose(command.angular_velocity_tool, 0)
    np.testing.assert_allclose(command.linear_velocity_base, 0)
    assert command.gripper_motion == 0


def test_deadzone_signs_and_legacy_decoder(config):
    np.testing.assert_allclose(decode(config, [0, 0, 1, .08, -.08, 1]).angular_velocity_tool, 0)
    mapping = replace(config.arms[0].gamepad, right_stick_x_sign=1., right_stick_y_sign=-1.)
    np.testing.assert_allclose(decode(config, [0, 0, 1, -1, 1, 1], mapping=mapping).angular_velocity_tool, [-.15, -.15, 0])
    legacy = decode_gamepad_values([0, 0, 1, 1, 1, 1], [0, 0, 1, 0], mapping, config)
    np.testing.assert_allclose(legacy[0], 0)
    assert legacy[1:] == (1, 0, 0)


@pytest.mark.parametrize("axes", [[0, 0, 1], [0, 0, 1, float("nan"), 0, 1], [0, 0, 1, 0, float("inf"), 1]])
def test_bad_joy_is_rejected(config, axes):
    with pytest.raises(ValueError):
        decode(config, axes)


def write_config(tmp_path, raw):
    raw["control"]["rtc_deployment_config"] = str(
        (CONFIG.parent / raw["control"]["rtc_deployment_config"]).resolve()
    )
    path = tmp_path / "gamepad.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_rollout_config(path)


def test_old_configuration_defaults_and_explicit_override(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    raw["control"].pop("gamepad_roll_speed")
    raw["control"]["gamepad_joint7_speed"] = .23
    mapping = raw["arms"][0]["gamepad"]
    mapping.pop("tool_roll_button_sign")
    mapping["joint7_button_sign"] = 1
    cfg = write_config(tmp_path, raw)
    assert cfg.gamepad_roll_speed == .23
    assert cfg.arms[0].gamepad.tool_roll_button_sign == 1
    raw["control"]["gamepad_roll_speed"] = .2
    mapping["tool_roll_button_sign"] = -1
    assert write_config(tmp_path, raw).gamepad_roll_speed == .2
    assert write_config(tmp_path, raw).arms[0].gamepad.tool_roll_button_sign == -1


@pytest.mark.parametrize("field,value", [("right_stick_x_axis", -1), ("right_stick_x_axis", 2), ("right_stick_y_sign", 0)])
def test_invalid_axis_configuration(tmp_path, field, value):
    raw = yaml.safe_load(CONFIG.read_text())
    raw["arms"][0]["gamepad"][field] = value
    with pytest.raises(ValueError, match="gamepad"):
        write_config(tmp_path, raw)


def test_rotation_uses_measured_tool_axes_and_one_exponential():
    current = Rotation.from_euler("xyz", [.6, -.4, .8]).as_matrix()
    target = Rotation.from_euler("xyz", [.62, -.4, .8]).as_matrix()
    angular = np.array([.1, -.12, .15])
    rotation, base = integrate_human_tool_rotation(current, target, angular,
        joint7_velocity=0, dt=.02, max_angular_speed=.3)
    np.testing.assert_allclose(base, current @ angular, atol=1e-12)
    np.testing.assert_allclose(rotation, Rotation.from_rotvec(base * .02).as_matrix() @ target, atol=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1)


@pytest.fixture
def robot(tmp_path, monkeypatch):
    monkeypatch.setattr("vla.robots.franka_robotiq.robot.time.monotonic", lambda: 10.)
    arm = FakeArm("not-hardware")
    arm.state.O_T_EE[:3, 3] = [.4, -.2, .3]
    arm.state.O_T_EE[:3, :3] = Rotation.from_euler("xyz", [.4, -.5, .8]).as_matrix()
    instance = FrankaRobotiqRobot(FrankaRobotiqConfig(id="test", calibration_dir=tmp_path),
        arm_factory=lambda _: arm, camera=FakeCamera(), gripper=FakeGripper())
    monkeypatch.setattr(instance, "_target_loop", lambda: instance._target_stop.wait())
    instance.connect()
    instance.begin_human_control()
    try:
        yield instance
    finally:
        instance.disconnect()


@pytest.mark.parametrize("online", [False, True])
def test_rotation_only_keeps_tcp_and_sends_one_cartesian_motion(robot, online):
    send = robot.send_human_target if online else robot.send_human_command
    position = robot._human_target_position.copy()
    for angular in ([.1, -.1, .1], [.1, 0, 0], [0, 0, -.3], [0, 0, 0]):
        count = len(robot._arm.moves)
        result = action_vector(send(np.zeros(3), 0, 0, .01,
            angular_velocity_tool=np.array(angular), **LIMITS))
        assert len(robot._arm.moves) == count + 1
        assert type(robot._arm.moves[-1][0]).__name__ == "CartesianVelocityMotion"
        np.testing.assert_array_equal(robot._human_target_position, position)
        np.testing.assert_allclose(result[:3], position, atol=1e-9)
        np.testing.assert_allclose(robot.command_debug_snapshot()["commanded_linear_velocity"], 0, atol=1e-9)
        assert np.linalg.norm(robot.command_debug_snapshot()["commanded_angular_velocity"]) <= .3 + 1e-12


def test_online_target_records_full_pose_without_servo_reintegration(robot):
    before = robot._arm.state.O_T_EE.copy()
    angular = np.array([.1, -.1, .15])
    result = action_vector(robot.send_human_target(np.array([.01, .01, .01]), 0, 1, 1 / 30,
        angular_velocity_tool=angular, **LIMITS))
    target = robot._target_command[0].copy()
    np.testing.assert_allclose(result[:3], before[:3, 3] + np.array([.01, .01, .01]) / 30)
    np.testing.assert_allclose(rot6d_to_matrix(result[3:9]), target[:3, :3])
    expected = Rotation.from_rotvec(before[:3, :3] @ angular / 30).as_matrix() @ before[:3, :3]
    np.testing.assert_allclose(target[:3, :3], expected)
    assert result[9] == 1 and robot._gripper.target == 1
    for _ in range(4):
        count = len(robot._arm.moves)
        robot._servo_target_once()
        assert len(robot._arm.moves) == count + 1
        np.testing.assert_array_equal(robot._target_command[0], target)
    robot._target_command = (*robot._target_command[:-1], 9.)
    with pytest.raises(TimeoutError, match="expired"):
        robot._servo_target_once()


def test_sft_online_target_anchors_to_measured_and_release_clears_tail(robot):
    pose = robot._arm.state.O_T_EE.copy()
    kwargs = dict(angular_velocity_tool=np.array([.1, .1, .2]), anchor_to_measured=True, **LIMITS)
    for _ in range(5):
        result = action_vector(robot.send_human_target(np.array([.01, 0, 0]), 0, 0, 1 / 30, **kwargs))
        np.testing.assert_allclose(result[:3], pose[:3, 3] + [.01 / 30, 0, 0])
    robot.send_human_target(np.zeros(3), 0, 0, 1 / 30,
        angular_velocity_tool=np.zeros(3), anchor_to_measured=True, **LIMITS)
    np.testing.assert_allclose(robot._target_command[0], pose)
    np.testing.assert_allclose(robot.command_debug_snapshot()["commanded_linear_velocity"], 0, atol=1e-12)
    np.testing.assert_allclose(robot.command_debug_snapshot()["commanded_angular_velocity"], 0, atol=1e-6)


def test_sft_online_gripper_release_holds_once(robot):
    holds = []
    robot._gripper.hold_position = lambda: holds.append(True)
    for motion in (1, 1, 0, 0):
        robot.send_human_target(np.zeros(3), 0, motion, 1 / 30,
            angular_velocity_tool=np.zeros(3), anchor_to_measured=True,
            immediate_gripper_hold=True, **LIMITS)
    assert holds == [True]


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("angular,joint7", [([float("nan"), 0, 0], 0), ([.1, 0], 0), ([.1, 0, 0], .1)])
def test_bad_or_conflicting_angular_commands_do_not_send(robot, online, angular, joint7):
    before = len(robot._arm.moves)
    position, rotation = robot._human_target_position.copy(), robot._human_target_rotation.copy()
    send = robot.send_human_target if online else robot.send_human_command
    with pytest.raises(ValueError):
        send(np.zeros(3), joint7, 0, .01, angular_velocity_tool=angular, **LIMITS)
    assert len(robot._arm.moves) == before
    np.testing.assert_array_equal(robot._human_target_position, position)
    np.testing.assert_array_equal(robot._human_target_rotation, rotation)


@pytest.mark.parametrize("online", [False, True])
def test_collector_combines_all_inputs_and_holds_on_stale_joy(config, robot, online):
    collector = object.__new__(FrankyRolloutCollector)
    collector.config, collector.arm, collector.robot = config, config.arms[0], robot
    collector.require_online = online
    collector._last_gamepad_tick = 9.99
    collector._episode_active = collector._intervention = True
    collector._faulted = collector._stale_joy_reported = False
    collector._phase, collector._command_sequence = "rl", 0
    collector._skip_reasons = Counter()
    events = []
    collector._emit = lambda *args, **kwargs: events.append(kwargs)
    collector._latch_fault = lambda error: pytest.fail(str(error))
    packet = SimpleNamespace(axes=[1, 1, -1, -1, 1, 1], buttons=[0, 0, 1, 1] + [0] * 7)
    collector.joy_receiver = SimpleNamespace(poll_latest=lambda: packet, require_fresh=lambda timeout: packet)
    before = len(robot._arm.moves)
    if online:
        collector.gamepad_tick()
        assert len(robot._arm.moves) == before
    collector.gamepad_tick(decide=True)
    assert len(robot._arm.moves) == before + 1
    assert collector._command.command_id == 1
    assert type(robot._arm.moves[-1][0]).__name__ == "CartesianVelocityMotion"
    assert all(value != 0 for value in events[-1]["linear"])
    assert all(value != 0 for value in events[-1]["angular_velocity_tool"])
    assert events[-1]["gripper_motion"] == 1
    position, rotation = robot._human_target_position.copy(), robot._human_target_rotation.copy()
    def stale(timeout):
        raise TimeoutError("stale joy")
    collector.joy_receiver.require_fresh = stale
    collector.gamepad_tick(decide=True)
    np.testing.assert_array_equal(robot._human_target_position, position)
    np.testing.assert_array_equal(robot._human_target_rotation, rotation)


def test_collector_sft_profile_only_sends_at_recording_boundary(config, robot):
    from vla.datacollection.start_teleop.rollout_gamepad import RolloutGamepadControls

    collector = object.__new__(FrankyRolloutCollector)
    collector.config, collector.arm, collector.robot = config, config.arms[0], robot
    collector.require_online = True
    collector._last_gamepad_tick = 9.99
    collector._episode_active = collector._intervention = True
    collector._faulted = collector._stale_joy_reported = False
    collector._phase, collector._command_sequence = "rl", 0
    collector._skip_reasons = Counter()
    collector._emit = lambda *args, **kwargs: None
    collector._latch_fault = lambda error: pytest.fail(str(error))
    neutral = SimpleNamespace(axes=[0, 0, 1, 0, 0, 1], buttons=[0] * 11)
    packet = SimpleNamespace(axes=[1, 1, -1, -1, 1, 1], buttons=[0, 0, 1, 1, 1] + [0] * 6)
    collector.joy_receiver = SimpleNamespace(poll_latest=lambda: packet, require_fresh=lambda timeout: packet)
    controls = RolloutGamepadControls(robot, collector.joy_receiver, report=lambda *args: None)
    collector._gamepad_controls = controls
    controls.observe(neutral, active=True, human=True)
    before = len(robot._arm.moves)
    for _ in range(9):
        collector.gamepad_tick()
        assert len(robot._arm.moves) == before
        collector.gamepad_tick(decide=True)
        before += 1
        assert len(robot._arm.moves) == before
    assert controls.speed_scale == 5
    np.testing.assert_allclose(collector._command.action[:3], robot._arm.state.O_T_EE[:3, 3] + [.1 / 30, .1 / 30, .05 / 30])
    assert robot._target_command[3]["max_linear_speed"] == .16
    expected = decode_cartesian_gamepad_values(packet.axes, packet.buttons, controls.config.mapping, controls.config.control)
    np.testing.assert_allclose(robot.command_debug_snapshot()["commanded_angular_velocity"],
        robot._arm.state.O_T_EE[:3, :3] @ expected.angular_velocity_tool, atol=1e-6)
