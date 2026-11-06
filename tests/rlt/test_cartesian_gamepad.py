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
from vla.control.geometry import action_vector, integrate_human_tool_rotation, rot6d_to_matrix
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
    path = tmp_path / "gamepad.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_rollout_config(path)


def test_old_configuration_defaults_and_explicit_override(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    raw["control"].pop("gamepad_roll_speed", None)
    raw["control"]["gamepad_joint7_speed"] = .23
    mapping = raw["arms"][0]["gamepad"]
    mapping.pop("tool_roll_button_sign", None)
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
