from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vla.datacollection.franka_robotiq_rollout.collect_franky import FrankyRolloutCollector
from vla.datacollection.franka_robotiq_rollout.gamepad import decode_cartesian_gamepad_values
from vla.datacollection.franka_robotiq_rollout.joy_transport import JoyPacket
from vla.datacollection.start_teleop.rollout_gamepad import RolloutGamepadControls
from vla.datacollection.start_teleop.gamepad_teleop import upright_tcp_rotation, axis_aligned_upright_rotation


def packet(*buttons, axes=None):
    values = [0] * 11
    for button in buttons:
        values[button] = 1
    return JoyPacket(0, 1, tuple(axes or [0, 0, 1, 0, 0, 1]), tuple(values))


def state(position=(.4, 0, .3), tilt=0):
    pose = np.eye(4)
    pose[:3, 3] = position
    pose[:3, :3] = Rotation.from_euler("xyz", [np.pi, tilt, 0]).as_matrix()
    return SimpleNamespace(O_T_EE=pose)


@pytest.fixture
def controls():
    result = RolloutGamepadControls(None, None, report=lambda *args: None)
    result.capture_initial_pose(state())
    return result


def observe(controls, value, *, active=True, human=True):
    return controls.observe(value, active=active, human=human)


def test_profile_reuses_sft_calibration_and_limits(controls):
    assert controls.config.mapping.right_stick_rotation_deg == -45
    assert controls.config.speed_modes.high_multiplier == 5
    assert controls.config.robot.max_linear_velocity == .16
    assert controls.critical_button == 5


def test_lb_combined_motion_ramps_xyz_only_without_stop(controls):
    observe(controls, packet())
    value = packet(4, 3, 2, axes=[1, 1, -1, -1, 1, 1])
    baseline = decode_cartesian_gamepad_values(value.axes, value.buttons, controls.config.mapping, controls.config.control)
    outputs = []
    for _ in range(9):
        observe(controls, value)
        outputs.append(controls.make_command(value, state(), 1 / 30))
    assert controls.high_speed and controls.speed_scale == 5
    for linear, angular, gripper, limits in outputs:
        assert np.linalg.norm(linear) > 0
        np.testing.assert_allclose(angular, baseline.angular_velocity_tool)
        assert gripper == baseline.gripper_motion
        assert limits["max_linear_speed"] <= .16
    np.testing.assert_allclose(outputs[-1][0], baseline.linear_velocity_base * 5)
    assert all(np.linalg.norm(a[0]) <= np.linalg.norm(b[0]) for a, b in zip(outputs, outputs[1:]))
    observe(controls, packet())
    observe(controls, value)
    assert not controls.high_speed
    controls.reset_human()
    observe(controls, value)
    assert not controls.high_speed  # Held LB across re-entry is not a new press.


def test_rb_edges_work_in_policy_and_human_but_reset_each_episode(controls):
    assert not observe(controls, packet(5), active=False)
    assert not observe(controls, packet(5), human=False)
    assert not observe(controls, packet(), human=False)
    assert observe(controls, packet(5), human=False)
    assert not observe(controls, packet(5), human=False)
    observe(controls, packet())
    assert observe(controls, packet(5, 4, 3, axes=[1, 0, 1, 0, 0, 1]))
    assert controls.high_speed
    controls.reset_episode()
    assert not observe(controls, packet(5))


@pytest.mark.parametrize("button", [6, 7])
def test_pose_shortcuts_ignored_outside_human_and_on_held_entry(controls, button):
    observe(controls, packet(), human=False)
    observe(controls, packet(button), human=False)
    observe(controls, packet(button), human=True)
    controls.make_command(packet(button), state(tilt=.3), 1 / 30)
    assert controls._pose_return is None


def test_menu_keeps_xyz_and_view_returns_to_start_at_high_speed(controls):
    observe(controls, packet())
    moved = state((.5, 0, .3), tilt=.3)
    observe(controls, packet(7))
    linear, angular, _, limits = controls.make_command(packet(7), moved, 1 / 30)
    np.testing.assert_allclose(linear, 0)
    assert np.linalg.norm(angular) > 0
    np.testing.assert_allclose(controls._pose_return.position, moved.O_T_EE[:3, 3])
    observe(controls, packet())
    observe(controls, packet(6))
    linear, _, _, limits = controls.make_command(packet(6), moved, 1 / 30)
    assert limits["max_linear_speed"] == .16
    assert linear[0] == pytest.approx(-.16)
    np.testing.assert_allclose(controls._pose_return.position, [.4, 0, .3])


@pytest.mark.parametrize("roll_degrees", [-65, 65])
def test_menu_restores_captured_heading_after_tool_roll(controls, roll_degrees):
    initial = state()
    moved = state((.5, 0, .3))
    moved.O_T_EE[:3, :3] = initial.O_T_EE[:3, :3] @ Rotation.from_euler(
        "zx", [roll_degrees, 20], degrees=True,
    ).as_matrix()
    observe(controls, packet())
    observe(controls, packet(7))
    linear, angular, _, _ = controls.make_command(packet(7), moved, 1 / 30)
    np.testing.assert_allclose(controls._pose_return.rotation, initial.O_T_EE[:3, :3], atol=1e-12)
    np.testing.assert_allclose(controls._pose_return.position, moved.O_T_EE[:3, 3])
    np.testing.assert_allclose(linear, 0)
    assert np.linalg.norm(angular) > 0


def test_menu_uses_configured_dataset_home_xyz_but_keeps_direct_return_path(controls):
    controls.configure_dataset_home_position(np.array([.3, -.3, .36]))
    observe(controls, packet())
    moved = state((.5, 0, .3), tilt=.3)
    observe(controls, packet(7))
    controls.make_command(packet(7), moved, 1 / 30)
    assert controls._pose_return.label == "upright"
    np.testing.assert_allclose(controls._pose_return.position, [.3, -.3, .36])
    np.testing.assert_allclose(
        controls._pose_return.rotation,
        upright_tcp_rotation(moved.O_T_EE[:3, :3]),
    )


@pytest.mark.parametrize("heading, expected_x", [(44, [1, 0, 0]), (46, [0, 1, 0]),
                                                (170, [-1, 0, 0]), (-80, [0, -1, 0])])
def test_axis_aligned_home_uses_nearest_dataset_heading(controls, heading, expected_x):
    reference = Rotation.from_euler("xyz", [175, 3, heading], degrees=True).as_matrix()
    target_rotation = axis_aligned_upright_rotation(reference)
    np.testing.assert_allclose(target_rotation[:, 0], expected_x)
    np.testing.assert_allclose(target_rotation[:, 2], [0, 0, -1])
    np.testing.assert_allclose(target_rotation.T @ target_rotation, np.eye(3))
    assert np.linalg.det(target_rotation) == pytest.approx(1)
    controls.configure_dataset_home_position([.3, -.3, .36], reference)
    observe(controls, packet())
    observe(controls, packet(7))
    controls.make_command(packet(7), state(tilt=.3), 1 / 30)
    np.testing.assert_allclose(controls._pose_return.rotation, target_rotation)
    assert controls._pose_return_path == []


def test_pose_request_is_latched_until_boundary_and_manual_input_cancels(controls):
    observe(controls, packet())
    observe(controls, packet(7))
    observe(controls, packet())  # A short button press between recording ticks.
    controls.make_command(packet(), state(tilt=.3), 1 / 30)
    assert controls._pose_return is not None
    observe(controls, packet(2, 1))  # Opposing buttons still cancel automatic return.
    assert controls._pose_return is None
    controls.reset_human()
    assert controls.speed_scale == 1 and controls._shortcut_packet is None


def test_dataset_home_uses_lift_transit_settle_and_manual_cancel(controls):
    controls._read_state = lambda: state()
    target = np.array([.3, -.3, .36])
    rotation = state().O_T_EE[:3, :3]
    controls.request_dataset_home_return(target, rotation, .45, .005)
    assert controls._pose_return.label == "home lift"
    np.testing.assert_allclose(controls._pose_return.position, [.4, 0, .45])
    linear, _, gripper, limits = controls.make_command(packet(), state(), 1 / 30)
    assert limits["max_linear_speed"] == pytest.approx(.16)
    assert np.linalg.norm(linear) > .04
    assert gripper == 0
    controls._pose_return_command(state((.4, 0, .446)), controls.config.control)
    assert controls._pose_return.label == "home transit"
    np.testing.assert_allclose(controls._pose_return.position, [.3, -.3, .45])
    controls._pose_return_command(state((.3, -.3, .45)), controls.config.control)
    assert controls._pose_return.label == "home settle"
    controls.observe(packet(2), active=True, human=True)
    assert controls._pose_return is None and not controls._pose_return_path


def collector(controls):
    value = object.__new__(FrankyRolloutCollector)
    value._gamepad_controls = controls
    value._episode_active = value._intervention = value.require_online = True
    value._faulted = value._at_boundary = value._critical_entered = False
    value._pending_event = value._pending_frame = None
    value._phase, value._segment_id = "vla", 0
    value._emit = lambda *args, **kwargs: None
    return value


def test_human_rb_changes_phase_only_at_boundary_and_cannot_reenter(controls):
    value = collector(controls)
    value.toggle_phase()
    assert value._phase == "vla" and value._pending_event == ("phase", None)
    value._pending_event, value._at_boundary = None, True
    value.toggle_phase()
    assert value._phase == "rl" and value._intervention and value._critical_entered
    value.toggle_phase()
    assert value._phase == "vla" and value._intervention
    value.toggle_phase()
    assert value._phase == "vla"


def test_rb_does_not_overwrite_pending_save(controls):
    value = collector(controls)
    value._pending_event = ("finish", "success")
    value.toggle_phase()
    assert value._pending_event == ("finish", "success")


def test_bad_special_button_packet_rejected(controls):
    with pytest.raises(ValueError, match="buttons"):
        observe(controls, JoyPacket(0, 1, (0, 0, 1, 0, 0, 1), (0, 0, 0, 0)))
