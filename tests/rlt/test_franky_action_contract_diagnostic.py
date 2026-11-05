"""Current-behavior diagnostics, NOT acceptance of a unified controller.

Run production action/recording methods with fake devices and capture the
CartesianVelocityMotion constructor boundary. Replace these expectations with
adapter equivalence tests when the versioned shared executor is implemented.
"""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tests.rlt.test_franka_pi05_robot import FakeArm, FakeCamera, FakeGripper
from tests.rlt.test_gamepad_recording import snapshot
from vla.datacollection.start_teleop.gamepad_recording import IntervalAssembler, command_from_debug
from vla.robots.franka_robotiq.config import FrankaRobotiqConfig
from vla.robots.franka_robotiq.robot import ACTION_FIELDS, FrankaRobotiqRobot, action_vector


LIMITS = dict(damping=.05, max_joint_speed=.3, joint_limit_margin=.05,
              position_gain=8., orientation_gain=3., max_linear_speed=.16,
              max_angular_speed=.30, gripper_speed_scale=.4)


@pytest.fixture
def robot_pair(tmp_path, monkeypatch):
    import franky
    # Capture actual submission arguments; never instantiate a real Robot.
    def motion(velocity, duration, dynamics):
        twist = velocity.end_effector_twist
        return NS(linear=np.array(twist.linear), angular=np.array(twist.angular))
    monkeypatch.setattr(franky, 'CartesianVelocityMotion', motion)
    robots = []
    for name in ('demonstration', 'replay'):
        arm = FakeArm('unused')
        arm.state.O_T_EE = snapshot().pose.copy()
        config = FrankaRobotiqConfig(
            id=name, calibration_dir=tmp_path / name, max_linear_velocity=.16,
            max_angular_velocity=.30, target_smoothing_alpha=0., velocity_smoothing=0.,
            min_command_position_delta=0., min_command_rotation_delta=0.,
            human_linear_velocity_control=True, human_angular_velocity_control=True,
            automatic_motion_recovery=False,
        )
        robot = FrankaRobotiqRobot(config, arm_factory=lambda _, arm=arm: arm,
                                  camera=FakeCamera(), gripper=FakeGripper())
        robot.connect()
        robots.append(robot)
    yield robots
    for robot in robots:
        robot._target_thread = None  # No background worker was started by this fixture.
        robot.disconnect()


def submitted(robot):
    motion = robot._arm.moves[-1][0]
    return np.r_[motion.linear, motion.angular]


@pytest.mark.parametrize('route,dt', [('interval_demo', 1/30), ('takeover', .01), ('online_takeover', 1/30)])
def test_current_recorded_label_replay_has_dt_over_tau_velocity_gap(robot_pair, route, dt):
    demonstration, replay = robot_pair
    demonstration.begin_human_control()
    # Frozen measurements isolate action semantics from tracking/timestamp error.
    # Both controllers start at the same measured pose and empty filter history.
    if route == 'online_takeover':
        demonstration._target_thread = NS(is_alive=lambda: True)  # Step servo explicitly, no thread.
    actual, reproduced = [], []
    for frame in range(3):
        stamp = 1 + frame / 30
        velocity = np.array([.03 * (-1 if frame == 1 else 1), 0., 0.])
        omega = np.array([0., 0., .12])
        if route == 'online_takeover':
            label = demonstration.send_human_target(
                velocity, 0., 0., dt, angular_velocity_tool=omega,
                anchor_to_measured=True, **LIMITS)
        else:
            label = demonstration.send_human_command(
                velocity, 0., 0., .01, angular_velocity_tool=omega, **LIMITS)
            if route == 'interval_demo':
                debug = demonstration.command_debug_snapshot()
                debug['human_command_at_ns'] = round(stamp * 1e9)
                assembler = IntervalAssembler(snapshot(stamp), command_from_debug(debug), False, 30)
                _, _, label, _, _ = assembler.observe(snapshot(stamp + dt), False)
        # Float32 serialization mimics dataset labels, not shared live arrays.
        label = np.array(json.loads(json.dumps(action_vector(label).astype(np.float32).tolist())))
        actual.append(submitted(demonstration))
        replay.send_action(dict(zip(ACTION_FIELDS, label, strict=True)))
        reproduced.append(submitted(replay))
    np.testing.assert_allclose(actual, [[.03, 0, 0, 0, 0, .12], [-.03, 0, 0, 0, 0, .12], [.03, 0, 0, 0, 0, .12]], atol=1e-7)
    np.testing.assert_allclose(reproduced, np.array(actual) * dt / .15, atol=2e-6)
    assert not np.allclose(reproduced, actual)  # Known mismatch; no xfail or claimed equivalence.


def test_current_send_action_return_reapplies_target_filter_on_replay(robot_pair):
    original, replay = robot_pair
    for robot in robot_pair:
        robot.config.target_smoothing_alpha = .25
        robot.config.velocity_smoothing = .30
        robot._last_smoothed_target = snapshot().pose.copy()
        robot._last_linear_velocity[:] = [.01, 0., 0.]
    action = np.r_[snapshot().pose[:3, 3] + [.003, 0., 0.], [1, 0, 0, 0, 1, 0], .75]
    returned = original.send_action(action)
    replay.send_action(returned)
    # Return value is postfilter: replay filters it again with identical history.
    assert returned[ACTION_FIELDS[0]] == pytest.approx(action[0] - .00075)
    assert submitted(original)[0] == pytest.approx(.0135)
    assert submitted(replay)[0] == pytest.approx(.010875)


def test_current_idle_stop_retains_velocity_history(robot_pair):
    robot, _ = robot_pair
    robot._last_linear_velocity[:] = [.01, 0, 0]
    robot._last_angular_velocity[:] = [0, 0, .02]
    robot._last_smoothed_target = snapshot().pose.copy()
    assert not robot._motion_started
    robot.stop()
    assert robot._last_smoothed_target is None
    np.testing.assert_array_equal(robot._last_linear_velocity, [.01, 0, 0])
    np.testing.assert_array_equal(robot._last_angular_velocity, [0, 0, .02])
