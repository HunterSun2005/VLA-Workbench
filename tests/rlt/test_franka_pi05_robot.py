from __future__ import annotations

import os
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vla.robots.franka_robotiq.camera import CameraSnapshot, RealSenseCapture, prepare_image
from vla.robots.franka_robotiq.config import FrankaRobotiqConfig
from vla.robots.franka_robotiq.gripper import (
    RobotiqModbus,
    action_to_request,
    command_registers,
    parse_status_registers,
)
from vla.robots.franka_robotiq.robot import (
    ACTION_FIELDS,
    STATE_FIELDS,
    TARGET_SMOOTHING_ALPHA,
    FrankaRobotiqRobot,
    limit_pose,
    matrix_to_rot6d,
    pose_step_to_twist,
    rot6d_to_matrix,
    solve_human_joint_velocity,
    smooth_pose_target,
    state_vector,
)


class FakeCamera:
    def __init__(self, timestamp_offset_s=0.0):
        self.connected = False
        self.closed = False
        self.timestamp_offset_s = timestamp_offset_s

    def connect(self):
        self.connected = True

    def check_health(self):
        if not self.connected:
            raise RuntimeError("camera disconnected")

    def images(self):
        return {name: np.zeros((224, 224, 3), dtype=np.uint8) for name in ("base", "left_wrist")}

    def snapshot(self):
        now = time.monotonic() + self.timestamp_offset_s
        return CameraSnapshot(
            images=self.images(),
            timestamps={name: now for name in ("base", "left_wrist")},
            pair_skew_s=0.0,
            oldest_age_s=0.0,
        )

    def close(self):
        self.closed = True


class FakeGripper:
    def __init__(self):
        self.connected = False
        self.target = None
        self.closed = False
        self.rate_scale = None

    def connect(self):
        self.connected = True

    def check_health(self):
        if not self.connected:
            raise RuntimeError("gripper disconnected")

    def position(self):
        return 0.75

    def set_target(self, value, *, rate_scale=1.0):
        self.target = value
        self.rate_scale = rate_scale

    def hold_position(self):
        self.set_target(self.position())

    def close(self):
        self.closed = True


class FakeResponse:
    def __init__(self, registers=None):
        self.registers = registers

    def isError(self):
        return False


class FakeModbusClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.writes = []
        self.closed = False

    def connect(self):
        return True

    def read_holding_registers(self, address, count, device_id=None, slave=None):
        # Activated and half-open at gPO=128.
        return FakeResponse([0xF900, 0, 0x8000])

    def write_registers(self, address, registers, device_id=None, slave=None):
        self.writes.append((address, registers, device_id or slave))
        return FakeResponse()

    def close(self):
        self.closed = True


class SequencedModbusClient(FakeModbusClient):
    def __init__(self, statuses, **kwargs):
        super().__init__(**kwargs)
        self.statuses = list(statuses)
        self.last_status = self.statuses[-1]

    def read_holding_registers(self, address, count, device_id=None, slave=None):
        if self.statuses:
            self.last_status = self.statuses.pop(0)
        return FakeResponse(self.last_status)


class FakeArm:
    def __init__(self, ip):
        self.ip = ip
        self.relative_dynamics_factor = None
        self.has_errors = False
        self.state = SimpleNamespace(
            O_T_EE=np.eye(4),
            q=np.array([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]),
            O_F_ext_hat_K=np.zeros(6),
            tau_J=np.arange(7), tau_ext_hat_filtered=np.arange(7)/2, dq=np.arange(7)/100,
        )
        jacobian = np.zeros((6, 7))
        jacobian[:6, :6] = np.eye(6)
        self.model = SimpleNamespace(zero_jacobian=lambda _frame, _state: jacobian)
        self.moves = []
        self.stop_calls = 0
        self.poll_calls = 0
        self.join_calls = 0
        self.recover_calls = 0
        self.is_in_control = False
        self.motion_error = None
        self.join_error = None
        self.join_result = True
        self.recovery_error = None
        self.recovery_result = True

    def move(self, motion, asynchronous=False):
        if self.motion_error is not None:
            self.is_in_control = False
            raise self.motion_error
        self.moves.append((motion, asynchronous))
        self.is_in_control = bool(asynchronous)

    def stop(self):
        self.stop_calls += 1
        self.is_in_control = False

    def poll_motion(self):
        self.poll_calls += 1
        if self.motion_error is not None:
            self.is_in_control = False
            raise self.motion_error
        return False

    def join_motion(self, timeout=None):
        self.join_calls += 1
        if self.join_error is not None:
            self.is_in_control = False
            raise self.join_error
        return self.join_result

    def recover_from_errors(self):
        self.recover_calls += 1
        if self.recovery_error is not None:
            raise self.recovery_error
        self.motion_error = None
        self.has_errors = False
        return self.recovery_result


def test_plugin_registration_and_lerobot_factory(tmp_path) -> None:
    from vla.adapters.lerobot import registry

    registry.register()
    from lerobot.robots.config import RobotConfig
    from lerobot.robots.utils import make_robot_from_config

    assert RobotConfig._choice_registry["franka_robotiq"] is FrankaRobotiqConfig
    cfg = FrankaRobotiqConfig(id="test", calibration_dir=tmp_path)
    robot = make_robot_from_config(cfg)
    assert isinstance(robot, FrankaRobotiqRobot)
    assert tuple(robot.observation_features)[:10] == STATE_FIELDS
    assert tuple(robot.action_features) == ACTION_FIELDS


def test_pose_state_and_rot6d_round_trip() -> None:
    rng = np.random.default_rng(7)
    for _ in range(20):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, -1] *= -1
        np.testing.assert_allclose(rot6d_to_matrix(matrix_to_rot6d(q)), q, atol=1e-7)
    pose = np.eye(4)
    pose[:3, 3] = [0.4, -0.1, 0.3]
    state = state_vector(SimpleNamespace(O_T_EE=pose), 0.6)
    np.testing.assert_allclose(state, [0.4, -0.1, 0.3, 1, 0, 0, 0, 1, 0, 0.6])


def test_camera_preserves_full_frame_with_centered_padding() -> None:
    image = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
    expected = np.pad(cv2.resize(image, (224, 168), interpolation=cv2.INTER_AREA),
                      ((28, 28), (0, 0), (0, 0)))
    np.testing.assert_array_equal(prepare_image(image), expected)


def _camera_capture(clock) -> RealSenseCapture:
    return RealSenseCapture(
        {"base": "base", "left_wrist": "wrist"},
        width=640,
        height=480,
        fps=30,
        image_size=224,
        timeout_s=1.0,
        max_pair_skew_s=0.025,
        max_frame_age_s=0.075,
        clock=clock,
    )


def test_camera_snapshot_selects_latest_synchronized_pair() -> None:
    now = 10.0
    capture = _camera_capture(lambda: now)
    capture._frames["base"].append((9.94, np.full((2, 2, 3), 1, dtype=np.uint8)))
    capture._frames["base"].append((9.99, np.full((2, 2, 3), 2, dtype=np.uint8)))
    capture._frames["left_wrist"].append((9.97, np.full((2, 2, 3), 3, dtype=np.uint8)))

    snapshot = capture.snapshot()

    assert snapshot.pair_skew_s == pytest.approx(0.02)
    assert snapshot.oldest_age_s == pytest.approx(0.03)
    assert snapshot.images["base"][0, 0, 0] == 2
    assert snapshot.images["left_wrist"][0, 0, 0] == 3


@pytest.mark.parametrize(
    "base_time,wrist_time,error",
    [(9.99, 9.95, "pair_skew"), (9.90, 9.90, "oldest_age")],
)
def test_camera_snapshot_rejects_skewed_or_stale_frames(base_time, wrist_time, error) -> None:
    capture = _camera_capture(lambda: 10.0)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    capture._frames["base"].append((base_time, image))
    capture._frames["left_wrist"].append((wrist_time, image))

    with pytest.raises(TimeoutError, match=error):
        capture.snapshot()


def test_workspace_linear_and_angular_step_limits() -> None:
    current = np.eye(4)
    target = np.eye(4)
    target[:3, 3] = [2.0, 0.2, -1.0]
    angle = 1.0
    target[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    limited = limit_pose(
        current, target, workspace_min=(0.2, -0.45, 0.05), workspace_max=(0.75, 0.45, 0.75),
        max_linear_step=0.01, max_angular_step=0.1,
    )
    assert np.linalg.norm(limited[:3, 3]) == pytest.approx(0.01)
    assert np.arccos((np.trace(limited[:3, :3]) - 1) / 2) == pytest.approx(0.1)


def test_human_joint_velocity_solver_preserves_xyz_and_controls_joint7() -> None:
    jacobian = np.zeros((6, 7))
    jacobian[:3, :3] = np.eye(3)
    velocity = solve_human_joint_velocity(
        jacobian,
        np.array([0.02, -0.01, 0.0, 0.0, 0.0, 0.0]),
        joint_positions=np.array([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]),
        joint7_velocity=0.3,
        joint7_active=True,
        damping=1e-4,
        max_joint_speed=0.3,
        joint_limit_margin=0.05,
        dt=0.01,
    )
    np.testing.assert_allclose(velocity[:3], [0.02, -0.01, 0.0], atol=1e-6)
    assert velocity[6] == pytest.approx(0.3, abs=1e-6)


def test_human_joint_velocity_solver_blocks_motion_through_joint_limit() -> None:
    jacobian = np.zeros((6, 7))
    jacobian[:6, :6] = np.eye(6)
    joints = np.array([2.70, 0.0, 0.0, -1.0, 0.0, 1.0, 3.0])
    velocity = solve_human_joint_velocity(
        jacobian,
        np.zeros(6),
        joint_positions=joints,
        joint7_velocity=0.3,
        joint7_active=True,
        damping=0.05,
        max_joint_speed=0.3,
        joint_limit_margin=0.05,
        dt=0.01,
    )
    assert velocity[6] <= 0.0


def test_limited_absolute_pose_converts_to_base_frame_twist() -> None:
    current = np.eye(4)
    limited = np.eye(4)
    limited[:3, 3] = [0.004, -0.002, 0.0]
    angle = 0.03
    limited[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    linear, angular = pose_step_to_twist(current, limited, 0.1)
    np.testing.assert_allclose(linear, [0.04, -0.02, 0.0])
    np.testing.assert_allclose(angular, [0.0, 0.0, 0.3], atol=1e-8)


def test_tracking_time_constant_must_be_positive(tmp_path) -> None:
    with pytest.raises(ValueError, match="tracking_tau"):
        FrankaRobotiqConfig(
            id="test",
            calibration_dir=tmp_path,
            velocity_tracking_tau_s=0.0,
        )


def test_control_startup_opens_gripper_before_motion(tmp_path) -> None:
    class OpeningGripper(FakeGripper):
        def __init__(self):
            super().__init__()
            self.positions = iter((0.0, 0.5, 0.96))
            self.last_position = 0.0

        def position(self):
            self.last_position = next(self.positions, self.last_position)
            return self.last_position

    arm, camera, gripper = FakeArm("unused"), FakeCamera(), OpeningGripper()
    cfg = FrankaRobotiqConfig(
        id="test",
        calibration_dir=tmp_path,
        gripper_startup_open_fraction=0.95,
        gripper_startup_open_timeout_s=0.2,
    )
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()

    measured = robot.prepare_for_control()

    assert measured == pytest.approx(0.96)
    assert gripper.target == 1.0
    assert robot._commanded_gripper == 1.0
    robot.disconnect()


def test_control_startup_rejects_gripper_that_remains_closed(tmp_path) -> None:
    class ClosedGripper(FakeGripper):
        def position(self):
            return 0.0

    arm, camera, gripper = FakeArm("unused"), FakeCamera(), ClosedGripper()
    cfg = FrankaRobotiqConfig(
        id="test",
        calibration_dir=tmp_path,
        gripper_startup_open_timeout_s=0.01,
    )
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()

    with pytest.raises(TimeoutError, match="did not open"):
        robot.prepare_for_control()
    assert gripper.target == 1.0
    robot.disconnect()


def test_absolute_pose_target_smoothing_uses_position_ema_and_rotation_slerp() -> None:
    previous = np.eye(4)
    previous[:3, 3] = [0.1, -0.2, 0.3]
    current = np.eye(4)
    current[:3, 3] = [0.5, 0.2, 0.7]
    angle = np.pi / 2
    current[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]

    smoothed = smooth_pose_target(previous, current)

    np.testing.assert_allclose(
        smoothed[:3, 3],
        TARGET_SMOOTHING_ALPHA * previous[:3, 3]
        + (1.0 - TARGET_SMOOTHING_ALPHA) * current[:3, 3],
    )
    smoothed_angle = np.arctan2(smoothed[1, 0], smoothed[0, 0])
    assert smoothed_angle == pytest.approx(0.75 * angle)


def test_robot_sends_absolute_async_motion_and_stops(monkeypatch, tmp_path) -> None:
    arm, camera, gripper = FakeArm("unused"), FakeCamera(), FakeGripper()
    cfg = FrankaRobotiqConfig(
        id="test",
        calibration_dir=tmp_path,
        control_frequency=10,
        velocity_motion_duration_ms=200,
        target_smoothing_alpha=0.4,
    )
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()
    action = np.array([0.5, 0, 0, 1, 0, 0, 0, 1, 0, 0.25], dtype=np.float32)
    sent = robot.send_action(action)
    assert arm.moves[0][1] is True
    assert type(arm.moves[0][0]).__name__ == "CartesianVelocityMotion"
    assert gripper.target is None
    debug = robot.command_debug_snapshot()
    np.testing.assert_allclose(debug["predicted_xyz"], action[:3])
    np.testing.assert_allclose(debug["predicted_action"], action)
    np.testing.assert_allclose(debug["command_measured_pose"], np.eye(4))
    np.testing.assert_allclose(
        np.asarray(debug["smoothed_pose"])[:3, 3],
        [sent[name] for name in ACTION_FIELDS[:3]],
    )
    assert debug["command_completed_monotonic_s"] >= debug["command_pose_sample_monotonic_s"]
    assert debug["predicted_gripper"] == pytest.approx(0.25)
    assert debug["commanded_gripper"] == 0.0
    assert len(debug["commanded_linear_velocity"]) == 3
    sent_position = np.array([sent[name] for name in ACTION_FIELDS[:3]])
    assert sent["gripper"] == 0.0
    assert np.linalg.norm(sent_position) == pytest.approx(
        cfg.max_linear_velocity * cfg.velocity_tracking_tau_s
    )
    assert np.linalg.norm(debug["unsmoothed_linear_velocity"]) == pytest.approx(
        cfg.max_linear_velocity
    )
    assert sent_position[2] > 0  # workspace z minimum participates in the limited step
    first_smoothed = np.asarray(debug["smoothed_xyz"])
    second_action = action.copy()
    second_action[1] = 0.4
    second_sent = robot.send_action(second_action)
    assert arm.poll_calls == 0
    second_debug = robot.command_debug_snapshot()
    expected_smoothed = (
        cfg.target_smoothing_alpha * first_smoothed
        + (1.0 - cfg.target_smoothing_alpha) * np.asarray(second_debug["limited_xyz"])
    )
    np.testing.assert_allclose(second_debug["smoothed_xyz"], expected_smoothed)
    np.testing.assert_allclose(
        [second_sent[name] for name in ACTION_FIELDS[:3]], expected_smoothed
    )
    arm.motion_error = RuntimeError("motion failed")
    robot.check_health()
    assert arm.recover_calls == 1
    assert not robot._motion_started
    robot.disconnect()
    assert robot._last_smoothed_target is None
    assert arm.stop_calls == 0
    assert arm.join_calls == 1
    assert camera.closed and gripper.closed


def test_angular_velocity_ema_is_disabled_by_default(tmp_path) -> None:
    arm, camera, gripper = FakeArm("unused"), FakeCamera(), FakeGripper()
    cfg = FrankaRobotiqConfig(
        id="test",
        calibration_dir=tmp_path,
        velocity_smoothing=0.3,
    )
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()
    angle = 0.1
    action = np.array(
        [
            0.5,
            0.0,
            0.2,
            np.cos(angle),
            np.sin(angle),
            0.0,
            -np.sin(angle),
            np.cos(angle),
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    robot.send_action(action)

    debug = robot.command_debug_snapshot()
    assert debug["smooth_angular_velocity"] is False
    np.testing.assert_allclose(
        debug["commanded_angular_velocity"],
        debug["unsmoothed_angular_velocity"],
    )
    np.testing.assert_allclose(
        debug["commanded_linear_velocity"],
        (1.0 - cfg.velocity_smoothing) * np.asarray(debug["unsmoothed_linear_velocity"]),
    )
    robot.disconnect()


def test_angular_velocity_ema_can_be_enabled(tmp_path) -> None:
    arm, camera, gripper = FakeArm("unused"), FakeCamera(), FakeGripper()
    cfg = FrankaRobotiqConfig(
        id="test",
        calibration_dir=tmp_path,
        velocity_smoothing=0.3,
        smooth_angular_velocity=True,
    )
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()
    angle = 0.1
    action = np.array(
        [
            0.5,
            0.0,
            0.2,
            np.cos(angle),
            np.sin(angle),
            0.0,
            -np.sin(angle),
            np.cos(angle),
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    robot.send_action(action)

    debug = robot.command_debug_snapshot()
    assert debug["smooth_angular_velocity"] is True
    np.testing.assert_allclose(
        debug["commanded_angular_velocity"],
        (1.0 - cfg.velocity_smoothing) * np.asarray(debug["unsmoothed_angular_velocity"]),
    )
    robot.disconnect()


@pytest.mark.parametrize("fail", [False, True])
def test_native_constructor_restores_callers_scheduler(monkeypatch, tmp_path, fail):
    robot = FrankaRobotiqRobot(FrankaRobotiqConfig(id="test", calibration_dir=tmp_path),
                              arm_factory=lambda _: None, camera=FakeCamera(), gripper=FakeGripper())
    robot._native_realtime = True
    current = [os.SCHED_OTHER, os.sched_param(0)]
    monkeypatch.setattr(os, "sched_getscheduler", lambda _: current[0])
    monkeypatch.setattr(os, "sched_getparam", lambda _: current[1])
    monkeypatch.setattr(os, "sched_setscheduler", lambda _, policy, param: current.__setitem__(slice(None), [policy, param]))
    def constructor(_):
        current[:] = [os.SCHED_FIFO, os.sched_param(99)]
        if fail:
            raise RuntimeError("constructor failed")
        return "arm"
    robot._arm_factory = constructor
    if fail:
        with pytest.raises(RuntimeError, match="constructor failed"):
            robot._connect_arm()
    else:
        robot._connect_arm()
        assert robot._arm == "arm"
    assert current == [os.SCHED_OTHER, os.sched_param(0)]


def test_communication_snapshot_reads_cached_state(tmp_path):
    robot = FrankaRobotiqRobot(FrankaRobotiqConfig(id="test", calibration_dir=tmp_path),
                              arm_factory=lambda _: None, camera=FakeCamera(), gripper=FakeGripper())
    robot._arm = SimpleNamespace(
        state=SimpleNamespace(control_command_success_rate=float(np.float32(.9)), time=3151.803),
        is_in_control=True,
    )
    fields = robot.communication_debug_snapshot()
    assert fields["control_command_success_rate"] == float(np.float32(.9))
    assert fields["robot_state_time_s"] == 3151.803
    assert fields["is_in_control"] is True


@pytest.mark.parametrize("cpu", [None, 6])
@pytest.mark.parametrize("fail", [False, True])
def test_native_motion_inherits_realtime_and_restores_caller(monkeypatch, tmp_path, cpu, fail):
    robot = FrankaRobotiqRobot(FrankaRobotiqConfig(id="test", calibration_dir=tmp_path, realtime_cpu=cpu),
                              arm_factory=lambda _: None, camera=FakeCamera(), gripper=FakeGripper())
    robot._native_realtime = True
    current = [os.SCHED_OTHER, os.sched_param(0)]
    affinity = [{0, 1}]
    monkeypatch.setattr(os, "cpu_count", lambda: 24)
    monkeypatch.setattr(os, "sched_getscheduler", lambda _: current[0])
    monkeypatch.setattr(os, "sched_getparam", lambda _: current[1])
    monkeypatch.setattr(os, "sched_setscheduler", lambda _, policy, param: current.__setitem__(slice(None), [policy, param]))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: affinity[0])
    monkeypatch.setattr(os, "sched_setaffinity", lambda _, value: affinity.__setitem__(0, value))
    def move(motion, asynchronous):
        assert asynchronous and motion == "motion"
        assert current == [os.SCHED_FIFO, os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO))]
        assert affinity[0] == ({cpu} if cpu is not None else {0, 1})
        if fail:
            raise RuntimeError("motion failed")
    robot._arm = SimpleNamespace(move=move)
    if fail:
        with pytest.raises(RuntimeError, match="motion failed"):
            robot._start_async_motion("motion")
    else:
        robot._start_async_motion("motion")
    assert current == [os.SCHED_OTHER, os.sched_param(0)]
    assert affinity[0] == {0, 1}


def test_robot_pins_new_native_realtime_thread(monkeypatch, tmp_path) -> None:
    arm, camera, gripper = FakeArm("unused"), FakeCamera(), FakeGripper()
    cfg = FrankaRobotiqConfig(id="test", calibration_dir=tmp_path, realtime_cpu=6)
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()
    housekeeping = {0, 1, 2, 3, 4, *range(8, 24)}
    monkeypatch.setattr("vla.robots.franka_robotiq.robot.os.sched_getaffinity", lambda _: housekeeping)
    pinned = []
    monkeypatch.setattr(
        "vla.robots.franka_robotiq.robot.os.sched_setaffinity",
        lambda tid, cpus: pinned.append((tid, cpus)),
    )
    action = np.array([0.5, 0, 0, 1, 0, 0, 0, 1, 0, 0.25], dtype=np.float32)
    robot.send_action(action)
    assert pinned == [(0, {6}), (0, housekeeping)]


def test_robot_refreshes_velocity_for_unchanged_target_and_commands_gripper(monkeypatch, tmp_path) -> None:
    arm, camera, gripper = FakeArm("unused"), FakeCamera(), FakeGripper()
    cfg = FrankaRobotiqConfig(id="test", calibration_dir=tmp_path)
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()
    action = np.array([0.5, 0, 0, 1, 0, 0, 0, 1, 0, 0.0], dtype=np.float32)

    first_sent = robot.send_action(action)
    assert len(arm.moves) == 1
    action[9] = 0.9
    gated_sent = robot.send_action(action)

    assert len(arm.moves) == 2
    assert gripper.target == 1.0
    np.testing.assert_allclose(
        [gated_sent[name] for name in ACTION_FIELDS[:9]],
        [first_sent[name] for name in ACTION_FIELDS[:9]],
    )
    debug = robot.command_debug_snapshot()
    assert debug["motion_command_requested"] is True
    assert debug["motion_command_sent"] is True
    assert debug["command_position_delta"] < cfg.min_command_position_delta
    assert debug["command_rotation_delta"] < cfg.min_command_rotation_delta
    robot.disconnect()


def test_robot_refreshes_zero_velocity_inside_measured_pose_deadband(monkeypatch, tmp_path) -> None:
    arm, camera, gripper = FakeArm("unused"), FakeCamera(), FakeGripper()
    cfg = FrankaRobotiqConfig(
        id="test",
        calibration_dir=tmp_path,
        workspace_min=(-1.0, -1.0, -1.0),
        workspace_max=(1.0, 1.0, 1.0),
    )
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()
    action = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=np.float32)
    robot.send_action(action)

    action[0] = 0.0001
    robot.send_action(action)
    action[0] = 0.0002
    robot.send_action(action)
    assert len(arm.moves) == 3
    np.testing.assert_array_equal(robot.command_debug_snapshot()["commanded_linear_velocity"], np.zeros(3))

    action[0] = 0.0003
    robot.send_action(action)
    assert len(arm.moves) == 4
    assert arm.poll_calls == 0
    assert robot.command_debug_snapshot()["command_position_delta"] > cfg.min_command_position_delta
    robot.disconnect()


@pytest.mark.parametrize("position_reached,rotation_reached", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("stop_source", ["measured_pose", "hold_command"])
def test_robot_clears_velocity_history_when_measured_target_is_reached(
    tmp_path, position_reached, rotation_reached, stop_source,
) -> None:
    arm, camera, gripper = FakeArm("unused"), FakeCamera(), FakeGripper()
    cfg = FrankaRobotiqConfig(
        id="test",
        calibration_dir=tmp_path,
        workspace_min=(-1.0, -1.0, -1.0),
        workspace_max=(1.0, 1.0, 1.0),
        smooth_angular_velocity=True,
    )
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()
    rotation = np.array([
        [np.cos(0.02), -np.sin(0.02), 0.0],
        [np.sin(0.02), np.cos(0.02), 0.0],
        [0.0, 0.0, 1.0],
    ])
    action = np.concatenate(([0.004, 0.0, 0.0], matrix_to_rot6d(rotation), [0.0]))
    try:
        robot.send_action(action)
        assert np.linalg.norm(robot._last_linear_velocity) > 0.0
        assert np.linalg.norm(robot._last_angular_velocity) > 0.0
        if position_reached:
            if stop_source == "measured_pose":
                arm.state.O_T_EE[:3, 3] = action[:3]
            else:
                action[:3] = arm.state.O_T_EE[:3, 3]
        if rotation_reached:
            if stop_source == "measured_pose":
                arm.state.O_T_EE[:3, :3] = rotation
            else:
                action[3:9] = matrix_to_rot6d(arm.state.O_T_EE[:3, :3])
        robot.send_action(action)
        assert len(arm.moves) == 2
        debug = robot.command_debug_snapshot()
        if position_reached:
            np.testing.assert_array_equal(debug["commanded_linear_velocity"], np.zeros(3))
            np.testing.assert_array_equal(robot._last_linear_velocity, np.zeros(3))
        else:
            assert np.linalg.norm(debug["commanded_linear_velocity"]) > 0.0
        if rotation_reached:
            np.testing.assert_array_equal(debug["commanded_angular_velocity"], np.zeros(3))
            np.testing.assert_array_equal(robot._last_angular_velocity, np.zeros(3))
        else:
            assert np.linalg.norm(debug["commanded_angular_velocity"]) > 0.0
    finally:
        robot.disconnect()


def test_robot_sends_reduced_velocity_while_approaching_unchanged_target(tmp_path) -> None:
    arm, camera, gripper = FakeArm("unused"), FakeCamera(), FakeGripper()
    cfg = FrankaRobotiqConfig(
        id="test",
        calibration_dir=tmp_path,
        workspace_min=(-1.0, -1.0, -1.0),
        workspace_max=(1.0, 1.0, 1.0),
        target_smoothing_alpha=0.0,
        velocity_smoothing=0.0,
    )
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()
    action = np.array([0.004, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=np.float64)
    try:
        robot.send_action(action)
        initial_velocity = robot._last_linear_velocity.copy()
        arm.state.O_T_EE[0, 3] = 0.002
        robot.send_action(action)
        assert len(arm.moves) == 2
        assert robot.command_debug_snapshot()["command_position_delta"] == pytest.approx(0.0)
        np.testing.assert_allclose(robot._last_linear_velocity, initial_velocity * 0.5)
    finally:
        robot.disconnect()


def test_robot_recovers_poll_error_and_sends_current_command(monkeypatch, tmp_path) -> None:
    arm, camera, gripper = FakeArm("unused"), FakeCamera(), FakeGripper()
    cfg = FrankaRobotiqConfig(id="test", calibration_dir=tmp_path)
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()
    action = np.array([0.5, 0, 0, 1, 0, 0, 0, 1, 0, 0.0], dtype=np.float32)
    robot.send_action(action)
    arm.motion_error = RuntimeError("communication_constraints_violation")
    action[1] = 0.4

    robot.check_health()
    robot.send_action(action)

    assert arm.recover_calls == 1
    assert len(arm.moves) == 2
    assert robot.command_debug_snapshot()["motion_recovered"] is True
    events = robot.drain_motion_events()
    assert any(event["event"] == "franky_motion_failed" and event["source"] == "poll" for event in events)
    assert any(event["event"] == "franky_motion_recovery_completed" for event in events)
    robot.disconnect()


def test_robot_recovers_move_error_and_retries_on_next_frame(monkeypatch, tmp_path) -> None:
    arm, camera, gripper = FakeArm("unused"), FakeCamera(), FakeGripper()
    cfg = FrankaRobotiqConfig(id="test", calibration_dir=tmp_path)
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()
    arm.motion_error = RuntimeError("move rejected")
    action = np.array([0.5, 0, 0, 1, 0, 0, 0, 1, 0, 0.9], dtype=np.float32)

    failed_action = robot.send_action(action)

    assert arm.recover_calls == 1
    assert len(arm.moves) == 0
    assert gripper.target is None
    np.testing.assert_allclose([failed_action[name] for name in ACTION_FIELDS[:3]], [0.0, 0.0, 0.0])
    assert robot.command_debug_snapshot()["motion_recovered"] is True
    assert robot.command_debug_snapshot()["consecutive_motion_failures"] == 1
    events = robot.drain_motion_events()
    assert any(event["event"] == "franky_motion_failed" and event["source"] == "move" for event in events)
    successful_action = robot.send_action(action)
    assert len(arm.moves) == 1
    assert gripper.target == 1.0
    assert np.linalg.norm([successful_action[name] for name in ACTION_FIELDS[:3]]) > 0
    assert robot.command_debug_snapshot()["motion_command_sent"] is True
    assert robot.command_debug_snapshot()["consecutive_motion_failures"] == 1
    robot.check_health()
    assert robot._consecutive_motion_failures == 1
    action[1] = 0.4
    robot.send_action(action)
    assert robot._consecutive_motion_failures == 0
    robot.disconnect()


def test_robot_stops_after_five_consecutive_motion_command_failures(tmp_path) -> None:
    arm, camera, gripper = FakeArm("unused"), FakeCamera(), FakeGripper()
    cfg = FrankaRobotiqConfig(id="test", calibration_dir=tmp_path, max_consecutive_motion_failures=5)
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()
    arm.motion_error = RuntimeError("persistent move rejection")

    def recover_without_clearing_move_error():
        arm.recover_calls += 1
        arm.has_errors = False
        return True

    arm.recover_from_errors = recover_without_clearing_move_error
    action = np.array([0.5, 0, 0, 1, 0, 0, 0, 1, 0, 0.9], dtype=np.float32)
    for expected_failures in range(1, 5):
        robot.send_action(action)
        assert robot.command_debug_snapshot()["consecutive_motion_failures"] == expected_failures
        assert gripper.target is None
    with pytest.raises(RuntimeError, match="5 consecutive"):
        robot.send_action(action)
    assert arm.recover_calls == 5
    assert gripper.target is None
    robot.disconnect()


def test_robot_rejects_recovery_that_leaves_arm_faulted(tmp_path) -> None:
    arm, camera, gripper = FakeArm("unused"), FakeCamera(), FakeGripper()
    cfg = FrankaRobotiqConfig(id="test", calibration_dir=tmp_path)
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()
    arm.has_errors = True

    def ineffective_recovery():
        arm.recover_calls += 1
        return False

    arm.recover_from_errors = ineffective_recovery
    with pytest.raises(RuntimeError, match="reported failure"):
        robot.check_health()
    assert arm.recover_calls == 1
    events = robot.drain_motion_events()
    assert any(
        event["event"] == "franky_motion_failed" and event["source"] == "health_check"
        for event in events
    )
    robot.disconnect()


def test_robot_rejects_recovery_when_faulted_motion_cannot_be_joined(tmp_path) -> None:
    arm, camera, gripper = FakeArm("unused"), FakeCamera(), FakeGripper()
    cfg = FrankaRobotiqConfig(id="test", calibration_dir=tmp_path)
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()
    action = np.array([0.5, 0, 0, 1, 0, 0, 0, 1, 0, 0.0], dtype=np.float32)
    robot.send_action(action)
    arm.motion_error = RuntimeError("communication_constraints_violation")
    arm.join_result = False

    def poll_fault_keeps_control_active():
        arm.poll_calls += 1
        raise arm.motion_error

    def stop_without_finishing_control():
        arm.stop_calls += 1

    arm.poll_motion = poll_fault_keeps_control_active
    arm.stop = stop_without_finishing_control
    with pytest.raises(RuntimeError, match="cleanup failed"):
        robot.check_health()
    assert arm.recover_calls == 0
    events = robot.drain_motion_events()
    assert any(
        event["event"] == "franky_motion_cleanup_failed" and event["operation"] == "complete"
        for event in events
    )


def test_robot_waits_for_fresh_state_after_recovery(tmp_path) -> None:
    arm, camera, gripper = FakeArm("unused"), FakeCamera(), FakeGripper()

    class FixedDuration:
        def to_sec(self):
            return 1.0

    arm.state = SimpleNamespace(O_T_EE=np.eye(4), time=FixedDuration())
    cfg = FrankaRobotiqConfig(
        id="test",
        calibration_dir=tmp_path,
        motion_recovery_timeout_s=0.002,
        motion_recovery_poll_interval_s=0.001,
    )
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()
    arm.has_errors = True

    with pytest.raises(RuntimeError, match="remains faulted"):
        robot.check_health()
    events = robot.drain_motion_events()
    assert any(
        event["event"] == "franky_motion_recovery_failed"
        and event["operation"] == "state_refresh"
        for event in events
    )


def test_robot_rejects_camera_frames_too_old_for_arm_state(tmp_path) -> None:
    arm, camera, gripper = FakeArm("unused"), FakeCamera(timestamp_offset_s=-0.08), FakeGripper()
    cfg = FrankaRobotiqConfig(id="test", calibration_dir=tmp_path)
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()

    with pytest.raises(TimeoutError, match="observation is stale"):
        robot.get_observation()
    robot.disconnect()


def test_robot_gripper_uses_hysteresis_and_only_commands_transitions(monkeypatch, tmp_path) -> None:
    arm, camera, gripper = FakeArm("unused"), FakeCamera(), FakeGripper()
    cfg = FrankaRobotiqConfig(id="test", calibration_dir=tmp_path)
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()  # FakeGripper reports exactly 0.75; the test version treats that as closed.
    observation = robot.get_observation()
    assert observation["gripper_position"] == pytest.approx(0.75 * 0.085)
    action = np.array([0.5, 0, 0, 1, 0, 0, 0, 1, 0, 0.0], dtype=np.float32)

    action[9] = 0.8
    assert robot.send_action(action)["gripper"] == 0.0
    assert gripper.target is None  # No state transition, so no repeated Modbus target.
    action[9] = 0.9
    assert robot.send_action(action)["gripper"] == 1.0
    assert gripper.target == 1.0
    gripper.target = None
    action[9] = 0.8
    assert robot.send_action(action)["gripper"] == 1.0
    assert gripper.target is None
    action[9] = 0.75
    assert robot.send_action(action)["gripper"] == 0.0
    assert gripper.target == 0.0
    robot.disconnect()


def test_closed_gripper_holds_open_command_for_two_seconds(monkeypatch, tmp_path) -> None:
    now = 7.0
    monkeypatch.setattr("vla.robots.franka_robotiq.robot.time.monotonic", lambda: now)
    arm, camera, gripper = FakeArm("unused"), FakeCamera(), FakeGripper()
    cfg = FrankaRobotiqConfig(id="test", calibration_dir=tmp_path)
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()
    action = np.array([0.5, 0, 0, 1, 0, 0, 0, 1, 0, 0.875], dtype=np.float32)

    assert robot.send_action(action)["gripper"] == 1.0
    now = 10.0
    action[9] = 0.75
    assert robot.send_action(action)["gripper"] == 0.0
    action[9] = 0.875
    now = 11.999
    assert robot.send_action(action)["gripper"] == 0.0
    now = 12.0
    assert robot.send_action(action)["gripper"] == 1.0
    assert gripper.target == 1.0
    robot.disconnect()


def test_robot_stop_uses_cartesian_velocity_stop_motion(monkeypatch, tmp_path) -> None:
    arm, camera, gripper = FakeArm("unused"), FakeCamera(), FakeGripper()
    cfg = FrankaRobotiqConfig(id="test", calibration_dir=tmp_path)
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()
    action = np.array([0.5, 0, 0, 1, 0, 0, 0, 1, 0, 0.25], dtype=np.float32)
    robot.send_action(action)
    robot.disconnect()
    assert type(arm.moves[-1][0]).__name__ == "CartesianVelocityStopMotion"
    assert arm.moves[-1][1] is False
    assert arm.stop_calls == 0
    assert arm.join_calls == 0
    assert camera.closed and gripper.closed


def test_robot_rollout_snapshot_and_human_mode_use_native_velocity_motions(tmp_path) -> None:
    arm, camera, gripper = FakeArm("unused"), FakeCamera(), FakeGripper()
    arm.state.O_T_EE = np.eye(4)
    arm.state.O_T_EE[:3, 3] = [0.4, 0.0, 0.3]
    arm.state.O_F_ext_hat_K = np.arange(6, dtype=float)
    cfg = FrankaRobotiqConfig(id="test", calibration_dir=tmp_path)
    robot = FrankaRobotiqRobot(
        cfg,
        arm_factory=lambda _: arm,
        camera=camera,
        gripper=gripper,
    )
    robot.connect()

    snapshot = robot.get_rollout_snapshot()
    assert snapshot.policy_state.shape == (10,)
    np.testing.assert_allclose(snapshot.base_wrench, np.arange(6))
    np.testing.assert_allclose(snapshot.joint_positions, arm.state.q)
    np.testing.assert_allclose(snapshot.joint_torques, arm.state.tau_J)
    np.testing.assert_allclose(snapshot.external_joint_torques, arm.state.tau_ext_hat_filtered)
    np.testing.assert_allclose(snapshot.joint_velocities, arm.state.dq)
    arm.state.tau_J[:] = -1
    np.testing.assert_allclose(snapshot.joint_torques, np.arange(7))  # Owned copy.

    robot.begin_human_control()
    sent = robot.send_human_command(
        np.array([0.01, 0.0, 0.0]),
        0.0,
        -0.5,
        0.01,
        damping=0.05,
        max_joint_speed=0.3,
        joint_limit_margin=0.05,
        position_gain=8.0,
        orientation_gain=3.0,
        max_linear_speed=0.04,
        max_angular_speed=0.3,
        gripper_speed_scale=0.4,
    )
    assert type(arm.moves[-1][0]).__name__ == "CartesianVelocityMotion"
    assert sent["gripper"] == 0.0
    assert gripper.rate_scale == pytest.approx(0.2)

    robot.send_human_command(
        np.array([0.01, -0.02, 0.01]),
        -0.2,
        0.0,
        0.01,
        damping=0.05,
        max_joint_speed=0.3,
        joint_limit_margin=0.05,
        position_gain=8.0,
        orientation_gain=3.0,
        max_linear_speed=0.04,
        max_angular_speed=0.3,
        gripper_speed_scale=0.4,
    )
    assert type(arm.moves[-1][0]).__name__ == "JointVelocityMotion"
    debug = robot.command_debug_snapshot()
    np.testing.assert_allclose(
        debug["commanded_linear_velocity"], [0.0116, -0.0216, 0.0108]
    )
    assert debug["commanded_joint_velocity"][0] > 0.0
    assert debug["commanded_joint_velocity"][1] < 0.0
    assert debug["commanded_joint_velocity"][2] > 0.0
    assert debug["commanded_joint_velocity"][6] == pytest.approx(-0.2, abs=1e-3)
    assert any(type(motion).__name__ == "CartesianVelocityStopMotion" for motion, _ in arm.moves)
    robot.end_human_control()
    assert type(arm.moves[-1][0]).__name__ == "JointVelocityStopMotion"
    robot.disconnect()


def test_robot_raises_when_motion_recovery_fails(monkeypatch, tmp_path) -> None:
    arm, camera, gripper = FakeArm("unused"), FakeCamera(), FakeGripper()
    cfg = FrankaRobotiqConfig(id="test", calibration_dir=tmp_path)
    robot = FrankaRobotiqRobot(cfg, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    robot.connect()
    action = np.array([0.5, 0, 0, 1, 0, 0, 0, 1, 0, 0.25], dtype=np.float32)
    robot.send_action(action)
    arm.has_errors = True
    arm.recovery_error = RuntimeError("recovery rejected")
    with pytest.raises(RuntimeError, match="motion recovery failed"):
        robot.check_health()
    assert arm.recover_calls == 1
    arm.motion_error = RuntimeError("communication_constraints_violation")
    robot.stop()
    robot.stop()  # Cleanup is idempotent.
    assert arm.stop_calls == 1
    assert arm.join_calls == 1


@pytest.mark.parametrize("action,expected_request", [(0, 255), (0.5, 128), (1, 0), (-2, 255), (2, 0)])
def test_robotiq_continuous_action_mapping(action, expected_request) -> None:
    assert action_to_request(action) == expected_request


def test_robotiq_activation_command_and_status_parse() -> None:
    assert command_registers(activate=True, position=64, speed=128, force=255) == [0x0900, 0x0040, 0x80FF]
    status = parse_status_registers([0xF900, 0x0040, 0x2080])
    assert status.activation_requested
    assert status.activation_status == 3
    assert status.activated
    assert status.position_request == 64
    assert status.position == 32
    assert status.open_fraction == pytest.approx(1 - 32 / 255)


def test_already_activated_gripper_connect_holds_measured_position() -> None:
    clients = []

    def factory(**kwargs):
        client = FakeModbusClient(**kwargs)
        clients.append(client)
        return client

    gripper = RobotiqModbus(port="mock", client_factory=factory)
    gripper.connect()
    try:
        assert gripper.position() == pytest.approx(1 - 128 / 255)
        assert clients[0].writes == []
    finally:
        gripper.close()
    assert clients[0].closed


@pytest.mark.parametrize("transient_fault", [0x05, 0x07])
def test_gripper_activation_waits_through_transient_fault(transient_fault) -> None:
    clients = []
    statuses = [
        [0x0000, 0, 0x8000],  # Initial inactive state.
        [0x0000, 0, 0x8000],  # Reset acknowledged.
        [0x0900, transient_fault << 8, 0x8000],
        [0xF900, 0, 0x8000],  # Activation complete.
    ]

    def factory(**kwargs):
        client = SequencedModbusClient(statuses, **kwargs)
        clients.append(client)
        return client

    gripper = RobotiqModbus(port="mock", client_factory=factory)
    gripper.connect()
    try:
        assert gripper.position() == pytest.approx(1 - 128 / 255)
        assert clients[0].writes == [
            (RobotiqModbus.COMMAND_ADDRESS, [0x0000, 0x0080, 0x8080], 9),
            (RobotiqModbus.COMMAND_ADDRESS, [0x0900, 0x0080, 0x8080], 9),
        ]
    finally:
        gripper.close()


def test_gripper_recovers_initial_communication_timeout() -> None:
    clients = []
    statuses = [
        [0xF900, 0x0900, 0x8000],  # Previously active, but communication timed out.
        [0x0000, 0, 0x8000],  # Reset acknowledged and fault cleared.
        [0xF900, 0, 0x8000],  # Reactivation complete.
    ]

    def factory(**kwargs):
        client = SequencedModbusClient(statuses, **kwargs)
        clients.append(client)
        return client

    gripper = RobotiqModbus(port="mock", client_factory=factory)
    gripper.connect()
    try:
        assert gripper.position() == pytest.approx(1 - 128 / 255)
        assert clients[0].writes == [
            (RobotiqModbus.COMMAND_ADDRESS, [0x0000, 0x0080, 0x8080], 9),
            (RobotiqModbus.COMMAND_ADDRESS, [0x0900, 0x0080, 0x8080], 9),
        ]
    finally:
        gripper.close()


def test_gripper_activation_hard_fault_closes_serial_client() -> None:
    clients = []
    statuses = [
        [0x0000, 0, 0x8000],
        [0x0000, 0, 0x8000],
        [0x0900, 0x0E00, 0x8000],
    ]

    def factory(**kwargs):
        client = SequencedModbusClient(statuses, **kwargs)
        clients.append(client)
        return client

    gripper = RobotiqModbus(port="mock", client_factory=factory)
    with pytest.raises(RuntimeError, match="activation fault 0x0e"):
        gripper.connect()
    assert clients[0].closed
    assert gripper._client is None


def test_gripper_initial_hard_fault_closes_serial_client() -> None:
    clients = []

    def factory(**kwargs):
        client = SequencedModbusClient([[0x0000, 0x0E00, 0x8000]], **kwargs)
        clients.append(client)
        return client

    gripper = RobotiqModbus(port="mock", client_factory=factory)
    with pytest.raises(RuntimeError, match="activation fault 0x0e"):
        gripper.connect()
    assert clients[0].writes == []
    assert clients[0].closed
    assert gripper._client is None


def test_gripper_activation_timeout_closes_serial_client() -> None:
    clients = []

    def factory(**kwargs):
        client = SequencedModbusClient([[0x0000, 0, 0x8000]], **kwargs)
        clients.append(client)
        return client

    gripper = RobotiqModbus(port="mock", client_factory=factory)
    with pytest.raises(TimeoutError, match="activation"):
        gripper.connect(activation_timeout_s=0)
    assert clients[0].closed
    assert gripper._client is None


def test_robotiq_state_timeout_and_worker_fault() -> None:
    now = [10.0]
    gripper = RobotiqModbus(port="mock", clock=lambda: now[0], state_timeout_s=0.5)
    gripper._status = parse_status_registers([0xF900, 0, 0])
    gripper._status_time = 9.0
    with pytest.raises(TimeoutError, match="stale"):
        gripper.position()
    gripper._error = OSError("serial failed")
    with pytest.raises(RuntimeError, match="serial failed"):
        gripper.position()
