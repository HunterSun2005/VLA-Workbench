from __future__ import annotations

import os
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vla.sensors.realsense import CameraSnapshot, RealSenseCapture, prepare_image
from vla.control.geometry import ACTION_FIELDS, STATE_FIELDS, TARGET_SMOOTHING_ALPHA, limit_pose, matrix_to_rot6d, pose_step_to_twist, rot6d_to_matrix, solve_human_joint_velocity, smooth_pose_target, state_vector


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
        self.collision_calls = []

    def set_collision_behavior(self, *thresholds):
        self.collision_calls.append(thresholds)

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
