from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import os
import threading
import time
from typing import Any, Callable

import numpy as np

from lerobot.robots import Robot

from vla.robots.franka_robotiq.camera import RealSenseCapture
from vla.robots.franka_robotiq.config import FrankaRobotiqConfig
from vla.robots.franka_robotiq.gripper import RobotiqModbus
from vla.robots.franka_robotiq.resource_lock import HardwareLease


log = logging.getLogger(__name__)


STATE_FIELDS = (
    "tcp_position_x", "tcp_position_y", "tcp_position_z",
    "tcp_rot6d_col0_x", "tcp_rot6d_col0_y", "tcp_rot6d_col0_z",
    "tcp_rot6d_col1_x", "tcp_rot6d_col1_y", "tcp_rot6d_col1_z",
    "gripper_position",
)
ACTION_FIELDS = (
    "desired_tcp_position_x", "desired_tcp_position_y", "desired_tcp_position_z",
    "desired_tcp_rot6d_col0_x", "desired_tcp_rot6d_col0_y", "desired_tcp_rot6d_col0_z",
    "desired_tcp_rot6d_col1_x", "desired_tcp_rot6d_col1_y", "desired_tcp_rot6d_col1_z",
    "gripper",
)
TARGET_SMOOTHING_ALPHA = 0.25
GRIPPER_OPEN_WIDTH_M = 0.085
FR3_JOINT_LOWER = np.array(
    [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159],
    dtype=np.float64,
)
FR3_JOINT_UPPER = np.array(
    [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159],
    dtype=np.float64,
)


@dataclass(frozen=True)
class FrankaRolloutSnapshot:
    observed_at_ns: int
    pose: np.ndarray
    joint_positions: np.ndarray
    base_wrench: np.ndarray
    joint_torques: np.ndarray
    external_joint_torques: np.ndarray
    joint_velocities: np.ndarray
    gripper_width: float
    gripper_open_fraction: float
    images: dict[str, np.ndarray]
    image_timestamps: dict[str, float]

    @property
    def policy_state(self) -> np.ndarray:
        return np.concatenate(
            (
                self.pose[:3, 3],
                matrix_to_rot6d(self.pose[:3, :3]),
                [self.gripper_width],
            )
        ).astype(np.float32)


def _clamp_vector_norm(value: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm == 0.0 or norm <= maximum:
        return value
    return value * (maximum / norm)


def solve_human_joint_velocity(
    jacobian: np.ndarray,
    cartesian_velocity: np.ndarray,
    *,
    joint_positions: np.ndarray,
    joint7_velocity: float,
    joint7_active: bool,
    damping: float,
    max_joint_speed: float,
    joint_limit_margin: float,
    dt: float,
) -> np.ndarray:
    """Map a Cartesian command and optional wrist command to bounded FR3 joint velocity."""
    jacobian = np.asarray(jacobian, dtype=np.float64)
    cartesian_velocity = np.asarray(cartesian_velocity, dtype=np.float64)
    joint_positions = np.asarray(joint_positions, dtype=np.float64)
    if jacobian.shape != (6, 7) or cartesian_velocity.shape != (6,):
        raise ValueError("human Jacobian/velocity must have shapes (6, 7) and (6,)")
    if joint_positions.shape != (7,):
        raise ValueError("human joint positions must have shape (7,)")
    scalars = (joint7_velocity, damping, max_joint_speed, joint_limit_margin, dt)
    if not all(np.isfinite(value) for value in scalars):
        raise ValueError("human control scalars must be finite")
    if not np.isfinite(jacobian).all() or not np.isfinite(cartesian_velocity).all():
        raise ValueError("human control input must be finite")
    if damping <= 0.0 or max_joint_speed <= 0.0 or dt <= 0.0:
        raise ValueError("human damping, max speed, and dt must be positive")

    if joint7_active:
        joint7_row = np.zeros((1, 7), dtype=np.float64)
        joint7_row[0, 6] = 1.0
        objective = np.vstack((jacobian[:3], joint7_row))
        target = np.concatenate((cartesian_velocity[:3], [joint7_velocity]))
    else:
        objective = jacobian
        target = cartesian_velocity
    regularized = objective @ objective.T + damping**2 * np.eye(len(objective))
    velocity = objective.T @ np.linalg.solve(regularized, target)

    lower = FR3_JOINT_LOWER + joint_limit_margin
    upper = FR3_JOINT_UPPER - joint_limit_margin
    minimum_velocity = (lower - joint_positions) / dt
    maximum_velocity = (upper - joint_positions) / dt
    return np.clip(
        velocity,
        np.maximum(-max_joint_speed, minimum_velocity),
        np.minimum(max_joint_speed, maximum_velocity),
    )


def integrate_human_tool_rotation(
    current_rotation: np.ndarray,
    target_rotation: np.ndarray,
    angular_velocity_tool: np.ndarray | None,
    *,
    joint7_velocity: float,
    dt: float,
    max_angular_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine tool-axis rates before one SO(3) increment; never rotate XYZ."""
    angular = np.zeros(3) if angular_velocity_tool is None else np.asarray(angular_velocity_tool, dtype=float)
    if angular.shape != (3,) or not np.isfinite(angular).all():
        raise ValueError("human tool angular velocity must be a finite 3-vector")
    if not np.isfinite([joint7_velocity, dt, max_angular_speed]).all() or dt <= 0 or max_angular_speed <= 0:
        raise ValueError("invalid human rotation limits")
    if joint7_velocity != 0 and np.any(angular != 0):
        raise ValueError("cannot combine joint-7 control with tool angular velocity")
    angular_base = current_rotation @ _clamp_vector_norm(angular, max_angular_speed)
    speed = float(np.linalg.norm(angular_base))
    rotation = target_rotation.copy()
    if speed > 1e-8:
        rotation = _axis_angle_matrix(angular_base / speed, speed * dt) @ rotation
    return rotation, angular_base


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation matrix must be finite with shape (3, 3)")
    return np.concatenate((matrix[:, 0], matrix[:, 1]))


def rot6d_to_matrix(rot6d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    value = np.asarray(rot6d, dtype=np.float64)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("rot6d must be a finite 6-vector")
    first = value[:3]
    norm = np.linalg.norm(first)
    if norm < eps:
        raise ValueError("rot6d first column has zero norm")
    first = first / norm
    second = value[3:] - np.dot(first, value[3:]) * first
    norm = np.linalg.norm(second)
    if norm < eps:
        raise ValueError("rot6d columns are collinear")
    second = second / norm
    return np.column_stack((first, second, np.cross(first, second)))


def pose_matrix(value: Any) -> np.ndarray:
    if hasattr(value, "matrix"):
        value = value.matrix
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size == 16:
        matrix = matrix.reshape(4, 4, order="F") if matrix.shape != (4, 4) else matrix
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"Franky O_T_EE must be a finite 4x4 pose, got {matrix.shape}")
    return matrix.copy()


def state_vector(robot_state: Any, gripper_position: float) -> np.ndarray:
    matrix = pose_matrix(robot_state.O_T_EE)
    value = np.concatenate((matrix[:3, 3], matrix_to_rot6d(matrix[:3, :3]), [gripper_position]))
    if value.shape != (10,) or not np.isfinite(value).all():
        raise ValueError("invalid Franka/Robotiq state")
    return value.astype(np.float32)


def action_vector(action: dict[str, Any] | np.ndarray) -> np.ndarray:
    if isinstance(action, dict):
        try:
            value = np.array([float(np.asarray(action[name]).reshape(-1)[0]) for name in ACTION_FIELDS])
        except KeyError as error:
            raise ValueError(f"action is missing field {error.args[0]!r}") from error
    else:
        value = np.asarray(action, dtype=np.float64)
    if value.shape != (10,) or not np.isfinite(value).all():
        raise ValueError(f"action must be a finite 10-vector, got {value.shape}")
    return value


def _rotation_angle(rotation: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))


def _rotation_axis(rotation: np.ndarray, angle: float) -> np.ndarray:
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0])
    axis = np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]
    )
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        values, vectors = np.linalg.eigh(rotation)
        axis = vectors[:, np.argmin(np.abs(values - 1.0))]
        return axis / np.linalg.norm(axis)
    return axis / norm


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def limit_pose(
    current: np.ndarray,
    target: np.ndarray,
    *,
    workspace_min: tuple[float, float, float],
    workspace_max: tuple[float, float, float],
    max_linear_step: float,
    max_angular_step: float,
) -> np.ndarray:
    current, target = pose_matrix(current), pose_matrix(target)
    result = target.copy()
    requested = np.clip(target[:3, 3], workspace_min, workspace_max)
    delta = requested - current[:3, 3]
    distance = np.linalg.norm(delta)
    if distance > max_linear_step:
        delta *= max_linear_step / distance
    result[:3, 3] = current[:3, 3] + delta

    relative = target[:3, :3] @ current[:3, :3].T
    angle = _rotation_angle(relative)
    if angle > max_angular_step:
        relative = _axis_angle_matrix(_rotation_axis(relative, angle), max_angular_step)
        result[:3, :3] = relative @ current[:3, :3]
    return result


def smooth_pose_target(
    previous: np.ndarray,
    current: np.ndarray,
    alpha: float = TARGET_SMOOTHING_ALPHA,
) -> np.ndarray:
    """Smooth consecutive absolute pose targets in position and SO(3)."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError("target smoothing alpha must be in [0, 1)")
    previous, current = pose_matrix(previous), pose_matrix(current)
    result = current.copy()
    result[:3, 3] = alpha * previous[:3, 3] + (1.0 - alpha) * current[:3, 3]

    relative = current[:3, :3] @ previous[:3, :3].T
    angle = _rotation_angle(relative)
    if angle < 1e-8:
        result[:3, :3] = previous[:3, :3]
    else:
        step = _axis_angle_matrix(_rotation_axis(relative, angle), (1.0 - alpha) * angle)
        result[:3, :3] = step @ previous[:3, :3]
    return result


def pose_step_to_twist(
    current: np.ndarray,
    limited: np.ndarray,
    tracking_tau_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a limited pose error into a base-frame tracking twist."""
    if tracking_tau_s <= 0:
        raise ValueError("tracking time constant must be positive")
    current, limited = pose_matrix(current), pose_matrix(limited)
    linear = (limited[:3, 3] - current[:3, 3]) / tracking_tau_s
    relative = limited[:3, :3] @ current[:3, :3].T
    angle = _rotation_angle(relative)
    angular = (
        _rotation_axis(relative, angle) * (angle / tracking_tau_s)
        if angle >= 1e-8
        else np.zeros(3)
    )
    return linear, angular


class FrankaRobotiqRobot(Robot):
    config_class = FrankaRobotiqConfig
    name = "franka_robotiq"

    def __init__(
        self,
        config: FrankaRobotiqConfig,
        *,
        arm_factory: Callable[[str], Any] | None = None,
        camera: Any | None = None,
        gripper: Any | None = None,
        hardware_lease_factory: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self._arm_factory = arm_factory
        self._native_realtime = arm_factory is None
        # Injected arm factories represent simulators or externally owned hardware.
        self._hardware_lease_factory = hardware_lease_factory or (
            HardwareLease if arm_factory is None else None
        )
        self._hardware_lease = None
        self._arm = None
        self._arm_lock = threading.RLock()
        self._target_lock = threading.RLock()
        self._target_stop = threading.Event()
        self._target_thread: threading.Thread | None = None
        self._target_command = None
        self._target_error: BaseException | None = None
        self._motion_started = False
        self._motion_mode: str | None = None
        self._motion_faulted = False
        self._last_linear_velocity = np.zeros(3, dtype=np.float64)
        self._last_angular_velocity = np.zeros(3, dtype=np.float64)
        self._last_smoothed_target: np.ndarray | None = None
        self._last_sent_target: np.ndarray | None = None
        self._last_command_debug: dict[str, Any] = {}
        self._last_observation_debug: dict[str, Any] = {}
        self._motion_events: deque[dict[str, Any]] = deque(maxlen=64)
        self._motion_event_sequence = 0
        self._consecutive_motion_failures = 0
        self._recovery_since_last_success = False
        self._commanded_gripper: float | None = None
        self._human_target_position: np.ndarray | None = None
        self._human_target_rotation: np.ndarray | None = None
        self._human_gripper_target: float | None = None
        self._human_gripper_direction = 0
        self._last_gripper_close_time = 0.0
        self._camera = (camera or RealSenseCapture(
            config.camera_serials,
            width=config.camera_width, height=config.camera_height, fps=config.camera_fps,
            image_size=config.image_size, timeout_s=config.camera_timeout_s,
            max_pair_skew_s=config.camera_max_pair_skew_s,
            max_frame_age_s=config.observation_max_image_age_s,
        )) if config.cameras_enabled else None
        self._gripper = gripper or RobotiqModbus(
            port=config.gripper_port, baudrate=config.gripper_baudrate,
            bytesize=config.gripper_bytesize, parity=config.gripper_parity,
            stopbits=config.gripper_stopbits, device_id=config.gripper_device_id,
            timeout_s=config.gripper_timeout_s, state_timeout_s=config.gripper_state_timeout_s,
            command_timeout_s=config.gripper_command_timeout_s,
            deadband=config.gripper_deadband, max_delta_per_s=config.gripper_max_delta_per_s,
            speed=config.gripper_speed, force=config.gripper_force,
        )

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        features: dict[str, type | tuple[int, int, int]] = {name: float for name in STATE_FIELDS}
        shape = (self.config.image_size, self.config.image_size, 3)
        if self.config.cameras_enabled:
            features.update({role: shape for role in self.config.camera_serials})
        return features

    @property
    def action_features(self) -> dict[str, type]:
        return {name: float for name in ACTION_FIELDS}

    @property
    def is_connected(self) -> bool:
        return self._arm is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        if not self.is_connected:
            raise RuntimeError("robot is not connected")
        self._arm.relative_dynamics_factor = self.config.dynamics_factor

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError("robot is already connected")
        if self._arm_factory is None:
            import franky

            self._arm_factory = franky.Robot
        try:
            if self._hardware_lease_factory is not None:
                self._hardware_lease = self._hardware_lease_factory(
                    self.config.robot_ip, self.config.gripper_port,
                )
            self._connect_arm()
            self.configure()
            self._gripper.connect()
            measured_gripper = self._gripper.position()
            self._commanded_gripper = float(measured_gripper > self.config.gripper_action_threshold)
            self._last_gripper_close_time = 0.0
            if self._camera is not None:
                self._camera.connect()
            self.check_health()
        except BaseException:
            self.disconnect()
            raise

    def check_health(self) -> None:
        if self._target_error is not None:
            raise RuntimeError(f"human target servo failed: {self._target_error}") from self._target_error
        if not self.is_connected:
            raise RuntimeError("robot is not connected")
        if self._motion_started:
            try:
                with self._arm_lock:
                    self._arm.poll_motion()
                    self._motion_started = self._arm_in_control()
            except Exception as error:
                with self._arm_lock:
                    self._recover_motion_error(error, source="poll")
        if self._arm_has_errors():
            state = self._arm.state
            current_errors = getattr(state, "current_errors", None)
            success_rate = getattr(state, "control_command_success_rate", None)
            details = []
            if current_errors is not None:
                details.append(f"current_errors={current_errors}")
            if success_rate is not None:
                details.append(f"control_command_success_rate={success_rate}")
            suffix = f": {', '.join(details)}" if details else ""
            with self._arm_lock:
                self._recover_motion_error(
                    RuntimeError(f"Franky reports a robot motion fault{suffix}"),
                    source="health_check",
                )
        self._gripper.check_health()
        if self._camera is not None:
            self._camera.check_health()

    def get_control_state(self) -> Any:
        """Read health-checked arm state without depending on camera observations."""
        self.check_health()
        with self._arm_lock:
            state = self._arm.state
            self._telemetry_state = state
            return state

    def recover_motion_fault(self) -> None:
        """Explicit, bounded recovery; never establishes a new human target."""
        if not self.is_connected:
            raise RuntimeError("robot is not connected")
        self.stop()
        try:
            self._gripper.check_health()
        except (RuntimeError, TimeoutError):
            self._gripper.recover()
        self.end_human_control()
        with self._arm_lock:
            if self._motion_faulted or self._arm_has_errors():
                self._recover_motion_error(
                    RuntimeError("operator requested recovery"), source="manual", manual=True,
                )
        self.check_health()

    def prepare_for_control(self) -> float:
        """Open the gripper before an armed control session starts."""
        if not self.is_connected:
            raise RuntimeError("robot is not connected")
        threshold = self.config.gripper_startup_open_fraction
        deadline = time.monotonic() + self.config.gripper_startup_open_timeout_s
        self._gripper.set_target(1.0)
        while True:
            measured = self._gripper.position()
            if measured >= threshold:
                self._commanded_gripper = 1.0
                self._last_gripper_close_time = 0.0
                return measured
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Robotiq did not open before control startup: "
                    f"measured={measured:.3f}, required={threshold:.3f}"
                )
            time.sleep(0.02)

    def get_observation(self) -> dict[str, Any]:
        if self._camera is None:
            raise RuntimeError("camera observations are disabled")
        self.check_health()
        camera_snapshot = self._camera.snapshot()
        gripper_width = self._gripper.position() * GRIPPER_OPEN_WIDTH_M
        with self._arm_lock:
            robot_state = self._arm.state
            state = state_vector(robot_state, gripper_width)
            state_sample_time = time.monotonic()
        image_state_skew_s = max(
            abs(state_sample_time - timestamp) for timestamp in camera_snapshot.timestamps.values()
        )
        if image_state_skew_s > self.config.observation_max_image_age_s:
            raise TimeoutError(
                "RealSense/robot-state observation is stale: "
                f"skew={image_state_skew_s * 1000.0:.1f}ms"
            )
        observation = {name: float(value) for name, value in zip(STATE_FIELDS, state)}
        observation.update(camera_snapshot.images)
        self._last_observation_debug = {
            "state_sample_monotonic_s": state_sample_time,
            "robot_state_time_s": self._state_time_s(robot_state),
            "measured_state": state.tolist(),
            "camera_receive_monotonic_s": dict(camera_snapshot.timestamps),
            "camera_pair_skew_ms": camera_snapshot.pair_skew_s * 1000.0,
            "camera_oldest_age_ms": camera_snapshot.oldest_age_s * 1000.0,
            "image_state_skew_ms": image_state_skew_s * 1000.0,
        }
        return observation

    def get_camera_preview_frames(self):
        """Latest independent RGB caches for Web previews; no FCI or camera IO."""
        if self._camera is None:
            return {}
        return self._camera.latest_frames()

    def get_camera_preview_snapshot(self):
        """Read existing RGB buffers only; no FCI access or new camera pipeline."""
        if self._camera is None:
            raise RuntimeError("camera observations are disabled")
        return self._camera.snapshot()

    def get_rollout_snapshot(self) -> FrankaRolloutSnapshot:
        """Capture synchronized Franky state, Robotiq state, and policy images."""
        if self._camera is None:
            raise RuntimeError("camera observations are disabled")
        self.check_health()
        camera_snapshot = self._camera.snapshot()
        gripper_open_fraction = self._gripper.position()
        with self._arm_lock:
            robot_state = self._arm.state
            self._telemetry_state = robot_state
            pose = pose_matrix(robot_state.O_T_EE)
            joint_positions = np.asarray(robot_state.q, dtype=np.float64).reshape(-1).copy()
            base_wrench = np.asarray(
                robot_state.O_F_ext_hat_K, dtype=np.float64
            ).reshape(-1).copy()
            joint_torques = np.asarray(robot_state.tau_J, dtype=np.float64).reshape(-1).copy()
            external_joint_torques = np.asarray(robot_state.tau_ext_hat_filtered, dtype=np.float64).reshape(-1).copy()
            joint_velocities = np.asarray(robot_state.dq, dtype=np.float64).reshape(-1).copy()
            state_sample_time = time.monotonic()
        if any(v.shape != (7,) or not np.isfinite(v).all()
               for v in (joint_torques, external_joint_torques, joint_velocities)):
            raise ValueError("Franky joint torque and velocity fields must be finite 7-vectors")
        if joint_positions.shape != (7,) or base_wrench.shape != (6,):
            raise ValueError(
                "Franky rollout state must provide seven joints and a six-axis base wrench"
            )
        if not np.isfinite(joint_positions).all() or not np.isfinite(base_wrench).all():
            raise ValueError("Franky rollout state contains NaN or infinity")
        image_state_skew_s = max(
            abs(state_sample_time - timestamp)
            for timestamp in camera_snapshot.timestamps.values()
        )
        if image_state_skew_s > self.config.observation_max_image_age_s:
            raise TimeoutError(
                "RealSense/robot-state rollout observation is stale: "
                f"skew={image_state_skew_s * 1000.0:.1f}ms"
            )
        self._last_observation_debug = {
            "camera_pair_skew_ms": camera_snapshot.pair_skew_s * 1000.0,
            "camera_oldest_age_ms": camera_snapshot.oldest_age_s * 1000.0,
            "image_state_skew_ms": image_state_skew_s * 1000.0,
        }
        return FrankaRolloutSnapshot(
            observed_at_ns=int(state_sample_time * 1e9),
            pose=pose,
            joint_positions=joint_positions,
            base_wrench=base_wrench,
            joint_torques=joint_torques,
            external_joint_torques=external_joint_torques,
            joint_velocities=joint_velocities,
            gripper_width=float(gripper_open_fraction * GRIPPER_OPEN_WIDTH_M),
            gripper_open_fraction=float(gripper_open_fraction),
            images={key: np.asarray(value).copy() for key, value in camera_snapshot.images.items()},
            image_timestamps=dict(camera_snapshot.timestamps),
        )

    def measured_pose(self) -> np.ndarray:
        if not self.is_connected:
            raise RuntimeError("robot is not connected")
        with self._arm_lock:
            return pose_matrix(self._arm.state.O_T_EE)

    def send_action(self, action: dict[str, Any] | np.ndarray) -> dict[str, float]:
        if not self.is_connected:
            raise RuntimeError("robot is not connected")
        if self._motion_mode == "joint_velocity":
            self.end_human_control()
        values = action_vector(action)
        target = np.eye(4)
        target[:3, 3] = values[:3]
        target[:3, :3] = rot6d_to_matrix(values[3:9])
        current = self.measured_pose()
        command_pose_sample_monotonic_s = time.monotonic()
        limited = limit_pose(
            current, target,
            workspace_min=self.config.workspace_min, workspace_max=self.config.workspace_max,
            max_linear_step=self.config.max_linear_velocity * self.config.velocity_tracking_tau_s,
            max_angular_step=self.config.max_angular_velocity * self.config.velocity_tracking_tau_s,
        )
        smoothed = (
            limited.copy()
            if self._last_smoothed_target is None
            else smooth_pose_target(
                self._last_smoothed_target,
                limited,
                alpha=self.config.target_smoothing_alpha,
            )
        )
        position_target_reached = (
            float(np.linalg.norm(limited[:3, 3] - current[:3, 3]))
            <= self.config.min_command_position_delta
        )
        rotation_target_reached = _rotation_angle(
            limited[:3, :3] @ current[:3, :3].T
        ) <= self.config.min_command_rotation_delta
        if position_target_reached:
            smoothed[:3, 3] = current[:3, 3]
        if rotation_target_reached:
            smoothed[:3, :3] = current[:3, :3]
        self._last_smoothed_target = smoothed.copy()
        if self._last_sent_target is None:
            command_position_delta = None
            command_rotation_delta = None
        else:
            command_position_delta = float(
                np.linalg.norm(smoothed[:3, 3] - self._last_sent_target[:3, 3])
            )
            command_rotation_delta = _rotation_angle(
                smoothed[:3, :3] @ self._last_sent_target[:3, :3].T
            )
        unsmoothed_linear, unsmoothed_angular = pose_step_to_twist(
            current,
            smoothed,
            self.config.velocity_tracking_tau_s,
        )
        alpha = self.config.velocity_smoothing
        linear = alpha * self._last_linear_velocity + (1.0 - alpha) * unsmoothed_linear
        angular = unsmoothed_angular
        if self.config.smooth_angular_velocity:
            angular = alpha * self._last_angular_velocity + (1.0 - alpha) * angular
        if position_target_reached:
            linear = np.zeros(3)
        if rotation_target_reached:
            angular = np.zeros(3)
        motion_command_sent = False
        motion_recovered = self._recovery_since_last_success
        recovery_blocks_gripper = False
        motion_failure_error: BaseException | None = None
        import franky

        motion = franky.CartesianVelocityMotion(
            franky.RobotVelocity(franky.Twist(linear, angular)),
            franky.Duration(self.config.velocity_motion_duration_ms),
            self.config.dynamics_factor,
        )
        with self._arm_lock:
            try:
                self._start_async_motion(motion)
            except Exception as error:
                self._recover_motion_error(error, source="move")
                motion_recovered = True
                recovery_blocks_gripper = True
                motion_failure_error = error
            else:
                self._motion_started = True
                self._motion_mode = "cartesian_velocity"
                self._motion_faulted = False
                self._last_linear_velocity = linear
                self._last_angular_velocity = angular
                self._last_smoothed_target = smoothed.copy()
                self._last_sent_target = smoothed.copy()
                motion_command_sent = True
        if motion_command_sent:
            if self._recovery_since_last_success:
                self._recovery_since_last_success = False
            else:
                self._consecutive_motion_failures = 0
        predicted_gripper = float(values[9])
        target_gripper = float(np.clip(predicted_gripper, 0.0, 1.0))
        gripper_command_sent = False
        gripper_command_allowed = (
            not recovery_blocks_gripper
            and motion_command_sent
        )
        if gripper_command_allowed:
            if self._commanded_gripper is None:
                self._commanded_gripper = float(target_gripper > self.config.gripper_action_threshold)
            elif self._commanded_gripper == 1.0 and target_gripper <= self.config.gripper_close_trigger:
                self._commanded_gripper = 0.0
                self._last_gripper_close_time = time.monotonic()
                gripper_command_sent = True
            elif (
                self._commanded_gripper == 0.0
                and target_gripper >= self.config.gripper_open_trigger
                and time.monotonic() - self._last_gripper_close_time >= self.config.gripper_post_close_lock_s
            ):
                self._commanded_gripper = 1.0
                gripper_command_sent = True
        commanded_gripper = self._commanded_gripper
        if gripper_command_sent:
            self._gripper.set_target(commanded_gripper)
        raw_relative_rotation = target[:3, :3] @ current[:3, :3].T
        limited_relative_rotation = limited[:3, :3] @ current[:3, :3].T
        smoothed_relative_rotation = smoothed[:3, :3] @ current[:3, :3].T
        self._last_command_debug = {
            "command_pose_sample_monotonic_s": command_pose_sample_monotonic_s,
            "command_completed_monotonic_s": time.monotonic(),
            "predicted_action": values.tolist(),
            "command_measured_pose": current.tolist(),
            "limited_pose": limited.tolist(),
            "smoothed_pose": smoothed.tolist(),
            "predicted_xyz": values[:3].tolist(),
            "measured_xyz": current[:3, 3].tolist(),
            "prediction_error_xyz": (values[:3] - current[:3, 3]).tolist(),
            "limited_xyz": limited[:3, 3].tolist(),
            "limited_delta_xyz": (limited[:3, 3] - current[:3, 3]).tolist(),
            "smoothed_xyz": smoothed[:3, 3].tolist(),
            "target_smoothing_delta_xyz": (smoothed[:3, 3] - limited[:3, 3]).tolist(),
            "prediction_rotation_error_rad": _rotation_angle(raw_relative_rotation),
            "limited_rotation_step_rad": _rotation_angle(limited_relative_rotation),
            "smoothed_rotation_step_rad": _rotation_angle(smoothed_relative_rotation),
            "target_smoothing_alpha": self.config.target_smoothing_alpha,
            "motion_command_requested": True,
            "motion_command_sent": motion_command_sent,
            "position_target_reached": position_target_reached,
            "rotation_target_reached": rotation_target_reached,
            "motion_recovered": motion_recovered,
            "consecutive_motion_failures": self._consecutive_motion_failures,
            "max_consecutive_motion_failures": self.config.max_consecutive_motion_failures,
            "command_position_delta": command_position_delta,
            "command_rotation_delta": command_rotation_delta,
            "min_command_position_delta": self.config.min_command_position_delta,
            "min_command_rotation_delta": self.config.min_command_rotation_delta,
            "velocity_tracking_tau_s": self.config.velocity_tracking_tau_s,
            "smooth_angular_velocity": self.config.smooth_angular_velocity,
            "unsmoothed_linear_velocity": unsmoothed_linear.tolist(),
            "unsmoothed_angular_velocity": unsmoothed_angular.tolist(),
            "commanded_linear_velocity": linear.tolist(),
            "commanded_angular_velocity": angular.tolist(),
            "predicted_gripper": predicted_gripper,
            "target_gripper": target_gripper,
            "commanded_gripper": commanded_gripper,
            "gripper_command_sent": gripper_command_sent,
            "gripper_command_allowed": gripper_command_allowed,
            "gripper_action_threshold": self.config.gripper_action_threshold,
            "gripper_close_trigger": self.config.gripper_close_trigger,
            "gripper_open_trigger": self.config.gripper_open_trigger,
            "gripper_post_close_lock_s": self.config.gripper_post_close_lock_s,
        }
        effective_pose = smoothed if motion_command_sent else self._last_sent_target
        if effective_pose is None:
            effective_pose = current
        self._last_command_debug["effective_xyz"] = effective_pose[:3, 3].tolist()
        if self._consecutive_motion_failures >= self.config.max_consecutive_motion_failures:
            raise RuntimeError(
                f"{self._consecutive_motion_failures} consecutive Franky motion commands failed"
            ) from motion_failure_error
        sent = np.concatenate(
            (effective_pose[:3, 3], matrix_to_rot6d(effective_pose[:3, :3]), [commanded_gripper])
        )
        return {name: float(value) for name, value in zip(ACTION_FIELDS, sent)}

    def begin_human_control(self) -> None:
        if not self.is_connected:
            raise RuntimeError("robot is not connected")
        self.stop()
        with self._arm_lock:
            pose = pose_matrix(self._arm.state.O_T_EE)
        measured_gripper = self._gripper.position()
        self._human_target_position = pose[:3, 3].copy()
        self._human_target_rotation = pose[:3, :3].copy()
        self._human_gripper_target = float(measured_gripper)
        self._human_gripper_direction = 0
        self._commanded_gripper = float(measured_gripper > self.config.gripper_action_threshold)

    def send_human_command(
        self,
        linear_velocity: np.ndarray,
        joint7_velocity: float,
        gripper_motion: float,
        dt: float,
        *,
        damping: float,
        max_joint_speed: float,
        joint_limit_margin: float,
        position_gain: float,
        orientation_gain: float,
        max_linear_speed: float,
        max_angular_speed: float,
        gripper_speed_scale: float,
        angular_velocity_tool: np.ndarray | None = None,
    ) -> dict[str, float]:
        if self._human_target_position is None or self._human_target_rotation is None:
            raise RuntimeError("begin_human_control() must be called before human commands")
        linear_velocity = np.asarray(linear_velocity, dtype=np.float64)
        if linear_velocity.shape != (3,) or not np.isfinite(linear_velocity).all():
            raise ValueError("human linear velocity must be a finite 3-vector")
        scalars = (
            joint7_velocity,
            gripper_motion,
            dt,
            damping,
            max_joint_speed,
            joint_limit_margin,
            position_gain,
            orientation_gain,
            max_linear_speed,
            max_angular_speed,
            gripper_speed_scale,
        )
        if not np.isfinite(scalars).all():
            raise ValueError("human control parameters must be finite")
        if (
            dt <= 0.0
            or damping <= 0.0
            or max_joint_speed <= 0.0
            or joint_limit_margin < 0.0
            or position_gain < 0.0
            or orientation_gain < 0.0
            or max_linear_speed <= 0.0
            or max_angular_speed <= 0.0
            or not 0.0 <= gripper_speed_scale <= 1.0
        ):
            raise ValueError("human control parameters are outside their valid ranges")

        import franky

        with self._arm_lock:
            state = self._arm.state
            current = pose_matrix(state.O_T_EE)
            joints = np.asarray(state.q, dtype=np.float64).reshape(-1)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError("Franky human control requires seven finite joint positions")
        direct_angular = self.config.human_angular_velocity_control and angular_velocity_tool is not None
        rotation, angular_base = integrate_human_tool_rotation(
            current[:3, :3], current[:3, :3] if direct_angular else self._human_target_rotation, angular_velocity_tool,
            joint7_velocity=joint7_velocity, dt=dt, max_angular_speed=max_angular_speed,
        )
        joint7_active = abs(float(joint7_velocity)) > 0.0
        linear_velocity = _clamp_vector_norm(linear_velocity, max_linear_speed)
        self._human_target_rotation = rotation
        if self.config.human_linear_velocity_control:
            # Rebase on measurement each cycle; never chase accumulated position
            # error after a release, reversal, or partial-axis release. Clip only
            # the requested velocity at workspace boundaries: do not generate an
            # unrequested return movement if the measured pose is already outside.
            lower = np.minimum((np.asarray(self.config.workspace_min) - current[:3, 3]) / dt, 0.0)
            upper = np.maximum((np.asarray(self.config.workspace_max) - current[:3, 3]) / dt, 0.0)
            commanded_linear = np.clip(linear_velocity, lower, upper)
            self._human_target_position = current[:3, 3] + commanded_linear * dt
        else:
            self._human_target_position = np.clip(
                self._human_target_position + linear_velocity * dt,
                self.config.workspace_min,
                self.config.workspace_max,
            )
            commanded_linear = linear_velocity + position_gain * (
                self._human_target_position - current[:3, 3]
            )
        if self.config.human_max_vertical_velocity is not None:
            # Increasing the XY speed ceiling must not also increase Z tracking speed.
            commanded_linear[2] = np.clip(
                commanded_linear[2], -self.config.human_max_vertical_velocity,
                self.config.human_max_vertical_velocity,
            )
        commanded_linear = _clamp_vector_norm(commanded_linear, max_linear_speed)
        if joint7_active:
            # Solve XYZ and joint 7 together so X/B can be held while translating.
            # Follow the measured wrist rotation to avoid undoing the intentional
            # joint-7 motion when Cartesian control resumes.
            self._human_target_rotation = current[:3, :3].copy()
            commanded_angular = np.zeros(3, dtype=np.float64)
            jacobian = np.asarray(
                self._arm.model.zero_jacobian(franky.Frame.EndEffector, state),
                dtype=np.float64,
            ).reshape(6, 7)
            joint_velocity = solve_human_joint_velocity(
                jacobian,
                np.concatenate((commanded_linear, commanded_angular)),
                joint_positions=joints,
                joint7_velocity=joint7_velocity,
                joint7_active=True,
                damping=damping,
                max_joint_speed=max_joint_speed,
                joint_limit_margin=joint_limit_margin,
                dt=dt,
            )
            motion_mode = "joint_velocity"
            motion = franky.JointVelocityMotion(
                joint_velocity,
                franky.Duration(self.config.velocity_motion_duration_ms),
                self.config.dynamics_factor,
            )
        else:
            if direct_angular:
                # No accumulating orientation error: releasing any angular input
                # removes its velocity immediately, even if another axis stays held.
                # Franky's native acceleration/jerk-limited deceleration still applies.
                commanded_angular = angular_base
            else:
                relative = self._human_target_rotation @ current[:3, :3].T
                angle = _rotation_angle(relative)
                commanded_angular = (
                    _rotation_axis(relative, angle) * angle * orientation_gain
                    if angle >= 1e-8
                    else np.zeros(3, dtype=np.float64)
                )
                commanded_angular += angular_base
            commanded_angular = _clamp_vector_norm(commanded_angular, max_angular_speed)
            joint_velocity = np.zeros(7, dtype=np.float64)
            motion_mode = "cartesian_velocity"
            # This is the same Franky command path used by the proven RTC VLA
            # deployment. It avoids depending on a model Jacobian that returns
            # an all-zero Cartesian solution on the deployed Franky build.
            motion = franky.CartesianVelocityMotion(
                franky.RobotVelocity(franky.Twist(commanded_linear, commanded_angular)),
                franky.Duration(self.config.velocity_motion_duration_ms),
                self.config.dynamics_factor,
            )

        if self._motion_started and self._motion_mode != motion_mode:
            self.stop()
        with self._arm_lock:
            try:
                self._start_async_motion(motion)
                human_command_at_ns = time.monotonic_ns()
            except Exception as error:
                self._recover_motion_error(error, source="human_move")
                raise RuntimeError(f"Franky human command failed: {error}") from error
            self._motion_started = True
            self._motion_mode = motion_mode
            self._motion_faulted = False

        predicted_pose = current.copy()
        predicted_pose[:3, 3] = np.clip(
            current[:3, 3] + commanded_linear * dt,
            self.config.workspace_min,
            self.config.workspace_max,
        )
        if not joint7_active:
            angular_speed = float(np.linalg.norm(commanded_angular))
            if angular_speed > 1e-8:
                predicted_pose[:3, :3] = _axis_angle_matrix(
                    commanded_angular / angular_speed,
                    angular_speed * dt,
                ) @ current[:3, :3]

        measured_gripper = self._gripper.position()
        direction = 0 if gripper_motion == 0.0 else (1 if gripper_motion > 0.0 else -1)
        if direction == 0:
            self._human_gripper_target = float(measured_gripper)
        else:
            self._human_gripper_target = 1.0 if direction > 0 else 0.0
        rate_scale = 1.0 if direction == 0 else float(
            np.clip(abs(gripper_motion) * gripper_speed_scale, 0.0, 1.0)
        )
        if direction != self._human_gripper_direction or direction != 0:
            if direction == 0 and self.config.human_gripper_immediate_hold:
                self._gripper.hold_position()
            else:
                self._gripper.set_target(self._human_gripper_target, rate_scale=rate_scale)
        self._human_gripper_direction = direction

        self._last_command_debug = {
            "motion_mode": motion_mode,
            "motion_command_sent": True,
            "measured_xyz": current[:3, 3].tolist(),
            "human_target_xyz": self._human_target_position.tolist(),
            "commanded_linear_velocity": commanded_linear.tolist(),
            "commanded_angular_velocity": commanded_angular.tolist(),
            "human_angular_velocity_control": direct_angular,
            "human_linear_velocity_control": self.config.human_linear_velocity_control,
            "human_command_at_ns": human_command_at_ns,
            "commanded_joint_velocity": joint_velocity.tolist(),
            "joint7_active": joint7_active,
            "human_gripper_direction": direction,
            "human_gripper_target": self._human_gripper_target,
            "human_gripper_rate_scale": rate_scale,
        }
        sent = np.concatenate(
            (
                predicted_pose[:3, 3],
                matrix_to_rot6d(predicted_pose[:3, :3]),
                [self._human_gripper_target],
            )
        )
        return {name: float(value) for name, value in zip(ACTION_FIELDS, sent)}

    def send_human_target(
        self, linear_velocity, joint7_velocity, gripper_motion, dt,
        *, angular_velocity_tool=None, anchor_to_measured=False,
        immediate_gripper_hold=False, **limits,
    ):
        """Latch one observation-rate target; the servo never integrates Joy input."""
        import franky

        self.check_health()
        if self._human_target_position is None:
            raise RuntimeError("begin_human_control() is required before a human target")
        values = np.asarray([*linear_velocity, joint7_velocity, gripper_motion, dt], dtype=float)
        if not np.isfinite(values).all() or dt <= 0:
            raise ValueError("invalid human target input")
        if np.asarray(linear_velocity).shape != (3,):
            raise ValueError("human linear velocity must be a finite 3-vector")
        if not np.isfinite(limits["max_linear_speed"]) or limits["max_linear_speed"] <= 0:
            raise ValueError("invalid human linear speed limit")
        with self._target_lock:
            if self._target_error is not None:
                raise RuntimeError(f"human target servo failed: {self._target_error}")
            with self._arm_lock:
                state = self._arm.state
                current = pose_matrix(state.O_T_EE)
                joints = np.asarray(state.q, dtype=float).copy()
            target = current.copy()
            linear = _clamp_vector_norm(np.asarray(linear_velocity, dtype=float), limits["max_linear_speed"])
            # Observation-rate anchoring avoids accumulating an unexecuted human
            # target while preserving the recorded absolute-action contract.
            position_origin = current[:3, 3] if anchor_to_measured else self._human_target_position
            rotation_origin = current[:3, :3] if anchor_to_measured else self._human_target_rotation
            target[:3, 3] = np.clip(
                position_origin + linear * dt,
                self.config.workspace_min, self.config.workspace_max,
            )
            target[:3, :3], _ = integrate_human_tool_rotation(
                current[:3, :3], rotation_origin, angular_velocity_tool,
                joint7_velocity=joint7_velocity, dt=dt, max_angular_speed=limits["max_angular_speed"],
            )
            joint_goal = None
            if joint7_velocity:
                jacobian = np.asarray(self._arm.model.zero_jacobian(franky.Frame.EndEffector, state)).reshape(6, 7)
                velocity = solve_human_joint_velocity(
                    jacobian, np.concatenate((linear, np.zeros(3))), joint_positions=joints,
                    joint7_velocity=joint7_velocity, joint7_active=True,
                    damping=limits["damping"], max_joint_speed=limits["max_joint_speed"],
                    joint_limit_margin=limits["joint_limit_margin"], dt=dt,
                )
                predicted = pose_matrix(self._arm.model.pose(
                    franky.Frame.EndEffector, joints + velocity * dt,
                    franky.Affine(pose_matrix(state.F_T_EE)), franky.Affine(pose_matrix(state.EE_T_K)),
                ))
                if not np.isfinite(predicted).all() or not np.allclose(predicted[:3, :3].T @ predicted[:3, :3], np.eye(3), atol=1e-5):
                    raise ValueError("invalid Franky forward-kinematics target")
                target[:3, :3] = predicted[:3, :3]
                joint_goal = float(joints[6] + velocity[6] * dt)
            self._human_target_position = target[:3, 3].copy()
            self._human_target_rotation = target[:3, :3].copy()
            measured = self._gripper.position()
            gripper = measured if gripper_motion == 0 else float(gripper_motion > 0)
            if immediate_gripper_hold and gripper_motion == 0:
                if self._human_gripper_direction != 0:
                    self._gripper.hold_position()
            else:
                self._gripper.set_target(gripper, rate_scale=1.0 if gripper_motion == 0 else float(np.clip(abs(gripper_motion) * limits["gripper_speed_scale"], 0, 1)))
            self._human_gripper_direction = int(np.sign(gripper_motion))
            self._human_gripper_target = gripper
            self._target_command = (target, joint_goal, dt, dict(limits), time.monotonic() + self.config.velocity_motion_duration_ms / 1000)
            self._servo_target_once()
            if self._target_thread is None or not self._target_thread.is_alive():
                self._target_stop.clear()
                self._target_thread = threading.Thread(target=self._target_loop, name="franka-human-target", daemon=True)
                self._target_thread.start()
            sent = np.concatenate((target[:3, 3], matrix_to_rot6d(target[:3, :3]), [gripper]))
            return dict(zip(ACTION_FIELDS, sent, strict=True))

    def _servo_target_once(self) -> None:
        import franky

        target, joint_goal, dt, limits, deadline = self._target_command
        if time.monotonic() >= deadline:
            raise TimeoutError("human target expired; a fresh observation is required")
        with self._arm_lock:
            state = self._arm.state
            current = pose_matrix(state.O_T_EE)
            linear, angular = pose_step_to_twist(current, target, dt)
            linear = _clamp_vector_norm(linear, limits["max_linear_speed"])
            angular = _clamp_vector_norm(angular, limits["max_angular_speed"])
            mode = "joint_velocity" if joint_goal is not None else "cartesian_velocity"
            if self._motion_started and self._motion_mode != mode:
                self._arm.stop()
                self._arm.join_motion(self.config.motion_cleanup_timeout_s)
                self._motion_started = False
            if joint_goal is not None:
                joints = np.asarray(state.q, dtype=float)
                jacobian = np.asarray(self._arm.model.zero_jacobian(franky.Frame.EndEffector, state)).reshape(6, 7)
                velocity = solve_human_joint_velocity(
                    jacobian, np.concatenate((linear, angular)), joint_positions=joints,
                    joint7_velocity=(joint_goal - joints[6]) / dt, joint7_active=True,
                    damping=limits["damping"], max_joint_speed=limits["max_joint_speed"],
                    joint_limit_margin=limits["joint_limit_margin"], dt=0.01,
                )
                motion = franky.JointVelocityMotion(velocity, franky.Duration(self.config.velocity_motion_duration_ms), self.config.dynamics_factor)
            else:
                motion = franky.CartesianVelocityMotion(franky.RobotVelocity(franky.Twist(linear, angular)), franky.Duration(self.config.velocity_motion_duration_ms), self.config.dynamics_factor)
            self._start_async_motion(motion)
            self._motion_started, self._motion_mode = True, mode
            self._last_command_debug = {
                "motion_mode": mode,
                "motion_command_sent": True,
                "commanded_linear_velocity": linear.tolist(),
                "commanded_angular_velocity": angular.tolist(),
                "human_target_xyz": target[:3, 3].tolist(),
            }

    def _target_loop(self) -> None:
        while not self._target_stop.wait(0.01):
            try:
                with self._target_lock:
                    if self._target_command is None:
                        return
                    self._servo_target_once()
            except BaseException as error:
                self._target_error = error
                self._target_stop.set()
                try:
                    self.stop()
                except BaseException:
                    pass
                self._target_error = error
                return

    def end_human_control(self) -> None:
        try:
            self.stop()
        finally:
            try:
                measured_gripper = self._gripper.position() if self.is_connected else None
                if measured_gripper is not None:
                    if self.config.human_gripper_immediate_hold:
                        self._gripper.hold_position()
                    else:
                        self._gripper.set_target(measured_gripper)
                    self._commanded_gripper = float(
                        measured_gripper > self.config.gripper_action_threshold
                    )
            finally:
                self._human_target_position = None
                self._human_target_rotation = None
                self._human_gripper_target = None
                self._human_gripper_direction = 0

    def _connect_arm(self) -> None:
        """libfranka elevates its constructor's caller; do not leak that to workers."""
        if not self._native_realtime:
            self._arm = self._arm_factory(self.config.robot_ip)
            return
        policy, priority = os.sched_getscheduler(0), os.sched_getparam(0)
        try:
            self._arm = self._arm_factory(self.config.robot_ip)
        finally:
            os.sched_setscheduler(0, policy, priority)

    def _start_async_motion(self, motion: Any) -> None:
        """Start Franky's native thread with the dedicated CPU affinity inherited."""
        cpu = self.config.realtime_cpu
        if cpu is not None and cpu >= (os.cpu_count() or 1):
            raise RuntimeError(f"configured Franky realtime CPU {cpu} is not online")

        caller_affinity = os.sched_getaffinity(0)
        policy, priority = os.sched_getscheduler(0), os.sched_getparam(0)
        try:
            # pthread/std::thread inherits its creator's affinity. Restrict only
            # this Python caller while Franky constructs the native control
            # thread, then return it to the housekeeping pool immediately.
            if cpu is not None:
                os.sched_setaffinity(0, {cpu})
            if self._native_realtime:
                os.sched_setscheduler(0, os.SCHED_FIFO,
                                      os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO)))
            self._arm.move(motion, asynchronous=True)
        finally:
            try:
                if self._native_realtime:
                    os.sched_setscheduler(0, policy, priority)
            finally:
                if cpu is not None:
                    os.sched_setaffinity(0, caller_affinity)

    def _record_motion_event(self, event: str, **fields: Any) -> None:
        self._motion_event_sequence += 1
        self._motion_events.append(
            {
                "event": event,
                "motion_event_sequence": self._motion_event_sequence,
                **fields,
            }
        )

    @staticmethod
    def _state_time_s(state: Any) -> float | None:
        value = getattr(state, "time", None)
        if value is None:
            return None
        to_sec = getattr(value, "to_sec", None)
        try:
            return float(to_sec() if callable(to_sec) else value)
        except (TypeError, ValueError):
            return None

    def _arm_in_control(self) -> bool:
        value = getattr(self._arm, "is_in_control", self._motion_started)
        return bool(value() if callable(value) else value)

    def _state_has_errors(self, _state: Any) -> bool:
        # franky.Errors has no Python __bool__; testing the wrapper object
        # directly would therefore report True even when all flags are clear.
        return self._arm_has_errors()

    def _motion_status_fields(self, state: Any | None = None) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        if state is None:
            try:
                state = self._arm.state
            except BaseException as error:
                fields["state_error"] = repr(error)
        if state is not None:
            current_errors = getattr(state, "current_errors", None)
            if current_errors is not None:
                fields["current_errors"] = repr(current_errors)
            success_rate = getattr(state, "control_command_success_rate", None)
            if success_rate is not None:
                fields["control_command_success_rate"] = float(success_rate)
            state_time_s = self._state_time_s(state)
            if state_time_s is not None:
                fields["robot_state_time_s"] = state_time_s
        try:
            fields["is_in_control"] = self._arm_in_control()
        except BaseException as error:
            fields["control_state_error"] = repr(error)
        return fields

    def communication_debug_snapshot(self) -> dict[str, Any]:
        """Read cached Franky telemetry; never open another FCI connection."""
        with self._arm_lock:
            return self._motion_status_fields()

    def _cleanup_faulted_motion(self, *, source: str) -> None:
        started = time.monotonic()
        try:
            active_before = self._arm_in_control()
        except BaseException as error:
            self._record_motion_event(
                "franky_motion_cleanup_failed",
                source=source,
                operation="is_in_control",
                error=repr(error),
            )
            raise RuntimeError("could not determine whether Franky control is active") from error

        self._record_motion_event(
            "franky_motion_cleanup_started",
            source=source,
            active_before=active_before,
        )
        stop_error: BaseException | None = None
        if active_before:
            try:
                self._arm.stop()
            except BaseException as error:
                stop_error = error
                self._record_motion_event(
                    "franky_motion_cleanup_failed",
                    source=source,
                    operation="stop",
                    error=repr(error),
                )

        joined = False
        join_error: BaseException | None = None
        try:
            joined = bool(self._arm.join_motion(self.config.motion_cleanup_timeout_s))
        except BaseException as error:
            # Franky joins the native thread before rethrowing its stored
            # control exception. Confirm the control state below before
            # deciding whether cleanup itself failed.
            join_error = error
            self._record_motion_event(
                "franky_motion_cleanup_failed",
                source=source,
                operation="join",
                error=repr(error),
            )

        try:
            active_after = self._arm_in_control()
        except BaseException as error:
            self._record_motion_event(
                "franky_motion_cleanup_failed",
                source=source,
                operation="is_in_control_after_join",
                error=repr(error),
            )
            raise RuntimeError("could not confirm Franky motion cleanup") from error
        cleanup_confirmed = not active_after
        self._record_motion_event(
            "franky_motion_cleanup_completed" if cleanup_confirmed else "franky_motion_cleanup_failed",
            source=source,
            operation="complete",
            active_before=active_before,
            active_after=active_after,
            stop_error=None if stop_error is None else repr(stop_error),
            join_result=joined,
            join_error=None if join_error is None else repr(join_error),
            elapsed_ms=(time.monotonic() - started) * 1000.0,
        )
        if not cleanup_confirmed:
            raise RuntimeError(
                "Franky asynchronous motion did not stop within "
                f"{self.config.motion_cleanup_timeout_s:.3f}s"
            )
        self._motion_started = False
        self._motion_mode = None

    def _wait_for_recovered_state(
        self,
        previous_state_time_s: float | None,
        *,
        source: str,
    ) -> tuple[Any, float]:
        started = time.monotonic()
        deadline = started + self.config.motion_recovery_timeout_s
        last_error: BaseException | None = None
        last_state: Any | None = None
        while True:
            try:
                last_state = self._arm.state
                state_time_s = self._state_time_s(last_state)
                state_is_fresh = (
                    previous_state_time_s is None
                    or state_time_s is None
                    or state_time_s > previous_state_time_s
                )
                if state_is_fresh and not self._state_has_errors(last_state):
                    return last_state, time.monotonic() - started
            except BaseException as error:
                last_error = error
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                details = self._motion_status_fields(last_state)
                self._record_motion_event(
                    "franky_motion_recovery_failed",
                    source=source,
                    operation="state_refresh",
                    error=None if last_error is None else repr(last_error),
                    wait_ms=(time.monotonic() - started) * 1000.0,
                    **details,
                )
                raise RuntimeError("Franky remains faulted after motion recovery") from last_error
            time.sleep(min(self.config.motion_recovery_poll_interval_s, remaining))

    def _recover_motion_error(self, error: BaseException, *, source: str, manual: bool = False) -> None:
        self._motion_faulted = True
        if not manual and not self.config.automatic_motion_recovery:
            self._cleanup_faulted_motion(source=source)
            raise RuntimeError(f"Franky {source} fault; manual recovery required: {error}") from error
        initial_status = self._motion_status_fields()
        previous_state_time_s = initial_status.get("robot_state_time_s")
        self._record_motion_event(
            "franky_motion_failed",
            source=source,
            error=repr(error),
            **initial_status,
        )
        log.warning("Franky %s error; attempting bounded recovery: %r", source, error)
        recovery_started = time.monotonic()
        try:
            self._cleanup_faulted_motion(source=source)
        except BaseException as cleanup_error:
            self._record_motion_event(
                "franky_motion_recovery_failed",
                source=source,
                operation="cleanup",
                error=repr(cleanup_error),
            )
            raise RuntimeError(f"Franky motion cleanup failed after: {error}") from cleanup_error
        self._record_motion_event("franky_motion_recovery_started", source=source)
        try:
            recovered = bool(self._arm.recover_from_errors())
        except BaseException as recovery_error:
            self._record_motion_event(
                "franky_motion_recovery_failed",
                source=source,
                operation="recover_from_errors",
                error=repr(recovery_error),
            )
            raise RuntimeError(f"Franky motion recovery failed after: {error}") from recovery_error
        if not recovered:
            self._record_motion_event(
                "franky_motion_recovery_failed",
                source=source,
                operation="recover_from_errors_result",
                recover_result=False,
                **self._motion_status_fields(),
            )
            raise RuntimeError(f"Franky motion recovery reported failure after: {error}")
        try:
            recovered_state, state_wait_s = self._wait_for_recovered_state(
                previous_state_time_s,
                source=source,
            )
        except BaseException as state_error:
            raise RuntimeError(f"Franky remains faulted after motion recovery: {error}") from state_error

        self._motion_faulted = False
        self._last_linear_velocity.fill(0.0)
        self._last_angular_velocity.fill(0.0)
        self._last_smoothed_target = None
        self._last_sent_target = None
        self._consecutive_motion_failures += 1
        self._recovery_since_last_success = True
        self._last_command_debug["consecutive_motion_failures"] = self._consecutive_motion_failures
        if self._consecutive_motion_failures >= self.config.max_consecutive_motion_failures:
            raise RuntimeError(
                f"{self._consecutive_motion_failures} consecutive Franky motion commands failed"
            ) from error
        self._record_motion_event(
            "franky_motion_recovery_completed",
            source=source,
            recover_result=True,
            state_wait_ms=state_wait_s * 1000.0,
            elapsed_ms=(time.monotonic() - recovery_started) * 1000.0,
            **self._motion_status_fields(recovered_state),
        )
        log.info("Franky recovered from %s error; cleared velocity smoothing state", source)

    def _arm_has_errors(self) -> bool:
        has_errors = self._arm.has_errors
        return bool(has_errors() if callable(has_errors) else has_errors)

    def command_debug_snapshot(self) -> dict[str, Any]:
        return dict(self._last_command_debug)

    def observation_debug_snapshot(self) -> dict[str, Any]:
        return dict(self._last_observation_debug)

    def drain_motion_events(self) -> list[dict[str, Any]]:
        with self._arm_lock:
            events = list(self._motion_events)
            self._motion_events.clear()
            return events

    def stop(self) -> None:
        self._target_stop.set()
        thread = self._target_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.config.motion_cleanup_timeout_s)
            if thread.is_alive():
                raise RuntimeError("human servo did not stop; control remains disabled")
        with self._target_lock:
            self._target_command = None
        self._target_thread = None
        self._target_error = None
        self._last_smoothed_target = None
        self._last_sent_target = None
        self._consecutive_motion_failures = 0
        self._recovery_since_last_success = False
        if self._arm is None or not self._motion_started:
            self._motion_mode = None
            return
        error: BaseException | None = None
        with self._arm_lock:
            try:
                import franky

                if self._motion_mode == "joint_velocity":
                    stop_motion = franky.JointVelocityStopMotion()
                else:
                    stop_motion = franky.CartesianVelocityStopMotion()
                self._arm.move(stop_motion)
            except BaseException as caught:
                error = caught
                # A terminal libfranka fault can reject a new stop motion. Fall
                # back to Robot.stop() and consume the asynchronous result.
                try:
                    self._arm.stop()
                except BaseException:
                    pass
                try:
                    self._arm.join_motion(1.0)
                except BaseException:
                    pass
            finally:
                self._motion_started = False
                self._motion_mode = None
                self._last_linear_velocity.fill(0.0)
                self._last_angular_velocity.fill(0.0)
                self._last_smoothed_target = None
                self._last_sent_target = None
                self._consecutive_motion_failures = 0
                self._recovery_since_last_success = False
        if error is not None and not self._motion_faulted:
            raise RuntimeError(f"failed to stop Franky motion cleanly: {error}") from error

    def disconnect(self) -> None:
        gripper_closed = False
        try:
            self.stop()
        finally:
            try:
                self._gripper.close()
                gripper_closed = True
            finally:
                try:
                    if self._camera is not None:
                        self._camera.close()
                finally:
                    if gripper_closed:
                        self._arm = None
                        if self._hardware_lease is not None:
                            self._hardware_lease.close()
                            self._hardware_lease = None
                self._motion_started = False
                self._motion_mode = None
                self._motion_faulted = False
                self._last_linear_velocity.fill(0.0)
                self._last_angular_velocity.fill(0.0)
                self._last_smoothed_target = None
                self._last_sent_target = None
                self._last_command_debug = {}
                self._last_observation_debug = {}
                self._motion_events.clear()
                self._consecutive_motion_failures = 0
                self._recovery_since_last_success = False
                self._commanded_gripper = None
                self._last_gripper_close_time = 0.0
