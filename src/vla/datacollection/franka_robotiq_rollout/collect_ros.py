"""Collect Franka VLA/RLT rollouts with optional gamepad intervention.

Run from the repository root with the ROS2 Python interpreter::

    PYTHONPATH=src:${PYTHONPATH:-} /usr/bin/python3 \
        -m vla.datacollection.franka_robotiq_rollout.collect \
        --config datacollection/franka_robotiq_rollout/configs/single_left.yaml
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import traceback
from collections import Counter
from dataclasses import dataclass

import numpy as np
import PyKDL
import rclpy
from control_msgs.action import GripperCommand
from franka_msgs.msg import FrankaRobotState
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import parameter_value_to_python
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import Float32, Float64MultiArray

from vla.datacollection.franka_robotiq_single_left import schema
from vla.datacollection.franka_robotiq_single_left.auto_record import (
    AsyncVideoWriter,
    KeyboardEvents,
    RealSenseCamera,
    prepare_image,
)
from vla.datacollection.franka_robotiq_single_left.v30_writer import V30DatasetWriter
from vla.datacollection.start_teleop.franka_kinematics import (
    build_kinematic_model,
    joint7_step,
)
from .config import ArmConfig, RolloutConfig, load_rollout_config
from .gamepad import decode_gamepad_values, gripper_direction_transition
from .gamepad import gamepad_is_neutral
from .policy_client import AsyncPolicyClient, PolicyResult
from .safety import (
    FrankaHealthMonitor,
    action_from_joint_command,
    clamp_gripper_action_roundoff,
    clamp_pose_step,
    compose_pose,
    flange_pose_from_tcp,
    gamepad_tracking_twist,
    policy_state_from_raw,
    pose_step_to_twist,
    smooth_pose_target,
    validate_absolute_action,
    validate_rollout_episode,
)


COLLECTOR_HUMAN = 0
COLLECTOR_PI05 = 1
COLLECTOR_RLT_ACTOR = 2
COLLECTOR_NAMES = {
    COLLECTOR_HUMAN: "HUMAN",
    COLLECTOR_PI05: "VLA",
    COLLECTOR_RLT_ACTOR: "RLT_ACTOR",
}
MIN_EPISODE_FRAMES = 12

EXTRA_FEATURES = {
    "complementary_info.is_intervention": {
        "dtype": "float32",
        "shape": [1],
        "names": ["is_intervention"],
    },
    "complementary_info.phase": {
        "dtype": "float32",
        "shape": [1],
        "names": ["phase"],
    },
    "complementary_info.collector_policy_id": {
        "dtype": "int64",
        "shape": [1],
        "names": ["collector_policy_id"],
        "info": {"codebook": {"0": "human", "1": "pi0.5", "2": "pi_rlt_actor"}},
    },
}


def _prefixed_fields(arms: tuple[ArmConfig, ...], fields: list[str]) -> tuple[str, ...]:
    if len(arms) == 1:
        return tuple(fields)
    return tuple(f"{arm.name}_{field}" for arm in arms for field in fields)


def _clamp_norm(values: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm <= maximum or norm == 0.0:
        return values
    return values * (maximum / norm)


def _gripper_width(message: JointState, joint_name: str) -> float:
    return schema.gripper_width(message, joint_name)


def _gripper_open_fraction(message: JointState, joint_name: str) -> float:
    return float(
        np.clip(
            _gripper_width(message, joint_name) / schema.ROBOTIQ_MAX_OPENING_WIDTH_M,
            0.0,
            1.0,
        )
    )


def _state_part(
    robot_state: FrankaRobotState,
    gripper_state: JointState,
    gripper_joint_name: str,
    velocity: np.ndarray,
    joint_positions: np.ndarray,
) -> np.ndarray:
    return schema.state_vector(
        robot_state,
        gripper_state,
        velocity,
        gripper_joint_name=gripper_joint_name,
        joint_positions=joint_positions,
    )


@dataclass
class GamepadTarget:
    position: np.ndarray
    rotation: np.ndarray
    gripper: float
    joints: np.ndarray
    flange_to_tcp_position: np.ndarray
    flange_to_tcp_rotation: np.ndarray
    gripper_direction: int = 0


@dataclass(frozen=True)
class ArmCommandSnapshot:
    joints: np.ndarray
    gripper: float


@dataclass(frozen=True)
class CommandSnapshot:
    arms: dict[str, ArmCommandSnapshot]
    source: int
    phase: float
    published_at_ns: int


class CartesianServo:
    def __init__(self, model, config: RolloutConfig) -> None:
        self.model = model
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(model.chain)
        self.jacobian_solver = PyKDL.ChainJntToJacSolver(model.chain)
        self.config = config
        self._last_smoothed_position: np.ndarray | None = None
        self._last_smoothed_rotation: np.ndarray | None = None
        self._last_linear_velocity = np.zeros(3, dtype=np.float64)
        self._last_angular_velocity = np.zeros(3, dtype=np.float64)

    def reset_tracking(self) -> None:
        self._last_smoothed_position = None
        self._last_smoothed_rotation = None
        self._last_linear_velocity.fill(0.0)
        self._last_angular_velocity.fill(0.0)

    def _joints(self, values: np.ndarray) -> PyKDL.JntArray:
        joints = PyKDL.JntArray(7)
        for index, value in enumerate(values):
            joints[index] = float(value)
        return joints

    def forward(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the base-to-flange pose for the configured KDL chain."""
        frame = PyKDL.Frame()
        if self.fk_solver.JntToCart(self._joints(joints), frame) < 0:
            raise RuntimeError("KDL forward kinematics failed")
        position = np.array([frame.p.x(), frame.p.y(), frame.p.z()])
        rotation = np.array([[frame.M[row, col] for col in range(3)] for row in range(3)])
        return position, rotation

    def forward_tcp(
        self,
        joints: np.ndarray,
        flange_to_tcp_position: np.ndarray,
        flange_to_tcp_rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        flange_position, flange_rotation = self.forward(joints)
        return compose_pose(
            flange_position,
            flange_rotation,
            flange_to_tcp_position,
            flange_to_tcp_rotation,
        )

    def step(
        self,
        joints: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        flange_to_tcp_position: np.ndarray,
        flange_to_tcp_rotation: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        current_flange_position, current_flange_rotation = self.forward(joints)
        current_tcp_position, current_tcp_rotation = compose_pose(
            current_flange_position,
            current_flange_rotation,
            flange_to_tcp_position,
            flange_to_tcp_rotation,
        )
        limited_tcp_position, limited_tcp_rotation = clamp_pose_step(
            current_tcp_position,
            current_tcp_rotation,
            target_position,
            target_rotation,
            max_linear_step=(
                self.config.max_linear_velocity * self.config.velocity_tracking_tau_s
            ),
            max_angular_step=(
                self.config.max_angular_velocity * self.config.velocity_tracking_tau_s
            ),
        )
        if self._last_smoothed_position is not None:
            limited_tcp_position, limited_tcp_rotation = smooth_pose_target(
                self._last_smoothed_position,
                self._last_smoothed_rotation,
                limited_tcp_position,
                limited_tcp_rotation,
                alpha=self.config.target_smoothing_alpha,
            )
        self._last_smoothed_position = limited_tcp_position.copy()
        self._last_smoothed_rotation = limited_tcp_rotation.copy()
        target_flange_position, target_flange_rotation = flange_pose_from_tcp(
            limited_tcp_position,
            limited_tcp_rotation,
            flange_to_tcp_position,
            flange_to_tcp_rotation,
        )
        linear, angular = pose_step_to_twist(
            current_flange_position,
            current_flange_rotation,
            target_flange_position,
            target_flange_rotation,
            tracking_tau_s=self.config.velocity_tracking_tau_s,
        )
        alpha = self.config.velocity_smoothing
        linear = alpha * self._last_linear_velocity + (1.0 - alpha) * linear
        if self.config.smooth_angular_velocity:
            angular = alpha * self._last_angular_velocity + (1.0 - alpha) * angular
        linear = _clamp_norm(linear, self.config.max_linear_velocity)
        angular = _clamp_norm(angular, self.config.max_angular_velocity)
        self._last_linear_velocity = linear.copy()
        self._last_angular_velocity = angular.copy()
        return self._joint_step(joints, np.concatenate((linear, angular)), dt)

    def step_gamepad(
        self,
        joints: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        flange_to_tcp_position: np.ndarray,
        flange_to_tcp_rotation: np.ndarray,
        target_velocity: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Match FineTeleopNode's command-state Cartesian servo."""
        current_flange_position, current_flange_rotation = self.forward(joints)
        target_flange_position, target_flange_rotation = flange_pose_from_tcp(
            target_position,
            target_rotation,
            flange_to_tcp_position,
            flange_to_tcp_rotation,
        )
        linear, angular = gamepad_tracking_twist(
            current_flange_position,
            current_flange_rotation,
            target_flange_position,
            target_flange_rotation,
            target_velocity,
            position_gain=self.config.gamepad_position_gain,
            orientation_gain=self.config.gamepad_orientation_gain,
            max_linear_speed=2.0 * self.config.gamepad_xy_speed,
            max_angular_speed=self.config.gamepad_max_angular_speed,
        )
        return self._joint_step(joints, np.concatenate((linear, angular)), dt)

    def _joint_step(self, joints: np.ndarray, twist: np.ndarray, dt: float) -> np.ndarray:
        jacobian_kdl = PyKDL.Jacobian(7)
        if self.jacobian_solver.JntToJac(self._joints(joints), jacobian_kdl) < 0:
            raise RuntimeError("KDL Jacobian calculation failed")
        jacobian = np.array(
            [[jacobian_kdl[row, column] for column in range(7)] for row in range(6)]
        )
        regularized = jacobian @ jacobian.T + self.config.damping**2 * np.eye(6)
        velocity = jacobian.T @ np.linalg.solve(regularized, twist)
        velocity = np.clip(velocity, -self.config.max_joint_speed, self.config.max_joint_speed)
        return np.clip(
            joints + velocity * dt,
            self.model.lower_limits,
            self.model.upper_limits,
        )


class RolloutCollector(Node):
    def __init__(self, config: RolloutConfig, *, demonstration: bool = False) -> None:
        if demonstration:
            from vla.datacollection.franka_robotiq_dual.gamepad_record import validate_dual_config
            validate_dual_config(config)
        super().__init__("vla_franka_rollout_collector")
        self.config = config
        self.demonstration = demonstration
        self._teleop_enabled = False
        self._subtask_index = 0
        self._episode_task_index = 0
        self._last_demo_images: dict[str, int] = {}
        self._extra_features = (
            {"complementary_info.phase": EXTRA_FEATURES["complementary_info.phase"]}
            if demonstration else dict(EXTRA_FEATURES)
        )
        if config.subtasks:
            self._extra_features["complementary_info.subtask_index"] = {
                "dtype": "int64", "shape": [1], "names": ["subtask_index"],
                "info": {"codebook": {str(i): name for i, name in enumerate(config.subtasks)}},
            }
        self._lock = threading.RLock()
        self._latest: dict[str, tuple[int, object]] = {}
        self._cameras: dict[str, RealSenseCamera] = {}
        self._video_writers: dict[str, AsyncVideoWriter] = {}
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._annotations = {key: [] for key in self._extra_features}
        self._frame_times: list[int] = []
        self._skip_reasons = Counter()
        self._last_pose: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        self._last_command_snapshot: CommandSnapshot | None = None
        self._latest_policy_result: PolicyResult | None = None
        self._policy_action: np.ndarray | None = None
        self._reset_policy_on_request = True
        self._intervention = False
        self._gamepad_targets: dict[str, GamepadTarget] = {}
        self._gamepad_stale_reported = False
        self._last_gamepad_tick = time.monotonic()
        self._last_gamepad_input_log = 0.0
        self._phase = "vla"
        self._last_applied_control: tuple[int, str] | None = None
        self._policy_responses_in_phase = 0
        self._log_next_policy_request = True
        self._recording = False
        self._closed = False
        self._faulted = False
        self._fault_reason: str | None = None
        self._gripper_goal_futures: set[object] = set()

        self.policy_client = None
        if not demonstration:
            self.policy_client = AsyncPolicyClient(config.socket_path, config.policy_timeout_s)
            self._validate_policy_contract()
        metadata = {
            "recording_schema_version": 3,
            "rlt_episode_metadata_fields": {
                "rl_intervals": "RL frame intervals with episode outcome.",
                "human_intervention_intervals": "Human intervention frame intervals.",
            },
        }
        if demonstration:
            from vla.datacollection.franka_robotiq_dual.gamepad_record import recording_contract
            metadata.update(collection_kind="human_demonstration", gamepad_recording_version=1,
                            recording_contract=recording_contract(config))
        else:
            metadata["policy_checkpoint_fingerprint"] = self.policy_client.contract["checkpoint_fingerprint"]
        if config.subtasks:
            metadata["subtasks"] = list(config.subtasks)
        if demonstration:
            metadata["multi_task"] = len(config.tasks) > 1
        self._health_monitor = FrankaHealthMonitor(
            config.min_control_command_success_rate,
            config.low_success_rate_samples,
        )
        self.writer = V30DatasetWriter(
            config.dataset_root,
            config.fps,
            config.task,
            tuple(camera.feature_key for camera in config.cameras),
            *config.image_size,
            state_fields=_prefixed_fields(config.arms, schema.STATE_FIELDS),
            action_fields=_prefixed_fields(config.arms, schema.ACTION_FIELDS),
            extra_features=self._extra_features,
            robot_type="franka_robotiq_dual" if len(config.arms) == 2 else "franka_robotiq",
            info_metadata=metadata,
            strict_metadata=True,
            allow_multiple_tasks=demonstration and len(config.tasks) > 1,
        )
        self.servos = {
            arm.name: CartesianServo(self._load_kinematic_model(arm), config)
            for arm in config.arms
        }
        self._setup_ros_io()
        self._start_cameras()
        # Poll fresh base frames frequently; demonstrations must not duplicate
        # one camera exposure just because the 30 Hz timer and camera drift.
        self.create_timer(1.0 / (config.gamepad_control_rate if demonstration else config.fps), self._tick)
        self.create_timer(1.0 / config.gamepad_control_rate, self._gamepad_tick)
        control_profile = (
            str(config.rtc_deployment_config)
            if config.rtc_deployment_config is not None
            else "rollout YAML"
        )
        print(
            "Cartesian control: "
            f"source={control_profile}, linear={config.max_linear_velocity:.3f}m/s, "
            f"angular={config.max_angular_velocity:.3f}rad/s, "
            f"tracking_tau={config.velocity_tracking_tau_s:.3f}s, "
            f"target_alpha={config.target_smoothing_alpha:.2f}, "
            f"velocity_alpha={config.velocity_smoothing:.2f}",
            flush=True,
        )
        print(
            "Gamepad control: "
            f"rate={config.gamepad_control_rate:.0f}Hz, "
            f"xy={config.gamepad_xy_speed:.3f}m/s, z={config.gamepad_z_speed:.3f}m/s, "
            f"position_gain={config.gamepad_position_gain:.1f}",
            flush=True,
        )
        print(f"Rollout collector ready; dataset={config.dataset_root}", flush=True)

    def _validate_policy_contract(self) -> None:
        contract = self.policy_client.contract
        expected_cameras = [camera.feature_key for camera in self.config.cameras]
        expected_robot_type = (
            "franka_robotiq_dual" if len(self.config.arms) == 2 else "franka_robotiq"
        )
        expected_state_names = list(_prefixed_fields(self.config.arms, schema.TCP_STATE_FIELDS))
        expected_action_names = list(_prefixed_fields(self.config.arms, schema.ACTION_FIELDS))
        expected_camera_shapes = {
            key: [*self.config.image_size, 3] for key in expected_cameras
        }
        mismatches = []
        if int(contract["state_dim"]) != self.config.policy_state_dim:
            mismatches.append(f"state_dim server={contract['state_dim']} local={self.config.policy_state_dim}")
        if int(contract["action_dim"]) != self.config.action_dim:
            mismatches.append(f"action_dim server={contract['action_dim']} local={self.config.action_dim}")
        if list(contract["camera_keys"]) != expected_cameras:
            mismatches.append(f"cameras server={contract['camera_keys']} local={expected_cameras}")
        if contract.get("camera_shapes_hwc") != expected_camera_shapes:
            mismatches.append(
                f"camera_shapes server={contract.get('camera_shapes_hwc')} "
                f"local={expected_camera_shapes}"
            )
        if contract.get("state_names") != expected_state_names:
            mismatches.append("state feature names/order do not match")
        if contract.get("action_names") != expected_action_names:
            mismatches.append("action feature names/order do not match")
        if contract.get("robot_type") != expected_robot_type:
            mismatches.append(
                f"robot_type server={contract.get('robot_type')!r} local={expected_robot_type!r}"
            )
        server_tasks = tuple(contract.get("tasks", [contract.get("task", "")]))
        if tuple(self.config.tasks) != server_tasks:
            mismatches.append(
                f"tasks server={server_tasks!r} local={tuple(self.config.tasks)!r}"
            )
        if float(contract.get("fps", -1)) != float(self.config.fps):
            mismatches.append(f"fps server={contract.get('fps')} local={self.config.fps}")
        if contract.get("actions_stored_as") != "absolute":
            mismatches.append(
                f"checkpoint actions_stored_as={contract.get('actions_stored_as')!r}, expected 'absolute'"
            )
        if bool(contract.get("use_relative_actions", True)):
            mismatches.append("checkpoint enables relative actions; rollout requires absolute actions")
        if mismatches:
            raise ValueError("policy rollout contract mismatch: " + "; ".join(mismatches))

    def _load_kinematic_model(self, arm: ArmConfig):
        service_name = f"{arm.robot_state_publisher.rstrip('/')}/get_parameters"
        client = self.create_client(GetParameters, service_name)
        if not client.wait_for_service(timeout_sec=30.0):
            raise RuntimeError(f"parameter service unavailable: {service_name}")
        request = GetParameters.Request()
        request.names = ["robot_description"]
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        response = future.result()
        if response is None or not response.values:
            raise RuntimeError(f"failed to read robot_description from {service_name}")
        description = parameter_value_to_python(response.values[0])
        return build_kinematic_model(
            description,
            arm.base_link,
            arm.tip_link,
            self.config.joint_limit_margin,
        )

    def _setup_ros_io(self) -> None:
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._joint_publishers = {}
        self._gripper_publishers = {}
        self._gripper_speed_publishers = {}
        self._gripper_action_clients = {}
        self._subscription_handles = []
        for arm in self.config.arms:
            prefix = arm.name
            subscriptions = (
                (FrankaRobotState, arm.robot_state_topic, f"{prefix}.robot_state"),
                (JointState, arm.measured_joint_topic, f"{prefix}.measured_joint"),
                (JointState, arm.gripper_state_topic, f"{prefix}.gripper_state"),
                (Joy, arm.gamepad.joy_topic, f"{prefix}.gamepad"),
            )
            for message_type, topic, key in subscriptions:
                self._subscription_handles.append(
                    self.create_subscription(message_type, topic, self._callback(key), qos)
                )
            self._joint_publishers[prefix] = self.create_publisher(
                JointState, arm.command_joint_topic, 10
            )
            self._gripper_publishers[prefix] = self.create_publisher(
                Float32, arm.command_gripper_topic, 10
            )
            self._gripper_speed_publishers[prefix] = self.create_publisher(
                Float64MultiArray, arm.gripper_speed_topic, 10
            )
            self._gripper_action_clients[prefix] = ActionClient(
                self,
                GripperCommand,
                arm.gripper_action_topic,
            )

    def _start_cameras(self) -> None:
        started = []
        try:
            for camera in self.config.cameras:
                instance = RealSenseCamera(camera.name, camera.serial, self._camera_callback)
                instance.start()
                instance.wait_ready(10.0)
                self._cameras[camera.name] = instance
                started.append(instance)
        except BaseException:
            for camera in reversed(started):
                camera.close()
            raise

    def _callback(self, key: str):
        return lambda message: self._store_latest(key, message)

    def _store_latest(self, key: str, value: object, timestamp_ns: int | None = None) -> None:
        with self._lock:
            self._latest[key] = (timestamp_ns or time.monotonic_ns(), value)

    def _camera_callback(self, role: str, timestamp_ns: int, rgb: np.ndarray) -> None:
        height, width = self.config.image_size
        prepared = prepare_image(rgb)
        if prepared.shape[:2] != (height, width):
            import cv2

            prepared = cv2.resize(prepared, (width, height), interpolation=cv2.INTER_AREA)
        self._store_latest(f"camera.{role}", prepared, timestamp_ns)

    def _snapshot(
        self, *, raise_on_invalid: bool = False
    ) -> tuple[dict[str, object], dict[str, np.ndarray]] | None:
        now_ns = time.monotonic_ns()
        maximum_age_ns = int(self.config.sync_tolerance_ms * 1_000_000)
        required = []
        for arm in self.config.arms:
            required.extend(
                (
                    f"{arm.name}.robot_state",
                    f"{arm.name}.measured_joint",
                    f"{arm.name}.gripper_state",
                )
            )
        required.extend(f"camera.{camera.name}" for camera in self.config.cameras)
        values: dict[str, object] = {}
        timestamps: list[int] = []
        for key in required:
            item = self._latest.get(key)
            if item is None:
                self._skip_reasons[f"missing {key}"] += 1
                if raise_on_invalid:
                    raise RuntimeError(f"required rollout input is missing: {key}")
                return None
            timestamp_ns, value = item
            if now_ns - timestamp_ns > maximum_age_ns:
                self._skip_reasons[f"stale {key}"] += 1
                if raise_on_invalid:
                    age_ms = (now_ns - timestamp_ns) / 1_000_000.0
                    raise RuntimeError(f"rollout input is stale: {key}, age={age_ms:.1f}ms")
                return None
            values[key] = value
            timestamps.append(timestamp_ns)
        skew_ns = max(timestamps) - min(timestamps)
        if skew_ns > maximum_age_ns:
            self._skip_reasons["inputs exceed cross-source sync tolerance"] += 1
            if raise_on_invalid:
                raise RuntimeError(
                    "rollout inputs exceed cross-source sync tolerance: "
                    f"skew={skew_ns / 1_000_000.0:.1f}ms"
                )
            return None
        images = {
            camera.feature_key: np.asarray(values[f"camera.{camera.name}"])
            for camera in self.config.cameras
        }
        expected_shape = (*self.config.image_size, 3)
        for key, image in images.items():
            if image.shape != expected_shape or image.dtype != np.uint8:
                raise ValueError(
                    f"{key} must be uint8 RGB with shape {expected_shape}, "
                    f"got {image.shape}/{image.dtype}"
                )
        return values, images

    def _measured_joints(self, arm: ArmConfig, message: JointState) -> np.ndarray:
        model = self.servos[arm.name].model
        positions = dict(zip(message.name, message.position))
        if all(name in positions for name in model.joint_names):
            result = np.array([positions[name] for name in model.joint_names], dtype=float)
        elif len(message.position) >= 7:
            result = np.asarray(message.position[:7], dtype=float)
        else:
            raise ValueError(f"measured joints missing for {arm.name}")
        if result.shape != (7,) or not np.isfinite(result).all():
            raise ValueError(f"measured joints are invalid for {arm.name}")
        return result

    def _raw_state(self, values: dict[str, object], now_ns: int) -> np.ndarray:
        parts = []
        for arm in self.config.arms:
            robot_state = values[f"{arm.name}.robot_state"]
            position, rotation = schema.measured_pose_from_robot_state(robot_state)
            previous = self._last_pose.get(arm.name)
            velocity = schema.base_tcp_velocity(
                previous[0] if previous else None,
                previous[1] if previous else None,
                previous[2] if previous else None,
                position,
                rotation,
                now_ns,
            )
            self._last_pose[arm.name] = (position.copy(), rotation.copy(), now_ns)
            parts.append(
                _state_part(
                    robot_state,
                    values[f"{arm.name}.gripper_state"],
                    arm.gripper_joint_name,
                    velocity,
                    self._measured_joints(arm, values[f"{arm.name}.measured_joint"]),
                )
            )
        return np.concatenate(parts)

    def _policy_state(self, raw_state: np.ndarray) -> np.ndarray:
        return policy_state_from_raw(raw_state, len(self.config.arms))

    def _reset_servo_tracking(self) -> None:
        for servo in self.servos.values():
            servo.reset_tracking()

    def _invalidate_policy(self) -> None:
        if self.policy_client is not None:
            self.policy_client.invalidate()

    def _neutral_gamepads(self) -> bool:
        values, problem = self._gamepad_control_snapshot()
        if values is None:
            print(f"Release/start blocked: {problem}", flush=True)
            return False
        return all(gamepad_is_neutral(values[f"{arm.name}.gamepad"].axes,
                                      values[f"{arm.name}.gamepad"].buttons,
                                      arm.gamepad, self.config) for arm in self.config.arms)

    def enable_teleop(self) -> None:
        """Enable both gamepads together for unrecorded demonstration positioning."""
        with self._lock:
            if not self.demonstration or self._recording or self._faulted or self._teleop_enabled:
                return
            problems = self._preflight_problems()
            if problems:
                print(f"Cannot enable: {'; '.join(problems)}", flush=True)
                return
            if not self._neutral_gamepads():
                print("Cannot enable: check robot/cameras/Joy and release BOTH gamepads", flush=True)
                return
            self._teleop_enabled = True
            self.toggle_intervention()
            if not self._intervention:
                self._teleop_enabled = False

    def pause_teleop(self) -> None:
        with self._lock:
            if self._recording:
                self.finish_episode(None, discard=True)
            self._stop_grippers_at_measured_position()
            self._teleop_enabled = False
            self._intervention = False
            self._gamepad_targets = {}
            self._last_command_snapshot = None
            self._reset_servo_tracking()

    def next_subtask(self) -> None:
        with self._lock:
            if not self._recording or not self.config.subtasks:
                return
            # A stage transition is annotation only: never change prompt, phase,
            # control ownership, policy queue or reward/terminal semantics.
            if self._subtask_index + 1 < len(self.config.subtasks):
                self._subtask_index += 1
            print(f"Subtask {self._subtask_index}: {self.config.subtasks[self._subtask_index]}", flush=True)

    def next_task(self) -> None:
        """Select the next complete-task prompt before starting an episode."""
        with self._lock:
            if self._recording or len(self.config.tasks) < 2:
                return
            self._episode_task_index = (self._episode_task_index + 1) % len(self.config.tasks)
            print(f"Task {self._episode_task_index}: {self.config.tasks[self._episode_task_index]!r}", flush=True)

    def _actual_action(
        self,
        values: dict[str, object],
        snapshot: CommandSnapshot,
    ) -> np.ndarray:
        parts = []
        for arm in self.config.arms:
            command = snapshot.arms[arm.name]
            parts.append(
                action_from_joint_command(
                    command.joints,
                    command.gripper,
                    values[f"{arm.name}.robot_state"],
                    self.servos[arm.name].forward,
                )
            )
        return np.concatenate(parts)

    def _record_frame(
        self,
        raw_state: np.ndarray,
        values: dict[str, object],
        images: dict[str, np.ndarray],
        now_ns: int,
    ) -> None:
        snapshot = self._last_command_snapshot
        if not self._recording or snapshot is None:
            return
        if self.demonstration and now_ns - snapshot.published_at_ns > int(self.config.control_discontinuity_timeout_s * 1e9):
            raise RuntimeError("demonstration command stream is stale")
        action = self._actual_action(values, snapshot)
        for key, writer in self._video_writers.items():
            writer.write(images[key])
        self._states.append(raw_state)
        self._actions.append(action)
        self._annotations["complementary_info.phase"].append([snapshot.phase])
        if not self.demonstration:
            self._annotations["complementary_info.is_intervention"].append(
                [float(snapshot.source == COLLECTOR_HUMAN)])
            self._annotations["complementary_info.collector_policy_id"].append([snapshot.source])
        if self.config.subtasks:
            self._annotations["complementary_info.subtask_index"].append([self._subtask_index])
        self._frame_times.append(now_ns)

    def set_control_mode(self, mode: str) -> None:
        """Called on the coordinator thread, between command ticks."""
        self.policy_client.set_control_mode(mode)
        self._web_mode = mode
        self._web_switch_pending = True
        self._latest_policy_result = None
        self._reset_policy_on_request = True
        self._policy_action = None
        self._reset_servo_tracking()
        if not self._intervention and self._recording:
            self._republish_arm_hold()
            self._stop_grippers_at_measured_position()

    def _consume_policy_result(self) -> None:
        if self.policy_client is None:
            return
        result = self.policy_client.poll_latest()
        if result is None:
            return
        if self._intervention:
            return
        managed = getattr(result, "control_mode", None) is not None
        if managed and result.mode_version != self.policy_client.mode_version:
            return
        if not managed and result.requested_phase != self._phase:
            print(
                "Discarded policy response after local phase change "
                f"id={result.request_id} requested_phase={result.requested_phase} "
                f"current_phase={self._phase}",
                flush=True,
            )
            return
        returned_phase = "rl" if result.phase > 0.5 else "vla"
        if getattr(result, "control_mode", None) != "auto" and returned_phase != result.requested_phase:
            raise RuntimeError(
                "policy phase mismatch: "
                f"id={result.request_id} requested={result.requested_phase} "
                f"returned={returned_phase} source_type={result.source_type}"
            )
        if getattr(result, "control_mode", None) != "auto" and result.requested_phase == "vla" and result.source_type > 0.5:
            raise RuntimeError(
                "policy source mismatch: "
                f"id={result.request_id} requested=vla but source_type={result.source_type}"
            )
        action = clamp_gripper_action_roundoff(
            result.action,
            arm_count=len(self.config.arms),
        )
        validate_absolute_action(
            action,
            arm_count=len(self.config.arms),
            workspace_bounds=[
                (np.asarray(arm.workspace_min), np.asarray(arm.workspace_max))
                for arm in self.config.arms
            ],
        )
        self._latest_policy_result = result
        if managed:
            self._web_switch_pending = False
        self._policy_action = action

    def _publish_policy(self, values: dict[str, object], dt: float) -> bool:
        if self._policy_action is None or self._latest_policy_result is None:
            return False
        if time.monotonic() - self._latest_policy_result.received_at > self.config.policy_timeout_s:
            raise RuntimeError("latest policy action exceeded policy_timeout_s")
        decoded = validate_absolute_action(
            self._policy_action,
            arm_count=len(self.config.arms),
            workspace_bounds=[
                (np.asarray(arm.workspace_min), np.asarray(arm.workspace_max))
                for arm in self.config.arms
            ],
        )
        commands: dict[str, tuple[np.ndarray, float]] = {}
        action_diagnostics = []
        for arm, (position, rotation, gripper) in zip(self.config.arms, decoded):
            measured = self._measured_joints(arm, values[f"{arm.name}.measured_joint"])
            tool_position, tool_rotation = schema.pose_from_stamped(
                values[f"{arm.name}.robot_state"].f_t_ee
            )
            command = self.servos[arm.name].step(
                measured,
                position,
                rotation,
                tool_position,
                tool_rotation,
                dt,
            )
            commands[arm.name] = (command, gripper)
            action_diagnostics.append(
                f"{arm.name}_xyz={np.round(position, 4).tolist()} "
                f"{arm.name}_gripper={gripper:.4f} "
                f"{arm.name}_max_abs_dq={np.max(np.abs(command - measured)):.6f}"
            )
        source_type = self._latest_policy_result.source_type
        source = (
            COLLECTOR_RLT_ACTOR if source_type > 0.5 else COLLECTOR_PI05
        )
        self._publish_commands(commands, source, self._latest_policy_result.phase)
        self._policy_responses_in_phase += 1
        response_index = self._policy_responses_in_phase
        if response_index <= 3 or response_index % 30 == 0:
            result = self._latest_policy_result
            returned_phase = "rl" if result.phase > 0.5 else "vla"
            print(
                "Policy action published "
                f"id={result.request_id} requested_phase={result.requested_phase} "
                f"returned_phase={returned_phase} "
                f"source={COLLECTOR_NAMES[int(source)]} "
                f"latency_ms={result.latency_s * 1000.0:.1f} "
                + " ".join(action_diagnostics),
                flush=True,
            )
        self._policy_action = None
        return True

    def _publish_gripper_speed(self, arm: ArmConfig, speed: float) -> None:
        message = Float64MultiArray()
        message.data = [float(np.clip(speed, 0.0, 1.0))]
        self._gripper_speed_publishers[arm.name].publish(message)

    def _send_gripper_goal(self, arm: ArmConfig, open_fraction: float) -> None:
        fraction = float(np.clip(open_fraction, 0.0, 1.0))
        goal = GripperCommand.Goal()
        goal.command.position = (
            arm.gripper_closed_position
            + fraction * (arm.gripper_open_position - arm.gripper_closed_position)
        )
        goal.command.max_effort = arm.gripper_max_effort
        future = self._gripper_action_clients[arm.name].send_goal_async(goal)
        self._gripper_goal_futures.add(future)
        future.add_done_callback(
            lambda finished, arm_name=arm.name: self._gripper_goal_response(
                arm_name, finished
            )
        )

    def _gripper_goal_response(self, arm_name: str, future: object) -> None:
        self._gripper_goal_futures.discard(future)
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                raise RuntimeError(f"{arm_name} Robotiq goal was rejected")
            result_future = goal_handle.get_result_async()
            self._gripper_goal_futures.add(result_future)
            result_future.add_done_callback(
                lambda finished, name=arm_name: self._gripper_goal_result(
                    name, finished
                )
            )
        except Exception as error:
            print(f"{arm_name} Robotiq goal failed: {error}", file=sys.stderr, flush=True)

    def _gripper_goal_result(self, arm_name: str, future: object) -> None:
        self._gripper_goal_futures.discard(future)
        try:
            future.result()
        except Exception as error:
            print(f"{arm_name} Robotiq action failed: {error}", file=sys.stderr, flush=True)

    def _latest_gripper_fraction(self, arm: ArmConfig) -> float | None:
        item = self._latest.get(f"{arm.name}.gripper_state")
        if item is None:
            return None
        return _gripper_open_fraction(item[1], arm.gripper_joint_name)

    def _update_gamepad_gripper(
        self,
        arm: ArmConfig,
        target: GamepadTarget,
        measured_open_fraction: float,
        motion: float,
    ) -> None:
        self._publish_gripper_speed(
            arm,
            abs(motion) * self.config.gamepad_gripper_speed,
        )
        direction, goal = gripper_direction_transition(
            target.gripper_direction,
            motion,
            measured_open_fraction,
        )
        target.gripper_direction = direction
        target.gripper = float(np.clip(measured_open_fraction, 0.0, 1.0))
        if goal is not None:
            self._send_gripper_goal(arm, goal)

    def _stop_grippers_at_measured_position(self) -> None:
        for arm in self.config.arms:
            self._publish_gripper_speed(arm, 0.0)
            measured = self._latest_gripper_fraction(arm)
            if measured is not None:
                self._send_gripper_goal(arm, measured)
            target = self._gamepad_targets.get(arm.name)
            if target is not None:
                target.gripper_direction = 0
                if measured is not None:
                    target.gripper = measured

    def _restore_policy_gripper_speed(self) -> None:
        for arm in self.config.arms:
            self._publish_gripper_speed(arm, self.config.policy_gripper_speed)

    def _publish_gamepad(self, values: dict[str, object], dt: float) -> bool:
        commands: dict[str, tuple[np.ndarray, float]] = {}
        active_inputs = []
        for arm in self.config.arms:
            target = self._gamepad_targets.get(arm.name)
            if target is None:
                return False
            gamepad_message = values[f"{arm.name}.gamepad"]
            velocity, x_button, b_button, gripper_motion = decode_gamepad_values(
                gamepad_message.axes,
                gamepad_message.buttons,
                arm.gamepad,
                self.config,
            )
            joint7_motion = x_button - b_button
            if np.linalg.norm(velocity) > 0.0 or joint7_motion != 0.0 or gripper_motion != 0.0:
                active_inputs.append(
                    f"{arm.name}:linear={np.round(velocity, 4).tolist()},"
                    f"j7={joint7_motion:+.0f},gripper={gripper_motion:+.2f}"
                )
            target.position = np.clip(
                target.position + velocity * dt,
                np.asarray(arm.workspace_min),
                np.asarray(arm.workspace_max),
            )
            joint7_target = joint7_step(
                target.joints[6],
                x_button,
                b_button,
                self.config.gamepad_joint7_speed,
                arm.gamepad.joint7_button_sign,
                dt,
                self.servos[arm.name].model.lower_limits[6],
                self.servos[arm.name].model.upper_limits[6],
            )
            command = self.servos[arm.name].step_gamepad(
                target.joints,
                target.position,
                target.rotation,
                target.flange_to_tcp_position,
                target.flange_to_tcp_rotation,
                velocity,
                dt,
            )
            if x_button != b_button:
                command[6] = joint7_target
                _, target.rotation = self.servos[arm.name].forward_tcp(
                    command,
                    target.flange_to_tcp_position,
                    target.flange_to_tcp_rotation,
                )
            target.joints = command.copy()
            measured_gripper = _gripper_open_fraction(
                values[f"{arm.name}.gripper_state"],
                arm.gripper_joint_name,
            )
            self._update_gamepad_gripper(
                arm,
                target,
                measured_gripper,
                gripper_motion,
            )
            commands[arm.name] = (command, target.gripper)
        self._publish_commands(
            commands,
            COLLECTOR_HUMAN,
            float(self._phase == "rl"),
            publish_gripper=False,
        )
        now = time.monotonic()
        if active_inputs and now - self._last_gamepad_input_log >= 0.5:
            print("Gamepad input " + "; ".join(active_inputs), flush=True)
            self._last_gamepad_input_log = now
        return True

    def _gamepad_control_snapshot(self) -> tuple[dict[str, object] | None, str | None]:
        now_ns = time.monotonic_ns()
        values: dict[str, object] = {}
        for arm in self.config.arms:
            required = (
                (f"{arm.name}.measured_joint", self.config.gamepad_state_timeout_s),
                (f"{arm.name}.gripper_state", self.config.gamepad_state_timeout_s),
                (f"{arm.name}.gamepad", self.config.gamepad_joy_timeout_s),
            )
            for key, timeout_s in required:
                item = self._latest.get(key)
                if item is None:
                    return None, f"missing {key}"
                timestamp_ns, value = item
                age_s = (now_ns - timestamp_ns) / 1_000_000_000.0
                if age_s > timeout_s:
                    return None, f"stale {key}, age={age_s:.3f}s"
                values[key] = value
        return values, None

    def _publish_gamepad_hold(self) -> None:
        if set(self._gamepad_targets) != {arm.name for arm in self.config.arms}:
            return
        self._publish_commands(
            {
                arm.name: (
                    self._gamepad_targets[arm.name].joints,
                    self._gamepad_targets[arm.name].gripper,
                )
                for arm in self.config.arms
            },
            COLLECTOR_HUMAN,
            float(self._phase == "rl"),
            publish_gripper=False,
        )

    def _gamepad_tick(self) -> None:
        with self._lock:
            now = time.monotonic()
            minimum_dt = 1.0 / (2.0 * self.config.gamepad_control_rate)
            dt = min(max(now - self._last_gamepad_tick, minimum_dt), 0.05)
            self._last_gamepad_tick = now
            if not (self._recording or self._teleop_enabled) or self._faulted:
                return
            if not self._intervention:
                self._republish_arm_hold()
                return
            try:
                values, problem = self._gamepad_control_snapshot()
                if values is None:
                    if self.demonstration:
                        raise RuntimeError(f"demonstration interrupted: {problem}")
                    if not self._gamepad_stale_reported:
                        print(f"Gamepad input/state unavailable; holding command: {problem}", flush=True)
                        self._stop_grippers_at_measured_position()
                        self._gamepad_stale_reported = True
                    self._publish_gamepad_hold()
                    return
                if self._gamepad_stale_reported:
                    print("Gamepad input/state restored", flush=True)
                    self._gamepad_stale_reported = False
                self._publish_gamepad(values, dt)
            except (KeyError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                self._skip_reasons[f"gamepad control error: {error}"] += 1
                self._latch_fault(error)

    def _publish_commands(
        self,
        commands: dict[str, tuple[np.ndarray, float]],
        source: int,
        phase: float,
        *,
        publish_gripper: bool = True,
    ) -> None:
        expected = {arm.name for arm in self.config.arms}
        if set(commands) != expected:
            raise ValueError(
                f"command arm set mismatch: provided={sorted(commands)}, expected={sorted(expected)}"
            )
        snapshots: dict[str, ArmCommandSnapshot] = {}
        for arm in self.config.arms:
            joints, gripper = commands[arm.name]
            self._publish_arm_command(
                arm,
                joints,
                gripper,
                publish_gripper=publish_gripper,
            )
            snapshots[arm.name] = ArmCommandSnapshot(
                joints=np.asarray(joints, dtype=np.float64).copy(),
                gripper=float(np.clip(gripper, 0.0, 1.0)),
            )
        self._last_command_snapshot = CommandSnapshot(
            arms=snapshots,
            source=int(source),
            phase=float(phase),
            published_at_ns=time.monotonic_ns(),
        )
        applied = (int(source), "rl" if float(phase) > 0.5 else "vla")
        if applied != self._last_applied_control:
            print(
                f"Applied control source={COLLECTOR_NAMES[int(source)]} "
                f"phase={applied[1]}",
                flush=True,
            )
            self._last_applied_control = applied

    def _publish_arm_command(
        self,
        arm: ArmConfig,
        joints: np.ndarray,
        gripper: float,
        *,
        publish_gripper: bool = True,
    ) -> None:
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError(f"invalid joint command for {arm.name}: {joints}")
        if not np.isfinite(gripper):
            raise ValueError(f"invalid gripper command for {arm.name}: {gripper}")
        joint_message = JointState()
        joint_message.header.stamp = self.get_clock().now().to_msg()
        joint_message.header.frame_id = arm.base_link
        joint_message.name = list(self.servos[arm.name].model.joint_names)
        joint_message.position = joints.tolist()
        self._joint_publishers[arm.name].publish(joint_message)
        if publish_gripper:
            gripper_message = Float32()
            gripper_message.data = float(np.clip(gripper, 0.0, 1.0))
            self._gripper_publishers[arm.name].publish(gripper_message)

    def _republish_arm_hold(self) -> None:
        snapshot = self._last_command_snapshot
        if snapshot is None:
            return
        for arm in self.config.arms:
            command = snapshot.arms[arm.name]
            self._publish_arm_command(
                arm,
                command.joints,
                command.gripper,
                publish_gripper=False,
            )

    def _initialize_episode_hold(self, values: dict[str, object]) -> None:
        snapshots = {}
        for arm in self.config.arms:
            joints = self._measured_joints(
                arm,
                values[f"{arm.name}.measured_joint"],
            )
            gripper = _gripper_open_fraction(
                values[f"{arm.name}.gripper_state"],
                arm.gripper_joint_name,
            )
            self._publish_arm_command(arm, joints, gripper)
            snapshots[arm.name] = ArmCommandSnapshot(
                joints=joints.copy(),
                gripper=gripper,
            )
        self._last_command_snapshot = CommandSnapshot(
            arms=snapshots,
            source=COLLECTOR_HUMAN if self.demonstration else COLLECTOR_PI05,
            phase=0.0,
            published_at_ns=time.monotonic_ns(),
        )

    def _submit_policy(
        self,
        raw_state: np.ndarray,
        images: dict[str, np.ndarray],
    ) -> None:
        if self._intervention:
            return
        submitted = self.policy_client.submit(
            state=self._policy_state(raw_state),
            images=images,
            task=self.config.tasks[self._episode_task_index],
            phase=self._phase,
            reset_episode=self._reset_policy_on_request,
        )
        if submitted is not None:
            if self._log_next_policy_request:
                print(
                    "Policy request submitted "
                    f"id={submitted} requested_phase={self._phase} "
                    f"reset_episode={self._reset_policy_on_request}",
                    flush=True,
                )
                self._log_next_policy_request = False
            self._reset_policy_on_request = False

    def _validate_robot_health(self, values: dict[str, object]) -> None:
        for arm in self.config.arms:
            self._health_monitor.check(
                arm.name,
                values[f"{arm.name}.robot_state"],
            )

    def _camera_failures(self) -> dict[str, BaseException]:
        return {
            name: camera.error()
            for name, camera in self._cameras.items()
            if camera.error() is not None
        }

    def _preflight_problems(self) -> list[str]:
        problems = []
        failures = self._camera_failures()
        if failures:
            problems.append(f"camera failures={failures}")
        for arm in self.config.arms:
            for label, topic in (
                ("robot state", arm.robot_state_topic),
                ("measured joints", arm.measured_joint_topic),
                ("gripper state", arm.gripper_state_topic),
                ("gamepad", arm.gamepad.joy_topic),
            ):
                if self.count_publishers(topic) < 1:
                    problems.append(f"{arm.name} {label} has no publisher: {topic}")
            for label, topic in (
                ("arm command", arm.command_joint_topic),
                ("gripper command", arm.command_gripper_topic),
                ("gripper speed", arm.gripper_speed_topic),
            ):
                if self.count_subscribers(topic) < 1:
                    problems.append(f"{arm.name} {label} has no controller subscriber: {topic}")
            if not self._gripper_action_clients[arm.name].server_is_ready():
                problems.append(
                    f"{arm.name} gripper action server is unavailable: "
                    f"{arm.gripper_action_topic}"
                )
        try:
            if self.policy_client is not None:
                self.policy_client.raise_if_failed()
            snapshot = self._snapshot(raise_on_invalid=True)
            if snapshot is not None:
                self._validate_robot_health(snapshot[0])
        except (KeyError, ValueError, RuntimeError) as error:
            problems.append(str(error))
        return problems

    def _latch_fault(self, error: BaseException) -> None:
        if self._faulted:
            return
        self._faulted = True
        self._teleop_enabled = False
        self._fault_reason = f"{type(error).__name__}: {error}"
        self._stop_grippers_at_measured_position()
        self._recording = False
        self._intervention = False
        self._gamepad_targets = {}
        self._invalidate_policy()
        self._policy_action = None
        self._latest_policy_result = None
        self._last_command_snapshot = None
        self._last_applied_control = None
        self._reset_servo_tracking()
        episode_index = self.writer.next_episode_index()
        cleanup_errors = []
        for writer in self._video_writers.values():
            try:
                writer.abort()
            except BaseException as cleanup_error:
                cleanup_errors.append(str(cleanup_error))
        self._video_writers = {}
        try:
            self.writer.discard_uncommitted_files(episode_index)
        except BaseException as cleanup_error:
            cleanup_errors.append(str(cleanup_error))
        self._states = []
        self._actions = []
        self._annotations = {key: [] for key in self._extra_features}
        details = traceback.format_exc()
        print(
            "\nROLLOUT FAULT LATCHED; commands disabled until collector restart: "
            f"{self._fault_reason}\n{details}",
            file=sys.stderr,
            flush=True,
        )
        if cleanup_errors:
            print(f"Fault cleanup errors: {cleanup_errors}", file=sys.stderr, flush=True)

    def _tick(self) -> None:
        with self._lock:
            if self._faulted:
                return
            try:
                if self.policy_client is not None:
                    self.policy_client.raise_if_failed()
                failures = self._camera_failures()
                if failures:
                    raise RuntimeError(f"RealSense capture failed: {failures}")
                snapshot = self._snapshot(raise_on_invalid=self._recording or self._teleop_enabled)
                if snapshot is None:
                    return
                values, images = snapshot
                self._validate_robot_health(values)
                now_ns = time.monotonic_ns()
                if self.demonstration and self._recording:
                    base_ns = self._latest["camera.base"][0]
                    previous = self._last_demo_images.get("base")
                    if previous is not None and now_ns - previous > int(1.5e9 / self.config.fps):
                        raise RuntimeError("demonstration camera interval exceeded 1.5 frame periods")
                    if base_ns == previous:
                        return
                    stamps = {camera.name: self._latest[f"camera.{camera.name}"][0] for camera in self.config.cameras}
                    if any(stamps[name] <= old for name, old in self._last_demo_images.items()):
                        return
                    self._last_demo_images = stamps
                    now_ns = base_ns
                raw_state = self._raw_state(values, now_ns)
                if not self._recording:
                    return
                self._record_frame(raw_state, values, images, now_ns)
                self._consume_policy_result()
                if not self._intervention:
                    dt = 1.0 / self.config.fps
                    self._publish_policy(values, dt)
                    self._submit_policy(raw_state, images)
            except (KeyError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                self._skip_reasons[f"control error: {error}"] += 1
                self._latch_fault(error)

    def start_episode(self) -> None:
        with self._lock:
            if self._recording:
                return
            if self._faulted:
                print(
                    f"Cannot start rollout: collector is fault-latched ({self._fault_reason})",
                    flush=True,
                )
                return
            if self.demonstration and (not self._teleop_enabled or not self._neutral_gamepads()):
                print("Press s to enable; release BOTH gamepads before n=record", flush=True)
                return
            problems = self._preflight_problems()
            if problems:
                print(f"Rollout start blocked: {'; '.join(problems)}", flush=True)
                return
            snapshot = self._snapshot(raise_on_invalid=True)
            if snapshot is None:
                print("Rollout start blocked: synchronized robot state is unavailable", flush=True)
                return
            episode_index = self.writer.next_episode_index()
            self.writer.discard_uncommitted_files(episode_index)
            self._video_writers = {
                camera.feature_key: AsyncVideoWriter(
                    self.writer.video_path(episode_index, camera.feature_key),
                    width=self.config.image_size[1],
                    height=self.config.image_size[0],
                    fps=self.config.fps,
                )
                for camera in self.config.cameras
            }
            self._states = []
            self._actions = []
            self._annotations = {key: [] for key in self._extra_features}
            self._frame_times = []
            self._last_demo_images = {}
            self._skip_reasons.clear()
            self._last_pose.clear()
            self._last_command_snapshot = None
            self._last_applied_control = None
            self._policy_responses_in_phase = 0
            self._log_next_policy_request = True
            self._reset_servo_tracking()
            self._invalidate_policy()
            self._phase = "vla"
            self._subtask_index = 0
            self._reset_policy_on_request = True
            if not self.demonstration:
                self._restore_policy_gripper_speed()
            self._initialize_episode_hold(snapshot[0])
            self._recording = True
            source = "HUMAN demonstration" if self.demonstration else "requested policy phase=vla"
            print(f"Recording episode {episode_index}; task={self.config.tasks[self._episode_task_index]!r}; {source}", flush=True)

    @staticmethod
    def _intervals(values: list[int | float], predicate) -> list[dict[str, int]]:
        intervals = []
        start = None
        for index, value in enumerate(values + [None]):
            active = index < len(values) and predicate(value)
            if active and start is None:
                start = index
            elif not active and start is not None:
                intervals.append({"start_frame": start, "end_frame": index})
                start = None
        return intervals

    def finish_episode(self, outcome: str | None, *, discard: bool = False) -> None:
        with self._lock:
            if not self._recording:
                return
            if self.demonstration and outcome != "success":
                discard = True
            if (not discard and outcome == "success" and self.config.subtasks
                    and self._subtask_index != len(self.config.subtasks) - 1):
                print("Success requires the final subtask; Tab advances a completed stage", flush=True)
                return
            should_save = (
                not discard
                and outcome in {"success", "failure"}
                and len(self._states) >= MIN_EPISODE_FRAMES
            )
            if should_save:
                try:
                    validate_rollout_episode(
                        np.asarray(self._states),
                        np.asarray(self._actions),
                        arm_count=len(self.config.arms),
                        workspace_bounds=[
                            (np.asarray(arm.workspace_min), np.asarray(arm.workspace_max))
                            for arm in self.config.arms
                        ],
                        maximum_gripper_width=schema.ROBOTIQ_MAX_OPENING_WIDTH_M,
                    )
                except (ValueError, np.linalg.LinAlgError) as error:
                    self._latch_fault(error)
                    return
            self._stop_grippers_at_measured_position()
            self._recording = False
            self._teleop_enabled = False
            self._intervention = False
            self._gamepad_targets = {}
            self._invalidate_policy()
            self._policy_action = None
            self._latest_policy_result = None
            self._last_command_snapshot = None
            self._last_applied_control = None
            self._reset_servo_tracking()
            self._reset_policy_on_request = True
            episode_index = self.writer.next_episode_index()
            if discard or outcome not in {"success", "failure"} or len(self._states) < MIN_EPISODE_FRAMES:
                for writer in self._video_writers.values():
                    writer.abort()
                self.writer.discard_uncommitted_files(episode_index)
                print(f"Discarded rollout episode {episode_index}", flush=True)
            else:
                try:
                    for writer in self._video_writers.values():
                        writer.close()
                    collector_values = [COLLECTOR_HUMAN] * len(self._states) if self.demonstration else [
                        int(value[0])
                        for value in self._annotations[
                            "complementary_info.collector_policy_id"
                        ]
                    ]
                    phase_values = [
                        float(value[0])
                        for value in self._annotations["complementary_info.phase"]
                    ]
                    rl_intervals = self._intervals(
                        phase_values, lambda value: float(value) > 0.5
                    )
                    for interval in rl_intervals:
                        interval["outcome"] = outcome
                    metadata = {
                        "episode_success": outcome,
                        "rl_intervals": rl_intervals,
                        "human_intervention_intervals": self._intervals(
                            collector_values,
                            lambda value: int(value) == COLLECTOR_HUMAN,
                        ),
                    }
                    if self.demonstration:
                        metadata["collection_source"] = "human_demonstration"
                        metadata["critical_intervals"] = rl_intervals
                        # Demonstrations are not policy interventions.
                        metadata["human_intervention_intervals"] = []
                    if self.config.subtasks:
                        indices = [int(value[0]) for value in self._annotations["complementary_info.subtask_index"]]
                        metadata["subtask_intervals"] = [
                            {**interval, "subtask_index": index, "name": name}
                            for index, name in enumerate(self.config.subtasks)
                            for interval in self._intervals(indices, lambda value, i=index: value == i)
                        ]
                    image_stats = {
                        key: writer.image_stats.as_dict()
                        for key, writer in self._video_writers.items()
                    }
                    saved = self.writer.save_episode(
                        np.asarray(self._states),
                        np.asarray(self._actions),
                        image_stats,
                        frame_features={
                            key: np.asarray(values)
                            for key, values in self._annotations.items()
                        },
                        extra_episode_metadata=metadata,
                        task=self.config.tasks[self._episode_task_index],
                    )
                    print(
                        f"Saved rollout episode {saved}: outcome={outcome}, "
                        f"frames={len(self._states)}",
                        flush=True,
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    self._latch_fault(error)
                    return
            self._video_writers = {}
            self._states = []
            self._actions = []
            self._annotations = {key: [] for key in self._extra_features}

    def toggle_intervention(self) -> None:
        with self._lock:
            if self.demonstration and self._intervention:
                print("Demonstration remains HUMAN; use d to pause/discard", flush=True)
                return
            if not self._intervention:
                if not (self._recording or self._teleop_enabled):
                    print("Ignoring gamepad intervention outside an active episode")
                    return
                if not self._neutral_gamepads():
                    print("Release BOTH gamepads before takeover", flush=True)
                    return
                now_ns = time.monotonic_ns()
                targets: dict[str, GamepadTarget] = {}
                for arm in self.config.arms:
                    required = (
                        (f"{arm.name}.robot_state", self.config.gamepad_state_timeout_s),
                        (f"{arm.name}.measured_joint", self.config.gamepad_state_timeout_s),
                        (f"{arm.name}.gripper_state", self.config.gamepad_state_timeout_s),
                        (f"{arm.name}.gamepad", self.config.gamepad_joy_timeout_s),
                    )
                    if any(key not in self._latest for key, _ in required):
                        print(f"Cannot enter intervention: {arm.name} gamepad/state is missing")
                        return
                    if any(
                        now_ns - self._latest[key][0] > int(timeout_s * 1_000_000_000)
                        for key, timeout_s in required
                    ):
                        print(f"Cannot enter intervention: {arm.name} gamepad/state is stale")
                        return
                    measured = self._measured_joints(
                        arm, self._latest[f"{arm.name}.measured_joint"][1]
                    )
                    robot_state = self._latest[f"{arm.name}.robot_state"][1]
                    tool_position, tool_rotation = schema.pose_from_stamped(robot_state.f_t_ee)
                    position, rotation = self.servos[arm.name].forward_tcp(
                        measured,
                        tool_position,
                        tool_rotation,
                    )
                    gripper = _gripper_open_fraction(
                        self._latest[f"{arm.name}.gripper_state"][1],
                        arm.gripper_joint_name,
                    )
                    targets[arm.name] = GamepadTarget(
                        position=position,
                        rotation=rotation,
                        gripper=gripper,
                        joints=measured.copy(),
                        flange_to_tcp_position=tool_position,
                        flange_to_tcp_rotation=tool_rotation,
                    )
                self._gamepad_targets = targets
                self._stop_grippers_at_measured_position()
                self._intervention = True
            else:
                self._stop_grippers_at_measured_position()
                self._intervention = False
                self._gamepad_targets = {}
                self._restore_policy_gripper_speed()
            self._last_gamepad_tick = time.monotonic()
            self._last_gamepad_input_log = 0.0
            self._gamepad_stale_reported = False
            self._reset_servo_tracking()
            self._invalidate_policy()
            self._policy_action = None
            self._latest_policy_result = None
            if not self._intervention:
                self._reset_policy_on_request = True
                self._policy_responses_in_phase = 0
                self._log_next_policy_request = True
            requested = "active" if self._intervention else "inactive"
            next_source = "gamepad" if self._intervention else "policy"
            print(
                f"Requested gamepad intervention={requested}; "
                f"waiting for first {next_source} command",
                flush=True,
            )

    def toggle_phase(self) -> None:
        with self._lock:
            if self.demonstration and not self._recording:
                return
            if self.demonstration:
                self._phase = "vla" if self._phase == "rl" else "rl"
                print(f"Critical annotation={'ON' if self._phase == 'rl' else 'OFF'}; HUMAN control", flush=True)
                return
            if self._intervention and not self.demonstration:
                print("Ignoring phase toggle while gamepad intervention is active")
                return
            self._phase = "vla" if self._phase == "rl" else "rl"
            self._invalidate_policy()
            self._policy_action = None
            self._latest_policy_result = None
            self._reset_servo_tracking()
            self._policy_responses_in_phase = 0
            self._log_next_policy_request = True
            print(
                f"Requested policy phase={self._phase}; "
                "waiting for server acknowledgement",
                flush=True,
            )

    def discard_latest(self) -> None:
        if self._recording:
            self.finish_episode(None, discard=True)
        else:
            removed = self.writer.discard_latest_episode()
            print(f"Discarded latest episode {removed}")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._recording:
                self.finish_episode(None, discard=True)
            else:
                self._stop_grippers_at_measured_position()
            self._teleop_enabled = False
            self._closed = True
        if self.policy_client is not None:
            self.policy_client.close()
        for camera in reversed(tuple(self._cameras.values())):
            camera.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    config = load_rollout_config(parse_args().config)
    run(config)


def run(config: RolloutConfig, *, demonstration: bool = False, resume: bool = False) -> None:
    from vla.core.writer_lock import WriterLease

    with WriterLease(config.dataset_root):
        if demonstration:
            from vla.datacollection.franka_robotiq_dual.gamepad_record import validate_dataset_root, validate_resume_contract
            validate_dataset_root(config.dataset_root, resume=resume)
            if resume:
                validate_resume_contract(config)
        _run(config, demonstration=demonstration)


def _run(config: RolloutConfig, *, demonstration: bool = False) -> None:
    rclpy.init()
    try:
        collector = RolloutCollector(config, demonstration=demonstration)
    except BaseException:
        rclpy.try_shutdown()
        raise
    executor = MultiThreadedExecutor(num_threads=max(4, len(config.arms) * 3))
    executor.add_node(collector)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    keymap = {b"\t": "subtask", b"t": "task"}
    if demonstration:
        keymap.update({b"s": "enable", b"n": "right", b"y": "success",
                       b"x": "discard", b"d": "pause", b"c": "quit"})
        print("STANDBY: s=enable both gamepads, n=record, y=success/save, x=discard, "
              "d=pause, r=critical ON/OFF, Tab=next subtask, c/Esc=exit. "
              "ROS control faults require a restart.", flush=True)
    else:
        print("Rollout task selection: press t between episodes to cycle configured prompts.", flush=True)
    keyboard = KeyboardEvents(keymap=keymap)
    try:
        stop = False
        while not stop:
            for event in keyboard.consume():
                if event == "enable":
                    collector.enable_teleop()
                elif event == "pause":
                    collector.pause_teleop()
                elif event == "discard":
                    collector.finish_episode(None, discard=True)
                elif event == "subtask":
                    collector.next_subtask()
                elif event == "task":
                    collector.next_task()
                elif event == "right":
                    collector.start_episode()
                elif event == "left":
                    if demonstration:
                        collector.finish_episode(None, discard=True)
                    else:
                        collector.discard_latest()
                elif event == "intervention":
                    collector.toggle_intervention()
                elif event == "phase":
                    collector.toggle_phase()
                elif event == "success":
                    collector.finish_episode("success")
                elif event == "failure":
                    collector.finish_episode("failure")
                elif event == "quit":
                    stop = True
                    break
            time.sleep(0.03)
    finally:
        collector.close()
        keyboard.close()
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        collector.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
