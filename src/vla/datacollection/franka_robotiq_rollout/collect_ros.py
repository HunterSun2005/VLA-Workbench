"""Collect Franka VLA/RLT rollouts with optional gamepad intervention.

Run from the repository root with the ROS2 Python interpreter::

    PYTHONPATH=src:${PYTHONPATH:-} /usr/bin/python3 \
        -m vla.datacollection.franka_robotiq_rollout.collect \
        --config datacollection/franka_robotiq_rollout/configs/single_left.yaml
"""

from __future__ import annotations

from vla.terminal import (console_print as print, console_status, console_debug, terminal_session,
                          TerminalConfirmation, diagnostic_path, current_reporter)

import os
from dataclasses import asdict
import sys
import threading
import time
import traceback
from collections import Counter
from dataclasses import dataclass

import numpy as np
import rclpy
from control_msgs.action import GripperCommand
from franka_msgs.msg import FrankaRobotState
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import parameter_value_to_python
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import Float32, Float64MultiArray, UInt64
from vla.actions import ActionSpec, matrix_to_rot6d
from vla.actions.spec import fingerprint, validate_array
from vla.control.ros import ControlCoordinator, ArmMeasurement
from vla.control.ros.transport import joint_command, controller_identity

from vla.datacollection.franka_robotiq_single_left import schema
from vla.datacollection.franka_robotiq_single_left.auto_record import (
    AsyncVideoWriter,
    RealSenseCamera,
    prepare_image,
)
from vla.datacollection.franka_robotiq_single_left.v30_writer import V30DatasetWriter
from vla.datacollection.start_teleop.franka_kinematics import (
    build_kinematic_model,
    joint7_step,
)
from .config import ArmConfig, RolloutConfig
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


from vla.control.ros.cartesian_servo import CartesianServo


class RolloutCollector(Node):
    def __init__(self, config: RolloutConfig, *, demonstration: bool = False) -> None:
        try:
            self._initialize(config, demonstration=demonstration)
        except BaseException:
            # Construction never enables motion. Release partially opened I/O.
            for camera in reversed(tuple(getattr(self, '_cameras', {}).values())):
                try:
                    camera.close()
                except Exception:
                    pass
            client = getattr(self, 'policy_client', None)
            if client is not None:
                client.close()
            self.destroy_node()
            raise

    def _initialize(self, config: RolloutConfig, *, demonstration: bool = False) -> None:
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
        for key, dim, dtype in [
            ("observation.anchor", config.action_dim, "float64"),
            ("command.joint", 8 * len(config.arms), "float64"),
            ("command.fk_tcp", 10 * len(config.arms), "float64"),
            ("command.identity", 4, "int64"),
            ("command.source", 1, "int64"),
            ("diagnostic.controller", 3 * len(config.arms), "int64"),
            ("diagnostic.filtered_joint", 7 * len(config.arms), "float64"),
        ]:
            self._extra_features[key] = {"dtype": dtype, "shape": [dim], "names": None}
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
        self._high_speed = {a.name: False for a in config.arms}
        self._speed_scale = {a.name: 1. for a in config.arms}
        self._previous_buttons = {}
        self._initial_pose = {}
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
        self.require_online = bool(self.policy_client and self.policy_client.contract.get('online_enabled'))
        self._full_episode = bool(self.require_online and self.policy_client.contract.get('online_full_episode'))
        self._processing = False
        self._pending_online_save = None
        self._pending_finish = None
        self._pending_phase = None
        self._phase_exited = False
        self._at_boundary = False
        self._pending_frame = None
        self._segment_id = 0
        self._episode_identity = {}
        if self.require_online:
            from vla.core.online_contract import TIMING_FEATURES, load_run
            self._extra_features.update(TIMING_FEATURES)
            self._annotations = {key: [] for key in self._extra_features}
            run = load_run(config.dataset_root)
            if any(run[key] != self.policy_client.contract[key] for key in ('run_uuid', 'dataset_uuid')):
                raise ValueError('online dataset/server identity mismatch')
        metadata = {
            "action_spec": config.action_spec.to_dict(),
            "actions_stored_as": "absolute",
            "action_recording_point": "accepted_before_tracking",
            "recording_schema_version": 4 if self.require_online else 3,
            **({key: run[key] for key in ("run_uuid", "dataset_uuid")} if self.require_online else {}),
            "rlt_episode_metadata_fields": {
                "rl_intervals": "RL frame intervals with episode outcome.",
                "human_intervention_intervals": "Human intervention frame intervals.",
            },
        }
        from vla.datacollection.gripper_contract import ROS_GRIPPER_RECORDING_VERSION
        metadata["gripper_recording_version"] = ROS_GRIPPER_RECORDING_VERSION
        metadata["action_gripper_source"] = "command_goal_or_latched_hold"
        if demonstration:
            from vla.datacollection.franka_robotiq_dual.gamepad_record import recording_contract
            metadata.update(collection_kind="human_demonstration", gamepad_recording_version=2,
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
        self._kinematics_identity = {}
        self.servos = {
            arm.name: CartesianServo(self._load_kinematic_model(arm), config)
            for arm in config.arms
        }
        self._setup_ros_io()
        self.coordinator = ControlCoordinator(
            config.action_spec, self.servos,
            {a.name: (np.asarray(a.workspace_min), np.asarray(a.workspace_max)) for a in config.arms},
            self._coordinated_publish, self._request_controller_reset,
            state_timeout_s=config.gamepad_state_timeout_s,
            command_timeout_s=config.control_discontinuity_timeout_s)
        self._last_gripper_goal = {}
        self._publish_sequence = 0
        from vla.actions.metadata import RecordingContract
        tool = self._tool_definition()
        metadata['recording_contract'] = RecordingContract(
            config.action_spec,
            fingerprint({a.name: os.environ.get(f'{a.name.upper()}_ROBOT_IP', a.robot_state_publisher) for a in config.arms}),
            fingerprint(tool), fingerprint(self._kinematics_identity),
            fingerprint({k: str(v) for k, v in asdict(config).items()
                         if k not in ('dataset_root', 'task', 'tasks', 'socket_path', 'joy_socket_path')})).to_dict()
        metadata['tool_definition'] = tool
        self.writer = V30DatasetWriter(
            config.dataset_root,
            config.fps,
            config.task,
            tuple(camera.feature_key for camera in config.cameras),
            *config.image_size,
            state_fields=_prefixed_fields(config.arms, schema.STATE_FIELDS),
            action_fields=config.action_spec.action_fields,
            extra_features=self._extra_features,
            robot_type="franka_robotiq_dual" if len(config.arms) == 2 else "franka_robotiq",
            info_metadata=metadata,
            strict_metadata=True,
            allow_multiple_tasks=demonstration and len(config.tasks) > 1,
        )

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

    def _tool_definition(self):
        deadline = time.monotonic() + 30
        while any(f'{a.name}.{field}' not in self._latest for a in self.config.arms
                  for field in ('robot_state', 'measured_joint', 'gripper_state')):
            if time.monotonic() > deadline:
                raise TimeoutError('measured robot, joint and gripper states required before controller initialization')
            rclpy.spin_once(self, timeout_sec=.1)
        tool = {}
        for arm in self.config.arms:
            position, rotation = schema.pose_from_stamped(self._latest[f'{arm.name}.robot_state'][1].f_t_ee)
            tool[arm.name] = {'position': position.tolist(), 'rotation': rotation.tolist(),
                              'base': arm.base_link, 'tip': arm.tip_link}
        return tool

    def _validate_policy_contract(self) -> None:
        contract = self.policy_client.contract
        expected_cameras = [camera.feature_key for camera in self.config.cameras]
        expected_robot_type = (
            "franka_robotiq_dual" if len(self.config.arms) == 2 else "franka_robotiq"
        )
        available_state_names = set(_prefixed_fields(self.config.arms, schema.STATE_FIELDS))
        expected_state_names = contract.get('state_names', [])
        expected_action_names = list(self.config.action_spec.action_fields)
        expected_camera_shapes = {
            key: [*self.config.image_size, 3] for key in expected_cameras
        }
        mismatches = []
        if int(contract["state_dim"]) != len(expected_state_names):
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
        if not expected_state_names or len(set(expected_state_names)) != len(expected_state_names) or not set(expected_state_names).issubset(available_state_names):
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
        try:
            self.config.action_spec.require_compatible(ActionSpec.from_dict(contract["action_spec"]), training=False)
        except (KeyError, ValueError) as error:
            mismatches.append(f"checkpoint action contract: {error}")
        if contract.get("wire_action_representation") != "absolute":
            mismatches.append("policy transport must return absolute targets")
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
        self._kinematics_identity[arm.name] = fingerprint(description)
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
        self._reset_publishers = {}
        self._joint_publishers = {}
        self._gripper_publishers = {}
        self._gripper_drive_publishers = {}
        self._gripper_speed_publishers = {}
        self._gripper_action_clients = {}
        self._subscription_handles = []
        for arm in self.config.arms:
            prefix = arm.name
            self._reset_publishers[prefix] = self.create_publisher(UInt64, f"/{prefix}/teleop/reset", 10)
            self._subscription_handles.append(self.create_subscription(
                JointState, f"/{prefix}/teleop/controller_state",
                lambda msg, name=prefix: self._controller_ack(name, msg), 10))
            subscriptions = (
                (FrankaRobotState, arm.robot_state_topic, f"{prefix}.robot_state"),
                (JointState, arm.measured_joint_topic, f"{prefix}.measured_joint"),
                (JointState, arm.gripper_state_topic, f"{prefix}.gripper_state"),
                (Joy, arm.gamepad.joy_topic, f"{prefix}.gamepad"),
                (JointState, f"/{prefix}/gello/joint_states", f"{prefix}.gello_joint"),
                (Float32, f"/{prefix}/gello/gripper_width_percent", f"{prefix}.gello_gripper"),
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
            self._gripper_drive_publishers[prefix] = self.create_publisher(
                Float32, arm.gripper_drive_topic, 10
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
        prepared = prepare_image(rgb, self.config.image_size)
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

    def _raw_anchor(self, raw_state):
        from vla.actions.dataset import anchors_from_state
        return anchors_from_state(np.asarray(raw_state)[None], _prefixed_fields(self.config.arms, schema.STATE_FIELDS),
                                  self.config.action_spec, gripper_unit='metres')[0]

    def _policy_state(self, raw_state):
        names = self.policy_client.contract['state_names']
        available = dict(zip(_prefixed_fields(self.config.arms, schema.STATE_FIELDS), raw_state))
        if any(name not in available for name in names):
            raise ValueError('checkpoint requests unavailable measured fields')
        return np.asarray([available[name] for name in names], dtype=np.float32)

    def _reset_servo_tracking(self) -> None:
        for servo in self.servos.values():
            servo.reset_tracking()

    def _invalidate_policy(self) -> None:
        if self.policy_client is not None:
            self.policy_client.invalidate()

    def _neutral_gamepads(self) -> bool:
        if self.config.input_source == 'gello':
            return all(f'{a.name}.gello_joint' in self._latest for a in self.config.arms)
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
            self._control_boundary(reason="pause")
            self._invalidate_policy()

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

    def _actual_action(self, values, snapshot):
        command = self.coordinator.accepted
        if command is None:
            raise RuntimeError("no accepted command to record")
        return command.target.copy()

    def _observe_control(self):
        for arm in self.config.arms:
            keys = [f"{arm.name}.{suffix}" for suffix in ("measured_joint", "robot_state", "gripper_state")]
            if any(key not in self._latest for key in keys):
                continue
            joint, robot, gripper = [self._latest[key] for key in keys]
            p, r = schema.pose_from_stamped(robot[1].f_t_ee)
            self.coordinator.observe(arm.name, ArmMeasurement(
                self._measured_joints(arm, joint[1]), _gripper_open_fraction(gripper[1], arm.gripper_joint_name),
                p, r, min(joint[0], robot[0], gripper[0])))

    def _controller_ack(self, arm, message):
        with self._lock:
            if not hasattr(self, 'coordinator'):
                return
            try:
                generation, command_id, fault = controller_identity(message)
                if generation != self.coordinator.generation:
                    return
                if fault:
                    raise RuntimeError(f'{arm} controller command watchdog fault')
                self._observe_control()
                self.coordinator.acknowledge(arm, int(generation), int(command_id),
                                             message.position, time.monotonic_ns())
            except (ValueError, RuntimeError) as error:
                self._latch_fault(error)

    def _request_controller_reset(self, generation):
        for publisher in self._reset_publishers.values():
            message = UInt64()
            message.data = generation
            publisher.publish(message)

    def _coordinated_publish(self, name, joints, gripper, generation, command_id):
        arm = next(a for a in self.config.arms if a.name == name)
        self._publish_sequence += 1
        message = joint_command(self.servos[name].model.joint_names, joints,
                                generation, command_id, self._publish_sequence,
                                self.get_clock().now().to_msg())
        self._joint_publishers[name].publish(message)
        if self.coordinator.owner != 'pose' and self._last_gripper_goal.get(name) != gripper:
            self._send_gripper_goal(arm, gripper)
            self._last_gripper_goal[name] = gripper
        goal = Float32()
        goal.data = gripper
        self._gripper_publishers[name].publish(goal)

    def _control_boundary(self, owner=None, *, fault=False, reason='control boundary'):
        self.coordinator.transition(owner, fault=fault, reason=reason)
        self._last_command_snapshot = None
        self._input_action = None
        self._last_gripper_goal.clear()
        self._pending_frame = None
        self._segment_id += 1

    def _track_control(self, dt):
        command = self.coordinator.tick(dt)
        if command is None:
            return
        source = COLLECTOR_HUMAN if command.source in ('gamepad', 'gello', 'pose') else (
            COLLECTOR_RLT_ACTOR if self._latest_policy_result and self._latest_policy_result.source_type > .5 else COLLECTOR_PI05)
        self._last_command_snapshot = CommandSnapshot(
            arms={arm: ArmCommandSnapshot(value['joints'], value['gripper'])
                  for arm, value in self.coordinator.submitted.items()},
            source=source, phase=float(self._phase == 'rl'), published_at_ns=time.monotonic_ns())

    def _record_frame(
        self,
        raw_state: np.ndarray,
        values: dict[str, object],
        images: dict[str, np.ndarray],
        now_ns: int,
    ) -> None:
        snapshot = self._last_command_snapshot
        if not self._recording or snapshot is None or self.coordinator.accepted is None:
            return
        if self.demonstration and now_ns - snapshot.published_at_ns > int(self.config.control_discontinuity_timeout_s * 1e9):
            raise RuntimeError("demonstration command stream is stale")
        action = self._actual_action(values, snapshot)
        for key, writer in self._video_writers.items():
            writer.write(images[key])
        self._states.append(raw_state)
        self._actions.append(action)
        command = self.coordinator.accepted
        names = _prefixed_fields(self.config.arms, schema.STATE_FIELDS)
        measured = dict(zip(names, raw_state))
        for name in self.config.action_spec.measured_fields:
            if name.endswith('gripper_position'):
                measured[name] = float(np.clip(measured[name] / schema.ROBOTIQ_MAX_OPENING_WIDTH_M, 0, 1))
        self._annotations['observation.anchor'].append(self.config.action_spec.anchor_from_fields(measured))
        self._annotations['command.joint'].append(np.concatenate([
            np.r_[self.coordinator.submitted[a.name]['joints'], self.coordinator.submitted[a.name]['gripper']]
            for a in self.config.arms]))
        self._annotations['command.fk_tcp'].append(np.concatenate([
            np.r_[self.coordinator.submitted[a.name]['fk_tcp'], self.coordinator.submitted[a.name]['gripper']]
            for a in self.config.arms]))
        self._annotations['command.identity'].append([
            command.generation, command.command_id, command.accepted_at_ns, command.observed_at_ns])
        self._annotations['command.source'].append([snapshot.source])
        if self.require_online:
            for name, value in {'observed_at_ns': now_ns, 'command_at_ns': command.accepted_at_ns,
                                'command_id': command.command_id, 'segment_id': self._segment_id,
                                'action_valid': 0, 'segment_terminal': 0}.items():
                self._annotations['complementary_info.'+name].append([value])
            self._pending_frame = len(self._states)-1
        diagnostics = [self.coordinator.diagnostics.get(a.name, {}) for a in self.config.arms]
        self._annotations['diagnostic.controller'].append(np.concatenate([
            [d.get('generation', -1), d.get('command_id', -1), d.get('received_at_ns', -1)] for d in diagnostics]))
        self._annotations['diagnostic.filtered_joint'].append(np.concatenate([
            d.get('filtered_joints', np.zeros(7)) for d in diagnostics]))
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
        self._control_boundary(self.config.input_source if self._intervention else "policy", reason="policy mode change")
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
        action = validate_array(result.action, self.config.action_spec)
        self._latest_policy_result = result
        if managed:
            self._web_switch_pending = False
        self._policy_action = action

    def _publish_policy(self, values, dt):
        if self._policy_action is None or self._latest_policy_result is None:
            return False
        if time.monotonic() - self._latest_policy_result.received_at > self.config.control_discontinuity_timeout_s:
            raise RuntimeError("latest policy action stream expired")
        if self.coordinator.mode != 'active':
            return False
        accepted = self.coordinator.accept(self._policy_action, source='policy', generation=self.coordinator.generation, slot_id=time.monotonic_ns()*30//1_000_000_000)
        return accepted is not None

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
            self._latch_fault(error)
            print(f"{arm_name} Robotiq goal failed: {error}", file=sys.stderr, flush=True)

    def _gripper_goal_result(self, arm_name: str, future: object) -> None:
        self._gripper_goal_futures.discard(future)
        try:
            result = future.result()
            if getattr(result, "status", 4) not in (4, 5):
                raise RuntimeError(f"gripper execution status {result.status}")
        except Exception as error:
            self._latch_fault(error)
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
        if goal is not None:
            target.gripper = goal
        target.gripper_direction = direction

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
            from .gamepad import decode_cartesian_gamepad_values
            from scipy.spatial.transform import Rotation
            from vla.control.geometry import upright_tcp_rotation
            decoded = decode_cartesian_gamepad_values(gamepad_message.axes, gamepad_message.buttons, arm.gamepad, self.config)
            buttons = list(gamepad_message.buttons)
            old = self._previous_buttons.get(arm.name, [0]*len(buttons))
            edge = lambda index: len(buttons) > index and buttons[index] and not (len(old) > index and old[index])
            if edge(4):
                self._high_speed[arm.name] = not self._high_speed[arm.name]
            if edge(5) and self._recording:
                self.toggle_phase()
            if edge(6):
                target.rotation = upright_tcp_rotation(target.rotation)
            if edge(7):
                target.position, target.rotation = [v.copy() for v in self._initial_pose[arm.name]]
            self._previous_buttons[arm.name] = buttons
            scale = self._speed_scale[arm.name]
            goal = 2. if self._high_speed[arm.name] else 1.
            scale += np.clip(goal-scale, -dt/.3, dt/.3)
            self._speed_scale[arm.name] = float(scale)
            velocity = _clamp_norm(decoded.linear_velocity_base * scale, self.config.max_linear_velocity)
            angular = _clamp_norm(decoded.angular_velocity_tool * scale, self.config.max_angular_velocity)
            target.position = np.clip(target.position + velocity*dt, arm.workspace_min, arm.workspace_max)
            target.rotation = target.rotation @ Rotation.from_rotvec(angular*dt).as_matrix()
            gripper_motion = decoded.gripper_motion
            target.joints = self.coordinator._measurements[arm.name].joints.copy()
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
            commands[arm.name] = (target.position, target.rotation, target.gripper)
        action = self.coordinator.gamepad_target(commands, 1.0 / self.config.fps)
        self._input_action = action
        now = time.monotonic()
        if active_inputs and now - self._last_gamepad_input_log >= 0.5:
            console_debug("Gamepad input " + "; ".join(active_inputs))
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

    def _gamepad_tick(self):
        with self._lock:
            now = time.monotonic()
            dt = min(max(now-self._last_gamepad_tick, .001), .05)
            self._last_gamepad_tick = now
            try:
                self._observe_control()
                self._online_save_tick()
                if getattr(self, '_recovery_pending', False) and self.coordinator.mode == 'standby':
                    self._faulted, self._fault_reason, self._recovery_pending = False, None, False
                if self.coordinator.mode == 'resetting':
                    self.coordinator.tick(dt)
                    return
                if self._faulted or not (self._recording or self._teleop_enabled):
                    return
                if self.coordinator.mode != 'active':
                    return
                if self._intervention:
                    if self.config.input_source == 'gello':
                        self._publish_gello()
                    else:
                        values, problem = self._gamepad_control_snapshot()
                        if values is None:
                            raise RuntimeError(f'input stream interrupted: {problem}')
                        self._publish_gamepad(values, dt)
                self._track_control(dt)
            except (KeyError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                self._latch_fault(error)

    def _publish_gello(self):
        joints, grippers = {}, {}
        for arm in self.config.arms:
            q = self._latest.get(f'{arm.name}.gello_joint')
            g = self._latest.get(f'{arm.name}.gello_gripper')
            if q is None or g is None or time.monotonic_ns()-min(q[0], g[0]) > int(self.config.gamepad_joy_timeout_s*1e9):
                raise RuntimeError('GELLO input stream interrupted')
            joints[arm.name] = self._measured_joints(arm, q[1])
            grippers[arm.name] = schema.gripper_action(g[1])
        action = self.coordinator.gello_target(joints, grippers)
        self._input_action = action

    def _republish_arm_hold(self):
        # Tracking and controller heartbeat run independently at 100 Hz.
        return

    def _initialize_episode_hold(self, values):
        self._control_boundary(self.config.input_source if self.demonstration else 'policy', reason='episode start')

    def _submit_policy(
        self,
        raw_state: np.ndarray,
        images: dict[str, np.ndarray],
    ) -> None:
        if self._intervention:
            return
        submitted = self.policy_client.submit(
            state=self._policy_state(raw_state),
            anchor=self._raw_anchor(raw_state),
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
        try:
            self._control_boundary(fault=True, reason=self._fault_reason)
        except Exception as stop_error:
            self._fault_reason += f"; stop failed: {stop_error}"
        try:
            self._stop_grippers_at_measured_position()
        except Exception as stop_error:
            self._fault_reason += f'; gripper stop failed: {stop_error}'
        self._pending_finish = None
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
                if not (self._recording or self._teleop_enabled):
                    return
                self._complete_interval(now_ns)
                if self._pending_phase is not None:
                    self._record_frame(raw_state, values, images, now_ns)
                    if self._states:
                        self._annotations['complementary_info.segment_terminal'][-1] = [1]
                    self._phase = self._pending_phase
                    self._pending_phase = None
                    self._phase_exited = True
                    self._invalidate_policy()
                    self._policy_action = self._latest_policy_result = None
                    self._reset_policy_on_request = True
                    self._control_boundary('policy', reason='critical phase ended')
                    return
                if self._pending_finish is not None:
                    outcome = self._pending_finish
                    self._pending_finish = None
                    self._record_frame(raw_state, values, images, now_ns)
                    if self.require_online and self._states:
                        self._annotations['complementary_info.segment_terminal'][-1] = [int(self._phase == 'rl')]
                    self._at_boundary = True
                    try:
                        self.finish_episode(outcome)
                    finally:
                        self._at_boundary = False
                    return
                self._consume_policy_result()
                previous = self.coordinator.command_id
                if self._intervention:
                    if self.coordinator.mode == 'active' and getattr(self, '_input_action', None) is not None:
                        self.coordinator.accept(self._input_action, source=self.config.input_source,
                                                generation=self.coordinator.generation, slot_id=time.monotonic_ns()*30//1_000_000_000)
                else:
                    self._publish_policy(values, 1/self.config.fps)
                    self._submit_policy(raw_state, images)
                if self.coordinator.command_id != previous:
                    self._track_control(1/self.config.gamepad_control_rate)
                    self._record_frame(raw_state, values, images, now_ns)
            except (KeyError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                self._skip_reasons[f"control error: {error}"] += 1
                self._latch_fault(error)

    def _complete_interval(self, now_ns):
        if not self.require_online or self._pending_frame is None:
            return
        index = self._pending_frame
        observed = self._annotations['complementary_info.observed_at_ns'][index][0]
        command = self._annotations['complementary_info.command_at_ns'][index][0]
        elapsed = (now_ns-observed)/1e9
        if .5/self.config.fps <= elapsed <= 1.5/self.config.fps and observed <= command < now_ns:
            self._annotations['complementary_info.action_valid'][index] = [1]
        else:
            self._segment_id += 1
        self._pending_frame = None

    def _online_save_tick(self):
        if self._pending_online_save is None or self.coordinator.mode != 'standby':
            return
        episode, outcome = self._pending_online_save
        self._pending_online_save = None
        def process():
            try:
                self.policy_client.process_saved_episode(dataset_root=self.config.dataset_root,
                                                         episode_id=episode, outcome=outcome)
            except Exception as error:
                with self._lock:
                    self._faulted, self._fault_reason = True, f'online training failed: {error}'
            finally:
                self._processing = False
        self._online_worker = threading.Thread(target=process, name='ros-online-save', daemon=True)
        self._online_worker.start()

    def start_episode(self) -> None:
        with self._lock:
            if self._recording or self._processing:
                return
            if self._faulted:
                print(
                    f"Cannot start rollout: collector is fault-latched ({self._fault_reason})",
                    flush=True,
                )
                return
            if self.demonstration and (not self._teleop_enabled or not self._neutral_gamepads()):
                print("请先启用遥操作；开始录制前松开双侧手柄输入。", flush=True)
                return
            problems = self._preflight_problems()
            if problems:
                print(f"Rollout start blocked: {'; '.join(problems)}", flush=True)
                return
            snapshot = self._snapshot(raise_on_invalid=True)
            if snapshot is None:
                print("Rollout start blocked: synchronized robot state is unavailable", flush=True)
                return
            if self.require_online:
                self._episode_identity = self.policy_client.begin_episode()
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
            self._phase = "rl" if self._full_episode else "vla"
            self._subtask_index = 0
            self._phase_exited = False
            self._pending_phase = None
            self._pending_finish = None
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

    def _validate_recorded_episode(self, states, actions, **kwargs):
        if self.config.action_space == 'tcp':
            return validate_rollout_episode(states, actions, **kwargs)
        validate_array(actions, self.config.action_spec)
        if np.asarray(states).shape != (len(actions), 50 * len(self.config.arms)) or not np.isfinite(states).all():
            raise ValueError('invalid measured state stream')

    def finish_episode(self, outcome: str | None, *, discard: bool = False) -> None:
        with self._lock:
            if self.require_online and self._recording and not discard and not self._at_boundary:
                self._pending_finish = outcome
                return
            if not self._recording:
                return
            if self.demonstration and outcome != "success":
                discard = True
            if (not discard and outcome == "success" and self.config.subtasks
                    and self._subtask_index != len(self.config.subtasks) - 1):
                print("成功保存需要完成最后阶段；完成当前阶段后执行下一阶段操作。", flush=True)
                return
            should_save = (
                not discard
                and outcome in {"success", "failure"}
                and len(self._states) >= MIN_EPISODE_FRAMES
            )
            if should_save:
                try:
                    self._validate_recorded_episode(
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
            self._control_boundary(reason="episode end")
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
                        **self._episode_identity,
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
                    if self.require_online:
                        self._processing = True
                        self._pending_online_save = (saved, outcome)
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
                print("演示采集保持人工控制；暂停会丢弃当前回合。", flush=True)
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
                        (f"{arm.name}.gello_joint" if self.config.input_source == "gello" else f"{arm.name}.gamepad", self.config.gamepad_joy_timeout_s),
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
                    self._initial_pose.setdefault(arm.name, (position.copy(), rotation.copy()))
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
            self._control_boundary(self.config.input_source if self._intervention else 'policy', reason='takeover/return')
            requested = "active" if self._intervention else "inactive"
            next_source = "gamepad" if self._intervention else "policy"
            print(
                f"Requested gamepad intervention={requested}; "
                f"waiting for first {next_source} command",
                flush=True,
            )

    def recover_fault(self):
        with self._lock:
            if self._recording or self._video_writers:
                raise RuntimeError('finish storage cleanup before recovery')
            if self.require_online:
                self.policy_client.recover_online()
            self._observe_control()
            self.coordinator._fresh()
            values = {key: value[1] for key, value in self._latest.items()}
            self._validate_robot_health(values)
            if self._camera_failures():
                raise RuntimeError('camera fault remains')
            self._control_boundary(reason='explicit fault recovery')
            self._recovery_pending = True
            return True

    def toggle_phase(self) -> None:
        with self._lock:
            if self._full_episode:
                return
            if self.demonstration and not self._recording:
                return
            if self.demonstration:
                self._phase = "vla" if self._phase == "rl" else "rl"
                print(f"Critical annotation={'ON' if self._phase == 'rl' else 'OFF'}; HUMAN control", flush=True)
                return
            if self._intervention and not self.demonstration:
                print("Ignoring phase toggle while gamepad intervention is active")
                return
            if self.require_online:
                if self._phase == 'rl':
                    self._pending_phase = 'vla'
                    return
                if self._phase_exited:
                    raise RuntimeError('online critical phase may occur only once per episode')
            self._phase = "vla" if self._phase == "rl" else "rl"
            self._control_boundary('policy', reason='policy phase changed')
            self._reset_policy_on_request = True
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
                self._control_boundary(reason="session close")
            self._teleop_enabled = False
            self._closed = True
        if self.policy_client is not None:
            self.policy_client.close()
        for camera in reversed(tuple(self._cameras.values())):
            camera.close()



def main():
    from vla.cli.ros_session import main as shared_main
    shared_main(['--workflow', 'deploy', *sys.argv[1:]])


if __name__ == '__main__':
    main()
