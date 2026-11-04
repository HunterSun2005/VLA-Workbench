"""Record synchronized dual Franka/Robotiq SFT trajectories as LeRobot v3.0.

Run this module with the ROS Python 3.10 interpreter from the repository root::

    /usr/bin/python3 -m vla.datacollection.franka_robotiq_dual.auto_record
"""

from __future__ import annotations

from vla.terminal import (console_print as print, console_status, console_debug, terminal_session,
                          TerminalConfirmation, diagnostic_path, current_reporter)

import sys
import threading
import time
import traceback
from collections import Counter

import numpy as np
import rclpy
from franka_msgs.msg import FrankaRobotState
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

from vla.datacollection.franka_robotiq_single_left import schema as single_schema
from vla.datacollection.franka_robotiq_single_left.auto_record import (
    AsyncVideoWriter,
    CAMERA_READY_TIMEOUT_S,
    CommandPoseKinematics,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    KeyboardEvents,
    RealSenseCamera,
    prepare_image,
)
from vla.datacollection.franka_robotiq_single_left.v30_writer import V30DatasetWriter

from . import schema


OUTPUT_ROOT = schema.OUTPUT_ROOT
TASK = schema.TASK
FPS = schema.FPS
MAX_SYNC_DELTA_NS = schema.MAX_SYNC_DELTA_NS
MIN_FRAME_INTERVAL_NS = int(0.8 * 1_000_000_000 / FPS)
MIN_EPISODE_FRAMES = 2
ERROR_LOG_PATH = OUTPUT_ROOT / "meta/recorder_errors.log"


def report_exception(context: str) -> None:
    details = traceback.format_exc()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    message = f"[{timestamp}] {context}\n{details}"
    reporter = current_reporter()
    if reporter is not None:
        reporter.event(f"{context}；详细堆栈：{reporter.path}", level="error", details=details)
        return
    print(f"\n{message}", file=sys.stderr, flush=True)
    try:
        ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ERROR_LOG_PATH.open("a", encoding="utf-8") as stream:
            stream.write(message)
            if not message.endswith("\n"):
                stream.write("\n")
    except OSError as error:
        print(f"Could not write recorder error log: {error}", file=sys.stderr)


class DirectRealSenseCapture:
    def __init__(self, callback) -> None:
        self._cameras = {
            role: RealSenseCamera(role, serial, callback, error_reporter=report_exception)
            for role, serial in schema.CAMERA_SERIALS.items()
        }

    def start(self) -> None:
        started = []
        try:
            # Base frames drive recording, so start wrist cameras first.
            for role in [*(r for r in self._cameras if r != "base"), "base"]:
                camera = self._cameras[role]
                camera.start()
                started.append(camera)
            for camera in started:
                camera.wait_ready(CAMERA_READY_TIMEOUT_S)
        except BaseException:
            for camera in reversed(started):
                camera.close()
            raise

    def failures(self) -> dict[str, BaseException]:
        return {
            role: camera.error()
            for role, camera in self._cameras.items()
            if camera.error() is not None
        }

    def close(self) -> None:
        for camera in reversed(tuple(self._cameras.values())):
            camera.close()


class DualLeRobotRecorder(Node):
    def __init__(self) -> None:
        super().__init__("franka_dual_lerobot_v30_recorder")
        self.dataset = V30DatasetWriter(
            OUTPUT_ROOT,
            FPS,
            TASK,
            tuple(schema.CAMERA_FEATURES),
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            state_fields=schema.STATE_FIELDS,
            action_fields=schema.ACTION_FIELDS,
            robot_type="franka_robotiq_dual",
            info_metadata={
                "recording_schema_version": 3,
                "arm_order": [arm.name for arm in schema.ARMS],
            },
        )
        self._validate_existing_dataset()
        self.is_recording = False
        self._closed = False
        self._lock = threading.RLock()
        self._latest: dict[str, tuple[int, object]] = {}
        self._last_frame_time_ns: int | None = None
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._frame_times_ns: list[int] = []
        self._writers: dict[str, AsyncVideoWriter] = {}
        self._skip_reasons = Counter()
        self._last_saved_episode: int | None = None
        self._previous_pose: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        self._kinematics = {
            arm.name: CommandPoseKinematics(
                self._load_robot_description(arm),
                arm.base_link,
                arm.tip_link,
            )
            for arm in schema.ARMS
        }

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._subscription_handles = []
        for arm in schema.ARMS:
            for message_type, topic, key in (
                (FrankaRobotState, arm.robot_state_topic, arm.robot_state_key),
                (JointState, arm.command_joint_topic, arm.command_joint_key),
                (JointState, arm.gripper_state_topic, arm.gripper_state_key),
                (Float32, arm.gripper_command_topic, arm.gripper_command_key),
            ):
                self._subscription_handles.append(
                    self.create_subscription(message_type, topic, self._latest_callback(key), qos)
                )

        self._camera_capture = DirectRealSenseCapture(self._on_camera_frame)
        self._camera_capture.start()
        print("Streaming dual-arm LeRobot v3.0 absolute rot6d recorder started.")
        print(f"Dataset: {OUTPUT_ROOT}; existing episodes: {self.dataset.episode_count}")
        for arm in schema.ARMS:
            print(
                f"{arm.name}: robot_state={arm.robot_state_topic}, "
                f"final_arm_command={arm.command_joint_topic}, "
                f"gripper_state={arm.gripper_state_topic}, "
                f"gripper_command={arm.gripper_command_topic}"
            )
        print("[Right] start/save  [Left] discard current/latest  [Esc/q] save and exit")

    def _validate_existing_dataset(self) -> None:
        if self.dataset.frame_count == 0:
            return
        version = int(self.dataset.info.get("recording_schema_version", 0))
        if version < 3:
            raise RuntimeError(
                f"existing dataset {self.dataset.root} uses recording schema {version}, "
                "whose dual-arm TCP actions may come from stale o_t_ee_d values; refusing "
                "to mix it with FK-derived actions. Set DUAL_FRANKA_DATASET_ROOT to a new "
                "empty directory."
            )

    def _load_robot_description(self, arm: schema.ArmSchema) -> str:
        from rcl_interfaces.srv import GetParameters
        from rclpy.parameter import parameter_value_to_python

        service = f"{arm.robot_state_publisher.rstrip('/')}/get_parameters"
        client = self.create_client(GetParameters, service)
        if not client.wait_for_service(timeout_sec=30.0):
            raise RuntimeError(f"parameter service is unavailable: {service}")
        request = GetParameters.Request(names=["robot_description"])
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        response = future.result()
        if response is None or not response.values:
            raise RuntimeError(f"failed to read robot_description from {service}")
        description = parameter_value_to_python(response.values[0])
        if not isinstance(description, str) or not description:
            raise RuntimeError(f"robot_description is empty: {service}")
        return description

    @property
    def required_latest_keys(self) -> tuple[str, ...]:
        arm_keys = tuple(
            key
            for arm in schema.ARMS
            for key in (
                arm.robot_state_key,
                arm.command_joint_key,
                arm.gripper_state_key,
                arm.gripper_command_key,
            )
        )
        return (*(role for role in schema.CAMERA_SERIALS if role != "base"), *arm_keys)

    def _store_latest(self, key: str, message: object, timestamp_ns: int | None = None) -> None:
        with self._lock:
            self._latest[key] = (timestamp_ns or time.monotonic_ns(), message)

    def _latest_callback(self, key: str):
        return lambda message: self._store_latest(key, message)

    def _on_camera_frame(self, role: str, timestamp_ns: int, rgb: np.ndarray) -> None:
        rgb = prepare_image(rgb)
        if role == "base":
            self._store_latest(role, rgb, timestamp_ns)
            self._on_base_image(rgb, timestamp_ns)
        else:
            self._store_latest(role, rgb, timestamp_ns)

    def _on_base_image(self, base_rgb: np.ndarray, now_ns: int) -> None:
        with self._lock:
            if not self.is_recording:
                return
            if (
                self._last_frame_time_ns is not None
                and now_ns - self._last_frame_time_ns < MIN_FRAME_INTERVAL_NS
            ):
                return

            messages: dict[str, object] = {}
            for key in self.required_latest_keys:
                latest = self._latest.get(key)
                if latest is None:
                    self._skip_reasons[f"missing {key}"] += 1
                    return
                timestamp_ns, message = latest
                if now_ns - timestamp_ns > MAX_SYNC_DELTA_NS:
                    self._skip_reasons[f"stale {key}"] += 1
                    return
                messages[key] = message

            try:
                state_parts = {}
                action_parts = {}
                current_poses = {}
                for arm in schema.ARMS:
                    robot_state = messages[arm.robot_state_key]
                    position, rotation = single_schema.measured_pose_from_robot_state(robot_state)
                    previous = self._previous_pose.get(arm.name)
                    velocity = single_schema.base_tcp_velocity(
                        previous[0] if previous else None,
                        previous[1] if previous else None,
                        previous[2] if previous else None,
                        position,
                        rotation,
                        now_ns,
                    )
                    state_parts[arm.name] = schema.arm_state_vector(
                        arm,
                        robot_state,
                        messages[arm.gripper_state_key],
                        velocity,
                        position,
                        rotation,
                    )
                    flange_position, flange_rotation = self._kinematics[arm.name].pose(
                        messages[arm.command_joint_key]
                    )
                    commanded_position, commanded_rotation = single_schema.tcp_pose_from_flange(
                        flange_position,
                        flange_rotation,
                        robot_state,
                    )
                    action_parts[arm.name] = schema.arm_action_vector(
                        commanded_position,
                        commanded_rotation,
                        messages[arm.gripper_command_key],
                    )
                    current_poses[arm.name] = (position.copy(), rotation.copy(), now_ns)

                state = schema.dual_state_vector(state_parts)
                action = schema.dual_action_vector(action_parts)
                images = {
                    "observation.images.base": base_rgb,
                    **{f"observation.images.{role}": messages[role] for role in schema.CAMERA_SERIALS if role != "base"},
                }
                for key, writer in self._writers.items():
                    writer.write(images[key])
                self._states.append(state)
                self._actions.append(action)
                self._frame_times_ns.append(now_ns)
                self._previous_pose = current_poses
                self._last_frame_time_ns = now_ns
                if len(self._states) % FPS == 0:
                    console_status(f"RECORDING frames={len(self._states)} elapsed={len(self._states)/FPS:.1f}s")
                    console_debug(f"Recorded {len(self._states)} frames; encoders={self._diagnostics()}")
            except (KeyError, ValueError, RuntimeError) as error:
                self._skip_reasons[f"invalid frame: {error}"] += 1

    def _diagnostics(self) -> dict:
        return {key: writer.diagnostics() for key, writer in self._writers.items()}

    def _print_skip_summary(self) -> None:
        if self._skip_reasons:
            print(
                "Skipped base frames: "
                + ", ".join(
                    f"{key}={value}" for key, value in self._skip_reasons.most_common()
                )
            )

    def missing_publishers(self) -> list[str]:
        topics = [
            topic
            for arm in schema.ARMS
            for topic in (
                arm.robot_state_topic,
                arm.command_joint_topic,
                arm.gripper_state_topic,
                arm.gripper_command_topic,
            )
        ]
        return [topic for topic in topics if self.count_publishers(topic) == 0]

    def unconsumed_command_topics(self) -> list[str]:
        # The recorder itself is one subscriber. A second subscriber proves the
        # final command is also consumed by the live controller/driver.
        return [
            topic
            for arm in schema.ARMS
            for topic in (arm.command_joint_topic, arm.gripper_command_topic)
            if self.count_subscribers(topic) < 2
        ]

    def start_recording(self) -> None:
        with self._lock:
            if self.is_recording:
                return
            failures = self._camera_capture.failures()
            missing = self.missing_publishers()
            unconsumed = self.unconsumed_command_topics()
            if failures or missing or unconsumed:
                print(
                    "Recording blocked; "
                    f"camera_failures={failures}, missing_topics={missing}, "
                    f"commands_without_hardware_consumer={unconsumed}"
                )
                return
            episode_index = self.dataset.next_episode_index()
            self.dataset.discard_uncommitted_files(episode_index)
            writers = {}
            try:
                for key in schema.CAMERA_FEATURES:
                    writers[key] = AsyncVideoWriter(self.dataset.video_path(episode_index, key))
            except BaseException:
                for writer in writers.values():
                    writer.abort()
                self.dataset.discard_uncommitted_files(episode_index)
                raise
            self._writers = writers
            self._states = []
            self._actions = []
            self._frame_times_ns = []
            self._skip_reasons.clear()
            self._last_frame_time_ns = None
            self._previous_pose.clear()
            self.is_recording = True
            print(f"\nRecording dual-arm episode {episode_index} with streaming H.264 encoding.")

    def _close_writers(self) -> None:
        first_error = None
        for writer in self._writers.values():
            try:
                writer.close()
            except BaseException as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error

    def _abort_writers(self) -> None:
        for writer in self._writers.values():
            try:
                writer.abort()
            except BaseException:
                pass

    def _quality_summary(self, episode_index: int, save_seconds: float) -> None:
        times = np.asarray(self._frame_times_ns, dtype=np.float64)
        fps = 0.0
        if len(times) > 1 and times[-1] > times[0]:
            fps = (len(times) - 1) / ((times[-1] - times[0]) / 1e9)
        actions = np.asarray(self._actions).reshape(len(self._actions), len(schema.ARMS), 10)
        reports = schema.validate_episode_features(
            np.asarray(self._states),
            np.asarray(self._actions),
        )
        action_spans = {
            name: (
                float(report["translation_span_m"]),
                float(report["rotation_span_rad"]),
            )
            for name, report in reports.items()
        }
        gripper_ranges = {
            arm.name: (
                float(actions[:, index, 9].min()),
                float(actions[:, index, 9].max()),
            )
            for index, arm in enumerate(schema.ARMS)
        }
        print(
            f"Episode {episode_index}: frames={len(actions)}, avg_fps={fps:.2f}, "
            f"save_time={save_seconds:.2f}s, "
            f"action_spans={action_spans}, "
            f"gripper_commands={gripper_ranges}, encoders={self._diagnostics()}"
        )
        self._print_skip_summary()

    def stop_recording(self, discard: bool = False) -> None:
        with self._lock:
            if not self.is_recording:
                if discard:
                    print("No active recording to discard.")
                return
            self.is_recording = False
            episode_index = self.dataset.next_episode_index()
            try:
                if discard or len(self._states) < MIN_EPISODE_FRAMES:
                    self._abort_writers()
                    self.dataset.discard_uncommitted_files(episode_index)
                    print(f"Discarded episode {episode_index} (frames={len(self._states)}).")
                    self._print_skip_summary()
                else:
                    started = time.monotonic()
                    schema.validate_episode_features(
                        np.asarray(self._states),
                        np.asarray(self._actions),
                    )
                    self._close_writers()
                    image_stats = {
                        key: writer.image_stats.as_dict()
                        for key, writer in self._writers.items()
                    }
                    saved_index = self.dataset.save_episode(
                        np.asarray(self._states),
                        np.asarray(self._actions),
                        image_stats,
                    )
                    self._last_saved_episode = saved_index
                    elapsed = time.monotonic() - started
                    print(
                        f"Saved episode {saved_index}; data={self.dataset.data_path(saved_index)}"
                    )
                    self._quality_summary(saved_index, elapsed)
            except BaseException:
                self._abort_writers()
                self.dataset.discard_uncommitted_files(episode_index)
                raise
            finally:
                self._writers = {}
                self._states = []
                self._actions = []
                self._frame_times_ns = []

    def discard_last_saved_episode(self) -> None:
        with self._lock:
            if self.is_recording:
                self.stop_recording(discard=True)
                return
            if self._last_saved_episode is None:
                print("No episode saved in this recorder session to discard.")
                return
            removed = self.dataset.discard_latest_episode()
            self._last_saved_episode = None
            print(f"Discarded latest episode {removed}.")

    def close(self, save_active: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            if self.is_recording:
                self.stop_recording(discard=not save_active)
            self._closed = True
        self._camera_capture.close()


@terminal_session("gello-dual-record")
def main() -> None:
    rclpy.init()
    recorder = DualLeRobotRecorder()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(recorder)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    confirmation = TerminalConfirmation()
    keyboard = KeyboardEvents(confirmation=confirmation)
    try:
        should_quit = False
        while not should_quit:
            confirmation.expire()
            for event in confirmation.prioritize(keyboard.consume()):
                if event == "force_quit":
                    should_quit = True
                    break  # EOF/forced exit must not return to the command loop after a save failure.
                try:
                    states, saved, recording = recorder._states, recorder._last_saved_episode, recorder.is_recording
                    event = confirmation.route(event,
                        destructive=event == "left" and (recording or saved is not None),
                        valid=lambda frames=states, previous=saved, active=recording: recorder._states is frames and recorder._last_saved_episode == previous and recorder.is_recording == active,
                        label="丢弃当前未保存回合" if recording else "删除本次会话最近已保存回合")
                    if event is None:
                        continue
                    if event == "right":
                        (
                            recorder.stop_recording()
                            if recorder.is_recording
                            else recorder.start_recording()
                        )
                    elif event == "left":
                        recorder.discard_last_saved_episode()
                    elif event == "quit":
                        recorder.close(save_active=True)
                        should_quit = True
                        break
                except Exception:
                    report_exception(
                        f"keyboard command {event!r} failed; recorder remains active"
                    )
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("Interrupted; saving active recording before exit.")
    finally:
        try:
            recorder.close(save_active=True)
        finally:
            keyboard.close()
            executor.shutdown()
            spin_thread.join(timeout=5.0)
            recorder.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
