"""Record absolute Franka/Robotiq trajectories directly as LeRobot v3.0.

Run this module with the ROS Python 3.10 interpreter from the repository root::

    /usr/bin/python3 -m vla.datacollection.franka_robotiq_single_left.auto_record
"""

from __future__ import annotations

from vla.terminal import (console_print as print, console_status, console_debug, terminal_session,
                          TerminalConfirmation, diagnostic_path, current_reporter)

import os
import queue
import select
import shutil
import subprocess
import sys
import termios
import threading
import time
import traceback
import tty
from collections import Counter, deque

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from franka_msgs.msg import FrankaRobotState
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

from vla.datacollection.start_teleop.franka_kinematics import build_kinematic_model

from . import schema
from .v30_writer import V30DatasetWriter
from vla.datacollection.gripper_contract import gello_recording_metadata, gripper_consumer_requirement


OUTPUT_ROOT = schema.OUTPUT_ROOT
TASK = schema.TASK
FPS = schema.FPS
MAX_SYNC_DELTA_NS = schema.MAX_SYNC_DELTA_NS
MIN_FRAME_INTERVAL_NS = int(0.8 * 1_000_000_000 / FPS)
MIN_EPISODE_FRAMES = 2
VIDEO_CRF = 23
VIDEO_PRESET = "veryfast"
VIDEO_QUEUE_SIZE = 8
IMAGE_STATS_STRIDE = 5
ERROR_LOG_PATH = OUTPUT_ROOT / "meta/recorder_errors.log"
CAMERA_CAPTURE_WIDTH = 640
CAMERA_CAPTURE_HEIGHT = 480
CAMERA_CAPTURE_FPS = 30
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
CAMERA_READY_TIMEOUT_S = 10.0
CAMERA_SERIALS = {
    "base": os.environ.get("BASE_CAMERA_SERIAL", "260722303785"),
    "left_wrist": os.environ.get("LEFT_WRIST_CAMERA_SERIAL", "346222073620"),
}
if os.environ.get("SIDE_CAMERA_SERIAL"):
    CAMERA_SERIALS["side"] = os.environ["SIDE_CAMERA_SERIAL"]
CAMERA_FEATURES = {
    "observation.images.base": "base",
    "observation.images.left_wrist": "left wrist",
}
if "side" in CAMERA_SERIALS:
    CAMERA_FEATURES["observation.images.side"] = "side"
TOPIC_TYPES = {
    schema.TOPIC_ARM_COMMAND: JointState,
    schema.TOPIC_GRIPPER: JointState,
    schema.TOPIC_GRIPPER_COMMAND: Float32,
    schema.TOPIC_ROBOT_STATE: FrankaRobotState,
}
LATEST_KEYS = {
    schema.TOPIC_ARM_COMMAND: "arm_command",
    schema.TOPIC_GRIPPER: "gripper",
    schema.TOPIC_GRIPPER_COMMAND: "teleop_gripper_command",
    schema.TOPIC_ROBOT_STATE: "robot_state",
}
REQUIRED_LATEST_KEYS = (*(role for role in CAMERA_SERIALS if role != "base"), *LATEST_KEYS.values())

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


class ImageRunningStats:
    def __init__(self) -> None:
        self.samples = 0
        self.pixels = 0
        self.total = np.zeros(3, dtype=np.float64)
        self.total_square = np.zeros(3, dtype=np.float64)
        self.minimum = np.full(3, np.inf, dtype=np.float64)
        self.maximum = np.full(3, -np.inf, dtype=np.float64)

    def update(self, rgb: np.ndarray) -> None:
        values = cv2.resize(rgb, (64, 48), interpolation=cv2.INTER_AREA)
        values = values.reshape(-1, 3).astype(np.float64) / 255.0
        self.samples += 1
        self.pixels += len(values)
        self.total += values.sum(axis=0)
        self.total_square += np.square(values).sum(axis=0)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def as_dict(self) -> dict[str, np.ndarray]:
        if self.pixels == 0:
            raise ValueError("cannot serialize empty image statistics")
        mean = self.total / self.pixels
        variance = np.maximum(self.total_square / self.pixels - np.square(mean), 0.0)

        def shaped(value: np.ndarray) -> np.ndarray:
            return np.asarray(value, dtype=np.float32).reshape(3, 1, 1)

        # Quantiles are approximated by extrema/mean because image frames are sampled
        # online. Numeric state/action quantiles are always computed exactly.
        return {
            "min": shaped(self.minimum),
            "max": shaped(self.maximum),
            "mean": shaped(mean),
            "std": shaped(np.sqrt(variance)),
            "count": np.array([self.samples], dtype=np.int64),
            "q01": shaped(self.minimum),
            "q10": shaped(self.minimum),
            "q50": shaped(mean),
            "q90": shaped(self.maximum),
            "q99": shaped(self.maximum),
        }


class FfmpegVideoWriter:
    def __init__(self, path, width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT, fps: int = FPS) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required for streaming video capture")
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("video width, height, and fps must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.width}x{self.height}",
                "-r", str(self.fps), "-i", "-", "-an", "-vcodec", "libx264",
                "-preset", VIDEO_PRESET, "-pix_fmt", "yuv420p", "-crf", str(VIDEO_CRF),
                "-movflags", "+faststart", str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, rgb: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        rgb = np.asarray(rgb)
        expected = (self.height, self.width, 3)
        if rgb.shape != expected or rgb.dtype != np.uint8:
            raise ValueError(f"video frame must be uint8 RGB with shape {expected}, got {rgb.shape}/{rgb.dtype}")
        self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
            self.process.stdin = None
        return_code = self.process.wait()
        stderr = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
        if return_code:
            raise RuntimeError(f"ffmpeg failed for {self.path}: {stderr.strip()}")

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        if self.process.stdin is not None:
            self.process.stdin.close()
            self.process.stdin = None


class AsyncVideoWriter:
    def __init__(
        self,
        path,
        width: int = IMAGE_WIDTH,
        height: int = IMAGE_HEIGHT,
        fps: int = FPS,
    ) -> None:
        self.writer = FfmpegVideoWriter(path, width=width, height=height, fps=fps)
        self.image_stats = ImageRunningStats()
        self._queue = queue.Queue(maxsize=VIDEO_QUEUE_SIZE)
        self._lock = threading.Lock()
        self._closed = False
        self._abort = threading.Event()
        self._error = None
        self._frame_count = 0
        self._blocked_count = 0
        self._max_queue_size = 0
        self._thread = threading.Thread(target=self._worker, name=f"ffmpeg-{path.parent.name}", daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            while not self._abort.is_set():
                frame = self._queue.get()
                if frame is None:
                    return
                self.writer.write(frame)
                if self._frame_count % IMAGE_STATS_STRIDE == 0:
                    self.image_stats.update(frame)
                self._frame_count += 1
        except BaseException as error:
            with self._lock:
                self._error = error

    def _raise_if_error(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"video writer failed: {error}") from error

    def write(self, rgb: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed video writer")
        rgb = np.asarray(rgb)
        expected = (self.writer.height, self.writer.width, 3)
        if rgb.shape != expected or rgb.dtype != np.uint8:
            raise ValueError(
                f"video frame must be uint8 RGB with shape {expected}, "
                f"got {rgb.shape}/{rgb.dtype}"
            )
        blocked = False
        while True:
            self._raise_if_error()
            try:
                self._queue.put(rgb, timeout=0.05)
                break
            except queue.Full:
                blocked = True
        with self._lock:
            self._blocked_count += int(blocked)
            self._max_queue_size = max(self._max_queue_size, self._queue.qsize())

    def diagnostics(self) -> dict[str, int]:
        with self._lock:
            return {"queue": self._queue.qsize(), "blocked": self._blocked_count, "max_queue": self._max_queue_size}

    def close(self) -> None:
        if self._closed:
            self._raise_if_error()
            return
        self._closed = True
        while self._thread.is_alive():
            self._raise_if_error()
            try:
                self._queue.put(None, timeout=0.05)
                break
            except queue.Full:
                continue
        self._thread.join()
        self._raise_if_error()
        self.writer.close()

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._abort.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self.writer.abort()
        self._thread.join(timeout=2.0)


class KeyboardEvents:
    def __init__(self, keymap: dict[bytes, str] | None = None, *, confirmation=None) -> None:
        self.confirmation = confirmation
        self._keymap = keymap or {}
        self._events = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(self._fd) if os.isatty(self._fd) else None
        if self._old_termios is not None:
            tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _push(self, event: str) -> None:
        with self._lock:
            self._events.append(event)

    def _worker(self) -> None:
        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.1)
            if not ready:
                continue
            char = os.read(self._fd, 1)
            if self.confirmation is not None and char in (b"", b"\x03", b"\x04"):
                self._push("force_quit")
                return
            if char == b"\x1b":
                suffix = os.read(self._fd, 2) if select.select([self._fd], [], [], 0.05)[0] else b""
                if self.confirmation is not None and self.confirmation.pending and not suffix:
                    self._push("cancel")
                    continue
                self._push("right" if suffix == b"[C" else "left" if suffix == b"[D" else "quit")
            elif self.confirmation is not None and self.confirmation.key_event(char):
                self._push(self.confirmation.key_event(char))
            elif char.lower() in self._keymap:
                self._push(self._keymap[char.lower()])
            elif char in {b"q", b"Q", b"\x03", b"\x04", b""}:
                self._push("quit")
            elif char == b" ":
                self._push("intervention")
            elif char in {b"s", b"S"}:
                self._push("success")
            elif char in {b"f", b"F"}:
                self._push("failure")
            elif char in {b"r", b"R"}:
                self._push("phase")

    def consume(self) -> list[str]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        if self._old_termios is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)


class RealSenseCamera:
    def __init__(self, role: str, serial: str, callback, error_reporter=report_exception) -> None:
        self.role = role
        self.serial = serial
        self._callback = callback
        self._error_reporter = error_reporter
        self._pipeline = rs.pipeline()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._error = None
        self._started = False
        self._thread = threading.Thread(target=self._worker, name=f"realsense-{role}", daemon=True)

    def start(self) -> None:
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, CAMERA_CAPTURE_WIDTH, CAMERA_CAPTURE_HEIGHT, rs.format.rgb8, CAMERA_CAPTURE_FPS)
        profile = self._pipeline.start(config)
        self._started = True
        name = profile.get_device().get_info(rs.camera_info.name)
        print(f"RealSense {self.role}: {name}, serial={self.serial}")
        self._thread.start()

    def _worker(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError:
                    if self._stop.is_set():
                        return
                    raise
                color = frames.get_color_frame()
                if color:
                    rgb = np.asanyarray(color.get_data()).copy()
                    self._ready.set()
                    self._callback(self.role, time.monotonic_ns(), rgb)
        except BaseException as error:
            with self._lock:
                self._error = error
            self._error_reporter(f"RealSense {self.role} capture failed")

    def error(self):
        with self._lock:
            return self._error

    def wait_ready(self, timeout: float) -> None:
        if not self._ready.wait(timeout):
            error = self.error()
            if error is not None:
                raise RuntimeError(f"RealSense {self.role} failed: {error}") from error
            raise TimeoutError(f"timed out waiting for RealSense {self.role} serial={self.serial}")

    def close(self) -> None:
        self._stop.set()
        if self._started:
            try:
                self._pipeline.stop()
            except RuntimeError:
                pass
        self._thread.join(timeout=2.0)


class DirectRealSenseCapture:
    def __init__(self, callback) -> None:
        self._cameras = {role: RealSenseCamera(role, serial, callback) for role, serial in CAMERA_SERIALS.items()}

    def start(self) -> None:
        started = []
        try:
            for role in [*(r for r in self._cameras if r != "base"), "base"]:
                self._cameras[role].start()
                started.append(self._cameras[role])
            for camera in started:
                camera.wait_ready(CAMERA_READY_TIMEOUT_S)
        except BaseException:
            for camera in reversed(started):
                camera.close()
            raise

    def failures(self) -> dict:
        return {role: camera.error() for role, camera in self._cameras.items() if camera.error() is not None}

    def close(self) -> None:
        for camera in reversed(tuple(self._cameras.values())):
            camera.close()


def prepare_image(
    rgb: np.ndarray, image_size: tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH)
) -> np.ndarray:
    """Fit the entire camera view into the requested (height, width)."""
    from vla.core.images import resize_rgb_with_pad

    return resize_rgb_with_pad(rgb, *image_size)


class CommandPoseKinematics:
    """Convert the final joint command consumed by the controller into TCP pose."""

    def __init__(
        self,
        robot_description: str,
        base_link: str = "left_fr3_link0",
        tip_link: str = "left_fr3_link8",
    ) -> None:
        import PyKDL

        self._kdl = PyKDL
        self.model = build_kinematic_model(
            robot_description,
            base_link,
            tip_link,
            0.0,
        )
        self._solver = PyKDL.ChainFkSolverPos_recursive(self.model.chain)

    def pose(self, message) -> tuple[np.ndarray, np.ndarray]:
        joints = schema.commanded_joint_positions(message, self.model.joint_names)
        values = self._kdl.JntArray(len(joints))
        for index, value in enumerate(joints):
            values[index] = float(value)
        frame = self._kdl.Frame()
        if self._solver.JntToCart(values, frame) < 0:
            raise RuntimeError("KDL forward kinematics failed for final arm command")
        position = np.array([frame.p.x(), frame.p.y(), frame.p.z()], dtype=np.float64)
        rotation = np.array(
            [[frame.M[row, column] for column in range(3)] for row in range(3)],
            dtype=np.float64,
        )
        return position, rotation


class DirectLeRobotRecorder(Node):
    def __init__(self) -> None:
        super().__init__("franka_single_left_lerobot_v30_recorder")
        self.dataset = V30DatasetWriter(
            OUTPUT_ROOT, FPS, TASK, tuple(CAMERA_FEATURES), IMAGE_HEIGHT, IMAGE_WIDTH,
            info_metadata=gello_recording_metadata({"left": schema.TOPIC_GRIPPER_COMMAND}),
            strict_metadata=True,
        )
        self._validate_existing_dataset()
        self.is_recording = False
        self._closed = False
        self._lock = threading.RLock()
        self._latest = {}
        self._last_frame_time_ns = None
        self._states = []
        self._actions = []
        self._frame_times_ns = []
        self._writers = {}
        self._skip_reasons = Counter()
        self._last_saved_episode = None
        self._previous_position = None
        self._previous_rotation = None
        self._previous_pose_time_ns = None
        self._kinematics = CommandPoseKinematics(self._load_robot_description())
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._subscription_handles = [
            self.create_subscription(msg_type, topic, self._latest_callback(LATEST_KEYS[topic]), qos)
            for topic, msg_type in TOPIC_TYPES.items()
        ]
        self._camera_capture = DirectRealSenseCapture(self._on_camera_frame)
        self._camera_capture.start()
        print("Streaming LeRobot v3.0 absolute rot6d recorder started.")
        print(f"Dataset: {OUTPUT_ROOT}; existing episodes: {self.dataset.episode_count}")
        print(
            "ROS inputs: "
            f"robot_state={schema.TOPIC_ROBOT_STATE}, "
            f"final_arm_command={schema.TOPIC_ARM_COMMAND}, "
            f"gripper_state={schema.TOPIC_GRIPPER}, "
            f"final_gripper_command={schema.TOPIC_GRIPPER_COMMAND}"
        )
        print("[Right] start/save  [Left] discard current/latest  [Esc/q] save and exit")

    def _validate_existing_dataset(self) -> None:
        if self.dataset.frame_count == 0:
            return
        import json

        stats_path = self.dataset.root / "meta/stats.json"
        if not stats_path.is_file():
            raise RuntimeError(f"existing dataset has frames but no stats: {stats_path}")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        try:
            action_min = np.asarray(stats["action"]["min"], dtype=np.float64)
            action_max = np.asarray(stats["action"]["max"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"existing dataset has invalid action stats: {stats_path}") from error
        if action_min.shape != (len(schema.ACTION_FIELDS),) or action_max.shape != action_min.shape:
            raise RuntimeError(f"existing dataset has invalid action stats shape: {stats_path}")
        translation_span = float(np.linalg.norm(action_max[:3] - action_min[:3]))
        rot6d_span = float(np.linalg.norm(action_max[3:9] - action_min[3:9]))
        if (
            translation_span < schema.ACTION_MIN_TRANSLATION_SPAN_M
            and rot6d_span < schema.ACTION_MIN_ROTATION_SPAN_RAD
        ):
            raise RuntimeError(
                f"existing dataset {self.dataset.root} contains degenerate constant TCP actions "
                f"across {self.dataset.frame_count} frames; refusing to append. Set "
                "LEFT_FRANKA_DATASET_ROOT to a new empty directory."
            )

    def _store_latest(self, key, message, timestamp_ns=None) -> None:
        with self._lock:
            self._latest[key] = (timestamp_ns or time.monotonic_ns(), message)

    def _load_robot_description(self) -> str:
        from rcl_interfaces.srv import GetParameters
        from rclpy.parameter import parameter_value_to_python

        service = "/left/robot_state_publisher/get_parameters"
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
            raise RuntimeError("robot_description is empty")
        return description

    def _latest_callback(self, key):
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
            if self._last_frame_time_ns is not None and now_ns - self._last_frame_time_ns < MIN_FRAME_INTERVAL_NS:
                return
            messages = {}
            for key in REQUIRED_LATEST_KEYS:
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
                robot_state = messages["robot_state"]
                position, rotation = schema.measured_pose_from_robot_state(robot_state)
                velocity = schema.base_tcp_velocity(
                    self._previous_position, self._previous_rotation, self._previous_pose_time_ns,
                    position, rotation, now_ns,
                )
                state = schema.state_vector(robot_state, messages["gripper"], velocity, position, rotation)
                flange_position, flange_rotation = self._kinematics.pose(messages["arm_command"])
                commanded_position, commanded_rotation = schema.tcp_pose_from_flange(
                    flange_position,
                    flange_rotation,
                    robot_state,
                )
                action = schema.action_vector_from_pose(
                    commanded_position,
                    commanded_rotation,
                    messages["teleop_gripper_command"],
                )
                images = {
                    "observation.images.base": base_rgb,
                    **{f"observation.images.{role}": messages[role] for role in CAMERA_SERIALS if role != "base"},
                }
                for key, writer in self._writers.items():
                    writer.write(images[key])
                self._states.append(state)
                self._actions.append(action)
                self._frame_times_ns.append(now_ns)
                self._previous_position = position.copy()
                self._previous_rotation = rotation.copy()
                self._previous_pose_time_ns = now_ns
                self._last_frame_time_ns = now_ns
                if len(self._states) % FPS == 0:
                    console_status(f"RECORDING frames={len(self._states)} elapsed={len(self._states)/FPS:.1f}s")
                    console_debug(f"Recorded {len(self._states)} frames; encoders={self._diagnostics()}")
            except (KeyError, ValueError, RuntimeError) as error:
                self._skip_reasons[f"invalid frame: {error}"] += 1

    def _diagnostics(self) -> dict:
        return {key: writer.diagnostics() for key, writer in self._writers.items()}

    def missing_publishers(self) -> list[str]:
        return [topic for topic in TOPIC_TYPES if self.count_publishers(topic) == 0]

    def unconsumed_command_topics(self) -> list[str]:
        requirements = [(schema.TOPIC_ARM_COMMAND, 2), gripper_consumer_requirement(schema.TOPIC_GRIPPER_COMMAND)]
        return [topic for topic, minimum in requirements
                if self.count_subscribers(topic) < minimum]

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
                for key in CAMERA_FEATURES:
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
            self._previous_position = None
            self._previous_rotation = None
            self._previous_pose_time_ns = None
            self.is_recording = True
            print(f"\nRecording episode {episode_index} with streaming H.264 encoding.")

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
        actions = np.asarray(self._actions)
        report = schema.validate_episode_features(np.asarray(self._states), actions)
        print(
            f"Episode {episode_index}: frames={len(actions)}, avg_fps={fps:.2f}, save_time={save_seconds:.2f}s, "
            f"action_translation_span={report['translation_span_m']:.4f}m, "
            f"action_rotation_span={report['rotation_span_rad']:.4f}rad, "
            f"gripper_command=[{actions[:, 9].min():.4f}, {actions[:, 9].max():.4f}], encoders={self._diagnostics()}"
        )
        console_debug(f"State ranges ({len(schema.STATE_FIELDS)}D): {np.asarray(report['state_range']).tolist()}")
        console_debug(f"Action ranges ({len(schema.ACTION_FIELDS)}D): {np.asarray(report['action_range']).tolist()}")
        if self._skip_reasons:
            print("Skipped base frames: " + ", ".join(f"{key}={value}" for key, value in self._skip_reasons.most_common()))

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
                else:
                    started = time.monotonic()
                    schema.validate_episode_features(np.asarray(self._states), np.asarray(self._actions))
                    self._close_writers()
                    image_stats = {key: writer.image_stats.as_dict() for key, writer in self._writers.items()}
                    saved_index = self.dataset.save_episode(np.asarray(self._states), np.asarray(self._actions), image_stats)
                    self._last_saved_episode = saved_index
                    elapsed = time.monotonic() - started
                    print(f"Saved episode {saved_index}; data={self.dataset.data_path(saved_index)}")
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


@terminal_session("gello-single-record")
def main() -> None:
    rclpy.init()
    recorder = DirectLeRobotRecorder()
    executor = MultiThreadedExecutor(num_threads=4)
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
                        recorder.stop_recording() if recorder.is_recording else recorder.start_recording()
                    elif event == "left":
                        recorder.discard_last_saved_episode()
                    elif event == "quit":
                        recorder.close(save_active=True)
                        should_quit = True
                        break
                except Exception:
                    report_exception(f"keyboard command {event!r} failed; recorder remains active")
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
