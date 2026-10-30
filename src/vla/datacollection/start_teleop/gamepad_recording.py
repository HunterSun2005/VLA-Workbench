"""Camera-rate human demonstrations, integrated from successful servo commands.

No policy imports or hardware connections. Disk IO belongs to EpisodeWorker.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import uuid

import numpy as np
import yaml

from vla.datacollection.franka_robotiq_single_left import schema
from vla.datacollection.franka_robotiq_single_left.v30_writer import (
    V30DatasetWriter, atomic_json, _sync_directory,
)
from vla.datacollection.franka_robotiq_rollout.common import FfmpegVideoWriter, ImageRunningStats
from vla.datacollection.start_teleop.gamepad_config import load_config
from vla.core.writer_lock import WriterLease
from vla.robots.franka_robotiq.robot import _axis_angle_matrix


VERSION = 1
CAMERAS = ("observation.images.base", "observation.images.left_wrist")
FEATURES = {
    f"complementary_info.{name}": {"dtype": dtype, "shape": [1], "names": [name]}
    for name, dtype in {
        "phase": "float32", "observed_at_ns": "int64", "interval_end_ns": "int64",
        "base_image_at_ns": "int64", "left_wrist_image_at_ns": "int64",
    }.items()
}


@dataclass(frozen=True)
class RecordingConfig:
    fps: int = 30
    critical_button: int = 5
    queue_size: int = 256
    min_frames: int = 12
    max_duration_s: int = 600

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < (0 if name == "critical_button" else 1):
                raise ValueError(f"recording.{name} must be a valid integer")
        if self.min_frames < 12:
            raise ValueError("recording.min_frames must be >= 12")


def load_recording_config(path):
    control = load_config(Path(path), recording=True)
    recording = RecordingConfig(**yaml.safe_load(Path(path).read_text()).get("recording", {}))
    if recording.fps != control.robot.camera_fps or recording.fps != 30:
        raise ValueError("recording and cameras must both use 30 FPS")
    if control.control.control_rate != 100 or control.robot.control_frequency != 100 or control.robot.image_size != 224:
        raise ValueError("recording requires 100 Hz control and 224-pixel images")
    occupied = [control.speed_modes.toggle_button, control.pose_shortcuts.view_button,
                control.pose_shortcuts.menu_button, control.mapping.a_button, control.mapping.b_button,
                control.mapping.x_button, control.mapping.y_button]
    if recording.critical_button in occupied:
        raise ValueError("critical button must not share a motion, speed or pose shortcut button")
    if recording.critical_button == control.pose_shortcuts.yaw_reset_button:
        raise ValueError("critical button must not share the yaw reset button")
    if not all((control.robot.human_linear_velocity_control, control.robot.human_angular_velocity_control,
                control.robot.human_gripper_immediate_hold)):
        raise ValueError("recording requires direct human velocity control and gripper hold")
    if len(set(control.robot.camera_serials.values())) != 2:
        raise ValueError("recording requires two distinct cameras")
    return control, recording


def dataset_contract(control, recording, task):
    if not task or not task.strip():
        raise ValueError("a nonempty task is required")
    settings = asdict(control)
    # Disabled shortcuts must not invalidate pre-existing dataset contracts.
    if settings["pose_shortcuts"]["yaw_reset_button"] is None:
        settings["pose_shortcuts"].pop("yaw_reset_button")
    settings["robot"].pop("calibration_dir", None)
    return json.loads(json.dumps({
        "version": VERSION, "collection_kind": "human_demonstration", "task": task,
        "settings": settings, "recording": asdict(recording),
        "state_fields": schema.STATE_FIELDS, "action_fields": schema.ACTION_FIELDS,
        "action_encoding": "interval_integrated_base_velocity_absolute_tcp_rot6d_v1",
        "state_gripper_unit": "metres", "action_gripper_unit": "open_fraction",
        "camera_clock": "host_monotonic_receive", "source": "demonstration",
    }, default=str))


class CriticalPhase:
    def __init__(self, button):
        self.button = button
        self.reset()

    def reset(self):
        self.state, self.ready = "before", False

    @property
    def active(self):
        return self.state == "critical"

    def update(self, buttons):
        if self.button >= len(buttons):
            raise ValueError("Joy message is missing the critical button")
        pressed = bool(buttons[self.button])
        if not pressed:
            self.ready = True
            return None
        if not self.ready:
            return None
        self.ready = False
        if self.state == "after":
            return "Critical re-entry rejected: one contiguous interval per episode"
        self.state = "critical" if self.state == "before" else "after"
        return f"Critical {'ON' if self.active else 'OFF'} from the next recording interval"


def validate_snapshot(snapshot):
    pose = np.asarray(snapshot.pose)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("invalid measured pose")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1):
        raise ValueError("invalid measured rotation")
    if snapshot.observed_at_ns <= 0 or not 0 <= snapshot.gripper_open_fraction <= 1:
        raise ValueError("invalid observation time/gripper")
    if set(snapshot.images) != {"base", "left_wrist"} or set(snapshot.image_timestamps) != {"base", "left_wrist"}:
        raise ValueError("both camera images/timestamps are required")
    for image in snapshot.images.values():
        if image.shape != (224, 224, 3) or image.dtype != np.uint8:
            raise ValueError("images must be 224x224 RGB uint8")
    times = list(snapshot.image_timestamps.values())
    if not np.isfinite(times).all() or min(times) <= 0 or max(times) - min(times) > .025:
        raise ValueError("invalid camera pair timestamps")
    if any(not 0 <= snapshot.observed_at_ns / 1e9 - t <= .075 for t in times):
        raise ValueError("stale/future camera observation")


def state_vector(snapshot, previous):
    velocity = schema.base_tcp_velocity(
        previous.pose[:3, 3] if previous else None, previous.pose[:3, :3] if previous else None,
        previous.observed_at_ns if previous else None,
        snapshot.pose[:3, 3], snapshot.pose[:3, :3], snapshot.observed_at_ns,
    )
    result = np.concatenate((snapshot.pose[:3, 3], schema.matrix_to_rot6d(snapshot.pose[:3, :3]),
                             [snapshot.gripper_width], snapshot.base_wrench, velocity, snapshot.joint_positions))
    if result.shape != (29,) or not np.isfinite(result).all():
        raise ValueError("invalid 29D demonstration state")
    return result.astype(np.float32)


def command_from_debug(debug):
    command = {"at_ns": int(debug["human_command_at_ns"]),
               "linear": list(debug["commanded_linear_velocity"]),
               "angular": list(debug["commanded_angular_velocity"]),
               "gripper": float(debug["human_gripper_target"])}
    if command["at_ns"] <= 0 or np.shape(command["linear"]) != (3,) or np.shape(command["angular"]) != (3,):
        raise ValueError("invalid command shape/time")
    if not np.isfinite([*command["linear"], *command["angular"], command["gripper"]]).all() or not 0 <= command["gripper"] <= 1:
        raise ValueError("invalid command values")
    return command


class IntervalAssembler:
    """Ordered command/observation stream. The final unclosed frame is never emitted."""
    def __init__(self, first, seed, phase, fps):
        validate_snapshot(first)
        if seed["at_ns"] > first.observed_at_ns:
            raise ValueError("seed command is newer than initial observation")
        self.pending, self.previous = first, None
        self.pose, self.at_ns = first.pose.copy(), first.observed_at_ns
        self.command, self.phase, self.fps = seed, float(phase), fps

    def _advance(self, at_ns):
        if at_ns < self.at_ns:
            raise ValueError("out-of-order command/observation")
        dt = (at_ns - self.at_ns) / 1e9
        if (at_ns - self.command["at_ns"]) / 1e9 > .1:
            raise ValueError("command trace gap exceeds watchdog")
        self.pose[:3, 3] += np.asarray(self.command["linear"]) * dt
        angular = np.asarray(self.command["angular"])
        speed = np.linalg.norm(angular)
        if speed > 1e-12:
            self.pose[:3, :3] = _axis_angle_matrix(angular / speed, speed * dt) @ self.pose[:3, :3]
        self.at_ns = at_ns

    def accept_command(self, command):
        if command["at_ns"] <= self.command["at_ns"]:
            raise ValueError("non-increasing command timestamp")
        self._advance(command["at_ns"])
        self.command = command

    def observe(self, snapshot, phase):
        validate_snapshot(snapshot)
        dt = (snapshot.observed_at_ns - self.pending.observed_at_ns) / 1e9
        if not .5 / self.fps <= dt <= 1.5 / self.fps:
            raise ValueError("recording interval discontinuity")
        if any(snapshot.image_timestamps[k] <= self.pending.image_timestamps[k] for k in snapshot.image_timestamps):
            raise ValueError("duplicate or reversed camera frame")
        self._advance(snapshot.observed_at_ns)
        action = np.concatenate((self.pose[:3, 3], schema.matrix_to_rot6d(self.pose[:3, :3]), [self.command["gripper"]]))
        row = (self.pending, state_vector(self.pending, self.previous), action.astype(np.float32), self.phase,
               snapshot.observed_at_ns)
        self.previous, self.pending = self.pending, snapshot
        self.pose, self.phase = snapshot.pose.copy(), float(phase)
        return row


class RecordingDatasetWriter(V30DatasetWriter):
    def _recover_temporary_files(self):
        # Never let the generic writer delete diagnostic files from old attempts.
        pass

    def _write_episode_rows(self, rows):
        # Data must be durable before the atomic episode-row commit. Keep this
        # stronger contract local to recording instead of changing other writers.
        if not rows.empty:
            path = self.data_path(int(rows.iloc[-1]["episode_index"]))
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            _sync_directory(path.parent)
        super()._write_episode_rows(rows)


class RecordingStore:
    def __init__(self, root, control, recording, task, *, resume=False):
        self.root = Path(root).expanduser().resolve()
        self.contract = dataset_contract(control, recording, task)
        self.lease = WriterLease(self.root)
        self.worker = None
        try:
            info_path = self.root / "meta/info.json"
            nonempty = self.root.exists() and any(self.root.iterdir())
            if not resume and nonempty:
                raise FileExistsError("nonempty dataset: use a new directory or explicit --resume")
            if resume:
                if not info_path.is_file():
                    raise ValueError("--resume requires an existing gamepad recording dataset")
                info = json.loads(info_path.read_text())
                if info.get("recording_contract") != self.contract:
                    raise ValueError("recording dataset contract mismatch")
            self.writer = RecordingDatasetWriter(
                self.root, recording.fps, task, CAMERAS, 224, 224, extra_features=FEATURES,
                info_metadata={"collection_kind": "human_demonstration", "gamepad_recording_version": VERSION,
                               "recording_contract": self.contract}, strict_metadata=True,
            )
            self.recording = recording
            self._quarantine_uncommitted()
        except BaseException:
            self.lease.close()
            raise

    def _quarantine_uncommitted(self):
        # Only paths reserved by THIS format for the next, uncommitted episode.
        index = self.writer.next_episode_index()
        paths = [self.writer.data_path(index), *(self.writer.video_path(index, key) for key in CAMERAS),
                 self.root / "commands" / f"episode-{index:08d}.jsonl"]
        present = [path for path in paths if path.exists()]
        if present:
            archive = self.root / ".recording_attempts" / str(uuid.uuid4())
            archive.mkdir(parents=True)
            for i, path in enumerate(present):
                os.replace(path, archive / f"uncommitted-{i}-{path.name}")
            atomic_json(archive / "recovery.json", {"uncommitted_episode": index, "original_paths": [str(p.relative_to(self.root)) for p in present]})

    def start(self, first, seed):
        if self.worker is not None:
            raise RuntimeError("previous recording has not finished")
        self.worker = EpisodeWorker(self, first, seed)
        return self.worker

    def close(self):
        if self.worker is not None:
            if not self.worker.finishing.is_set():
                self.worker.cancel_recording("collector closing without success confirmation")
            # An explicit y confirmation survives an immediate orderly exit.
            self.worker.thread.join(timeout=30 if self.worker.finishing.is_set() else 5)
            if self.worker.thread.is_alive():
                self.worker.cancel_recording("recording worker shutdown timeout")
                for video in self.worker.videos.values():
                    video.abort()
                self.worker.thread.join(timeout=5)
            if self.worker.thread.is_alive():
                raise RuntimeError("recording worker did not stop; writer lease retained")
        self.lease.close()


class EpisodeWorker:
    def __init__(self, store, first, seed):
        self.store, self.config = store, store.recording
        self.index = store.writer.next_episode_index()
        self.attempt = store.root / ".recording_attempts" / str(uuid.uuid4())
        self.events = queue.Queue(maxsize=self.config.queue_size)
        self.cancel, self.done, self.finishing = threading.Event(), threading.Event(), threading.Event()
        self.cancel_details = None
        self.error, self.saved_index = None, None
        self.frames = self.critical_frames = 0
        self.videos = {}
        self.assembler = IntervalAssembler(first, seed, False, self.config.fps)
        self.thread = threading.Thread(target=self._run, name="gamepad-recording", daemon=True)
        self.thread.start()

    def put(self, kind, value):
        self.check()
        try:
            self.events.put_nowait((kind, value))
            if kind == "finish":
                self.finishing.set()
        except queue.Full:
            self.cancel_recording("recording queue full; episode invalidated")
            raise RuntimeError("recording queue full; episode invalidated") from None

    def cancel_recording(self, reason, fault_snapshot=None):
        # Publish before signalling the worker; no disk I/O or queue wait on the
        # control thread, including when the event queue is full.
        if not self.cancel.is_set():
            self.cancel_details = {"reason": str(reason), "controller_fault": fault_snapshot}
            self.cancel.set()

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"recording worker failed: {self.error}") from self.error
        if self.cancel.is_set():
            raise RuntimeError("recording was cancelled")

    def _run(self):
        states, actions = [], []
        extras = {key: [] for key in FEATURES}
        stats = {key: ImageRunningStats() for key in CAMERAS}
        try:
            self.attempt.mkdir(parents=True)
            atomic_json(self.attempt / "attempt.json", {"episode_index": self.index, "status": "recording"})
            for key in CAMERAS:
                self.videos[key] = FfmpegVideoWriter(self.attempt / f"{key}.mp4", width=224, height=224, fps=self.config.fps)
            with (self.attempt / "commands.jsonl").open("w") as trace:
                trace.write(json.dumps({"kind": "seed", **self.assembler.command}) + "\n")
                while not self.cancel.is_set():
                    try:
                        kind, value = self.events.get(timeout=.05)
                    except queue.Empty:
                        continue
                    if kind == "finish":
                        break
                    if kind == "command":
                        self.assembler.accept_command(value)
                        trace.write(json.dumps({"kind": "command", **value}) + "\n")
                        continue
                    snapshot, phase = value
                    pending, state, action, label, end_ns = self.assembler.observe(snapshot, phase)
                    for key in CAMERAS:
                        image = pending.images[key.removeprefix("observation.images.")]
                        self.videos[key].write(image)
                        stats[key].update(image)
                    states.append(state)
                    actions.append(action)
                    values = [label, pending.observed_at_ns, end_ns,
                              int(pending.image_timestamps["base"] * 1e9), int(pending.image_timestamps["left_wrist"] * 1e9)]
                    for key, number in zip(FEATURES, values, strict=True):
                        extras[key].append(number)
                    trace.write(json.dumps({"kind": "interval", "frame_index": len(states) - 1,
                                            "observed_at_ns": pending.observed_at_ns, "end_ns": end_ns,
                                            "phase": label}) + "\n")
                    self.frames, self.critical_frames = len(states), self.critical_frames + int(label)
                    if self.frames >= self.config.max_duration_s * self.config.fps:
                        raise RuntimeError("maximum episode duration reached; episode invalidated")
                trace.flush()
                os.fsync(trace.fileno())
            if self.cancel.is_set():
                for video in self.videos.values():
                    video.abort()
                atomic_json(self.attempt / "attempt.json", {
                    "episode_index": self.index, "status": "discarded",
                    **(self.cancel_details or {"reason": "recording cancelled"}),
                })
                return
            for video in self.videos.values():
                video.close()
            for key in CAMERAS:
                output = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                     "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1",
                     str(self.attempt / f"{key}.mp4")],
                    check=True, text=True, capture_output=True, timeout=30,
                )
                if int(output.stdout.strip()) != self.frames:
                    raise ValueError("encoded video/numeric frame count mismatch")
            if self.frames < self.config.min_frames:
                raise ValueError(f"need at least {self.config.min_frames} complete frames, got {self.frames}")
            self.check()
            np.savez(self.attempt / "frames.npz", states=states, actions=actions, **extras)
            self._publish(states, actions, extras, stats)
        except BaseException as error:
            self.error = error
            for video in self.videos.values():
                try:
                    video.abort()
                except Exception:
                    pass
            if self.attempt.is_dir():
                try:
                    atomic_json(self.attempt / "error.json", {
                        "error": repr(error), "episode_index": self.index,
                        **(self.cancel_details or {}),
                    })
                except OSError:
                    pass
        finally:
            self.done.set()

    def _publish(self, states, actions, extras, stats):
        writer = self.store.writer
        schema.validate_episode_features(np.asarray(states), np.asarray(actions), require_action_motion=False)
        phase = extras["complementary_info.phase"]
        active = [i for i, value in enumerate(phase) if value]
        intervals = [] if not active else [{"start_frame": active[0], "end_frame": active[-1] + 1}]
        if active and len(active) != active[-1] - active[0] + 1:
            raise ValueError("multiple critical intervals are not supported")
        # Refuse overwrite even after a partially published previous attempt.
        files = [(self.attempt / f"{key}.mp4", writer.video_path(self.index, key)) for key in CAMERAS]
        trace_path = self.store.root / "commands" / f"episode-{self.index:08d}.jsonl"
        files.append((self.attempt / "commands.jsonl", trace_path))
        if writer.data_path(self.index).exists() or any(dst.exists() for _, dst in files):
            raise FileExistsError("uncommitted output exists; restart with --resume to quarantine it")
        for source, target in files:
            self.check()
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(source, target)
            _sync_directory(target.parent)
        self.check()
        self.saved_index = writer.save_episode(
            np.asarray(states), np.asarray(actions), {key: value.as_dict() for key, value in stats.items()},
            frame_features={key: np.asarray(value).reshape(-1, 1) for key, value in extras.items()},
            extra_episode_metadata={"episode_success": "success", "collection_source": "human_demonstration",
                                    "critical_intervals": json.dumps(intervals), "critical_human_frames": len(active),
                                    "command_trace": str(trace_path.relative_to(self.store.root))},
        )
        atomic_json(self.attempt / "attempt.json", {"episode_index": self.index, "status": "committed"})
