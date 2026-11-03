"""Single-left gamepad demonstrations: no GELLO, policy server or online training."""
from __future__ import annotations

from vla.terminal import (console_print as print, console_status, console_debug, terminal_session,
                          TerminalConfirmation, diagnostic_path, current_reporter)

import argparse
from contextlib import nullcontext
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import select
import shutil
import signal
import sys
import tempfile
import time

from vla.datacollection.start_teleop.gamepad_teleop import (
    AsyncReporter, GamepadTeleop, JoyReceiver, JoySidecars, keyboard, preflight, _interrupt,
)
from vla.datacollection.start_teleop.gamepad_recording import (
    CriticalPhase, RecordingStore, command_from_debug, load_recording_config, validate_snapshot,
)
from vla.robots.franka_robotiq.robot import FrankaRobotiqRobot
from vla.datacollection.start_teleop.gamepad_preview import CameraPreview


DEFAULT_CONFIG = Path(__file__).parent / "configs/single_left_gamepad_record.yaml"


class RecordingSession:
    """Main-thread coordinator. Never encodes video or waits for the disk worker."""
    def __init__(self, control, recording, robot, receiver, store, *, report=print, clock=time.monotonic):
        self.recording, self.robot, self.store = recording, robot, store
        self.report, self.clock = report, clock
        self.controller = GamepadTeleop(control, robot, receiver, clock=clock, report=report, on_command=self.on_command)
        self.phase = CriticalPhase(recording.critical_button)
        self.last_command = None
        self.worker = None
        self.saving = False
        self.storage_fault = False
        self.last_snapshot = None
        self.next_sample = 0.
        self.started_at = 0.
        self.last_report = 0.
        self.preview = None

    def on_command(self, debug):
        self.last_command = command_from_debug(debug)
        if self.worker is not None and not self.saving:
            self.worker.put("command", self.last_command)

    def begin(self):
        if self.worker is not None or self.storage_fault:
            self.report("Cannot record: previous save/cleanup pending or storage fault; restart with --resume after a storage fault")
            return
        if self.controller.mode != "active":
            self.report("请先启用遥操作，再开始录制。")
            return
        try:
            packet = self.controller._packet()
            if not self.controller._neutral(packet):
                self.report("Release all inputs before starting a recording")
                return
            self.controller._read_state()
            snapshot = self.robot.get_rollout_snapshot()
            validate_snapshot(snapshot)
            if self.last_command is None:
                raise RuntimeError("wait for a successful neutral control command before recording")
            if any(abs(v) > 1e-12 for key in ("linear", "angular") for v in self.last_command[key]):
                self.report("Wait for a successful zero-velocity command before recording")
                return
            self.phase.reset()
            # A held button cannot arm; the first post-start rising edge is valid.
            self.phase.update(packet.buttons)
            self.worker = self.store.start(snapshot, self.last_command)
            self.last_snapshot = snapshot
            self.started_at = snapshot.observed_at_ns / 1e9
            self.next_sample = self.started_at + 1 / self.recording.fps
            self.report(f"RECORDING episode={self.worker.index}；可标记成功／失败并保存，或丢弃当前回合。")
        except Exception as error:
            self.controller.pause(f"recording start failed: {error}")

    def finish(self, success=False, reason="discard requested", *, save_failure=False):
        worker = self.worker
        if worker is None:
            self.controller.pause(reason)
            return
        if self.saving:
            return
        # End motion before initiating any disk flush; the pending final interval
        # is intentionally omitted, including the native braking trajectory.
        self.saving = True
        if self.controller.mode == "active":
            self.controller.pause("success requested" if success else reason)
        if self.controller.mode == "fault":
            success = False
            save_failure = False
            reason = self.controller.reason
        if success or save_failure:
            try:
                worker.put("finish", {"success": bool(success), "reason": str(reason)})
                self.report("SAVING: motion disabled while videos/data are finalized")
            except Exception as error:
                self.storage_fault = True
                worker.cancel_recording(f"Save rejected: {error}", self.controller.fault_snapshot)
                self.report(f"Save rejected: {error}")
        else:
            worker.cancel_recording(reason, self.controller.fault_snapshot)
            self.report(f"DISCARDED current episode: {reason}; not added to the dataset")
            self.report(f"Recording diagnostics: {worker.attempt}/attempt.json")

    def _poll_completion(self):
        worker = self.worker
        if worker is None or not worker.done.is_set():
            return
        if not self.saving:
            self.controller.pause("recording worker stopped unexpectedly", fault=True)
        if worker.error is not None:
            self.storage_fault = True
            self.report(f"RECORDING FAULT: {worker.error}; diagnostic files: {worker.attempt}; restart with --resume")
        elif worker.saved_index is not None:
            outcome = "success" if worker.success else "failure"
            self.report(f"Saved episode={worker.saved_index} frames={worker.frames} critical+Human={worker.critical_frames} {outcome}")
            if not worker.critical_frames:
                self.report("No critical frames: saved as an SFT demonstration with phase=0")
        self.worker = self.store.worker = None
        self.saving = False
        self.phase.reset()

    def tick(self):
        self._poll_completion()
        if self.worker is not None and not self.saving:
            try:
                self.worker.check()
                if self.controller.mode != "active":
                    raise RuntimeError("teleop interrupted; discontinuous episodes cannot resume")
                packet = self.controller._packet()
                message = self.phase.update(packet.buttons)
                if message:
                    self.report(message)
                now = self.clock()
                if now - self.started_at >= self.recording.max_duration_s:
                    raise RuntimeError("maximum episode duration reached")
                if now >= self.next_sample:
                    if now - self.last_snapshot.observed_at_ns / 1e9 > 1.5 / self.recording.fps:
                        raise RuntimeError("recording sample deadline missed")
                    snapshot = self.robot.get_rollout_snapshot()
                    elapsed = (snapshot.observed_at_ns - self.last_snapshot.observed_at_ns) / 1e9
                    if elapsed <= 0 or elapsed > 1.5 / self.recording.fps:
                        raise RuntimeError("camera/recording interval exceeded 1.5 frame periods")
                    fresh = all(snapshot.image_timestamps[k] > self.last_snapshot.image_timestamps[k] for k in snapshot.image_timestamps)
                    if fresh and elapsed >= .5 / self.recording.fps:
                        validate_snapshot(snapshot)
                        self.worker.put("observation", (snapshot, self.phase.active))
                        self.last_snapshot = snapshot
                        self.next_sample += 1 / self.recording.fps
                if now - self.last_report >= 1:
                    self.last_report = now
                    console_status(f"RECORDING episode={self.worker.index} frames={self.worker.frames} elapsed={now-self.started_at:.1f}s "
                                f"speed={'HIGH' if self.controller.high_speed else 'LOW'} critical={self.phase.state}")
            except Exception as error:
                self.finish(False, str(error))
        self.controller.tick()
        if self.worker is not None and not self.saving and self.controller.mode != "active":
            self.finish(False, self.controller.reason)
        if self.preview is not None:
            self.preview.update_status({
                "mode": "SAVING/CLEANUP" if self.saving else "RECORDING" if self.worker else self.controller.mode.upper(),
                "episode": self.worker.index if self.worker else "-",
                "frames": self.worker.frames if self.worker else 0,
                "critical": self.phase.state,
                "speed": "HIGH" if self.controller.high_speed else "LOW",
            })

    def key(self, key):
        if key == "s":
            if self.worker is not None and self.saving or self.storage_fault:
                self.report("Cannot enable while saving/cleaning up or after a storage fault")
            else:
                self.controller.enable()
        elif key == "n":
            self.begin()
        elif key == "y":
            if self.worker is not None:
                self.finish(True)
        elif key == "f":
            if self.worker is not None:
                self.finish(False, "operator marked failure", save_failure=True)
        elif key in ("x", "d"):
            self.finish(False, "operator discarded episode" if key == "x" else "operator paused")
        elif key == "r":
            if self.worker is None:
                self.controller.recover()


def recording_preflight(control, recording):
    preflight(control)
    for module in ("pyrealsense2", "pyarrow", "pandas", "cv2"):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(f"missing recording dependency: {module}")
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"missing recording dependency: {executable}")
    # Enumeration only; never open a RealSense pipeline in --check-only mode.
    import pyrealsense2 as rs
    available = {device.get_info(rs.camera_info.serial_number) for device in rs.context().query_devices()}
    missing = set(control.robot.camera_serials.values()) - available
    if missing:
        raise RuntimeError(f"missing recording cameras: {sorted(missing)}")


def run(control, recording, root, task, resume, *, preview=True):
    if not sys.stdin.isatty():
        raise PermissionError("--execute requires an interactive terminal")
    store = RecordingStore(root, control, recording, task, resume=resume)
    robot, session = None, None
    try:
        print(f"Left Franka {control.robot.robot_ip}; cameras={control.robot.camera_serials}; dataset={store.root}")
        print("Clear the workspace; keep the emergency stop accessible. No home/open target is sent.")
        if control.robot.realtime_cpu is not None:
            os.sched_setaffinity(0, os.sched_getaffinity(0) - {control.robot.realtime_cpu})
        with tempfile.TemporaryDirectory(prefix="vla-gamepad-record-") as directory:
            runtime = Path(directory)
            receiver = JoyReceiver(runtime / "joy.sock")
            try:
                with JoySidecars(control, receiver.path, runtime) as sidecars, (nullcontext(print) if current_reporter() else AsyncReporter()) as reporter:
                    robot = FrankaRobotiqRobot(replace(control.robot, calibration_dir=runtime / "calibration"))
                    robot.connect()
                    session = RecordingSession(control, recording, robot, receiver, store, report=reporter)
                    viewer = CameraPreview(runtime, robot.get_camera_preview_snapshot, enabled=preview, report=reporter)
                    print("STANDBY: s=enable, n=record, y=success/save, f=failure/save, x=discard (confirm), d=pause+discard immediately, r=recover, c/Esc=exit, Ctrl+C=immediate stop; RB=critical")
                    try:
                        viewer.start()
                        session.preview = viewer
                        confirmation = TerminalConfirmation()
                        with keyboard() as fd:
                            while True:
                                started = time.monotonic()
                                sidecars.check()
                                session.tick()
                                confirmation.expire()
                                if select.select([fd], [], [], 0)[0]:
                                    keys = os.read(fd, 32).decode(errors="ignore").lower()
                                    if not keys or "\x04" in keys or "\x03" in keys:
                                        break
                                    if "d" in keys:
                                        # A pause in the same input batch must never be swallowed
                                        # while dropping keys typed before a confirmation prompt.
                                        confirmation.route("d")
                                        session.key("d")
                                        continue
                                    should_exit = False
                                    for key in keys:
                                        event = confirmation.key_event(key.encode()) or key
                                        worker = session.worker
                                        destructive = worker is not None and not session.saving and event in ("x", "c", "\x1b")
                                        event = confirmation.route(event, destructive=destructive,
                                            valid=lambda target=worker: session.worker is target and not session.saving and session.controller.mode != "fault",
                                            label="退出并丢弃当前未保存回合" if event != "x" else "丢弃当前未保存回合")
                                        if event is None:
                                            if confirmation.pending:
                                                break  # Drop keys buffered before the prompt was displayed.
                                            continue
                                        if event in ("c", "\x1b"):
                                            should_exit = True
                                            break
                                        session.key(event)
                                    if should_exit:
                                        break
                                time.sleep(max(0, 1 / control.control.control_rate - (time.monotonic() - started)))
                    finally:
                        session.finish(False, "shutdown")
                        viewer.close()
                        robot.disconnect()
            finally:
                receiver.close()
    finally:
        try:
            if robot is not None and robot.is_connected:
                robot.disconnect()
        finally:
            store.close()


@terminal_session("gamepad-record")
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-preview", action="store_true", help="Disable the camera preview window (headless recording)")
    args = parser.parse_args()
    if args.execute and (args.dataset_root is None or not args.task or not args.task.strip()):
        parser.error("--execute requires --dataset-root and nonempty --task")
    try:
        control, recording = load_recording_config(args.config.expanduser().resolve())
        recording_preflight(control, recording)
        if not args.execute:
            print("Recording preflight passed; no robot connection/motion or dataset initialization requested")
            return
        previous = signal.signal(signal.SIGTERM, _interrupt)
        try:
            run(control, recording, args.dataset_root, args.task, args.resume, preview=not args.no_preview)
        finally:
            signal.signal(signal.SIGTERM, previous)
    except KeyboardInterrupt:
        print("Gamepad recording stopped; unconfirmed episode was not saved", flush=True)
    except Exception as error:
        print(f"Gamepad recording failed: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
