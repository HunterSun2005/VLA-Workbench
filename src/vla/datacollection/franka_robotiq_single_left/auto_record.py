"""Shared camera/input helpers and the single-arm GELLO ROS session entry point."""
from __future__ import annotations

from collections import deque
import os
import select
import sys
import termios
import threading
import time
import traceback
import tty
import numpy as np
from vla.terminal import console_print as print, current_reporter

CAMERA_CAPTURE_WIDTH = 640
CAMERA_CAPTURE_HEIGHT = 480
CAMERA_CAPTURE_FPS = 30
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


def report_exception(context: str) -> None:
    details = traceback.format_exc()
    reporter = current_reporter()
    if reporter is not None:
        reporter.event(context, level='error', details=details)
    else:
        print(f'{context}\n{details}', file=sys.stderr, flush=True)


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
        import pyrealsense2 as rs
        self._pipeline = rs.pipeline()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._error = None
        self._started = False
        self._thread = threading.Thread(target=self._worker, name=f"realsense-{role}", daemon=True)

    def start(self) -> None:
        import pyrealsense2 as rs
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


def prepare_image(
    rgb: np.ndarray, image_size: tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH)
) -> np.ndarray:
    """Fit the entire camera view into the requested (height, width)."""
    from vla.core.images import resize_rgb_with_pad

    return resize_rgb_with_pad(rgb, *image_size)


def main() -> None:
    from vla.cli.ros_session import main as shared_main
    shared_main(['--arms', 'single_left', '--source', 'gello', *sys.argv[1:]])


if __name__ == '__main__':
    main()
