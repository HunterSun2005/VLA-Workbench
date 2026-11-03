"""ROS-independent keyboard and streaming-video helpers for rollout."""

from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import queue
import select
import shutil
import subprocess
import sys
import termios
import threading
import tty

import cv2
import numpy as np


VIDEO_CRF = 23
VIDEO_PRESET = "veryfast"
VIDEO_QUEUE_SIZE = 8
IMAGE_STATS_STRIDE = 5


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
    def __init__(self, path: Path, *, width: int, height: int, fps: int) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required for streaming video capture")
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("video width, height, and fps must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-preset",
                VIDEO_PRESET,
                "-pix_fmt",
                "yuv420p",
                "-crf",
                str(VIDEO_CRF),
                "-movflags",
                "+faststart",
                str(path),
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
            raise ValueError(
                f"video frame must be uint8 RGB with shape {expected}, "
                f"got {rgb.shape}/{rgb.dtype}"
            )
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
    def __init__(self, path: Path, *, width: int, height: int, fps: int) -> None:
        self.writer = FfmpegVideoWriter(path, width=width, height=height, fps=fps)
        self.image_stats = ImageRunningStats()
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=VIDEO_QUEUE_SIZE)
        self._lock = threading.Lock()
        self._closed = False
        self._abort = threading.Event()
        self._error: BaseException | None = None
        self._frame_count = 0
        self._blocked_count = 0
        self._max_queue_size = 0
        self._thread = threading.Thread(
            target=self._worker,
            name=f"ffmpeg-{path.parent.name}",
            daemon=True,
        )
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


def keyboard_event(char: bytes, *, sft: bool = False) -> str | None:
    char = char.lower()
    if char in {b"q", b"\x03", b"\x04", b""}:
        return "quit"
    if sft:
        return {b"s": "enable", b"n": "right", b"y": "success", b"f": "failure",
                b"x": "left", b"d": "pause", b"r": "recover", b"c": "quit",
                b"h": "home", b" ": "intervention"}.get(char)
    return {b" ": "intervention", b"s": "success", b"f": "failure",
            b"r": "phase", b"c": "recover"}.get(char)


class KeyboardEvents:
    def __init__(self, *, sft: bool = False, confirmation=None) -> None:
        self.confirmation = confirmation
        self._sft = sft
        self._events: deque[str] = deque()
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
                if not self._sft or not suffix:
                    self._push("right" if suffix == b"[C" else "left" if suffix == b"[D" else "quit")
            else:
                event = (self.confirmation.key_event(char) if self.confirmation is not None else None) or keyboard_event(char, sft=self._sft)
                if event is not None:
                    self._push(event)

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
