"""Latest-frame preview: camera-buffer reader thread -> mmap -> GUI subprocess.

No hardware imports. GUI crashes cannot take down the control process, and
the control loop only replaces a small status dictionary (never waits for IPC).
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time

import numpy as np


WINDOW = "VLA - Camera Preview"
SHAPE = (2, 224, 224, 3)
HEADER_SIZE = 4096
SLOT_SIZE = HEADER_SIZE + int(np.prod(SHAPE))
ROLES = ("base", "left_wrist")


def write_slot(slot, images, status):
    """One writer, one reader, latest frame only; odd sequence means in progress."""
    for role in ROLES:
        if images[role].shape != SHAPE[1:] or images[role].dtype != np.uint8:
            raise ValueError("preview requires 224x224 RGB uint8 images")
    payload = json.dumps(status, ensure_ascii=True).encode("utf-8")
    if len(payload) > HEADER_SIZE - 12:
        raise ValueError("preview status too long")
    sequence = struct.unpack_from("<Q", slot, 0)[0]
    sequence += 1 if sequence % 2 == 0 else 2
    struct.pack_into("<Q", slot, 0, sequence)
    struct.pack_into("<I", slot, 8, len(payload))
    slot[12:12 + len(payload)] = payload
    pixels = np.ndarray(SHAPE, dtype=np.uint8, buffer=slot, offset=HEADER_SIZE)
    for index, role in enumerate(ROLES):
        pixels[index] = images[role]
    del pixels
    struct.pack_into("<Q", slot, 0, sequence + 1)


def read_slot(slot):
    sequence = struct.unpack_from("<Q", slot, 0)[0]
    if sequence == 0 or sequence % 2:
        return None
    length = struct.unpack_from("<I", slot, 8)[0]
    if not 0 < length <= HEADER_SIZE - 12:
        return None
    payload = bytes(slot[12:12 + length])
    images = np.ndarray(SHAPE, dtype=np.uint8, buffer=slot, offset=HEADER_SIZE).copy()
    if struct.unpack_from("<Q", slot, 0)[0] != sequence:
        return None  # Writer overtook the reader: skip display, never block it.
    try:
        return images, json.loads(payload)
    except (ValueError, UnicodeError):
        return None


def render_frame(images, status, *, now=None):
    import cv2

    now = time.monotonic() if now is None else now
    canvas = np.zeros((560, 896, 3), dtype=np.uint8)
    for index, role in enumerate(ROLES):
        image = cv2.cvtColor(images[index], cv2.COLOR_RGB2BGR)
        canvas[60:508, index * 448:(index + 1) * 448] = cv2.resize(image, (448, 448))
        cv2.putText(canvas, "BASE" if role == "base" else "LEFT WRIST", (index * 448 + 12, 51),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (220, 220, 220), 1, cv2.LINE_AA)
    critical = status.get("critical") == "critical"
    phase_label = "RLT FULL" if status.get("full_episode") else f"critical {'ON' if critical else 'OFF'}"
    label = (f"{status.get('mode', 'WAITING')} | EP {status.get('episode', '-')} | "
             f"frames {status.get('frames', 0)} | {phase_label} | "
             f"{status.get('speed', 'LOW')}")
    cv2.putText(canvas, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .57,
                (0, 180, 255) if critical else (120, 230, 120), 1, cv2.LINE_AA)
    age = now - status.get("captured_at", 0.)
    if status.get("camera_error") or age > .5 or age < 0:
        message = "CAMERA UNAVAILABLE / STALE - do not treat this as a live image"
        cv2.rectangle(canvas, (0, 255), (895, 295), (0, 0, 100), -1)
        cv2.putText(canvas, message, (10, 280), cv2.FONT_HERSHEY_SIMPLEX, .57, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "q/Esc/window close: preview only. Recording keys: use the TERMINAL.", (12, 538),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def display_loop(slot, tk=None, image_tk=None):
    """GUI calls stay on the child process main thread."""
    if tk is None:
        import tkinter as tk
    if image_tk is None:
        from PIL import ImageTk as image_tk
    from PIL import Image

    root = tk.Tk()
    root.title(WINDOW)
    root.resizable(False, False)
    label = tk.Label(root)
    label.pack()
    last = (np.zeros(SHAPE, dtype=np.uint8), {})
    closed, failure = False, []

    def close(_event=None):
        nonlocal closed
        if not closed:
            closed = True
            root.destroy()

    def update():
        nonlocal last
        if closed:
            return
        try:
            frame = read_slot(slot)
            if frame is not None:
                last = frame
            rendered = render_frame(*last)
            image = image_tk.PhotoImage(Image.fromarray(rendered[:, :, ::-1].copy()), master=root)
            label.configure(image=image)
            label.image = image  # Tk needs a live Python reference.
            root.after(33, update)
        except Exception as error:
            failure.append(error)
            close()

    root.protocol("WM_DELETE_WINDOW", close)
    root.bind("<Escape>", close)
    root.bind("<Key-q>", close)
    try:
        root.after(0, update)
        root.mainloop()
    finally:
        close()
    if failure:
        raise RuntimeError(f"preview rendering failed: {failure[0]}") from failure[0]


class CameraPreview:
    def __init__(self, runtime, snapshot_source, *, enabled=True, report=print):
        self.runtime, self.snapshot_source = Path(runtime), snapshot_source
        self.enabled, self.report = enabled, report
        self.status = {}
        self._stop = threading.Event()
        self._thread = self._process = self._slot = self._file = self._log = None

    def update_status(self, status):
        # Assignment only; no GUI, image copy, camera access, serialization or IO.
        self.status = dict(status)

    def start(self):
        if not self.enabled:
            return self
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            self.report("Camera preview disabled: no desktop display. Recording remains available (--no-preview).")
            return self
        try:
            path = self.runtime / "camera-preview.mmap"
            self._file = path.open("w+b")
            self._file.truncate(SLOT_SIZE)
            self._slot = mmap.mmap(self._file.fileno(), SLOT_SIZE)
            self._log = (self.runtime / "camera-preview.log").open("w")
            self._process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--slot", str(path)],
                stdin=subprocess.DEVNULL, stdout=self._log, stderr=self._log,
                start_new_session=True, cwd=self.runtime,
            )
            self._thread = threading.Thread(target=self._feed, name="gamepad-camera-preview", daemon=True)
            self._thread.start()
            self.report("Camera preview enabled; closing its window does not stop recording. Use the terminal for recording keys.")
        except Exception as error:
            self.report(f"Camera preview unavailable: {error}; recording remains available (--no-preview).")
            self.close()
        return self

    def __enter__(self):
        return self.start()

    def _feed(self):
        images = {role: np.zeros(SHAPE[1:], dtype=np.uint8) for role in ROLES}
        captured_at = 0.
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                if self._process.poll() is not None:
                    if self._process.returncode == 0:
                        self.report("Camera preview closed; recording continues in the terminal.")
                    else:
                        detail = (self.runtime / "camera-preview.log").read_text(errors="replace")[-1500:]
                        self.report(f"Camera preview exited ({self._process.returncode}); recording continues. "
                                    f"Check desktop/Tk/Pillow support or use --no-preview. {detail}")
                    return
                status = dict(self.status)
                try:
                    snapshot = self.snapshot_source()
                    images = snapshot.images
                    captured_at = min(snapshot.timestamps.values())
                except Exception as error:
                    status["camera_error"] = str(error)[:200]
                status["captured_at"] = captured_at
                write_slot(self._slot, images, status)
                self._stop.wait(max(0., 1 / 30 - (time.monotonic() - started)))
        except Exception as error:
            self.report(f"Camera preview disabled: {error}; recording continues.")

    def close(self):
        self._stop.set()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout=1)
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1)
        # Never unmap memory beneath a feeder still inside a source callback.
        if self._thread is None or not self._thread.is_alive():
            if self._slot is not None:
                self._slot.close()
            if self._file is not None:
                self._file.close()
        if self._log is not None:
            self._log.close()

    def __exit__(self, *_):
        self.close()


def main():
    parser = argparse.ArgumentParser(description="Display existing gamepad recording frames; never opens cameras")
    parser.add_argument("--slot", type=Path, required=True)
    args = parser.parse_args()
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with args.slot.open("rb") as stream, mmap.mmap(stream.fileno(), SLOT_SIZE, access=mmap.ACCESS_READ) as slot:
        display_loop(slot)


if __name__ == "__main__":
    main()
