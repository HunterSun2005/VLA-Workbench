from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product
import threading
import time
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CameraSnapshot:
    images: dict[str, np.ndarray]
    timestamps: dict[str, float]
    pair_skew_s: float
    oldest_age_s: float


def prepare_image(rgb: np.ndarray, image_size: int = 224) -> np.ndarray:
    """Match the SFT recorder's square center crop followed by INTER_AREA resize."""
    import cv2

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected HWC uint8 RGB image, got {rgb.shape}/{rgb.dtype}")
    height, width = rgb.shape[:2]
    size = min(height, width)
    y0, x0 = (height - size) // 2, (width - size) // 2
    return np.ascontiguousarray(
        cv2.resize(
            rgb[y0 : y0 + size, x0 : x0 + size],
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )
    )


class RealSenseCapture:
    def __init__(
        self,
        serials: dict[str, str],
        *,
        width: int,
        height: int,
        fps: int,
        image_size: int,
        timeout_s: float,
        max_pair_skew_s: float,
        max_frame_age_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.serials = dict(serials)
        self.width, self.height, self.fps = width, height, fps
        self.image_size, self.timeout_s = image_size, timeout_s
        self.max_pair_skew_s = float(max_pair_skew_s)
        self.max_frame_age_s = float(max_frame_age_s)
        self._clock = clock
        self._pipelines: dict[str, object] = {}
        self._threads: list[threading.Thread] = []
        self._frames: dict[str, deque[tuple[float, np.ndarray]]] = {
            role: deque(maxlen=4) for role in self.serials
        }
        self._last_snapshot: CameraSnapshot | None = None
        self._errors: dict[str, BaseException] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def connect(self) -> None:
        import pyrealsense2 as rs

        self._stop.clear()
        with self._lock:
            for frames in self._frames.values():
                frames.clear()
            self._errors.clear()
            self._last_snapshot = None
        try:
            for role, serial in self.serials.items():
                pipeline, config = rs.pipeline(), rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
                pipeline.start(config)
                self._pipelines[role] = pipeline
                thread = threading.Thread(target=self._worker, args=(role, pipeline), daemon=True)
                thread.start()
                self._threads.append(thread)
            deadline = self._clock() + max(3.0, self.timeout_s * 5)
            while self._clock() < deadline:
                self.check_health(require_all=False)
                with self._lock:
                    if all(self._frames[role] for role in self.serials):
                        return
                time.sleep(0.02)
            raise TimeoutError("timed out waiting for RealSense camera frames")
        except BaseException:
            self.close()
            raise

    def _worker(self, role: str, pipeline) -> None:
        try:
            while not self._stop.is_set():
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color = frames.get_color_frame()
                if color:
                    captured_at = self._clock()
                    image = prepare_image(np.asanyarray(color.get_data()).copy(), self.image_size)
                    with self._lock:
                        self._frames[role].append((captured_at, image))
        except BaseException as error:
            if not self._stop.is_set():
                with self._lock:
                    self._errors[role] = error

    def check_health(self, *, require_all: bool = True) -> None:
        with self._lock:
            errors = dict(self._errors)
            latest = {
                role: frames[-1][0]
                for role, frames in self._frames.items()
                if frames
            }
        if errors:
            role, error = next(iter(errors.items()))
            raise RuntimeError(f"RealSense {role} failed: {error}") from error
        if require_all:
            now = self._clock()
            for role in self.serials:
                if role not in latest or now - latest[role] > self.timeout_s:
                    raise TimeoutError(f"RealSense {role} frame is stale")

    def snapshot(self) -> CameraSnapshot:
        self.check_health()
        now = self._clock()
        roles = tuple(self.serials)
        with self._lock:
            buffered = {role: tuple(self._frames[role]) for role in roles}
        candidates = []
        for frames in product(*(buffered[role] for role in roles)):
            timestamps = tuple(frame[0] for frame in frames)
            pair_skew_s = max(timestamps) - min(timestamps)
            oldest_age_s = now - min(timestamps)
            if pair_skew_s <= self.max_pair_skew_s and oldest_age_s <= self.max_frame_age_s:
                candidates.append((min(timestamps), frames, pair_skew_s, oldest_age_s))
        if not candidates:
            latest = {role: buffered[role][-1][0] for role in roles}
            pair_skew_s = max(latest.values()) - min(latest.values())
            oldest_age_s = now - min(latest.values())
            raise TimeoutError(
                "no synchronized RealSense frame pair: "
                f"pair_skew={pair_skew_s * 1000.0:.1f}ms "
                f"oldest_age={oldest_age_s * 1000.0:.1f}ms"
            )
        _, frames, pair_skew_s, oldest_age_s = max(candidates, key=lambda candidate: candidate[0])
        snapshot = CameraSnapshot(
            images={role: frame[1].copy() for role, frame in zip(roles, frames)},
            timestamps={role: frame[0] for role, frame in zip(roles, frames)},
            pair_skew_s=pair_skew_s,
            oldest_age_s=oldest_age_s,
        )
        with self._lock:
            self._last_snapshot = snapshot
        return snapshot

    def images(self) -> dict[str, np.ndarray]:
        return self.snapshot().images

    def debug_snapshot(self) -> dict[str, float | None]:
        with self._lock:
            snapshot = self._last_snapshot
        return {
            "camera_pair_skew_ms": None if snapshot is None else snapshot.pair_skew_s * 1000.0,
            "camera_oldest_age_ms": None if snapshot is None else snapshot.oldest_age_s * 1000.0,
        }

    def close(self) -> None:
        self._stop.set()
        for pipeline in reversed(tuple(self._pipelines.values())):
            try:
                pipeline.stop()
            except RuntimeError:
                pass
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()
        self._pipelines.clear()
