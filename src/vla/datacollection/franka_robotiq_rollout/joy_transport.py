"""Versioned local transport for ROS Joy samples."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import socket
import time


JOY_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class JoyPacket:
    sequence: int
    monotonic_ns: int
    axes: tuple[float, ...]
    buttons: tuple[int, ...]


def encode_joy_packet(packet: JoyPacket) -> bytes:
    return json.dumps(
        {
            "version": JOY_PROTOCOL_VERSION,
            "sequence": packet.sequence,
            "monotonic_ns": packet.monotonic_ns,
            "axes": packet.axes,
            "buttons": packet.buttons,
        },
        separators=(",", ":"),
    ).encode("ascii")


def decode_joy_packet(payload: bytes) -> JoyPacket:
    try:
        value = json.loads(payload.decode("ascii"))
        if value.get("version") != JOY_PROTOCOL_VERSION:
            raise ValueError("unsupported Joy protocol version")
        packet = JoyPacket(
            sequence=int(value["sequence"]),
            monotonic_ns=int(value["monotonic_ns"]),
            axes=tuple(float(item) for item in value["axes"]),
            buttons=tuple(int(item) for item in value["buttons"]),
        )
    except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid Joy bridge packet") from error
    if packet.sequence < 0 or packet.monotonic_ns <= 0:
        raise ValueError("Joy sequence and timestamp must be non-negative")
    if not packet.axes or not packet.buttons:
        raise ValueError("Joy packet must contain axes and buttons")
    if not all(math.isfinite(item) for item in packet.axes):
        raise ValueError("Joy axes must be finite")
    if any(item not in {0, 1} for item in packet.buttons):
        raise ValueError("Joy buttons must be binary")
    return packet


class JoyReceiver:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            if not self.path.is_socket():
                raise FileExistsError(f"refusing to replace non-socket Joy path: {self.path}")
            self.path.unlink()
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._socket.bind(str(self.path))
        os.chmod(self.path, 0o600)
        self._socket.setblocking(False)
        self._latest: JoyPacket | None = None
        self._closed = False

    def poll_latest(self) -> JoyPacket | None:
        if self._closed:
            return self._latest
        while True:
            try:
                payload = self._socket.recv(65536)
            except BlockingIOError:
                return self._latest
            packet = decode_joy_packet(payload)
            # The bridge sequence restarts at zero whenever its ROS process is
            # restarted. Monotonic time is host-wide, so prefer it when
            # selecting the newest datagram and use sequence only as a tie
            # breaker. The loop also drains a full queue before freshness is
            # evaluated.
            if self._latest is None or (
                packet.monotonic_ns,
                packet.sequence,
            ) > (
                self._latest.monotonic_ns,
                self._latest.sequence,
            ):
                self._latest = packet

    def require_fresh(self, timeout_s: float, *, now_ns: int | None = None) -> JoyPacket:
        packet = self.poll_latest()
        if packet is None:
            raise RuntimeError("Joy bridge has not delivered an input sample")
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        age_s = (now_ns - packet.monotonic_ns) / 1_000_000_000.0
        if age_s < 0.0 or age_s > timeout_s:
            raise TimeoutError(f"Joy input is stale: age={age_s:.3f}s")
        return packet

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._socket.close()
        if self.path.is_socket():
            self.path.unlink()
