"""Small versioned Unix stream protocol for the Franky worker.

The protocol deliberately contains only JSON and fixed-size numeric arrays.  A
worker connection handles one request and one response; this keeps reconnects
and ownership boundaries explicit while the client sends targets from a
bounded background queue.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import socket
import struct
import threading
import time
from typing import Any, Mapping


IPC_VERSION = 1
MAX_PACKET = 2 * 1024 * 1024


def encode_packet(value: Mapping[str, Any]) -> bytes:
    if not isinstance(value, Mapping):
        raise TypeError("worker packet must be a mapping")
    payload = json.dumps(
        {"version": IPC_VERSION, **dict(value)},
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > MAX_PACKET:
        raise ValueError("worker packet is too large")
    return struct.pack("!I", len(payload)) + payload


def _read_exact(sock: socket.socket, size: int) -> bytes:
    value = bytearray()
    while len(value) < size:
        chunk = sock.recv(size - len(value))
        if not chunk:
            raise ConnectionError("worker connection closed")
        value.extend(chunk)
    return bytes(value)


def decode_packet(sock: socket.socket) -> dict[str, Any]:
    length = struct.unpack("!I", _read_exact(sock, 4))[0]
    if not 0 < length <= MAX_PACKET:
        raise ValueError("worker packet is too large")
    value = json.loads(_read_exact(sock, length).decode("utf-8"))
    if not isinstance(value, dict) or value.get("version") != IPC_VERSION:
        raise ValueError("unsupported Franky worker IPC version")
    return value


def _validate_pose(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        raise ValueError("TCP pose must be xyz + quaternion_xyzw")
    pose = [float(item) for item in value]
    if not all(__import__("math").isfinite(item) for item in pose):
        raise ValueError("TCP pose must be finite")
    norm = sum(item * item for item in pose[3:]) ** 0.5
    if norm < 1e-9:
        raise ValueError("TCP quaternion must be non-zero")
    return pose


@dataclass(frozen=True)
class WorkerState:
    arm: str
    observed_at_ns: int
    tcp_pose: tuple[float, ...]
    joints: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    joint_torque: tuple[float, ...] = ()
    external_joint_torque: tuple[float, ...] = ()
    force: tuple[float, ...] = ()
    torque: tuple[float, ...] = ()
    target_joints: tuple[float, ...] = ()
    flange_to_tcp_pose: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    desired_tcp_pose: tuple[float, ...] = ()
    generation: int = -1
    sequence: int = -1
    tracker_running: bool = False
    state_valid: bool = True
    fault: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkerState":
        pose = tuple(_validate_pose(value.get("tcp_pose")))
        joints = tuple(float(x) for x in value.get("joints", ()))
        velocity = tuple(float(x) for x in value.get("joint_velocity", ()))
        if len(joints) != 7 or len(velocity) != 7:
            raise ValueError("worker state must contain seven joints and velocities")
        joint_torque = tuple(float(x) for x in value.get("joint_torque", ()))
        external_torque = tuple(float(x) for x in value.get("external_joint_torque", ()))
        if ((joint_torque and len(joint_torque) != 7)
                or (external_torque and len(external_torque) != 7)):
            raise ValueError("worker state torques must contain seven joints when present")
        return cls(
            arm=str(value["arm"]),
            observed_at_ns=int(value["observed_at_ns"]),
            tcp_pose=pose,
            joints=joints,
            joint_velocity=velocity,
            joint_torque=joint_torque,
            external_joint_torque=external_torque,
            force=tuple(float(x) for x in value.get("force", ())),
            torque=tuple(float(x) for x in value.get("torque", ())),
            target_joints=tuple(float(x) for x in value.get("target_joints", ())),
            flange_to_tcp_pose=tuple(_validate_pose(value.get("flange_to_tcp_pose", (0, 0, 0, 0, 0, 0, 1)))),
            desired_tcp_pose=tuple(_validate_pose(value["desired_tcp_pose"])) if value.get("desired_tcp_pose") else (),
            generation=int(value.get("generation", -1)),
            sequence=int(value.get("sequence", -1)),
            tracker_running=bool(value.get("tracker_running", False)),
            state_valid=bool(value.get("state_valid", not value.get("fault"))),
            fault=str(value["fault"]) if value.get("fault") else None,
        )


class FrankyWorkerClient:
    """Non-blocking target client with a bounded latest-target queue."""

    def __init__(self, path: str | os.PathLike[str], *, timeout_s: float = .5,
                 queue_size: int = 8, session_id: str | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        self.timeout_s = float(timeout_s)
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=queue_size)
        self._states: dict[str, WorkerState] = {}
        self._compliance: dict[str, float] = {}
        self._binding_fingerprint: dict[str, str] = {}
        self._lock = threading.RLock()
        self._request_lock = threading.Lock()
        self._fault: str | None = None
        self._session_id = str(session_id) if session_id is not None else None
        self._generation = -1
        self._request_id = 0
        self._last_health = 0.0
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, name="franky-ipc", daemon=True)
        self._thread.start()
        if session_id is not None:
            self._request_id += 1
            self._enqueue({"kind": "health", "session_id": self._session_id,
                           "generation": self._generation, "request_id": self._request_id})

    @property
    def fault(self) -> str | None:
        with self._lock:
            return self._fault

    def states(self) -> dict[str, WorkerState]:
        with self._lock:
            return dict(self._states)

    def compliance_values(self) -> dict[str, float]:
        with self._lock:
            return dict(self._compliance)

    def binding_fingerprint(self) -> dict[str, str]:
        with self._lock:
            return dict(self._binding_fingerprint)

    def _enqueue(self, message: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            # Targets are replaceable; lifecycle requests are not.
            if message["kind"] == "target":
                try:
                    with self._queue.mutex:
                        index = next((i for i, item in enumerate(self._queue.queue)
                                      if item.get("kind") == "target"), None)
                        if index is None:
                            return
                        del self._queue.queue[index]
                        self._queue.not_full.notify()
                    self._queue.put_nowait(message)
                except (queue.Empty, ValueError):
                    return
            else:
                with self._queue.mutex:
                    index = next((i for i, item in enumerate(self._queue.queue)
                                  if item.get("kind") == "target"), None)
                    if index is None:
                        raise RuntimeError("Franky worker command queue is full")
                    del self._queue.queue[index]
                    self._queue.not_full.notify()
                self._queue.put_nowait(message)

    def target(self, session_id: str, generation: int, sequence: int,
               arms: Mapping[str, Any], timestamp_ns: int | None = None) -> None:
        targets = {}
        for arm, value in arms.items():
            pose = value["pose"] if isinstance(value, Mapping) else value
            targets[str(arm)] = {"pose": _validate_pose(pose)}
        self._generation = int(generation)
        self._request_id += 1
        self._enqueue({"kind": "target", "request_id": self._request_id, "session_id": str(session_id),
                       "generation": int(generation), "sequence": int(sequence),
                       "timestamp_ns": int(timestamp_ns or time.monotonic_ns()),
                       "arms": targets})

    def reset(self, session_id: str, generation: int) -> None:
        self._generation = int(generation)
        self._request_id += 1
        self._enqueue({"kind": "reset", "session_id": str(session_id),
                       "generation": int(generation), "request_id": self._request_id})

    def stop(self, session_id: str, generation: int) -> None:
        self._generation = int(generation)
        self._request_id += 1
        self._enqueue({"kind": "stop", "session_id": str(session_id),
                       "generation": int(generation), "request_id": self._request_id})

    def recover(self, session_id: str, generation: int) -> None:
        self._generation = int(generation)
        self._request_id += 1
        self._enqueue({"kind": "recover", "session_id": str(session_id),
                       "generation": int(generation), "request_id": self._request_id})

    def compliance(self, session_id: str, generation: int,
                   request_id: str, values: Mapping[str, Any]) -> None:
        self._enqueue({"kind": "compliance", "session_id": str(session_id),
                       "generation": int(generation), "request_id": str(request_id),
                       "values": dict(values)})

    def _request(self, message: Mapping[str, Any]) -> dict[str, Any]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout_s)
            sock.connect(str(self.path))
            sock.sendall(encode_packet(message))
            return decode_packet(sock)

    def request_now(self, message: Mapping[str, Any]) -> dict[str, Any]:
        """Perform a lifecycle request synchronously and publish its state."""
        if message.get("kind") in {"stop", "recover", "reset"}:
            with self._queue.mutex:
                retained = [item for item in self._queue.queue if item.get("kind") != "target"]
                self._queue.queue.clear()
                self._queue.queue.extend(retained)
        with self._request_lock:
            result = self._request(message)
        if result.get("error"):
            raise RuntimeError(str(result["error"]))
        with self._lock:
            for value in result.get("states", ()):
                state = WorkerState.from_dict(value)
                self._states[state.arm] = state
            if isinstance(result.get("compliance"), dict):
                self._compliance = {str(k): float(v) for k, v in result["compliance"].items()}
            if isinstance(result.get("binding_fingerprint"), dict):
                self._binding_fingerprint = {str(k): str(v) for k, v in result["binding_fingerprint"].items()}
            self._fault = str(result["fault"]) if result.get("fault") else None
        return result

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                message = self._queue.get(timeout=.1)
            except queue.Empty:
                if self._session_id and time.monotonic() - self._last_health >= .2:
                    self._last_health = time.monotonic()
                    try:
                        self._request_id += 1
                        self._queue.put_nowait({"kind": "health", "session_id": self._session_id,
                                                "generation": self._generation,
                                                "request_id": self._request_id})
                    except queue.Full:
                        pass
                continue
            try:
                self.request_now(message)
            except Exception as error:
                with self._lock:
                    self._fault = str(error)

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=max(.2, self.timeout_s + .2))
