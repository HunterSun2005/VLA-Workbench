"""Small versioned Unix stream protocol for the Franky worker.

The protocol deliberately contains only JSON and fixed-size numeric arrays.  A
worker connection handles one request and one response; this keeps reconnects
and ownership boundaries explicit while the client sends targets from a
single outstanding target and a priority lifecycle queue.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from collections import deque
import uuid
import socket
import struct
import threading
import time
from typing import Any, Mapping


IPC_VERSION = 2
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


def _read_exact(sock: socket.socket, size: int, deadline: float | None = None) -> bytes:
    value = bytearray()
    while len(value) < size:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'worker packet deadline exceeded ({len(value)}/{size} bytes)')
            sock.settimeout(remaining)
        chunk = sock.recv(size - len(value))
        if not chunk:
            raise ConnectionError("worker connection closed")
        value.extend(chunk)
    return bytes(value)


def decode_packet(sock: socket.socket, *, deadline: float | None = None) -> dict[str, Any]:
    length = struct.unpack("!I", _read_exact(sock, 4, deadline))[0]
    if not 0 < length <= MAX_PACKET:
        raise ValueError("worker packet is too large")
    value = json.loads(_read_exact(sock, length, deadline).decode("utf-8"))
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
    reference_tcp_pose: tuple[float, ...] = ()
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
        import math
        for key, width in (('joints', 7), ('joint_velocity', 7), ('joint_torque', 7),
                           ('external_joint_torque', 7), ('target_joints', 7), ('force', 3), ('torque', 3)):
            values = value.get(key, ())
            if values and (len(values) != width or not all(math.isfinite(float(x)) for x in values)):
                raise ValueError(f'invalid worker state {key}')
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
            reference_tcp_pose=tuple(_validate_pose(value["reference_tcp_pose"])) if value.get("reference_tcp_pose") else (),
            generation=int(value.get("generation", -1)),
            sequence=int(value.get("sequence", -1)),
            tracker_running=bool(value.get("tracker_running", False)),
            state_valid=bool(value.get("state_valid", not value.get("fault"))),
            fault=str(value["fault"]) if value.get("fault") else None,
        )


class FrankyWorkerClient:
    """Bounded background IPC; lifecycle fences also cancel dequeued targets."""
    def __init__(self, path, *, timeout_s=.5, target_timeout_s=.1, queue_size=8, session_id=None):
        self.path = Path(path).expanduser().resolve()
        self.timeout_s = float(timeout_s)
        self.target_timeout_s = float(target_timeout_s)
        self._target_pending = None
        self._queue = deque()
        self._queue_size = queue_size
        self._condition = threading.Condition(threading.RLock())
        self._lock = self._condition
        self._request_lock = threading.Lock()
        self._states = {}
        self._compliance = {}
        self._binding_fingerprint = {}
        self._results = {}
        self._fault = None
        self.transport_failed = False
        self._session_id = session_id
        self._generation = -1
        self._epoch = 0
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, name='franky-ipc', daemon=True)
        self._thread.start()

    @property
    def fault(self):
        with self._lock:
            return self._fault

    def states(self):
        with self._lock:
            return dict(self._states)

    def compliance_values(self):
        with self._lock:
            return dict(self._compliance)

    def binding_fingerprint(self):
        with self._lock:
            return dict(self._binding_fingerprint)

    def result(self, request_id):
        with self._lock:
            return self._results.get(str(request_id))

    @property
    def target_available(self):
        with self._lock:
            return not self._closed.is_set() and not self._fault and self._target_pending is None and not self._queue

    def _message(self, kind, session_id=None, generation=None, **values):
        return {'kind': kind, 'session_id': session_id or self._session_id,
                'generation': self._generation if generation is None else int(generation),
                'request_id': uuid.uuid4().hex, **values}

    def _fence(self, message):
        # Called under the condition lock, before the lifecycle message is visible.
        self._generation = max(self._generation, int(message['generation']))
        self._epoch += 1
        for _, pending in self._queue:
            if pending['kind'] == 'target':
                self._results[pending['request_id']] = {'request_status': 'cancelled'}
                self._target_pending = None
        self._queue = deque(item for item in self._queue if item[1]['kind'] != 'target'
                            and item[1]['generation'] >= self._generation)

    def _enqueue(self, message):
        with self._condition:
            if self._closed.is_set():
                raise RuntimeError('worker client is closed')
            lifecycle = message['kind'] in ('stop', 'reset', 'recover', 'compliance')
            if lifecycle:
                self._fence(message)
            if len(self._queue) >= self._queue_size:
                self._fault = 'Franky worker lifecycle queue is full'
                raise RuntimeError(self._fault)
            if message['kind'] == 'stop':
                self._queue.appendleft((self._epoch, message))
            else:
                self._queue.append((self._epoch, message))
            self._condition.notify()
            return message['request_id']

    def target(self, session_id, generation, sequence, arms, timestamp_ns=None, *, slot_id):
        targets = {str(name): {'pose': _validate_pose(value['pose'] if isinstance(value, Mapping) else value)}
                   for name, value in arms.items()}
        with self._condition:
            if not self.target_available:
                raise RuntimeError('Franky target channel is busy or unavailable')
            message = self._message('target', session_id, generation, sequence=int(sequence),
                                    slot_id=int(slot_id), arms=targets,
                                    timestamp_ns=time.monotonic_ns() if timestamp_ns is None else int(timestamp_ns))
            self._target_pending = message['request_id']
            self._queue.append((self._epoch, message))
            self._condition.notify()
            return message['request_id']

    def reset(self, session_id, generation):
        return self._enqueue(self._message('reset', session_id, generation))

    def stop(self, session_id, generation):
        return self._enqueue(self._message('stop', session_id, generation))

    def recover(self, session_id, generation):
        return self._enqueue(self._message('recover', session_id, generation))

    def compliance(self, session_id, generation, request_id, values):
        return self._enqueue(self._message('compliance', session_id, generation,
                                    request_id=str(request_id), values=dict(values)))

    def _request(self, message):
        # Serialize before connecting: the peer should not wait for local work.
        packet = encode_packet(message)
        started = time.monotonic()
        deadline = (message["timestamp_ns"] / 1e9 + self.target_timeout_s
                    if message["kind"] == "target" else started + self.timeout_s)
        stage = 'connect'
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("request deadline exceeded before connect")
                sock.settimeout(remaining)
                sock.connect(str(self.path))
                stage = 'send'
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('request deadline exceeded before send')
                sock.settimeout(remaining)
                sock.sendall(packet)
                stage = 'receive'
                return decode_packet(sock, deadline=deadline)
            except (OSError, ValueError) as error:
                self.transport_failed = True
                raise RuntimeError(
                    f'Franky IPC {message.get("kind")} {stage} failed '
                    f'(request_id={message.get("request_id")}, generation={message.get("generation")}, '
                    f'elapsed={time.monotonic()-started:.3f}s, socket={self.path}): '
                    f'{type(error).__name__}: {error}') from error

    def _exchange(self, message):
        result = self._request(message)
        if (result.get('session_id') != message['session_id']
                or result.get('request_id') != str(message['request_id'])
                or result.get('kind') != 'state'
                or type(result.get('generation')) is not int
                or result['generation'] < message['generation']):
            raise RuntimeError('worker response identity mismatch')
        if message['kind'] == 'target' and result.get('request_status') == 'applied':
            if (result['generation'] != message['generation']
                    or result.get('sequence') != message['sequence']
                    or result.get('slot_id') != message['slot_id']
                    or not message['timestamp_ns'] <= result.get('applied_at_ns', -1) <= time.monotonic_ns()
                    or set(result.get('references', {})) != set(message['arms'])):
                raise RuntimeError('worker target acknowledgement mismatch')
            for pose in result['references'].values():
                _validate_pose(pose)
        states = {value['arm']: WorkerState.from_dict(value) for value in result.get('states', ())}
        error = result.get('error') or result.get('fault')
        if message['kind'] == 'target' and result.get('request_status') != 'applied':
            error = error or 'target was not acknowledged as applied'
        if any(not value.state_valid or value.fault for value in states.values()):
            error = error or 'invalid Franky state'
        with self._lock:
            if message['generation'] < self._generation:
                self._results[str(message['request_id'])] = {'request_status': 'cancelled'}
                return result
            self._states = states
            self._compliance = dict(result.get('compliance', {}))
            self._binding_fingerprint = dict(result.get('binding_fingerprint', {}))
            self._results[str(message['request_id'])] = result
            while len(self._results) > 128:
                self._results.pop(next(iter(self._results)))
            if error:
                self._fault = self._fault or str(error)
            elif message['kind'] == 'recover' and result.get('request_status') == 'applied':
                self._fault = None
        if error:
            raise RuntimeError(str(error))
        return result

    def request_now(self, message):
        message = dict(message)
        with self._lock:
            if message['kind'] in ('stop', 'reset', 'recover', 'compliance'):
                self._fence(message)
        try:
            with self._request_lock:
                return self._exchange(message)
        except Exception as error:
            with self._lock:
                self._fault = self._fault or str(error)
            raise

    def _run(self):
        while not self._closed.is_set():
            with self._condition:
                if not self._queue:
                    self._condition.wait(.1)
                if self._closed.is_set():
                    break
                if self._queue:
                    epoch, message = self._queue.popleft()
                elif self._session_id:
                    epoch, message = self._epoch, self._message('health')
                else:
                    continue
            try:
                with self._request_lock:
                    with self._lock:
                        if message['kind'] == 'target' and (epoch != self._epoch or self._fault):
                            continue
                    self._exchange(message)
            except Exception as error:
                with self._lock:
                    self._fault = self._fault or str(error)
                    self._queue = deque(item for item in self._queue if item[1]['kind'] != 'target')
                    self._results.setdefault(str(message['request_id']), {'request_status': 'failed', 'error': str(error)})
                    while len(self._results) > 128:
                        self._results.pop(next(iter(self._results)))

            finally:
                with self._condition:
                    if message['kind'] == 'target' and self._target_pending == message['request_id']:
                        self._target_pending = None

    def close(self):
        with self._condition:
            self._closed.set()
            self._epoch += 1
            self._queue.clear()
            self._condition.notify_all()
        self._thread.join(timeout=self.timeout_s + .2)
