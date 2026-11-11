"""Franky Cartesian impedance worker.

The module is importable without Franky installed.  The binding is imported
only when a real arm is opened, which lets ROS and protocol tests run in the
system Python environment.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from vla.core.hardware_lease import HardwareLease
from .ipc import decode_packet, encode_packet
from .binding import check_binding


@dataclass
class Compliance:
    translational_stiffness: float = 1000.0
    rotational_stiffness: float = 50.0
    nullspace_stiffness: float = 20.0
    translational_clip: float = .008
    rotational_clip: float = .04
    max_step: float = .03
    max_step_rad: float = .10
    max_delta_tau: float = .2
    gains_time_constant: float = .1

    def validate(self) -> "Compliance":
        values = asdict(self)
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("Franky compliance parameters must be finite")
        if not (0 < self.translational_stiffness <= 1200):
            raise ValueError("translational_stiffness must be in (0, 1200]")
        if not (0 < self.rotational_stiffness <= 80):
            raise ValueError("rotational_stiffness must be in (0, 80]")
        if self.nullspace_stiffness < 0 or self.translational_clip < .005:
            raise ValueError("nullspace stiffness or translational clip is unsafe")
        if self.rotational_clip < .02 or self.max_step <= 0 or self.max_step_rad <= 0:
            raise ValueError("rotational clip and target steps must be positive")
        if self.max_delta_tau <= 0 or self.gains_time_constant <= 0:
            raise ValueError("tracker dynamics must be positive")
        return self

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None,
                     current: "Compliance | None" = None) -> "Compliance":
        result = cls(**(asdict(current) if current else {}))
        for key, value in (values or {}).items():
            if key not in asdict(result):
                raise ValueError(f"unknown Franky compliance parameter: {key}")
            setattr(result, key, float(value))
        return result.validate()


def _pose_to_affine(franky: Any, pose: np.ndarray) -> Any:
    pose = np.asarray(pose, dtype=np.float64)
    quat = pose[3:] / np.linalg.norm(pose[3:])
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quat).as_matrix()
    matrix[:3, 3] = pose[:3]
    return franky.Affine(matrix)


def _affine_pose(value: Any) -> np.ndarray:
    """Normalize Franky Affine or libfranka 4x4 column-major arrays."""
    if hasattr(value, "translation") and hasattr(value, "quaternion"):
        return np.r_[np.asarray(value.translation, dtype=float),
                     np.asarray(value.quaternion, dtype=float)]
    matrix = np.asarray(value, dtype=float)
    if matrix.size == 16:
        matrix = matrix.reshape(4, 4, order="F")
        return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()]
    raise ValueError("Franky state pose is not an Affine or 4x4 matrix")


class FrankyArm:
    """One exclusive CartesianImpedanceTracker session."""

    def __init__(self, name: str, robot_ip: str, *, gripper_port: str | None = None,
                 compliance: Compliance | None = None, realtime_config: str = "ignore"):
        self.name = name
        self.robot_ip = robot_ip
        self.compliance = (compliance or Compliance()).validate()
        self.realtime_config = realtime_config
        # The Franky worker owns the FCI arm exclusively. Gripper ownership
        # remains with the ROS peripheral stack, so gripper_port is optional.
        self._lease = None
        self._gripper_port = gripper_port
        self.franky = None
        self.robot = None
        self.tracker = None
        self.target_pose: np.ndarray | None = None
        self.target_generation = -1
        self.target_sequence = -1
        self.fault: str | None = None
        self.binding_fingerprint: dict[str, str] = {}
        self._robot_stamp = None
        self._robot_stamp_at = time.monotonic()
        self.state_timeout_s = .5

    def open(self) -> None:
        try:
            self.binding_fingerprint = check_binding()
            import franky
        except Exception as error:
            if self._lease:
                self._lease.close()
            raise RuntimeError(f"Franky binding unavailable: {error}") from error
        self.franky = franky
        self._lease = HardwareLease(self.robot_ip, self._gripper_port)
        mode = getattr(franky.RealtimeConfig, "Enforce" if self.realtime_config == "enforce" else "Ignore")
        self.robot = franky.Robot(self.robot_ip, realtime_config=mode)
        self.robot.relative_dynamics_factor = .2
        self.robot.set_collision_behavior(
            [80.0] * 4 + [11.0] * 3,
            [100.0] * 3 + [25.0] * 3,
        )

    def _state(self) -> dict[str, Any]:
        if self.robot is None:
            raise RuntimeError("Franky arm is not open")
        # Surface a tracker thread that stopped between target packets. The
        # collector polls health independently of command traffic, so an
        # asynchronous Franky exception cannot remain hidden until the next
        # user action.
        self._check_tracker()
        raw = self.robot.state
        if self.tracker is not None and hasattr(raw, 'time'):
            stamp = raw.time.to_sec()
            if stamp != self._robot_stamp:
                self._robot_stamp, self._robot_stamp_at = stamp, time.monotonic()
            elif time.monotonic() - self._robot_stamp_at > self.state_timeout_s:
                raise RuntimeError('Franky measured state timestamp stalled')
        affine = raw.O_T_EE
        pose = _affine_pose(affine)
        desired = getattr(raw, "O_T_EE_d", None)
        desired_pose = _affine_pose(desired) if desired is not None else self.target_pose
        flange = getattr(raw, "F_T_EE", None)
        flange_pose = (_affine_pose(flange) if flange is not None
                       else np.array([0., 0., 0., 0., 0., 0., 1.]))
        q = np.asarray(raw.q, dtype=float)
        dq = np.asarray(raw.dq, dtype=float)
        qd = np.asarray(getattr(raw, "q_d", q), dtype=float)
        tau = np.asarray(getattr(raw, "tau_J", np.zeros(7)), dtype=float)
        tau_ext = np.asarray(getattr(raw, "tau_ext_hat_filtered", np.zeros(7)), dtype=float)
        force = np.asarray(getattr(raw, "K_F_ext_hat_K", np.zeros(6)), dtype=float)
        if any(not np.isfinite(value).all() for value in (pose, q, dq, qd, tau, tau_ext, force)):
            raise RuntimeError("non-finite Franky measured state")
        return {"arm": self.name, "observed_at_ns": time.monotonic_ns(),
                "tcp_pose": pose.tolist(), "joints": q.tolist(),
                "joint_velocity": dq.tolist(), "joint_torque": tau.tolist(),
                "external_joint_torque": tau_ext.tolist(), "target_joints": qd.tolist(),
                "flange_to_tcp_pose": flange_pose.tolist(),
                "desired_tcp_pose": desired_pose.tolist() if desired_pose is not None else [],
                "force": force[:3].tolist(), "torque": force[3:].tolist(),
                "generation": self.target_generation, "sequence": self.target_sequence,
                "tracker_running": bool(self.tracker is not None and self._tracker_running()),
                "state_valid": True,
                "fault": self.fault}

    def _tracker_running(self) -> bool:
        if self.tracker is None:
            return False
        value = getattr(self.tracker, "is_running", False)
        try:
            return bool(value() if callable(value) else value)
        except Exception as error:
            self.fault = f"Cartesian tracker state unavailable: {error}"
            return False

    def _check_tracker(self) -> None:
        if self.fault:
            raise RuntimeError(self.fault)
        if self.tracker is not None and not self._tracker_running():
            try:
                self.robot.poll_motion()
            except Exception as error:
                self.fault = f"Cartesian tracker stopped: {error}"
            else:
                self.fault = "Cartesian tracker stopped unexpectedly"
            raise RuntimeError(self.fault)

    def _ensure_tracker(self) -> None:
        self._check_tracker()
        if self.tracker is not None:
            return
        # Only explicit recover may clear robot errors.
        self.robot.join_motion()
        nullspace = np.asarray(self.robot.state.q, dtype=np.float64).copy()
        c = self.compliance
        self.tracker = self.franky.CartesianImpedanceTracker(
            self.robot,
            translational_stiffness=c.translational_stiffness,
            rotational_stiffness=c.rotational_stiffness,
            nullspace_target=nullspace,
            nullspace_stiffness=c.nullspace_stiffness,
            translational_error_clip=np.full(3, c.translational_clip),
            rotational_error_clip=np.full(3, c.rotational_clip),
            max_delta_tau=c.max_delta_tau,
            gains_time_constant=c.gains_time_constant,
        )
        # Seed the target accumulator from the measured TCP. The first client
        # packet is therefore limited by max_step instead of causing a pose
        # discontinuity.
        self.target_pose = _affine_pose(self.robot.state.O_T_EE)

    def set_target(self, pose: list[float], generation: int, sequence: int) -> None:
        self._ensure_tracker()
        pose_array = np.array(pose, dtype=np.float64, copy=True)
        pose_array[3:] /= np.linalg.norm(pose_array[3:])
        if self.target_pose is not None:
            delta = pose_array[:3] - self.target_pose[:3]
            distance = float(np.linalg.norm(delta))
            if distance > self.compliance.max_step:
                pose_array[:3] = self.target_pose[:3] + delta * self.compliance.max_step / distance
            previous = Rotation.from_quat(self.target_pose[3:])
            current = Rotation.from_quat(pose_array[3:])
            delta_r = current * previous.inv()
            vector = delta_r.as_rotvec()
            angle = float(np.linalg.norm(vector))
            if angle > self.compliance.max_step_rad:
                pose_array[3:] = (Rotation.from_rotvec(vector * self.compliance.max_step_rad / angle) * previous).as_quat()
        self.tracker.set_target(_pose_to_affine(self.franky, pose_array))
        self.target_pose = pose_array
        self.target_generation = generation
        self.target_sequence = sequence
        self._check_tracker()

    def reset(self, generation: int) -> None:
        self.stop(recover=False)
        if self.fault:
            raise RuntimeError(self.fault)
        self.target_generation = generation
        self.target_sequence = -1

    def stop(self, *, recover: bool = False) -> None:
        if self.tracker is not None:
            try:
                self.tracker.stop()
            finally:
                self.tracker = None
        if self.robot is not None:
            self.robot.join_motion()
        self.target_pose = None
        if recover and self.robot is not None:
            self.robot.recover_from_errors()

    def recover(self) -> None:
        if self.robot is None:
            raise RuntimeError("Franky arm is not open")
        if self.tracker is not None:
            try:
                self.tracker.stop()
            finally:
                self.tracker = None
        self.robot.recover_from_errors()
        self.fault = None
        self.target_pose = None
        self._robot_stamp = None

    def update_compliance(self, values: Mapping[str, Any]) -> Compliance:
        next_value = Compliance.from_mapping(values, self.compliance)
        if self.tracker is not None and any(key in values for key in (
                "translational_clip", "rotational_clip", "max_delta_tau", "gains_time_constant")):
            pose = self.target_pose.copy() if self.target_pose is not None else None
            self.stop()
            self.compliance = next_value
            if pose is not None:
                self.set_target(pose.tolist(), self.target_generation, self.target_sequence)
        else:
            self.compliance = next_value
            if self.tracker is not None:
                self.tracker.set_gains(
                    translational_stiffness=next_value.translational_stiffness,
                    rotational_stiffness=next_value.rotational_stiffness,
                    nullspace_stiffness=next_value.nullspace_stiffness,
                )
        return self.compliance

    def close(self) -> None:
        try:
            self.stop(recover=False)
        finally:
            # Destroy the FCI connection before making its lease available.
            self.robot = None
            if self._lease:
                self._lease.close()
                self._lease = None


class FrankyWorker:
    """One generation-fenced owner, with a watchdog independent of client traffic."""
    IMPLEMENTATION_VERSION = 3

    def __init__(self, arms, *, session_id, compliance=None, target_timeout_s=.1,
                 state_timeout_s=.5):
        if not session_id or not arms:
            raise ValueError('worker requires a session and at least one arm')
        self.session_id = str(session_id)
        base = Compliance.from_mapping(compliance)
        self.arms = {name: FrankyArm(name, str(value['robot_ip']),
                                    compliance=Compliance.from_mapping({}, base))
                     for name, value in arms.items()}
        self._lock = threading.RLock()
        self.generation = self.sequence = self.timestamp_ns = -1
        self.fault = None
        self.binding_fingerprint = {}
        self.target_timeout_s, self.state_timeout_s = float(target_timeout_s), float(state_timeout_s)
        if not all(math.isfinite(x) and x > 0 for x in (self.target_timeout_s, self.state_timeout_s)):
            raise ValueError('worker timeouts must be positive and finite')
        for arm in self.arms.values():
            arm.state_timeout_s = self.state_timeout_s
        self._armed = self._streaming = False
        self._connected = False
        self._last_target = self._last_contact = time.monotonic()
        self._requests = {}
        self._closed = threading.Event()
        self._watchdog = None
        self._compliance = asdict(base)

    def open(self):
        try:
            for arm in self.arms.values():
                arm.open()
                self.binding_fingerprint.update(arm.binding_fingerprint)
        except BaseException:
            self.close()
            raise
        self._watchdog = threading.Thread(target=self._watch, name='franky-watchdog', daemon=True)
        self._watchdog.start()

    def _stop_all(self):
        errors = []
        self._armed = self._streaming = False
        for arm in self.arms.values():
            try:
                arm.stop()
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError('; '.join(errors))

    def latch_fault(self, reason):
        with self._lock:
            self.fault = self.fault or str(reason)
            try:
                self._stop_all()
            except Exception as error:
                self.fault += f'; stop failed: {error}'

    def _watch(self):
        while not self._closed.wait(.01):
            with self._lock:
                if self.fault:
                    continue
                now = time.monotonic()
                if self._connected and now - self._last_contact > self.state_timeout_s:
                    self.latch_fault('Franky client heartbeat expired')
                elif self._streaming and now - self._last_target > self.target_timeout_s:
                    self.latch_fault('Franky target stream expired')
                elif self._streaming:
                    self.snapshot()  # Poll asynchronous tracker failures even without requests.

    def request(self, message):
        with self._lock:
            if message.get('session_id') != self.session_id:
                raise ValueError('wrong Franky worker session')
            kind = message.get('kind')
            request_id = message.get('request_id')
            generation = message.get('generation')
            if (not isinstance(request_id, (str, int)) or isinstance(request_id, bool)
                    or not str(request_id) or len(str(request_id)) > 128
                    or type(generation) is not int or generation < -1):
                raise ValueError('invalid worker request id or generation')
            key = str(request_id)
            signature = json.dumps(message, sort_keys=True)
            if key in self._requests:
                old_signature, response = self._requests[key]
                if old_signature != signature:
                    raise ValueError('worker request id reused with different content')
                # A retry never repeats a motion or refreshes a watchdog.
                if self.fault != response.get('fault') or self.generation != response.get('generation'):
                    return {**self.snapshot(), 'request_id': key, 'request_status': 'ignored'}
                return response
            self._last_contact = time.monotonic()
            self._connected = True
            status = 'applied'
            error = None
            try:
                if kind == 'health':
                    pass
                elif kind not in ('target', 'reset', 'stop', 'recover', 'compliance'):
                    raise ValueError(f'unknown Franky worker request: {kind}')
                elif generation < self.generation:
                    status = 'ignored'
                elif kind == 'stop':
                    self._stop_all()
                    self.generation = generation
                elif kind == 'recover':
                    # join_motion may report the already-latched control error.
                    # Recovery is allowed only once all control threads stopped.
                    try:
                        self._stop_all()
                    except Exception:
                        if any(arm.robot is not None and arm.robot.is_in_control
                               for arm in self.arms.values()):
                            raise
                    for arm in self.arms.values():
                        arm.recover()
                        arm.update_compliance(self._compliance)
                    self.fault = None
                    self.generation = generation
                    self.sequence = self.timestamp_ns = -1
                elif self.fault:
                    raise RuntimeError(f'explicit recovery required: {self.fault}')
                elif kind == 'reset':
                    if generation <= self.generation:
                        status = 'ignored'
                    else:
                        self._stop_all()
                        for arm in self.arms.values():
                            arm.reset(generation)
                        self.generation = generation
                        self.sequence = self.timestamp_ns = -1
                        self._armed = True
                elif generation != self.generation:
                    raise ValueError('target/compliance generation has not been reset')
                elif kind == 'target':
                    sequence, stamp = message.get('sequence'), message.get('timestamp_ns')
                    if type(sequence) is not int or sequence < 0 or type(stamp) is not int:
                        raise ValueError('invalid target sequence/timestamp')
                    if sequence <= self.sequence or stamp <= self.timestamp_ns:
                        status = 'ignored'
                    else:
                        if not self._armed:
                            raise RuntimeError('worker requires a new reset before targets')
                        age = (time.monotonic_ns() - stamp) / 1e9
                        if not 0 <= age <= self.target_timeout_s:
                            raise ValueError('stale or future Franky target')
                        targets = message.get('arms', {})
                        if set(targets) != set(self.arms):
                            raise ValueError('target must contain every configured Franky arm')
                        from .ipc import _validate_pose
                        poses = {name: _validate_pose(value['pose']) for name, value in targets.items()}
                        for name, pose in poses.items():
                            self.arms[name].set_target(pose, generation, sequence)
                        self.sequence, self.timestamp_ns = sequence, stamp
                        self._last_target = time.monotonic()
                        self._streaming = True
                elif kind == 'compliance':
                    values = message.get('values', {})
                    next_value = Compliance.from_mapping(values, Compliance.from_mapping(self._compliance))
                    for arm in self.arms.values():
                        arm.update_compliance(values)
                    self._compliance = asdict(next_value)  # Commit only after all arms succeed.
                    self._last_target = time.monotonic()  # One bounded interval for target resumption.
            except Exception as exc:
                error = str(exc)
                if kind != 'health':
                    self.latch_fault(f'{kind}: {error}')
            response = {**self.snapshot(), 'request_id': key,
                        'request_status': 'failed' if error else status}
            if error:
                response['error'] = error
            # A synchronous startup/rebuild may consume time while the caller
            # awaits its reply. Heartbeat silence starts after that reply.
            self._last_contact = time.monotonic()
            self._requests[key] = (signature, response)
            while len(self._requests) > 256:
                self._requests.pop(next(iter(self._requests)))
            return response

    def snapshot(self):
        with self._lock:
            states = []
            for arm in self.arms.values():
                try:
                    state = arm._state()
                    if not state.get('state_valid', True) or state.get('fault'):
                        raise RuntimeError(state.get('fault') or 'invalid measured state')
                    states.append(state)
                except Exception as error:
                    self.latch_fault(f'{arm.name}: {error}')
                    states.append({'arm': arm.name, 'observed_at_ns': time.monotonic_ns(),
                                   'tcp_pose': [0, 0, 0, 0, 0, 0, 1], 'joints': [0]*7,
                                   'joint_velocity': [0]*7, 'state_valid': False, 'fault': str(error)})
            return {'kind': 'state', 'session_id': self.session_id,
                    'generation': self.generation, 'sequence': self.sequence,
                    'states': states, 'fault': self.fault,
                    'implementation_version': self.IMPLEMENTATION_VERSION,
                    'binding_fingerprint': dict(self.binding_fingerprint),
                    'compliance': dict(self._compliance)}

    def close(self):
        self._closed.set()
        if self._watchdog and self._watchdog is not threading.current_thread():
            self._watchdog.join(timeout=1.)
        errors = []
        for arm in reversed(tuple(self.arms.values())):
            try:
                arm.close()
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError('; '.join(errors))


class FrankyWorkerServer:
    def __init__(self, worker, path):
        self.worker = worker
        self.path = Path(path).expanduser().resolve()
        self._closed = threading.Event()
        self._listener = None
        self._socket_identity = None

    def serve(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.worker.open()
            if self.path.exists():
                # Never unlink a socket belonging to another worker, even if
                # it uses different robot leases. Stale sockets require cleanup.
                raise FileExistsError(f'worker socket already exists: {self.path}')
            listener.bind(str(self.path))
            stat = self.path.stat()
            self._socket_identity = (stat.st_dev, stat.st_ino)
            os.chmod(self.path, 0o600)
            listener.listen(16)
            listener.settimeout(.1)
            self._listener = listener
            while not self._closed.is_set():
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._closed.is_set():
                        break
                    raise
                with connection:
                    connection.settimeout(.05)
                    try:
                        message = decode_packet(connection)
                        response = self.worker.request(message)
                        connection.sendall(encode_packet(response))
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        self.worker.latch_fault(f'IPC: {error}')
        finally:
            listener.close()
            try:
                self.worker.close()
            finally:
                if self.path.exists() and self._socket_identity is not None:
                    stat = self.path.stat()
                    if (stat.st_dev, stat.st_ino) == self._socket_identity:
                        self.path.unlink()

    def close(self):
        self._closed.set()
        if self._listener is not None:
            try:
                self._listener.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._listener.close()
