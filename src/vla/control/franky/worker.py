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
import sys
import threading
import time
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from vla.core.hardware_lease import HardwareLease
from .ipc import decode_packet, encode_packet
from .binding import check_binding
from .tcp_frame import TCPFrame, TCP_FRAME
from .pose_correction import PoseErrorCompensator


@dataclass
class Compliance:
    translational_stiffness: float = 1100.0
    rotational_stiffness: float = 65.0
    nullspace_stiffness: float = 20.0
    translational_clip: float = .008
    rotational_clip: float = .04
    max_step: float = .03
    max_step_rad: float = .10
    max_delta_tau: float = .2
    gains_time_constant: float = .1
    tracking_integral_gain: float = 0.
    tracking_max_offset_m: float = .01
    tracking_max_offset_rad: float = .20
    tracking_max_linear_rate: float = .002
    tracking_max_angular_rate: float = .03

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
        for key, lower, upper in (
                ('tracking_integral_gain', 0., 1.),
                ('tracking_max_offset_m', 0., .02), ('tracking_max_offset_rad', 0., .35),
                ('tracking_max_linear_rate', 0., .01), ('tracking_max_angular_rate', 0., .10)):
            value = getattr(self, key)
            if not lower <= value <= upper or (key != 'tracking_integral_gain' and value == 0):
                raise ValueError(f'{key} outside allowed range ({lower}, {upper})')
        return self

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None,
                     current: "Compliance | None" = None) -> "Compliance":
        result = cls(**(asdict(current) if current else {}))
        for key, value in (values or {}).items():
            if key not in asdict(result):
                raise ValueError(f"unknown Franky compliance parameter: {key}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number")
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
                 compliance: Compliance | None = None, realtime_config: str = "ignore",
                 workspace_bounds=None):
        self.name = name
        self.robot_ip = robot_ip
        self.compliance = (compliance or Compliance()).validate()
        self.realtime_config = realtime_config
        self.workspace_bounds = None
        if workspace_bounds is not None:
            bounds = np.asarray(workspace_bounds, dtype=float)
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
                raise ValueError('invalid Franky workspace bounds')
            self.workspace_bounds = bounds.copy()
        # The Franky worker owns the FCI arm exclusively. Gripper ownership
        # remains with the ROS peripheral stack, so gripper_port is optional.
        self._lease = None
        self._gripper_port = gripper_port
        self.franky = None
        self.robot = None
        self.tracker = None
        self.target_pose: np.ndarray | None = None
        self.nominal_pose: np.ndarray | None = None
        self.correction = PoseErrorCompensator()
        self._target_time = None
        self._tcp_frame: TCPFrame | None = None
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

    def _frame(self, raw=None) -> TCPFrame:
        raw = self.robot.state if raw is None else raw
        flange = getattr(raw, 'F_T_EE', None)
        if flange is None:
            raise RuntimeError('Franky measured F_T_EE is required for TCP frame calibration')
        hardware_tool = _affine_pose(flange)
        if self._tcp_frame is None:
            self._tcp_frame = TCPFrame(hardware_tool)
        else:
            self._tcp_frame.check_tool(hardware_tool)
        return self._tcp_frame

    def _state(self) -> dict[str, Any]:
        if self.robot is None:
            raise RuntimeError("Franky arm is not open")
        # Surface a tracker thread that stopped between target packets. The
        # collector polls health independently of command traffic, so an
        # asynchronous Franky exception cannot remain hidden until the next
        # user action.
        self._check_tracker()
        raw = self.robot.state
        self._check_state_time(raw)
        frame = self._frame(raw)
        hardware_pose = _affine_pose(raw.O_T_EE)
        pose = frame.to_logical(hardware_pose)
        desired = getattr(raw, "O_T_EE_d", None)
        desired_pose = frame.to_logical(_affine_pose(desired)) if desired is not None else self.target_pose
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
                "flange_to_tcp_pose": frame.logical_tool.tolist(),
                "hardware_flange_to_tcp_pose": frame.hardware_tool.tolist(),
                "hardware_tcp_pose": hardware_pose.tolist(),
                "hardware_reference_tcp_pose": (frame.to_hardware(self.target_pose).tolist()
                                                if self.target_pose is not None else []),
                "desired_tcp_pose": desired_pose.tolist() if desired_pose is not None else [],
                "reference_tcp_pose": self.target_pose.tolist() if self.target_pose is not None else [],
                "nominal_tcp_pose": self.nominal_pose.tolist() if self.nominal_pose is not None else [],
                "tracking_offset": self.correction.offset.tolist(),
                "tracking_status": self.correction.status,
                "force": force[:3].tolist(), "torque": force[3:].tolist(),
                "generation": self.target_generation, "sequence": self.target_sequence,
                "tracker_running": bool(self.tracker is not None and self._tracker_running()),
                "state_valid": True,
                "fault": self.fault}

    def _check_state_time(self, raw):
        if self.tracker is not None and hasattr(raw, 'time'):
            stamp = float(raw.time.to_sec())
            if not math.isfinite(stamp) or (self._robot_stamp is not None and stamp < self._robot_stamp):
                raise RuntimeError('invalid or regressing Franky measured state timestamp')
            if stamp != self._robot_stamp:
                self._robot_stamp, self._robot_stamp_at = stamp, time.monotonic()
            elif time.monotonic() - self._robot_stamp_at > self.state_timeout_s:
                raise RuntimeError('Franky measured state timestamp stalled')

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

    def _start_tracker(self) -> None:
        self._check_tracker()
        if self.tracker is not None:
            return
        # Only explicit recover may clear robot errors.
        self.robot.join_motion()
        raw = self.robot.state
        self._frame(raw)  # Reject unknown/changed tools before starting control.
        nullspace = np.asarray(raw.q, dtype=np.float64).copy()
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
        raw = self.robot.state
        self.target_pose = self._frame(raw).to_logical(_affine_pose(raw.O_T_EE))
        self.nominal_pose = self.target_pose.copy()
        self.correction.reset()
        self._target_time = None

    def set_target(self, pose: list[float], generation: int, sequence: int) -> None:
        self._check_tracker()
        if self.tracker is None:
            raise RuntimeError("tracker is not armed; explicit reset required")
        pose_array = np.array(pose, dtype=np.float64, copy=True)
        pose_array[3:] /= np.linalg.norm(pose_array[3:])
        pose_array = self._limit_target(pose_array, self.nominal_pose)
        nominal = pose_array.copy()
        now = time.monotonic()
        dt = min(1/30, max(0., now-self._target_time)) if self._target_time is not None else 0.
        raw = self.robot.state
        self._check_state_time(raw)
        frame = self._frame(raw)
        measured = frame.to_logical(_affine_pose(raw.O_T_EE))
        pose_array = self.correction.update(pose_array, measured, dt, self.compliance)
        if self.workspace_bounds is not None:
            projected = np.clip(pose_array[:3], *self.workspace_bounds)
            if np.linalg.norm(projected-pose_array[:3]) > 1e-12:
                pose_array[:3] = projected
                self.correction.offset[:3] = projected-nominal[:3]
                self.correction.status = 'workspace_limited'
        limited = self._limit_target(pose_array, self.target_pose)
        if (np.linalg.norm(limited[:3]-pose_array[:3]) > 1e-12
                or (Rotation.from_quat(limited[3:]) * Rotation.from_quat(pose_array[3:]).inv()).magnitude() > 1e-12):
            # The last step limiter also participates in anti-windup.
            self.correction.offset[:3] = limited[:3]-nominal[:3]
            self.correction.offset[3:] = (Rotation.from_quat(limited[3:]) * Rotation.from_quat(nominal[3:]).inv()).as_rotvec()
            self.correction.status = 'step_limited'
        self.tracker.set_target(_pose_to_affine(self.franky, frame.to_hardware(limited)))
        self.target_pose, self.nominal_pose = limited, nominal
        self._target_time = now
        self.target_generation = generation
        self.target_sequence = sequence
        self._check_tracker()

    def _limit_target(self, pose_array, previous_pose):
        pose_array = pose_array.copy()
        if previous_pose is not None:
            delta = pose_array[:3] - previous_pose[:3]
            distance = float(np.linalg.norm(delta))
            if distance > self.compliance.max_step:
                pose_array[:3] = previous_pose[:3] + delta * self.compliance.max_step / distance
            previous = Rotation.from_quat(previous_pose[3:])
            current = Rotation.from_quat(pose_array[3:])
            delta_r = current * previous.inv()
            vector = delta_r.as_rotvec()
            angle = float(np.linalg.norm(vector))
            if angle > self.compliance.max_step_rad:
                pose_array[3:] = (Rotation.from_rotvec(vector * self.compliance.max_step_rad / angle) * previous).as_quat()
        return pose_array

    def reset(self, generation: int) -> None:
        self.stop(recover=False)
        if self.fault:
            raise RuntimeError(self.fault)
        self.target_generation = generation
        self.target_sequence = -1
        self._start_tracker()

    def stop(self, *, recover: bool = False) -> None:
        try:
            if self.tracker is not None:
                try:
                    self.tracker.stop()
                finally:
                    self.tracker = None
            if self.robot is not None:
                self.robot.join_motion()
        finally:
            self.target_pose = None
            self.nominal_pose = None
            self.correction.reset()
            self._target_time = None
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
        self.nominal_pose = None
        self.correction.reset()
        self._target_time = None
        self._robot_stamp = None

    def update_compliance(self, values: Mapping[str, Any]) -> Compliance:
        next_value = Compliance.from_mapping(values, self.compliance)
        changed = {key for key, value in asdict(next_value).items()
                   if value != getattr(self.compliance, key)}
        if not changed:
            return self.compliance
        if self.tracker is not None and changed.intersection({
                "translational_clip", "rotational_clip", "max_delta_tau", "gains_time_constant",
                "tracking_integral_gain", "tracking_max_offset_m", "tracking_max_offset_rad",
                "tracking_max_linear_rate", "tracking_max_angular_rate"}):
            pose = self.nominal_pose.copy() if self.nominal_pose is not None else None
            self.stop()
            self.compliance = next_value
            if pose is not None:
                self._start_tracker()
                self.set_target(pose.tolist(), self.target_generation, self.target_sequence)
        else:
            self.compliance = next_value
            if self.tracker is not None and changed.intersection({
                    "translational_stiffness", "rotational_stiffness", "nullspace_stiffness"}):
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
            self._tcp_frame = None
            if self._lease:
                self._lease.close()
                self._lease = None


class FrankyWorker:
    """One generation-fenced owner, with a watchdog independent of client traffic."""
    IMPLEMENTATION_VERSION = 7

    def __init__(self, arms, *, session_id, compliance=None, target_timeout_s=.1,
                 state_timeout_s=.5):
        if not session_id or not arms:
            raise ValueError('worker requires a session and at least one arm')
        self.session_id = str(session_id)
        base = Compliance.from_mapping(compliance)
        self.arms = {name: FrankyArm(name, str(value['robot_ip']),
                                    compliance=Compliance.from_mapping({}, base),
                                    workspace_bounds=value.get('workspace_bounds'))
                     for name, value in arms.items()}
        self._lock = threading.RLock()
        self.generation = self.sequence = self.timestamp_ns = self.slot_id = -1
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
            first_fault = self.fault is None
            self.fault = self.fault or str(reason)
            try:
                self._stop_all()
            except Exception as error:
                self.fault += f'; stop failed: {error}'
            if first_fault:
                print(f'Franky worker fault session={self.session_id} generation={self.generation} '
                      f'sequence={self.sequence}: {self.fault}', file=sys.stderr, flush=True)

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
            receipt = {}
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
                    for arm in self.arms.values():
                        arm.target_generation, arm.target_sequence = generation, -1
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
                    self.sequence = self.timestamp_ns = self.slot_id = -1
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
                        self.sequence = self.timestamp_ns = self.slot_id = -1
                        self._armed = self._streaming = True
                        self._last_target = time.monotonic()
                elif generation != self.generation:
                    raise ValueError('target/compliance generation has not been reset')
                elif kind == 'target':
                    slot = message.get('slot_id')
                    if type(slot) is not int or slot < 0:
                        raise ValueError('invalid target slot_id')
                    sequence, stamp = message.get('sequence'), message.get('timestamp_ns')
                    if type(sequence) is not int or sequence < 0 or type(stamp) is not int:
                        raise ValueError('invalid target sequence/timestamp')
                    if sequence <= self.sequence or stamp <= self.timestamp_ns:
                        status = 'ignored'
                    else:
                        if slot <= self.slot_id:
                            raise ValueError('expired or out of order target slot')
                        if not self._armed:
                            raise RuntimeError('worker requires a new reset before targets')
                        if self._streaming and time.monotonic() - self._last_target > self.target_timeout_s:
                            raise RuntimeError('Franky target stream expired')
                        age = (time.monotonic_ns() - stamp) / 1e9
                        if not 0 <= age <= self.target_timeout_s:
                            raise ValueError('stale or future Franky target')
                        targets = message.get('arms', {})
                        if set(targets) != set(self.arms):
                            raise ValueError('target must contain every configured Franky arm')
                        from .ipc import _validate_pose
                        poses = {name: _validate_pose(value['pose']) for name, value in targets.items()}
                        receipt['partial_application'] = {}
                        for name, pose in poses.items():
                            self.arms[name].set_target(pose, generation, sequence)
                            receipt['partial_application'][name] = {
                                'reference': self.arms[name].target_pose.tolist(),
                                'applied_at_ns': time.monotonic_ns()}
                        self.sequence, self.timestamp_ns, self.slot_id = sequence, stamp, slot
                        self._last_target = time.monotonic()
                        self._streaming = True
                        receipt.update({'slot_id': slot, 'applied_at_ns': time.monotonic_ns(),
                                   'references': {n: a.target_pose.tolist() for n, a in self.arms.items()}})
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
            response = {**self.snapshot(), **receipt, 'request_id': key,
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
                    'tcp_frame': TCP_FRAME,
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
                    # Receive a whole frame within the configured communication
                    # deadline. A hard-coded 50 ms wait used to close healthy
                    # clients paused briefly after connect (e.g. by the GIL).
                    deadline = time.monotonic() + self.worker.state_timeout_s
                    stage = 'receive'
                    message = {}
                    try:
                        message = decode_packet(connection, deadline=deadline)
                        stage = 'dispatch'
                        response = self.worker.request(message)
                        stage = 'respond'
                        connection.settimeout(self.worker.state_timeout_s)
                        connection.sendall(encode_packet(response))
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        self.worker.latch_fault(
                            f'IPC {stage} kind={message.get("kind")} request_id={message.get("request_id")}: '
                            f'{type(error).__name__}: {error}')
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
