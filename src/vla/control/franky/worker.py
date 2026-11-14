"""Franky Cartesian velocity worker.

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
from .velocity_control import VelocityControl, tracking_twist, smooth_pose, validate_context


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
    """One exclusive official CartesianVelocityMotion session."""

    def __init__(self, name: str, robot_ip: str, *, gripper_port: str | None = None,
                 velocity_control: VelocityControl | None = None, realtime_config: str = "ignore",
                 workspace_bounds=None, realtime_cpu=None):
        self.name = name
        self.robot_ip = robot_ip
        self.velocity_control = (velocity_control or VelocityControl()).validate()
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
        self.servo_ready = False
        self.target_pose: np.ndarray | None = None
        self.nominal_pose: np.ndarray | None = None
        self.context = None
        self.last_twist = np.zeros(6)
        self.last_servo_ns = None
        if realtime_cpu is not None and (type(realtime_cpu) is not int or realtime_cpu < 0):
            raise ValueError("realtime_cpu must be a nonnegative CPU number")
        self.realtime_cpu = realtime_cpu
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
        affinity = os.sched_getaffinity(0)
        policy, parameter = os.sched_getscheduler(0), os.sched_getparam(0)
        try:
            self.robot = franky.Robot(self.robot_ip, realtime_config=mode)
        finally:
            os.sched_setscheduler(0, policy, parameter)
            os.sched_setaffinity(0, affinity)
        self.robot.relative_dynamics_factor = .1
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
        # Surface a native motion thread that stopped between target packets. The
        # collector polls health independently of command traffic, so an
        # asynchronous Franky exception cannot remain hidden until the next
        # user action.
        self._check_motion()
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
                "commanded_twist": self.last_twist.tolist(),
                "force": force[:3].tolist(), "torque": force[3:].tolist(),
                "generation": self.target_generation, "sequence": self.target_sequence,
                "servo_ready": self.servo_ready,
                "state_valid": True,
                "fault": self.fault}

    def _check_state_time(self, raw):
        if hasattr(raw, 'time'):
            stamp = float(raw.time.to_sec())
            if not math.isfinite(stamp) or (self._robot_stamp is not None and stamp < self._robot_stamp):
                raise RuntimeError('invalid or regressing Franky measured state timestamp')
            if stamp != self._robot_stamp:
                self._robot_stamp, self._robot_stamp_at = stamp, time.monotonic()
            elif time.monotonic() - self._robot_stamp_at > self.state_timeout_s:
                raise RuntimeError('Franky measured state timestamp stalled')

    def _check_motion(self):
        if self.fault:
            raise RuntimeError(self.fault)
        if self.target_pose is not None:
            self.robot.poll_motion()
            if not self.robot.is_in_control:
                raise RuntimeError('Cartesian velocity motion stopped unexpectedly')

    def set_target(self, pose, generation, sequence, context):
        if not self.servo_ready or self.fault:
            raise RuntimeError('velocity servo is not armed; explicit reset required')
        from .ipc import _validate_pose
        target = np.asarray(_validate_pose(pose), dtype=float)
        target[3:] /= np.linalg.norm(target[3:])
        context = validate_context(context)
        if self.workspace_bounds is not None and np.any((target[:3] < self.workspace_bounds[0]-1e-9) | (target[:3] > self.workspace_bounds[1]+1e-9)):
            raise ValueError('absolute target outside workspace')
        previous = self.target_pose
        if previous is not None and self.velocity_control.target_smoothing_enabled:
            target = smooth_pose(previous, target, self.velocity_control.target_smoothing_alpha)
        self.nominal_pose = np.asarray(pose, dtype=float).copy()
        self.target_pose = target
        self.context = context
        self.target_generation, self.target_sequence = generation, sequence
        self.first_receipt = None
        return None

    def _move(self, motion):
        # Native control threads inherit this thread's scheduling and affinity.
        # Restore both even if native startup fails; never leak RT policy to IPC.
        affinity = os.sched_getaffinity(0)
        policy, parameter = os.sched_getscheduler(0), os.sched_getparam(0)
        try:
            if self.realtime_cpu is not None:
                if self.realtime_cpu not in affinity:
                    raise RuntimeError('configured realtime CPU is unavailable')
                os.sched_setaffinity(0, {self.realtime_cpu})
            if self.realtime_config == 'enforce':
                os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(20))
            self.robot.move(motion, asynchronous=True, limit_rate=False, cutoff_frequency=100.)
        finally:
            os.sched_setscheduler(0, policy, parameter)
            os.sched_setaffinity(0, affinity)

    def servo_once(self):
        if not self.servo_ready or self.target_pose is None:
            return None
        if self.fault:
            raise RuntimeError(self.fault)
        self.robot.poll_motion()
        raw = self.robot.state
        self._check_state_time(raw)
        frame = self._frame(raw)
        measured = frame.to_logical(_affine_pose(raw.O_T_EE))
        observed = time.monotonic_ns()
        twist = tracking_twist(measured, self.target_pose, self.velocity_control)
        if self.velocity_control.velocity_smoothing_enabled:
            dt = (observed-self.last_servo_ns)/1e9 if self.last_servo_ns is not None else 1/30
            alpha = self.velocity_control.velocity_smoothing_alpha ** (30*min(.1, max(0., dt)))
            twist = alpha*self.last_twist + (1-alpha)*twist
        if self.workspace_bounds is not None:
            # The 100 ms native motion must not carry a command across a bound.
            projected = np.clip(measured[:3]+twist[:3]*.1, *self.workspace_bounds)
            twist[:3] = (projected-measured[:3])/.1
            if np.any(measured[:3] < self.workspace_bounds[0]-1e-6) or np.any(measured[:3] > self.workspace_bounds[1]+1e-6):
                raise RuntimeError('measured TCP outside workspace')
        if not np.isfinite(twist).all():
            raise RuntimeError('non-finite derived Cartesian velocity')
        motion = self.franky.CartesianVelocityMotion(
            self.franky.RobotVelocity(self.franky.Twist(twist[:3], twist[3:])),
            self.franky.Duration(100), self.velocity_control.dynamics_factor)
        self._move(motion)
        applied = time.monotonic_ns()
        self.last_twist, self.last_servo_ns = twist, observed
        event = {'arm': self.name, 'generation': self.target_generation,
                'sequence': self.target_sequence, 'source': self.context['source'], 'observed_at_ns': observed,
                'applied_at_ns': applied, 'measured_tcp': measured.tolist(),
                'absolute_target': self.nominal_pose.tolist(), 'reference_tcp': self.target_pose.tolist(),
                'twist': twist.tolist()}
        if self.first_receipt is None:
            self.first_receipt = event
        return event

    def reset(self, generation):
        self.stop()
        if self.fault:
            raise RuntimeError(self.fault)
        raw = self.robot.state
        self._check_state_time(raw)
        self._frame(raw)
        self.target_generation, self.target_sequence = generation, -1
        self.servo_ready = True

    def stop(self, *, recover=False):
        self.servo_ready = False
        try:
            if self.robot is not None and self.robot.is_in_control:
                try:
                    self._move(self.franky.CartesianVelocityStopMotion())
                    if self.robot.join_motion(1.) is False:
                        raise TimeoutError('velocity stop timed out')
                except Exception:
                    self.robot.stop()
                    if self.robot.join_motion(1.) is False:
                        raise TimeoutError('native velocity thread did not stop')
                    raise
        finally:
            self.target_pose = self.nominal_pose = self.context = None
            self.last_twist = np.zeros(6)
            self.last_servo_ns = None
        if recover and self.robot is not None:
            self.robot.recover_from_errors()

    def recover(self):
        self.stop()
        self.robot.recover_from_errors()
        self.fault = None
        self._robot_stamp = None

    def update_velocity_control(self, values):
        config = VelocityControl.from_mapping(values, self.velocity_control)
        # No retained target may resume because a parameter update succeeded.
        self.stop()
        self.velocity_control = config
        self.servo_ready = True
        return config

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
    IMPLEMENTATION_VERSION = 9

    def __init__(self, arms, *, session_id, velocity_control=None, target_timeout_s=.1,
                 state_timeout_s=.5):
        if not session_id or not arms:
            raise ValueError('worker requires a session and at least one arm')
        self.session_id = str(session_id)
        base = VelocityControl.from_mapping(velocity_control)
        self.arms = {name: FrankyArm(name, str(value['robot_ip']),
                                    velocity_control=VelocityControl.from_mapping({}, base),
                                    workspace_bounds=value.get('workspace_bounds'),
                                    realtime_cpu=value.get('realtime_cpu'),
                                    realtime_config=value.get('realtime_config', 'ignore'))
                     for name, value in arms.items()}
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
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
        self._audit = []
        self._audit_sequence = 0
        self._closed = threading.Event()
        self._watchdog = None
        self._velocity_control = asdict(base)

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
            self._condition.notify_all()
            if first_fault:
                print(f'Franky worker fault session={self.session_id} generation={self.generation} '
                      f'sequence={self.sequence}: {self.fault}', file=sys.stderr, flush=True)

    def _watch(self):
        deadline = time.monotonic() + .01
        while not self._closed.wait(max(0., deadline-time.monotonic())):
            now = time.monotonic()
            deadline += max(1, int((now-deadline)/.01)+1)*.01
            with self._lock:
                if self.fault:
                    continue
                now = time.monotonic()
                if self._connected and now - self._last_contact > self.state_timeout_s:
                    self.latch_fault('Franky client heartbeat expired')
                elif self._streaming and now - self._last_target > self.target_timeout_s:
                    self.latch_fault('Franky target stream expired')
                elif self._streaming:
                    try:
                        if len(self._audit) + len(self.arms) > 4096:
                            raise RuntimeError('velocity audit buffer full; manual recovery required')
                        for arm in self.arms.values():
                            self._record_servo(arm.servo_once())
                        self._condition.notify_all()
                    except Exception as error:
                        self.latch_fault(f'velocity servo: {error}')

    def _record_servo(self, event):
        if event is None:
            return
        if len(self._audit) >= 4096:
            raise RuntimeError('velocity audit buffer full; manual recovery required')
        self._audit_sequence += 1
        self._audit.append(dict(event, audit_id=self._audit_sequence, slot_id=self.slot_id))

    def request(self, message):
        with self._lock:
            if message.get('session_id') != self.session_id:
                raise ValueError('wrong Franky worker session')
            ack = message.get('audit_ack', 0)
            if type(ack) is not int or not 0 <= ack <= self._audit_sequence:
                raise ValueError('invalid velocity audit acknowledgement')
            self._audit = [event for event in self._audit if event['audit_id'] > ack]
            kind = message.get('kind')
            request_id = message.get('request_id')
            generation = message.get('generation')
            if (not isinstance(request_id, (str, int)) or isinstance(request_id, bool)
                    or not str(request_id) or len(str(request_id)) > 128
                    or type(generation) is not int or generation < -1):
                raise ValueError('invalid worker request id or generation')
            key = str(request_id)
            signature = json.dumps({key:value for key,value in message.items() if key != 'audit_ack'}, sort_keys=True)
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
                elif kind not in ('target', 'reset', 'stop', 'recover', 'velocity_control'):
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
                        arm.velocity_control = VelocityControl.from_mapping(self._velocity_control)
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
                        self._armed, self._streaming = True, False
                        self._last_target = time.monotonic()
                elif generation != self.generation:
                    raise ValueError('target/velocity_control generation has not been reset')
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
                        contexts = {name: validate_context(targets[name].get('context')) for name in poses}
                        if len(self._audit) + len(self.arms) > 4096:
                            raise RuntimeError('velocity audit buffer full; manual recovery required')
                        self.slot_id = slot
                        for name, pose in poses.items():
                            self.arms[name].set_target(pose, generation, sequence, contexts[name])
                        self.sequence, self.timestamp_ns, self.slot_id = sequence, stamp, slot
                        self._last_target = time.monotonic()
                        self._streaming = True
                        deadline = stamp/1e9 + self.target_timeout_s
                        while not self.fault and any(arm.first_receipt is None for arm in self.arms.values()):
                            remaining = deadline-time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError('first velocity submission acknowledgement expired')
                            self._condition.wait(remaining)
                        if self.fault:
                            raise RuntimeError(self.fault)
                        receipt.update({'slot_id': slot,
                            'applied_at_ns': max(arm.first_receipt['applied_at_ns'] for arm in self.arms.values()),
                            'references': {n: arm.target_pose.tolist() for n, arm in self.arms.items()},
                            'partial_application': {n: arm.first_receipt for n, arm in self.arms.items()}})
                elif kind == 'velocity_control':
                    values = message.get('values', {})
                    next_value = VelocityControl.from_mapping(values, VelocityControl.from_mapping(self._velocity_control))
                    for arm in self.arms.values():
                        arm.update_velocity_control(values)
                    self._velocity_control = asdict(next_value)  # Commit only after all arms succeed.
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
                    'velocity_control': dict(self._velocity_control), 'servo_events': self._audit[:128]}

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
