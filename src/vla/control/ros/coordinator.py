"""Single command owner. ROS adapters supply measured state and publishing callbacks."""
from dataclasses import dataclass
from threading import RLock
import time
import numpy as np
from vla.actions import ActionSpec, AcceptedCommand, matrix_to_rot6d, rot6d_to_matrix
from vla.actions.spec import immutable_array, validate_array


@dataclass(frozen=True)
class ArmMeasurement:
    joints: np.ndarray
    gripper: float
    tool_position: np.ndarray
    tool_rotation: np.ndarray
    observed_at_ns: int

    def __post_init__(self):
        for name, shape in [('joints', (7,)), ('tool_position', (3,)), ('tool_rotation', (3, 3))]:
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'invalid measured {name}')
            object.__setattr__(self, name, immutable_array(value))
        matrix_to_rot6d(self.tool_rotation)
        if not 0 <= self.gripper <= 1 or self.observed_at_ns < 0:
            raise ValueError('invalid measured gripper or timestamp')


class ControlCoordinator:
    """30 Hz accepted actions, 100 Hz tracking; safety transitions bypass both clocks.

    publish receives absolute joint targets. reset requests a measured-position
    reset in each controller's real-time loop. A transition completes only after
    all controllers acknowledge its generation. No model or ROS imports here.
    """
    def __init__(self, spec, servos, bounds, publish, reset, *, state_timeout_s=.5,
                 command_timeout_s=.5, reset_timeout_s=1., clock=time.monotonic_ns):
        if spec.representation != 'absolute':
            raise ValueError('ROS accepts only absolute targets')
        self.spec, self.servos, self.bounds = spec, servos, bounds
        self.publish, self.request_reset, self.clock = publish, reset, clock
        self.state_timeout_ns = int(state_timeout_s * 1e9)
        self.command_timeout_ns = int(command_timeout_s * 1e9)
        self.reset_timeout_ns = int(reset_timeout_s * 1e9)
        self.generation = clock()
        self.command_id = 0
        self.owner = None
        self.mode = 'standby'
        self.reason = None
        self.accepted = None
        self.submitted = {}
        self.diagnostics = {}
        self._measurements = {}
        self._acknowledged = set()
        self._last_accept = None
        self._last_slot = -1
        self.partial_submission = {}
        self._pending_owner = None
        self._lock = RLock()

    def observe(self, arm, measurement):
        with self._lock:
            if arm not in self.spec.arms:
                raise ValueError('unknown arm')
            previous = self._measurements.get(arm)
            if previous is not None and measurement.observed_at_ns < previous.observed_at_ns:
                raise ValueError('out of order observation')
            self._measurements[arm] = measurement

    def _fresh(self):
        now = self.clock()
        if set(self._measurements) != set(self.spec.arms) or any(
                not 0 <= now - m.observed_at_ns <= self.state_timeout_ns for m in self._measurements.values()):
            raise RuntimeError('missing or stale robot measurement')

    def transition(self, owner=None, *, reason='control boundary', fault=False):
        with self._lock:
            self.generation += 1
            self.command_id = 0
            self.accepted = None
            self.submitted = {}
            self._last_accept = None
            self._last_slot = -1
            self._acknowledged.clear()
            self.owner = None
            self._pending_owner = None if fault else owner
            self.mode = 'fault' if fault else 'resetting'
            self.reason = reason
            self._reset_started = self.clock()
            for servo in self.servos.values():
                servo.reset_tracking()
            try:
                self.request_reset(self.generation)
            except Exception as error:
                self.mode = 'fault'
                self._pending_owner = None
                self.reason = f'reset submission failed: {error}'
                raise
            return self.generation

    def acknowledge(self, arm, generation, command_id, filtered_joints, received_at_ns):
        with self._lock:
            joints = np.asarray(filtered_joints, dtype=float)
            if arm not in self.spec.arms or joints.shape != (7,) or not np.isfinite(joints).all():
                raise ValueError('invalid controller acknowledgement')
            if generation != self.generation:
                return False
            self.diagnostics[arm] = {'generation': generation, 'command_id': command_id,
                                     'filtered_joints': joints.copy(), 'received_at_ns': received_at_ns}
            if command_id == 0:
                self._acknowledged.add(arm)
            if self.mode == 'resetting' and self._acknowledged == set(self.spec.arms):
                try:
                    self._fresh()
                except RuntimeError as error:
                    self.mode, self.reason = 'fault', str(error)
                    self._pending_owner = None
                    return False
                self.owner = self._pending_owner
                self.mode = 'active' if self.owner else 'standby'
            return True

    def accept(self, target, *, source, generation, slot_id):
        with self._lock:
            if self.mode != 'active' or source != self.owner or generation != self.generation:
                raise RuntimeError('inactive or revoked command owner/generation')
            self._fresh()
            now = self.clock()
            if type(slot_id) is not int or slot_id < self._last_slot:
                raise ValueError('expired or out of order slot id')
            if slot_id == self._last_slot:
                return self.accepted
            command_id = slot_id + 1
            target = validate_array(target, self.spec).copy()
            if target.shape != (self.spec.dim,):
                raise ValueError('ROS accepts one target at a time')
            for i, arm in enumerate(self.spec.arms):
                off = i * self.spec.width
                if self.spec.space == 'joint':
                    model = self.servos[arm].model
                    target[off:off+7] = np.clip(target[off:off+7], model.lower_limits, model.upper_limits)
                else:
                    target[off:off+3] = np.clip(target[off:off+3], *self.bounds[arm])
                    target[off+3:off+9] = matrix_to_rot6d(rot6d_to_matrix(target[off+3:off+9]))
            self.accepted = AcceptedCommand(self.spec, target, source, command_id, generation,
                                            min(m.observed_at_ns for m in self._measurements.values()), now)
            self.command_id, self._last_accept = command_id, now
            self._last_slot = slot_id
            return self.accepted

    def gello_target(self, joints, grippers):
        parts = []
        for arm in self.spec.arms:
            q = np.asarray(joints[arm])
            if self.spec.space == 'joint':
                parts.append(np.r_[q, grippers[arm]])
            else:
                m = self._measurements[arm]
                p, r = self.servos[arm].forward_tcp(q, m.tool_position, m.tool_rotation)
                parts.append(np.r_[p, matrix_to_rot6d(r), grippers[arm]])
        return np.concatenate(parts)

    def gamepad_target(self, tcp_targets, dt):
        """Cartesian input is IK-derived only when the selected action space is joint."""
        parts = []
        for arm in self.spec.arms:
            p, r, gripper = tcp_targets[arm]
            if self.spec.space == 'joint':
                m = self._measurements[arm]
                q = self.servos[arm].step(m.joints, p, r, m.tool_position, m.tool_rotation, dt)
                parts.append(np.r_[q, gripper])
            else:
                parts.append(np.r_[p, matrix_to_rot6d(r), gripper])
        return np.concatenate(parts)

    def tick(self, dt=.01):
        with self._lock:
            now = self.clock()
            if self.mode == 'resetting' and now-self._reset_started > self.reset_timeout_ns:
                self.mode, self.reason = 'fault', 'controller reset acknowledgement timed out'
                self._pending_owner = None
                raise RuntimeError(self.reason)
            if self.mode != 'active':
                return None
            try:
                self._fresh()
                command = self.accepted
                if command is None:
                    return None
                if now-command.accepted_at_ns > self.command_timeout_ns:
                    raise RuntimeError('accepted command stream expired')
                if not 0 < dt <= .05:
                    raise ValueError('invalid tracker period')
                submitted = {}
                for i, arm in enumerate(self.spec.arms):
                    block = command.target[i*self.spec.width:(i+1)*self.spec.width]
                    m = self._measurements[arm]
                    if self.spec.space == 'joint':
                        q = block[:7].copy()
                    else:
                        q = self.servos[arm].step(m.joints, block[:3], rot6d_to_matrix(block[3:9]),
                                                  m.tool_position, m.tool_rotation, dt)
                    p, r = self.servos[arm].forward_tcp(q, m.tool_position, m.tool_rotation)
                    submitted[arm] = {'joints': q.copy(), 'fk_tcp': np.r_[p, matrix_to_rot6d(r)],
                                      'gripper': float(block[-1])}
                # Validate every arm before the first publish; a partial publish faults all arms.
                self.partial_submission = {}
                for arm, target in submitted.items():
                    self.publish(arm, target['joints'], target['gripper'], command.generation, command.command_id)
                    self.partial_submission[arm] = target
                self.submitted = submitted
                return command
            except Exception as error:
                self.transition(reason=str(error), fault=True)
                raise
