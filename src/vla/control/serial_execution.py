"""Serial chunk scheduling on the ROS clock, with explicit inference holds.

No prefetch, retiming of the request, RTC prefix, or independent control loop.
An ACK for the final action is required before the next observation is requested.
"""
from dataclasses import dataclass
import math

import numpy as np

from vla.actions.spec import ActionSpec, fingerprint, immutable_array, validate_array
from .execution import Executor, ExecutionRecord, ExecutionState, PlanRequest


@dataclass(frozen=True)
class SerialExecutionContract:
    action_spec: ActionSpec
    control_fingerprint: str
    kinematics_fingerprint: str
    tool_fingerprint: str
    timeout_s: float = 5.0
    version: int = 2
    control_hz: int = 30
    frame_stride: int = 1
    execution_mode: str = 'serial'

    def __post_init__(self):
        if (self.action_spec != ActionSpec() or self.version != 2 or self.control_hz != 30
                or self.frame_stride != 1 or self.execution_mode != 'serial'):
            raise ValueError('serial LingBot requires single-left absolute TCP at 30 Hz')
        if (isinstance(self.timeout_s, bool) or not math.isfinite(self.timeout_s)
                or not 0 < self.timeout_s <= 300):
            raise ValueError('invalid serial inference timeout')
        if not all((self.control_fingerprint, self.kinematics_fingerprint, self.tool_fingerprint)):
            raise ValueError('serial execution requires control/kinematics/tool fingerprints')

    @property
    def horizons(self):
        return {'vla': 50}

    def to_dict(self):
        return {**self.__dict__, 'action_spec': self.action_spec.to_dict(), 'horizons': self.horizons}

    @property
    def digest(self):
        return fingerprint(self.to_dict())


class SerialExecutor(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_action_slot = None
        self._awaiting_final_ack = False

    def invalidate(self, state=ExecutionState.WAITING, reason='control boundary'):
        super().invalidate(state, reason)
        self._last_action_slot = None
        self._awaiting_final_ack = False

    def enable(self, hold, *, episode_id):
        super().enable(hold, episode_id=episode_id)
        self.state = ExecutionState.WAITING_INFERENCE

    def needs_plan(self, slot):
        return (self.state == ExecutionState.WAITING_INFERENCE and self.inflight is None
                and not self.queue and not self._awaiting_final_ack)

    def request(self, *, slot, anchor, observation_id, observed_at_ns, policy):
        if not self.needs_plan(slot):
            return None
        if policy != 'vla':
            raise ValueError('serial deployment supports VLA only')
        self.plan_version += 1
        request = PlanRequest(self.session_id, self.episode_id, self.generation, self.request_id,
                              observation_id, self.plan_version, self.model_version, self.contract.digest,
                              slot, slot, observed_at_ns + int(self.contract.timeout_s*1e9),
                              observed_at_ns, 50, 1, anchor, np.empty((0, 10), dtype=np.float32))
        self.request_id += 1
        self.inflight = request
        self.events.append(('request', request))
        return request

    def install(self, response, now_ns, *, start_slot):
        self.events.append(('candidate', response))
        request, expected = response.request, self.inflight
        reason = None
        if expected is None or request.identity != expected.identity:
            reason = 'stale_or_unrequested'
        elif (request.packet()[0] != expected.packet()[0]
              or not np.array_equal(request.prefix, expected.prefix)
              or not np.array_equal(request.anchor, expected.anchor)):
            reason = 'request_context_mismatch'
        elif self.state != ExecutionState.WAITING_INFERENCE or self.queue or self._awaiting_final_ack:
            reason = 'disabled'
        elif now_ns >= request.deadline_ns:
            reason = 'late'
        if reason:
            self.events.append(('reject', request.identity, reason))
            if expected is not None and request.identity == expected.identity:
                self.invalidate(reason='invalid serial response: ' + reason)
            return False
        try:
            if (response.actions.shape != (50, 10) or response.sources != ('vla',)*50
                    or response.phases != (0,)*50 or response.context
                    or not np.array_equal(response.actions, response.vla_reference)):
                raise ValueError('serial response must contain exactly 50 VLA actions')
            validate_array(response.actions, self.contract.action_spec)
            start = start_slot
            if type(start) is not int or start < 0 or start <= self.last_slot:
                raise ValueError('serial installation requires a fresh control slot')
            projected = [immutable_array(validate_array(self.project(a), self.contract.action_spec))
                         for a in response.actions]
        except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError):
            # Even a caller that catches validation errors cannot later install
            # this request and silently resume. Projection must be all-or-none.
            self.invalidate(reason='invalid serial candidate or projection')
            raise
        for i, target in enumerate(projected):
            self.queue[start+i] = (target, 'vla', 0, request.plan_version, i, response.actions[i])
        self._last_action_slot = start+49
        self._awaiting_final_ack = True
        self.inflight = None
        self.state = ExecutionState.POLICY
        self.events.append(('install_serial', response, {'start_slot': start, 'installed_at_ns': now_ns}))
        return True

    def acknowledge(self, record):
        if (record is not None and record.generation == self.generation and record.source == 'vla'
                and record.slot_id == self._last_action_slot and record.action_index == 49
                and self._awaiting_final_ack):
            self._awaiting_final_ack = False
            self._last_action_slot = None
            self.state = ExecutionState.WAITING_INFERENCE
            self.events.append(('serial_chunk_complete', self.generation, record.plan_version, record.slot_id))

    def advance(self, slot, now_ns):
        if slot == self.last_slot:
            return self.current
        if slot < self.last_slot:
            raise ValueError('out of order execution slot')
        if self.state not in (ExecutionState.POLICY, ExecutionState.WAITING_INFERENCE):
            self.last_slot, self.current = slot, None
            return None
        if self.inflight is not None and now_ns >= self.inflight.deadline_ns:
            self.invalidate(reason='serial inference timed out')
            self.last_slot = slot
            return None
        if self.queue:
            if min(self.queue) != slot:
                self.invalidate(reason='serial action slot missed')
                self.last_slot = slot
                return None
            target, source, phase, plan, index, candidate = self.queue.pop(slot)
            self.hold = target
        elif self.state == ExecutionState.POLICY and not self._awaiting_final_ack:
            self.invalidate(reason='unexpected serial queue exhaustion')
            self.last_slot = slot
            return None
        else:
            target, source, phase, plan, index, candidate = self.hold, 'system_hold', 0, -1, -1, self.hold
        self.last_slot = slot
        self.current = ExecutionRecord(slot, self.generation, plan, index, candidate, target, source, phase, now_ns)
        return self.current

    def execution_context(self, slot, anchor):
        return None  # This deployment does not produce RLT bootstrap contexts.
