"""LingBot VLA-only RTC with atomic validation and manual fault recovery."""
from dataclasses import dataclass

import numpy as np

from vla.actions.spec import ActionSpec, immutable_array, validate_array
from vla_rollout_protocol.external import validate_rtc_budget
from .execution import ExecutionContract, Executor, ExecutionState


@dataclass(frozen=True)
class LingBotRTCContract(ExecutionContract):
    inference_slots: int = 14
    guard_slots: int = 1
    rtc_max_delay: int = 15
    execution_mode: str = 'rtc'

    def __post_init__(self):
        super().__post_init__()
        if (self.action_spec != ActionSpec() or dict(self.horizons) != {'vla': 50}
                or self.frame_stride != 1 or self.execution_mode != 'rtc'
                or type(self.rtc_max_delay) is not int or not 0 < self.rtc_max_delay < 50):
            raise ValueError('LingBot RTC requires single-left TCP, VLA horizon 50 and stride 1')
        validate_rtc_budget(self.inference_slots, self.guard_slots, self.rtc_max_delay)


class LingBotRTCExecutor(Executor):
    def invalidate(self, state=ExecutionState.WAITING, reason='control boundary'):
        super().invalidate(state, reason)
        self._last_active_slot = None

    def request(self, **kwargs):
        if kwargs['policy'] != 'vla':
            raise ValueError('LingBot RTC supports VLA only')
        slot = kwargs['slot']
        if self.needs_plan(slot) and self.state == ExecutionState.POLICY:
            if any(s not in self.queue for s in range(slot, slot+self.contract.freeze_steps)):
                self.invalidate(reason='missing committed RTC slot')
                return None
        return super().request(**kwargs)

    def install(self, response, now_ns):
        self.events.append(('candidate', response, {'checked_at_ns': now_ns,
                                                  'network_received_at_ns': response.received_at_ns}))
        expected, request = self.inflight, response.request
        # Old generations must never disturb a newer active request.
        if expected is None or request.identity != expected.identity:
            self.events.append(('reject', request.identity, 'stale_or_unrequested'))
            return False
        reason = None
        if (request.packet()[0] != expected.packet()[0]
                or not np.array_equal(request.prefix, expected.prefix)
                or not np.array_equal(request.anchor, expected.anchor)):
            reason = 'RTC response context mismatch'
        elif self.state not in (ExecutionState.PREPARING, ExecutionState.POLICY):
            reason = 'RTC response received while execution is disabled'
        elif now_ns >= request.deadline_ns:
            reason = (f'late RTC response: request={request.request_id}, '
                      f'elapsed={(now_ns-request.observed_at_ns)/1e6:.1f}ms, '
                      f'budget={(request.deadline_ns-request.observed_at_ns)/1e6:.1f}ms, '
                      f'late_by={(now_ns-request.deadline_ns)/1e6:.1f}ms')
            if response.received_at_ns is not None:
                reason += (f', network_elapsed={(response.received_at_ns-request.observed_at_ns)/1e6:.1f}ms, '
                           f'handoff={(now_ns-response.received_at_ns)/1e6:.1f}ms')
        elif self.last_slot >= request.replace_slot:
            reason = (f'RTC response missed installation boundary: request={request.request_id}, '
                      f'last_slot={self.last_slot}, replace_slot={request.replace_slot}')
        if reason:
            self.events.append(('reject', request.identity, reason))
            self.invalidate(reason=reason)
            return False
        try:
            d = request.start_idx
            if (response.actions.shape != (50, 10) or response.sources != ('vla',)*50
                    or response.phases != (0,)*50 or response.context
                    or not np.array_equal(response.actions, response.vla_reference)
                    or not np.array_equal(response.actions[:d], request.prefix)):
                raise ValueError('invalid LingBot RTC actions, provenance or committed prefix')
            validate_array(response.actions, self.contract.action_spec)
            # Project every suffix before touching the installed queue. The prefix
            # already contains accepted/projected targets and must not be projected twice.
            projected = [immutable_array(validate_array(self.project(a), self.contract.action_spec))
                         for a in response.actions[d:]]
            if any(a.shape != (10,) for a in projected):
                raise ValueError('projection must return one TCP target')
        except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError):
            self.invalidate(reason='invalid RTC candidate or projection')
            raise
        retained = {s: e for s, e in self.queue.items() if s < request.replace_slot}
        for i, target in enumerate(projected, d):
            retained[request.start_slot+i] = (target, 'vla', 0, request.plan_version, i, response.actions[i])
        self.queue = retained
        self.inflight = None
        self.events.append(('install', response, {'installed_at_ns': now_ns,
                                                'replace_slot': request.replace_slot}))
        return True

    def advance(self, slot, now_ns):
        if self.state in (ExecutionState.PREPARING, ExecutionState.POLICY):
            if self.inflight is not None and now_ns >= self.inflight.deadline_ns:
                self.invalidate(reason='RTC inference deadline expired')
            elif self._last_active_slot is not None and slot > self._last_active_slot+1:
                self.invalidate(reason='missed RTC execution slot')
        record = super().advance(slot, now_ns)
        if record is not None:
            self._last_active_slot = slot
        return record
