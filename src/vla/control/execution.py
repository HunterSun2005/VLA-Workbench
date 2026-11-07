"""ROS-owned execution timeline. Pure NumPy; no ROS, Torch, I/O or wall clock sleeps.

All mutations run on the control thread. Workers return immutable candidates;
only this owner may install them. Slot numbers never restart at a mode boundary.
"""
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping
import math
import numpy as np

from vla.actions.spec import ActionSpec, immutable_array, validate_array, fingerprint


class ExecutionState(str, Enum):
    STANDBY = 'standby'
    PREPARING = 'preparing'
    HUMAN = 'human'
    POLICY = 'policy'
    WAITING = 'waiting_resume'
    STORAGE = 'storage_blocked'
    FAULT = 'fault'


@dataclass(frozen=True)
class ExecutionContract:
    action_spec: ActionSpec
    horizons: Mapping[str, int]
    control_fingerprint: str
    kinematics_fingerprint: str
    tool_fingerprint: str
    frame_stride: int = 1
    inference_slots: int = 4
    guard_slots: int = 1
    control_hz: int = 30
    tracking_hz: int = 100
    version: int = 2

    def __post_init__(self):
        if self.version != 2 or (self.control_hz, self.tracking_hz) != (30, 100):
            raise ValueError('unsupported execution contract')
        if self.action_spec.representation != 'absolute':
            raise ValueError('executor requires absolute targets')
        if any(type(v) is not int or v < 1 for v in
               (self.frame_stride, self.inference_slots, self.guard_slots)):
            raise ValueError('execution budgets and stride must be positive integers')
        if not all((self.control_fingerprint, self.kinematics_fingerprint, self.tool_fingerprint)):
            raise ValueError('execution requires effective control, URDF and TCP fingerprints')
        horizons = dict(self.horizons)
        if not horizons or any(type(h) is not int or h < 2*self.freeze_steps for h in horizons.values()):
            raise ValueError(f'every policy horizon must satisfy H >= 2d (d={self.freeze_steps})')
        object.__setattr__(self, 'horizons', MappingProxyType(horizons))

    @property
    def freeze_steps(self):
        return math.ceil((self.inference_slots+self.guard_slots)/self.frame_stride)

    def to_dict(self):
        return {**self.__dict__, 'action_spec': self.action_spec.to_dict(), 'horizons': dict(self.horizons)}

    @property
    def digest(self):
        return fingerprint(self.to_dict())


@dataclass(frozen=True)
class PlanRequest:
    session_id: str
    episode_id: str
    generation: int
    request_id: int
    observation_id: int
    plan_version: int
    model_version: str
    contract_digest: str
    start_slot: int
    replace_slot: int
    deadline_ns: int
    observed_at_ns: int
    horizon: int
    frame_stride: int
    anchor: np.ndarray
    prefix: np.ndarray

    def __post_init__(self):
        if not all((self.session_id, self.episode_id, self.model_version, self.contract_digest)):
            raise ValueError('missing plan identity')
        if min(self.generation, self.request_id, self.observation_id, self.plan_version,
               self.start_slot, self.observed_at_ns) < 0:
            raise ValueError('negative plan identity/time')
        if self.frame_stride < 1 or self.horizon < 1 or self.deadline_ns < self.observed_at_ns:
            raise ValueError('invalid request timing')
        d, remainder = divmod(self.replace_slot-self.start_slot, self.frame_stride)
        anchor, prefix = np.asarray(self.anchor), np.asarray(self.prefix)
        if remainder or not 0 <= d < self.horizon or prefix.shape != (d, anchor.size):
            raise ValueError('invalid fixed commitment boundary')
        if anchor.ndim != 1 or not np.isfinite(anchor).all() or not np.isfinite(prefix).all():
            raise ValueError('invalid request anchor/prefix')
        object.__setattr__(self, 'anchor', immutable_array(anchor))
        object.__setattr__(self, 'prefix', immutable_array(prefix))

    @property
    def start_idx(self):
        return len(self.prefix)

    @property
    def identity(self):
        return (self.session_id, self.episode_id, self.generation, self.request_id,
                self.observation_id, self.plan_version, self.model_version, self.contract_digest)

    def packet(self):
        return ({'kind': 'plan', **{k: v for k, v in self.__dict__.items() if k not in ('anchor', 'prefix')}},
                {'observation.anchor': self.anchor, 'committed_prefix': self.prefix})

    @classmethod
    def from_packet(cls, header, arrays):
        keys = cls.__dataclass_fields__.keys() - {'anchor', 'prefix'}
        return cls(**{k: header[k] for k in keys}, anchor=arrays['observation.anchor'],
                   prefix=arrays['committed_prefix'])


@dataclass(frozen=True)
class PlanResponse:
    request: PlanRequest
    actions: np.ndarray
    sources: tuple[str, ...]
    phases: tuple[int, ...]
    vla_reference: np.ndarray
    context: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        shape = (len(self.actions), self.request.anchor.size)
        if not 2*self.request.start_idx <= shape[0] <= self.request.horizon:
            raise ValueError('candidate has no valid suffix or exceeds requested range')
        for name in ('actions', 'vla_reference'):
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'invalid full {name} chunk')
            object.__setattr__(self, name, immutable_array(value))
        if len(self.sources) != shape[0] or len(self.phases) != shape[0]:
            raise ValueError('per-action provenance required')
        if set(self.sources)-{'vla', 'actor'} or set(self.phases)-{0, 1}:
            raise ValueError('invalid candidate provenance')
        object.__setattr__(self, 'sources', tuple(self.sources))
        object.__setattr__(self, 'phases', tuple(self.phases))
        object.__setattr__(self, 'context', MappingProxyType({k: immutable_array(v) for k, v in self.context.items()}))

    def packet(self):
        header, arrays = self.request.packet()
        return ({**header, 'kind': 'plan_result', 'sources': self.sources, 'phases': self.phases},
                {**arrays, 'actions': self.actions, 'vla_reference': self.vla_reference,
                 **{'context.'+k: v for k, v in self.context.items()}})

    @classmethod
    def from_packet(cls, header, arrays):
        return cls(PlanRequest.from_packet(header, arrays), arrays['actions'], tuple(header['sources']),
                   tuple(header['phases']), arrays['vla_reference'],
                   {k[8:]: v for k, v in arrays.items() if k.startswith('context.')})


@dataclass(frozen=True)
class ExecutionRecord:
    slot_id: int
    generation: int
    plan_version: int
    action_index: int
    candidate: np.ndarray
    accepted: np.ndarray
    source: str
    phase: int
    accepted_at_ns: int
    submitted_at_ns: int | None = None
    outcome: str = 'accepted'
    diagnostics: tuple = ()

    def __post_init__(self):
        for name in ('candidate', 'accepted'):
            object.__setattr__(self, name, immutable_array(getattr(self, name)))


class FixedDeadline:
    """Integer deadlines share an epoch; late polls skip, never catch up in bursts."""
    def __init__(self, epoch_ns, hz):
        self.epoch_ns, self.hz, self.last = epoch_ns, hz, -1

    def deadline(self, slot):
        return self.epoch_ns + (slot*1_000_000_000+self.hz-1)//self.hz

    def poll(self, now_ns):
        slot = (now_ns-self.epoch_ns)*self.hz//1_000_000_000
        if slot <= self.last or slot < 0:
            return None
        self.last = slot
        return slot


class Executor:
    def __init__(self, contract, *, session_id, model_version, epoch_ns, project=None):
        self.contract, self.session_id, self.model_version = contract, session_id, model_version
        self.control_clock = FixedDeadline(epoch_ns, contract.control_hz)
        self.tracking_clock = FixedDeadline(epoch_ns, contract.tracking_hz)
        self.state = ExecutionState.STANDBY
        self.generation = self.plan_version = self.request_id = 0
        self.episode_id = ''
        self.queue = {}
        self.inflight = None
        self.locked_until = -1
        self.last_slot = -1
        self.events = []
        self.hold = None
        self.current = None
        self.project = project or (lambda target: target)

    def invalidate(self, state=ExecutionState.WAITING, reason='control boundary'):
        self.generation += 1
        self.queue.clear()
        self.inflight = None
        self.locked_until = -1
        self.current = None
        self.state = ExecutionState(state)
        self.events.append(('invalidate', self.generation, self.last_slot, reason))

    def enable(self, hold, *, episode_id):
        """Explicit user enable/resume. A response can never call this method."""
        self.invalidate(ExecutionState.PREPARING, 'manual enable')
        self.hold = immutable_array(validate_array(hold, self.contract.action_spec))
        if self.hold.shape != (self.contract.action_spec.dim,):
            raise ValueError('hold must be one absolute target')
        self.episode_id = str(episode_id)

    def needs_plan(self, slot):
        if self.inflight is not None or self.state not in (ExecutionState.PREPARING, ExecutionState.POLICY):
            return False
        budget = self.contract.freeze_steps*self.contract.frame_stride
        return not self.queue or max(self.queue)+1-slot <= 2*budget

    def request(self, *, slot, anchor, observation_id, observed_at_ns, policy):
        if not self.needs_plan(slot):
            return None
        r = self.contract.frame_stride
        start = ((slot+r-1)//r)*r
        boundary = max(start+self.contract.freeze_steps*r, self.locked_until)
        d = (boundary-start)//r
        horizon = self.contract.horizons[policy]
        if horizon < 2*d:
            return None  # A locked Actor block is still too long: request closer to its end.
        prefix = []
        for first in range(start, boundary, r):
            entries = [self.queue.get(s) for s in range(first, first+r)]
            if self.state == ExecutionState.PREPARING:
                target = self.hold
            elif any(e is None for e in entries):
                return None
            else:
                target = entries[0][0]
                if any(not np.array_equal(e[0], target) for e in entries):
                    raise ValueError('commitment is not aligned to model action grid')
            prefix.append(target)
        self.plan_version += 1
        request = PlanRequest(self.session_id, self.episode_id, self.generation, self.request_id,
                              observation_id, self.plan_version, self.model_version, self.contract.digest,
                              start, boundary, min(self.control_clock.deadline(boundary-self.contract.guard_slots),
                                  observed_at_ns + self.contract.inference_slots*1_000_000_000//self.contract.control_hz),
                              observed_at_ns, horizon, r, anchor, np.asarray(prefix))
        self.request_id += 1
        # Freeze both values and provenance until the response's immutable boundary.
        if self.state == ExecutionState.PREPARING:
            for s in range(slot, boundary):
                self.queue[s] = (self.hold, 'system_hold', 0, -1, -1, self.hold)
        self.inflight = request
        self.events.append(('request', request))
        return request

    def install(self, response, now_ns):
        self.events.append(('candidate', response))
        request = response.request
        expected = self.inflight
        reason = None
        if expected is None or request.identity != expected.identity:
            reason = 'stale_or_unrequested'
        elif (request.packet()[0] != expected.packet()[0]
              or not np.array_equal(request.prefix, expected.prefix)
              or not np.array_equal(request.anchor, expected.anchor)):
            reason = 'request_context_mismatch'
        elif self.state not in (ExecutionState.PREPARING, ExecutionState.POLICY):
            reason = 'disabled'
        elif now_ns >= request.deadline_ns or self.last_slot >= request.replace_slot:
            reason = 'late'
        if reason:
            if expected is not None and request.identity == expected.identity:
                self.inflight = None
            self.events.append(('reject', request.identity, reason))
            return False
        validate_array(response.actions, self.contract.action_spec)
        d, r = request.start_idx, request.frame_stride
        horizon = len(response.actions)
        if horizon not in self.contract.horizons.values():
            raise ValueError('response horizon does not match a declared policy')
        if request.replace_slot < self.locked_until:
            raise ValueError('cannot replace locked Actor suffix')
        projected = [immutable_array(self.project(action)) for action in response.actions[d:]]
        for s in list(self.queue):
            if s >= request.replace_slot:
                del self.queue[s]
        for i in range(d, horizon):
            accepted = projected[i-d]
            for s in range(request.start_slot+i*r, request.start_slot+(i+1)*r):
                self.queue[s] = (accepted, response.sources[i], response.phases[i],
                                 request.plan_version, i, response.actions[i])
        if 'actor' in response.sources[d:]:
            self.locked_until = request.start_slot+horizon*r
        self.inflight = None
        self.events.append(('install', response))
        return True

    def advance(self, slot, now_ns):
        if slot == self.last_slot:
            return self.current
        if slot < self.last_slot:
            raise ValueError('out of order execution slot')
        self.last_slot = slot
        for old in [s for s in self.queue if s < slot]:
            del self.queue[old]
        self.current = None
        if self.state not in (ExecutionState.PREPARING, ExecutionState.POLICY):
            return None
        entry = self.queue.pop(slot, None)
        if entry is None:
            self.invalidate(ExecutionState.WAITING, 'queue exhausted')
            return None
        action, source, phase, plan, index, candidate = entry
        if source != 'system_hold':
            self.state = ExecutionState.POLICY
        self.current = ExecutionRecord(slot, self.generation, plan, index, candidate, action, source, phase, now_ns)
        return self.current

    def execution_context(self, slot, anchor):
        """Capture the real next-state commitment for later constrained bootstrap."""
        r = self.contract.frame_stride
        H = self.contract.horizons.get('rlt', self.contract.horizons.get('actor'))
        if H is None or slot % r or self.current is None or self.current.slot_id != slot:
            return None
        boundary = max(slot+self.contract.freeze_steps*r, self.locked_until,
                       self.inflight.replace_slot if self.inflight else 0)
        d = (boundary-slot)//r
        if d >= H:
            return None
        prefix = []
        for s in range(slot, boundary, r):
            if s == slot:
                prefix.append(self.current.accepted)
            elif s in self.queue:
                prefix.append(self.queue[s][0])
            else:
                return None
        return {'slot_id': slot, 'generation': self.generation, 'model_version': self.model_version,
                'horizon': H, 'start_idx': d, 'prefix': immutable_array(prefix),
                'anchor': immutable_array(anchor)}
