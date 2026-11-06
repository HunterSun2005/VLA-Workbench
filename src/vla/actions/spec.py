"""Versioned, serializable contracts. This package has no ROS or model dependency."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

import numpy as np

VERSION = 1
ROTATION = 'columns_0_1_base_relative_left_multiply'
ANCHOR = 'chunk_start_raw_measured_named_fields'
TCP_FIELDS = ('tcp_position_x', 'tcp_position_y', 'tcp_position_z',
              'tcp_rot6d_col0_x', 'tcp_rot6d_col0_y', 'tcp_rot6d_col0_z',
              'tcp_rot6d_col1_x', 'tcp_rot6d_col1_y', 'tcp_rot6d_col1_z')
JOINT_FIELDS = tuple(f'joint_position_{i}' for i in range(1, 8))


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ActionSpec:
    space: str = 'tcp'
    arms: tuple[str, ...] = ('left',)
    representation: str = 'absolute'
    version: int = VERSION
    anchor_rule: str = ANCHOR
    rotation: str = ROTATION
    storage: str = 'absolute'

    def __post_init__(self):
        object.__setattr__(self, 'arms', tuple(self.arms))
        if self.space not in ('joint', 'tcp') or self.representation not in ('absolute', 'chunk_relative'):
            raise ValueError('action space must be joint/tcp; representation absolute/chunk_relative')
        if self.arms not in (('left',), ('left', 'right')):
            raise ValueError('arms must be left or left,right, in that order')
        if (self.version, self.anchor_rule, self.rotation, self.storage) != (VERSION, ANCHOR, ROTATION, 'absolute'):
            raise ValueError('unsupported action contract version, anchor, rotation or storage')

    @property
    def width(self):
        return 8 if self.space == 'joint' else 10

    @property
    def dim(self):
        return len(self.arms) * self.width

    @property
    def measured_fields(self):
        fields = (JOINT_FIELDS if self.space == 'joint' else TCP_FIELDS) + ('gripper_position',)
        return tuple((f'{arm}_' if len(self.arms) > 1 else '') + name
                     for arm in self.arms for name in fields)

    @property
    def action_fields(self):
        fields = tuple('desired_' + field for field in
                       (JOINT_FIELDS if self.space == 'joint' else TCP_FIELDS)) + ('gripper',)
        return tuple((f'{arm}_' if len(self.arms) > 1 else '') + name
                     for arm in self.arms for name in fields)

    def to_dict(self):
        return {**asdict(self), 'arms': list(self.arms)}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def require_compatible(self, other, *, training=True):
        a, b = self.to_dict(), other.to_dict()
        if not training:
            a.pop('representation'); b.pop('representation')
        if a != b:
            raise ValueError(f'incompatible action contracts: {a} != {b}')

    def anchor_from_fields(self, measured: Mapping):
        missing = set(self.measured_fields) - set(measured)
        if missing:
            raise ValueError(f'missing raw measured anchor fields: {sorted(missing)}')
        return validate_array(np.stack([measured[name] for name in self.measured_fields], axis=-1), self)


def validate_array(value, spec):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != spec.dim or not np.isfinite(array).all():
        raise ValueError(f'action/anchor must be finite with final dimension {spec.dim}')
    gripper = array[..., spec.width - 1::spec.width]
    if np.any((gripper < 0) | (gripper > 1)):
        raise ValueError('gripper target/measurement must be an absolute opening in [0, 1]')
    return array


def immutable_array(value):
    # Backed by immutable bytes: even setflags(write=True) cannot mutate a request anchor.
    value = np.asarray(value, dtype=np.float64)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class ObservationSnapshot:
    measured: Mapping[str, float]
    observed_at_ns: int

    def __post_init__(self):
        from types import MappingProxyType
        values = {name: float(value) for name, value in self.measured.items()}
        if self.observed_at_ns < 0 or not all(np.isfinite(v) for v in values.values()):
            raise ValueError('invalid observation timestamp or measurement')
        object.__setattr__(self, 'measured', MappingProxyType(values))


@dataclass(frozen=True)
class ChunkContext:
    spec: ActionSpec
    anchor: np.ndarray
    observed_at_ns: int
    generation: int
    request_id: int

    def __post_init__(self):
        value = validate_array(self.anchor, self.spec)
        if value.shape != (self.spec.dim,) or min(self.observed_at_ns, self.generation, self.request_id) < 0:
            raise ValueError('invalid chunk context')
        object.__setattr__(self, 'anchor', immutable_array(value))

    @classmethod
    def capture(cls, spec, observation, generation, request_id):
        return cls(spec, spec.anchor_from_fields(observation.measured),
                   observation.observed_at_ns, generation, request_id)


@dataclass(frozen=True)
class AcceptedCommand:
    spec: ActionSpec
    target: np.ndarray
    source: str
    command_id: int
    generation: int
    observed_at_ns: int
    accepted_at_ns: int

    def __post_init__(self):
        value = validate_array(self.target, self.spec)
        if value.shape != (self.spec.dim,) or self.accepted_at_ns < self.observed_at_ns:
            raise ValueError('invalid command shape or timestamps')
        object.__setattr__(self, 'target', immutable_array(value))
