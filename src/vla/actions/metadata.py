"""Strict provenance shared by datasets, checkpoints, and replay caches."""
from dataclasses import dataclass
import json
from pathlib import Path
from .spec import ActionSpec, fingerprint
from .statistics import SamplingSpec, validate_statistics


@dataclass(frozen=True)
class RecordingContract:
    action_spec: ActionSpec
    device_fingerprint: str
    tool_fingerprint: str
    kinematics_fingerprint: str
    control_fingerprint: str
    derived_from: str | None = None

    def __post_init__(self):
        if self.action_spec.representation != 'absolute':
            raise ValueError('recordings always store absolute targets')
        if not all((self.device_fingerprint, self.tool_fingerprint,
                    self.kinematics_fingerprint, self.control_fingerprint)):
            raise ValueError('recording requires device, TCP tool, kinematics and control identities')
        if self.derived_from not in (None, 'joint_command_fk', 'tcp_submitted_joints'):
            raise ValueError('unsupported action derivation')

    def to_dict(self):
        return {**self.__dict__, 'action_spec': self.action_spec.to_dict(),
                'version': 1, 'backend': 'ros2_joint_impedance',
                'recording_point': 'accepted_before_tracking' if self.derived_from is None else 'derived'}


def validate_mixture(contracts, *, allow_derived=False):
    if not contracts:
        raise ValueError('no recording contracts')
    reference = dict(contracts[0])
    for contract in contracts:
        if contract.get('derived_from') and not allow_derived:
            raise ValueError('derived actions require explicit training provenance consent')
        a, b = dict(reference), dict(contract)
        if allow_derived:
            for value in (a, b):
                value.pop('derived_from', None)
                value.pop('recording_point', None)
        if a != b:
            raise ValueError('incompatible recording contracts; do not mix control chains')
    return {'allow_derived': allow_derived, 'sources': contracts}


def save_model_contract(path, *, spec, sampling, statistics, data_fingerprint, sources):
    validate_statistics(statistics, spec, sampling, data_fingerprint)
    value = {'version': 1, 'action_spec': spec.to_dict(), 'statistics': statistics,
             'sources': sources}
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    return value


def load_model_contract(path):
    value = json.loads(Path(path).read_text())
    if value.get('version') != 1:
        raise ValueError('missing or unsupported checkpoint action contract')
    spec = ActionSpec.from_dict(value['action_spec'])
    stats = value['statistics']
    sampling = SamplingSpec(**stats['contract']['sampling'])
    validate_statistics(stats, spec, sampling, stats['contract']['data_fingerprint'])
    return spec, sampling, value


def require_model_contract(expected, actual):
    if fingerprint(expected) != fingerprint(actual):
        raise ValueError('checkpoint/replay action contract mismatch; rebuild cache')
