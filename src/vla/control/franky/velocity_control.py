"""Absolute TCP feedback math; no SDK, device access, or command queue."""
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.spatial.transform import Rotation


def limit_norm(value, maximum):
    value = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(value))
    return value * min(1., maximum / norm) if norm else value.copy()


@dataclass
class VelocityControl:
    max_linear_speed: float = .05
    max_angular_speed: float = .20
    tracking_tau_s: float = .15
    dynamics_factor: float = .10
    target_smoothing_enabled: bool = False
    velocity_smoothing_enabled: bool = False
    target_smoothing_alpha: float = .25
    velocity_smoothing_alpha: float = .30

    def validate(self):
        for key, value in asdict(self).items():
            if key.endswith('_enabled'):
                if type(value) is not bool:
                    raise ValueError(f'{key} must be boolean')
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{key} must be a finite number')
        for key, cap in (('max_linear_speed', .16), ('max_angular_speed', .45),
                         ('tracking_tau_s', 1.), ('dynamics_factor', 1.)):
            if not 0 < getattr(self, key) <= cap:
                raise ValueError(f'{key} must be in (0, {cap}]')
        for key in ('target_smoothing_alpha', 'velocity_smoothing_alpha'):
            if not 0 <= getattr(self, key) < 1:
                raise ValueError(f'{key} must be in [0, 1)')
        return self

    @classmethod
    def from_mapping(cls, values=None, current=None):
        if values is not None and not isinstance(values, dict):
            raise ValueError('velocity_control must be an object')
        result = asdict(current or cls())
        unknown = set(values or {}) - set(result)
        if unknown:
            raise ValueError('unknown Franky velocity_control parameter: ' + ', '.join(sorted(unknown)))
        result.update(values or {})
        return cls(**result).validate()


def pose_error(measured, target):
    return (target[:3] - measured[:3],
            (Rotation.from_quat(target[3:]) * Rotation.from_quat(measured[3:]).inv()).as_rotvec())


def smooth_pose(previous, target, alpha):
    result = target.copy()
    position, rotation = pose_error(previous, target)
    result[:3] = previous[:3] + (1-alpha) * position
    result[3:] = (Rotation.from_rotvec((1-alpha)*rotation) * Rotation.from_quat(previous[3:])).as_quat()
    return result


def tracking_twist(measured, target, config):
    """One execution law for every absolute TCP target, independent of source."""
    position, rotation = pose_error(measured, target)
    linear = limit_norm(position, config.max_linear_speed * config.tracking_tau_s) / config.tracking_tau_s
    angular = limit_norm(rotation, config.max_angular_speed * config.tracking_tau_s) / config.tracking_tau_s
    if np.linalg.norm(position) <= .0005:
        linear[:] = 0
    if np.linalg.norm(rotation) <= .01:
        angular[:] = 0
    return np.r_[linear, angular]


def validate_context(value):
    if not isinstance(value, dict) or value.get('source') not in ('policy', 'gamepad', 'gello', 'pose'):
        raise ValueError('absolute target requires a valid source context')
    result = dict(value)
    if set(result) != {'source', 'dt'}:
        raise ValueError('absolute target context accepts only source and dt; input masks belong in the frontend')
    dt = result.get('dt')
    if isinstance(dt, bool) or not isinstance(dt, (float, int)) or not math.isfinite(dt) or not 0 < dt <= 1/30 + 1e-6:
        raise ValueError('invalid absolute target dt')
    return result
