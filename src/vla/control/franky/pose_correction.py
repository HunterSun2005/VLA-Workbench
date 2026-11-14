"""Bounded, opt-in static error correction on accepted target packets only.

This is not another execution clock. The worker calls it once per accepted
30 Hz target; health queries and the 1000 Hz impedance loop never advance it.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def limit_norm(vector, maximum):
    norm = float(np.linalg.norm(vector))
    return vector * min(1., maximum / norm) if norm else vector.copy()


class PoseErrorCompensator:
    SETTLE_S = .5

    def __init__(self):
        self.reset()

    def reset(self):
        self.offset = np.zeros(6)
        self.previous = None
        self.settled_s = 0.
        self.status = 'disabled'

    @staticmethod
    def apply(target, offset):
        result = target.copy()
        result[:3] += offset[:3]
        result[3:] = (Rotation.from_rotvec(offset[3:]) * Rotation.from_quat(target[3:])).as_quat()
        return result

    def update(self, target, measured, dt, config):
        target, measured = np.asarray(target, dtype=float), np.asarray(measured, dtype=float)
        if (target.shape != (7,) or measured.shape != (7,)
                or not np.isfinite(target).all() or not np.isfinite(measured).all()
                or np.linalg.norm(target[3:]) < 1e-9 or np.linalg.norm(measured[3:]) < 1e-9
                or not np.isfinite(dt) or not 0 <= dt <= 1/30+1e-9):
            raise ValueError('invalid pose correction state/period')
        if not config.tracking_integral_gain and not np.any(self.offset):
            self.reset()
            return target.copy()
        changed = (self.previous is None or np.linalg.norm(target[:3]-self.previous[:3]) > 1e-9
                   or (Rotation.from_quat(target[3:]) * Rotation.from_quat(self.previous[3:]).inv()).magnitude() > 1e-9)
        self.previous = target.copy()
        self.settled_s = 0. if changed else self.settled_s + dt
        position_error = target[:3]-measured[:3]
        rotation_error = (Rotation.from_quat(target[3:]) * Rotation.from_quat(measured[3:]).inv()).as_rotvec()
        captured = np.linalg.norm(position_error) <= .03 and np.linalg.norm(rotation_error) <= .35
        if not config.tracking_integral_gain or changed or not captured or self.settled_s < self.SETTLE_S:
            # Unwind rather than carrying a learnt static offset into a new path.
            self.offset[:3] -= limit_norm(self.offset[:3], config.tracking_max_linear_rate * dt)
            self.offset[3:] -= limit_norm(self.offset[3:], config.tracking_max_angular_rate * dt)
            self.status = 'unwinding' if np.any(self.offset) else 'settling'
            if not captured:
                self.settled_s = 0.
                self.status = 'outside_capture_range'
            return self.apply(target, self.offset) if np.any(self.offset) else target.copy()

        current = self.apply(target, self.offset)
        candidate = self.offset.copy()
        saturated = False
        for part, error, rate, bound, deadband in (
                (slice(0, 3), position_error, config.tracking_max_linear_rate,
                 config.tracking_max_offset_m, .0005),
                (slice(3, 6), rotation_error, config.tracking_max_angular_rate,
                 config.tracking_max_offset_rad, np.deg2rad(.3))):
            if np.linalg.norm(error) <= deadband:
                continue
            proposed = candidate[part] + limit_norm(config.tracking_integral_gain * error * dt, rate * dt)
            limited = limit_norm(proposed, bound)
            saturated |= np.linalg.norm(proposed-limited) > 1e-12
            candidate[part] = limited

        proposed_pose = self.apply(target, candidate)
        # Anti-windup: do not push further into the inner loop's component clips.
        for part, old_error, new_error, clip in (
                (slice(0, 3), current[:3]-measured[:3], proposed_pose[:3]-measured[:3], config.translational_clip),
                (slice(3, 6), self._quaternion_error(current, measured),
                 self._quaternion_error(proposed_pose, measured), config.rotational_clip)):
            if np.any((np.abs(new_error) > clip) & (np.abs(new_error) > np.abs(old_error)+1e-12)):
                candidate[part] = self.offset[part]
                saturated = True
        self.offset = candidate
        self.status = 'saturated' if saturated else 'correcting'
        return self.apply(target, self.offset)

    @staticmethod
    def _quaternion_error(target, measured):
        # Native Franky clips the quaternion vector part expressed in base axes,
        # not the rotation vector (whose magnitude is the full angle).
        relative = Rotation.from_quat(target[3:]) * Rotation.from_quat(measured[3:]).inv()
        quat = relative.as_quat()
        return quat[:3] * (1. if quat[3] >= 0 else -1.)
