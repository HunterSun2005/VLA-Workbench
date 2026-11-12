"""Synchronized Cartesian references; advanced only by accepted control slots."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class PoseReference:
    def __init__(self, start_position, start_rotation, goal_position, goal_rotation, *,
                 linear_speed=.10, angular_speed=.45):
        self.start = np.asarray(start_position, dtype=float).copy()
        self.goal = np.asarray(goal_position, dtype=float).copy()
        self.rotation = np.asarray(start_rotation, dtype=float).copy()
        self.goal_rotation = np.asarray(goal_rotation, dtype=float).copy()
        for p in (self.start, self.goal):
            if p.shape != (3,) or not np.isfinite(p).all():
                raise ValueError('invalid reference position')
        from vla.actions import matrix_to_rot6d
        matrix_to_rot6d(self.rotation)
        matrix_to_rot6d(self.goal_rotation)
        if not np.isfinite([linear_speed, angular_speed]).all() or min(linear_speed, angular_speed) <= 0:
            raise ValueError('reference speeds must be finite and positive')
        self.delta = self.goal-self.start
        self.vector = Rotation.from_matrix(self.goal_rotation @ self.rotation.T).as_rotvec()
        # max d(10u^3-15u^4+6u^5)/du = 15/8. Both components
        # share this duration and progress, including when one has zero length.
        self.duration = max(.3, 1.875*np.linalg.norm(self.delta)/linear_speed,
                            1.875*np.linalg.norm(self.vector)/angular_speed)
        self.elapsed = 0.
        self.progress = 0.

    @property
    def done(self):
        return self.elapsed >= self.duration

    def step(self, dt):
        if not np.isfinite(dt) or not 0 < dt <= 1/30+1e-9:
            raise ValueError('reference requires one 30 Hz control slot')
        self.elapsed = min(self.duration, self.elapsed+dt)
        u = self.elapsed/self.duration
        self.progress = u**3*(10+u*(-15+6*u))
        if self.done:
            self.progress = 1.
            return self.goal.copy(), self.goal_rotation.copy()
        return (self.start+self.progress*self.delta,
                Rotation.from_rotvec(self.progress*self.vector).as_matrix() @ self.rotation)
