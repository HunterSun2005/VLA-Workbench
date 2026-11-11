"""Forward-only TCP kinematics for GELLO; no IK or command integration."""
from __future__ import annotations
import numpy as np
import PyKDL
from vla.datacollection.franka_robotiq_rollout.safety import compose_pose


class ForwardKinematics:
    def __init__(self, model):
        self.model = model
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(model.chain)

    def reset_tracking(self):
        pass  # Stateless; shared coordinator also supports isolated legacy tests.

    def _joints(self, values: np.ndarray) -> PyKDL.JntArray:
        joints = PyKDL.JntArray(7)
        for index, value in enumerate(values):
            joints[index] = float(value)
        return joints

    def forward(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the base-to-flange pose for the configured KDL chain."""
        frame = PyKDL.Frame()
        if self.fk_solver.JntToCart(self._joints(joints), frame) < 0:
            raise RuntimeError("KDL forward kinematics failed")
        position = np.array([frame.p.x(), frame.p.y(), frame.p.z()])
        rotation = np.array([[frame.M[row, col] for col in range(3)] for row in range(3)])
        return position, rotation

    def forward_tcp(
        self,
        joints: np.ndarray,
        flange_to_tcp_position: np.ndarray,
        flange_to_tcp_rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        flange_position, flange_rotation = self.forward(joints)
        return compose_pose(
            flange_position,
            flange_rotation,
            flange_to_tcp_position,
            flange_to_tcp_rotation,
        )

