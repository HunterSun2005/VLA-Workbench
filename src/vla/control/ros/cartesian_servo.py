"""Production Cartesian tracking and KDL IK, importable without ROS nodes or Torch."""
from __future__ import annotations
import numpy as np
import PyKDL
from vla.datacollection.franka_robotiq_rollout.config import RolloutConfig
from vla.datacollection.franka_robotiq_rollout.safety import (
    clamp_pose_step, compose_pose, flange_pose_from_tcp, gamepad_tracking_twist,
    pose_step_to_twist, smooth_pose_target,
)


def _clamp_norm(values, maximum):
    norm = float(np.linalg.norm(values))
    return values if norm <= maximum or norm == 0 else values * (maximum / norm)


class CartesianServo:
    def __init__(self, model, config: RolloutConfig) -> None:
        self.model = model
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(model.chain)
        self.jacobian_solver = PyKDL.ChainJntToJacSolver(model.chain)
        self.config = config
        self._last_smoothed_position: np.ndarray | None = None
        self._last_smoothed_rotation: np.ndarray | None = None
        self._last_linear_velocity = np.zeros(3, dtype=np.float64)
        self._last_angular_velocity = np.zeros(3, dtype=np.float64)

    def reset_tracking(self) -> None:
        self._last_smoothed_position = None
        self._last_smoothed_rotation = None
        self._last_linear_velocity.fill(0.0)
        self._last_angular_velocity.fill(0.0)

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

    def step(
        self,
        joints: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        flange_to_tcp_position: np.ndarray,
        flange_to_tcp_rotation: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        current_flange_position, current_flange_rotation = self.forward(joints)
        current_tcp_position, current_tcp_rotation = compose_pose(
            current_flange_position,
            current_flange_rotation,
            flange_to_tcp_position,
            flange_to_tcp_rotation,
        )
        limited_tcp_position, limited_tcp_rotation = clamp_pose_step(
            current_tcp_position,
            current_tcp_rotation,
            target_position,
            target_rotation,
            max_linear_step=(
                self.config.max_linear_velocity * self.config.velocity_tracking_tau_s
            ),
            max_angular_step=(
                self.config.max_angular_velocity * self.config.velocity_tracking_tau_s
            ),
        )
        if self._last_smoothed_position is not None:
            limited_tcp_position, limited_tcp_rotation = smooth_pose_target(
                self._last_smoothed_position,
                self._last_smoothed_rotation,
                limited_tcp_position,
                limited_tcp_rotation,
                alpha=self.config.target_smoothing_alpha,
            )
        self._last_smoothed_position = limited_tcp_position.copy()
        self._last_smoothed_rotation = limited_tcp_rotation.copy()
        target_flange_position, target_flange_rotation = flange_pose_from_tcp(
            limited_tcp_position,
            limited_tcp_rotation,
            flange_to_tcp_position,
            flange_to_tcp_rotation,
        )
        linear, angular = pose_step_to_twist(
            current_flange_position,
            current_flange_rotation,
            target_flange_position,
            target_flange_rotation,
            tracking_tau_s=self.config.velocity_tracking_tau_s,
        )
        alpha = self.config.velocity_smoothing
        linear = alpha * self._last_linear_velocity + (1.0 - alpha) * linear
        if self.config.smooth_angular_velocity:
            angular = alpha * self._last_angular_velocity + (1.0 - alpha) * angular
        linear = _clamp_norm(linear, self.config.max_linear_velocity)
        angular = _clamp_norm(angular, self.config.max_angular_velocity)
        self._last_linear_velocity = linear.copy()
        self._last_angular_velocity = angular.copy()
        return self._joint_step(joints, np.concatenate((linear, angular)), dt)

    def step_gamepad(
        self,
        joints: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        flange_to_tcp_position: np.ndarray,
        flange_to_tcp_rotation: np.ndarray,
        target_velocity: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Match FineTeleopNode's command-state Cartesian servo."""
        current_flange_position, current_flange_rotation = self.forward(joints)
        target_flange_position, target_flange_rotation = flange_pose_from_tcp(
            target_position,
            target_rotation,
            flange_to_tcp_position,
            flange_to_tcp_rotation,
        )
        linear, angular = gamepad_tracking_twist(
            current_flange_position,
            current_flange_rotation,
            target_flange_position,
            target_flange_rotation,
            target_velocity,
            position_gain=self.config.gamepad_position_gain,
            orientation_gain=self.config.gamepad_orientation_gain,
            max_linear_speed=2.0 * self.config.gamepad_xy_speed,
            max_angular_speed=self.config.gamepad_max_angular_speed,
        )
        return self._joint_step(joints, np.concatenate((linear, angular)), dt)

    def _joint_step(self, joints: np.ndarray, twist: np.ndarray, dt: float) -> np.ndarray:
        jacobian_kdl = PyKDL.Jacobian(7)
        if self.jacobian_solver.JntToJac(self._joints(joints), jacobian_kdl) < 0:
            raise RuntimeError("KDL Jacobian calculation failed")
        jacobian = np.array(
            [[jacobian_kdl[row, column] for column in range(7)] for row in range(6)]
        )
        regularized = jacobian @ jacobian.T + self.config.damping**2 * np.eye(6)
        velocity = jacobian.T @ np.linalg.solve(regularized, twist)
        velocity = np.clip(velocity, -self.config.max_joint_speed, self.config.max_joint_speed)
        return np.clip(
            joints + velocity * dt,
            self.model.lower_limits,
            self.model.upper_limits,
        )

