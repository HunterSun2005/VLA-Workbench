"""Historical ROS/IK validation only; never used by Franky sessions."""
from __future__ import annotations
import numpy as np
import PyKDL
from scipy.spatial.transform import Rotation
from dataclasses import dataclass


@dataclass(frozen=True)
class LegacyServoConfig:
    """Fixed historical harness parameters, isolated from session configuration."""
    fps: int = 30
    damping: float = .05
    max_joint_speed: float = .3
    velocity_tracking_tau_s: float = .15
    target_smoothing_alpha: float = .25
    velocity_smoothing: float = .3
    smooth_angular_velocity: bool = True
    gamepad_position_gain: float = 8.
    gamepad_orientation_gain: float = 3.
    gamepad_xy_speed: float = .02
    gamepad_max_angular_speed: float = .3
    max_linear_velocity: float = .04
    max_angular_velocity: float = .3

from vla.datacollection.franka_robotiq_rollout.safety import (
    clamp_pose_step, compose_pose, flange_pose_from_tcp, gamepad_tracking_twist,
    pose_step_to_twist, smooth_pose_target,
)

CARTESIAN_TRACKING = 'tcp_reference_coordinated_limits_v2'


def _clamp_norm(values, maximum):
    norm = float(np.linalg.norm(values))
    return values if norm <= maximum or norm == 0 else values * (maximum / norm)


class CartesianServo:
    def __init__(self, model, config: LegacyServoConfig) -> None:
        self.model = model
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(model.chain)
        self.jacobian_solver = PyKDL.ChainJntToJacSolver(model.chain)
        self.config = config
        self._last_smoothed_position: np.ndarray | None = None
        self._last_smoothed_rotation: np.ndarray | None = None
        self._last_linear_velocity = np.zeros(3, dtype=np.float64)
        self._last_angular_velocity = np.zeros(3, dtype=np.float64)
        self._command_joints: np.ndarray | None = None

    def reset_tracking(self) -> None:
        self._command_joints = None
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
        if not 0 < dt <= .05:
            raise ValueError('invalid Cartesian tracker period')
        # Keep the trajectory state between ticks. Re-seeding IK from measured
        # joints every tick makes the impedance error only velocity * dt, so
        # a slow/stationary arm never receives the requested position target.
        command_joints = joints if self._command_joints is None else self._command_joints
        following_limit = self.config.max_joint_speed * self.config.velocity_tracking_tau_s
        # Saturation and fault detection must not share an exact boundary: a
        # fresh sample can move slightly while a target sits at the soft limit.
        # Allow one 30 Hz action interval of joint travel as a guard band; this
        # is not additional commanded travel. Larger errors still latch fault.
        fault_limit = following_limit + self.config.max_joint_speed / self.config.fps
        error = np.abs(command_joints - joints)
        if np.any(error > fault_limit + 1e-9):
            index = int(np.argmax(error))
            raise RuntimeError(
                f'Cartesian joint tracking error: {self.model.joint_names[index]} '
                f'error={error[index]:.6f} rad exceeds {fault_limit:.6f} rad '
                f'(measured={joints[index]:.6f}, target={command_joints[index]:.6f})')
        if (np.any(command_joints < self.model.lower_limits - 1e-12)
                or np.any(command_joints > self.model.upper_limits + 1e-12)):
            raise RuntimeError('measured joints outside Cartesian tracking envelope')
        measured_flange_position, measured_flange_rotation = self.forward(joints)
        current_tcp_position, current_tcp_rotation = compose_pose(
            measured_flange_position,
            measured_flange_rotation,
            flange_to_tcp_position,
            flange_to_tcp_rotation,
        )
        current_flange_position, current_flange_rotation = self.forward(command_joints)
        command_tcp_position, command_tcp_rotation = compose_pose(
            current_flange_position, current_flange_rotation,
            flange_to_tcp_position, flange_to_tcp_rotation)
        limited_tcp_position, limited_tcp_rotation = clamp_pose_step(
            command_tcp_position,
            command_tcp_rotation,
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
        linear, angular = pose_step_to_twist(
            command_tcp_position,
            command_tcp_rotation,
            limited_tcp_position,
            limited_tcp_rotation,
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
        # Hold a previously submitted target inside the guard band rather than
        # snapping it to a moving measurement. Only motion back toward the soft
        # envelope is permitted on a saturated joint. Scale the entire IK step
        # together, so limits do not turn pure rotation into translation.
        lower = np.minimum(command_joints, joints-following_limit)
        upper = np.maximum(command_joints, joints+following_limit)
        result = self._joint_step(
            command_joints, np.concatenate((linear, angular)), dt,
            reference_offset=current_flange_rotation @ flange_to_tcp_position,
            lower=lower, upper=upper)
        # The measured-pose envelope limits the submitted trajectory, not the
        # goal fed into the proportional servo. Applying it to that goal adds
        # plant lag to the tracking time constant and throttles even feasible
        # constant-velocity motion. Preserve the whole joint-step direction.
        position_limit = max(self.config.max_linear_velocity*self.config.velocity_tracking_tau_s,
                             float(np.linalg.norm(command_tcp_position-current_tcp_position)))
        rotation_limit = max(self.config.max_angular_velocity*self.config.velocity_tracking_tau_s,
                             Rotation.from_matrix(command_tcp_rotation@current_tcp_rotation.T).magnitude())

        def within_envelope(q):
            p, r = self.forward_tcp(q, flange_to_tcp_position, flange_to_tcp_rotation)
            return (np.linalg.norm(p-current_tcp_position) <= position_limit + 1e-12
                    and Rotation.from_matrix(r@current_tcp_rotation.T).magnitude() <= rotation_limit + 1e-12)

        if not within_envelope(result):
            delta = result-command_joints
            low, high = 0., 1.
            for _ in range(12):
                middle = (low+high)*.5
                if within_envelope(command_joints+middle*delta):
                    low = middle
                else:
                    high = middle
            result = command_joints+low*delta
        self._command_joints = result.copy()
        return result

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

    def _joint_step(self, joints: np.ndarray, twist: np.ndarray, dt: float, *,
                    reference_offset=None, lower=None, upper=None) -> np.ndarray:
        jacobian_kdl = PyKDL.Jacobian(7)
        if self.jacobian_solver.JntToJac(self._joints(joints), jacobian_kdl) < 0:
            raise RuntimeError("KDL Jacobian calculation failed")
        if reference_offset is not None:
            # KDL returns the flange Jacobian. At the TCP,
            # v_tcp = v_flange + omega x (R_flange * tool_translation).
            jacobian_kdl.changeRefPoint(PyKDL.Vector(*map(float, reference_offset)))
        jacobian = np.array(
            [[jacobian_kdl[row, column] for column in range(7)] for row in range(6)]
        )
        regularized = jacobian @ jacobian.T + self.config.damping**2 * np.eye(6)
        velocity = jacobian.T @ np.linalg.solve(regularized, twist)
        velocity *= min(1., self.config.max_joint_speed / max(float(np.max(np.abs(velocity))), 1e-12))
        delta = velocity * dt
        lower = self.model.lower_limits if lower is None else np.maximum(lower, self.model.lower_limits)
        upper = self.model.upper_limits if upper is None else np.minimum(upper, self.model.upper_limits)
        scale = 1.
        for index, step in enumerate(delta):
            if step > 0:
                scale = min(scale, max(0., (upper[index]-joints[index])/step))
            elif step < 0:
                scale = min(scale, max(0., (lower[index]-joints[index])/step))
        return joints + scale * delta

