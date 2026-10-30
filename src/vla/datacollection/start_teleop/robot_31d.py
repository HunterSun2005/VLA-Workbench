"""Franky snapshot adapter for the independent 31D recorder."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from vla.robots.franka_robotiq.robot import (
    FrankaRobotiqRobot,
    FrankaRolloutSnapshot,
)


@dataclass(frozen=True)
class Franka31DRolloutSnapshot(FrankaRolloutSnapshot):
    robot_state: object
    joint_torques: np.ndarray
    external_joint_torques: np.ndarray
    joint_velocities: np.ndarray


class FrankaRobotiq31DRobot(FrankaRobotiqRobot):
    """Reuse the existing hardware path and append libfranka state values."""

    def get_rollout_snapshot(self) -> Franka31DRolloutSnapshot:
        snapshot = super().get_rollout_snapshot()
        with self._arm_lock:
            robot_state = self._arm.state
            joint_torques = np.asarray(robot_state.tau_J, dtype=np.float64).reshape(-1).copy()
            external_joint_torques = np.asarray(
                robot_state.tau_ext_hat_filtered, dtype=np.float64
            ).reshape(-1).copy()
            joint_velocities = np.asarray(robot_state.dq, dtype=np.float64).reshape(-1).copy()
            observed_at_ns = int(time.monotonic() * 1e9)
        vectors = (joint_torques, external_joint_torques, joint_velocities)
        if any(value.shape != (7,) or not np.isfinite(value).all() for value in vectors):
            raise ValueError("Franka 31D torque and velocity fields must be finite 7-vectors")
        return Franka31DRolloutSnapshot(
            observed_at_ns=observed_at_ns,
            pose=snapshot.pose,
            joint_positions=snapshot.joint_positions,
            base_wrench=snapshot.base_wrench,
            gripper_width=snapshot.gripper_width,
            gripper_open_fraction=snapshot.gripper_open_fraction,
            images=snapshot.images,
            image_timestamps=snapshot.image_timestamps,
            robot_state=robot_state,
            joint_torques=joint_torques,
            external_joint_torques=external_joint_torques,
            joint_velocities=joint_velocities,
        )
