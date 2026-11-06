"""Shared collection annotations and offline conversion of legacy measured snapshots."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from vla.control.geometry import FrankaRolloutSnapshot
from vla.datacollection.franka_robotiq_single_left import schema

COLLECTOR_HUMAN = 0
COLLECTOR_PI05 = 1
COLLECTOR_RLT_ACTOR = 2
COLLECTOR_NAMES = {
    COLLECTOR_HUMAN: "HUMAN",
    COLLECTOR_PI05: "VLA",
    COLLECTOR_RLT_ACTOR: "RLT_ACTOR",
}
MIN_EPISODE_FRAMES = 12

EXTRA_FEATURES = {
    "complementary_info.is_intervention": {
        "dtype": "float32",
        "shape": [1],
        "names": ["is_intervention"],
    },
    "complementary_info.phase": {
        "dtype": "float32",
        "shape": [1],
        "names": ["phase"],
    },
    "complementary_info.collector_policy_id": {
        "dtype": "int64",
        "shape": [1],
        "names": ["collector_policy_id"],
        "info": {"codebook": {"0": "human", "1": "pi0.5", "2": "pi_rlt_actor"}},
    },
}


def episode_source_frame_counts(
    phase_values: list[float], collector_values: list[int]
) -> dict[str, int]:
    if len(phase_values) != len(collector_values):
        raise ValueError("phase and collector annotations must have the same length")
    critical = [float(value) > 0.5 for value in phase_values]
    return {
        "critical_actor": sum(
            active and int(source) == COLLECTOR_RLT_ACTOR
            for active, source in zip(critical, collector_values, strict=True)
        ),
        "critical_human": sum(
            active and int(source) == COLLECTOR_HUMAN
            for active, source in zip(critical, collector_values, strict=True)
        ),
        "critical_vla": sum(
            active and int(source) == COLLECTOR_PI05
            for active, source in zip(critical, collector_values, strict=True)
        ),
    }


@dataclass(frozen=True)
class CommandSnapshot:
    action: np.ndarray
    source: int
    phase: float
    applied_at: float
    command_id: int = 0


def rollout_state(
    snapshot: FrankaRolloutSnapshot,
    previous: tuple[np.ndarray, np.ndarray, int] | None,
) -> np.ndarray:
    position = snapshot.pose[:3, 3]
    rotation = snapshot.pose[:3, :3]
    velocity = schema.base_tcp_velocity(
        previous[0] if previous else None,
        previous[1] if previous else None,
        previous[2] if previous else None,
        position,
        rotation,
        snapshot.observed_at_ns,
    )
    value = np.concatenate(
        (
            position,
            schema.matrix_to_rot6d(rotation),
            [snapshot.gripper_width],
            snapshot.base_wrench,
            velocity,
            snapshot.joint_positions,
            snapshot.joint_torques,
            snapshot.external_joint_torques,
            snapshot.joint_velocities,
        )
    ).astype(np.float32)
    if value.shape != (len(schema.STATE_FIELDS),) or not np.isfinite(value).all():
        raise ValueError(f"invalid rollout state: {value.shape}")
    return value
