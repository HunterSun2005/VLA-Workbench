from __future__ import annotations

from vla.core.interfaces import ChunkTransition, Observation, VLAOutput
from vla.core.config import RLTConfig, OfflineRLConfig
from vla.core.vla_adapter import VLAAdapter, DummyVLAAdapter
from vla.core.rl_token import RLTokenModule
from vla.core.actor import ChunkActor
from vla.core.critic import ChunkCritic, TwinCritic
from vla.core.losses import discounted_chunk_return, critic_loss, actor_loss
from vla.core.replay_buffer import ReplayBuffer
from vla.core.utils import (
    soft_update,
    flatten_chunk,
    unflatten_chunk,
    compute_discount_vector,
    build_mlp,
    filter_encoder_only,
    infer_actor_architecture,
)
from vla.core.policy import RLTPolicy
from vla.core.algorithm import RLTAlgorithm
from vla.core.collector import Environment, DummyEnvironment, execute_chunk
from vla.core.rewards import build_reward_seq

__all__ = [
    "ChunkTransition",
    "Observation",
    "VLAOutput",
    "RLTConfig",
    "OfflineRLConfig",
    "VLAAdapter",
    "DummyVLAAdapter",
    "RLTokenModule",
    "ChunkActor",
    "ChunkCritic",
    "TwinCritic",
    "discounted_chunk_return",
    "critic_loss",
    "actor_loss",
    "ReplayBuffer",
    "RLTPolicy",
    "RLTAlgorithm",
    "soft_update",
    "flatten_chunk",
    "unflatten_chunk",
    "compute_discount_vector",
    "build_mlp",
    "filter_encoder_only",
    "infer_actor_architecture",
    "Environment",
    "DummyEnvironment",
    "execute_chunk",
    "build_reward_seq",
]
