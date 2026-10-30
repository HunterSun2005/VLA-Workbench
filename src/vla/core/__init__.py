"""Core public API; keep collection contracts usable without Torch."""
from importlib import import_module

_EXPORTS = {
    'ChunkTransition': ('vla.core.interfaces', 'ChunkTransition'),
    'Observation': ('vla.core.interfaces', 'Observation'),
    'VLAOutput': ('vla.core.interfaces', 'VLAOutput'),
    'RLTConfig': ('vla.core.config', 'RLTConfig'),
    'OfflineRLConfig': ('vla.core.config', 'OfflineRLConfig'),
    'VLAAdapter': ('vla.core.vla_adapter', 'VLAAdapter'),
    'DummyVLAAdapter': ('vla.core.vla_adapter', 'DummyVLAAdapter'),
    'RLTokenModule': ('vla.core.rl_token', 'RLTokenModule'),
    'ChunkActor': ('vla.core.actor', 'ChunkActor'),
    'ChunkCritic': ('vla.core.critic', 'ChunkCritic'),
    'TwinCritic': ('vla.core.critic', 'TwinCritic'),
    'discounted_chunk_return': ('vla.core.losses', 'discounted_chunk_return'),
    'critic_loss': ('vla.core.losses', 'critic_loss'),
    'actor_loss': ('vla.core.losses', 'actor_loss'),
    'ReplayBuffer': ('vla.core.replay_buffer', 'ReplayBuffer'),
    'soft_update': ('vla.core.utils', 'soft_update'),
    'flatten_chunk': ('vla.core.utils', 'flatten_chunk'),
    'unflatten_chunk': ('vla.core.utils', 'unflatten_chunk'),
    'compute_discount_vector': ('vla.core.utils', 'compute_discount_vector'),
    'build_mlp': ('vla.core.utils', 'build_mlp'),
    'filter_encoder_only': ('vla.core.utils', 'filter_encoder_only'),
    'infer_actor_architecture': ('vla.core.utils', 'infer_actor_architecture'),
    'RLTPolicy': ('vla.core.policy', 'RLTPolicy'),
    'RLTAlgorithm': ('vla.core.algorithm', 'RLTAlgorithm'),
    'Environment': ('vla.core.collector', 'Environment'),
    'DummyEnvironment': ('vla.core.collector', 'DummyEnvironment'),
    'execute_chunk': ('vla.core.collector', 'execute_chunk'),
    'build_reward_seq': ('vla.core.rewards', 'build_reward_seq'),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value
