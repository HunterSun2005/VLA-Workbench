"""Small behavioral checks for actor and twin critic; training integration lives separately."""
from __future__ import annotations

import torch

from vla.core.actor import ChunkActor
from vla.core.critic import TwinCritic


def test_actor_forward_sampling_and_gradients():
    actor = ChunkActor(state_dim=8, chunk_dim=12, hidden_dim=16, num_layers=2)
    state, ref = torch.randn(4, 8), torch.randn(4, 12)
    mu, std = actor(state, ref)
    assert mu.shape == std.shape == (4, 12)
    torch.testing.assert_close(std, torch.full_like(std, 0.05))
    action, sampled_mu = actor.sample(state, ref)
    assert action.shape == sampled_mu.shape == (4, 12)
    torch.testing.assert_close(sampled_mu, mu)
    action.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())


def test_reference_dropout_changes_actor_input_only_during_training():
    # Compare the real forward path to zero-reference inference, rather than
    # testing a separately generated random mask that never reaches the actor.
    torch.manual_seed(0)
    actor = ChunkActor(state_dim=8, chunk_dim=12, hidden_dim=16, ref_dropout_p=1.0)
    state, ref = torch.randn(4, 8), torch.ones(4, 12)
    dropped, _ = actor(state, ref, training=True)
    zero_ref, _ = actor(state, torch.zeros_like(ref), training=False)
    torch.testing.assert_close(dropped, zero_ref)
    actor.ref_dropout_p = 0.0
    kept, _ = actor(state, ref, training=True)
    inference, _ = actor(state, ref, training=False)
    torch.testing.assert_close(kept, inference)
    assert not torch.allclose(inference, zero_ref)


def test_twin_critic_outputs_minimum_and_gradients():
    critic = TwinCritic(state_dim=8, chunk_dim=12, hidden_dim=16, num_layers=2)
    state, action = torch.randn(4, 8), torch.randn(4, 12)
    q1, q2 = critic(state, action)
    assert q1.shape == q2.shape == (4, 1)
    minimum = critic.min_q(state, action)
    torch.testing.assert_close(minimum, torch.minimum(q1, q2))
    minimum.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())
