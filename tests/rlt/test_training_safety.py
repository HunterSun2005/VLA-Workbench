from __future__ import annotations

import copy
import multiprocessing
from types import SimpleNamespace

import pytest
import torch

from vla.core.losses import actor_loss
from vla.core.online_replay import PersistentReplayBuffer, ReplayTransitionRecord
from vla.core.training_contract import supervision_reference
from vla.core.writer_lock import WriterLease
from tests.rlt.test_online_consistency import session_at
from tests.rlt.test_online_training import _Policy, _transition


class OutputActor:
    def __init__(self, values):
        self.mu = torch.tensor([values], requires_grad=True)

    def forward(self, *args, **kwargs):
        return self.mu, None


class SumQ:
    def min_q(self, x, actions):
        self.actions = actions.detach().clone()
        return actions.sum(-1, keepdim=True)


def loss_batch(steps=4):
    return {"state_vec": torch.zeros(1, 2), "ref_chunk_flat": torch.zeros(1, 4),
            "reward_seq": torch.zeros(1, 4), "actual_steps": torch.tensor([steps])}


def test_actor_q_uses_deployment_bounds_and_bc_restores_saturated_output():
    actor, critic = OutputActor([2., -2., .2, -.2]), SumQ()
    actor_loss(actor, critic, loss_batch(), beta=1).backward()
    assert critic.actions.tolist() == [[1., -1., pytest.approx(.2), pytest.approx(-.2)]]
    assert actor.mu.grad[0, 0] > 0
    assert actor.mu.grad[0, 1] < 0


@pytest.mark.parametrize("beta", [0., 1.])
def test_actor_padding_has_no_q_or_bc_gradient(beta):
    actor, critic = OutputActor([.2, .3, .4, .5]), SumQ()
    actor_loss(actor, critic, loss_batch(1), beta=beta).backward()
    assert torch.equal(critic.actions[:, 1:], torch.zeros(1, 3))
    assert torch.equal(actor.mu.grad[:, 1:], torch.zeros(1, 3))
    assert actor.mu.grad[0, 0] != 0


@pytest.mark.parametrize("steps", [0, -1, 5, 1.5, float("nan")])
def test_invalid_actual_steps_fail(steps):
    with pytest.raises(ValueError, match="actual_steps"):
        actor_loss(OutputActor([.2] * 4), SumQ(), loss_batch(steps), beta=1)


def commit(session, key, records):
    return session.replay.commit_episode(episode_key=key, episode_id=0, outcome="success",
        warmup=True, warmup_env_steps=1000, records=records, utd_ratio=5)


def test_empty_replay_never_ends_warmup(tmp_path):
    session = session_at(tmp_path)
    try:
        commit(session, "empty", [])
        session._drain_pending()
        assert session.is_warmup
        assert session.replay.warmup_env_steps == 1000
        assert session.replay.trainable_env_steps == 0
        assert session.scheduler.critic_updates == session.scheduler.actor_updates == 0
    finally:
        session.close()


def test_trainable_steps_deduplicate_overlap_and_batch_gates_activation(tmp_path):
    session = session_at(tmp_path)
    session.config.batch_size = 3
    try:
        records = [ReplayTransitionRecord(i, _transition(i)) for i in (0, 2)]
        commit(session, "a", records)
        commit(session, "a", records)
        assert session.replay.trainable_env_steps == 6
        session._drain_pending()
        assert session.is_warmup and session.scheduler.critic_updates == 0
        commit(session, "b", [ReplayTransitionRecord(10, _transition())])
        session._drain_pending()
        assert not session.is_warmup
        assert session.scheduler.critic_updates == 50
        session._restore_checkpoint(session.replay.active_checkpoint)
        assert not session.is_warmup
    finally:
        session.close()


def _try_lease(root, pipe):
    try:
        with WriterLease(root):
            pipe.send("acquired")
    except RuntimeError:
        pipe.send("busy")
    finally:
        pipe.close()


def test_writer_lock_cross_process_and_release(tmp_path):
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "replay"
    lease = WriterLease(root)
    try:
        parent, child = context.Pipe()
        process = context.Process(target=_try_lease, args=(root, child))
        process.start()
        assert parent.poll(30) and parent.recv() == "busy"
        process.join(30)
        assert process.exitcode == 0
    finally:
        lease.close()
    with WriterLease(root):
        pass


def test_two_replay_writers_fail_and_init_error_releases_lease(tmp_path):
    kwargs = dict(capacity=10, contract={}, seed=7)
    replay = PersistentReplayBuffer(tmp_path, resume=False, **kwargs)
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            PersistentReplayBuffer(tmp_path, resume=True, **kwargs)
    finally:
        replay.close()
    with pytest.raises(ValueError, match="contract"):
        PersistentReplayBuffer(tmp_path, resume=True, **{**kwargs, "contract": {"bad": True}})
    restored = PersistentReplayBuffer(tmp_path, resume=True, **kwargs)
    restored.close()


def test_human_reference_is_per_frame_and_does_not_mutate_vla():
    ref, executed = torch.zeros(4, 2), torch.ones(4, 2)
    annotations = {1: {"is_intervention": 1}, 3: {"collector_policy_id": 0}}
    corrected = supervision_reference(executed, ref, annotations, 0)
    assert corrected[:, 0].tolist() == [0, 1, 0, 1]
    assert not ref.any()
    assert torch.equal(supervision_reference(executed, ref, None, 0), ref)


def test_offline_cache_applies_human_reference_without_changing_next_proposal():
    from vla.cli.build_transition_cache_v2 import _encoded_to_transitions
    encoded = {i: {"state_vec": torch.zeros(2), "exec_chunk": torch.ones(4, 2),
        "ref_chunk": torch.zeros(4, 2)} for i in range(5)}
    from tests.rlt.execution_fixtures import encoded_anchors, human_rows
    result = _encoded_to_transitions(encoded_anchors(encoded), list(range(5)), 4, 4, True, 0,
        frame_annotations=human_rows(5))
    assert result[0]["ref_chunk"][:, 0].tolist() == [1, 1, 1, 1]
    assert not result[0]["next_ref_chunk"].any()


def test_dataset_lease_prevents_other_replay_root_and_releases_partial_acquire(tmp_path):
    dataset, first, second = [tmp_path / name for name in ("dataset", "replay1", "replay2")]
    with WriterLease(dataset, first):
        with pytest.raises(RuntimeError, match="already owns"):
            WriterLease(second, dataset)
        with WriterLease(second):
            pass


def _hold_lease(root, pipe):
    import time
    with WriterLease(root):
        pipe.send("ready")
        while True:
            time.sleep(.1)


def test_writer_process_exit_releases_lease(tmp_path):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_hold_lease, args=(tmp_path / "replay", child))
    process.start()
    try:
        assert parent.poll(30) and parent.recv() == "ready"
    finally:
        process.terminate()
        process.join(30)
        parent.close()
        child.close()
    with WriterLease(tmp_path / "replay"):
        pass


def native_policy():
    from vla.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
    base = _Policy()
    policy = object.__new__(ChunkACPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = base.config
    policy.config.actor_update_interval = 2
    policy.actor, policy.critic, policy.target_critic = base.actor, base.critic, base.target_critic
    policy.register_buffer("_critic_step", torch.zeros((), dtype=torch.long))
    object.__setattr__(policy, "_rl_token_policy", None)
    policy.execution_projection = base.execution_projection
    return policy


def test_native_forward_is_pure_and_success_hook_advances_once():
    policy = native_policy().eval()
    with torch.no_grad():
        for param in policy.critic.parameters():
            param.add_(1)
    before = copy.deepcopy(policy.state_dict())
    batch = {key: value.unsqueeze(0) for key, value in _transition().items()}
    with torch.no_grad():
        policy.forward(batch)
        policy.forward(batch)
    assert all(torch.equal(value, before[key]) for key, value in policy.state_dict().items())
    policy.on_optimizer_step()
    assert policy._critic_step.item() == 1
    assert any(not torch.equal(value, before[key]) for key, value in policy.state_dict().items() if key.startswith("target_critic."))
    _, info = policy.forward(batch)
    assert "loss_actor" in info


@pytest.mark.parametrize("skipped,raises", [(False, False), (True, False), (False, True)])
def test_native_adapter_skips_failed_optimizer_steps(skipped, raises):
    from vla.cli.train_actor_critic_native import install_ac_update_hook
    policy = native_policy()
    def update_policy(policy, accelerator):
        if raises:
            raise RuntimeError("optimizer failed")
        return None, {}
    trainer = SimpleNamespace(update_policy=update_policy)
    install_ac_update_hook(trainer)
    accelerator = SimpleNamespace(unwrap_model=lambda model, **kwargs: model, optimizer_step_was_skipped=skipped)
    if raises:
        with pytest.raises(RuntimeError):
            trainer.update_policy(policy, accelerator)
    else:
        trainer.update_policy(policy, accelerator)
    assert policy._critic_step.item() == int(not skipped and not raises)
