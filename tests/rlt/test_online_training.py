from __future__ import annotations

import copy
import io
from types import SimpleNamespace

import pytest
import torch

from vla.core.actor import ChunkActor
from vla.core.critic import TwinCritic
from vla.core.losses import actor_loss
from vla.core.online_replay import PersistentReplayBuffer, ReplayTransitionRecord
from vla.core.phase_controller import PhaseController
from vla.adapters.lerobot.policies.action_modifier import RLTActionModifier
from vla.core.online_training import (
    _OnlineTrainingProgress,
    OnlineRLTTrainingConfig,
    OnlineUpdateScheduler,
    build_critical_transition_specs,
    encode_saved_episode,
    resolve_state_indices,
)
import vla.core.online_training as online_training

C = 4
ACTION_DIM = 2
STATE_DIM = 5


def _transition(value: float = 0.0) -> dict[str, torch.Tensor]:
    from tests.rlt.execution_fixtures import execution_context
    return execution_context({
        "state_vec": torch.full((STATE_DIM,), value),
        "exec_chunk": torch.zeros(C, ACTION_DIM),
        "ref_chunk": torch.zeros(C, ACTION_DIM),
        "reward_seq": torch.zeros(C),
        "next_state_vec": torch.full((STATE_DIM,), value + 1),
        "next_ref_chunk": torch.zeros(C, ACTION_DIM),
        "done": torch.tensor(0.0),
        "intervention": torch.tensor(0.0),
        "actual_steps": torch.tensor(C),
        "source": torch.tensor(1),
        "episode_id": torch.tensor(int(value)),
        "is_critical": torch.tensor(1.0),
    }, start_slot=int(value))


def _replay(tmp_path, *, resume=False, capacity=10):
    return PersistentReplayBuffer(
        tmp_path,
        capacity=capacity,
        contract={"chunk_length": C},
        resume=resume,
        seed=7,
    )


def test_persistent_replay_is_idempotent_and_restores_progress(tmp_path):
    replay = _replay(tmp_path, capacity=1)
    assert (
        replay.commit_episode(
            episode_key="ep0",
            episode_id=0,
            outcome="success",
            warmup=True,
            warmup_env_steps=4,
            records=[ReplayTransitionRecord(0, _transition())],
            utd_ratio=5,
        )
        == 1
    )
    assert (
        replay.commit_episode(
            episode_key="ep0",
            episode_id=0,
            outcome="success",
            warmup=True,
            warmup_env_steps=4,
            records=[ReplayTransitionRecord(0, _transition())],
            utd_ratio=5,
        )
        == 0
    )
    replay.commit_episode(
        episode_key="ep1",
        episode_id=1,
        outcome="failure",
        warmup=False,
        warmup_env_steps=0,
        records=[ReplayTransitionRecord(2, _transition(1))],
        utd_ratio=5,
    )
    assert len(replay) == 1
    assert replay.produced_credits == 40
    assert replay.warmup_env_steps == 4
    replay.checkpoint_progress(
        checkpoint="checkpoint-a",
        consumed_credits=10,
        critic_updates=10,
        actor_updates=5,
    )
    replay.close()

    restored = _replay(tmp_path, resume=True, capacity=1)
    assert restored.active_checkpoint == "checkpoint-a"
    assert restored.consumed_credits == 10
    assert restored.actor_updates == 5
    assert restored.sample(1)["episode_id"].item() == 1
    restored.close()


def test_critical_specs_use_stride_and_one_terminal_window():
    annotations = {frame: {"phase": 1.0} for frame in range(7)}
    specs = build_critical_transition_specs(
        annotations,
        episode_start=0,
        episode_stop=7,
        chunk_length=C,
        stride=2,
    )
    assert [
        (s.anchor_frame, s.next_frame, s.actual_steps, s.terminal) for s in specs
    ] == [
        (0, 4, 4, False),
        (2, 6, 4, False),
        (4, 6, 3, True),
    ]


class _MustNotRun(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("warmup must not run Actor")

    def encode(self, *args, **kwargs):
        return torch.zeros(args[0].shape[0], 3)


def test_online_warmup_keeps_critical_phase_but_executes_vla():
    phase = PhaseController(mode="manual")
    phase.trigger_critical()
    modifier = RLTActionModifier(
        rl_token=_MustNotRun(),
        actor=_MustNotRun(),
        phase_ctrl=phase,
        chunk_length=C,
        action_dim=ACTION_DIM,
        proprio_dim=2,
        online_warmup=True,
    )
    vla = torch.arange(12 * ACTION_DIM, dtype=torch.float32).view(1, 12, ACTION_DIM)
    result = modifier.compute_chunk(vla, torch.zeros(1, 2), torch.zeros(1, 1, 1))

    assert torch.equal(result, vla[:, :C])
    metadata = [modifier.pop_step_metadata() for _ in range(C)]
    assert [item.phase for item in metadata] == [1.0] * C
    assert [item.source_type for item in metadata] == [0.0] * C


class _FakeHFDataset:
    column_names = [
        "complementary_info.is_intervention",
        "complementary_info.phase",
        "complementary_info.collector_policy_id",
    ]

    def __init__(self):
        self.values = {
            "complementary_info.is_intervention": [[0], [0], [1], [0], [0], [0], [0]],
            "complementary_info.phase": [[1]] * 7,
            "complementary_info.collector_policy_id": [
                [1],
                [1],
                [0],
                [1],
                [1],
                [1],
                [1],
            ],
        }

    def select(self, indices):
        indices = list(indices)
        return {
            key: [values[index] for index in indices]
            for key, values in self.values.items()
        }


class _FakeDataset(torch.utils.data.Dataset):
    hf_dataset = _FakeHFDataset()

    def __len__(self):
        return 7

    def __getitem__(self, index):
        return {
            "observation.state": torch.tensor([float(index), 0.0]),
            "action": torch.full((C, ACTION_DIM), float(index + 1)),
            "task": "task",
        }


class _FakeCapture:
    def consume(self):
        return torch.ones(1, 1, 1)


class _FakePI05:
    def predict_action_chunk(self, batch):
        count = batch["observation.state"].shape[0]
        return torch.full((count, C, ACTION_DIM), 9.0)


class _FakeRLToken:
    def encode(self, prefix):
        return torch.full((prefix.shape[0], 3), 2.0)


def test_episode_encoding_replaces_intervention_reference_and_rewards_terminal():
    records, warmup_steps = encode_saved_episode(
        dataset=_FakeDataset(),
        episode_id=3,
        episode_start=0,
        episode_stop=7,
        episode_success=True,
        warmup=False,
        pi05=_FakePI05(),
        rl_token=_FakeRLToken(),
        preprocessor=lambda batch: batch,
        capture=_FakeCapture(),
        chunk_length=C,
        stride=2,
        action_dim=ACTION_DIM,
        proprio_dim=2,
        batch_size=1,
        num_workers=0,
        device="cpu",
        task_instruction=None,
    )
    assert warmup_steps == 7
    assert len(records) == 3
    first = records[0].transition
    assert torch.equal(first["ref_chunk"][2], first["exec_chunk"][2])
    assert torch.equal(first["ref_chunk"][1], torch.full((ACTION_DIM,), 9.0))
    terminal = records[-1].transition
    assert terminal["done"].item() == 1
    assert terminal["actual_steps"].item() == 3
    assert terminal["reward_seq"].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert torch.count_nonzero(terminal["exec_chunk"][3:]) == 0


def test_episode_encoding_selects_named_state_before_preprocessing():
    class WideStateDataset(_FakeDataset):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            item["observation.state"] = torch.tensor(
                [float(index), -100.0, float(index + 10)]
            )
            return item

    seen_shapes = []

    def preprocessor(batch):
        seen_shapes.append(tuple(batch["observation.state"].shape))
        return batch

    records, _ = encode_saved_episode(
        dataset=WideStateDataset(),
        episode_id=3,
        episode_start=0,
        episode_stop=7,
        episode_success=False,
        warmup=True,
        pi05=_FakePI05(),
        rl_token=_FakeRLToken(),
        preprocessor=preprocessor,
        capture=_FakeCapture(),
        chunk_length=C,
        stride=2,
        action_dim=ACTION_DIM,
        proprio_dim=2,
        batch_size=1,
        num_workers=0,
        device="cpu",
        task_instruction=None,
        state_indices=(0, 2),
    )

    assert set(seen_shapes) == {(1, 2)}
    assert records[0].transition["state_vec"][-2:].tolist() == [0.0, 10.0]


def test_online_state_indices_follow_dataset_feature_names():
    dataset = SimpleNamespace(
        meta=SimpleNamespace(
            features={
                "observation.state": {
                    "names": ["tcp_x", "force_x", "gripper", "joint_1"]
                }
            }
        )
    )

    assert resolve_state_indices(dataset, ("tcp_x", "gripper"), 2) == (0, 2)
    with pytest.raises(ValueError, match="missing checkpoint fields"):
        resolve_state_indices(dataset, ("tcp_x", "tcp_y"), 2)


class _Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            device="cpu",
            gamma=0.99,
            beta=0.3,
            prefix_bc_weight=0.3,
            action_contract=None,
            tau=0.005,
            chunk_length=C,
            target_q_clip=100.0,
        )
        self.actor = ChunkActor(STATE_DIM, C * ACTION_DIM, hidden_dim=16, num_layers=1)
        self.critic = TwinCritic(STATE_DIM, C * ACTION_DIM, hidden_dim=16, num_layers=1)
        self.target_critic = copy.deepcopy(self.critic)
        self.register_buffer("_critic_step", torch.zeros((), dtype=torch.long))

    @staticmethod
    def execution_projection(mu, anchor):
        return mu.clamp(-1, 1)

    @staticmethod
    def _coerce_batch(batch):
        result = dict(batch)
        result["exec_chunk_flat"] = result["exec_chunk"].flatten(start_dim=-2)
        result["ref_chunk_flat"] = result["ref_chunk"].flatten(start_dim=-2)
        result["next_exec_chunk_flat"] = result["next_exec_chunk"].flatten(start_dim=-2)
        result["next_ref_flat"] = result["next_ref_chunk"].flatten(start_dim=-2)
        return result

    def _actor_loss_without_critic_grads(self, batch):
        parameters = list(self.critic.parameters())
        for parameter in parameters:
            parameter.requires_grad_(False)
        try:
            return actor_loss(self.actor, self.critic, batch, beta=self.config.beta)
        finally:
            for parameter in parameters:
                parameter.requires_grad_(True)


def test_utd_credits_produce_exact_critic_actor_counts(tmp_path):
    replay = _replay(tmp_path)
    policy = _Policy()
    scheduler = OnlineUpdateScheduler(
        policy=policy,
        replay=replay,
        config=OnlineRLTTrainingConfig(
            replay_root=str(tmp_path),
            batch_size=1,
            grad_clip_norm=0,
        ),
    )
    replay.commit_episode(
        episode_key="ep0",
        episode_id=0,
        outcome="success",
        warmup=True,
        warmup_env_steps=4,
        records=[ReplayTransitionRecord(0, _transition())],
        utd_ratio=5,
    )
    first = scheduler.drain()
    assert first["critic_updates"] == 20
    assert first["actor_updates"] == 10
    replay.commit_episode(
        episode_key="ep1",
        episode_id=1,
        outcome="failure",
        warmup=False,
        warmup_env_steps=0,
        records=[ReplayTransitionRecord(2, _transition(1))],
        utd_ratio=5,
    )
    second = scheduler.drain()
    assert second["critic_updates"] == 20
    assert second["actor_updates"] == 10
    assert scheduler.critic_updates == 40
    assert scheduler.actor_updates == 20
    replay.close()


def test_online_training_progress_updates_one_terminal_line() -> None:
    stream = io.StringIO()
    messages = []
    progress = _OnlineTrainingProgress(
        10, stream=stream, refresh_interval_s=0, callback=messages.append
    )

    progress.update(0, critic_loss_value=None, actor_updates=0, actor_loss_value=None)
    progress.update(5, critic_loss_value=1.25, actor_updates=2, actor_loss_value=0.5)
    progress.update(10, critic_loss_value=0.75, actor_updates=5, actor_loss_value=0.25)

    output = stream.getvalue()
    assert output.count("\n") == 1
    assert output.count("\r") == 3
    assert "10/10 (100.0%)" in output
    assert "actor_updates=5" in output
    assert messages[-1] == {
        "completed_updates": 10,
        "total_updates": 10,
        "critic_loss": 0.75,
        "actor_updates": 5,
        "actor_loss": 0.25,
    }


def test_actor_reuses_even_critic_batch_after_target_update(tmp_path, monkeypatch):
    replay = _replay(tmp_path)
    replay.commit_episode(
        episode_key="ep0",
        episode_id=0,
        outcome="success",
        warmup=False,
        warmup_env_steps=0,
        records=[ReplayTransitionRecord(0, _transition())],
        utd_ratio=5,
    )
    policy = _Policy()
    scheduler = OnlineUpdateScheduler(
        policy=policy,
        replay=replay,
        config=OnlineRLTTrainingConfig(
            replay_root=str(tmp_path), batch_size=1, grad_clip_norm=0
        ),
    )
    events = []
    critic_batches = []

    def fake_critic_loss(critic, target_critic, actor, batch, **kwargs):
        del target_critic, actor, kwargs
        critic_batches.append(id(batch["state_vec"]))
        events.append("critic_loss")
        return next(critic.parameters()).sum() * 0 + 1

    def fake_actor_loss(batch):
        assert id(batch["state_vec"]) == critic_batches[-1]
        events.append("actor_loss_same_batch")
        return next(policy.actor.parameters()).sum() * 0 + 1

    critic_step = scheduler.critic_optimizer.step
    actor_step = scheduler.actor_optimizer.step
    scheduler.critic_optimizer.step = lambda *args, **kwargs: (
        events.append("critic_step"),
        critic_step(*args, **kwargs),
    )[-1]
    scheduler.actor_optimizer.step = lambda *args, **kwargs: (
        events.append("actor_step"),
        actor_step(*args, **kwargs),
    )[-1]
    policy._actor_loss_without_critic_grads = fake_actor_loss
    monkeypatch.setattr(online_training, "critic_loss", fake_critic_loss)
    monkeypatch.setattr(
        online_training, "soft_update", lambda *args: events.append("target_update")
    )

    scheduler.drain()

    expected = []
    for step in range(1, 21):
        expected.extend(['critic_loss', 'critic_step', 'target_update'])
        if step % 2 == 0:
            expected.extend(['actor_loss_same_batch', 'actor_step'])
    assert events == expected
    replay.close()
