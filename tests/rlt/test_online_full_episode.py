from collections import Counter
from dataclasses import replace
import sys
from types import SimpleNamespace

import pytest

from vla.cli.serve_franka_rollout import parse_args, FrankaRolloutPolicyServer
from vla.core.online_replay import ReplayTransitionRecord
from vla.core.online_training import OnlineRLTTrainingConfig, encode_saved_episode
from tests.rlt.test_online_consistency import session_at
from tests.rlt.test_online_training import (
    _transition, _FakeDataset, _FakeHFDataset, _FakeCapture, _FakePI05, _FakeRLToken, C, ACTION_DIM,
)


def test_full_episode_requires_inherited_weights():
    config = OnlineRLTTrainingConfig(replay_root="unused", full_episode=True, warmup_steps=0)
    with pytest.raises(ValueError, match="inherited"):
        config.validate(chunk_length=20, actor_update_interval=2)
    replace(config, inherit_actor_critic=True).validate(chunk_length=20, actor_update_interval=2)


def test_cli_requires_online_inheritance(monkeypatch):
    args = ["serve", "--policy-path", "/unused", "--socket-path", "/unused", "--task", "test", "--online-full-episode"]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit):
        parse_args()
    monkeypatch.setattr(sys, "argv", args + ["--online-inherit-actor-critic", "--online-replay-root", "/unused/replay"])
    assert parse_args().online_full_episode


def test_full_episode_trains_only_after_a_full_batch(tmp_path):
    session = session_at(tmp_path)
    session.config = replace(session.config, full_episode=True, inherit_actor_critic=True,
                             warmup_steps=0, batch_size=2)
    session.scheduler.config = session.config
    try:
        assert not session.is_warmup
        for episode_id in (0, 1):
            session.replay.commit_episode(episode_key=str(episode_id), episode_id=episode_id,
                outcome="success", warmup=False, warmup_env_steps=4,
                records=[ReplayTransitionRecord(0, _transition())], utd_ratio=5,
                checkpoint=session._save_checkpoint())
            session._drain_pending()
            assert session.scheduler.critic_updates == (0 if episode_id == 0 else 10)
        assert session.scheduler.actor_updates == 5
        assert session.replay.actor_enabled
        assert session.replay.warmup_env_steps == 0
    finally:
        session.close()




def test_server_rejects_vla_request_in_full_episode_mode():
    server = object.__new__(FrankaRolloutPolicyServer)
    server.online_session = SimpleNamespace(require_ready=lambda: None,
        config=SimpleNamespace(full_episode=True))
    server.task, server.phase = "test", "rl"
    with pytest.raises(ValueError, match="only accepts phase=rl"):
        server.handle({"kind": "infer", "task": "test", "phase": "vla"}, {})




@pytest.mark.parametrize("bad_field", [None, "phase", "collector_policy_id"])
def test_full_episode_encoding_checks_entire_episode(bad_field):
    dataset = _FakeDataset()
    dataset.hf_dataset = _FakeHFDataset()
    dataset.hf_dataset.values["complementary_info.collector_policy_id"] = [[2]] * 7
    if bad_field is not None:
        dataset.hf_dataset.values[f"complementary_info.{bad_field}"][0] = [0 if bad_field == "phase" else 1]
    args = dict(dataset=dataset, episode_id=0, episode_start=0, episode_stop=7,
        episode_success=True, warmup=False, pi05=_FakePI05(), rl_token=_FakeRLToken(),
        preprocessor=lambda batch: batch, capture=_FakeCapture(), chunk_length=C,
        stride=2, action_dim=ACTION_DIM, proprio_dim=2, batch_size=1, num_workers=0,
        device="cpu", task_instruction=None, require_full_episode=True)
    if bad_field:
        with pytest.raises(ValueError, match="throughout"):
            encode_saved_episode(**args)
    else:
        records, steps = encode_saved_episode(**args)
        assert len(records) == 3 and steps == 7
