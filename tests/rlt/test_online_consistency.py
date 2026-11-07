from __future__ import annotations

import copy
import json
from types import SimpleNamespace
import uuid

import numpy as np
import pandas as pd
import pytest
import torch

from vla.core.online_contract import (
    TIMING_FEATURES, bind_run, load_run, model_digest, reject_unaligned_offline_dataset,
)
from vla.core.online_replay import PersistentReplayBuffer, ReplayTransitionRecord
from vla.core.online_training import (
    OnlineRLTSession, OnlineRLTTrainingConfig, OnlineUpdateScheduler,
    build_critical_transition_specs,
)
from vla.datacollection.franka_robotiq_single_left.v30_writer import V30DatasetWriter, repair_committed_metadata
from vla.datacollection.franka_robotiq_rollout.recording import EXTRA_FEATURES, CommandSnapshot
from tests.rlt.test_online_training import _Policy, _transition


def timed_rows(length=7, *, terminal=True):
    return {
        index: {
            "phase": 1, "is_intervention": 0, "collector_policy_id": 1,
            "observed_at_ns": 1_000_000_000 + index * 33_333_333,
            "command_at_ns": 1_001_000_000 + index * 33_333_333,
            "command_id": index, "segment_id": 0,
            "slot_id": index, "submitted_at_ns": 1_002_000_000+index*33_333_333,
            "supervision_valid": 1, "observation_valid": 1, "plan_version": 1, "action_index": index,
            "task_terminal": int(terminal and index == length-1),
            "action_valid": int(index < length - 1),
            "segment_terminal": int(terminal and index == length - 1),
        } for index in range(length)
    }


def specs(rows):
    return build_critical_transition_specs(rows, episode_start=0, episode_stop=len(rows), chunk_length=4, stride=2, fps=30)


def test_timed_windows_use_real_endpoint_not_last_action():
    result = specs(timed_rows())
    assert [(s.anchor_frame, s.next_frame, s.actual_steps, s.terminal) for s in result] == [(0, 4, 4, False), (2, 6, 4, True)]
    assert specs(timed_rows(4))[-1].next_frame == 3
    assert specs(timed_rows(4))[-1].actual_steps == 3


def test_internal_gap_is_not_terminal_and_drops_partial_windows():
    rows = timed_rows(10)
    rows[3]["action_valid"] = 0
    for index in range(4, 10):
        rows[index]["segment_id"] = 1
        rows[index]["observed_at_ns"] += 1_000_000_000
        rows[index]["command_at_ns"] += 1_000_000_000
    result = specs(rows)
    assert all(item.anchor_frame >= 4 for item in result)
    assert sum(item.terminal for item in result) == 1
    assert result[-1].next_frame == 9


@pytest.mark.parametrize("field,value", [("command_at_ns", 1), ("segment_id", 9), ("observed_at_ns", 9_000_000_000)])
def test_bad_timing_is_rejected(field, value):
    rows = timed_rows()
    rows[2][field] = value
    with pytest.raises(ValueError):
        specs(rows)


def test_second_critical_interval_is_rejected():
    rows = timed_rows(9)
    rows[3]["phase"] = 0
    with pytest.raises(ValueError, match="one contiguous"):
        specs(rows)


def test_run_identity_and_contract_cannot_be_reused(tmp_path):
    first = bind_run(tmp_path / "a", {"warmup": 1000, "task": "a"}, resume=False)
    second = bind_run(tmp_path / "b", {"warmup": 1000, "task": "a"}, resume=False)
    assert first["dataset_uuid"] != second["dataset_uuid"]
    assert bind_run(tmp_path / "a", {"warmup": 1000, "task": "a"}, resume=True) == first
    with pytest.raises(ValueError, match="warmup"):
        bind_run(tmp_path / "a", {"warmup": 2000, "task": "a"}, resume=True)
    with pytest.raises(ValueError, match="legacy"):
        load_run(tmp_path / "old")
    with pytest.raises(ValueError, match="schema-5"):
        reject_unaligned_offline_dataset({"recording_schema_version": 4})
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.bin").write_bytes(b"original")
    before = model_digest(weights)
    (weights / "model.bin").write_bytes(b"replacement")
    assert model_digest(weights) != before


def session_at(tmp_path):
    session = object.__new__(OnlineRLTSession)
    session.root = tmp_path / "replay"
    session.dataset = SimpleNamespace(root=tmp_path / "dataset", repo_id="local/test")
    session.run = bind_run(session.dataset.root, {"test": True}, resume=False)
    session.replay = PersistentReplayBuffer(session.root, capacity=20, contract=session.run, resume=False, seed=7)
    session.config = OnlineRLTTrainingConfig(replay_root=str(session.root), batch_size=1, warmup_steps=2, grad_clip_norm=0)
    session.policy = _Policy()
    session.policy.config.proprio_dim = 2
    session.policy.config.action_dim = 2
    session.policy.quiesce_for_online_training = lambda: None
    session.policy.reset = lambda: None
    session.policy.set_online_warmup = lambda enabled: None
    session.scheduler = OnlineUpdateScheduler(policy=session.policy, replay=session.replay, config=session.config)
    session.status, session.fault_reason = "READY", None
    session.task_instruction, session.fps, session.state_feature_names = "test", 30, None
    session.rl_token_policy = SimpleNamespace(_pi05=None, rl_token=None)
    session.preprocessor = session.capture = None
    session._episode_warmup = True
    session._checkpoint_scheduler_progress()
    return session


def save_episode(session):
    root = session.dataset.root
    writer = V30DatasetWriter(
        root, 30, "test", ("observation.images.camera",), 2, 2,
        state_fields=("x", "y"), action_fields=("x", "y"),
        extra_features={**EXTRA_FEATURES, **TIMING_FEATURES},
        info_metadata={"recording_schema_version": 5, "run_uuid": session.run["run_uuid"], "dataset_uuid": session.run["dataset_uuid"]},
    )
    video = writer.video_path(0, "observation.images.camera")
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"fake video: never decoded")
    rows = timed_rows(3)
    features = {}
    for key in {**EXTRA_FEATURES, **TIMING_FEATURES}:
        name = key.removeprefix("complementary_info.")
        features[key] = np.array([[row[name]] for row in rows.values()])
    image_stats = {name: np.ones((3, 1, 1)) for name in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")}
    image_stats["count"] = np.array([3])
    writer.save_episode(
        np.zeros((3, 2)), np.array([[1, 2], [3, 4], [1000, 1000]]),
        {"observation.images.camera": image_stats}, frame_features=features,
        extra_episode_metadata={
            "run_uuid": session.run["run_uuid"], "dataset_uuid": session.run["dataset_uuid"],
            "episode_uuid": str(uuid.uuid4()), "episode_success": "success",
            "online_warmup": True, "policy_revision": session.replay.active_checkpoint,
        },
    )
    return writer


def mock_encoding(monkeypatch, session):
    session._fresh_dataset = lambda: SimpleNamespace(meta=SimpleNamespace(episodes=pd.DataFrame(session._saved_rows())))
    monkeypatch.setattr("vla.core.online_training.encode_saved_episode", lambda **kwargs: ([ReplayTransitionRecord(0, _transition())], 2))


def test_saved_episode_is_durable_outbox_and_endpoints_do_not_pollute_stats(tmp_path, monkeypatch):
    session = session_at(tmp_path)
    try:
        writer = save_episode(session)
        mock_encoding(monkeypatch, session)
        assert len(session.replay) == 0
        repair_committed_metadata(session.dataset.root)
        stats = json.loads((session.dataset.root / "meta/stats.json").read_text())
        assert stats["action"]["max"] == [3.0, 4.0]
        assert session.recover()["online_status"] == "READY"
        assert len(session.replay) == 1
        assert (session.scheduler.critic_updates, session.scheduler.actor_updates) == (20, 10)
        result = session.process_saved_episode(episode_id=0, outcome="success")
        assert result["duplicate"]
        assert session.replay.produced_credits == 20
        assert writer.frame_count == 3
    finally:
        session.replay.close()


def test_actor_failure_locks_session_and_recovery_matches_baseline(tmp_path, monkeypatch):
    session = session_at(tmp_path)
    try:
        save_episode(session)
        mock_encoding(monkeypatch, session)
        original_actor = session.policy._actor_loss_without_critic_grads
        def fail(_batch):
            raise RuntimeError("injected actor failure")
        session.policy._actor_loss_without_critic_grads = fail
        with pytest.raises(RuntimeError, match="injected"):
            session.process_saved_episode(episode_id=0, outcome="success")
        assert session.status == "FAULTED"
        assert not session.policy.training
        assert session.scheduler.consumed_credits == 1
        assert session.scheduler.critic_updates == 1
        assert session.scheduler.actor_updates == 0
        with pytest.raises(RuntimeError, match="FAULTED"):
            session.process_saved_episode(episode_id=0, outcome="success")
        with pytest.raises(RuntimeError, match="faulted"):
            session.scheduler.drain()
        assert session.recover()["online_status"] == "FAULTED"
        session.policy._actor_loss_without_critic_grads = original_actor
        baseline_checkpoint = session.replay.active_checkpoint
        assert session.recover()["online_status"] == "READY"
        recovered = copy.deepcopy(session._model_state())
        assert (session.scheduler.critic_updates, session.scheduler.actor_updates) == (20, 10)
        # Build the uninterrupted baseline from the same durable pre-training state.
        session.replay._set_state("consumed_credits", "0")
        session.replay._set_state("critic_updates", "0")
        session.replay._set_state("actor_updates", "0")
        session.replay._set_state("actor_enabled", "0")
        session.replay._connection.commit()
        session._restore_checkpoint(baseline_checkpoint)
        session.scheduler.drain()
        for key, tensor in recovered.items():
            torch.testing.assert_close(tensor, session._model_state()[key], rtol=0, atol=0)
    finally:
        session.replay.close()


def test_changed_dataset_latches_recovery(tmp_path, monkeypatch):
    session = session_at(tmp_path)
    try:
        writer = save_episode(session)
        mock_encoding(monkeypatch, session)
        writer.video_path(0, "observation.images.camera").write_bytes(b"changed")
        assert session.recover()["online_status"] == "FAULTED"
        assert "changed file" in session.fault_reason
        assert len(session.replay) == 0
    finally:
        session.replay.close()






@pytest.mark.parametrize("fail_save", [1, 2])
def test_checkpoint_failure_before_or_after_replay_commit_is_recoverable(tmp_path, monkeypatch, fail_save):
    session = session_at(tmp_path)
    try:
        save_episode(session)
        mock_encoding(monkeypatch, session)
        original = session._save_checkpoint
        calls = 0
        def fail(**kwargs):
            nonlocal calls
            calls += 1
            if calls == fail_save:
                raise OSError("injected checkpoint disk failure")
            return original(**kwargs)
        session._save_checkpoint = fail
        with pytest.raises(OSError, match="injected"):
            session.process_saved_episode(episode_id=0, outcome="success")
        assert session.status == "FAULTED"
        assert len(session.replay) == int(fail_save == 2)
        session._save_checkpoint = original
        assert session.recover()["online_status"] == "READY"
        assert (session.replay.produced_credits, session.replay.critic_updates, session.replay.actor_updates) == (20, 20, 10)
    finally:
        session.replay.close()


def test_encoding_failure_then_retry_reconciles_dataset(tmp_path, monkeypatch):
    session = session_at(tmp_path)
    try:
        save_episode(session)
        mock_encoding(monkeypatch, session)
        def fail(**kwargs):
            raise RuntimeError("injected encoder failure")
        monkeypatch.setattr("vla.core.online_training.encode_saved_episode", fail)
        with pytest.raises(RuntimeError, match="encoder"):
            session.process_saved_episode(episode_id=0, outcome="success")
        assert len(session.replay) == 0
        mock_encoding(monkeypatch, session)
        assert session.recover()["online_status"] == "READY"
        assert session.replay.produced_credits == 20
    finally:
        session.replay.close()


def test_committed_replay_survives_lost_in_memory_mirror(tmp_path, monkeypatch):
    session = session_at(tmp_path)
    try:
        save_episode(session)
        mock_encoding(monkeypatch, session)
        original = session.replay.commit_episode
        def fail(**kwargs):
            original(**kwargs)
            session.replay._entries.clear()
            raise RuntimeError("failure after durable commit")
        session.replay.commit_episode = fail
        with pytest.raises(RuntimeError, match="durable"):
            session.process_saved_episode(episode_id=0, outcome="success")
        session.replay.commit_episode = original
        assert session.recover()["online_status"] == "READY"
        assert len(session.replay) == 1
        assert session.replay.actor_updates == 10
    finally:
        session.replay.close()


def test_changed_episode_metadata_and_missing_dataset_block_resume(tmp_path, monkeypatch):
    session = session_at(tmp_path)
    try:
        save_episode(session)
        mock_encoding(monkeypatch, session)
        session.recover()
        from vla.datacollection.franka_robotiq_single_left.v30_writer import EPISODES_PATH
        path = session.dataset.root / EPISODES_PATH
        rows = pd.read_parquet(path)
        rows.loc[0, "episode_success"] = "failure"
        rows.to_parquet(path, index=False)
        assert session.recover()["online_status"] == "FAULTED"
        assert "metadata changed" in session.fault_reason
        path.rename(path.with_suffix(".backup"))
        assert session.recover()["online_status"] == "FAULTED"
        assert "missing Dataset" in session.fault_reason
    finally:
        session.replay.close()


def test_server_refuses_inference_while_training_faulted(tmp_path):
    from vla.cli.serve_franka_rollout import FrankaRolloutPolicyServer
    session = session_at(tmp_path)
    try:
        session.status = "FAULTED"
        session.fault_reason = "injected"
        server = object.__new__(FrankaRolloutPolicyServer)
        server.online_session = session
        with pytest.raises(RuntimeError, match="FAULTED"):
            server.handle({"kind": "plan"}, {})
        with pytest.raises(ValueError, match="run_uuid"):
            server.handle({"kind": "recover_online", "run_uuid": "other"}, {})
    finally:
        session.replay.close()




@pytest.mark.parametrize('action_stride', [1, 2, 4])
def test_timed_encoder_assembles_actions_without_crossing_endpoints(action_stride, tmp_path):
    from tests.rlt.test_execution_learning import recorded_plan, contract
    from vla.core.online_training import encode_saved_episode
    from vla.core.online_contract import TIMING_FIELDS, durable_json, file_digest
    rows, events = recorded_plan()
    plan = events[0][1]
    plan['request']['frame_stride'] = action_stride
    expanded = []
    for i in range(10*action_stride+1):
        row = dict(rows[min(i//action_stride, 10)])
        row.update(slot_id=i, action_index=i//action_stride,
                   observed_at_ns=1_000_000_000+i*33_333_333,
                   command_at_ns=1_000_100_000+i*33_333_333)
        expanded.append(row)
    raw = [{**{'complementary_info.'+k: [row.get(k, 0)] for k in (*TIMING_FIELDS, 'collector_policy_id', 'phase', 'is_intervention')},
            'action': row['action'], 'observation.anchor': row['anchor']} for row in expanded]
    audit = tmp_path/'execution'/'episode-000000'
    durable_json(audit/'events.json', {'schema_version': 5, 'events': events})
    durable_json(audit/'manifest.json', {'schema_version': 5, 'episode_index': 0,
                 'files': {'events.json': file_digest(audit/'events.json')}})
    dataset = SimpleNamespace(root=tmp_path, meta=SimpleNamespace(root=tmp_path), hf_dataset=raw)
    records, steps = encode_saved_episode(
        dataset=dataset, episode_id=0, episode_start=0, episode_stop=len(raw),
        episode_success=True, warmup=False, pi05=None, rl_token=None, preprocessor=None,
        capture=None, chunk_length=10, stride=1, action_dim=8, proprio_dim=2,
        batch_size=1, num_workers=0, device='cpu', task_instruction=None,
        require_timing=True, action_stride=action_stride, execution_contract=contract())
    item, = [record.transition for record in records]
    assert item['prefix_mask'].sum() == item['actor_mask'].sum() == 5
    assert item['interval_slot_ids'].tolist() == list(range(10*action_stride))
    assert item['reward_seq'].sum() == 1 and item['done'] == 1
    assert item['reward_offsets'][-1] == pytest.approx(10*action_stride-1, abs=1e-5)
    np.testing.assert_allclose(item['absolute_exec_chunk'][:5, :7], 0)
    np.testing.assert_allclose(item['absolute_exec_chunk'][5:, :7], .2)


def test_client_restarts_failed_worker_only_after_server_recovery(monkeypatch):
    import queue
    import threading
    from vla.datacollection.franka_robotiq_rollout.policy_client import AsyncPolicyClient

    client = object.__new__(AsyncPolicyClient)
    client._lock = threading.Lock()
    client._error = RuntimeError("dead inference worker")
    client._pending = False
    client._requests = queue.Queue(1)
    client._results = queue.Queue()
    client._next_request_id = 5
    client._min_valid_request_id = 0
    client.timeout_s = 1
    client._closed = False
    starts = []
    class Thread:
        def __init__(self, **kwargs):
            pass
        def join(self, timeout):
            pass
        def is_alive(self):
            return False
        def start(self):
            starts.append("worker")
    client._thread = Thread()
    monkeypatch.setattr("vla.datacollection.franka_robotiq_rollout.policy_client.threading.Thread", Thread)
    def fail(kind):
        raise RuntimeError("server remains faulted")
    client._online_request = fail
    with pytest.raises(RuntimeError, match="server remains"):
        client.recover_online()
    assert client._error is not None and not starts
    client._online_request = lambda kind: {"online_status": "READY"}
    client.recover_online()
    assert client._error is None and starts == ["worker"]
    assert client._min_valid_request_id == 5


def test_recovery_progress_uses_one_line():
    import io
    from vla.datacollection.franka_robotiq_rollout.policy_client import _RemoteTrainingProgress
    stream = io.StringIO()
    progress = _RemoteTrainingProgress(stream=stream)
    progress.update({"stage": "encoding", "episode_id": 0})
    progress.update({"completed_updates": 0, "total_updates": 5, "actor_updates": 0})
    progress.update({"completed_updates": 5, "total_updates": 5, "actor_updates": 2})
    progress.update({"stage": "checkpoint"})
    progress.close()
    assert stream.getvalue().count("\n") == 1
    assert stream.getvalue().count("\r") == 4


def test_warmup_and_initialization_contract_is_readable_without_models(tmp_path):
    from vla.adapters.lerobot.policies.configuration_rlt_ac import ChunkACConfig
    from dataclasses import asdict
    config = ChunkACConfig()
    run = bind_run(tmp_path / "dataset", {"policy": asdict(config)}, resume=False)
    assert bind_run(tmp_path / "dataset", {"policy": asdict(config)}, resume=True) == run


@pytest.mark.parametrize("inherit", [False, True])
def test_online_inheritance_cli_is_explicit(monkeypatch, inherit):
    import sys
    from vla.cli.serve_franka_rollout import parse_args

    argv = ["serve", "--policy-path", "/unused/ac", "--socket-path", "/unused/socket", "--task", "test"]
    if inherit:
        argv.append("--online-inherit-actor-critic")
    monkeypatch.setattr(sys, "argv", argv)
    assert parse_args().online_inherit_actor_critic is inherit


@pytest.mark.parametrize("inherit,full_episode", [(False, False), (True, False), (True, True)])
def test_full_session_start_and_resume_without_gpu_or_robot(tmp_path, monkeypatch, inherit, full_episode):
    from dataclasses import dataclass, replace
    from vla.adapters.lerobot.policies import modeling_rlt_ac, processor_rlt_token
    import vla.core.online_training as training

    model = tmp_path / "frozen_model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"test frozen weights")

    @dataclass
    class Config:
        device: str = "cpu"
        chunk_length: int = 4
        actor_update_interval: int = 2
        action_dim: int = 2
        proprio_dim: int = 2
        rl_token_dim: int = 3
        gamma: float = 0.99
        prefix_bc_weight: float = 0.3
        beta: float = 0.3
        tau: float = 0.005
        target_q_clip: float = 100
        rl_token_pretrained_path: str = str(model)
        vla_pretrained_path: str = str(model)
        actor_hidden_dim: int = 16
        actor_num_layers: int = 1
        actor_fixed_std: float = 0.1
        actor_ref_dropout_p: float = 0.0
        actor_activation: str = "relu"
        actor_layer_norm: bool = False
        actor_residual: bool = False
        critic_hidden_dim: int = 16
        critic_num_layers: int = 1
        critic_activation: str = "relu"
        critic_layer_norm: bool = False
        critic_residual: bool = False
        token_pool_size: int = 1
        image_only: bool = False
        num_per_camera: int = 1
        active_camera_indices: tuple = (0,)

    class Policy(_Policy):
        def __init__(self):
            super().__init__()
            self.config = Config()
        def _ensure_rl_token_policy(self):
            return SimpleNamespace(config=None, _pi05=None)
        def _compute_num_image_tokens(self):
            return 1
        def quiesce_for_online_training(self):
            pass
        def set_online_warmup(self, enabled):
            self.online_warmup = enabled
        def reset(self):
            pass

    class Capture:
        def __init__(self, **kwargs):
            pass
        def attach(self, policy):
            pass
        def detach(self):
            pass

    monkeypatch.setattr(modeling_rlt_ac, "ChunkACPolicy", Policy)
    monkeypatch.setattr(processor_rlt_token, "make_rlt_token_pre_post_processors", lambda **kwargs: (None, None))
    monkeypatch.setattr(training, "PrefixOutputCapture", Capture)
    config = OnlineRLTTrainingConfig(replay_root=str(tmp_path / "replay"), inherit_actor_critic=inherit,
                                    full_episode=full_episode, warmup_steps=0 if full_episode else 1000)
    dataset = SimpleNamespace(root=tmp_path / "dataset", repo_id="local/test")
    runtime = {"checkpoint_fingerprint": "source-checkpoint"}
    def start(policy=None, **kwargs):
        return OnlineRLTSession(
            policy=policy if policy is not None else Policy(), dataset=dataset,
            fps=30, task_instruction="test", config=kwargs.pop("config", config),
            runtime_contract=kwargs.pop("runtime_contract", runtime), **kwargs,
        )
    source = Policy()
    source._critic_step.fill_(50000)
    original = {name: copy.deepcopy(getattr(source, name).state_dict()) for name in ("actor", "critic", "target_critic")}
    if inherit:
        with pytest.raises(ValueError, match="fingerprint"):
            start(resume=False, runtime_contract={})
        assert not dataset.root.exists()
        invalid = Policy()
        with torch.no_grad():
            next(invalid.actor.parameters()).fill_(float("nan"))
        with pytest.raises(ValueError, match="non-finite"):
            start(policy=invalid, resume=False)
        assert not dataset.root.exists()
    first = start(policy=source, resume=False)
    assert first.status == "READY"
    assert first.policy.online_warmup is not full_episode
    assert first.is_warmup is not full_episode
    assert first.policy._critic_step.item() == 0
    assert first.scheduler.actor_optimizer.state == {}
    assert first.scheduler.critic_optimizer.state == {}
    if inherit:
        for name, state in original.items():
            for key, value in state.items():
                torch.testing.assert_close(value, getattr(first.policy, name).state_dict()[key], rtol=0, atol=0)
        assert all(not parameter.requires_grad for parameter in first.policy.target_critic.parameters())
    else:
        assert "inherit_actor_critic" not in first.run["contract"]["training"]
        assert not torch.equal(next(iter(original["actor"].values())), next(first.policy.actor.parameters()))
    identity = first.begin_episode()
    assert identity["online_warmup"] is not full_episode
    assert identity["policy_revision"]
    # Save online-updated weights and verify resume does not reseed from offline weights.
    with torch.no_grad():
        next(first.policy.actor.parameters()).add_(0.25)
    first._checkpoint_scheduler_progress()
    before = copy.deepcopy(first._model_state())
    first.close()
    resumed = start(resume=True)
    assert resumed.status == "READY"
    for key, value in before.items():
        torch.testing.assert_close(value, resumed._model_state()[key], rtol=0, atol=0)
    resumed.close()
    with pytest.raises(ValueError, match="training|full-episode"):
        start(config=replace(config, inherit_actor_critic=not inherit), resume=True)
    with pytest.raises(ValueError, match="training"):
        start(config=replace(config, full_episode=not full_episode, inherit_actor_critic=True,
                             warmup_steps=1000 if full_episode else 0), resume=True)
    with pytest.raises(ValueError, match="runtime"):
        start(runtime_contract={"checkpoint_fingerprint": "different-source"}, resume=True)
    with pytest.raises(ValueError, match="training|full-episode"):
        start(config=replace(config, warmup_steps=2000), resume=True)
    (model / "weights.bin").write_bytes(b"different model at same path")
    with pytest.raises(ValueError, match="sha256"):
        start(resume=True)
