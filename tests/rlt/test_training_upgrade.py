from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from vla.cli.upgrade_rlt_training import _tree_digest, upgrade
from vla.core.online_contract import RUN_MANIFEST, durable_json, file_digest, load_run, model_digest
from vla.core.online_replay import PersistentReplayBuffer, ReplayTransitionRecord
from vla.core.training_contract import REFERENCE_POLICY, validate_cache_upgrade, validate_training_cache
from tests.rlt.test_online_consistency import session_at, save_episode
from tests.rlt.test_online_training import _transition


def legacy_run(tmp_path, *, commit=True, trained=False):
    session = session_at(tmp_path)
    weights = tmp_path / "frozen_model"
    weights.mkdir()
    (weights / "weights.bin").write_bytes(b"unchanged frozen model")
    session.run["contract"] = {
        "chunk_length": 4, "stride": 2, "fps": 30, "utd_ratio": 5,
        "training": {"batch_size": 1, "warmup_steps": 2},
        "rl_token_pretrained_path": str(weights), "vla_pretrained_path": str(weights),
        "rl_token_sha256": model_digest(weights), "vla_sha256": model_digest(weights),
    }
    durable_json(session.dataset.root / RUN_MANIFEST, session.run)
    session.replay._set_state("contract", json.dumps(session.run, sort_keys=True, separators=(",", ":")))
    session.replay._connection.commit()
    save_episode(session)
    row = session._saved_rows()[0]
    if commit:
        transition = _transition()
        transition["actual_steps"] = torch.tensor(2)
        transition["done"] = torch.tensor(1.)
        session.replay.commit_episode(episode_key=session._episode_key(row), episode_id=0,
            outcome="success", warmup=True, warmup_env_steps=2,
            records=[ReplayTransitionRecord(0, transition)], utd_ratio=5,
            checkpoint=session.replay.active_checkpoint, episode_digest=session._episode_digest(row))
    if trained:
        session._drain_pending()
    checkpoint = session.root / "checkpoints" / session.replay.active_checkpoint
    state = torch.load(checkpoint / "training_state.pt", weights_only=False)
    state.pop("training_semantics_version")
    state.pop("actor_enabled")
    torch.save(state, checkpoint / "training_state.pt")
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    metadata["sha256"]["training_state.pt"] = file_digest(checkpoint / "training_state.pt")
    durable_json(checkpoint / "metadata.json", metadata)
    session.replay._set_state("format_version", "2")
    session.replay._connection.execute("DELETE FROM state WHERE key IN ('trainable_env_steps', 'actor_enabled')")
    session.replay._connection.commit()
    session.close()
    return session


def test_upgrade_preserves_sources_weights_and_pending_credits(tmp_path):
    session = legacy_run(tmp_path)
    sources = (_tree_digest(session.dataset.root), _tree_digest(session.root))
    output = tmp_path / "upgraded"
    report = upgrade(session.dataset.root, session.root, output, dry_run=True)
    assert report["trainable_env_steps"] == 2 and not report["actor_enabled"]
    assert not output.exists()
    assert sources == (_tree_digest(session.dataset.root), _tree_digest(session.root))
    report = upgrade(session.dataset.root, session.root, output)
    assert report["complete"]
    assert sources == (_tree_digest(session.dataset.root), _tree_digest(session.root))
    run = load_run(output / "dataset")
    assert run["dataset_uuid"] == session.run["dataset_uuid"]
    old_replay_root = session.root
    session.root, session.dataset.root, session.run = output / "replay", output / "dataset", run
    session.replay = PersistentReplayBuffer(session.root, capacity=20, contract=run, resume=True, seed=7)
    session.scheduler.replay = session.replay
    try:
        active = session.replay.active_checkpoint
        assert (old_replay_root / "checkpoints" / active / "model.safetensors").read_bytes() == (
            session.root / "checkpoints" / active / "model.safetensors").read_bytes()
        assert session.replay.produced_credits == 5 and session.replay.consumed_credits == 0
        assert session.recover()["online_status"] == "READY"
        assert not session.is_warmup
        assert session.replay.consumed_credits == 5 and session.replay.actor_updates == 2
        assert session.begin_episode()["dataset_uuid"] == run["dataset_uuid"]
    finally:
        session.close()


def test_upgrade_rejects_missing_episode_before_any_output(tmp_path):
    session = legacy_run(tmp_path, commit=False)
    with pytest.raises(ValueError, match="missing from Replay"):
        upgrade(session.dataset.root, session.root, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_upgrade_publish_failure_keeps_source_and_blocks_staging_resume(tmp_path, monkeypatch):
    session = legacy_run(tmp_path)
    before = (_tree_digest(session.dataset.root), _tree_digest(session.root))
    def fail(*args):
        raise OSError("injected publish failure")
    monkeypatch.setattr("vla.cli.upgrade_rlt_training.os.rename", fail)
    with pytest.raises(RuntimeError, match="not published"):
        upgrade(session.dataset.root, session.root, tmp_path / "output")
    assert before == (_tree_digest(session.dataset.root), _tree_digest(session.root))
    assert not (tmp_path / "output").exists()
    staged = list(tmp_path.glob(".output-upgrade-*"))
    assert len(staged) == 1
    with pytest.raises(ValueError, match="unpublished"):
        load_run(staged[0] / "dataset")


def test_old_replay_refuses_implicit_upgrade(tmp_path):
    session = legacy_run(tmp_path)
    before = _tree_digest(session.root)
    with pytest.raises(ValueError, match="upgrade_rlt_training"):
        PersistentReplayBuffer(session.root, capacity=20, contract=session.run, resume=True, seed=7)
    assert _tree_digest(session.root) == before


def test_upgrade_preserves_trained_optimizer_moments_and_actor_mode(tmp_path):
    session = legacy_run(tmp_path, trained=True)
    import sqlite3
    with sqlite3.connect(session.root / "replay.sqlite3") as connection:
        active = connection.execute("SELECT value FROM state WHERE key='active_checkpoint'").fetchone()[0]
    before = torch.load(session.root / "checkpoints" / active / "training_state.pt", weights_only=False)
    assert before["actor_optimizer"]["state"]
    output = tmp_path / "output"
    report = upgrade(session.dataset.root, session.root, output)
    assert report["actor_enabled"]
    after = torch.load(output / "replay" / "checkpoints" / active / "training_state.pt", weights_only=False)
    for key in before:
        if key in {"python_rng_state", "replay_rng_state"}:
            assert before[key] == after[key]
        else:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


@pytest.mark.parametrize("resource", ["model", "checkpoint"])
def test_upgrade_rejects_changed_resources(tmp_path, resource):
    session = legacy_run(tmp_path)
    if resource == "model":
        (tmp_path / "frozen_model" / "weights.bin").write_bytes(b"changed")
    else:
        path = next((session.root / "checkpoints").glob("*/model.safetensors"))
        path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed|checksum"):
        upgrade(session.dataset.root, session.root, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def cache_contract():
    return {"policy": {"model": "same"}, "dataset": {"task": "same"},
        "cache": {"training_semantics_version": 2, "annotation_policy": {"ref_chunk": REFERENCE_POLICY}}}


def test_offline_cache_upgrade_allows_only_supervision_change():
    current = cache_contract()
    previous = copy.deepcopy(current)
    previous["cache"].pop("training_semantics_version")
    previous["cache"]["annotation_policy"]["ref_chunk"] = "checkpoint_vla_proposal"
    with pytest.raises(ValueError, match="rebuild"):
        validate_training_cache(previous)
    validate_cache_upgrade(previous, current)
    changed = copy.deepcopy(current)
    changed["dataset"]["task"] = "different"
    with pytest.raises(ValueError, match="data/model"):
        validate_cache_upgrade(previous, changed)


def test_native_cache_guard_requires_explicit_upgrade_and_records_it(tmp_path, monkeypatch):
    from vla.cli.train_actor_critic_native import install_cache_guard
    from vla.adapters.lerobot.policies.configuration_rlt_ac import ChunkACConfig

    current = cache_contract()
    old = copy.deepcopy(current)
    old["cache"].pop("training_semantics_version")
    old["cache"]["annotation_policy"]["ref_chunk"] = "checkpoint_vla_proposal"
    cache, checkpoint = tmp_path / "new_cache", tmp_path / "old_run" / "pretrained_model"
    durable_json(cache / "cache_manifest.json", {"format_version": 1, "status": "complete", "contract": current})
    durable_json(checkpoint / "config.json", {"cache_contract": old})
    policy = object.__new__(ChunkACConfig)
    policy.cache_contract, policy.pretrained_path = old, checkpoint
    calls = []
    monkeypatch.setattr(ChunkACConfig, "_apply_cache_contract", lambda self: calls.append(self.cache_contract))
    cfg = SimpleNamespace(policy=policy, dataset=SimpleNamespace(repo_id=str(cache)), output_dir=tmp_path / "new_run")
    monkeypatch.setattr("vla.adapters.lerobot.registry.make_train_eval_datasets", lambda cfg: ("dataset", None))
    trainer = SimpleNamespace(make_train_eval_datasets=lambda cfg: "dataset")
    install_cache_guard(trainer)
    with pytest.raises(ValueError, match="accept-training"):
        trainer.make_train_eval_datasets(cfg)
    assert not cfg.output_dir.exists()
    install_cache_guard(trainer, accept_upgrade=True)
    assert trainer.make_train_eval_datasets(cfg) == ("dataset", None)
    assert calls == [current]
    assert policy.cache_contract == current
    assert json.loads((cfg.output_dir / "training_semantics_upgrade.json").read_text())["previous_contract"] == old
    cfg.output_dir = checkpoint.parent
    with pytest.raises(ValueError, match="separate output"):
        trainer.make_train_eval_datasets(cfg)
