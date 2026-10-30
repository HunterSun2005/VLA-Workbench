from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lerobot.configs.policies import PreTrainedConfig

from vla.adapters.lerobot.policies.cache_manifest import atomic_write_json, new_cache_manifest
from vla.adapters.lerobot.policies.configuration_rlt_ac import ChunkACConfig
from vla.adapters.lerobot.policies.dataset_rlt_ac import ChunkTransitionDataset
from vla.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
from vla.adapters.lerobot.checkpoint_contract import sha256_file
from vla.cli.validate_transition_cache import validate_transition_cache


def _contract(tmp_path: Path) -> dict:
    features = {
        "observation.state": {"dtype": "float32", "shape": [10], "names": [f"s{i}" for i in range(10)]},
        "action": {"dtype": "float32", "shape": [10], "names": [f"a{i}" for i in range(10)]},
        "observation.images.base": {"dtype": "video", "shape": [16, 16, 3]},
        "observation.images.left_wrist": {"dtype": "video", "shape": [16, 16, 3]},
    }
    return {
        "dataset": {
            "fps": 20.0,
            "features": features,
            "state_dim": 10,
            "action_dim": 10,
            "camera_keys": ["base", "left_wrist"],
        },
        "policy": {
            "sft_path": str(tmp_path / "sft"),
            "rl_token_path": str(tmp_path / "rl_token"),
            "action_dim": 10,
            "proprio_dim": 10,
            "camera_keys": ["base", "left_wrist"],
            "rl_token_dim": 32,
            "rl_token_num_rl_tokens": 1,
            "token_pool_size": 8,
            "image_only": True,
            "active_camera_indices": [1],
            "num_per_camera": 256,
        },
        "cache": {"chunk_length": 4},
    }


def _transition(value: float) -> dict[str, torch.Tensor]:
    return {
        "state_vec": torch.full((42,), value),
        "exec_chunk": torch.full((4, 10), value),
        "ref_chunk": torch.full((4, 10), value),
        "reward_seq": torch.zeros(4),
        "next_state_vec": torch.full((42,), value + 1),
        "next_ref_chunk": torch.full((4, 10), value + 1),
        "done": torch.tensor(0.0),
        "actual_steps": torch.tensor(4),
    }


def _make_cache(root: Path) -> Path:
    contract = _contract(root)
    manifest = new_cache_manifest(contract)
    manifest["status"] = "complete"
    manifest["shards"]["train"] = [
        {"episode_id": 2, "path": "shards/train/episode_000002.pt", "transitions": 2},
        {"episode_id": 5, "path": "shards/train/episode_000005.pt", "transitions": 1},
    ]
    for shard, values in zip(manifest["shards"]["train"], ([1.0, 2.0], [5.0])):
        path = root / shard["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        transitions = [_transition(value) for value in values]
        transitions[-1]["done"] = torch.tensor(1.0)
        torch.save(transitions, path)
        shard["sha256"] = sha256_file(path)
        shard["episode_success"] = False
    path = root / "cache_manifest.json"
    atomic_write_json(path, manifest)
    return path


def test_chunk_transition_dataset_loads_manifest_shards_lazily(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    dataset = ChunkTransitionDataset(tmp_path)
    assert len(dataset) == 3
    assert dataset.num_episodes == 2
    assert dataset.fps == 20.0
    assert dataset.meta.episodes.dataset_from_index == [0, 2]
    assert dataset.meta.episodes.dataset_to_index == [2, 3]
    assert dataset[0]["state_vec"][0].item() == 1.0
    assert dataset[2]["state_vec"][0].item() == 5.0
    assert dataset[-1]["state_vec"][0].item() == 5.0


def test_chunk_transition_dataset_keeps_all_loaded_shards_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_cache(tmp_path)
    real_load = torch.load
    loaded_paths: list[Path] = []

    def tracking_load(path, *args, **kwargs):
        loaded_paths.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", tracking_load)
    dataset = ChunkTransitionDataset(tmp_path)
    assert dataset[0]["state_vec"][0].item() == 1.0
    assert dataset[2]["state_vec"][0].item() == 5.0
    assert dataset[1]["state_vec"][0].item() == 2.0
    assert len(loaded_paths) == 2


def test_ac_config_is_populated_from_cache_manifest(tmp_path: Path) -> None:
    manifest_path = _make_cache(tmp_path)
    config = ChunkACConfig(cache_manifest_path=str(manifest_path), device="cpu")
    assert config.action_dim == 10
    assert config.proprio_dim == 10
    assert config.camera_keys == ["base", "left_wrist"]
    assert config.chunk_length == 4
    assert config.rl_token_dim == 32
    assert config.active_camera_indices == [1]
    assert config.rl_token_pretrained_path == str(tmp_path / "rl_token")


def test_ac_config_rejects_manifest_schema_override(tmp_path: Path) -> None:
    manifest_path = _make_cache(tmp_path)
    with pytest.raises(ValueError, match="action_dim"):
        ChunkACConfig(
            cache_manifest_path=str(manifest_path),
            action_dim=20,
            device="cpu",
        )


def test_ac_config_allows_relocated_checkpoint_paths(tmp_path: Path) -> None:
    manifest_path = _make_cache(tmp_path)
    config = ChunkACConfig(
        cache_manifest_path=str(manifest_path),
        vla_pretrained_path="/relocated/sft",
        rl_token_pretrained_path="/relocated/rl_token",
        device="cpu",
    )
    assert config.vla_pretrained_path == "/relocated/sft"
    assert config.rl_token_pretrained_path == "/relocated/rl_token"


def test_dataset_rejects_tampered_shard(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    shard = tmp_path / "shards/train/episode_000002.pt"
    shard.write_bytes(shard.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ChunkTransitionDataset(tmp_path)


def test_ac_manifest_accepts_franka_dual_arm_20d_schema(tmp_path: Path) -> None:
    manifest_path = _make_cache(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["contract"]["policy"].update(
        {
            "action_dim": 20,
            "proprio_dim": 20,
            "camera_keys": ["base", "left_wrist", "right_wrist"],
            "active_camera_indices": [1, 2],
        }
    )
    atomic_write_json(manifest_path, manifest)

    config = ChunkACConfig(cache_manifest_path=str(manifest_path), device="cpu")
    assert config.action_dim == 20
    assert config.proprio_dim == 20
    assert config.camera_keys == ["base", "left_wrist", "right_wrist"]
    assert config.active_camera_indices == [1, 2]


def test_ac_rejects_rl_token_camera_contract_mismatch() -> None:
    policy = object.__new__(ChunkACPolicy)
    policy.config = SimpleNamespace(
        action_dim=10,
        proprio_dim=10,
        camera_keys=["base", "left_wrist"],
        rl_token_dim=32,
        rl_token_num_rl_tokens=1,
        token_pool_size=8,
        image_only=True,
        active_camera_indices=[1],
        num_per_camera=256,
        chunk_length=4,
        cache_contract=None,
    )
    rl_token_policy = SimpleNamespace(
        config=SimpleNamespace(
            action_dim=10,
            proprio_dim=10,
            camera_keys=["base", "right_wrist"],
            rl_token_dim=32,
            rl_token_num_rl_tokens=1,
            token_pool_size=8,
            image_only=True,
            active_camera_indices=[1],
            num_per_camera=256,
            chunk_size=50,
        )
    )
    with pytest.raises(ValueError, match="camera_keys"):
        policy._validate_rl_token_arch(rl_token_policy)


def test_ac_initialization_defers_rl_token_backbone_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rl_config = SimpleNamespace(
        action_dim=10,
        proprio_dim=10,
        camera_keys=["base", "left_wrist"],
        rl_token_dim=32,
        rl_token_num_rl_tokens=1,
        token_pool_size=8,
        image_only=True,
        active_camera_indices=[1],
        num_per_camera=256,
        chunk_size=50,
    )
    monkeypatch.setattr(ChunkACPolicy, "_load_rl_token_config", lambda self: rl_config)
    monkeypatch.setattr(
        ChunkACPolicy,
        "_load_rl_token_policy",
        lambda self: pytest.fail("backbone must not load during AC initialization"),
    )
    config = ChunkACConfig(
        rl_token_pretrained_path="unused",
        action_dim=10,
        proprio_dim=10,
        camera_keys=["base", "left_wrist"],
        rl_token_dim=32,
        token_pool_size=8,
        image_only=True,
        active_camera_indices=[1],
        num_per_camera=256,
        device="cpu",
    )

    policy = ChunkACPolicy(config)

    assert policy._rl_token_policy is None
    assert sum(parameter.numel() for parameter in policy.parameters()) < 10_000_000

    backbone = SimpleNamespace()
    loads = []
    monkeypatch.setattr(
        ChunkACPolicy,
        "_load_rl_token_policy",
        lambda self: loads.append(True) or backbone,
    )
    policy.prepare_for_inference()
    policy.prepare_for_inference()
    assert policy._rl_token_policy is backbone
    assert loads == [True]


def test_saved_ac_config_keeps_embedded_contract_when_manifest_moves(tmp_path: Path) -> None:
    manifest_path = _make_cache(tmp_path / "cache")
    config = ChunkACConfig(cache_manifest_path=str(manifest_path), device="cpu")
    checkpoint = tmp_path / "ac_config"
    config.save_pretrained(checkpoint)
    manifest_path.unlink()

    loaded = PreTrainedConfig.from_pretrained(checkpoint)

    assert loaded.cache_contract is not None
    assert loaded.action_dim == 10
    assert loaded.camera_keys == ["base", "left_wrist"]


def test_validate_transition_cache_checks_all_shards(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    summary = validate_transition_cache(tmp_path, verify_checkpoints=False)
    assert summary.state_dim == 42
    assert summary.action_dim == 10
    assert summary.train_transitions == 3
    assert summary.episodes == 2
