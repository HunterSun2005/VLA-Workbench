from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vla.datacollection.franka_robotiq_single_left import schema
from vla.datacollection.franka_robotiq_single_left.v30_writer import V30DatasetWriter
from vla.cli.validate_rl_token_inputs import validate_rl_token_inputs


CAMERA_KEYS = (
    "observation.images.base",
    "observation.images.left_wrist",
)


def _image_stats(count: int) -> dict[str, np.ndarray]:
    stats = {
        key: np.full((3, 1, 1), 0.5, dtype=np.float32)
        for key in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
    }
    stats["count"] = np.array([count], dtype=np.int64)
    return stats


def _make_dataset(root: Path) -> None:
    # These tests represent an already-selected 29D training dataset, not raw capture.
    writer = V30DatasetWriter(root, 30, "local task", CAMERA_KEYS, 16, 16,
                              state_fields=schema.LEGACY_STATE_FIELDS)
    for key in CAMERA_KEYS:
        path = writer.video_path(0, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"local fixture")
    state_dim = len(schema.LEGACY_STATE_FIELDS)
    states = np.arange(4 * state_dim, dtype=np.float32).reshape(4, state_dim)
    actions = np.arange(4 * 10, dtype=np.float32).reshape(4, 10)
    writer.save_episode(
        states,
        actions,
        {key: _image_stats(len(states)) for key in CAMERA_KEYS},
    )


def _make_sft(
    root: Path,
    tokenizer: Path,
    *,
    state_dim: int = len(schema.LEGACY_STATE_FIELDS),
    relative: bool = False,
) -> None:
    root.mkdir(parents=True)
    tokenizer.mkdir()
    config = {
        "type": "pi05",
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [state_dim]},
            "observation.images.base": {"type": "VISUAL", "shape": [3, 16, 16]},
            "observation.images.left_wrist": {"type": "VISUAL", "shape": [3, 16, 16]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [10]}},
        "chunk_size": 50,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "use_relative_actions": relative,
    }
    (root / "config.json").write_text(json.dumps(config))
    steps = [
        {
            "registry_name": "delta_actions_processor",
            "config": {"enabled": relative},
        },
        {
            "registry_name": "tokenizer_processor",
            "config": {"tokenizer_name": str(tokenizer)},
        },
    ]
    (root / "policy_preprocessor.json").write_text(json.dumps({"steps": steps}))
    (root / "policy_postprocessor.json").write_text(json.dumps({"steps": []}))
    (root / "model.safetensors").write_bytes(b"synthetic-sft-weights")


def test_validate_rl_token_inputs_accepts_matching_local_fixture(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "sft"
    tokenizer = tmp_path / "tokenizer"
    _make_dataset(dataset)
    _make_sft(checkpoint, tokenizer)

    summary = validate_rl_token_inputs(dataset, checkpoint)

    assert summary.state_dim == len(schema.LEGACY_STATE_FIELDS)
    assert summary.action_dim == 10
    assert len(summary.state_names) == len(schema.LEGACY_STATE_FIELDS)
    assert len(summary.action_names) == 10
    assert summary.camera_keys == ("base", "left_wrist")
    assert summary.chunk_size == 50
    assert summary.use_relative_actions is False
    assert summary.action_stats == "absolute"
    assert summary.action_chunk_size == 50
    assert summary.actions_stored_as == "absolute"
    assert len(summary.dataset_info_sha256) == 64
    assert len(summary.dataset_stats_sha256) == 64


def test_validate_rl_token_inputs_rejects_schema_mismatch(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "sft"
    _make_dataset(dataset)
    _make_sft(checkpoint, tmp_path / "tokenizer", state_dim=10)

    with pytest.raises(ValueError, match="state_dim dataset=29, SFT=10"):
        validate_rl_token_inputs(dataset, checkpoint)


def test_validate_rl_token_inputs_rejects_action_stats_mismatch(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "sft"
    _make_dataset(dataset)
    _make_sft(checkpoint, tmp_path / "tokenizer", relative=True)

    with pytest.raises(ValueError, match="action_stats='absolute'.*expected 'relative'"):
        validate_rl_token_inputs(dataset, checkpoint)


def test_validate_rl_token_inputs_rejects_chunk_size_mismatch(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "sft"
    _make_dataset(dataset)
    _make_sft(checkpoint, tmp_path / "tokenizer")
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["evo_rlt"]["action_chunk_size"] = 25
    info_path.write_text(json.dumps(info))
    from vla.dataset_metadata import write_metadata_sidecar
    write_metadata_sidecar(dataset, info)

    with pytest.raises(ValueError, match="action_chunk_size=25.*SFT chunk_size=50"):
        validate_rl_token_inputs(dataset, checkpoint)


def test_validate_rl_token_inputs_rejects_relative_parquet_actions(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "sft"
    _make_dataset(dataset)
    _make_sft(checkpoint, tmp_path / "tokenizer")
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["evo_rlt"]["actions_stored_as"] = "relative"
    info_path.write_text(json.dumps(info))
    from vla.dataset_metadata import write_metadata_sidecar
    write_metadata_sidecar(dataset, info)

    with pytest.raises(ValueError, match="actions_stored_as='relative'.*expects absolute"):
        validate_rl_token_inputs(dataset, checkpoint)


def test_train_rl_token_delegates_to_registered_lerobot(monkeypatch) -> None:
    import vla.adapters.lerobot as adapter
    import lerobot.scripts.lerobot_train as lerobot_train
    from vla.cli import train_rl_token

    calls = []
    monkeypatch.setattr(adapter, "register", lambda: calls.append("register"))
    monkeypatch.setattr(lerobot_train, "main", lambda: calls.append("train"))

    train_rl_token.main()

    assert calls == ["register", "train"]
