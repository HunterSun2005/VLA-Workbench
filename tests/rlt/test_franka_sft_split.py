from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vla.diagnostics.franka_sft_split import (
    BALANCE_FEATURE_NAMES,
    choose_validation_episode_ids,
    ensure_episode_metadata_locations,
    episode_balance_features,
    inherit_vla_metadata,
    load_split_manifest,
    split_manifest_digest,
    training_schedule,
    validate_split_dataset_contract,
)


def _manifest(tmp_path: Path) -> dict:
    return {
        "schema_version": 1,
        "kind": "franka_pi05_sft_split",
        "source": {"repo_id": "local/source", "root": str(tmp_path / "source")},
        "splits": {
            "train": {
                "repo_id": "local/source_train",
                "root": str(tmp_path / "split/train"),
                "original_episode_ids": list(range(131)),
                "total_frames": 1000,
            },
            "val": {
                "repo_id": "local/source_val",
                "root": str(tmp_path / "split/val"),
                "original_episode_ids": list(range(131, 151)),
                "total_frames": 200,
            },
        },
    }


def _actions(length: int = 60) -> np.ndarray:
    actions = np.zeros((length, 10), dtype=np.float64)
    actions[:, 0] = np.linspace(0.4, 0.5, length)
    actions[:, 3] = 1.0
    actions[:, 7] = 1.0
    actions[length // 2 :, 9] = 1.0
    return actions


def test_episode_balance_features_are_finite_and_named() -> None:
    feature, audit = episode_balance_features(_actions())

    assert feature.shape == (len(BALANCE_FEATURE_NAMES),)
    assert np.isfinite(feature).all()
    assert audit["balance_features"]["episode_length"] == 60
    assert audit["balance_features"]["gripper_transition_rate"] > 0


def test_validation_selection_is_deterministic_balanced_and_chronological() -> None:
    episode_ids = list(range(151))
    x = np.linspace(0.0, 1.0, len(episode_ids))
    features = np.stack([x + index / 100 for index in range(len(BALANCE_FEATURE_NAMES))], axis=1)

    first, report = choose_validation_episode_ids(
        episode_ids, features, validation_episodes=20, seed=1000, candidate_count=500
    )
    second, _ = choose_validation_episode_ids(
        episode_ids, features, validation_episodes=20, seed=1000, candidate_count=500
    )

    assert first == second
    assert len(first) == len(set(first)) == 20
    assert report["chronological_bucket_count"] == 10
    for bucket in np.array_split(np.arange(151), 10):
        assert len(set(first) & set(bucket.tolist())) == 2


def test_split_manifest_contract_and_epoch_schedule(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "split_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_split_manifest(path)
    train = validate_split_dataset_contract(
        loaded,
        "train",
        dataset_repo_id="local/source_train",
        dataset_root=tmp_path / "split/train",
    )

    assert train["total_frames"] == 1000
    assert len(split_manifest_digest(loaded)) == 64
    assert training_schedule(loaded) == {
        "train_frames": 1000,
        "batch_size": 2,
        "steps_per_epoch": 500,
        "short_steps": 250,
        "warmup_steps": 25,
        "formal_steps": 1000,
        "save_freq": 250,
    }
    with pytest.raises(ValueError, match="root does not match"):
        validate_split_dataset_contract(
            loaded,
            "train",
            dataset_repo_id="local/source_train",
            dataset_root=tmp_path / "wrong",
        )


def test_legacy_episode_metadata_locations_are_backfilled(tmp_path: Path) -> None:
    metadata_path = tmp_path / "meta/episodes/chunk-002/file-003.parquet"
    metadata_path.parent.mkdir(parents=True)
    pd.DataFrame({"episode_index": [0, 1]}).to_parquet(metadata_path)

    class Meta:
        episodes = {0: {"length": 4}, 1: {"length": 5}}

    class Dataset:
        root = tmp_path
        meta = Meta()

    ensure_episode_metadata_locations(Dataset())

    assert Meta.episodes[0]["meta/episodes/chunk_index"] == 2
    assert Meta.episodes[0]["meta/episodes/file_index"] == 3
    assert Meta.episodes[1]["meta/episodes/chunk_index"] == 2
    assert Meta.episodes[1]["meta/episodes/file_index"] == 3


def test_materialized_split_inherits_absolute_action_metadata(tmp_path: Path) -> None:
    split_root = tmp_path / "train"
    info_path = split_root / "meta/info.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text(json.dumps({"total_frames": 10}), encoding="utf-8")
    source_info = {
        "evo_rlt": {
            "actions_stored_as": "absolute",
            "action_stats": "absolute",
            "action_chunk_size": 50,
            "custom_field": "preserved",
        }
    }

    inherit_vla_metadata(source_info, split_root)

    materialized = json.loads(info_path.read_text(encoding="utf-8"))
    assert materialized["evo_rlt"] == source_info["evo_rlt"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("actions_stored_as", "relative", "actions_stored_as"),
        ("action_stats", "relative", "action_stats"),
        ("action_chunk_size", 32, "action_chunk_size"),
    ],
)
def test_split_rejects_incompatible_action_metadata(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    split_root = tmp_path / "train"
    info_path = split_root / "meta/info.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text("{}", encoding="utf-8")
    metadata = {
        "actions_stored_as": "absolute",
        "action_stats": "absolute",
        "action_chunk_size": 50,
    }
    metadata[field] = value

    with pytest.raises(ValueError, match=message):
        inherit_vla_metadata({"evo_rlt": metadata}, split_root)
