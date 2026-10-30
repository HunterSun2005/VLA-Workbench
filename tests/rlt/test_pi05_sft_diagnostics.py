from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from vla.diagnostics.pi05_sft import (
    EXPECTED_ACTION_NAMES,
    checkpoint_label,
    checkpoint_tensor_schema,
    classify_action_window,
    create_manifest,
    decoded_action_errors,
    derive_sample_seed,
    evenly_spaced_anchors,
    load_or_create_manifest,
    aggregate_metric_by_episode,
    paired_bootstrap,
    read_checkpoint_config,
    schedule_diagnostics,
    select_stratified_episode_ids,
    selection_summary,
    summarize_flow,
)


IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _actions(length: int, *, moving: bool = False, gripper_transition: bool = False) -> np.ndarray:
    result = np.zeros((length, 10), dtype=np.float32)
    result[:, 3:9] = IDENTITY_ROT6D
    if moving:
        result[:, 0] = np.arange(length) * 0.001
    if gripper_transition:
        result[5:, 9] = 1.0
    return result


def test_sampling_is_deterministic_stratified_and_has_no_padding() -> None:
    first = select_stratified_episode_ids(list(range(151)), 15, 1000)
    second = select_stratified_episode_ids(list(range(151)), 15, 1000)
    assert first == second
    assert len(first) == len(set(first)) == 15
    for selected, bucket in zip(first, np.array_split(np.arange(151), 15), strict=True):
        assert selected in bucket

    anchors = evenly_spaced_anchors(length=100, chunk_size=50, count=32)
    assert len(anchors) == len(set(anchors)) == 32
    assert anchors[0] == 0
    assert anchors[-1] == 50


def test_action_groups_use_deployment_relevant_thresholds() -> None:
    assert classify_action_window(_actions(50)) == {
        "tcp_group": "tcp_static",
        "gripper_group": "gripper_stable",
    }
    assert classify_action_window(_actions(50, moving=True, gripper_transition=True)) == {
        "tcp_group": "tcp_moving",
        "gripper_group": "gripper_transition",
    }


def test_decoded_metrics_use_physical_position_rotation_and_continuous_gripper() -> None:
    target = _actions(50)[None]
    predicted = target.copy()
    predicted[:, :, 0] += 0.002
    predicted[:, :, 9] += 0.25
    angle = np.deg2rad(10.0)
    rotated = np.array([np.cos(angle), np.sin(angle), 0.0, -np.sin(angle), np.cos(angle), 0.0])
    predicted[:, :, 3:9] = rotated

    errors = decoded_action_errors(predicted, target)

    np.testing.assert_allclose(errors["position_mm"], 2.0)
    np.testing.assert_allclose(errors["rotation_deg"], 10.0, atol=1e-6)
    np.testing.assert_allclose(errors["gripper"], 0.25)
    invalid = predicted.copy()
    invalid[:, :, 3:9] = 0.0
    with pytest.raises(ValueError, match="zero norm"):
        decoded_action_errors(invalid, target)


class _FakeDataset:
    def __init__(self, root: Path) -> None:
        self.actions = np.concatenate((_actions(60), _actions(60, moving=True)))
        episodes = [
            {
                "episode_index": 0,
                "length": 60,
                "dataset_from_index": 0,
                "dataset_to_index": 60,
                "data/chunk_index": 0,
                "data/file_index": 0,
            },
            {
                "episode_index": 1,
                "length": 60,
                "dataset_from_index": 60,
                "dataset_to_index": 120,
                "data/chunk_index": 0,
                "data/file_index": 1,
            },
        ]
        self.meta = SimpleNamespace(
            episodes=episodes,
            info={"codebase_version": "v3.0"},
            total_episodes=2,
            total_frames=120,
            fps=30,
            features={"action": {"shape": [10], "names": list(EXPECTED_ACTION_NAMES)}},
            camera_keys=[],
            data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        )
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text("{}\n", encoding="utf-8")
        (root / "data" / "chunk-000").mkdir(parents=True)
        (root / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"episode-0")
        (root / "data" / "chunk-000" / "file-001.parquet").write_bytes(b"episode-1")

    def get_raw_item(self, index: int) -> dict:
        return {"action": self.actions[index]}


def test_manifest_is_reused_and_rejects_dataset_changes(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset = _FakeDataset(dataset_root)
    manifest_path = tmp_path / "manifest.json"
    kwargs = dict(
        dataset_repo_id="local/test",
        dataset_root=dataset_root,
        episode_count=2,
        anchors_per_episode=2,
        flow_repeats=4,
        inference_repeats=1,
        seed=1000,
    )

    created = load_or_create_manifest(manifest_path, dataset, **kwargs)
    reused = load_or_create_manifest(manifest_path, dataset, **kwargs)

    assert created == reused
    assert created["diagnostic_seen_data"] is True
    assert len(created["samples"]) == 4
    assert all(sample["frame_index"] <= 10 for sample in created["samples"])
    (dataset_root / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="fingerprint changed"):
        load_or_create_manifest(manifest_path, dataset, **kwargs)


def test_holdout_manifest_uses_all_episodes_and_separate_transition_anchors(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "val"
    dataset = _FakeDataset(dataset_root)
    dataset.actions[:60] = _actions(60, gripper_transition=True)
    dataset.actions[60:] = _actions(60, moving=True, gripper_transition=True)
    split_manifest = {
        "schema_version": 1,
        "kind": "franka_pi05_sft_split",
        "source": {"repo_id": "local/source", "root": str(tmp_path / "source")},
        "splits": {
            "train": {
                "repo_id": "local/train",
                "root": str(tmp_path / "train"),
                "original_episode_ids": [0, 1],
            },
            "val": {
                "repo_id": "local/val",
                "root": str(dataset_root),
                "original_episode_ids": [10, 11],
                "original_to_split_episode": {"10": 0, "11": 1},
            },
        },
    }

    manifest = create_manifest(
        dataset,
        dataset_repo_id="local/val",
        dataset_root=dataset_root,
        episode_count=2,
        anchors_per_episode=2,
        flow_repeats=1,
        inference_repeats=1,
        seed=1000,
        dataset_role="val",
        split_manifest=split_manifest,
        transition_anchors_per_episode=1,
    )

    assert manifest["schema_version"] == 2
    assert manifest["diagnostic_seen_data"] is False
    assert manifest["sampling"]["uniform_sample_count"] == 4
    assert manifest["sampling"]["transition_sample_count"] == 2
    assert {sample["original_episode_index"] for sample in manifest["samples"]} == {10, 11}


def test_episode_aggregation_and_holdout_composite_selection() -> None:
    samples = [
        {"episode_index": 0},
        {"episode_index": 0},
        {"episode_index": 1},
        {"episode_index": 1},
    ]
    ids, values = aggregate_metric_by_episode(np.array([1.0, 3.0, 2.0, 4.0]), samples)
    np.testing.assert_array_equal(ids, [0, 1])
    np.testing.assert_allclose(values, [2.0, 3.0])

    baseline = {
        "flow_full_50": np.array([2.0, 2.0]),
        "flow_first_10": np.array([2.0, 2.0]),
        "position_rmse_mm": np.array([2.0, 2.0]),
        "rotation_mean_deg": np.array([2.0, 2.0]),
        "gripper_mae": np.array([2.0, 2.0]),
        "gripper_transition_mae": np.array([2.0, 2.0]),
    }
    improved = {key: value * 0.8 for key, value in baseline.items()}
    selection = selection_summary(
        [Path("baseline/pretrained_model"), Path("improved/pretrained_model")],
        [baseline, improved],
    )
    assert selection["recommended_checkpoint"] == "improved"
    assert selection["conclusion"] == "holdout_composite_score"


def test_flow_summary_pairing_selection_and_seed_stability() -> None:
    samples = [
        {"tcp_group": "tcp_static", "gripper_group": "gripper_stable"},
        {"tcp_group": "tcp_moving", "gripper_group": "gripper_transition"},
    ]
    losses = np.ones((2, 50, 10), dtype=np.float64)
    summary = summarize_flow(losses, samples)
    assert summary["full_50"]["overall"]["mean"] == 1.0
    assert summary["slices_full_50"]["tcp_static"]["count"] == 1

    assert derive_sample_seed(1000, 1, 2, 3) == derive_sample_seed(1000, 1, 2, 3)
    paired = paired_bootstrap(np.array([2.0, 2.0]), np.array([1.0, 1.0]), seed=1000, resamples=100)
    assert paired["mean_delta_right_minus_left"] == -1.0
    assert paired["right_win_rate"] == 1.0

    metric_names = (
        "flow_full_50",
        "flow_first_10",
        "position_rmse_mm",
        "rotation_mean_deg",
        "gripper_mae",
    )
    worse = {name: np.array([2.0, 2.0]) for name in metric_names}
    better = {name: np.array([1.0, 1.0]) for name in metric_names}
    selection = selection_summary(
        [Path("040000/pretrained_model"), Path("080000/pretrained_model")],
        [worse, better],
    )
    assert selection["recommended_checkpoint"] == "080000"
    assert selection["conclusion"] == "dominates_all_primary_metrics"


def test_checkpoint_label_disambiguates_runs_at_the_same_step(tmp_path: Path) -> None:
    screen_a = tmp_path / "screen_a" / "checkpoints" / "025774" / "pretrained_model"
    screen_b = tmp_path / "screen_b" / "checkpoints" / "025774" / "pretrained_model"

    assert checkpoint_label(screen_a) == "screen_a@025774"
    assert checkpoint_label(screen_b) == "screen_b@025774"
    assert checkpoint_label(tmp_path / "baseline" / "pretrained_model") == "baseline"


def _make_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    config = {
        "type": "pi05",
        "action_feature_names": list(EXPECTED_ACTION_NAMES),
        "output_features": {"action": {"shape": [10]}},
        "chunk_size": 50,
        "use_relative_actions": False,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(
        {
            "model.action_in_proj.weight": torch.zeros(2, 2),
            "model.action_out_proj.weight": torch.zeros(2, 2),
            "model.paligemma_with_expert.gemma_expert.lm_head.weight": torch.zeros(2, 2),
        },
        path / "model.safetensors",
    )
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (path / name).write_bytes(b"placeholder")


def test_checkpoint_contract_and_scheduler_warning(tmp_path: Path) -> None:
    checkpoint = tmp_path / "100000" / "pretrained_model"
    _make_checkpoint(checkpoint)
    assert read_checkpoint_config(checkpoint)["type"] == "pi05"
    assert len(checkpoint_tensor_schema(checkpoint)) == 3
    train_config = {
        "batch_size": 2,
        "steps": 100000,
        "scheduler": {"num_warmup_steps": 1000, "num_decay_steps": 30000},
    }
    (checkpoint / "train_config.json").write_text(json.dumps(train_config), encoding="utf-8")

    schedule = schedule_diagnostics(checkpoint, 118518)

    assert schedule["steps_per_epoch"] == 59259
    assert schedule["lr_floor_epoch"] == pytest.approx(30000 / 59259)
    assert schedule["warnings"] == ["learning rate reaches its floor before one dataset epoch"]
