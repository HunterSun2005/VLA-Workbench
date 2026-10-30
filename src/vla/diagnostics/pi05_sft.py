from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


log = logging.getLogger(__name__)


SCHEMA_VERSION = 1
HOLDOUT_SCHEMA_VERSION = 2
EXPECTED_ACTION_NAMES = (
    "desired_tcp_position_x",
    "desired_tcp_position_y",
    "desired_tcp_position_z",
    "desired_tcp_rot6d_col0_x",
    "desired_tcp_rot6d_col0_y",
    "desired_tcp_rot6d_col0_z",
    "desired_tcp_rot6d_col1_x",
    "desired_tcp_rot6d_col1_y",
    "desired_tcp_rot6d_col1_z",
    "gripper",
)
PRIMARY_HORIZON = 10
FULL_HORIZON = 50
POSITION_MOTION_THRESHOLD_M = 8e-4
ROTATION_MOTION_THRESHOLD_RAD = 1e-2
GRIPPER_TRANSITION_THRESHOLD = 1e-2


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, default=_json_default)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(root: Path, relative_paths: Iterable[Path]) -> dict[str, Any]:
    root = Path(root).resolve()
    entries: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    for relative in sorted({Path(path) for path in relative_paths}, key=lambda path: path.as_posix()):
        absolute = root / relative
        if not absolute.is_file():
            raise FileNotFoundError(f"dataset fingerprint input is missing: {absolute}")
        file_digest = sha256_file(absolute)
        size = absolute.stat().st_size
        entry = {"path": relative.as_posix(), "size": size, "sha256": file_digest}
        entries.append(entry)
        combined.update(relative.as_posix().encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(size).encode("ascii"))
        combined.update(b"\0")
        combined.update(file_digest.encode("ascii"))
        combined.update(b"\n")
    return {"sha256": combined.hexdigest(), "files": entries}


def select_stratified_episode_ids(
    episode_ids: Sequence[int], count: int, seed: int
) -> list[int]:
    ids = np.asarray(sorted(int(value) for value in episode_ids), dtype=np.int64)
    if count <= 0:
        raise ValueError("episode_count must be positive")
    if count > len(ids):
        raise ValueError(f"cannot select {count} episodes from {len(ids)}")
    rng = np.random.default_rng(seed)
    selected = []
    for bucket in np.array_split(ids, count):
        selected.append(int(bucket[int(rng.integers(0, len(bucket)))]))
    return selected


def evenly_spaced_anchors(length: int, chunk_size: int, count: int) -> list[int]:
    valid_count = int(length) - int(chunk_size) + 1
    if valid_count <= 0:
        raise ValueError(f"episode length {length} is shorter than chunk size {chunk_size}")
    if count <= 0:
        raise ValueError("anchors_per_episode must be positive")
    if valid_count <= count:
        return list(range(valid_count))
    anchors = np.rint(np.linspace(0, valid_count - 1, count)).astype(np.int64)
    if len(np.unique(anchors)) != count:
        raise RuntimeError("evenly spaced anchor generation produced duplicates")
    return anchors.tolist()


def rot6d_to_matrix(value: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("rot6d must be a finite 6-vector")
    first = value[:3]
    first_norm = float(np.linalg.norm(first))
    if first_norm < eps:
        raise ValueError("rot6d first column has zero norm")
    first = first / first_norm
    second = value[3:] - np.dot(first, value[3:]) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < eps:
        raise ValueError("rot6d columns are collinear")
    second = second / second_norm
    return np.column_stack((first, second, np.cross(first, second)))


def rotation_geodesic_rad(left: np.ndarray, right: np.ndarray) -> float:
    relative = rot6d_to_matrix(left) @ rot6d_to_matrix(right).T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cosine))


def classify_action_window(actions: np.ndarray) -> dict[str, str]:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] < PRIMARY_HORIZON or actions.shape[1] != 10:
        raise ValueError(
            f"expected at least {PRIMARY_HORIZON} actions with shape (*, 10), got {actions.shape}"
        )
    head = actions[:PRIMARY_HORIZON]
    if not np.isfinite(head).all():
        raise ValueError("action window contains NaN or infinity")
    position_deltas = np.linalg.norm(np.diff(head[:, :3], axis=0), axis=1)
    rotation_deltas = np.array(
        [rotation_geodesic_rad(head[index + 1, 3:9], head[index, 3:9]) for index in range(9)]
    )
    gripper_deltas = np.abs(np.diff(head[:, 9]))
    tcp_group = (
        "tcp_moving"
        if np.any(position_deltas > POSITION_MOTION_THRESHOLD_M)
        or np.any(rotation_deltas > ROTATION_MOTION_THRESHOLD_RAD)
        else "tcp_static"
    )
    gripper_group = (
        "gripper_transition"
        if np.any(gripper_deltas > GRIPPER_TRANSITION_THRESHOLD)
        else "gripper_stable"
    )
    return {"tcp_group": tcp_group, "gripper_group": gripper_group}


def derive_sample_seed(base_seed: int, sample_id: int, repeat: int, stream: int) -> int:
    sequence = np.random.SeedSequence([int(base_seed), int(sample_id), int(repeat), int(stream)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _episode_rows(meta: Any) -> dict[int, dict[str, Any]]:
    rows = {}
    for row in meta.episodes:
        episode_id = int(row["episode_index"])
        if episode_id in rows:
            raise ValueError(f"duplicate episode index {episode_id}")
        rows[episode_id] = row
    return rows


def _selected_dataset_paths(
    meta: Any, rows: dict[int, dict[str, Any]], episode_ids: Sequence[int]
) -> list[Path]:
    paths: list[Path] = []
    for episode_id in episode_ids:
        row = rows[episode_id]
        paths.append(
            Path(meta.data_path.format(
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
            ))
        )
        for camera_key in meta.camera_keys:
            paths.append(
                Path(meta.video_path.format(
                    video_key=camera_key,
                    chunk_index=int(row[f"videos/{camera_key}/chunk_index"]),
                    file_index=int(row[f"videos/{camera_key}/file_index"]),
                ))
            )
    return paths


def _action_window_from_raw_dataset(dataset: Any, global_index: int) -> np.ndarray:
    actions = [
        np.asarray(dataset.get_raw_item(global_index + offset)["action"])
        for offset in range(PRIMARY_HORIZON)
    ]
    return np.stack(actions).astype(np.float64)


def _episode_actions(dataset: Any, row: dict[str, Any]) -> np.ndarray:
    start = int(row["dataset_from_index"])
    length = int(row["length"])
    return np.stack(
        [np.asarray(dataset.get_raw_item(start + offset)["action"]) for offset in range(length)]
    ).astype(np.float64)


def transition_focused_anchors(actions: np.ndarray, count: int) -> list[int]:
    if count <= 0:
        return []
    actions = np.asarray(actions, dtype=np.float64)
    valid_count = len(actions) - FULL_HORIZON + 1
    if valid_count <= 0:
        raise ValueError(f"episode length {len(actions)} is shorter than chunk size {FULL_HORIZON}")
    deltas = np.abs(np.diff(actions[:, 9])) > GRIPPER_TRANSITION_THRESHOLD
    candidates = [
        start
        for start in range(valid_count)
        if np.any(deltas[start : start + PRIMARY_HORIZON - 1])
    ]
    if len(candidates) <= count:
        return candidates
    positions = np.rint(np.linspace(0, len(candidates) - 1, count)).astype(np.int64)
    return [candidates[index] for index in positions]


def create_manifest(
    dataset: Any,
    *,
    dataset_repo_id: str,
    dataset_root: Path,
    episode_count: int,
    anchors_per_episode: int,
    flow_repeats: int,
    inference_repeats: int,
    seed: int,
    dataset_role: str = "seen",
    split_manifest: dict[str, Any] | None = None,
    transition_anchors_per_episode: int = 0,
) -> dict[str, Any]:
    from vla.diagnostics.franka_sft_split import (
        split_manifest_digest,
        validate_split_dataset_contract,
    )

    if dataset_role not in {"seen", "val"}:
        raise ValueError("dataset_role must be seen or val")
    if transition_anchors_per_episode < 0:
        raise ValueError("transition_anchors_per_episode must be non-negative")
    meta = dataset.meta
    root = Path(dataset_root).resolve()
    rows = _episode_rows(meta)
    split_context = None
    original_episode_ids: dict[int, int] = {}
    if dataset_role == "val":
        if split_manifest is None:
            raise ValueError("holdout diagnostics require a split manifest")
        split = validate_split_dataset_contract(
            split_manifest,
            "val",
            dataset_repo_id=dataset_repo_id,
            dataset_root=root,
        )
        if episode_count != len(rows):
            raise ValueError(
                f"holdout diagnostics must use every validation episode: expected={len(rows)}, "
                f"actual={episode_count}"
            )
        episode_ids = sorted(rows)
        mapping = {
            int(original): int(split_id)
            for original, split_id in split["original_to_split_episode"].items()
        }
        original_episode_ids = {split_id: original for original, split_id in mapping.items()}
        if set(original_episode_ids) != set(episode_ids):
            raise ValueError("validation episode mapping does not match the materialized dataset")
        split_context = {
            "role": "val",
            "manifest_sha256": split_manifest_digest(split_manifest),
            "original_episode_ids": [original_episode_ids[episode_id] for episode_id in episode_ids],
        }
    else:
        episode_ids = select_stratified_episode_ids(list(rows), episode_count, seed)
    samples = []
    sample_id = 0
    for episode_id in episode_ids:
        row = rows[episode_id]
        length = int(row["length"])
        actions = _episode_actions(dataset, row)
        anchors = [
            (local_index, "uniform")
            for local_index in evenly_spaced_anchors(length, FULL_HORIZON, anchors_per_episode)
        ]
        if dataset_role == "val":
            anchors.extend(
                (local_index, "gripper_transition")
                for local_index in transition_focused_anchors(actions, transition_anchors_per_episode)
            )
        for local_index, sample_kind in anchors:
            global_index = int(row["dataset_from_index"]) + local_index
            groups = classify_action_window(actions[local_index : local_index + PRIMARY_HORIZON])
            sample = {
                "sample_id": sample_id,
                "episode_index": episode_id,
                "frame_index": local_index,
                "dataset_index": global_index,
                "flow_seeds": [
                    derive_sample_seed(seed, sample_id, repeat, 0) for repeat in range(flow_repeats)
                ],
                "inference_seeds": [
                    derive_sample_seed(seed, sample_id, repeat, 1)
                    for repeat in range(inference_repeats)
                ],
                **groups,
            }
            if dataset_role == "val":
                sample["sample_kind"] = sample_kind
                sample["original_episode_index"] = original_episode_ids[episode_id]
            samples.append(sample)
            sample_id += 1

    metadata_paths = [
        path.relative_to(root)
        for path in sorted((root / "meta").rglob("*"))
        if path.is_file()
    ]
    selected_paths = _selected_dataset_paths(meta, rows, episode_ids)
    dataset_fingerprint = {
        "metadata": fingerprint_files(root, metadata_paths),
        "selected_data": fingerprint_files(root, selected_paths),
    }
    fingerprint_digest = hashlib.sha256()
    fingerprint_digest.update(dataset_fingerprint["metadata"]["sha256"].encode("ascii"))
    fingerprint_digest.update(dataset_fingerprint["selected_data"]["sha256"].encode("ascii"))
    dataset_fingerprint["sha256"] = fingerprint_digest.hexdigest()

    return {
        "schema_version": HOLDOUT_SCHEMA_VERSION if dataset_role == "val" else SCHEMA_VERSION,
        "diagnostic_seen_data": dataset_role == "seen",
        "warning": (
            "All sampled episodes were used for training; this is not an unbiased validation set."
            if dataset_role == "seen"
            else "Validation episodes are disjoint from the checkpoint training split."
        ),
        **({"split": split_context} if split_context is not None else {}),
        "dataset": {
            "repo_id": dataset_repo_id,
            "root": str(root),
            "codebase_version": meta.info["codebase_version"],
            "total_episodes": int(meta.total_episodes),
            "total_frames": int(meta.total_frames),
            "fps": int(meta.fps),
            "action_names": list(meta.features["action"]["names"]),
            "fingerprint": dataset_fingerprint,
        },
        "sampling": {
            "policy": "stratified_episode_uniform_frame_no_padding",
            "seed": seed,
            "episode_count": episode_count,
            "episode_ids": episode_ids,
            "anchors_per_episode": anchors_per_episode,
            "chunk_size": FULL_HORIZON,
            "primary_horizon": PRIMARY_HORIZON,
            "flow_repeats": flow_repeats,
            "inference_repeats": inference_repeats,
            "position_motion_threshold_m": POSITION_MOTION_THRESHOLD_M,
            "rotation_motion_threshold_rad": ROTATION_MOTION_THRESHOLD_RAD,
            "gripper_transition_threshold": GRIPPER_TRANSITION_THRESHOLD,
            **(
                {
                    "dataset_role": dataset_role,
                    "transition_anchors_per_episode": transition_anchors_per_episode,
                    "uniform_sample_count": sum(sample["sample_kind"] == "uniform" for sample in samples),
                    "transition_sample_count": sum(
                        sample["sample_kind"] == "gripper_transition" for sample in samples
                    ),
                    "bootstrap_unit": "episode",
                }
                if dataset_role == "val"
                else {}
            ),
        },
        "samples": samples,
    }


def validate_manifest(
    manifest: dict[str, Any],
    dataset: Any,
    *,
    dataset_repo_id: str,
    dataset_root: Path,
    episode_count: int,
    anchors_per_episode: int,
    flow_repeats: int,
    inference_repeats: int,
    seed: int,
    dataset_role: str = "seen",
    split_manifest: dict[str, Any] | None = None,
    transition_anchors_per_episode: int = 0,
) -> None:
    expected_schema = HOLDOUT_SCHEMA_VERSION if dataset_role == "val" else SCHEMA_VERSION
    if manifest.get("schema_version") != expected_schema:
        raise ValueError(f"unsupported manifest schema {manifest.get('schema_version')!r}")
    if manifest.get("diagnostic_seen_data") is not (dataset_role == "seen"):
        raise ValueError("diagnostic manifest seen-data declaration does not match dataset_role")
    expected_sampling = {
        "seed": seed,
        "episode_count": episode_count,
        "anchors_per_episode": anchors_per_episode,
        "chunk_size": FULL_HORIZON,
        "primary_horizon": PRIMARY_HORIZON,
        "flow_repeats": flow_repeats,
        "inference_repeats": inference_repeats,
    }
    if dataset_role == "val":
        expected_sampling.update(
            {
                "dataset_role": "val",
                "transition_anchors_per_episode": transition_anchors_per_episode,
                "bootstrap_unit": "episode",
            }
        )
    for key, expected in expected_sampling.items():
        actual = manifest["sampling"].get(key)
        if actual != expected:
            raise ValueError(f"manifest sampling mismatch for {key}: expected {expected}, got {actual}")
    if manifest["dataset"].get("repo_id") != dataset_repo_id:
        raise ValueError("manifest dataset repo_id does not match")
    if Path(manifest["dataset"].get("root", "")).resolve() != Path(dataset_root).resolve():
        raise ValueError("manifest dataset root does not match")
    if tuple(manifest["dataset"].get("action_names", ())) != EXPECTED_ACTION_NAMES:
        raise ValueError("manifest action names do not match the Franka 10D schema")

    regenerated = create_manifest(
        dataset,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        episode_count=episode_count,
        anchors_per_episode=anchors_per_episode,
        flow_repeats=flow_repeats,
        inference_repeats=inference_repeats,
        seed=seed,
        dataset_role=dataset_role,
        split_manifest=split_manifest,
        transition_anchors_per_episode=transition_anchors_per_episode,
    )
    if regenerated != manifest:
        raise ValueError("manifest content or dataset fingerprint changed; create a new manifest path")


def load_or_create_manifest(
    path: Path,
    dataset: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        validate_manifest(manifest, dataset, **kwargs)
        return manifest
    manifest = create_manifest(dataset, **kwargs)
    write_json_atomic(path, manifest)
    return manifest


def read_checkpoint_config(checkpoint: Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    for name in required:
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"checkpoint is missing {name}: {checkpoint}")
    with (checkpoint / "config.json").open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("type") != "pi05":
        raise ValueError(f"checkpoint {checkpoint} is not PI0.5")
    if tuple(config.get("action_feature_names") or ()) != EXPECTED_ACTION_NAMES:
        raise ValueError(f"checkpoint {checkpoint} does not use the expected Franka 10D action schema")
    if config.get("output_features", {}).get("action", {}).get("shape") != [10]:
        raise ValueError(f"checkpoint {checkpoint} action shape is not 10D")
    if int(config.get("chunk_size", -1)) != FULL_HORIZON:
        raise ValueError(f"checkpoint {checkpoint} chunk_size is not {FULL_HORIZON}")
    if config.get("use_relative_actions") is not False:
        raise ValueError(f"checkpoint {checkpoint} is not configured for absolute actions")
    return config


def checkpoint_tensor_schema(checkpoint: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    from safetensors import safe_open

    path = Path(checkpoint).resolve() / "model.safetensors"
    with safe_open(path, framework="pt", device="cpu") as tensors:
        schema = {
            key: (tuple(tensors.get_slice(key).get_shape()), tensors.get_slice(key).get_dtype())
            for key in tensors.keys()
        }
    required = {
        "model.action_in_proj.weight",
        "model.action_out_proj.weight",
        "model.paligemma_with_expert.gemma_expert.lm_head.weight",
    }
    missing = sorted(required - schema.keys())
    if missing:
        raise ValueError(f"checkpoint tensor schema is missing required weights: {missing}")
    return schema


def validate_checkpoint_contracts(checkpoints: Sequence[Path], dataset_meta: Any) -> list[dict[str, Any]]:
    if tuple(dataset_meta.features["action"].get("names") or ()) != EXPECTED_ACTION_NAMES:
        raise ValueError("dataset does not use the expected Franka 10D action names")
    if list(dataset_meta.features["action"].get("shape") or ()) != [10]:
        raise ValueError("dataset action feature is not 10D")
    configs = [read_checkpoint_config(path) for path in checkpoints]
    contract_keys = (
        "type",
        "input_features",
        "output_features",
        "chunk_size",
        "use_relative_actions",
        "action_feature_names",
        "normalization_mapping",
    )
    baseline = {key: configs[0].get(key) for key in contract_keys}
    baseline_tensor_schema = checkpoint_tensor_schema(checkpoints[0])
    for checkpoint, config in zip(checkpoints[1:], configs[1:], strict=True):
        current = {key: config.get(key) for key in contract_keys}
        if current != baseline:
            raise ValueError(f"checkpoint contract differs from the first checkpoint: {checkpoint}")
        if checkpoint_tensor_schema(checkpoint) != baseline_tensor_schema:
            raise ValueError(f"checkpoint tensor schema differs from the first checkpoint: {checkpoint}")
    return configs


def validate_checkpoint_training_split(
    checkpoints: Sequence[Path], split_manifest: dict[str, Any]
) -> None:
    from vla.diagnostics.franka_sft_split import validate_split_dataset_contract

    train = split_manifest["splits"]["train"]
    for checkpoint in checkpoints:
        config_path = Path(checkpoint) / "train_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"holdout checkpoint is missing train_config.json: {checkpoint}")
        with config_path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
        dataset = config.get("dataset") or {}
        validate_split_dataset_contract(
            split_manifest,
            "train",
            dataset_repo_id=dataset.get("repo_id"),
            dataset_root=Path(dataset.get("root", "")),
        )
        if dataset.get("episodes") is not None:
            raise ValueError(f"holdout checkpoint used an additional episode filter: {checkpoint}")
        if config.get("policy", {}).get("use_relative_actions") is not False:
            raise ValueError(f"holdout checkpoint is not absolute-action SFT: {checkpoint}")
        if Path(dataset["root"]).resolve() != Path(train["root"]).resolve():
            raise ValueError(f"holdout checkpoint training root differs from the manifest: {checkpoint}")


def strict_load_policy(checkpoint: Path, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint).resolve()
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = device
    policy = PI05Policy(config)
    state_dict = load_file(checkpoint / "model.safetensors", device="cpu")
    fixed = policy._fix_pytorch_state_dict_keys(state_dict, policy.config)
    remapped = {key if key.startswith("model.") else f"model.{key}": value for key, value in fixed.items()}
    policy.load_state_dict(remapped, strict=True)
    policy.to(device)
    policy.eval()
    del state_dict, fixed, remapped
    return policy


def make_checkpoint_processors(policy: Any, checkpoint: Path, device: str, tokenizer_path: Path | None):
    from lerobot.policies.factory import make_pre_post_processors

    overrides: dict[str, dict[str, Any]] = {"device_processor": {"device": device}}
    if tokenizer_path is not None:
        tokenizer_path = Path(tokenizer_path).resolve()
        if not (tokenizer_path / "tokenizer.json").is_file():
            raise FileNotFoundError(f"local tokenizer snapshot is incomplete: {tokenizer_path}")
        overrides["tokenizer_processor"] = {"tokenizer_name": str(tokenizer_path)}
    return make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(Path(checkpoint).resolve()),
        preprocessor_overrides=overrides,
    )


def _seeded_flow_inputs(
    samples: Sequence[dict[str, Any]], repeat: int, policy: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    noises = []
    times = []
    for sample in samples:
        rng = np.random.default_rng(int(sample["flow_seeds"][repeat]))
        noises.append(
            rng.standard_normal((FULL_HORIZON, policy.config.max_action_dim), dtype=np.float32)
        )
        beta = rng.beta(policy.config.time_sampling_beta_alpha, policy.config.time_sampling_beta_beta)
        times.append(beta * policy.config.time_sampling_scale + policy.config.time_sampling_offset)
    device = torch.device(policy.config.device)
    return (
        torch.from_numpy(np.stack(noises)).to(device=device),
        torch.tensor(times, dtype=torch.float32, device=device),
    )


def _seeded_inference_noise(samples: Sequence[dict[str, Any]], repeat: int, policy: Any) -> torch.Tensor:
    noises = []
    for sample in samples:
        rng = np.random.default_rng(int(sample["inference_seeds"][repeat]))
        noises.append(
            rng.standard_normal((FULL_HORIZON, policy.config.max_action_dim), dtype=np.float32)
        )
    return torch.from_numpy(np.stack(noises)).to(policy.config.device)


def compute_flow_loss_tensor(
    policy: Any, processed_batch: dict[str, torch.Tensor], noise: torch.Tensor, time: torch.Tensor
) -> torch.Tensor:
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, image_masks = policy._preprocess_images(processed_batch)
    tokens = processed_batch[OBS_LANGUAGE_TOKENS]
    masks = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = policy.prepare_action(processed_batch)
    losses = policy.model.forward(images, image_masks, tokens, masks, actions, noise=noise, time=time)
    losses = losses[:, :, : len(EXPECTED_ACTION_NAMES)]
    if losses.shape[1:] != (FULL_HORIZON, len(EXPECTED_ACTION_NAMES)):
        raise ValueError(f"unexpected PI0.5 loss shape {tuple(losses.shape)}")
    if not torch.isfinite(losses).all():
        raise ValueError("PI0.5 flow loss contains NaN or infinity")
    return losses.float().cpu()


def decoded_action_errors(predicted: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[2] != 10:
        raise ValueError(f"decoded action shape mismatch: predicted={predicted.shape}, target={target.shape}")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("decoded actions contain NaN or infinity")
    position_mm = np.linalg.norm(predicted[:, :, :3] - target[:, :, :3], axis=2) * 1000.0
    rotation_deg = np.empty(predicted.shape[:2], dtype=np.float64)
    for sample in range(predicted.shape[0]):
        for step in range(predicted.shape[1]):
            rotation_deg[sample, step] = math.degrees(
                rotation_geodesic_rad(predicted[sample, step, 3:9], target[sample, step, 3:9])
            )
    gripper = np.abs(predicted[:, :, 9] - target[:, :, 9])
    return {"position_mm": position_mm, "rotation_deg": rotation_deg, "gripper": gripper}


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("cannot summarize empty or non-finite values")
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def summarize_flow(sample_dim_losses: np.ndarray, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    losses = np.asarray(sample_dim_losses, dtype=np.float64)
    if losses.shape != (len(samples), FULL_HORIZON, len(EXPECTED_ACTION_NAMES)):
        raise ValueError(f"unexpected aggregated flow loss shape {losses.shape}")

    def horizon_summary(horizon: int) -> dict[str, Any]:
        selected = losses[:, :horizon]
        per_sample_dim = selected.mean(axis=1)
        per_sample = per_sample_dim.mean(axis=1)
        return {
            "overall": distribution_summary(per_sample),
            "per_dim": {
                name: distribution_summary(per_sample_dim[:, index])
                for index, name in enumerate(EXPECTED_ACTION_NAMES)
            },
            "groups": {
                "position": distribution_summary(per_sample_dim[:, :3].mean(axis=1)),
                "rotation_6d": distribution_summary(per_sample_dim[:, 3:9].mean(axis=1)),
                "gripper": distribution_summary(per_sample_dim[:, 9]),
            },
        }

    sample_loss = losses.mean(axis=(1, 2))
    slices = {}
    for group in ("tcp_static", "tcp_moving", "gripper_stable", "gripper_transition"):
        key = "tcp_group" if group.startswith("tcp_") else "gripper_group"
        mask = np.array([sample[key] == group for sample in samples])
        slices[group] = distribution_summary(sample_loss[mask]) if np.any(mask) else {"count": 0}
    return {
        "full_50": horizon_summary(FULL_HORIZON),
        "first_10": horizon_summary(PRIMARY_HORIZON),
        "slices_full_50": slices,
    }


def summarize_decoded(errors: dict[str, np.ndarray], samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def horizon_summary(horizon: int, mask: np.ndarray | None = None) -> dict[str, Any]:
        if mask is None:
            mask = np.ones(len(samples), dtype=bool)
        position = errors["position_mm"][mask, :horizon]
        rotation = errors["rotation_deg"][mask, :horizon]
        gripper = errors["gripper"][mask, :horizon]
        if not np.any(mask):
            return {"sample_count": 0}
        return {
            "sample_count": int(np.count_nonzero(mask)),
            "position": {
                **distribution_summary(position),
                "rmse_mm": float(np.sqrt(np.mean(np.square(position)))),
            },
            "rotation": distribution_summary(rotation),
            "gripper": {**distribution_summary(gripper), "mae": float(np.mean(gripper))},
        }

    slices = {}
    for group in ("tcp_static", "tcp_moving", "gripper_stable", "gripper_transition"):
        key = "tcp_group" if group.startswith("tcp_") else "gripper_group"
        mask = np.array([sample[key] == group for sample in samples])
        slices[group] = horizon_summary(PRIMARY_HORIZON, mask)
    return {
        "first_10": horizon_summary(PRIMARY_HORIZON),
        "full_50": horizon_summary(FULL_HORIZON),
        "slices_first_10": slices,
    }


def aggregate_metric_by_episode(
    values: np.ndarray, samples: Sequence[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(samples):
        raise ValueError("episode aggregation requires one scalar per sample")
    episode_ids = np.asarray([int(sample["episode_index"]) for sample in samples], dtype=np.int64)
    unique_ids = np.unique(episode_ids)
    aggregated = np.asarray([values[episode_ids == episode_id].mean() for episode_id in unique_ids])
    return unique_ids, aggregated


def _raw_batches(dataset: Any, samples: Sequence[dict[str, Any]], batch_size: int):
    from torch.utils.data._utils.collate import default_collate

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        items = [dataset[int(sample["dataset_index"])] for sample in batch_samples]
        yield batch_samples, default_collate(items)


def evaluate_checkpoint(
    checkpoint: Path,
    dataset: Any,
    samples: Sequence[dict[str, Any]],
    *,
    device: str,
    batch_size: int,
    flow_repeats: int,
    inference_repeats: int,
    tokenizer_path: Path | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    log.info("Loading checkpoint %s", checkpoint_label(checkpoint))
    policy = strict_load_policy(checkpoint, device)
    preprocessor, postprocessor = make_checkpoint_processors(policy, checkpoint, device, tokenizer_path)
    flow_batches: list[np.ndarray] = []
    decoded_batches: list[dict[str, np.ndarray]] = []
    total_batches = math.ceil(len(samples) / batch_size)
    try:
        with torch.inference_mode():
            for batch_index, (batch_samples, raw_batch) in enumerate(
                _raw_batches(dataset, samples, batch_size), start=1
            ):
                target = np.asarray(raw_batch["action"], dtype=np.float64)
                processed = preprocessor(raw_batch)
                repeated_losses = []
                for repeat in range(flow_repeats):
                    noise, time = _seeded_flow_inputs(batch_samples, repeat, policy)
                    repeated_losses.append(compute_flow_loss_tensor(policy, processed, noise, time).numpy())
                flow_batches.append(np.mean(np.stack(repeated_losses), axis=0))

                repeated_errors = []
                for repeat in range(inference_repeats):
                    noise = _seeded_inference_noise(batch_samples, repeat, policy)
                    normalized = policy.predict_action_chunk(processed, noise=noise)
                    predicted = np.asarray(postprocessor(normalized), dtype=np.float64)
                    repeated_errors.append(decoded_action_errors(predicted, target))
                decoded_batches.append(
                    {
                        key: np.mean(np.stack([value[key] for value in repeated_errors]), axis=0)
                        for key in repeated_errors[0]
                    }
                )
                if batch_index == 1 or batch_index % 25 == 0 or batch_index == total_batches:
                    log.info(
                        "Checkpoint %s: evaluated batch %d/%d",
                        checkpoint_label(checkpoint),
                        batch_index,
                        total_batches,
                    )
    finally:
        del preprocessor, postprocessor, policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    flow = np.concatenate(flow_batches, axis=0)
    errors = {
        key: np.concatenate([batch[key] for batch in decoded_batches], axis=0)
        for key in decoded_batches[0]
    }
    sample_kinds = np.asarray([sample.get("sample_kind", "uniform") for sample in samples])
    uniform_mask = sample_kinds == "uniform"
    transition_mask = sample_kinds == "gripper_transition"
    uniform_samples = [sample for sample, selected in zip(samples, uniform_mask, strict=True) if selected]
    uniform_flow = flow[uniform_mask]
    uniform_errors = {key: value[uniform_mask] for key, value in errors.items()}
    result = {
        "status": "ok",
        "checkpoint": str(Path(checkpoint).resolve()),
        "weights_sha256": None,
        "flow": summarize_flow(uniform_flow, uniform_samples),
        "decoded_action": summarize_decoded(uniform_errors, uniform_samples),
    }
    uniform_scalar_metrics = {
        "flow_full_50": uniform_flow.mean(axis=(1, 2)),
        "flow_first_10": uniform_flow[:, :PRIMARY_HORIZON].mean(axis=(1, 2)),
        "position_rmse_mm": np.sqrt(
            np.mean(np.square(uniform_errors["position_mm"][:, :PRIMARY_HORIZON]), axis=1)
        ),
        "rotation_mean_deg": uniform_errors["rotation_deg"][:, :PRIMARY_HORIZON].mean(axis=1),
        "gripper_mae": uniform_errors["gripper"][:, :PRIMARY_HORIZON].mean(axis=1),
    }
    sample_metrics = {
        key: aggregate_metric_by_episode(value, uniform_samples)[1]
        for key, value in uniform_scalar_metrics.items()
    }
    if np.any(transition_mask):
        transition_samples = [
            sample for sample, selected in zip(samples, transition_mask, strict=True) if selected
        ]
        transition_flow = flow[transition_mask]
        transition_errors = {key: value[transition_mask] for key, value in errors.items()}
        result["transition_focus"] = {
            "flow": summarize_flow(transition_flow, transition_samples),
            "decoded_action": summarize_decoded(transition_errors, transition_samples),
        }
        _, sample_metrics["gripper_transition_mae"] = aggregate_metric_by_episode(
            transition_errors["gripper"][:, :PRIMARY_HORIZON].mean(axis=1),
            transition_samples,
        )
    log.info("Checkpoint %s: hashing model.safetensors", checkpoint_label(checkpoint))
    result["weights_sha256"] = sha256_file(Path(checkpoint) / "model.safetensors")
    log.info("Checkpoint %s: diagnostics complete", checkpoint_label(checkpoint))
    return result, sample_metrics


def paired_bootstrap(
    left: np.ndarray, right: np.ndarray, *, seed: int, resamples: int = 10_000
) -> dict[str, float | int]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("paired bootstrap inputs must be non-empty matching vectors")
    deltas = right - left
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample_indices = rng.integers(0, len(deltas), size=len(deltas))
        bootstrap[index] = np.mean(deltas[sample_indices])
    return {
        "sample_count": int(len(deltas)),
        "mean_delta_right_minus_left": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "right_win_rate": float(np.mean(right < left)),
    }


def checkpoint_label(path: Path) -> str:
    resolved = Path(path).resolve()
    if resolved.name != "pretrained_model":
        return resolved.name
    step = resolved.parent
    if step.parent.name == "checkpoints":
        return f"{step.parent.parent.name}@{step.name}"
    return step.name


def compare_checkpoints(
    checkpoints: Sequence[Path], metrics: Sequence[dict[str, np.ndarray]], seed: int
) -> list[dict[str, Any]]:
    comparisons = []
    for left_index in range(len(checkpoints)):
        for right_index in range(left_index + 1, len(checkpoints)):
            comparisons.append(
                {
                    "left": checkpoint_label(checkpoints[left_index]),
                    "right": checkpoint_label(checkpoints[right_index]),
                    "lower_is_better": True,
                    "metrics": {
                        key: paired_bootstrap(
                            metrics[left_index][key],
                            metrics[right_index][key],
                            seed=derive_sample_seed(seed, left_index, right_index, metric_index + 2),
                        )
                        for metric_index, key in enumerate(metrics[left_index])
                    },
                }
            )
    return comparisons


def selection_summary(
    checkpoints: Sequence[Path], metrics: Sequence[dict[str, np.ndarray]]
) -> dict[str, Any]:
    names = [checkpoint_label(path) for path in checkpoints]
    means = [{key: float(np.mean(value)) for key, value in metric.items()} for metric in metrics]
    holdout_composite = all("gripper_transition_mae" in metric for metric in metrics)
    keys = (
        "flow_full_50",
        "position_rmse_mm",
        "rotation_mean_deg",
        "gripper_transition_mae" if holdout_composite else "gripper_mae",
    )
    best = {key: names[int(np.argmin([value[key] for value in means]))] for key in keys}
    pareto = []
    dominates_all = []
    for candidate in range(len(names)):
        dominated = False
        dominates = []
        for other in range(len(names)):
            if candidate == other:
                continue
            other_dominates = all(means[other][key] <= means[candidate][key] for key in keys) and any(
                means[other][key] < means[candidate][key] for key in keys
            )
            if other_dominates:
                dominated = True
            candidate_dominates = all(means[candidate][key] <= means[other][key] for key in keys) and any(
                means[candidate][key] < means[other][key] for key in keys
            )
            if candidate_dominates:
                dominates.append(other)
        if not dominated:
            pareto.append(names[candidate])
        if len(dominates) == len(names) - 1:
            dominates_all.append(names[candidate])
    result = {
        "best_by_metric": best,
        "pareto_front": pareto,
        "recommended_checkpoint": dominates_all[0] if len(dominates_all) == 1 else None,
        "conclusion": "dominates_all_primary_metrics" if len(dominates_all) == 1 else "inconclusive",
    }
    if holdout_composite:
        weights = {
            "flow_full_50": 0.20,
            "position_rmse_mm": 0.30,
            "rotation_mean_deg": 0.30,
            "gripper_transition_mae": 0.20,
        }
        baseline = means[0]
        candidates = []
        for name, values in zip(names, means, strict=True):
            ratios = {key: values[key] / baseline[key] for key in weights}
            candidates.append(
                {
                    "checkpoint": name,
                    "ratios_to_first_checkpoint": ratios,
                    "score": sum(weights[key] * ratios[key] for key in weights),
                    "eligible": all(value <= 1.05 for value in ratios.values()),
                }
            )
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        recommended = min(eligible, key=lambda value: (value["score"], names.index(value["checkpoint"])))
        result.update(
            {
                "composite": {
                    "baseline_checkpoint": names[0],
                    "weights": weights,
                    "regression_guardrail": 1.05,
                    "candidates": candidates,
                },
                "recommended_checkpoint": recommended["checkpoint"],
                "conclusion": "holdout_composite_score",
            }
        )
    return result


def schedule_diagnostics(checkpoint: Path, num_frames: int) -> dict[str, Any]:
    config_path = Path(checkpoint) / "train_config.json"
    if not config_path.is_file():
        return {"status": "unavailable", "reason": "train_config.json is missing"}
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    batch_size = int(config["batch_size"])
    steps_per_epoch = math.ceil(num_frames / batch_size)
    warmup_steps = int(config["scheduler"]["num_warmup_steps"])
    decay_steps = int(config["scheduler"]["num_decay_steps"])
    total_steps = int(config["steps"])
    floor_epoch = decay_steps / steps_per_epoch
    warnings = []
    if floor_epoch < 1.0:
        warnings.append("learning rate reaches its floor before one dataset epoch")
    return {
        "status": "ok",
        "batch_size": batch_size,
        "assumed_world_size": 1,
        "effective_batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup_steps,
        "warmup_epoch": warmup_steps / steps_per_epoch,
        "decay_steps": decay_steps,
        "lr_floor_epoch": floor_epoch,
        "total_steps": total_steps,
        "total_epochs": total_steps / steps_per_epoch,
        "warnings": warnings,
        "future_retraining_guidance": (
            "Express total and decay steps in dataset epochs; "
            "do not resume until diagnostics are reviewed."
        ),
    }


def run_diagnostics(
    *,
    dataset_repo_id: str,
    dataset_root: Path,
    checkpoints: Sequence[Path],
    manifest_path: Path,
    output_path: Path,
    tokenizer_path: Path | None,
    device: str = "cuda",
    batch_size: int = 2,
    episode_count: int = 15,
    anchors_per_episode: int = 32,
    flow_repeats: int = 4,
    inference_repeats: int = 1,
    seed: int = 1000,
    video_backend: str = "torchcodec",
    dataset_role: str = "seen",
    split_manifest_path: Path | None = None,
    transition_anchors_per_episode: int = 0,
) -> dict[str, Any]:
    import lerobot.policies.pi05.configuration_pi05  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from vla.diagnostics.franka_sft_split import (
        load_split_manifest,
        validate_split_dataset_contract,
    )

    if batch_size <= 0 or flow_repeats <= 0 or inference_repeats <= 0:
        raise ValueError("batch_size and repeat counts must be positive")
    if dataset_role not in {"seen", "val"}:
        raise ValueError("dataset_role must be seen or val")
    if dataset_role == "seen" and split_manifest_path is not None:
        raise ValueError("split_manifest_path is only valid for dataset_role=val")
    if dataset_role == "val" and split_manifest_path is None:
        raise ValueError("dataset_role=val requires split_manifest_path")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device {device!r}, but CUDA is not available")
    checkpoints = [Path(path).resolve() for path in checkpoints]
    if len(checkpoints) < 2:
        raise ValueError("at least two checkpoints are required for comparison")
    dataset_root = Path(dataset_root).resolve()
    split_manifest = load_split_manifest(split_manifest_path) if split_manifest_path is not None else None
    if split_manifest is not None:
        validate_split_dataset_contract(
            split_manifest,
            "val",
            dataset_repo_id=dataset_repo_id,
            dataset_root=dataset_root,
        )
        validate_checkpoint_training_split(checkpoints, split_manifest)
    meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    configs = validate_checkpoint_contracts(checkpoints, meta)
    first_config = PreTrainedConfig.from_pretrained(checkpoints[0], local_files_only=True)
    delta_timestamps = resolve_delta_timestamps(first_config, meta)
    dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        tolerance_s=0.04,
        video_backend=video_backend,
    )
    manifest_kwargs = {
        "dataset_repo_id": dataset_repo_id,
        "dataset_root": dataset_root,
        "episode_count": episode_count,
        "anchors_per_episode": anchors_per_episode,
        "flow_repeats": flow_repeats,
        "inference_repeats": inference_repeats,
        "seed": seed,
        "dataset_role": dataset_role,
        "split_manifest": split_manifest,
        "transition_anchors_per_episode": transition_anchors_per_episode,
    }
    manifest = load_or_create_manifest(manifest_path, dataset, **manifest_kwargs)
    samples = manifest["samples"]
    report: dict[str, Any] = {
        "schema_version": HOLDOUT_SCHEMA_VERSION if dataset_role == "val" else SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_seen_data": dataset_role == "seen",
        "warning": manifest["warning"],
        "dataset": manifest["dataset"],
        "sampling": manifest["sampling"],
        "manifest": str(Path(manifest_path).resolve()),
        **(
            {"split_manifest": str(Path(split_manifest_path).resolve())}
            if split_manifest_path is not None
            else {}
        ),
        "schedule": schedule_diagnostics(checkpoints[-1], int(meta.total_frames)),
        "checkpoints": [],
        "paired_comparisons": [],
        "selection": None,
    }
    metric_vectors = []
    successful_paths = []
    errors = []
    for checkpoint, config in zip(checkpoints, configs, strict=True):
        try:
            result, sample_metrics = evaluate_checkpoint(
                checkpoint,
                dataset,
                samples,
                device=device,
                batch_size=batch_size,
                flow_repeats=flow_repeats,
                inference_repeats=inference_repeats,
                tokenizer_path=tokenizer_path,
            )
            result["label"] = checkpoint_label(checkpoint)
            result["config"] = {
                "chunk_size": config["chunk_size"],
                "num_inference_steps": config["num_inference_steps"],
                "dtype": config["dtype"],
            }
            report["checkpoints"].append(result)
            metric_vectors.append(sample_metrics)
            successful_paths.append(checkpoint)
        except Exception as error:
            report["checkpoints"].append(
                {
                    "status": "failed",
                    "checkpoint": str(checkpoint),
                    "label": checkpoint_label(checkpoint),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            errors.append((checkpoint, error))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(successful_paths) >= 2:
        report["paired_comparisons"] = compare_checkpoints(successful_paths, metric_vectors, seed)
        report["selection"] = selection_summary(successful_paths, metric_vectors)
    write_json_atomic(output_path, report)
    if errors:
        details = "; ".join(f"{path}: {type(error).__name__}: {error}" for path, error in errors)
        raise RuntimeError(f"one or more checkpoint diagnostics failed; partial report written: {details}")
    return report
