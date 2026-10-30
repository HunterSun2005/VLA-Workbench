from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from copy import deepcopy
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SPLIT_SCHEMA_VERSION = 1
EXPECTED_STATE_NAMES = (
    "tcp_position_x",
    "tcp_position_y",
    "tcp_position_z",
    "tcp_rot6d_col0_x",
    "tcp_rot6d_col0_y",
    "tcp_rot6d_col0_z",
    "tcp_rot6d_col1_x",
    "tcp_rot6d_col1_y",
    "tcp_rot6d_col1_z",
    "gripper_position",
)
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
EXPECTED_CAMERA_KEYS = (
    "observation.images.base",
    "observation.images.left_wrist",
)
BALANCE_FEATURE_NAMES = (
    "episode_length",
    "position_mean_x",
    "position_mean_y",
    "position_mean_z",
    "position_span_x",
    "position_span_y",
    "position_span_z",
    "tcp_motion_rate",
    "gripper_transition_rate",
    "gripper_closed_rate",
    "gripper_open_rate",
    "gripper_mid_rate",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
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


def fingerprint_tree(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    combined = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for path in sorted((path for path in root.rglob("*") if path.is_file())):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(size).encode("ascii"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
        file_count += 1
        byte_count += size
    if file_count == 0:
        raise ValueError(f"cannot fingerprint empty dataset tree: {root}")
    return {"sha256": combined.hexdigest(), "file_count": file_count, "bytes": byte_count}


def split_manifest_digest(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_split_manifest(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SFT split manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError(f"unsupported SFT split manifest schema {manifest.get('schema_version')!r}")
    if manifest.get("kind") != "franka_pi05_sft_split":
        raise ValueError("file is not a Franka PI0.5 SFT split manifest")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "val"}:
        raise ValueError("SFT split manifest must contain exactly train and val splits")
    train_ids = set(int(value) for value in splits["train"].get("original_episode_ids", ()))
    val_ids = set(int(value) for value in splits["val"].get("original_episode_ids", ()))
    if not train_ids or not val_ids or train_ids & val_ids:
        raise ValueError("SFT split manifest has empty or overlapping episode splits")
    return manifest


def validate_split_dataset_contract(
    manifest: dict[str, Any], role: str, *, dataset_repo_id: str, dataset_root: Path
) -> dict[str, Any]:
    if role not in {"train", "val"}:
        raise ValueError("dataset role must be train or val")
    split = manifest["splits"][role]
    if split.get("repo_id") != dataset_repo_id:
        raise ValueError(
            f"{role} dataset repo_id does not match split manifest: "
            f"expected={split.get('repo_id')!r}, actual={dataset_repo_id!r}"
        )
    expected_root = Path(split.get("root", "")).resolve()
    if expected_root != Path(dataset_root).expanduser().resolve():
        raise ValueError(
            f"{role} dataset root does not match split manifest: "
            f"expected={expected_root}, actual={Path(dataset_root).expanduser().resolve()}"
        )
    return split


def _rot6d_matrices(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    first_raw = values[:, :3]
    second_raw = values[:, 3:]
    first_norm = np.linalg.norm(first_raw, axis=1)
    if np.any(first_norm < 1e-8):
        raise ValueError("rot6d contains a zero first column")
    first = first_raw / first_norm[:, None]
    second = second_raw - np.sum(first * second_raw, axis=1)[:, None] * first
    second_norm = np.linalg.norm(second, axis=1)
    if np.any(second_norm < 1e-8):
        raise ValueError("rot6d contains collinear columns")
    second = second / second_norm[:, None]
    return np.stack((first, second, np.cross(first, second)), axis=2)


def _rotation_deltas(values: np.ndarray) -> np.ndarray:
    matrices = _rot6d_matrices(values)
    if len(matrices) < 2:
        return np.empty(0, dtype=np.float64)
    relative = matrices[1:] @ np.transpose(matrices[:-1], (0, 2, 1))
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def episode_balance_features(actions: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 10 or len(actions) < 2:
        raise ValueError(f"expected an action episode with shape (N, 10), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("action episode contains NaN or infinity")
    position_deltas = np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1)
    rotation_deltas = _rotation_deltas(actions[:, 3:9])
    gripper_deltas = np.abs(np.diff(actions[:, 9]))
    moving = (position_deltas > 8e-4) | (rotation_deltas > 1e-2)
    closed = actions[:, 9] <= 0.1
    opened = actions[:, 9] >= 0.9
    feature = np.array(
        [
            len(actions),
            *actions[:, :3].mean(axis=0),
            *np.ptp(actions[:, :3], axis=0),
            moving.mean(),
            (gripper_deltas > 1e-2).mean(),
            closed.mean(),
            opened.mean(),
            (~closed & ~opened).mean(),
        ],
        dtype=np.float64,
    )
    audit = {
        "frames": len(actions),
        "position_step_p99_m": float(np.quantile(position_deltas, 0.99)),
        "position_step_max_m": float(position_deltas.max()),
        "rotation_step_p99_deg": float(np.degrees(np.quantile(rotation_deltas, 0.99))),
        "rotation_step_max_deg": float(np.degrees(rotation_deltas.max())),
        "gripper_step_p99": float(np.quantile(gripper_deltas, 0.99)),
        "gripper_step_max": float(gripper_deltas.max()),
        "balance_features": dict(zip(BALANCE_FEATURE_NAMES, feature.tolist(), strict=True)),
    }
    return feature, audit


def choose_validation_episode_ids(
    episode_ids: Sequence[int],
    feature_rows: np.ndarray,
    *,
    validation_episodes: int = 20,
    seed: int = 1000,
    candidate_count: int = 10_000,
) -> tuple[list[int], dict[str, Any]]:
    ids = np.asarray([int(value) for value in episode_ids], dtype=np.int64)
    features = np.asarray(feature_rows, dtype=np.float64)
    if ids.ndim != 1 or len(ids) != len(set(ids.tolist())) or len(ids) != len(features):
        raise ValueError("episode ids must be unique and align with feature rows")
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("balance features must be a finite 2D array")
    if validation_episodes <= 0 or validation_episodes >= len(ids):
        raise ValueError("validation_episodes must leave non-empty train and val splits")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")

    order = np.argsort(ids)
    ids = ids[order]
    features = features[order]
    bucket_count = min(10, validation_episodes)
    buckets = np.array_split(np.arange(len(ids)), bucket_count)
    base, remainder = divmod(validation_episodes, bucket_count)
    slots = [base + int(index < remainder) for index in range(bucket_count)]
    if any(slot > len(bucket) for slot, bucket in zip(slots, buckets, strict=True)):
        raise ValueError("not enough episodes to cover chronological validation buckets")

    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (features - features.mean(axis=0)) / scale
    global_mean = standardized.mean(axis=0)
    rng = np.random.default_rng(seed)
    best_ids: tuple[int, ...] | None = None
    best_score: tuple[float, float, tuple[int, ...]] | None = None
    best_distance: np.ndarray | None = None
    for _ in range(candidate_count):
        selected_positions = np.concatenate(
            [
                rng.choice(bucket, size=slot, replace=False)
                for bucket, slot in zip(buckets, slots, strict=True)
            ]
        )
        candidate_ids = tuple(sorted(ids[selected_positions].tolist()))
        distance = np.abs(standardized[selected_positions].mean(axis=0) - global_mean)
        score = (float(distance.max()), float(distance.mean()), candidate_ids)
        if best_score is None or score < best_score:
            best_score = score
            best_ids = candidate_ids
            best_distance = distance
    assert best_ids is not None and best_score is not None and best_distance is not None
    return list(best_ids), {
        "policy": "chronological_buckets_distribution_balanced",
        "seed": seed,
        "candidate_count": candidate_count,
        "chronological_bucket_count": bucket_count,
        "episodes_per_bucket": slots,
        "balance_feature_names": list(BALANCE_FEATURE_NAMES),
        "max_absolute_standardized_mean_difference": best_score[0],
        "mean_absolute_standardized_mean_difference": best_score[1],
        "absolute_standardized_mean_difference": dict(
            zip(BALANCE_FEATURE_NAMES, best_distance.tolist(), strict=True)
        ),
    }


def _load_info(root: Path) -> dict[str, Any]:
    from vla.dataset_metadata import load_info
    return load_info(root)


def validated_absolute_action_metadata(info: dict[str, Any]) -> dict[str, Any]:
    metadata = info.get("evo_rlt")
    if not isinstance(metadata, dict):
        raise ValueError("SFT source metadata must contain an evo_rlt object")
    if metadata.get("actions_stored_as") != "absolute":
        raise ValueError("SFT source evo_rlt.actions_stored_as must be 'absolute'")
    if metadata.get("action_stats") != "absolute":
        raise ValueError("SFT source evo_rlt.action_stats must be 'absolute'")
    if metadata.get("action_chunk_size") != 50:
        raise ValueError("SFT source evo_rlt.action_chunk_size must be 50")
    return deepcopy(metadata)


def inherit_vla_metadata(source_info: dict[str, Any], split_root: Path) -> None:
    """Restore custom metadata omitted by LeRobot's split_dataset helper."""
    source_metadata = validated_absolute_action_metadata(source_info)
    split_root = Path(split_root)
    split_info = _load_info(split_root)
    existing = split_info.get("evo_rlt")
    if existing is not None and existing != source_metadata:
        raise ValueError(f"materialized split has conflicting vla metadata: {split_root}")
    split_info["evo_rlt"] = source_metadata
    write_json_atomic(split_root / "meta/info.json", split_info)
    from vla.dataset_metadata import write_metadata_sidecar
    write_metadata_sidecar(split_root, split_info)


def _feature_names(info: dict[str, Any], key: str) -> tuple[str, ...]:
    feature = info.get("features", {}).get(key, {})
    names = tuple(feature.get("names") or ())
    if list(feature.get("shape") or ()) != [len(names)]:
        raise ValueError(f"{key} must be a named one-dimensional feature")
    return names


def _probe_video(path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_read_frames",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("ffprobe is required for SFT video auditing") from error
    except subprocess.CalledProcessError as error:
        raise ValueError(f"ffprobe rejected video {path}: {error.stderr.strip()}") from error
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"video must contain exactly one video stream: {path}")
    return streams[0]


def audit_source_dataset(
    root: Path,
    *,
    expected_camera_keys: Sequence[str] = EXPECTED_CAMERA_KEYS,
    verify_videos: bool = True,
    tolerance_s: float = 0.04,
) -> tuple[list[int], np.ndarray, dict[str, Any]]:
    root = Path(root).expanduser().resolve()
    info = _load_info(root)
    validated_absolute_action_metadata(info)
    if info.get("codebase_version") != "v3.0":
        raise ValueError("SFT source must be a LeRobot v3.0 dataset")
    if _feature_names(info, "observation.state") != EXPECTED_STATE_NAMES:
        raise ValueError("SFT source does not use the expected Franka 10D state schema")
    if _feature_names(info, "action") != EXPECTED_ACTION_NAMES:
        raise ValueError("SFT source does not use the expected Franka 10D action schema")
    camera_keys = tuple(
        key for key, value in info["features"].items() if value.get("dtype") in {"video", "image"}
    )
    if camera_keys != tuple(expected_camera_keys):
        raise ValueError(
            f"SFT camera order mismatch: expected={tuple(expected_camera_keys)}, actual={camera_keys}"
        )
    fps = int(info["fps"])
    episode_ids: list[int] = []
    feature_rows: list[np.ndarray] = []
    episode_audits: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path in sorted((root / "data").glob("*/*.parquet")):
        frame = pd.read_parquet(path)
        required = {"observation.state", "action", "timestamp", "episode_index", "frame_index"}
        if missing := sorted(required - set(frame.columns)):
            raise ValueError(f"data file {path} is missing columns {missing}")
        for episode_id in sorted(int(value) for value in frame["episode_index"].unique()):
            if episode_id in seen:
                raise ValueError(f"episode {episode_id} spans multiple parquet files")
            seen.add(episode_id)
            episode = frame[frame["episode_index"] == episode_id].sort_values("frame_index")
            states = np.stack(episode["observation.state"].to_numpy()).astype(np.float64)
            actions = np.stack(episode["action"].to_numpy()).astype(np.float64)
            if states.shape != (len(episode), 10) or actions.shape != (len(episode), 10):
                raise ValueError(f"episode {episode_id} has an invalid state/action shape")
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise ValueError(f"episode {episode_id} contains NaN or infinity")
            _rot6d_matrices(states[:, 3:9])
            feature, episode_audit = episode_balance_features(actions)
            timestamps = episode["timestamp"].to_numpy(dtype=np.float64)
            if len(timestamps) < 50 or np.any(np.diff(timestamps) <= 0):
                raise ValueError(f"episode {episode_id} is too short or has non-monotonic timestamps")
            timestamp_error = np.abs(np.diff(timestamps) - 1.0 / fps)
            if np.any(timestamp_error > tolerance_s):
                raise ValueError(f"episode {episode_id} timestamps exceed tolerance_s={tolerance_s}")
            episode_audits.append(
                {
                    "episode_index": episode_id,
                    "timestamp_error_max_s": float(timestamp_error.max()),
                    **episode_audit,
                }
            )
            episode_ids.append(episode_id)
            feature_rows.append(feature)
    if len(episode_ids) != int(info.get("total_episodes", -1)):
        raise ValueError(
            f"dataset episode count mismatch: metadata={info.get('total_episodes')}, data={len(episode_ids)}"
        )
    if sum(item["frames"] for item in episode_audits) != int(info.get("total_frames", -1)):
        raise ValueError("dataset frame count does not match metadata")

    video_report = {"verified": verify_videos, "file_count": 0}
    video_paths = sorted((root / "videos").glob("*/*/*.mp4"))
    expected_video_count = len(episode_ids) * len(camera_keys)
    if len(video_paths) != expected_video_count:
        raise ValueError(
            f"expected {expected_video_count} per-episode videos, found {len(video_paths)}"
        )
    if verify_videos:
        episode_lengths = {int(item["episode_index"]): int(item["frames"]) for item in episode_audits}
        for video_path in video_paths:
            stream = _probe_video(video_path)
            camera_key = video_path.parents[1].name
            expected_shape = info["features"][camera_key]["shape"]
            if [int(stream["height"]), int(stream["width"]), 3] != expected_shape:
                raise ValueError(f"video resolution does not match metadata: {video_path}")
            numerator, denominator = (int(value) for value in stream["r_frame_rate"].split("/"))
            if denominator == 0 or not math.isclose(numerator / denominator, fps, abs_tol=0.01):
                raise ValueError(f"video FPS does not match metadata: {video_path}")
            episode_id = int(video_path.stem.removeprefix("file-"))
            frame_count = int(stream["nb_read_frames"])
            if episode_id not in episode_lengths or frame_count != episode_lengths[episode_id]:
                raise ValueError(
                    f"video frame count does not match episode {episode_id}: "
                    f"expected={episode_lengths.get(episode_id)}, actual={frame_count}"
                )
    video_report["file_count"] = len(video_paths)
    audit = {
        "status": "ok",
        "schema": {
            "state_names": list(EXPECTED_STATE_NAMES),
            "action_names": list(EXPECTED_ACTION_NAMES),
            "camera_keys": list(camera_keys),
            "fps": fps,
        },
        "total_episodes": len(episode_ids),
        "total_frames": sum(item["frames"] for item in episode_audits),
        "videos": video_report,
        "episodes": sorted(episode_audits, key=lambda value: value["episode_index"]),
    }
    order = np.argsort(episode_ids)
    return (
        np.asarray(episode_ids, dtype=np.int64)[order].tolist(),
        np.asarray(feature_rows, dtype=np.float64)[order],
        audit,
    )


def _validate_materialized_split(root: Path, expected_episodes: int, expected_frames: int) -> None:
    info = _load_info(root)
    if int(info.get("total_episodes", -1)) != expected_episodes:
        raise ValueError(f"materialized split has the wrong episode count: {root}")
    if int(info.get("total_frames", -1)) != expected_frames:
        raise ValueError(f"materialized split has the wrong frame count: {root}")
    with (root / "meta/stats.json").open("r", encoding="utf-8") as stream:
        stats = json.load(stream)
    for key in ("observation.state", "action"):
        count = stats.get(key, {}).get("count")
        if count != [expected_frames]:
            raise ValueError(f"materialized {key} stats do not cover only split frames: {root}")


def ensure_episode_metadata_locations(dataset: Any) -> None:
    """Backfill metadata parquet locations used by newer LeRobot split tooling."""
    episodes = dataset.meta.episodes
    if episodes is None:
        raise ValueError("source dataset has no episode metadata")
    location_keys = ("meta/episodes/chunk_index", "meta/episodes/file_index")
    if isinstance(episodes, dict):
        episode_indices = [int(value) for value in episodes]
        if all(all(key in episode for key in location_keys) for episode in episodes.values()):
            return
    else:
        episode_indices = [int(value) for value in episodes["episode_index"]]
        if all(key in episodes.column_names for key in location_keys):
            return

    if len(episode_indices) != len(set(episode_indices)):
        raise ValueError("loaded episode metadata contains duplicate episode indices")

    locations: dict[int, tuple[int, int]] = {}
    metadata_root = Path(dataset.root) / "meta/episodes"
    for path in sorted(metadata_root.glob("chunk-*/*.parquet")):
        chunk_index = int(path.parent.name.removeprefix("chunk-"))
        file_index = int(path.stem.removeprefix("file-"))
        frame = pd.read_parquet(path, columns=["episode_index"])
        for value in frame["episode_index"]:
            episode_index = int(value)
            if episode_index in locations:
                raise ValueError(f"duplicate episode {episode_index} in episode metadata")
            locations[episode_index] = (chunk_index, file_index)

    expected = set(episode_indices)
    if set(locations) != expected:
        raise ValueError(
            "episode metadata parquet rows do not match loaded episodes: "
            f"missing={sorted(expected - set(locations))}, extra={sorted(set(locations) - expected)}"
        )
    if isinstance(episodes, dict):
        for episode_index, episode in episodes.items():
            chunk_index, file_index = locations[int(episode_index)]
            episode[location_keys[0]] = chunk_index
            episode[location_keys[1]] = file_index
        return

    for offset, key in enumerate(location_keys):
        if key not in episodes.column_names:
            episodes = episodes.add_column(key, [locations[value][offset] for value in episode_indices])
    dataset.meta.episodes = episodes


def prepare_franka_sft_split(
    source_root: Path,
    output_root: Path,
    *,
    source_repo_id: str,
    validation_episodes: int = 20,
    seed: int = 1000,
    candidate_count: int = 10_000,
    verify_videos: bool = True,
) -> Path:
    if "HF_HOME" not in os.environ:
        local_hf_home = Path.cwd() / "cache" / "huggingface"
        local_hf_home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(local_hf_home.resolve())

    from lerobot.datasets.dataset_tools import split_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source dataset does not exist: {source_root}")
    if output_root.exists():
        raise FileExistsError(f"split output already exists: {output_root}")
    if source_root == output_root or output_root.is_relative_to(source_root):
        raise ValueError("split output must be separate from the source dataset")

    episode_ids, features, audit = audit_source_dataset(source_root, verify_videos=verify_videos)
    source_info = _load_info(source_root)
    val_ids, selection = choose_validation_episode_ids(
        episode_ids,
        features,
        validation_episodes=validation_episodes,
        seed=seed,
        candidate_count=candidate_count,
    )
    val_set = set(val_ids)
    train_ids = [episode_id for episode_id in episode_ids if episode_id not in val_set]
    episode_frames = {int(item["episode_index"]): int(item["frames"]) for item in audit["episodes"]}

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.split-", dir=output_root.parent))
    staging = temporary_parent / "dataset"
    try:
        dataset = LeRobotDataset(source_repo_id, root=source_root, video_backend="pyav")
        ensure_episode_metadata_locations(dataset)
        created = split_dataset(
            dataset,
            splits={"train": train_ids, "val": val_ids},
            output_dir=staging,
        )
        final_roots = {role: output_root / role for role in ("train", "val")}
        manifest_splits: dict[str, Any] = {}
        for role, original_ids in (("train", train_ids), ("val", val_ids)):
            split_root = staging / role
            expected_frames = sum(episode_frames[episode_id] for episode_id in original_ids)
            inherit_vla_metadata(source_info, split_root)
            _validate_materialized_split(split_root, len(original_ids), expected_frames)
            manifest_splits[role] = {
                "repo_id": created[role].repo_id,
                "root": str(final_roots[role].resolve()),
                "original_episode_ids": original_ids,
                "original_to_split_episode": {
                    str(original_id): split_id for split_id, original_id in enumerate(sorted(original_ids))
                },
                "total_episodes": len(original_ids),
                "total_frames": expected_frames,
                "fingerprint": fingerprint_tree(split_root),
            }
        manifest = {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "kind": "franka_pi05_sft_split",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "repo_id": source_repo_id,
                "root": str(source_root),
                "fingerprint": fingerprint_tree(source_root),
            },
            "configuration": {
                "validation_episodes": validation_episodes,
                "seed": seed,
                "candidate_count": candidate_count,
                "absolute_actions": True,
            },
            "selection": selection,
            "audit": audit,
            "splits": manifest_splits,
        }
        write_json_atomic(staging / "split_manifest.json", manifest)
        if output_root.exists():
            raise FileExistsError(f"split output was created during preparation: {output_root}")
        os.replace(staging, output_root)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)
    return output_root / "split_manifest.json"


def training_schedule(split_manifest: dict[str, Any], batch_size: int = 2) -> dict[str, int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    train_frames = int(split_manifest["splits"]["train"]["total_frames"])
    steps_per_epoch = math.ceil(train_frames / batch_size)
    return {
        "train_frames": train_frames,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "short_steps": math.ceil(0.5 * steps_per_epoch),
        "warmup_steps": math.ceil(0.05 * steps_per_epoch),
        "formal_steps": 2 * steps_per_epoch,
        "save_freq": math.ceil(0.5 * steps_per_epoch),
    }
