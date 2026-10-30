"""Select and reorder state features in an immutable LeRobot v3.0 source dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from vla.paths import workspace
from typing import Any

import numpy as np
import pandas as pd
import yaml

from vla.cli.common import configure_logging

logger = configure_logging(__name__)

DEFAULT_DATASETS_ROOT = workspace() / "datasets"
FEATURE_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "dataset_features"
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
META_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


@dataclass(frozen=True)
class FeatureSelectionConfig:
    source_key: str
    selected_names: tuple[str, ...]
    source_names: tuple[str, ...]
    expected_action_dim: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract selected state features from a LeRobot v3.0 dataset."
    )
    parser.add_argument("--source-root", required=True, help="Source LeRobot v3.0 dataset directory.")
    parser.add_argument("--config", required=True, help="State feature-selection YAML.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="New dataset directory. Defaults to datasets/sft/<source>_<dim>d.",
    )
    parser.add_argument(
        "--prompt",
        "--replace-prompt",
        dest="prompt",
        default=None,
        help="Replace every task prompt in the output dataset with this text.",
    )
    parser.add_argument(
        "--action-stats",
        choices=("absolute", "relative"),
        default="absolute",
        help="Space used only for global action normalization statistics; parquet actions stay absolute.",
    )
    parser.add_argument(
        "--action-chunk-size",
        type=int,
        default=50,
        help="PI0.5 action chunk size used by official relative action statistics.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="Workers for relative statistics.")
    return parser.parse_args()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a YAML mapping")
    return value


def load_feature_config(path: str | Path) -> FeatureSelectionConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"feature config does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        raw = _require_mapping(yaml.safe_load(stream), "feature config")
    if raw.get("version") != 1:
        raise ValueError("feature config version must be 1")
    state = _require_mapping(raw.get("state"), "state")
    source_key = state.get("source_key", "observation.state")
    if source_key != "observation.state":
        raise ValueError("state.source_key must be observation.state")
    components = _require_mapping(state.get("components"), "state.components")
    component_features = _require_mapping(state.get("component_features"), "state.component_features")
    output_order = state.get("output_order")
    if not isinstance(output_order, list) or not output_order or not all(isinstance(item, str) for item in output_order):
        raise ValueError("state.output_order must be a non-empty list of component names")
    if len(output_order) != len(set(output_order)):
        raise ValueError("state.output_order must not contain duplicates")
    if set(output_order) != set(components) or set(components) != set(component_features):
        raise ValueError("components, component_features, and output_order must use identical keys")
    source_names = []
    selected_names = []
    for component in output_order:
        enabled = components[component]
        names = component_features[component]
        if not isinstance(enabled, bool):
            raise ValueError(f"state.components.{component} must be true or false")
        if not isinstance(names, list) or not names or not all(isinstance(name, str) and name for name in names):
            raise ValueError(f"state.component_features.{component} must be a non-empty name list")
        source_names.extend(names)
        if enabled:
            selected_names.extend(names)
    if len(source_names) != len(set(source_names)):
        raise ValueError("component feature names must be unique")
    if not selected_names:
        raise ValueError("at least one state component must be enabled")
    action = _require_mapping(raw.get("action", {}), "action")
    expected_action_dim = action.get("expected_dim")
    if expected_action_dim is not None and (not isinstance(expected_action_dim, int) or expected_action_dim <= 0):
        raise ValueError("action.expected_dim must be a positive integer or null")
    return FeatureSelectionConfig(
        source_key=source_key,
        selected_names=tuple(selected_names),
        source_names=tuple(source_names),
        expected_action_dim=expected_action_dim,
    )


def _load_info(root: Path) -> dict[str, Any]:
    from vla.core.online_contract import reject_unaligned_offline_dataset
    with (root / "meta/info.json").open(encoding="utf-8") as stream:
        info = json.load(stream)
    reject_unaligned_offline_dataset(info)
    return info


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=4)
        stream.write("\n")
    os.replace(temporary, path)


def _validate_paths(source: Path, output: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"source dataset does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("source-root and output-root must be separate, non-nested directories")


def _normalize_prompt(prompt: str | None) -> str | None:
    if prompt is None:
        return None
    normalized = prompt.strip()
    if not normalized:
        raise ValueError("prompt must contain non-whitespace text")
    if "\x00" in normalized:
        raise ValueError("prompt must not contain NUL characters")
    return normalized


def _default_output(source: Path, config: FeatureSelectionConfig) -> Path:
    name = source.parent.name if source.name in {"dataset", "lerobot_dataset"} else source.name
    return DEFAULT_DATASETS_ROOT / "sft" / f"{name}_{len(config.selected_names)}d"


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _validate_source(info: dict[str, Any], config: FeatureSelectionConfig) -> list[int]:
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"source must be LeRobot v3.0, got {info.get('codebase_version')!r}")
    features = info.get("features", {})
    if config.source_key not in features or "action" not in features:
        raise ValueError("source must contain observation.state and action")
    state = features[config.source_key]
    names = state.get("names")
    if not isinstance(names, list) or list(state.get("shape", [])) != [len(names)]:
        raise ValueError("observation.state must be a named 1D feature")
    if tuple(names) != config.source_names:
        missing = sorted(set(config.source_names) - set(names))
        extra = sorted(set(names) - set(config.source_names))
        raise ValueError(
            "source state schema does not exactly match the YAML declaration; "
            f"missing={missing}, extra={extra}, order_matches={tuple(names) == config.source_names}"
        )
    action_shape = list(features["action"].get("shape", []))
    if config.expected_action_dim is not None and action_shape != [config.expected_action_dim]:
        raise ValueError(f"expected {config.expected_action_dim}D action, got {action_shape}")
    return [names.index(name) for name in config.selected_names]


def feature_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    stats = {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": np.array([len(values)], dtype=np.int64),
    }
    for quantile in QUANTILES:
        stats[f"q{int(quantile * 100):02d}"] = np.quantile(values, quantile, axis=0)
    return stats


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _episode_metadata_paths(root: Path) -> list[Path]:
    return sorted((root / "meta/episodes").glob("*/*.parquet"))


def _local_repo_id(root: Path) -> str:
    """Use a root-specific id so Hugging Face never reuses another local dataset cache."""
    digest = hashlib.sha1(str(root.resolve()).encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return f"local/vla-{digest}"


def _rewrite_data_and_episode_stats(
    root: Path,
    selected_indices: list[int],
    prompt: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    all_states = []
    all_actions = []
    per_episode: dict[int, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    for path in sorted((root / "data").glob("*/*.parquet")):
        frame = pd.read_parquet(path)
        if "observation.state" not in frame or "action" not in frame or "episode_index" not in frame:
            raise ValueError(f"{path} lacks required state/action/episode_index columns")
        if prompt is not None and "task_index" not in frame:
            raise ValueError(f"{path} lacks task_index required for prompt replacement")
        source_state = np.stack(frame["observation.state"].to_numpy()).astype(np.float32, copy=False)
        selected_state = source_state[:, selected_indices]
        actions = np.stack(frame["action"].to_numpy()).astype(np.float32, copy=False)
        frame["observation.state"] = list(selected_state)
        if prompt is not None:
            frame["task_index"] = np.int64(0)
        temporary = path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
        all_states.append(selected_state)
        all_actions.append(actions)
        for episode_index in sorted(frame["episode_index"].unique()):
            mask = frame["episode_index"].to_numpy() == episode_index
            per_episode[int(episode_index)] = (feature_stats(selected_state[mask]), feature_stats(actions[mask]))
    if not all_states:
        raise ValueError(f"no parquet files found under {root / 'data'}")
    for path in _episode_metadata_paths(root):
        episodes = pd.read_parquet(path)
        if prompt is not None:
            if "tasks" not in episodes:
                raise ValueError(f"{path} lacks tasks required for prompt replacement")
            episodes["tasks"] = [[prompt] for _ in range(len(episodes))]
        drop = [
            column
            for column in episodes.columns
            if column.startswith("stats/observation.state/") or column.startswith("stats/action/")
        ]
        episodes = episodes.drop(columns=drop)
        for stat_key in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
            episodes[f"stats/observation.state/{stat_key}"] = [
                _jsonable(per_episode[int(index)][0][stat_key]) for index in episodes["episode_index"]
            ]
            episodes[f"stats/action/{stat_key}"] = [
                _jsonable(per_episode[int(index)][1][stat_key]) for index in episodes["episode_index"]
            ]
        temporary = path.with_suffix(".parquet.tmp")
        episodes.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    return np.concatenate(all_states), np.concatenate(all_actions)


def _relative_action_stats(root: Path, chunk_size: int, num_workers: int) -> dict[str, np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("action-chunk-size must be positive")
    from lerobot.datasets.compute_stats import compute_relative_action_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=_local_repo_id(root), root=root, video_backend="pyav")
    state_names = dataset.meta.features["observation.state"].get("names")
    action_names = dataset.meta.features["action"].get("names")
    action_dim = len(action_names or [])
    if action_dim < 10 or action_dim % 10 != 0 or len(state_names or []) < action_dim:
        raise ValueError(
            "relative action stats require aligned 10D TCP blocks in state and action"
        )
    gripper_indices = list(range(9, action_dim, 10))
    if any("gripper" not in str(action_names[index]).lower() for index in gripper_indices):
        raise ValueError(
            f"relative action stats require gripper dimensions at indices {gripper_indices}"
        )
    return compute_relative_action_stats(
        dataset.hf_dataset,
        dataset.meta.features,
        chunk_size=chunk_size,
        exclude_joints=["gripper"],
        num_workers=num_workers,
    )


def _update_metadata(
    root: Path,
    config: FeatureSelectionConfig,
    states: np.ndarray,
    actions: np.ndarray,
    action_stats: str,
    chunk_size: int,
    num_workers: int,
    prompt: str | None,
) -> None:
    info = _load_info(root)
    info["features"]["observation.state"] = {
        **info["features"]["observation.state"],
        "shape": [len(config.selected_names)],
        "names": list(config.selected_names),
    }
    info.setdefault("vla", {}).update(
        {
            "action_stats": action_stats,
            "action_chunk_size": chunk_size,
            "actions_stored_as": "absolute",
        }
    )
    if prompt is not None:
        info["total_tasks"] = 1
        tasks = pd.DataFrame(
            {"task": [prompt], "task_index": np.array([0], dtype=np.int64)}
        ).set_index("task")
        tasks_path = root / "meta/tasks.parquet"
        temporary = tasks_path.with_suffix(".parquet.tmp")
        tasks.to_parquet(temporary)
        os.replace(temporary, tasks_path)
    _write_json(root / "meta/info.json", info)
    stats_path = root / "meta/stats.json"
    existing = {}
    if stats_path.exists():
        with stats_path.open(encoding="utf-8") as stream:
            existing = json.load(stream)
    existing["observation.state"] = _jsonable(feature_stats(states))
    existing["action"] = _jsonable(feature_stats(actions))
    _write_json(stats_path, existing)
    if action_stats == "relative":
        existing["action"] = _jsonable(_relative_action_stats(root, chunk_size, num_workers))
        _write_json(stats_path, existing)


def validate_dataset(
    root: Path,
    state_dim: int,
    action_dim: int,
    expected_prompt: str | None = None,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=_local_repo_id(root),
        root=root,
        delta_timestamps={"action": [index / dataset_fps(root) for index in range(50)]},
        video_backend="pyav",
    )
    if dataset.num_frames < 1 or dataset.num_episodes < 1:
        raise ValueError("extracted dataset is empty")
    sample = dataset[0]
    if tuple(sample["observation.state"].shape) != (state_dim,):
        raise ValueError("official LeRobot load returned an unexpected state shape")
    if tuple(sample["action"].shape) != (50, action_dim):
        raise ValueError("official LeRobot load returned an unexpected action chunk shape")
    if not sample.get("task"):
        raise ValueError("official LeRobot load returned no task text")
    if expected_prompt is not None and sample["task"] != expected_prompt:
        raise ValueError(
            "official LeRobot load returned an unexpected task prompt: "
            f"expected={expected_prompt!r}, actual={sample['task']!r}"
        )


def dataset_fps(root: Path) -> int:
    return int(_load_info(root)["fps"])


def extract_dataset_features(
    source_root: str | Path,
    config_path: str | Path,
    output_root: str | Path | None = None,
    *,
    action_stats: str = "absolute",
    action_chunk_size: int = 50,
    num_workers: int = 0,
    prompt: str | None = None,
    validate: bool = True,
) -> Path:
    source = Path(source_root).expanduser().resolve()
    config = load_feature_config(config_path)
    prompt = _normalize_prompt(prompt)
    output = _default_output(source, config) if output_root is None else Path(output_root).expanduser().resolve()
    _validate_paths(source, output)
    if action_stats not in {"absolute", "relative"}:
        raise ValueError("action_stats must be absolute or relative")
    info = _load_info(source)
    selected_indices = _validate_source(info, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(tempfile.mkdtemp(prefix=f".{output.name}.extract-", dir=output.parent))
    staging = temporary_parent / "dataset"
    try:
        shutil.copytree(source, staging, copy_function=_link_or_copy)
        states, actions = _rewrite_data_and_episode_stats(staging, selected_indices, prompt)
        if actions.shape[1] >= 9:
            # One arm may hold an object while the other moves. Check all TCP
            # blocks before rejecting a dual dataset as stationary.
            offsets = range(0, actions.shape[1], 10) if actions.shape[1] in (10, 20) else (0,)
            spans = [
                (float(np.linalg.norm(np.ptp(actions[:, offset:offset+3], axis=0))),
                 float(np.linalg.norm(np.ptp(actions[:, offset+3:offset+9], axis=0))))
                for offset in offsets
            ]
            if all(translation < 5e-3 and rotation < 1e-2 for translation, rotation in spans):
                raise ValueError(
                    "source action TCP pose is degenerate: "
                    f"per-arm (translation_m, rot6d) spans={spans}; refusing to build a training dataset"
                )
        _update_metadata(
            staging,
            config,
            states,
            actions,
            action_stats,
            action_chunk_size,
            num_workers,
            prompt,
        )
        if validate:
            validate_dataset(
                staging,
                len(config.selected_names),
                actions.shape[1],
                expected_prompt=prompt,
            )
        if output.exists():
            raise FileExistsError(f"output directory was created during extraction: {output}")
        os.replace(staging, output)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)
    logger.info("Extracted dataset written to %s; source unchanged at %s", output, source)
    return output


def main() -> None:
    args = parse_args()
    extract_dataset_features(
        args.source_root,
        args.config,
        args.output_root,
        action_stats=args.action_stats,
        action_chunk_size=args.action_chunk_size,
        num_workers=args.num_workers,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    main()
