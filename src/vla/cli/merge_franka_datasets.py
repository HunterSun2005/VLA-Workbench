"""Merge compatible LeRobot v3 Franka datasets without mutating the inputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vla.datacollection.franka_robotiq_single_left.v30_writer import (
    DATA_PATH,
    EPISODES_PATH,
    VIDEO_PATH,
    feature_stats,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_info(root: Path) -> dict[str, Any]:
    path = root / "meta/info.json"
    if not path.is_file():
        raise ValueError(f"missing dataset metadata: {path}")
    info = json.loads(path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"{root}: expected LeRobot v3.0 metadata")
    return info


def _episode_rows(root: Path) -> pd.DataFrame:
    path = root / EPISODES_PATH
    if not path.is_file():
        raise ValueError(f"missing episode metadata: {path}")
    rows = pd.read_parquet(path)
    if rows.empty:
        raise ValueError(f"dataset has no episodes: {root}")
    rows = rows.sort_values("episode_index").reset_index(drop=True)
    expected = list(range(len(rows)))
    actual = rows["episode_index"].astype(int).tolist()
    if actual != expected:
        raise ValueError(f"{root}: episode_index must be contiguous from zero, got {actual[:5]!r}")
    return rows


def _camera_keys(info: dict[str, Any]) -> tuple[str, ...]:
    return tuple(key for key, value in info["features"].items() if value.get("dtype") == "video")


def _validate_contract(base: dict[str, Any], candidate: dict[str, Any], root: Path) -> None:
    for key in ("codebase_version", "robot_type", "fps", "chunks_size", "data_path", "video_path", "features"):
        if base.get(key) != candidate.get(key):
            raise ValueError(f"incompatible dataset contract at {root}: {key} differs")
    if candidate.get("total_episodes", 0) <= 0:
        raise ValueError(f"dataset has no committed episodes: {root}")


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ValueError(f"missing committed dataset file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _rewrite_episode_data(source: Path, destination: Path, episode: int, frame_start: int) -> int:
    table = pd.read_parquet(source)
    if len(table) == 0:
        raise ValueError(f"empty episode data: {source}")
    if "episode_index" not in table or "index" not in table:
        raise ValueError(f"episode data lacks index columns: {source}")
    table["episode_index"] = np.int64(episode)
    table["frame_index"] = np.arange(len(table), dtype=np.int64)
    table["index"] = np.arange(frame_start, frame_start + len(table), dtype=np.int64)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    table.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    return len(table)


def _merge_stats(episode_tables: list[pd.DataFrame], info: dict[str, Any], rows: pd.DataFrame) -> dict[str, Any]:
    states = [np.stack(table["observation.state"].to_numpy()) for table in episode_tables]
    actions = [np.stack(table["action"].to_numpy()) for table in episode_tables]
    result: dict[str, Any] = {
        "observation.state": _jsonable(feature_stats(np.concatenate(states))),
        "action": _jsonable(feature_stats(np.concatenate(actions))),
    }
    for camera in _camera_keys(info):
        prefix = f"stats/{camera}/"
        entries = []
        for _, row in rows.iterrows():
            entry = {column[len(prefix):]: np.asarray(value) for column, value in row.items() if column.startswith(prefix)}
            if entry:
                entries.append(entry)
        if entries:
            counts = np.asarray([float(item["count"][0]) for item in entries])
            total = counts.sum()
            result[camera] = {}
            for key in ("min", "max"):
                result[camera][key] = _jsonable(np.min(np.stack([item[key] for item in entries]), axis=0) if key == "min" else np.max(np.stack([item[key] for item in entries]), axis=0))
            means = np.stack([item["mean"] for item in entries])
            result[camera]["mean"] = _jsonable(np.sum(means * counts[:, None], axis=0) / total)
            result[camera]["count"] = [int(total)]
            for key in ("std", "q01", "q10", "q50", "q90", "q99"):
                result[camera][key] = _jsonable(np.average(np.stack([item[key] for item in entries]), axis=0, weights=counts))
    return result


def merge_franka_datasets(sources: list[Path], output_root: Path) -> Path:
    sources = [path.expanduser().resolve() for path in sources]
    output_root = output_root.expanduser().resolve()
    if len(sources) < 2:
        raise ValueError("at least two --source-root arguments are required")
    if output_root.exists():
        raise FileExistsError(f"output directory already exists: {output_root}")
    if output_root in sources:
        raise ValueError("output directory must differ from every source")

    infos = [_load_info(root) for root in sources]
    base_info = infos[0]
    rows_by_source = [_episode_rows(root) for root in sources]
    for root, info in zip(sources[1:], infos[1:], strict=True):
        _validate_contract(base_info, info, root)
    base_tasks = pd.read_parquet(sources[0] / "meta/tasks.parquet")
    for root in sources[1:]:
        tasks = pd.read_parquet(root / "meta/tasks.parquet")
        if not base_tasks.equals(tasks):
            raise ValueError(f"incompatible task table at {root}; task text/index must match")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_root.name + ".tmp.", dir=output_root.parent))
    try:
        (temporary / "meta").mkdir(parents=True)
        shutil.copy2(sources[0] / "meta/tasks.parquet", temporary / "meta/tasks.parquet")
        all_rows: list[pd.DataFrame] = []
        all_tables: list[pd.DataFrame] = []
        episode_offset = 0
        frame_offset = 0
        camera_keys = _camera_keys(base_info)
        for source_index, (root, rows) in enumerate(zip(sources, rows_by_source, strict=True)):
            for _, row in rows.iterrows():
                old_episode = int(row["episode_index"])
                new_episode = episode_offset + old_episode
                old_chunk, old_file = divmod(old_episode, int(base_info["chunks_size"]))
                new_chunk, new_file = divmod(new_episode, int(base_info["chunks_size"]))
                source_data = root / DATA_PATH.format(chunk_index=old_chunk, file_index=old_file)
                destination_data = temporary / DATA_PATH.format(chunk_index=new_chunk, file_index=new_file)
                length = _rewrite_episode_data(source_data, destination_data, new_episode, frame_offset)
                table = pd.read_parquet(destination_data)
                all_tables.append(table)
                for camera in camera_keys:
                    source_video = root / VIDEO_PATH.format(video_key=camera, chunk_index=old_chunk, file_index=old_file)
                    destination_video = temporary / VIDEO_PATH.format(video_key=camera, chunk_index=new_chunk, file_index=new_file)
                    _copy_file(source_video, destination_video)
                updated = row.copy()
                updated["episode_index"] = new_episode
                updated["dataset_from_index"] = frame_offset
                updated["dataset_to_index"] = frame_offset + length
                updated["data/chunk_index"] = new_chunk
                updated["data/file_index"] = new_file
                for camera in camera_keys:
                    updated[f"videos/{camera}/chunk_index"] = new_chunk
                    updated[f"videos/{camera}/file_index"] = new_file
                all_rows.append(pd.DataFrame([updated]))
                frame_offset += length
            episode_offset += len(rows)

        merged_rows = pd.concat(all_rows, ignore_index=True).sort_values("episode_index").reset_index(drop=True)
        episodes_path = temporary / EPISODES_PATH
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        merged_rows.to_parquet(episodes_path, index=False)
        info = dict(base_info)
        info["total_episodes"] = int(episode_offset)
        info["total_frames"] = int(frame_offset)
        info["total_tasks"] = int(len(base_tasks))
        info["splits"] = {"train": f"0:{episode_offset}"}
        (temporary / "meta/info.json").write_text(json.dumps(_jsonable(info), indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
        (temporary / "meta/stats.json").write_text(
            json.dumps(_merge_stats(all_tables, info, merged_rows), indent=4, default=_jsonable) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", type=Path, required=True, help="Input dataset; repeat in append order")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = merge_franka_datasets(args.source_root, args.output_root)
    info = json.loads((result / "meta/info.json").read_text(encoding="utf-8"))
    print(f"Merged dataset: {result}")
    print(f"episodes={info['total_episodes']} frames={info['total_frames']}")


if __name__ == "__main__":
    main()
