"""Small LeRobot v3.0 writer usable from the ROS Python 3.10 environment."""

from __future__ import annotations

from vla.dataset_metadata import load_info, write_metadata_sidecar

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import ACTION_FIELDS, STATE_FIELDS


CODEBASE_VERSION = "v3.0"
CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-000/file-000.parquet"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=4, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _numeric_array(value: Any) -> np.ndarray:
    """Convert nested Arrow/Pandas object arrays without scalar coercion warnings."""
    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value.astype(np.float64, copy=False)
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return np.stack([_numeric_array(item) for item in value])
    return np.asarray(value, dtype=np.float64)


def feature_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if len(values) == 0:
        raise ValueError("cannot compute statistics for an empty feature")
    return {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": np.array([len(values)], dtype=np.int64),
        "q01": np.quantile(values, 0.01, axis=0),
        "q10": np.quantile(values, 0.10, axis=0),
        "q50": np.quantile(values, 0.50, axis=0),
        "q90": np.quantile(values, 0.90, axis=0),
        "q99": np.quantile(values, 0.99, axis=0),
    }


def aggregate_feature_stats(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    counts = np.array([float(item["count"][0]) for item in items])
    total = counts.sum()
    means = np.stack([item["mean"] for item in items])
    variances = np.stack([np.square(item["std"]) for item in items])
    mean = np.sum(means * counts.reshape((-1,) + (1,) * (means.ndim - 1)), axis=0) / total
    weight_shape = (-1,) + (1,) * (variances.ndim - 1)
    variance = np.sum((variances + np.square(means - mean)) * counts.reshape(weight_shape), axis=0) / total
    result = {
        "min": np.min(np.stack([item["min"] for item in items]), axis=0),
        "max": np.max(np.stack([item["max"] for item in items]), axis=0),
        "mean": mean,
        "std": np.sqrt(np.maximum(variance, 0.0)),
        "count": np.array([int(total)], dtype=np.int64),
    }
    # Episode quantiles cannot be merged exactly. These are replaced by exact global
    # quantiles when numeric parquet data is available during metadata rebuild.
    for key in ("q01", "q10", "q50", "q90", "q99"):
        result[key] = np.sum(
            np.stack([item[key] for item in items]) * counts.reshape(weight_shape), axis=0
        ) / total
    return result


def _video_feature(height: int, width: int, fps: int) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }


def make_info(
    fps: int,
    camera_keys: tuple[str, ...],
    image_height: int,
    image_width: int,
    *,
    state_fields: tuple[str, ...] = tuple(STATE_FIELDS),
    action_fields: tuple[str, ...] = tuple(ACTION_FIELDS),
    extra_features: dict[str, dict[str, Any]] | None = None,
    robot_type: str = "franka_robotiq",
    info_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(state_fields)],
            "names": list(state_fields),
        },
        "action": {
            "dtype": "float32",
            "shape": [len(action_fields)],
            "names": list(action_fields),
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for key, feature in (extra_features or {}).items():
        if key in features or key.startswith("observation.images."):
            raise ValueError(f"extra feature collides with a reserved feature: {key}")
        features[key] = feature
    for key in camera_keys:
        features[key] = _video_feature(image_height, image_width, fps)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "chunks_size": CHUNKS_SIZE,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "fps": fps,
        "splits": {},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "evo_rlt": {
            "action_stats": "absolute",
            "action_chunk_size": 50,
            "actions_stored_as": "absolute",
        },
        "features": features,
    }
    if info_metadata:
        info.update(info_metadata)
    return info


def _fixed_array(values: np.ndarray, dtype: str) -> pa.Array:
    numpy_dtype, arrow_dtype = {
        "float32": (np.float32, pa.float32()),
        "float64": (np.float64, pa.float64()),
        "int64": (np.int64, pa.int64()),
    }.get(dtype, (None, None))
    if numpy_dtype is None:
        raise ValueError(f"unsupported fixed-size feature dtype: {dtype}")
    values = np.asarray(values, dtype=numpy_dtype)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f"fixed-size feature must have shape (frames, dim), got {values.shape}")
    # LeRobot represents shape-[1] numeric features as Arrow scalars. Writing
    # them as fixed-size lists makes Dataset.from_parquet reject the dataset.
    if values.shape[1] == 1:
        return pa.array(values[:, 0], type=arrow_dtype)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=arrow_dtype),
        values.shape[1],
    )


def _fixed_float(values: np.ndarray) -> pa.Array:
    return _fixed_array(values, "float32")


def write_episode_data(
    path: Path,
    states: np.ndarray,
    actions: np.ndarray,
    episode_index: int,
    global_start: int,
    task_index: int,
    fps: int,
    *,
    state_dim: int = len(STATE_FIELDS),
    action_dim: int = len(ACTION_FIELDS),
    frame_features: dict[str, np.ndarray] | None = None,
    feature_specs: dict[str, dict[str, Any]] | None = None,
) -> None:
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != state_dim or actions.shape != (len(states), action_dim):
        raise ValueError(f"invalid episode arrays: state={states.shape}, action={actions.shape}")
    length = len(states)
    arrays = [
        _fixed_float(states),
        _fixed_float(actions),
        pa.array(np.arange(length, dtype=np.float32) / np.float32(fps)),
        pa.array(np.arange(length, dtype=np.int64)),
        pa.array(np.full(length, episode_index, dtype=np.int64)),
        pa.array(np.arange(global_start, global_start + length, dtype=np.int64)),
        pa.array(np.full(length, task_index, dtype=np.int64)),
    ]
    names = ["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"]
    for key, values in (frame_features or {}).items():
        spec = (feature_specs or {}).get(key)
        if spec is None:
            raise ValueError(f"missing feature specification for frame feature {key!r}")
        shape = list(spec.get("shape", []))
        if len(shape) != 1 or shape[0] < 1:
            raise ValueError(f"frame feature {key!r} must be a named 1D vector")
        values = np.asarray(values)
        if values.shape != (length, shape[0]):
            raise ValueError(
                f"frame feature {key!r} has shape {values.shape}, expected {(length, shape[0])}"
            )
        arrays.append(_fixed_array(values, str(spec.get("dtype"))))
        names.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_arrays(arrays, names=names), temporary, compression="snappy")
    os.replace(temporary, path)


def repair_committed_metadata(root: Path) -> None:
    """Rebuild derived metadata after an episode-row commit survived a crash."""
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    features = info["features"]
    cameras = tuple(key for key, value in features.items() if value["dtype"] == "video")
    height, width = features[cameras[0]]["shape"][:2]
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    task = str(tasks.index[0])
    V30DatasetWriter(
        root, info["fps"], task, cameras, height, width,
        state_fields=tuple(features["observation.state"]["names"]),
        action_fields=tuple(features["action"]["names"]),
        extra_features={key: value for key, value in features.items() if key.startswith("complementary_info.")},
        robot_type=info["robot_type"],
    )


class V30DatasetWriter:
    """Transactional metadata manager for one-episode-per-file v3.0 datasets."""

    def __init__(
        self,
        root: str | Path,
        fps: int,
        task: str,
        camera_keys: tuple[str, ...],
        image_height: int,
        image_width: int,
        *,
        state_fields: tuple[str, ...] = tuple(STATE_FIELDS),
        action_fields: tuple[str, ...] = tuple(ACTION_FIELDS),
        extra_features: dict[str, dict[str, Any]] | None = None,
        robot_type: str = "franka_robotiq",
        info_metadata: dict[str, Any] | None = None,
        strict_metadata: bool = False,
        allow_multiple_tasks: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.fps = int(fps)
        self.task = task
        self.camera_keys = tuple(camera_keys)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.state_fields = tuple(state_fields)
        self.action_fields = tuple(action_fields)
        self.extra_features = dict(extra_features or {})
        self.robot_type = robot_type
        self.info_metadata = dict(info_metadata or {})
        self.strict_metadata = bool(strict_metadata)
        self.allow_multiple_tasks = bool(allow_multiple_tasks)
        if not self.state_fields or not self.action_fields:
            raise ValueError("state_fields and action_fields must not be empty")
        self.root.mkdir(parents=True, exist_ok=True)
        self.info = self._load_or_initialize_info()
        self._validate_info()
        # Reject incompatible datasets before removing any recovery files.
        self._recover_temporary_files()
        self._rebuild_metadata()

    @property
    def episode_count(self) -> int:
        return int(self.info["total_episodes"])

    @property
    def frame_count(self) -> int:
        return int(self.info["total_frames"])

    @property
    def task_index(self) -> int:
        tasks = self._read_tasks()
        return int(tasks.loc[self.task, "task_index"])

    def chunk_file(self, episode_index: int) -> tuple[int, int]:
        return episode_index // CHUNKS_SIZE, episode_index % CHUNKS_SIZE

    def data_path(self, episode_index: int) -> Path:
        chunk, file_index = self.chunk_file(episode_index)
        return self.root / DATA_PATH.format(chunk_index=chunk, file_index=file_index)

    def video_path(self, episode_index: int, video_key: str) -> Path:
        chunk, file_index = self.chunk_file(episode_index)
        return self.root / VIDEO_PATH.format(video_key=video_key, chunk_index=chunk, file_index=file_index)

    def next_episode_index(self) -> int:
        return self.episode_count

    def _recover_temporary_files(self) -> None:
        for path in self.root.rglob("*.tmp"):
            path.unlink(missing_ok=True)

    def _load_or_initialize_info(self) -> dict[str, Any]:
        path = self.root / "meta/info.json"
        if path.exists():
            return load_info(self.root)
        info = make_info(
            self.fps,
            self.camera_keys,
            self.image_height,
            self.image_width,
            state_fields=self.state_fields,
            action_fields=self.action_fields,
            extra_features=self.extra_features,
            robot_type=self.robot_type,
            info_metadata=self.info_metadata,
        )
        atomic_json(path, info)
        write_metadata_sidecar(self.root, info)
        return info

    def _validate_info(self) -> None:
        if self.info.get("codebase_version") != CODEBASE_VERSION:
            raise RuntimeError(f"expected LeRobot {CODEBASE_VERSION}, got {self.info.get('codebase_version')!r}")
        if int(self.info.get("fps", -1)) != self.fps:
            raise RuntimeError(f"existing dataset uses {self.info.get('fps')} fps, requested {self.fps}")
        actual_cameras = {key for key in self.info.get("features", {}) if key.startswith("observation.images.")}
        if actual_cameras != set(self.camera_keys):
            raise RuntimeError("existing dataset camera features do not match the requested camera schema")
        expected = make_info(
            self.fps,
            self.camera_keys,
            self.image_height,
            self.image_width,
            state_fields=self.state_fields,
            action_fields=self.action_fields,
            extra_features=self.extra_features,
            robot_type=self.robot_type,
            info_metadata=self.info_metadata,
        )["features"]
        if self.strict_metadata and set(self.info.get("features", {})) != set(expected):
            raise RuntimeError(
                "existing dataset feature keys do not match the requested rollout schema"
            )
        for key, feature in expected.items():
            actual = self.info.get("features", {}).get(key)
            if self.strict_metadata and actual != feature:
                raise RuntimeError(
                    f"existing feature {key!r} does not exactly match the requested rollout schema"
                )
            if (
                actual is None
                or list(actual.get("shape", [])) != list(feature["shape"])
                or actual.get("dtype") != feature.get("dtype")
                or actual.get("names") != feature.get("names")
            ):
                raise RuntimeError(f"existing feature {key!r} is missing or has an incompatible shape")
        if self.strict_metadata:
            if self.info.get("robot_type") != self.robot_type:
                raise RuntimeError(
                    f"existing dataset robot_type={self.info.get('robot_type')!r}, "
                    f"requested {self.robot_type!r}"
                )
            for key, expected_value in self.info_metadata.items():
                if self.info.get(key) != expected_value:
                    raise RuntimeError(
                        f"existing dataset metadata {key!r} does not match the current rollout"
                    )
            tasks = self._read_tasks()
            if not self.allow_multiple_tasks and len(tasks) and set(tasks.index) != {self.task}:
                raise RuntimeError(
                    "existing rollout dataset task does not match the requested task: "
                    f"existing={list(tasks.index)!r}, requested={self.task!r}"
                )

    def _read_tasks(self) -> pd.DataFrame:
        path = self.root / "meta/tasks.parquet"
        if path.exists():
            tasks = pd.read_parquet(path)
            tasks.index.name = "task"
            return tasks
        return pd.DataFrame(
            {"task_index": pd.Series(dtype="int64")},
            index=pd.Index([], name="task", dtype="object"),
        )

    def _write_tasks(self, frame: pd.DataFrame) -> None:
        path = self.root / "meta/tasks.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary)
        os.replace(temporary, path)

    def _ensure_task(self, task: str | None = None) -> int:
        task = self.task if task is None else str(task).strip()
        if not task:
            raise ValueError("task must be a nonempty string")
        tasks = self._read_tasks()
        if task not in tasks.index:
            tasks.loc[task, "task_index"] = len(tasks)
            tasks["task_index"] = tasks["task_index"].astype("int64")
            self._write_tasks(tasks)
        return int(tasks.loc[task, "task_index"])

    def task_index_for(self, task: str) -> int:
        """Return the stable task index, adding a task only in multi-task mode."""
        task = str(task).strip()
        if not self.allow_multiple_tasks and task != self.task:
            raise ValueError(f"single-task dataset cannot save task {task!r}; expected {self.task!r}")
        return self._ensure_task(task)

    def _read_episode_rows(self) -> pd.DataFrame:
        path = self.root / EPISODES_PATH
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def _write_episode_rows(self, rows: pd.DataFrame) -> None:
        path = self.root / EPISODES_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        if rows.empty:
            path.unlink(missing_ok=True)
            return
        temporary = path.with_suffix(".parquet.tmp")
        rows.to_parquet(temporary, index=False)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)

    def _rebuild_metadata(self) -> None:
        self._ensure_task()
        rows = self._read_episode_rows()
        tasks = self._read_tasks()
        if not tasks.empty:
            task_indices = tasks["task_index"].astype(int).tolist()
            if sorted(task_indices) != list(range(len(task_indices))):
                raise RuntimeError("task table indices must be contiguous from zero")
        if not rows.empty:
            rows = rows.sort_values("episode_index").reset_index(drop=True)
            expected_indices = list(range(len(rows)))
            if rows["episode_index"].astype(int).tolist() != expected_indices:
                raise RuntimeError("episode metadata is not contiguous from zero")
            for episode_index in expected_indices:
                if not self.data_path(episode_index).is_file():
                    raise RuntimeError(f"episode {episode_index} metadata exists but its data file is missing")
                row = rows.iloc[episode_index]
                values = row.get("tasks", [])
                if hasattr(values, "tolist"):
                    values = values.tolist()
                if isinstance(values, str):
                    values = [values]
                if not isinstance(values, (list, tuple)) or len(values) != 1 or str(values[0]) not in tasks.index:
                    raise RuntimeError(f"episode {episode_index} has an invalid task reference: {values!r}")
                task_index = int(tasks.loc[str(values[0]), "task_index"])
                actual = pd.read_parquet(self.data_path(episode_index), columns=["task_index"])["task_index"]
                if set(actual.astype(int).unique()) != {task_index}:
                    raise RuntimeError(
                        f"episode {episode_index} task_index does not match task {values[0]!r}"
                    )
                for key in self.camera_keys:
                    if not self.video_path(episode_index, key).is_file():
                        raise RuntimeError(f"episode {episode_index} video {key!r} is missing")
        total_frames = int(rows["length"].sum()) if not rows.empty else 0
        self.info.update(
            total_episodes=len(rows),
            total_frames=total_frames,
            total_tasks=len(tasks),
            splits={"train": f"0:{len(rows)}"} if len(rows) else {},
        )
        atomic_json(self.root / "meta/info.json", self.info)
        write_metadata_sidecar(self.root, self.info)
        self._rebuild_global_stats(rows)

    def _rebuild_global_stats(self, rows: pd.DataFrame) -> None:
        if rows.empty:
            (self.root / "meta/stats.json").unlink(missing_ok=True)
            return
        states = []
        actions = []
        for episode_index in rows["episode_index"].astype(int):
            columns = ["observation.state", "action"]
            valid_key = "complementary_info.action_valid"
            if valid_key in self.extra_features:
                columns.append(valid_key)
            table = pd.read_parquet(self.data_path(episode_index), columns=columns)
            states.append(np.stack(table["observation.state"].to_numpy()))
            valid = np.ones(len(table), dtype=bool) if valid_key not in table else np.stack(table[valid_key].to_numpy()).reshape(-1).astype(bool)
            actions.append(np.stack(table["action"].to_numpy())[valid])
        stats = {
            "observation.state": feature_stats(np.concatenate(states)),
            "action": feature_stats(np.concatenate(actions)),
        }
        # Preserve image statistics from episode metadata where possible.
        for key in self.camera_keys:
            prefix = f"stats/{key}/"
            entries = []
            for _, row in rows.iterrows():
                entry = {
                    column[len(prefix):]: _numeric_array(row[column])
                    for column in rows.columns
                    if column.startswith(prefix)
                }
                if entry:
                    entries.append(entry)
            if entries:
                stats[key] = aggregate_feature_stats(entries)
        atomic_json(self.root / "meta/stats.json", _jsonable(stats))

    def save_episode(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        image_stats: dict[str, dict[str, Any]],
        *,
        frame_features: dict[str, np.ndarray] | None = None,
        extra_episode_metadata: dict[str, Any] | None = None,
        task: str | None = None,
    ) -> int:
        episode_index = self.next_episode_index()
        length = len(states)
        if length < 1:
            raise ValueError("cannot save an empty episode")
        if set(frame_features or {}) != set(self.extra_features):
            raise ValueError(
                "frame feature columns do not match the declared extra features: "
                f"provided={sorted(frame_features or {})}, expected={sorted(self.extra_features)}"
            )
        for key in self.camera_keys:
            if not self.video_path(episode_index, key).is_file():
                raise FileNotFoundError(f"video is missing before metadata commit: {self.video_path(episode_index, key)}")
        episode_task = self.task if task is None else str(task).strip()
        task_index = self.task_index_for(episode_task)
        write_episode_data(
            self.data_path(episode_index), states, actions, episode_index, self.frame_count,
            task_index, self.fps,
            state_dim=len(self.state_fields),
            action_dim=len(self.action_fields),
            frame_features=frame_features,
            feature_specs=self.extra_features,
        )
        state_stats = feature_stats(np.asarray(states, dtype=np.float32))
        valid_key = "complementary_info.action_valid"
        valid = np.ones(length, dtype=bool) if valid_key not in (frame_features or {}) else np.asarray(frame_features[valid_key]).reshape(-1).astype(bool)
        if not valid.any():
            raise ValueError("cannot save an episode without complete actions")
        action_stats = feature_stats(np.asarray(actions, dtype=np.float32)[valid])
        chunk, file_index = self.chunk_file(episode_index)
        row: dict[str, Any] = {
            "episode_index": episode_index,
            "tasks": [episode_task],
            "length": length,
            "dataset_from_index": self.frame_count,
            "dataset_to_index": self.frame_count + length,
            "data/chunk_index": chunk,
            "data/file_index": file_index,
        }
        for key, value in (extra_episode_metadata or {}).items():
            if key in row or key.startswith("stats/") or key.startswith("videos/"):
                raise ValueError(f"episode metadata collides with a reserved column: {key}")
            row[key] = _jsonable(value)
        if "episode_uuid" in row:
            from vla.core.online_contract import file_digest, load_run

            run = load_run(self.root)
            for key in ("run_uuid", "dataset_uuid"):
                if row.get(key) != run[key]:
                    raise ValueError(f"online episode {key} mismatch")
            files = [self.data_path(episode_index), *(self.video_path(episode_index, key) for key in self.camera_keys)]
            for path in files:
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
                _sync_directory(path.parent)
            row["online_files"] = json.dumps({path.relative_to(self.root).as_posix(): file_digest(path) for path in files}, sort_keys=True)
        for key in self.camera_keys:
            row[f"videos/{key}/chunk_index"] = chunk
            row[f"videos/{key}/file_index"] = file_index
            row[f"videos/{key}/from_timestamp"] = 0.0
            row[f"videos/{key}/to_timestamp"] = length / self.fps
        for feature, stats in {"observation.state": state_stats, "action": action_stats, **image_stats}.items():
            for stat_name, value in stats.items():
                row[f"stats/{feature}/{stat_name}"] = _jsonable(np.asarray(value))
        rows = self._read_episode_rows()
        rows = pd.concat([rows, pd.DataFrame([row])], ignore_index=True)
        self._write_episode_rows(rows)
        self._rebuild_metadata()
        return episode_index

    def discard_latest_episode(self) -> int | None:
        rows = self._read_episode_rows()
        if rows.empty:
            return None
        episode_index = int(rows["episode_index"].max())
        rows = rows[rows["episode_index"].astype(int) != episode_index].reset_index(drop=True)
        self._write_episode_rows(rows)
        self.data_path(episode_index).unlink(missing_ok=True)
        for key in self.camera_keys:
            self.video_path(episode_index, key).unlink(missing_ok=True)
        self._rebuild_metadata()
        return episode_index

    def discard_uncommitted_files(self, episode_index: int) -> None:
        self.data_path(episode_index).unlink(missing_ok=True)
        for key in self.camera_keys:
            self.video_path(episode_index, key).unlink(missing_ok=True)
