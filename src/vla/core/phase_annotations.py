from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PHASE_ANNOTATIONS_FORMAT_VERSION = 1
PHASE_ANNOTATIONS_NAME = "phase_annotations.json"


def new_phase_annotations(*, dataset_info_sha256: str, fps: float) -> dict[str, Any]:
    if not dataset_info_sha256:
        raise ValueError("dataset_info_sha256 must not be empty")
    if fps <= 0:
        raise ValueError("fps must be positive")
    return {
        "format_version": PHASE_ANNOTATIONS_FORMAT_VERSION,
        "dataset_info_sha256": dataset_info_sha256,
        "fps": float(fps),
        "episodes": {},
    }


def _validate_interval(interval: Any, *, episode_id: int, length: int) -> tuple[int, int]:
    if not isinstance(interval, dict):
        raise ValueError(f"episode {episode_id} interval must be an object")
    try:
        start = interval["start_frame"]
        end = interval["end_frame"]
    except KeyError as error:
        raise ValueError(
            f"episode {episode_id} interval requires integer start_frame/end_frame"
        ) from error
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
    ):
        raise ValueError(f"episode {episode_id} interval bounds must be integers")
    if not 0 <= start < end <= length:
        raise ValueError(
            f"episode {episode_id} interval [{start}, {end}) is outside [0, {length})"
        )
    return start, end


def validate_phase_annotations(
    document: dict[str, Any],
    *,
    expected_info_sha256: str | None = None,
    episode_lengths: dict[int, int] | None = None,
    require_reviewed: bool = False,
) -> None:
    if not isinstance(document, dict):
        raise ValueError("phase annotations must be a JSON object")
    if document.get("format_version") != PHASE_ANNOTATIONS_FORMAT_VERSION:
        raise ValueError(
            f"unsupported phase annotation format {document.get('format_version')!r}; "
            f"expected {PHASE_ANNOTATIONS_FORMAT_VERSION}"
        )
    fingerprint = document.get("dataset_info_sha256")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("phase annotations require dataset_info_sha256")
    if expected_info_sha256 is not None and fingerprint != expected_info_sha256:
        raise ValueError(
            "phase annotation dataset fingerprint mismatch: "
            f"annotations={fingerprint}, dataset={expected_info_sha256}"
        )
    fps = document.get("fps")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
        raise ValueError("phase annotation fps must be positive")
    episodes = document.get("episodes")
    if not isinstance(episodes, dict):
        raise ValueError("phase annotations require an episodes object")

    for raw_episode_id, annotation in episodes.items():
        try:
            episode_id = int(raw_episode_id)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid phase annotation episode id: {raw_episode_id!r}") from error
        if str(episode_id) != str(raw_episode_id) or episode_id < 0:
            raise ValueError(f"invalid phase annotation episode id: {raw_episode_id!r}")
        if not isinstance(annotation, dict):
            raise ValueError(f"episode {episode_id} annotation must be an object")
        length = annotation.get("length")
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise ValueError(f"episode {episode_id} length must be a positive integer")
        if episode_lengths is not None:
            expected_length = episode_lengths.get(episode_id)
            if expected_length is None:
                raise ValueError(f"phase annotations reference unknown episode {episode_id}")
            if length != expected_length:
                raise ValueError(
                    f"episode {episode_id} length mismatch: annotations={length}, "
                    f"dataset={expected_length}"
                )
        reviewed = annotation.get("reviewed")
        if not isinstance(reviewed, bool):
            raise ValueError(f"episode {episode_id} reviewed must be boolean")
        if require_reviewed and not reviewed:
            raise ValueError(f"episode {episode_id} phase annotation has not been reviewed")
        intervals = annotation.get("intervals")
        if not isinstance(intervals, list):
            raise ValueError(f"episode {episode_id} intervals must be a list")
        previous_end = 0
        for index, interval in enumerate(intervals):
            start, end = _validate_interval(interval, episode_id=episode_id, length=length)
            if index > 0 and start < previous_end:
                raise ValueError(f"episode {episode_id} critical intervals overlap or are unsorted")
            previous_end = end


def load_phase_annotations(
    path: str | Path,
    *,
    expected_info_sha256: str | None = None,
    episode_lengths: dict[int, int] | None = None,
) -> dict[str, Any]:
    annotation_path = Path(path).expanduser().resolve()
    value = json.loads(annotation_path.read_text(encoding="utf-8"))
    validate_phase_annotations(
        value,
        expected_info_sha256=expected_info_sha256,
        episode_lengths=episode_lengths,
    )
    return value


def save_phase_annotations(path: str | Path, document: dict[str, Any]) -> Path:
    validate_phase_annotations(document)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


def set_episode_phase_annotation(
    document: dict[str, Any],
    *,
    episode_id: int,
    length: int,
    intervals: list[tuple[int, int]],
    reviewed: bool,
) -> None:
    ordered = sorted((int(start), int(end)) for start, end in intervals)
    annotation = {
        "length": int(length),
        "reviewed": bool(reviewed),
        "intervals": [
            {"start_frame": start, "end_frame": end}
            for start, end in ordered
        ],
    }
    candidate = {**document, "episodes": {**document.get("episodes", {})}}
    candidate["episodes"][str(int(episode_id))] = annotation
    validate_phase_annotations(candidate)
    document.setdefault("episodes", {})[str(int(episode_id))] = annotation


def episode_phase_intervals(
    document: dict[str, Any],
    *,
    episode_id: int,
    length: int,
    require_reviewed: bool = True,
) -> list[tuple[int, int]]:
    annotation = document.get("episodes", {}).get(str(episode_id))
    if annotation is None:
        raise ValueError(f"episode {episode_id} has no phase annotation")
    if int(annotation.get("length", -1)) != length:
        raise ValueError(
            f"episode {episode_id} length mismatch: annotations={annotation.get('length')}, "
            f"dataset={length}"
        )
    if require_reviewed and annotation.get("reviewed") is not True:
        raise ValueError(f"episode {episode_id} phase annotation has not been reviewed")
    return [
        (int(interval["start_frame"]), int(interval["end_frame"]))
        for interval in annotation.get("intervals", [])
    ]


def overlay_episode_phase_annotations(
    frame_annotations: dict[int, dict[str, float | int]] | None,
    document: dict[str, Any],
    *,
    episode_id: int,
    episode_start: int,
    episode_stop: int,
) -> dict[int, dict[str, float | int]]:
    length = episode_stop - episode_start
    intervals = episode_phase_intervals(
        document,
        episode_id=episode_id,
        length=length,
        require_reviewed=True,
    )
    result = {
        frame_index: dict(values)
        for frame_index, values in (frame_annotations or {}).items()
    }
    interval_index = 0
    for local_frame in range(length):
        while interval_index < len(intervals) and local_frame >= intervals[interval_index][1]:
            interval_index += 1
        is_critical = (
            interval_index < len(intervals)
            and intervals[interval_index][0] <= local_frame < intervals[interval_index][1]
        )
        result.setdefault(episode_start + local_frame, {})["phase"] = float(is_critical)
    return result
