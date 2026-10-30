"""Resize every video in a LeRobot v3.0 dataset without modifying the source.

Run from the repository root in the vla environment::

    python -m vla.datacollection.resize_lerobot_videos \
        --source-root datasets/franka_single_left_sft_10d_absolute \
        --output-root datasets/franka_single_left_sft_10d_absolute_224x224
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any


DEFAULT_WIDTH = 224
DEFAULT_HEIGHT = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Source LeRobot v3.0 dataset.")
    parser.add_argument("--output-root", required=True, help="New resized dataset directory.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--crf", type=int, default=18, help="libx264 CRF (lower is higher quality).")
    parser.add_argument("--preset", default="medium", help="libx264 encoding preset.")
    parser.add_argument(
        "--skip-lerobot-validation",
        action="store_true",
        help="Skip final official LeRobotDataset loading (ffprobe validation still runs).",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if path.name == "info.json" and path.parent.name == "meta":
        from vla.dataset_metadata import load_info
        return load_info(path.parent.parent)
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=4, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, path)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,pix_fmt,avg_frame_rate,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected exactly one video stream in {path}, got {len(streams)}")
    stream = streams[0]
    try:
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "codec": str(stream["codec_name"]),
            "pix_fmt": str(stream["pix_fmt"]),
            "fps": Fraction(stream["avg_frame_rate"]),
            "frames": int(stream["nb_read_frames"]),
            "duration": float(stream["duration"]),
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"incomplete video metadata for {path}: {stream}") from exc


def _transcode_video(
    source: Path,
    destination: Path,
    *,
    width: int,
    height: int,
    crf: int,
    preset: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.mp4")
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"scale={width}:{height}:flags=lanczos",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_transcode(source: Path, destination: Path, width: int, height: int) -> None:
    before = _probe_video(source)
    after = _probe_video(destination)
    if (after["width"], after["height"]) != (width, height):
        raise ValueError(f"wrong output resolution for {destination}: {after}")
    if after["codec"] != "h264" or after["pix_fmt"] != "yuv420p":
        raise ValueError(f"unexpected output codec for {destination}: {after}")
    if after["frames"] != before["frames"]:
        raise ValueError(
            f"frame count changed for {destination}: {before['frames']} -> {after['frames']}"
        )
    if after["fps"] != before["fps"]:
        raise ValueError(f"FPS changed for {destination}: {before['fps']} -> {after['fps']}")
    duration_tolerance = max(0.05, float(1 / before["fps"]))
    if abs(after["duration"] - before["duration"]) > duration_tolerance:
        raise ValueError(
            f"duration changed for {destination}: {before['duration']} -> {after['duration']}"
        )


def _video_feature_keys(info: dict[str, Any]) -> tuple[str, ...]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("meta/info.json has no features mapping")
    keys = tuple(key for key, feature in features.items() if feature.get("dtype") == "video")
    if not keys:
        raise ValueError("dataset contains no video features")
    return keys


def _update_info(
    info: dict[str, Any],
    video_keys: tuple[str, ...],
    *,
    width: int,
    height: int,
) -> None:
    original_sizes: dict[str, list[int]] = {}
    for key in video_keys:
        feature = info["features"][key]
        shape = feature.get("shape")
        if not isinstance(shape, list) or len(shape) != 3:
            raise ValueError(f"video feature {key} has an invalid shape: {shape}")
        original_sizes[key] = [int(shape[0]), int(shape[1])]
        feature["shape"] = [height, width, int(shape[2])]
        video_info = feature.setdefault("info", {})
        video_info.update(
            {
                "video.height": height,
                "video.width": width,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
            }
        )
    info.setdefault("evo_rlt", {})["video_resize"] = {
        "source_sizes": original_sizes,
        "target_size": [height, width],
        "method": "ffmpeg_lanczos_libx264",
    }


def _validate_with_lerobot(root: Path, info: dict[str, Any], video_keys: tuple[str, ...]) -> None:
    cache = Path(tempfile.mkdtemp(prefix=".hf-validation-cache-", dir=root.parent))
    previous_hf_home = os.environ.get("HF_HOME")
    previous_datasets_cache = os.environ.get("HF_DATASETS_CACHE")
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        digest = hashlib.sha256(str(root).encode()).hexdigest()[:12]
        dataset = LeRobotDataset(repo_id=f"local/resized-{digest}", root=root, video_backend="pyav")
        if dataset.num_frames < 1 or dataset.num_episodes < 1:
            raise ValueError("official LeRobot load returned an empty dataset")
        sample = dataset[0]
        for key in video_keys:
            image = sample.get(key)
            expected = (3, info["features"][key]["shape"][0], info["features"][key]["shape"][1])
            if image is None or tuple(image.shape) != expected:
                raise ValueError(
                    f"official LeRobot load returned {key} shape={getattr(image, 'shape', None)}"
                )
    finally:
        if previous_hf_home is None:
            os.environ.pop("HF_HOME", None)
        else:
            os.environ["HF_HOME"] = previous_hf_home
        if previous_datasets_cache is None:
            os.environ.pop("HF_DATASETS_CACHE", None)
        else:
            os.environ["HF_DATASETS_CACHE"] = previous_datasets_cache
        shutil.rmtree(cache, ignore_errors=True)


def resize_dataset_videos(
    source_root: str | Path,
    output_root: str | Path,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    crf: int = 18,
    preset: str = "medium",
    validate_lerobot: bool = True,
) -> Path:
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source dataset does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"output dataset already exists: {output}")
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("source-root and output-root must be separate, non-nested directories")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("width and height must be positive even integers for yuv420p")
    if not 0 <= crf <= 51:
        raise ValueError("crf must be between 0 and 51")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH")

    info_path = source / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing LeRobot metadata: {info_path}")
    info = _load_json(info_path)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"expected LeRobot v3.0, got {info.get('codebase_version')!r}")
    video_keys = _video_feature_keys(info)
    videos = sorted((source / "videos").glob("**/*.mp4"))
    if not videos:
        raise ValueError(f"no MP4 files found under {source / 'videos'}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(tempfile.mkdtemp(prefix=f".{output.name}.resize-", dir=output.parent))
    staging = temporary_parent / "dataset"
    staging.mkdir()
    try:
        video_paths = {path.relative_to(source) for path in videos}
        for path in sorted(item for item in source.glob("**/*") if item.is_file()):
            relative = path.relative_to(source)
            if relative not in video_paths:
                _link_or_copy(path, staging / relative)

        total = len(videos)
        for index, video in enumerate(videos, start=1):
            relative = video.relative_to(source)
            destination = staging / relative
            print(f"[{index}/{total}] {relative}", flush=True)
            _transcode_video(
                video,
                destination,
                width=width,
                height=height,
                crf=crf,
                preset=preset,
            )
            _validate_transcode(video, destination, width, height)

        _update_info(info, video_keys, width=width, height=height)
        _write_json_atomic(staging / "meta/info.json", info)
        from vla.dataset_metadata import write_metadata_sidecar
        write_metadata_sidecar(staging, info)
        if validate_lerobot:
            _validate_with_lerobot(staging, info, video_keys)
        if output.exists():
            raise FileExistsError(f"output directory was created during conversion: {output}")
        os.replace(staging, output)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)
    print(f"Resized dataset: {output}")
    print(f"Source unchanged: {source}")
    return output


def main() -> None:
    args = parse_args()
    resize_dataset_videos(
        args.source_root,
        args.output_root,
        width=args.width,
        height=args.height,
        crf=args.crf,
        preset=args.preset,
        validate_lerobot=not args.skip_lerobot_validation,
    )


if __name__ == "__main__":
    main()
