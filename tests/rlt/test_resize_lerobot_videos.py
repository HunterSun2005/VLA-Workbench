from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from vla.datacollection.resize_lerobot_videos import _probe_video, resize_dataset_videos


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)


def _make_source(root: Path) -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    video = root / "videos/observation.images.base/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [10]},
            "action": {"dtype": "float32", "shape": [10]},
            "observation.images.base": {
                "dtype": "video",
                "shape": [16, 16, 3],
                "info": {
                    "video.height": 16,
                    "video.width": 16,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.fps": 10,
                },
            },
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "data/chunk-000/file-000.parquet").write_bytes(b"unchanged")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=16x16:rate=10:duration=0.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    return video


def test_resize_dataset_videos_is_transactional_and_preserves_timing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source_video = _make_source(source)
    output = tmp_path / "output"

    resize_dataset_videos(source, output, width=24, height=20, validate_lerobot=False)

    source_info = json.loads((source / "meta/info.json").read_text(encoding="utf-8"))
    output_info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert source_info["features"]["observation.images.base"]["shape"] == [16, 16, 3]
    assert output_info["features"]["observation.images.base"]["shape"] == [20, 24, 3]
    assert output_info["features"]["observation.images.base"]["info"]["video.height"] == 20
    assert output_info["features"]["observation.images.base"]["info"]["video.width"] == 24
    assert output_info["evo_rlt"]["video_resize"]["target_size"] == [20, 24]

    output_video = output / source_video.relative_to(source)
    before = _probe_video(source_video)
    after = _probe_video(output_video)
    assert (after["height"], after["width"]) == (20, 24)
    assert after["frames"] == before["frames"]
    assert after["fps"] == before["fps"]
    assert (output / "data/chunk-000/file-000.parquet").read_bytes() == b"unchanged"

    with pytest.raises(FileExistsError, match="already exists"):
        resize_dataset_videos(source, output, validate_lerobot=False)
