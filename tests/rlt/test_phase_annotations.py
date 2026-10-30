from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vla.cli.annotate_phases import _render, _select_ui
from vla.cli.phase_annotation_web import WEB_ASSET, _parse_range
from vla.cli.build_transition_cache_v2 import _chunk_annotation
from vla.core.interfaces import TRANSITION_SOURCE_DEMO
from vla.core.phase_annotations import (
    episode_phase_intervals,
    load_phase_annotations,
    new_phase_annotations,
    overlay_episode_phase_annotations,
    save_phase_annotations,
    set_episode_phase_annotation,
)


def _document() -> dict:
    return new_phase_annotations(dataset_info_sha256="dataset-fingerprint", fps=30)


def test_phase_annotations_round_trip_and_overlay_existing_frame_metadata(tmp_path: Path) -> None:
    document = _document()
    set_episode_phase_annotation(
        document,
        episode_id=3,
        length=8,
        intervals=[(2, 5), (7, 8)],
        reviewed=True,
    )
    output = save_phase_annotations(tmp_path / "meta" / "phase_annotations.json", document)
    loaded = load_phase_annotations(
        output,
        expected_info_sha256="dataset-fingerprint",
        episode_lengths={3: 8},
    )

    annotations = overlay_episode_phase_annotations(
        {10: {"collector_policy_id": 2}, 12: {"phase": 0.0}},
        loaded,
        episode_id=3,
        episode_start=10,
        episode_stop=18,
    )

    assert annotations[10] == {"collector_policy_id": 2, "phase": 0.0}
    assert annotations[12]["phase"] == 1.0
    assert [annotations[index]["phase"] for index in range(10, 18)] == [
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
    ]
    vla_annotation = _chunk_annotation(annotations, start_frame=10, chunk_length=2)
    critical_annotation = _chunk_annotation(annotations, start_frame=12, chunk_length=2)
    assert vla_annotation["is_critical"] == 0.0
    assert critical_annotation["is_critical"] == 1.0
    assert critical_annotation["source"] == TRANSITION_SOURCE_DEMO


def test_unreviewed_episode_cannot_be_used_by_cache_builder() -> None:
    document = _document()
    set_episode_phase_annotation(
        document,
        episode_id=0,
        length=5,
        intervals=[],
        reviewed=False,
    )

    with pytest.raises(ValueError, match="has not been reviewed"):
        episode_phase_intervals(document, episode_id=0, length=5)
    with pytest.raises(ValueError, match="has no phase annotation"):
        episode_phase_intervals(document, episode_id=1, length=5)


def test_phase_annotation_contract_rejects_wrong_dataset_and_invalid_intervals(
    tmp_path: Path,
) -> None:
    document = _document()
    set_episode_phase_annotation(
        document,
        episode_id=0,
        length=5,
        intervals=[(1, 3)],
        reviewed=True,
    )
    output = save_phase_annotations(tmp_path / "annotations.json", document)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_phase_annotations(output, expected_info_sha256="different")

    invalid = json.loads(output.read_text())
    invalid["episodes"]["0"]["intervals"] = [
        {"start_frame": 1, "end_frame": 4},
        {"start_frame": 3, "end_frame": 5},
    ]
    output.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="overlap or are unsorted"):
        load_phase_annotations(output)


def test_annotation_view_keeps_status_and_controls_inside_a_stable_canvas() -> None:
    canvas = _render(
        cv2,
        [("observation.images.base", np.zeros((224, 224, 3), dtype=np.uint8))],
        display_height=480,
        episode_id=2,
        episode_position=1,
        episode_count=4,
        frame_index=20,
        frame_count=100,
        fps=30,
        intervals=[(10, 30)],
        pending_start=None,
        reviewed=False,
        playing=False,
    )

    assert canvas.shape == (480, 1050, 3)
    assert np.any(canvas[355:] != 0)


def test_annotation_auto_ui_falls_back_when_opencv_has_no_highgui_backend() -> None:
    assert _select_ui("auto", highgui_backend="", graphical_session=True) == "web"
    assert _select_ui("auto", highgui_backend="GTK3", graphical_session=True) == "opencv"
    assert _select_ui("web", highgui_backend="", graphical_session=False) == "web"
    with pytest.raises(RuntimeError, match="Use --ui web"):
        _select_ui("opencv", highgui_backend="", graphical_session=True)


def test_web_annotator_asset_and_video_ranges() -> None:
    html = WEB_ASSET.read_text(encoding="utf-8")
    assert "VLA Critical Phase Annotator" in html
    assert "/api/bootstrap" in html
    assert _parse_range(None, 100) == (0, 99, False)
    assert _parse_range("bytes=10-19", 100) == (10, 19, True)
    assert _parse_range("bytes=-10", 100) == (90, 99, True)
