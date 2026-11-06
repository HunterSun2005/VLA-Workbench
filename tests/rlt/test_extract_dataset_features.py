from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from vla.datacollection.franka_robotiq_single_left.v30_writer import V30DatasetWriter
from vla.cli import extract_dataset_features as extraction


def _image_stats(count: int) -> dict[str, np.ndarray]:
    stats = {
        key: np.full((3, 1, 1), 0.5, dtype=np.float32)
        for key in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
    }
    stats["count"] = np.array([count])
    return stats


def _make_dataset(root: Path, *, episodes: int = 2, length: int = 55, state_fields=None) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the video compatibility fixture")
    camera = "observation.images.base"
    from vla.datacollection.franka_robotiq_single_left.schema import STATE_FIELDS
    state_fields = STATE_FIELDS if state_fields is None else state_fields
    writer = V30DatasetWriter(root, 30, "synthetic absolute task", (camera,), 16, 16, state_fields=state_fields)
    all_states = []
    for episode_index in range(episodes):
        video = writer.video_path(episode_index, camera)
        video.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                "-i", "color=c=red:s=16x16:r=30", "-frames:v", str(length),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
            ],
            check=True,
        )
        states = np.arange(length * len(state_fields), dtype=np.float32).reshape(length, len(state_fields)) / 100 + episode_index
        actions = states[:, :10].copy()
        actions[:, :9] += 0.2
        actions[:, 9] = np.linspace(0.0, 1.0, length)
        writer.save_episode(states, actions, {camera: _image_stats(length)})
        all_states.append(states)
    return np.concatenate(all_states)


@pytest.mark.parametrize(
    ("config_name", "expected_dim"),
    [
        ("franka_single_left_tcp_rot6d_10d.yaml", 10),
        ("franka_single_left_tcp_rot6d_22d.yaml", 22),
    ],
)
def test_extractor_selects_state_and_official_lerobot_loads_video(
    tmp_path: Path, config_name: str, expected_dim: int
) -> None:
    source = tmp_path / "source"
    source_states = _make_dataset(source)
    source_info_before = (source / "meta/info.json").read_bytes()
    output = tmp_path / f"output-{expected_dim}d"

    result = extraction.extract_dataset_features(
        source,
        extraction.FEATURE_CONFIG_DIR / config_name,
        output,
        validate=True,
    )

    assert result == output
    assert (source / "meta/info.json").read_bytes() == source_info_before
    info = json.loads((output / "meta/info.json").read_text())
    assert info["features"]["observation.state"]["shape"] == [expected_dim]
    assert info["features"]["action"]["shape"] == [10]
    assert info["evo_rlt"] == {
        "action_stats": "absolute",
        "action_chunk_size": 50,
        "actions_stored_as": "absolute",
    }
    data = pd.concat(pd.read_parquet(path) for path in sorted((output / "data").glob("*/*.parquet")))
    selected = np.stack(data["observation.state"].to_numpy())
    np.testing.assert_array_equal(selected, source_states[:, :expected_dim])
    actions = np.stack(data["action"].to_numpy())
    assert actions.shape[1] == 10
    assert (output / "videos/observation.images.base/chunk-000/file-000.mp4").is_file()
    episodes = pd.read_parquet(output / "meta/episodes/chunk-000/file-000.parquet")
    assert episodes["length"].tolist() == [55, 55]
    assert "stats/observation.state/q99" in episodes


def test_legacy_relative_stats_refused_without_command_contract(tmp_path):
    source = tmp_path/'source'
    _make_dataset(source)
    before = [path.read_bytes() for path in sorted((source/'data').glob('*/*.parquet'))]
    with pytest.raises(ValueError, match='explicit VLA action contract'):
        extraction.extract_dataset_features(source,
            extraction.FEATURE_CONFIG_DIR/'franka_single_left_tcp_rot6d_10d.yaml', tmp_path/'relative',
            action_stats='relative', action_chunk_size=50, validate=False)
    assert before == [path.read_bytes() for path in sorted((source/'data').glob('*/*.parquet'))]


def test_extractor_replaces_prompt_everywhere(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_dataset(source)
    source_tasks_before = (source / "meta/tasks.parquet").read_bytes()
    output = tmp_path / "replaced-prompt"
    prompt = "Place the white lid firmly onto the base."

    extraction.extract_dataset_features(
        source,
        extraction.FEATURE_CONFIG_DIR / "franka_single_left_tcp_rot6d_10d.yaml",
        output,
        prompt=prompt,
        validate=True,
    )

    assert (source / "meta/tasks.parquet").read_bytes() == source_tasks_before
    info = json.loads((output / "meta/info.json").read_text())
    assert info["total_tasks"] == 1
    tasks = pd.read_parquet(output / "meta/tasks.parquet")
    assert tasks.index.name == "task"
    assert tasks.index.tolist() == [prompt]
    assert tasks["task_index"].tolist() == [0]
    for path in sorted((output / "data").glob("*/*.parquet")):
        assert pd.read_parquet(path, columns=["task_index"])["task_index"].unique().tolist() == [0]
    for path in sorted((output / "meta/episodes").glob("*/*.parquet")):
        episodes = pd.read_parquet(path, columns=["tasks"])
        assert episodes["tasks"].tolist() == [[prompt]] * len(episodes)


def test_extractor_rejects_degenerate_tcp_actions(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_dataset(source, episodes=1)
    path = next((source / "data").glob("*/*.parquet"))
    frame = pd.read_parquet(path)
    actions = np.stack(frame["action"].to_numpy())
    actions[:, :9] = actions[0, :9]
    frame["action"] = list(actions)
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="source action TCP pose is degenerate"):
        extraction.extract_dataset_features(
            source,
            extraction.FEATURE_CONFIG_DIR / "franka_single_left_tcp_rot6d_10d.yaml",
            tmp_path / "output",
            validate=False,
        )


def test_relative_processor_changes_pose_only_and_restores_absolute() -> None:
    from lerobot.processor.relative_action_processor import to_absolute_actions, to_relative_actions

    state = torch.arange(10, dtype=torch.float32).reshape(1, 10)
    absolute = state[:, None, :].repeat(1, 50, 1) + 0.5
    mask = [True] * 9 + [False]
    relative = to_relative_actions(absolute, state, mask)
    torch.testing.assert_close(relative[..., :9], torch.full((1, 50, 9), 0.5))
    torch.testing.assert_close(relative[..., 9], absolute[..., 9])
    torch.testing.assert_close(to_absolute_actions(relative, state, mask), absolute)


def test_extractor_refuses_existing_output_and_bad_source_schema(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_dataset(source, episodes=1)
    output = tmp_path / "existing"
    output.mkdir()
    config = extraction.FEATURE_CONFIG_DIR / "franka_single_left_tcp_rot6d_10d.yaml"
    with pytest.raises(FileExistsError, match="already exists"):
        extraction.extract_dataset_features(source, config, output)

    empty_prompt_output = tmp_path / "empty-prompt"
    with pytest.raises(ValueError, match="prompt must contain"):
        extraction.extract_dataset_features(
            source,
            config,
            empty_prompt_output,
            prompt="  \n",
        )
    assert not empty_prompt_output.exists()

    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.state"]["names"][0] = "wrong"
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="missing selected fields"):
        extraction.extract_dataset_features(source, config, tmp_path / "new-output")


@pytest.mark.parametrize("width", [29, 31, 50])
def test_named_extraction_accepts_legacy_and_full_sources(tmp_path, width):
    from vla.datacollection.franka_robotiq_single_left.schema import STATE_FIELDS, LEGACY_STATE_FIELDS
    from vla.datacollection.franka_robotiq_single_left.schema_31d import STATE_31D_FIELDS
    fields = {29: LEGACY_STATE_FIELDS, 31: STATE_31D_FIELDS, 50: STATE_FIELDS}[width]
    source = tmp_path / "source"
    states = _make_dataset(source, episodes=1, state_fields=fields)
    output = extraction.extract_dataset_features(source,
        extraction.FEATURE_CONFIG_DIR / "franka_single_left_tcp_rot6d_10d.yaml", tmp_path / "train", validate=False)
    frame = pd.read_parquet(next((output / "data").glob("*/*.parquet")))
    np.testing.assert_array_equal(np.stack(frame["observation.state"]), states[:, :10])


@pytest.mark.parametrize("width", [29, 31])
def test_full_dataset_extracts_original_training_subsets(tmp_path, width):
    from vla.datacollection.franka_robotiq_single_left.schema import STATE_FIELDS, LEGACY_STATE_FIELDS
    from vla.datacollection.franka_robotiq_single_left.schema_31d import STATE_31D_FIELDS
    expected = LEGACY_STATE_FIELDS if width == 29 else list(STATE_31D_FIELDS)
    source = tmp_path / "source"
    states = _make_dataset(source, episodes=1)
    output = extraction.extract_dataset_features(source,
        extraction.FEATURE_CONFIG_DIR / f"franka_single_left_state_{width}d.yaml", tmp_path / "train", validate=True)
    info = json.loads((output / "meta/info.json").read_text())
    assert info['features']['observation.state']['names'] == expected
    frame = pd.read_parquet(next((output / "data").glob("*/*.parquet")))
    np.testing.assert_array_equal(np.stack(frame['observation.state']), states[:, [STATE_FIELDS.index(n) for n in expected]])


def test_dual_full_named_selection_and_duplicate_rejection():
    from vla.datacollection.franka_robotiq_dual.schema import STATE_FIELDS
    config = extraction.load_feature_config(extraction.FEATURE_CONFIG_DIR / "franka_dual_tcp_rot6d_20d.yaml")
    # Source field order is metadata-defined, including when an offline source reorders columns.
    names = list(reversed(STATE_FIELDS))
    info = {'codebase_version': 'v3.0', 'features': {
        'observation.state': {'shape': [100], 'names': names}, 'action': {'shape': [20]}}}
    indices = extraction._validate_source(info, config)
    assert [names[i] for i in indices] == list(STATE_FIELDS[:10] + STATE_FIELDS[50:60])
    names[-1] = names[0]
    with pytest.raises(ValueError, match="unique"):
        extraction._validate_source(info, config)
