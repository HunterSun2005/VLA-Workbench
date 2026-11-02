from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from vla.datacollection.franka_robotiq_single_left import schema
from vla.datacollection.franka_robotiq_single_left.v30_writer import V30DatasetWriter


def _pose(position, quaternion):
    return SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=position[0], y=position[1], z=position[2]),
            orientation=SimpleNamespace(
                x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]
            ),
        )
    )


def _robot_state():
    zero_vector = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    base_force = SimpleNamespace(x=1.0, y=2.0, z=3.0)
    base_torque = SimpleNamespace(x=4.0, y=5.0, z=6.0)
    return SimpleNamespace(
        measured_joint_state=SimpleNamespace(
            name=[f"left_fr3_joint{index}" for index in range(7, 0, -1)],
            position=[index / 10 for index in range(7, 0, -1)],
            effort=[index for index in range(7, 0, -1)],
            velocity=[index / 100 for index in range(7, 0, -1)],
        ),
        tau_ext_hat_filtered=SimpleNamespace(
            name=[f"left_fr3_joint{index}" for index in range(7, 0, -1)],
            effort=[index / 2 for index in range(7, 0, -1)]),
        o_t_ee=_pose([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]),
        o_t_ee_d=_pose([4.0, 5.0, 6.0], [0.0, 0.0, 1.0, 0.0]),
        f_t_ee=_pose([0.0, 0.0, 0.174], [0.0, 0.0, 0.0, 1.0]),
        k_f_ext_hat_k=SimpleNamespace(
            wrench=SimpleNamespace(force=zero_vector, torque=zero_vector)
        ),
        o_f_ext_hat_k=SimpleNamespace(
            wrench=SimpleNamespace(force=base_force, torque=base_torque)
        ),
    )


def _gripper_state():
    return SimpleNamespace(name=[schema.ROBOTIQ_JOINT_NAME], position=[0.0])


def test_default_gripper_action_topic_uses_teleop_arbiter() -> None:
    assert schema.OUTPUT_ROOT == schema.REPO_ROOT / "datasets" / "lerobot_dataset_v2"
    assert schema.TOPIC_ARM_COMMAND == "/left/teleop/joint_states"
    assert schema.DEFAULT_GRIPPER_COMMAND_TOPIC == "/left/teleop/gripper_action_width_percent"


def test_rot6d_column_order_and_round_trip() -> None:
    for quaternion in Rotation.random(20, random_state=7).as_quat():
        matrix = schema.quaternion_to_matrix(quaternion)
        rot6d = schema.matrix_to_rot6d(matrix)
        np.testing.assert_allclose(rot6d, matrix[:, :2].T.reshape(-1), atol=1e-8)
        np.testing.assert_allclose(schema.rot6d_to_matrix(rot6d), matrix, atol=1e-7)


@pytest.mark.parametrize(
    "value,match",
    [
        (np.zeros(6), "first column"),
        (np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0]), "collinear"),
    ],
)
def test_rot6d_rejects_degenerate_inputs(value: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        schema.rot6d_to_matrix(value)


def test_state_uses_measured_pose_and_action_uses_desired_pose() -> None:
    message = _robot_state()
    state = schema.state_vector(message, _gripper_state(), np.zeros(6))
    action = schema.action_vector(message, SimpleNamespace(data=0.25))

    assert schema.STATE_FIELDS[:10] == [
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
    ]
    np.testing.assert_array_equal(state[:3], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(action[:3], [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(state[3:9], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(action[3:9], [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
    assert state.shape == (50,)
    assert action.shape == (10,)
    assert action[9] == pytest.approx(0.25)
    np.testing.assert_array_equal(state[10:16], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    np.testing.assert_allclose(state[22:29], np.arange(1, 8) / 10)
    np.testing.assert_allclose(state[29:36], np.arange(1, 8))
    np.testing.assert_allclose(state[36:43], np.arange(1, 8) / 2)
    np.testing.assert_allclose(state[43:50], np.arange(1, 8) / 100)


def test_command_joint_positions_accepts_controller_namespace_and_reorders() -> None:
    message = SimpleNamespace(
        name=["fr3_joint3", "fr3_joint1", "fr3_joint2"],
        position=[3.0, 1.0, 2.0],
    )
    result = schema.commanded_joint_positions(
        message,
        ("left_fr3_joint1", "left_fr3_joint2", "left_fr3_joint3"),
    )
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_tcp_pose_from_flange_applies_configured_tool_transform() -> None:
    message = _robot_state()
    position, rotation = schema.tcp_pose_from_flange(
        np.array([1.0, 2.0, 3.0]),
        Rotation.from_euler("z", 90, degrees=True).as_matrix(),
        message,
    )
    np.testing.assert_allclose(position, [1.0, 2.0, 3.174], atol=1e-8)
    np.testing.assert_allclose(rotation, Rotation.from_euler("z", 90, degrees=True).as_matrix())


def test_episode_validation_rejects_constant_tcp_action() -> None:
    states = np.zeros((4, 50), dtype=np.float32)
    states[:, 3] = 1.0
    states[:, 7] = 1.0
    actions = np.zeros((4, 10), dtype=np.float32)
    actions[:, 3] = 1.0
    actions[:, 7] = 1.0
    with pytest.raises(ValueError, match="commanded TCP action is constant"):
        schema.validate_episode_features(states, actions)


def test_episode_validation_rejects_stationary_target_jitter() -> None:
    states = np.zeros((4, 50), dtype=np.float32)
    actions = np.zeros((4, 10), dtype=np.float32)
    states[:, 3] = actions[:, 3] = 1.0
    states[:, 7] = actions[:, 7] = 1.0
    actions[:, 0] = np.linspace(0.0, 0.001, 4)
    with pytest.raises(ValueError, match="commanded TCP action is constant"):
        schema.validate_episode_features(states, actions)


def test_episode_validation_reports_all_feature_ranges() -> None:
    states = np.zeros((4, 50), dtype=np.float32)
    states[:, 3] = 1.0
    states[:, 7] = 1.0
    actions = np.zeros((4, 10), dtype=np.float32)
    actions[:, 0] = np.linspace(0.4, 0.5, 4)
    actions[:, 3] = 1.0
    actions[:, 7] = 1.0
    report = schema.validate_episode_features(states, actions)
    assert np.asarray(report["state_range"]).shape == (50,)
    assert np.asarray(report["action_range"]).shape == (10,)
    assert report["translation_span_m"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("command", "expected"),
    [(0.0, 0.0), (0.4, 0.4), (1.0, 1.0)],
)
def test_gripper_action_uses_final_teleop_width(command: float, expected: float) -> None:
    assert schema.gripper_action(SimpleNamespace(data=command)) == pytest.approx(expected)


@pytest.mark.parametrize("command", [-0.1, 1.1, float("nan")])
def test_gripper_action_rejects_invalid_command(command: float) -> None:
    with pytest.raises(ValueError):
        schema.gripper_action(SimpleNamespace(data=command))


def _image_stats(count: int) -> dict[str, np.ndarray]:
    stats = {
        key: np.ones((3, 1, 1), dtype=np.float32)
        for key in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
    }
    stats["count"] = np.array([count], dtype=np.int64)
    return stats


def _save_synthetic_episode(writer: V30DatasetWriter, offset: float = 0.0, length: int = 4) -> None:
    episode_index = writer.next_episode_index()
    for key in writer.camera_keys:
        path = writer.video_path(episode_index, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic video placeholder")
    states = np.arange(length * 50, dtype=np.float32).reshape(length, 50) + offset
    actions = np.arange(length * 10, dtype=np.float32).reshape(length, 10) + offset
    writer.save_episode(
        states,
        actions,
        {key: _image_stats(length) for key in writer.camera_keys},
    )


def test_v30_writer_resume_discard_and_metadata(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    cameras = ("observation.images.base", "observation.images.left_wrist")
    writer = V30DatasetWriter(root, 30, "test task", cameras, 16, 16)
    _save_synthetic_episode(writer)
    _save_synthetic_episode(writer, offset=100.0)

    resumed = V30DatasetWriter(root, 30, "test task", cameras, 16, 16)
    assert resumed.episode_count == 2
    assert resumed.frame_count == 8
    assert resumed.next_episode_index() == 2
    assert resumed.discard_latest_episode() == 1
    assert resumed.episode_count == 1
    assert resumed.frame_count == 4
    assert not resumed.data_path(1).exists()
    assert not resumed.video_path(1, cameras[0]).exists()

    info = json.loads((root / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["features"]["observation.state"]["shape"] == [50]
    assert info["features"]["action"]["shape"] == [10]
    episodes = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    assert episodes["episode_index"].tolist() == [0]
    assert episodes["data/file_index"].tolist() == [0]
    assert "stats/observation.state/q01" in episodes
    assert "videos/observation.images.base/file_index" in episodes


@pytest.mark.parametrize("field", ["effort", "velocity", "external_effort"])
@pytest.mark.parametrize("bad", [[], [0.] * 6, [float("nan")] * 7])
def test_full_state_rejects_missing_or_invalid_joint_dynamics(field, bad):
    message = _robot_state()
    target = message.tau_ext_hat_filtered if field == "external_effort" else message.measured_joint_state
    setattr(target, "effort" if field == "external_effort" else field, bad)
    with pytest.raises(ValueError, match="finite 7-vector"):
        schema.state_vector(message, _gripper_state(), np.zeros(6))


@pytest.mark.parametrize("width", [29, 31])
def test_full_writer_rejects_legacy_dataset_without_touching_files(tmp_path, width):
    from vla.datacollection.franka_robotiq_single_left.schema_31d import STATE_31D_FIELDS
    fields = schema.LEGACY_STATE_FIELDS if width == 29 else STATE_31D_FIELDS
    V30DatasetWriter(tmp_path, 30, "test", (), 16, 16, state_fields=fields)
    pending = tmp_path / "unfinished.tmp"
    pending.write_text("keep")
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError, match="incompatible shape"):
        V30DatasetWriter(tmp_path, 30, "test", (), 16, 16)
    assert before == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
