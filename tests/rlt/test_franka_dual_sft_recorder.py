from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vla.datacollection.franka_robotiq_dual import schema
from vla.datacollection.franka_robotiq_single_left.v30_writer import V30DatasetWriter


def _pose(position, quaternion=(0.0, 0.0, 0.0, 1.0)):
    return SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=position[0], y=position[1], z=position[2]),
            orientation=SimpleNamespace(
                x=quaternion[0],
                y=quaternion[1],
                z=quaternion[2],
                w=quaternion[3],
            ),
        )
    )


def _robot_state(measured, desired):
    zero = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    return SimpleNamespace(
        measured_joint_state=SimpleNamespace(
            name=[f"fr3_joint{index}" for index in range(1, 8)],
            position=[index / 10 for index in range(1, 8)],
            effort=list(range(1, 8)), velocity=[index / 100 for index in range(1, 8)],
        ),
        tau_ext_hat_filtered=SimpleNamespace(name=[], effort=[index / 2 for index in range(1, 8)]),
        o_t_ee=_pose(measured),
        o_t_ee_d=_pose(desired),
        f_t_ee=_pose([0.0, 0.0, 0.0]),
        k_f_ext_hat_k=SimpleNamespace(wrench=SimpleNamespace(force=zero, torque=zero)),
        o_f_ext_hat_k=SimpleNamespace(wrench=SimpleNamespace(force=zero, torque=zero)),
    )


def _gripper_state(joint_name: str, position: float):
    return SimpleNamespace(name=[joint_name], position=[position])


def _image_stats(count: int) -> dict[str, np.ndarray]:
    values = {
        key: np.ones((3, 1, 1), dtype=np.float32)
        for key in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
    }
    values["count"] = np.array([count], dtype=np.int64)
    return values


def test_dual_schema_is_left_then_right_and_uses_absolute_poses() -> None:
    left, right = schema.ARMS
    assert schema.OUTPUT_ROOT == schema.REPO_ROOT / "datasets" / "dual_lerobot_dataset"
    assert right.gripper_joint_name == "robotiq_85_left_knuckle_joint"
    left_message = _robot_state([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    right_message = _robot_state([7.0, 8.0, 9.0], [10.0, 11.0, 12.0])
    zero_velocity = np.zeros(6, dtype=np.float32)

    states = {
        "left": schema.arm_state_vector(
            left,
            left_message,
            _gripper_state(left.gripper_joint_name, 0.0),
            zero_velocity,
        ),
        "right": schema.arm_state_vector(
            right,
            right_message,
            _gripper_state(right.gripper_joint_name, 0.7929),
            zero_velocity,
        ),
    }
    actions = {
        "left": schema.arm_action_vector(
            np.array([4.0, 5.0, 6.0]), np.eye(3), SimpleNamespace(data=0.2)
        ),
        "right": schema.arm_action_vector(
            np.array([10.0, 11.0, 12.0]), np.eye(3), SimpleNamespace(data=0.8)
        ),
    }
    state = schema.dual_state_vector(states)
    action = schema.dual_action_vector(actions)

    assert state.shape == (100,)
    assert action.shape == (20,)
    np.testing.assert_array_equal(state[:3], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(state[50:53], [7.0, 8.0, 9.0])
    np.testing.assert_allclose(state[22:29], np.arange(1, 8) / 10)
    np.testing.assert_allclose(state[72:79], np.arange(1, 8) / 10)
    np.testing.assert_array_equal(action[:3], [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(action[10:13], [10.0, 11.0, 12.0])
    assert state[9] == pytest.approx(0.085)
    assert state[59] == pytest.approx(0.0)
    assert action[9] == pytest.approx(0.2)
    assert action[19] == pytest.approx(0.8)
    assert schema.STATE_FIELDS[0] == "left_tcp_position_x"
    assert schema.STATE_FIELDS[50] == "right_tcp_position_x"
    assert schema.ACTION_FIELDS[0] == "left_desired_tcp_position_x"
    assert schema.ACTION_FIELDS[10] == "right_desired_tcp_position_x"
    assert [arm.command_joint_topic for arm in schema.ARMS] == [
        "/left/teleop/joint_states",
        "/right/teleop/joint_states",
    ]


def test_dual_episode_validation_rejects_only_fully_stationary_tcp_actions() -> None:
    frames = 12
    states = np.zeros((frames, len(schema.STATE_FIELDS)), dtype=np.float32)
    actions = np.zeros((frames, len(schema.ACTION_FIELDS)), dtype=np.float32)
    for offset in (0, 10):
        actions[:, offset : offset + 3] = [0.4, 0.0, 0.3]
        actions[:, offset + 3] = 1.0
        actions[:, offset + 7] = 1.0

    with pytest.raises(ValueError, match="both commanded TCP actions are constant"):
        schema.validate_episode_features(states, actions)

    actions[:, 0] += np.linspace(0.0, 0.02, frames)
    reports = schema.validate_episode_features(states, actions)
    assert reports["left"]["translation_span_m"] == pytest.approx(0.02)
    assert reports["right"]["translation_span_m"] == pytest.approx(0.0)


def test_dual_writer_emits_100d_20d_schema_and_three_video_indices(tmp_path: Path) -> None:
    writer = V30DatasetWriter(
        tmp_path,
        schema.FPS,
        "dual test task",
        tuple(schema.CAMERA_FEATURES),
        16,
        16,
        state_fields=schema.STATE_FIELDS,
        action_fields=schema.ACTION_FIELDS,
        robot_type="franka_robotiq_dual",
        info_metadata={"arm_order": ["left", "right"]},
    )
    for key in writer.camera_keys:
        path = writer.video_path(0, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic video placeholder")
    length = 4
    writer.save_episode(
        np.zeros((length, 100), dtype=np.float32),
        np.zeros((length, 20), dtype=np.float32),
        {key: _image_stats(length) for key in writer.camera_keys},
    )

    info = json.loads((tmp_path / "meta/info.json").read_text())
    assert info["robot_type"] == "franka_robotiq_dual"
    assert info["arm_order"] == ["left", "right"]
    assert info["features"]["observation.state"]["shape"] == [100]
    assert info["features"]["action"]["shape"] == [20]
    assert list(info["features"])[-3:] == list(schema.CAMERA_FEATURES)
    episodes = pd.read_parquet(tmp_path / "meta/episodes/chunk-000/file-000.parquet")
    assert all(f"videos/{key}/file_index" in episodes for key in schema.CAMERA_FEATURES)


def test_dual_writer_keeps_one_prompt_per_episode_and_validates_task_indices(tmp_path: Path) -> None:
    writer = V30DatasetWriter(
        tmp_path, schema.FPS, "pick", tuple(schema.CAMERA_FEATURES), 16, 16,
        state_fields=schema.STATE_FIELDS, action_fields=schema.ACTION_FIELDS,
        robot_type="franka_robotiq_dual", allow_multiple_tasks=True,
    )
    stats = {key: _image_stats(4) for key in writer.camera_keys}
    for episode, task in enumerate(("pick", "insert")):
        for key in writer.camera_keys:
            path = writer.video_path(episode, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic video placeholder")
        writer.save_episode(np.zeros((4, 100), dtype=np.float32),
                            np.zeros((4, 20), dtype=np.float32), stats, task=task)
    tasks = pd.read_parquet(tmp_path / "meta/tasks.parquet")
    episodes = pd.read_parquet(tmp_path / "meta/episodes/chunk-000/file-000.parquet")
    data = pd.concat(
        pd.read_parquet(path)
        for path in sorted((tmp_path / "data/chunk-000").glob("file-*.parquet"))
    )
    assert list(tasks.index) == ["pick", "insert"]
    assert episodes["tasks"].tolist() == [["pick"], ["insert"]]
    task_indices = {
        int(index): values.tolist()
        for index, values in data.groupby("episode_index")["task_index"].unique().items()
    }
    assert task_indices == {0: [0], 1: [1]}
    data.loc[data.episode_index == 1, "task_index"] = 0
    first = tmp_path / "data/chunk-000/file-000.parquet"
    first_data = pd.read_parquet(first)
    first_data.loc[first_data.episode_index == 0, "task_index"] = 1
    first_data.to_parquet(first, index=False)
    with pytest.raises(RuntimeError, match="task_index"):
        V30DatasetWriter(
            tmp_path, schema.FPS, "pick", tuple(schema.CAMERA_FEATURES), 16, 16,
            state_fields=schema.STATE_FIELDS, action_fields=schema.ACTION_FIELDS,
            robot_type="franka_robotiq_dual", allow_multiple_tasks=True,
        )
