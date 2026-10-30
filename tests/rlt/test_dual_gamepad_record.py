from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vla.datacollection.franka_robotiq_dual.gamepad_record import (
    DEFAULT_CONFIG, main, recording_contract, validate_dataset_root, validate_dual_config,
)
from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
from vla.datacollection.franka_robotiq_rollout.gamepad import gamepad_is_neutral, decode_gamepad_values
from vla.datacollection.franka_robotiq_single_left import schema
from vla.cli.build_transition_cache_v2 import load_episode_frame_annotations, _chunk_annotation
from vla.cli.train_franka_pi05_sft import record_pi05_loss_metrics


def test_dual_loss_groups_include_both_arms():
    tracker = SimpleNamespace(metrics={})
    names = [f"{arm}_{name}" for arm in ("left", "right") for name in schema.ACTION_FIELDS]
    record_pi05_loss_metrics(tracker, {"loss": 2., "loss_per_dim": [1.] * 10 + [3.] * 10}, names)
    assert tracker.metrics["flow_loss/group/gripper"].avg == 2.
    assert tracker.metrics["flow_loss/group/left/position"].avg == 1.
    assert tracker.metrics["flow_loss/group/right/rotation_6d"].avg == 3.


def test_dual_shared_ac_optimizes_20d_chunks_without_loading_vla(tmp_path, monkeypatch):
    import torch
    from vla.adapters.lerobot.policies.configuration_rlt_token import RLTokenPolicyConfig
    from vla.adapters.lerobot.policies.configuration_rlt_ac import ChunkACPolicyConfig
    from vla.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
    camera_keys = ["base", "left_wrist", "right_wrist"]
    token = RLTokenPolicyConfig(action_dim=20, proprio_dim=20, camera_keys=camera_keys,
                               rl_token_dim=32, device="cpu")
    token.validate_features()
    token.save_pretrained(tmp_path / "token")
    monkeypatch.setattr(ChunkACPolicy, "_load_rl_token_policy", lambda self: pytest.fail("offline training loaded VLA"))
    policy = ChunkACPolicy(ChunkACPolicyConfig(
        rl_token_pretrained_path=str(tmp_path / "token"), rl_token_dim=32,
        camera_keys=camera_keys, action_dim=20, proprio_dim=20, chunk_length=20,
        actor_hidden_dim=32, critic_hidden_dim=32, device="cpu"))
    batch = {
        "state_vec": torch.randn(2, 52), "next_state_vec": torch.randn(2, 52),
        "exec_chunk": torch.randn(2, 20, 20), "ref_chunk": torch.randn(2, 20, 20),
        "next_ref_chunk": torch.randn(2, 20, 20), "reward_seq": torch.zeros(2, 20),
        "done": torch.tensor([0., 1.]), "actual_steps": torch.tensor([20, 20]),
    }
    optimizer = torch.optim.Adam(policy.get_optim_params(), lr=1e-3)
    before = next(policy.critic.parameters()).detach().clone()
    loss, _ = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, next(policy.critic.parameters()))
    assert policy._rl_token_policy is None


def test_dual_check_only_never_creates_dataset(monkeypatch, tmp_path):
    root = tmp_path / "untouched"
    monkeypatch.setattr(sys, "argv", ["record", "--check-only", "--dataset-root", str(root)])
    main()
    assert not root.exists()
    assert not list(tmp_path.iterdir())
    config = load_rollout_config(DEFAULT_CONFIG)
    validate_dual_config(config)
    with pytest.raises(ValueError, match="distinct"):
        validate_dual_config(replace(config, cameras=tuple(replace(c, serial="same") for c in config.cameras)))
    with pytest.raises(ValueError, match="ordered"):
        validate_dual_config(replace(config, arms=tuple(reversed(config.arms))))
    with pytest.raises(ValueError, match="distinct Joy"):
        validate_dual_config(replace(config, arms=(config.arms[0], replace(config.arms[1], gamepad=config.arms[0].gamepad))))


def test_explicit_resume_and_recording_contract(tmp_path):
    with pytest.raises(ValueError, match="existing dataset"):
        validate_dataset_root(tmp_path, resume=True)
    (tmp_path / "data.txt").write_text("existing")
    with pytest.raises(ValueError, match="not empty"):
        validate_dataset_root(tmp_path, resume=False)
    config = load_rollout_config(DEFAULT_CONFIG)
    assert recording_contract(config) == recording_contract(replace(config, dataset_root=tmp_path))
    assert recording_contract(config) != recording_contract(replace(config, gamepad_xy_speed=.01))


def test_two_gamepad_mapping_and_neutral_gate():
    config = load_rollout_config(DEFAULT_CONFIG)
    axes, buttons = [0., 0., 1., 0., 0., 1.], [0] * 11
    for arm in config.arms:
        assert gamepad_is_neutral(axes, buttons, arm.gamepad, config)
        pressed = axes.copy()
        pressed[2] = pressed[5] = -1.
        assert not gamepad_is_neutral(pressed, buttons, arm.gamepad, config)
        with pytest.raises(ValueError, match="invalid gamepad"):
            gamepad_is_neutral([float("nan")] * 6, buttons, arm.gamepad, config)
    left_axes = axes.copy()
    left_axes[1] = 1.
    left = decode_gamepad_values(left_axes, buttons, config.arms[0].gamepad, config)
    right = decode_gamepad_values(axes, buttons, config.arms[1].gamepad, config)
    assert left[0][0] > 0
    np.testing.assert_array_equal(right[0], [0., 0., 0.])


@pytest.fixture
def ros_module(monkeypatch):
    """Run real collector state transitions with ROS transport/kinematics faked."""
    class Node:
        def __init__(self, *_):
            pass

        def create_timer(self, *_):
            pass

    symbols = {
        "PyKDL": {}, "rclpy": {}, "rclpy.node": {"Node": Node},
        "rclpy.action": {"ActionClient": object},
        "rclpy.executors": {"MultiThreadedExecutor": object},
        "rclpy.parameter": {"parameter_value_to_python": lambda x: x},
        "rclpy.qos": {k: object for k in ("HistoryPolicy", "QoSProfile", "ReliabilityPolicy")},
        "control_msgs.action": {"GripperCommand": object},
        "franka_msgs.msg": {"FrankaRobotState": object},
        "rcl_interfaces.srv": {"GetParameters": object},
        "sensor_msgs.msg": {"JointState": object, "Joy": object},
        "std_msgs.msg": {"Float32": object, "Float64MultiArray": object},
        "vla.datacollection.franka_robotiq_single_left.auto_record": {
            k: object for k in ("AsyncVideoWriter", "KeyboardEvents", "RealSenseCamera", "prepare_image")},
        "vla.datacollection.start_teleop.franka_kinematics": {
            "build_kinematic_model": object, "joint7_step": object},
    }
    for name, attrs in symbols.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    name = "vla.datacollection.franka_robotiq_rollout._test_ros_collector"
    path = DEFAULT_CONFIG.parents[1] / "collect_ros.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def collector(ros_module, monkeypatch, tmp_path):
    module = ros_module
    cls = module.RolloutCollector
    for name in ("_setup_ros_io", "_start_cameras", "_stop_grippers_at_measured_position"):
        monkeypatch.setattr(cls, name, lambda self: None)
    monkeypatch.setattr(cls, "_load_kinematic_model", lambda self, arm: None)
    monkeypatch.setattr(cls, "_preflight_problems", lambda self: [])
    monkeypatch.setattr(cls, "_publish_arm_command", lambda *args, **kwargs: None)
    monkeypatch.setattr(cls, "_restore_policy_gripper_speed", lambda self: pytest.fail("demo used policy gripper"))
    monkeypatch.setattr(module, "AsyncPolicyClient", lambda *args: pytest.fail("demo opened policy socket"))
    monkeypatch.setattr(module, "CartesianServo", lambda *args: SimpleNamespace(
        reset_tracking=lambda: None, model=SimpleNamespace(joint_names=[f"j{i}" for i in range(7)]),
        forward_tcp=lambda *args: (np.array([.4, 0., .3]), np.eye(3))))

    class Video:
        def __init__(self, path, **kwargs):
            self.frames = []
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            self.image_stats = SimpleNamespace(as_dict=lambda: {
                **{key: np.zeros((3, 1, 1)) for key in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")},
                "count": np.array([len(self.frames)])})

        def write(self, image):
            self.frames.append(image)

        def close(self):
            pass

        def abort(self):
            pass

    monkeypatch.setattr(module, "AsyncVideoWriter", Video)
    config = replace(load_rollout_config(DEFAULT_CONFIG), dataset_root=tmp_path / "demo", subtasks=("pick", "insert"))
    c = cls(config, demonstration=True)
    pose = SimpleNamespace(pose=SimpleNamespace(
        position=SimpleNamespace(x=0., y=0., z=0.),
        orientation=SimpleNamespace(x=0., y=0., z=0., w=1.)))
    now = time.monotonic_ns()
    for arm in config.arms:
        for key, value in {
            "measured_joint": SimpleNamespace(name=[f"j{i}" for i in range(7)], position=[0.] * 7),
            "robot_state": SimpleNamespace(f_t_ee=pose),
            "gripper_state": SimpleNamespace(name=[arm.gripper_joint_name], position=[0.]),
            "gamepad": SimpleNamespace(axes=[0., 0., 1., 0., 0., 1.], buttons=[0] * 11),
        }.items():
            c._latest[f"{arm.name}.{key}"] = (now, value)
    images = {camera.feature_key: np.zeros((224, 224, 3), dtype=np.uint8) for camera in config.cameras}
    monkeypatch.setattr(c, "_snapshot", lambda **kwargs: ({key: item[1] for key, item in c._latest.items()}, images))
    yield c
    c.close()


def test_demo_enable_record_pause_and_fault(collector):
    c = collector
    c.start_episode()
    assert not c._recording
    c._latest["right.gamepad"][1].axes[1] = 1.
    c.enable_teleop()
    assert not c._teleop_enabled
    c._latest["right.gamepad"][1].axes[1] = 0.
    c.enable_teleop()
    assert c._teleop_enabled and c._intervention
    c.start_episode()
    assert c._recording and c._last_command_snapshot.source == 0
    c.toggle_phase()
    c.next_subtask()
    assert c._phase == "rl" and c._subtask_index == 1 and c._intervention
    c.toggle_intervention()
    assert c._intervention  # no policy to hand back to
    c.pause_teleop()
    assert not c._recording and not c._teleop_enabled and c.writer.episode_count == 0
    c.enable_teleop()
    c._latest.pop("right.gamepad")
    c._gamepad_tick()
    assert c._faulted and not c._teleop_enabled and not c._intervention
    c.enable_teleop()
    assert not c._teleop_enabled


def test_demo_save_preserves_stage_and_is_cache_demonstration(collector, monkeypatch):
    from datasets import Dataset
    c = collector
    c.enable_teleop()
    c.start_episode()
    action = np.tile([.4, 0., .3, 1., 0., 0., 0., 1., 0., 1.], 2).astype(np.float32)
    state = np.zeros(58, dtype=np.float32)
    for offset in (0, 29):
        state[offset:offset+9] = action[:9]
        state[offset+9] = .085
    monkeypatch.setattr(c, "_actual_action", lambda *args: action.copy())
    values, images = c._snapshot()
    for frame in range(12):
        action[10] += .001  # Left holds; only the right arm moves.
        if frame == 6:
            c.next_subtask()
            c.toggle_phase()
            c._last_command_snapshot = replace(c._last_command_snapshot, phase=1.)
        c._record_frame(state.copy(), values, images, time.monotonic_ns())
    c.finish_episode("success")
    assert c.writer.episode_count == 1 and not c._teleop_enabled
    info = json.loads((c.config.dataset_root / "meta/info.json").read_text())
    frame = pd.read_parquet(c.writer.data_path(0))
    assert info["features"]["action"]["shape"] == [20]
    assert info["features"]["observation.state"]["shape"] == [58]
    assert frame["complementary_info.subtask_index"].tolist() == [0] * 6 + [1] * 6
    assert "complementary_info.collector_policy_id" not in frame
    data = SimpleNamespace(hf_dataset=Dataset.from_pandas(frame), meta=SimpleNamespace(info=info))
    annotations = load_episode_frame_annotations(data, 0, 12)
    from vla.core.interfaces import TRANSITION_SOURCE_DEMO
    assert _chunk_annotation(annotations, start_frame=6, chunk_length=2) == {
        "intervention": 0., "source": TRANSITION_SOURCE_DEMO, "is_critical": 1.,
    }
    from vla.datacollection.franka_robotiq_dual.report import summarize
    report = summarize(c.config.dataset_root)
    assert report["successes"] == 1
    assert [stage["completion_rate"] for stage in report["stages"]] == [1., 1.]
    assert report["stages"][0]["intervention_frame_rate"] is None
    # Reopen with unchanged settings and string-keyed stage codebook.
    from vla.datacollection.franka_robotiq_single_left.v30_writer import V30DatasetWriter
    writer = V30DatasetWriter(
        c.config.dataset_root, c.config.fps, c.config.task,
        tuple(camera.feature_key for camera in c.config.cameras), *c.config.image_size,
        state_fields=c.writer.state_fields, action_fields=c.writer.action_fields,
        extra_features=c._extra_features, robot_type="franka_robotiq_dual",
        info_metadata=c.writer.info_metadata, strict_metadata=True,
    )
    assert writer.episode_count == 1
    from vla.cli.extract_dataset_features import extract_dataset_features, FEATURE_CONFIG_DIR
    before = (c.config.dataset_root / "meta/info.json").read_bytes()
    output = extract_dataset_features(
        c.config.dataset_root, FEATURE_CONFIG_DIR / "franka_dual_tcp_rot6d_20d.yaml",
        c.config.dataset_root.parent / "20d", validate=False,
    )
    extracted = pd.read_parquet(output / "data/chunk-000/file-000.parquet")
    assert np.stack(extracted["observation.state"]).shape == (12, 20)
    np.testing.assert_array_equal(np.stack(extracted["action"]), np.stack(frame["action"]))
    assert extracted["complementary_info.subtask_index"].tolist() == [0] * 6 + [1] * 6
    assert summarize(output) == report
    assert (c.config.dataset_root / "meta/info.json").read_bytes() == before


def test_demo_resume_rejects_changed_controls(collector, ros_module):
    c = collector
    # Contract comparison includes speed, prompt, stages, topics and camera order.
    with pytest.raises(RuntimeError, match="recording_contract"):
        ros_module.RolloutCollector(replace(c.config, gamepad_xy_speed=.01), demonstration=True)


def test_demo_success_requires_last_subtask(collector):
    c = collector
    c.enable_teleop()
    c.start_episode()
    c.finish_episode("success")
    assert c._recording
    c.finish_episode("failure")
    assert not c._recording and c.writer.episode_count == 0


def test_demo_camera_poll_does_not_duplicate_images_and_faults_on_gap(collector, ros_module, monkeypatch):
    c = collector
    c.enable_teleop()
    c.start_episode()
    clock = [time.monotonic_ns()]
    monkeypatch.setattr(ros_module.time, "monotonic_ns", lambda: clock[0])
    monkeypatch.setattr(c, "_validate_robot_health", lambda values: None)
    monkeypatch.setattr(c, "_raw_state", lambda *args: np.zeros(58))
    recorded = []
    monkeypatch.setattr(c, "_record_frame", lambda *args: recorded.append(args[-1]))
    for camera in c.config.cameras:
        c._latest[f"camera.{camera.name}"] = (clock[0], None)
    c._tick()
    c._tick()
    assert recorded == [clock[0]]
    clock[0] += 33_333_333
    for camera in c.config.cameras:
        c._latest[f"camera.{camera.name}"] = (clock[0], None)
    c._tick()
    assert len(recorded) == 2
    clock[0] += 100_000_000
    c._tick()
    assert c._faulted and not c._recording and not c._teleop_enabled
