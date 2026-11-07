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
    from vla.adapters.lerobot.policies.configuration_rlt_token import RLTokenConfig
    from vla.adapters.lerobot.policies.configuration_rlt_ac import ChunkACConfig
    from vla.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
    camera_keys = ["base", "left_wrist", "right_wrist"]
    token = RLTokenConfig(action_dim=20, proprio_dim=20, camera_keys=camera_keys,
                               rl_token_dim=32, device="cpu")
    token.validate_features()
    token.save_pretrained(tmp_path / "token")
    monkeypatch.setattr(ChunkACPolicy, "_load_rl_token_policy", lambda self: pytest.fail("offline training loaded VLA"))
    policy = ChunkACPolicy(ChunkACConfig(
        rl_token_pretrained_path=str(tmp_path / "token"), rl_token_dim=32,
        camera_keys=camera_keys, action_dim=20, proprio_dim=20, chunk_length=20,
        actor_hidden_dim=32, critic_hidden_dim=32, device="cpu"))
    batch = {
        "state_vec": torch.randn(2, 52), "next_state_vec": torch.randn(2, 52),
        "exec_chunk": torch.randn(2, 20, 20), "ref_chunk": torch.randn(2, 20, 20),
        "next_ref_chunk": torch.randn(2, 20, 20), "reward_seq": torch.zeros(2, 20),
        "done": torch.tensor([0., 1.]), "actual_steps": torch.tensor([20, 20]),
    }
    from tests.rlt.execution_fixtures import execution_context
    execution_context(batch)
    monkeypatch.setattr(policy, "execution_projection", lambda mu, anchor: mu.clamp(-1,1))
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

        def create_subscription(self, *_):
            pass

        def destroy_node(self):
            pass

    symbols = {
        "PyKDL": {}, "rclpy": {}, "rclpy.node": {"Node": Node},
        "rclpy.action": {"ActionClient": object},
        "rclpy.executors": {"MultiThreadedExecutor": object},
        "rclpy.parameter": {"parameter_value_to_python": lambda x: x},
        "rclpy.qos": {k: object for k in ("HistoryPolicy", "QoSProfile", "ReliabilityPolicy")},
        "control_msgs.action": {"GripperCommand": object},
        "franka_msgs.msg": {"FrankaRobotState": object},
        "rcl_interfaces.msg": {"ParameterEvent": object},
        "rcl_interfaces.srv": {"GetParameters": object},
        "sensor_msgs.msg": {"JointState": object, "Joy": object},
        "std_msgs.msg": {"Float32": object, "Float64MultiArray": object, "UInt64": object},
        "vla.datacollection.franka_robotiq_single_left.auto_record": {
            k: object for k in ("KeyboardEvents", "RealSenseCamera", "prepare_image")},
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


@pytest.fixture(params=[False, True], ids=["standard", "side"])
def collector(ros_module, monkeypatch, tmp_path, request):
    module = ros_module
    cls = module.RolloutCollector
    for name in ("_setup_ros_io", "_start_cameras", "_stop_grippers_at_measured_position"):
        monkeypatch.setattr(cls, name, lambda self: None)
    monkeypatch.setattr(cls, "_controller_parameters", lambda self: {})
    monkeypatch.setattr(cls, "_load_kinematic_model", lambda self, arm: None)
    monkeypatch.setattr(cls, "_preflight_problems", lambda self: [])
    monkeypatch.setattr(cls, "_coordinated_publish", lambda *args, **kwargs: None)
    monkeypatch.setattr(cls, "_tool_definition", lambda self: {a.name: {'position': [0,0,0], 'rotation': np.eye(3).tolist()} for a in self.config.arms})
    def reset(self, generation):
        self._observe_control()
        for arm in self.config.arms:
            self.coordinator.acknowledge(arm.name, generation, 0, np.zeros(7), time.monotonic_ns())
    monkeypatch.setattr(cls, '_request_controller_reset', reset)
    monkeypatch.setattr(cls, "_send_gripper_goal", lambda *args: None)
    monkeypatch.setattr(cls, "_publish_gripper_speed", lambda *args: None)
    monkeypatch.setattr(cls, "_restore_policy_gripper_speed", lambda self: pytest.fail("demo used policy gripper"))
    monkeypatch.setattr(module, "AsyncPolicyClient", lambda *args: pytest.fail("demo opened policy socket"))
    monkeypatch.setattr(module, "CartesianServo", lambda *args: SimpleNamespace(
        reset_tracking=lambda: None, model=SimpleNamespace(joint_names=[f"j{i}" for i in range(7)], lower_limits=np.full(7,-3.), upper_limits=np.full(7,3.)),
        step=lambda *args: np.zeros(7),
        forward_tcp=lambda *args: (np.array([.4, 0., .3]), np.eye(3))))

    config = replace(load_rollout_config(DEFAULT_CONFIG), dataset_root=tmp_path / "demo", subtasks=("pick", "insert"))
    if request.param:
        from vla.datacollection.franka_robotiq_rollout.config import CameraConfig
        config = replace(config, cameras=(*config.cameras, CameraConfig("side", "TEST_SIDE")))
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
    for camera in config.cameras:
        c._latest[f"camera.{camera.name}"] = (now, np.zeros((224,224,3), dtype=np.uint8))
    images = {camera.feature_key: np.zeros((224, 224, 3), dtype=np.uint8) for camera in config.cameras}
    monkeypatch.setattr(c, "_snapshot", lambda **kwargs: ({key: item[1] for key, item in c._latest.items()}, images))
    yield c
    c.close()


def record_command(c, action, state, values, images):
    stamp = getattr(c, '_test_stamp', time.monotonic_ns()) + 33_333_334
    c._test_stamp = stamp
    c.coordinator.clock = lambda: stamp
    c._slot_id += 1
    c._complete_interval(stamp)
    for key, (_, value) in list(c._latest.items()):
        c._latest[key] = (stamp, value)
    c._observe_control()
    command = c.coordinator.accept(action, source='gamepad', generation=c.coordinator.generation, slot_id=c._slot_id)
    c._track_control(.01)
    c._last_command_snapshot = replace(c._last_command_snapshot, published_at_ns=stamp)
    c._record_frame(state, values, images, stamp)
    return command


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
    assert c._recording and c.coordinator.owner == "gamepad"
    assert c._last_command_snapshot is None  # reset never fabricates a command label
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
    state = np.zeros(100, dtype=np.float32)
    for offset in (0, 50):
        state[offset:offset+9] = action[:9]
        state[offset+9] = .085

    values, images = c._snapshot()
    for frame in range(12):
        action[10] += .001  # Left holds; only the right arm moves.
        if frame == 6:
            c.next_subtask()
            c.toggle_phase()

        record_command(c, action.copy(), state.copy(), values, images)
    c._at_boundary = True
    c.finish_episode("success")
    c._save_worker.join(30)
    assert c.writer.episode_count == 1 and not c._teleop_enabled
    info = json.loads((c.config.dataset_root / "meta/info.json").read_text())
    frame = pd.read_parquet(c.writer.data_path(0))
    assert info["features"]["action"]["shape"] == [20]
    assert info["features"]["observation.state"]["shape"] == [100]
    assert frame["complementary_info.subtask_index"].tolist() == [0] * 6 + [1] * 6
    assert frame["complementary_info.collector_policy_id"].tolist() == [0]*12
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


def test_all_configured_views_participate_in_snapshot(collector, ros_module):
    c = collector
    now = time.monotonic_ns()
    for key, (_, value) in list(c._latest.items()):
        c._latest[key] = (now, value)
    for camera in c.config.cameras:
        c._latest[f"camera.{camera.name}"] = (now, np.zeros((224, 224, 3), dtype=np.uint8))
    _, images = ros_module.RolloutCollector._snapshot(c, raise_on_invalid=True)
    assert set(images) == {cam.feature_key for cam in c.config.cameras}
    role = c.config.cameras[-1].name
    del c._latest[f"camera.{role}"]
    with pytest.raises(RuntimeError, match=f"missing: camera.{role}"):
        ros_module.RolloutCollector._snapshot(c, raise_on_invalid=True)


def test_demo_camera_poll_does_not_duplicate_images_and_faults_on_gap(collector, ros_module, monkeypatch):
    c = collector
    c.enable_teleop()
    c.start_episode()
    clock = [time.monotonic_ns()]
    monkeypatch.setattr(ros_module.time, "monotonic_ns", lambda: clock[0])
    monkeypatch.setattr(c, "_validate_robot_health", lambda values: None)
    monkeypatch.setattr(c, "_raw_state", lambda *args: np.zeros(100))
    recorded = []
    monkeypatch.setattr(c, "_record_frame", lambda *args, **kwargs: recorded.append(args[-1]))
    for camera in c.config.cameras:
        c._latest[f"camera.{camera.name}"] = (clock[0], None)
    c._input_action = np.tile([.4,0,.3,1,0,0,0,1,0,1], 2)
    c.coordinator.clock = lambda: clock[0]
    c._tick()
    c._gamepad_tick()
    c._tick()
    assert recorded == [clock[0]]
    clock[0] += 33_333_334
    for camera in c.config.cameras:
        c._latest[f"camera.{camera.name}"] = (clock[0], None)
    c._tick()
    c._gamepad_tick()
    assert len(recorded) == 2
    clock[0] += 100_000_000
    c._tick()
    assert c._faulted and not c._recording and not c._teleop_enabled


def test_online_save_waits_for_terminal_observation_and_controller_reset(collector):
    import threading
    from vla.core.online_contract import TIMING_FEATURES
    c=collector
    c.enable_teleop(); c.start_episode()
    c.require_online=True
    c._extra_features.update(TIMING_FEATURES)
    c._annotations.update({key:[] for key in TIMING_FEATURES})
    c._phase='rl'; c._subtask_index=1
    action=np.tile([.4,0,.3,1,0,0,0,1,0,.7],2)
    state=np.zeros(100)
    for off in (0,50):
        state[off:off+9]=action[:9]
        state[off+9]=.04
    values,images=c._snapshot()
    stamp=[time.monotonic_ns()]
    c.coordinator.clock=lambda:stamp[0]
    for _ in range(12):
        c._slot_id += 1
        c.coordinator.accept(action, source='gamepad', generation=c.coordinator.generation, slot_id=c._slot_id)
        c._track_control(.01)
        c._last_command_snapshot=replace(c._last_command_snapshot,published_at_ns=stamp[0])
        c._record_frame(state.copy(),values,images,stamp[0])
        stamp[0]+=33_333_333
        c._complete_interval(stamp[0])
    assert all(v==[1] for v in c._annotations['complementary_info.action_valid'])
    c.finish_episode('success')
    assert c._pending_finish=='success' and c._recording
    c._record_frame(state.copy(),values,images,stamp[0])
    c._annotations['complementary_info.segment_terminal'][-1]=[1]
    c._at_boundary=True
    # Writer schema is configured at construction in real online sessions.
    c.writer.extra_features.update(TIMING_FEATURES)
    c.finish_episode('success')
    c._save_worker.join(30)
    assert not c._recording and c._pending_online_save==(0,'success')
    called=threading.Event()
    c.policy_client=SimpleNamespace(process_saved_episode=lambda **kwargs:called.set(), close=lambda:None)
    c.coordinator.mode='resetting'
    c._online_save_tick()
    assert not called.is_set()
    c.coordinator.mode='standby'
    c.executor=SimpleNamespace(session_id='s', generation=1)
    c._online_save_tick()
    c.executor=None
    assert called.wait(1)
    c._online_worker.join(1)


@pytest.mark.parametrize('space,arms', [('joint',1),('joint',2),('tcp',1),('tcp',2)])
def test_native_labels_are_accepted_commands_for_every_ros_contract(collector,ros_module,monkeypatch,space,arms):
    base=collector
    config=replace(base.config,action_space=space,arms=base.config.arms[:arms],
        cameras=tuple(cam for cam in base.config.cameras if arms==2 or cam.name!='right_wrist'),
        dataset_root=base.config.dataset_root.parent/f'{space}-{arms}',subtasks=())
    c=ros_module.RolloutCollector(config,demonstration=True)
    c._latest=dict(base._latest)
    values={key:v[1] for key,v in c._latest.items()}
    images={cam.feature_key:np.zeros((224,224,3),np.uint8) for cam in config.cameras}
    monkeypatch.setattr(c,'_snapshot',lambda **kwargs:(values,images))
    try:
        c.enable_teleop(); c.start_episode()
        block=([.1]*7+[.73]) if space=='joint' else [.41,.02,.31,1,0,0,0,1,0,.73]
        target=np.tile(block,arms)
        raw=np.zeros(50*arms)
        for off in range(0,len(raw),50):
            raw[off:off+9]=[.4,0,.3,1,0,0,0,1,0]
            raw[off+9]=.02
        for _ in range(12):
            command=record_command(c,target,raw.copy(),values,images)
            np.testing.assert_array_equal(c._actions[-1],command.target)
        c._at_boundary=True
        c.finish_episode('success')
        c._save_worker.join(30)
        assert c.writer.episode_count==1
        table=pd.read_parquet(c.writer.data_path(0))
        np.testing.assert_allclose(np.stack(table['action']),np.tile(target,(12,1)))
        assert np.stack(table['observation.anchor']).shape==(12,config.action_dim)
        if space=='tcp':
            assert not np.allclose(np.stack(table['action']),np.stack(table['command.fk_tcp']))
        else:
            np.testing.assert_allclose(np.stack(table['action']),np.stack(table['command.joint']))
    finally:
        c.close()


def test_save_retry_after_commit_does_not_rewrite_episode(collector, monkeypatch):
    import vla.core.online_contract as persistence
    import vla.datacollection.episode_journal as journal
    c = collector
    c.enable_teleop()
    c.start_episode()
    c._subtask_index = 1
    target = np.tile([.4, 0, .3, 1, 0, 0, 0, 1, 0, .5], 2)
    raw = np.zeros(100)
    for off in (0,50):
        raw[off:off+9] = target[:9]
        raw[off+9] = .04
    values, images = c._snapshot()
    for _ in range(12):
        record_command(c, target, raw.copy(), values, images)
    original = persistence.durable_json
    def fail_marker(path, value):
        if path.name == 'published.json':
            raise OSError(28, 'injected post-commit failure')
        return original(path, value)
    monkeypatch.setattr(persistence, 'durable_json', fail_marker)
    c._at_boundary = True
    c.finish_episode('success')
    c._save_worker.join(30)
    assert c.writer.episode_count == 1 and c._storage_blocked
    before = {p: persistence.file_digest(p) for p in c.config.dataset_root.rglob('*.mp4')}
    monkeypatch.setattr(persistence, 'durable_json', original)
    monkeypatch.setattr(journal, 'materialize', lambda *a, **kw: pytest.fail('retry rewrote committed video'))
    c.finish_episode('success')
    c._save_worker.join(30)
    assert c.writer.episode_count == 1 and not c._storage_blocked
    assert not c._recording and not c._teleop_enabled
    assert before == {p: persistence.file_digest(p) for p in before}


@pytest.mark.parametrize('incomplete', [False, True])
def test_stopped_collector_recovers_only_durable_prefix(collector, incomplete):
    from vla.cli.recover_episode import recover_episode
    from vla.datacollection.episode_journal import recover_staging
    c = collector
    c.enable_teleop()
    c.start_episode()
    target = np.tile([.4,0,.3,1,0,0,0,1,0,.5],2)
    raw = np.zeros(100)
    for off in (0,50):
        raw[off:off+9] = target[:9]
        raw[off+9] = .04
    values, images = c._snapshot()
    for _ in range(32):
        record_command(c, target, raw.copy(), values, images)
    staging = c._journal.path
    c.close()
    durable = recover_staging(staging)
    if incomplete:
        (staging/'complete.json').unlink()
        (staging/durable.segments[-1]['arrays']).write_bytes(b'crash tail')
    expected = 30 if incomplete else 32
    assert recover_episode(c.config.dataset_root, staging)['durable_frames'] == expected
    result = recover_episode(c.config.dataset_root, staging, outcome='success')
    assert result['episode_index'] == 0 and result['incomplete_tail'] is incomplete
    frame = pd.read_parquet(c.writer.data_path(0))
    assert len(frame) == expected
    assert not np.stack(frame['complementary_info.task_terminal']).any()
    np.testing.assert_allclose(np.stack(frame['action']), np.tile(target,(expected,1)))
    assert recover_episode(c.config.dataset_root,staging,outcome='success')['episode_index'] == 0
    assert not c._recording and not c._teleop_enabled
