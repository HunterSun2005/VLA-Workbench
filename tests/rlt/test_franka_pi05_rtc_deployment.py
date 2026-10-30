from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from vla.deployment.franka_pi05_rtc import (
    LEGACY_PADDING_CAMERAS,
    PHYSICAL_CAMERAS,
    DeploymentLoop,
    JsonlDebugTrace,
    MockFrankaRobotiqRobot,
    Pi05RTC,
    policy_observation,
    validate_checkpoint,
)
from vla.cli.deploy_franka_pi05_rtc import (
    _ControlSessionConfig,
    _HomeConfig,
    _build_robot_config,
    _confirm_execute,
    _confirm_restart,
    _control_timing,
    _load_yaml,
    _load_home_config,
    _move_robot_home,
    _parse_cpu_list,
    _run_control_sessions,
)
from vla.robots.franka_robotiq.robot import ACTION_FIELDS, STATE_FIELDS


def _checkpoint(tmp_path, **updates):
    root = tmp_path / "checkpoint"
    root.mkdir()
    inputs = {
        key: {"type": "VISUAL", "shape": [3, 224, 224]}
        for key in PHYSICAL_CAMERAS
    }
    config = {
        "type": "pi05", "input_features": {**inputs, "observation.state": {"shape": [10]}},
        "output_features": {"action": {"shape": [10]}}, "use_relative_actions": False,
        "chunk_size": 50, "action_feature_names": list(ACTION_FIELDS),
    }
    config.update(updates)
    (root / "config.json").write_text(json.dumps(config))
    (root / "policy_preprocessor.json").write_text(json.dumps({"steps": [{
        "registry_name": "tokenizer_processor", "config": {"tokenizer_name": "cached-tokenizer"}
    }]}))
    (root / "policy_postprocessor.json").write_text('{"steps": []}')
    stats = {
        "action.min": torch.tensor([0.4, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        "action.max": torch.tensor([0.7, 0.2, 0.6, 0.0, 1.0, 0.2, -1.0, 0.0, 0.2, 1.0]),
        "action.q01": torch.tensor([0.42, -0.18, 0.32, 0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0]),
        "action.q99": torch.tensor([0.68, 0.18, 0.58, 0.0, 0.9, 0.1, -0.9, 0.0, 0.1, 1.0]),
    }
    save_file(stats, root / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    (root / "model.safetensors").write_bytes(b"weights")
    return root


def test_checkpoint_requires_absolute_10d_and_two_camera_contract(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: "/cached/tokenizer")
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: object())
    contract = validate_checkpoint(_checkpoint(tmp_path))
    assert contract.tokenizer == "/cached/tokenizer"
    assert set(PHYSICAL_CAMERAS).isdisjoint(LEGACY_PADDING_CAMERAS)
    bad = tmp_path / "bad"
    bad.mkdir()
    root = _checkpoint(bad, use_relative_actions=True)
    with pytest.raises(ValueError, match="absolute actions"):
        validate_checkpoint(root)


def test_checkpoint_rejects_legacy_padding_camera_features(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: "/cached/tokenizer")
    root = _checkpoint(tmp_path)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    for key in LEGACY_PADDING_CAMERAS:
        config["input_features"][key] = {"type": "VISUAL", "shape": [3, 224, 224]}
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="exact two physical cameras"):
        validate_checkpoint(root)


def test_checkpoint_rejects_degenerate_action_unnormalizer(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: "/cached/tokenizer")
    root = _checkpoint(tmp_path)
    fixed = torch.tensor([0.55, -0.04, 0.41, 0.7, 0.71, -0.03, 0.71, -0.7, -0.02, 0.0])
    stats = {
        "action.min": fixed,
        "action.max": fixed.clone(),
        "action.q01": fixed.clone(),
        "action.q99": fixed.clone(),
    }
    save_file(stats, root / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    with pytest.raises(ValueError, match="degenerate constant TCP stats"):
        validate_checkpoint(root)


def test_policy_observation_only_supplies_two_real_cameras() -> None:
    raw = {name: index / 10 for index, name in enumerate(STATE_FIELDS)}
    raw.update({name: np.zeros((224, 224, 3), dtype=np.uint8) for name in ("base", "left_wrist")})
    observation = policy_observation(raw, "task")
    assert set(observation) == {"observation.state", "task", *PHYSICAL_CAMERAS}
    assert not set(LEGACY_PADDING_CAMERAS) & set(observation)


def test_jsonl_debug_trace_writes_structured_events(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    trace = JsonlDebugTrace(path)
    trace.emit("control_step", step=7, predicted_xyz=[0.1, 0.2, 0.3], predicted_gripper=0.5)
    trace.close()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["event"] == "control_step"
    assert records[0]["step"] == 7
    assert records[0]["predicted_xyz"] == [0.1, 0.2, 0.3]
    assert records[-1]["event"] == "trace_closed"
    assert records[-1]["dropped_records"] == 0


def test_deployment_yaml_defaults_and_noninteractive_arm_guard(monkeypatch) -> None:
    values = _load_yaml(Path("src/vla/configs/deployment/franka_single_left_pi05_rtc.yaml"))
    config = _build_robot_config(values)
    assert config.robot_ip == "172.16.0.2"
    assert config.camera_serials["base"] == "260722303785"
    assert config.gripper_device_id == 9
    assert config.realtime_cpu == 6
    assert _parse_cpu_list(values["deployment"]["housekeeping_cpus"]) == {
        *range(5), *range(8, 24)
    }
    assert config.gripper_action_threshold == pytest.approx(0.75)
    assert config.gripper_close_trigger == pytest.approx(0.75)
    assert config.gripper_open_trigger == pytest.approx(0.875)
    assert config.gripper_post_close_lock_s == pytest.approx(2.0)
    assert config.gripper_startup_open_fraction == pytest.approx(0.95)
    assert config.gripper_startup_open_timeout_s == pytest.approx(5.0)
    assert config.dynamics_factor == pytest.approx(0.10)
    assert config.max_linear_velocity == pytest.approx(0.05)
    assert config.max_angular_velocity == pytest.approx(0.20)
    assert config.target_smoothing_alpha == pytest.approx(0.25)
    assert config.velocity_smoothing == pytest.approx(0.30)
    assert config.smooth_angular_velocity is True
    assert config.velocity_tracking_tau_s == pytest.approx(0.15)
    assert config.min_command_position_delta == pytest.approx(0.0008)
    assert config.min_command_rotation_delta == pytest.approx(0.01)
    assert config.velocity_motion_duration_ms == 100
    assert config.max_consecutive_motion_failures == 5
    assert config.motion_cleanup_timeout_s == pytest.approx(1.0)
    assert config.motion_recovery_timeout_s == pytest.approx(1.0)
    assert config.motion_recovery_poll_interval_s == pytest.approx(0.02)
    assert config.camera_max_pair_skew_s == pytest.approx(0.050)
    assert config.observation_max_image_age_s == pytest.approx(0.100)
    assert config.workspace_min == (0.1, -0.55, 0.05)
    assert values["deployment"]["torch_num_threads"] == 1
    assert values["deployment"]["torch_num_interop_threads"] == 1
    assert values["deployment"]["control_discontinuity_timeout_ms"] == 100
    assert values["deployment"]["task"] == "Place the white lid firmly onto the jig."
    home = _load_home_config(values["deployment"], config)
    assert home is None
    assert values["deployment"]["home"]["target_action"] is None
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert _confirm_execute(False) is False
    assert _confirm_execute(True) is True
    assert _confirm_restart(RuntimeError("fault"), 0) is False


def test_cpu_list_parser_rejects_invalid_ranges() -> None:
    assert _parse_cpu_list("0-2,5,8-9") == {0, 1, 2, 5, 8, 9}
    with pytest.raises(ValueError, match="invalid CPU range"):
        _parse_cpu_list("7-3")


def test_control_timing_requires_matching_rates_and_bounded_timeout() -> None:
    values = _load_yaml(Path("src/vla/configs/deployment/franka_single_left_pi05_rtc.yaml"))
    config = _build_robot_config(values)
    assert _control_timing(values["deployment"], config) == (30.0, 100.0)
    with pytest.raises(ValueError, match="must equal"):
        _control_timing({"fps": 20, "control_discontinuity_timeout_ms": 100}, config)
    with pytest.raises(ValueError, match="greater than one control period"):
        _control_timing({"fps": 30, "control_discontinuity_timeout_ms": 30}, config)
    with pytest.raises(ValueError, match="must not exceed"):
        _control_timing({"fps": 30, "control_discontinuity_timeout_ms": 101}, config)


def test_mock_hardware_provides_complete_observation_and_never_executes() -> None:
    robot = MockFrankaRobotiqRobot((0.2, -0.45, 0.05), (0.75, 0.45, 0.75))
    robot.connect()
    observation = policy_observation(robot.get_observation(), "task")
    np.testing.assert_allclose(observation["observation.state"][:3], [0.475, 0.0, 0.4])
    assert observation["observation.state"][9] == pytest.approx(0.0425)
    assert set(PHYSICAL_CAMERAS) <= set(observation)
    with pytest.raises(AssertionError, match="never execute"):
        robot.send_action(np.zeros(10))
    robot.disconnect()
    assert not robot.is_connected


def test_control_session_restart_reuses_loaded_policy_and_rtc() -> None:
    class ResidentPolicy:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    class ResidentRTC:
        def __init__(self):
            self.stop_calls = 0
            self.published = []
            self.infer_calls = 0

        def stop(self):
            self.stop_calls += 1

        def publish(self, observation):
            self.published.append(observation)

        def infer(self, *, account_for_latency):
            assert account_for_latency is False
            self.infer_calls += 1

    class SessionRobot:
        def __init__(self):
            self.connected = False
            self.disconnect_calls = 0
            self.prepare_calls = 0

        def connect(self):
            self.connected = True

        def get_observation(self):
            return {"connected": self.connected}

        def prepare_for_control(self):
            self.prepare_calls += 1
            return 1.0

        def disconnect(self):
            self.disconnect_calls += 1
            self.connected = False

    class SessionLoop:
        instances = []

        def __init__(self, robot, rtc, **kwargs):
            self.robot = robot
            self.rtc = rtc
            self.session_id = kwargs["session_id"]
            self.fps = kwargs["fps"]
            self.execute = kwargs["execute"]
            self.__class__.instances.append(self)

        def preflight(self):
            return np.zeros(10, dtype=np.float32)

        def request_stop(self, reason):
            self.stop_reason = reason

        def run(self):
            assert self.execute
            if self.session_id == 0:
                raise RuntimeError("simulated Franka control fault")

    policy = ResidentPolicy()
    rtc = ResidentRTC()
    robots = []
    arm_prompts = []
    restart_prompts = []

    def robot_factory():
        robot = SessionRobot()
        robots.append(robot)
        return robot

    result = _run_control_sessions(
        robot_factory=robot_factory,
        loop_factory=SessionLoop,
        rtc=rtc,
        policy=policy,
        task="task",
        fps=30.0,
        execute_requested=True,
        trace=None,
        console_log_every=30,
        discontinuity_timeout_s=0.1,
        observation_builder=lambda observation, task: {**observation, "task": task},
        confirm_execute=lambda execute: arm_prompts.append(execute) or True,
        confirm_restart=lambda error, session_id: restart_prompts.append((error, session_id)) or True,
    )

    assert result == 0
    assert arm_prompts == [True]
    assert len(restart_prompts) == 1
    assert restart_prompts[0][1] == 0
    assert [loop.session_id for loop in SessionLoop.instances] == [0, 1]
    assert all(loop.rtc is rtc for loop in SessionLoop.instances)
    assert len(robots) == 2
    assert [robot.disconnect_calls for robot in robots] == [1, 1]
    assert [robot.prepare_calls for robot in robots] == [1, 1]
    assert policy.reset_calls == 4
    assert rtc.infer_calls == 2
    assert rtc.stop_calls == 6


def test_control_session_restart_reloads_session_config() -> None:
    class PolicyConfig:
        rtc_config = None

    class Policy:
        config = PolicyConfig()

        def __init__(self):
            self.rtc_processor_configs = []

        def reset(self):
            pass

        def init_rtc_processor(self):
            self.rtc_processor_configs.append(self.config.rtc_config)

    class RTC:
        def __init__(self):
            self.reconfigured = []
            self.published = []

        def stop(self):
            pass

        def reconfigure(self, **kwargs):
            self.reconfigured.append(kwargs)

        def publish(self, observation):
            self.published.append(observation)

        def infer(self, *, account_for_latency):
            assert account_for_latency is False

    class Robot:
        def __init__(self, name):
            self.name = name

        def connect(self):
            pass

        def get_observation(self):
            return {"robot": self.name}

        def prepare_for_control(self):
            return 1.0

        def disconnect(self):
            pass

    class Loop:
        instances = []

        def __init__(self, robot, rtc, **kwargs):
            self.robot = robot
            self.fps = kwargs["fps"]
            self.task = kwargs["task"]
            self.session_id = kwargs["session_id"]
            self.execute = kwargs["execute"]
            self.__class__.instances.append(self)

        def preflight(self):
            return np.zeros(10, dtype=np.float32)

        def request_stop(self, reason):
            pass

        def run(self):
            if self.session_id == 0:
                raise RuntimeError("reload test")

    rtc_configs = [object(), object()]
    loaded = []

    def session_config_loader():
        index = len(loaded)
        loaded.append(index)
        return _ControlSessionConfig(
            robot_factory=lambda index=index: Robot(f"robot-{index}"),
            task=f"task-{index}",
            fps=20.0 + index,
            console_log_every=5 + index,
            discontinuity_timeout_s=0.1,
            rtc_config=rtc_configs[index],
            queue_threshold=10 + index,
        )

    rtc = RTC()
    policy = Policy()
    result = _run_control_sessions(
        robot_factory=lambda: (_ for _ in ()).throw(AssertionError("static factory used")),
        loop_factory=Loop,
        rtc=rtc,
        policy=policy,
        task="static task",
        fps=30.0,
        execute_requested=True,
        trace=None,
        console_log_every=30,
        discontinuity_timeout_s=0.1,
        observation_builder=lambda observation, task: {**observation, "task": task},
        session_config_loader=session_config_loader,
        confirm_execute=lambda execute: True,
        confirm_restart=lambda error, session_id: True,
    )

    assert result == 0
    assert loaded == [0, 1]
    assert [loop.robot.name for loop in Loop.instances] == ["robot-0", "robot-1"]
    assert [loop.task for loop in Loop.instances] == ["task-0", "task-1"]
    assert [loop.fps for loop in Loop.instances] == [20.0, 21.0]
    assert [item["fps"] for item in rtc.reconfigured] == [20.0, 21.0]
    assert [item["queue_threshold"] for item in rtc.reconfigured] == [10, 11]
    assert [item["task"] for item in rtc.published] == ["task-0", "task-1"]
    assert policy.config.rtc_config is rtc_configs[1]
    assert policy.rtc_processor_configs == rtc_configs


def test_control_key_stops_only_hardware_session_and_keeps_model_restart_prompt() -> None:
    class Policy:
        def reset(self):
            pass

    class RTC:
        def stop(self):
            pass

        def publish(self, observation):
            pass

        def infer(self, *, account_for_latency):
            assert account_for_latency is False

    class Robot:
        def __init__(self):
            self.disconnect_calls = 0

        def connect(self):
            pass

        def get_observation(self):
            return {}

        def disconnect(self):
            self.disconnect_calls += 1

    class Loop:
        def __init__(self, robot, rtc, **kwargs):
            self.fps = kwargs["fps"]
            self.execute = kwargs["execute"]
            self.stop_reason = None

        def preflight(self):
            return np.zeros(10, dtype=np.float32)

        def request_stop(self, reason):
            self.stop_reason = reason

        def run(self):
            assert self.execute
            return self.stop_reason

    class StopKey:
        def __init__(self, callback, **kwargs):
            assert kwargs == {"home_enabled": False, "home_key": "h"}
            self.callback = callback
            self.closed = False

        def start(self):
            self.callback("keyboard")

        def close(self):
            self.closed = True

    robot = Robot()
    restart_reasons = []
    result = _run_control_sessions(
        robot_factory=lambda: robot,
        loop_factory=Loop,
        rtc=RTC(),
        policy=Policy(),
        task="task",
        fps=30.0,
        execute_requested=True,
        trace=None,
        console_log_every=30,
        discontinuity_timeout_s=0.1,
        observation_builder=lambda observation, task: observation,
        confirm_execute=lambda execute: True,
        confirm_restart=lambda error, session_id: restart_reasons.append((error, session_id)) or False,
        control_key_listener_factory=StopKey,
    )

    assert result == 0
    assert restart_reasons == [(None, 0)]
    assert robot.disconnect_calls == 1


def test_home_return_uses_lift_transit_and_settle_route() -> None:
    values = _load_yaml(Path("src/vla/configs/deployment/franka_single_left_pi05_rtc.yaml"))
    values["deployment"]["home"].update(enabled=True, target_action=[0.3, -0.3, 0.3, 1, 0, 0, 0, -1, 0, 1])
    config = _load_home_config(values["deployment"], _build_robot_config(values))
    assert config is not None

    class Robot:
        def __init__(self):
            self.pose = np.eye(4)
            self.pose[:3, 3] = [0.4, -0.2, 0.2]
            self.actions = []
            self.stop_calls = 0

        def measured_pose(self):
            return self.pose.copy()

        def send_action(self, action):
            action = np.asarray(action, dtype=np.float64)
            self.actions.append(action.copy())
            self.pose[:3, 3] = action[:3]
            from vla.cli.deploy_franka_pi05_rtc import _rot6d_to_matrix

            self.pose[:3, :3] = _rot6d_to_matrix(action[3:9])

        def stop(self):
            self.stop_calls += 1

    robot = Robot()
    one_step = dataclasses.replace(config, settled_steps=1)
    _move_robot_home(
        robot,
        one_step,
        fps=30.0,
        trace=None,
        session_id=0,
        sleep=lambda _: None,
    )

    assert len(robot.actions) == 3
    np.testing.assert_allclose(robot.actions[0][:3], [0.4, -0.2, 0.45])
    assert robot.actions[0][9] == pytest.approx(0.80)
    np.testing.assert_allclose(robot.actions[1][:3], [*one_step.target_action[:2], 0.45])
    np.testing.assert_allclose(robot.actions[2], one_step.target_action)
    assert robot.stop_calls == 3


def test_home_key_stops_policy_runs_home_and_requires_restart() -> None:
    class Policy:
        def reset(self):
            pass

    class RTC:
        def stop(self):
            pass

        def publish(self, observation):
            pass

        def infer(self, *, account_for_latency):
            assert account_for_latency is False

    class Robot:
        def connect(self):
            pass

        def get_observation(self):
            return {}

        def prepare_for_control(self):
            return 1.0

        def disconnect(self):
            pass

    class Loop:
        def __init__(self, robot, rtc, **kwargs):
            self.execute = kwargs["execute"]
            self.fps = kwargs["fps"]
            self.reason = None

        def preflight(self):
            return np.zeros(10, dtype=np.float32)

        def request_stop(self, reason):
            self.reason = reason

        def run(self):
            return self.reason

    class HomeKey:
        def __init__(self, callback, **kwargs):
            assert kwargs == {"home_enabled": True, "home_key": "h"}
            self.callback = callback

        def start(self):
            self.callback("home")

        def close(self):
            pass

    home = _HomeConfig(
        target_action=(0.3, -0.35, 0.37, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        clearance_z=0.45,
        gripper_hold_action=0.80,
    )
    moved = []
    restarts = []
    result = _run_control_sessions(
        robot_factory=Robot,
        loop_factory=Loop,
        rtc=RTC(),
        policy=Policy(),
        task="task",
        fps=30.0,
        execute_requested=True,
        trace=None,
        console_log_every=30,
        discontinuity_timeout_s=0.1,
        observation_builder=lambda observation, task: observation,
        session_config_loader=lambda: _ControlSessionConfig(
            robot_factory=Robot,
            task="task",
            fps=30.0,
            console_log_every=30,
            discontinuity_timeout_s=0.1,
            home=home,
        ),
        confirm_execute=lambda execute: True,
        confirm_restart=lambda error, session_id: restarts.append((error, session_id)) or False,
        control_key_listener_factory=HomeKey,
        home_mover=lambda robot, config, **kwargs: moved.append((robot, config, kwargs)),
    )

    assert result == 0
    assert len(moved) == 1
    assert moved[0][1] is home
    assert restarts == [(None, 0)]


class FakePolicy:
    def __init__(self, runtime=None):
        self.calls = []
        self.runtime = runtime
        self.reset_calls = 0

    def predict_action_chunk(self, batch, **kwargs):
        self.calls.append(kwargs)
        if self.runtime is not None:
            self.runtime.queue.get()
        return torch.arange(50 * 10, dtype=torch.float32).reshape(1, 50, 10)

    def reset(self):
        self.reset_calls += 1


class Clock:
    def __init__(self, *values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class LoopClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class ClockedStopEvent:
    def __init__(self, clock, oversleeps=()):
        self.clock = clock
        self.oversleeps = iter(oversleeps)
        self.stopped = False

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, timeout=None):
        if timeout is not None and not self.stopped:
            self.clock.advance(timeout + next(self.oversleeps, 0.0))
        return self.stopped


def _rtc(policy=None, clock=None, *, queue_threshold=30):
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    policy = policy or FakePolicy()
    runtime = Pi05RTC(
        policy, lambda obs: obs, lambda chunk: chunk + 1000,
        rtc_config=RTCConfig(enabled=True, execution_horizon=10), fps=10,
        queue_threshold=queue_threshold, clock=clock or Clock(0.0, 0.0),
    )
    return runtime, policy


def test_first_chunk_has_no_prefix_and_postprocessor_is_absolute() -> None:
    runtime, policy = _rtc()
    runtime.publish({"observation.state": torch.zeros(1, 10)})
    runtime.infer()
    assert policy.calls[0]["prev_chunk_left_over"] is None
    assert policy.calls[0]["inference_delay"] == 0
    np.testing.assert_array_equal(runtime.get().numpy(), np.arange(10) + 1000)


def test_rtc_reconfigure_replaces_session_queue_without_reloading_policy() -> None:
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    runtime, policy = _rtc(queue_threshold=30)
    old_queue = runtime.queue
    new_config = RTCConfig(enabled=True, execution_horizon=7)

    runtime.reconfigure(rtc_config=new_config, fps=20.0, queue_threshold=12)

    assert runtime.policy is policy
    assert runtime.queue is not old_queue
    assert runtime.queue.cfg is new_config
    assert runtime.fps == pytest.approx(20.0)
    assert runtime.queue_threshold == 12
    assert runtime.debug_snapshot()["queue_size"] == 0
    runtime.close()


def test_synchronous_prime_does_not_discard_chunk_for_latency() -> None:
    runtime, _ = _rtc(clock=Clock(0.0, 10.0))
    runtime.publish({"observation.state": torch.zeros(1, 10)})
    runtime.infer(account_for_latency=False)
    assert runtime.queue.qsize() == 50
    np.testing.assert_array_equal(runtime.get().numpy(), np.arange(10) + 1000)


def test_trace_provenance_survives_replacement_delay_and_reset() -> None:
    runtime, _ = _rtc(clock=time.perf_counter)
    runtime.publish({"observation.state": np.zeros(10)})
    runtime.infer(account_for_latency=False)
    runtime.get()
    first = runtime.debug_snapshot()["last_action_provenance"]
    assert first["chunk_action_index"] == 0
    request = runtime._request()
    runtime.get()
    runtime.get()  # two actions consumed during simulated inference
    runtime.infer(request, account_for_latency=False)
    action = runtime.get()
    second = runtime.debug_snapshot()["last_action_provenance"]
    assert second["chunk_id"] != first["chunk_id"]
    assert second["input_observation_id"] == first["input_observation_id"]
    assert second["chunk_action_index"] == 2
    np.testing.assert_array_equal(action.numpy(), np.arange(20, 30) + 1000)
    runtime.reset()
    assert runtime.debug_snapshot()["last_action_provenance"] == {}
    runtime.publish({"observation.state": np.ones(10)})
    runtime.infer(account_for_latency=False)
    runtime.get()
    third = runtime.debug_snapshot()["last_action_provenance"]
    assert third["generation"] != second["generation"]
    assert third["input_observation_id"] != second["input_observation_id"]
    assert third["chunk_action_index"] == 0
    runtime.close()


def test_next_chunk_uses_actual_consumed_delay_not_wall_clock_delay() -> None:
    policy = FakePolicy()
    runtime, _ = _rtc(policy, Clock(0.0, 0.15))
    policy.runtime = runtime
    old = torch.full((50, 10), -7.0)
    runtime.queue.merge(old, old + 100, 0)
    runtime.queue.get()
    runtime.publish({"observation.state": torch.zeros(1, 10)})
    runtime.infer()
    assert torch.equal(policy.calls[0]["prev_chunk_left_over"], old[1:])
    assert runtime.queue.qsize() == 49
    np.testing.assert_array_equal(runtime.get().numpy(), np.arange(10, 20) + 1000)


def test_one_frame_latency_rounding_difference_does_not_warn(caplog) -> None:
    policy = FakePolicy()
    runtime, _ = _rtc(policy, Clock(0.0, 0.15))
    policy.runtime = runtime
    old = torch.zeros(50, 10)
    runtime.queue.merge(old, old, 0)
    runtime.publish({"observation.state": torch.zeros(1, 10)})
    runtime.infer()
    assert "latency/consumption mismatch" not in caplog.text


def test_followup_chunk_can_enable_autograd_for_rtc_guidance() -> None:
    class GradientGuidancePolicy(FakePolicy):
        def predict_action_chunk(self, batch, **kwargs):
            if kwargs["prev_chunk_left_over"] is not None:
                with torch.enable_grad():
                    latent = torch.ones(1, requires_grad=True)
                    output = latent.square()
                    assert torch.autograd.grad(output, latent)[0].item() == pytest.approx(2.0)
            return super().predict_action_chunk(batch, **kwargs)

    runtime, _ = _rtc(GradientGuidancePolicy(), Clock(0.0, 0.0, 1.0, 1.0))
    runtime.publish({"observation.state": torch.zeros(1, 10)})
    runtime.infer()
    runtime.get()
    runtime.infer()
    assert runtime.queue.qsize() > 0


class FakeRobot:
    def __init__(self):
        self.sent = []
        self.stop_calls = 0

    def get_observation(self):
        result = {name: 0.0 for name in STATE_FIELDS}
        result.update({name: np.zeros((224, 224, 3), dtype=np.uint8) for name in ("base", "left_wrist")})
        return result

    def check_health(self):
        pass

    def send_action(self, action):
        self.sent.append(action)

    def stop(self):
        self.stop_calls += 1


class ClockedRobot(FakeRobot):
    def __init__(self, clock, stop_event, *, stop_after, first_command_stall_s=0.0):
        super().__init__()
        self.clock = clock
        self.stop_event = stop_event
        self.stop_after = stop_after
        self.first_command_stall_s = first_command_stall_s
        self.sent_at = []

    def send_action(self, action):
        self.sent.append(action)
        self.sent_at.append(self.clock())
        if len(self.sent) == 1 and self.first_command_stall_s:
            self.clock.advance(self.first_command_stall_s)
        if len(self.sent) >= self.stop_after:
            self.stop_event.set()


def test_disarmed_loop_never_sends_and_preflight_only_infers() -> None:
    runtime, _ = _rtc()
    robot = FakeRobot()
    loop = DeploymentLoop(robot, runtime, task="task", execute=False)
    action = loop.preflight()
    assert action.shape == (10,)
    assert robot.sent == []
    with pytest.raises(PermissionError, match="disarmed"):
        loop.run()


def test_queue_underrun_stops_robot() -> None:
    runtime, _ = _rtc()
    runtime.publish({"observation.state": torch.zeros(1, 10)})
    robot = FakeRobot()
    loop = DeploymentLoop(robot, runtime, task="task", execute=True)
    runtime.maybe_start = lambda: None
    with pytest.raises(RuntimeError, match="queue is empty"):
        loop.run()
    assert robot.stop_calls == 1


def test_empty_queue_waits_for_slow_refill_and_rebases_deadline() -> None:
    inference_started = threading.Event()
    release_inference = threading.Event()

    class SlowSecondInferencePolicy(FakePolicy):
        def predict_action_chunk(self, batch, **kwargs):
            if len(self.calls) == 1:
                inference_started.set()
                assert release_inference.wait(timeout=2)
            return super().predict_action_chunk(batch, **kwargs)

    class Trace:
        def __init__(self):
            self.records = []
            self.queue_wait_started = threading.Event()

        def emit(self, event, **fields):
            self.records.append((event, fields))
            if event == "rtc_queue_wait_started":
                self.queue_wait_started.set()

    class StopAfterTwoActionsRobot(FakeRobot):
        def __init__(self):
            super().__init__()
            self.loop = None

        def send_action(self, action):
            super().send_action(action)
            if len(self.sent) == 2:
                self.loop.request_stop("test-complete")

    runtime, policy = _rtc(
        SlowSecondInferencePolicy(),
        time.perf_counter,
        queue_threshold=40,
    )
    runtime.publish({"observation.state": torch.zeros(1, 10)})
    runtime.infer(account_for_latency=False)
    for _ in range(50):
        assert runtime.get() is not None
    assert runtime.get() is None

    trace = Trace()
    robot = StopAfterTwoActionsRobot()
    loop = DeploymentLoop(
        robot,
        runtime,
        task="task",
        fps=30,
        execute=True,
        trace=trace,
        discontinuity_timeout_s=0.1,
    )
    robot.loop = loop
    result = []
    errors = []

    def run_loop():
        try:
            result.append(loop.run())
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run_loop)
    thread.start()
    assert inference_started.wait(timeout=2)
    assert trace.queue_wait_started.wait(timeout=2)
    release_inference.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert errors == []
    assert result == ["test-complete"]
    assert len(policy.calls) == 2
    assert len(robot.sent) == 2
    assert robot.stop_calls == 1
    events = [event for event, _ in trace.records]
    assert "rtc_queue_wait_completed" in events
    control_steps = [fields for event, fields in trace.records if event == "control_step"]
    assert control_steps[0]["queue_wait_count"] == 1
    assert control_steps[0]["queue_wait_ms"] >= 0
    assert control_steps[1]["queue_wait_count"] == 0
    runtime.close()


def test_refill_wait_can_be_interrupted_by_stop_request() -> None:
    inference_started = threading.Event()
    release_inference = threading.Event()

    class BlockingPolicy(FakePolicy):
        def predict_action_chunk(self, batch, **kwargs):
            inference_started.set()
            assert release_inference.wait(timeout=2)
            return super().predict_action_chunk(batch, **kwargs)

    runtime, _ = _rtc(BlockingPolicy(), time.perf_counter, queue_threshold=40)
    runtime.publish({"observation.state": torch.zeros(1, 10)})
    runtime.maybe_start()
    assert inference_started.wait(timeout=2)
    stop_event = threading.Event()
    stop_event.set()

    assert runtime.wait_for_refill(stop_event) is False

    release_inference.set()
    runtime.stop()
    runtime.close()


def test_empty_queue_preserves_three_failure_limit_for_fast_refills() -> None:
    class FailingPolicy(FakePolicy):
        def predict_action_chunk(self, batch, **kwargs):
            self.calls.append(kwargs)
            raise RuntimeError("inference failed")

    runtime, policy = _rtc(FailingPolicy(), time.perf_counter, queue_threshold=40)
    robot = FakeRobot()
    loop = DeploymentLoop(robot, runtime, task="task", execute=True)

    with pytest.raises(RuntimeError, match="RTC inference failed"):
        loop.run()

    assert len(policy.calls) == 3
    assert robot.stop_calls == 1
    runtime.close()


def test_keyboard_stop_request_stops_robot_without_reporting_control_failure() -> None:
    class Trace:
        def __init__(self):
            self.records = []

        def emit(self, event, **fields):
            self.records.append((event, fields))

    runtime, _ = _rtc()
    robot = FakeRobot()
    trace = Trace()
    loop = DeploymentLoop(robot, runtime, task="task", execute=True, trace=trace, session_id=4)
    loop.request_stop("keyboard")

    assert loop.run() == "keyboard"
    assert robot.stop_calls == 1
    assert runtime.queue.qsize() == 0
    assert ("control_stop_requested", {"session_id": 4, "step": 0, "reason": "keyboard"}) in trace.records
    assert all(event != "control_failed" for event, _ in trace.records)


@pytest.mark.parametrize("stop_stage", ["observation", "dequeue"])
def test_stop_during_control_frame_does_not_send_another_action(monkeypatch, stop_stage) -> None:
    runtime, _ = _rtc()
    robot = FakeRobot()
    loop = DeploymentLoop(robot, runtime, task="task", execute=True)
    loop.preflight()
    if stop_stage == "observation":
        get_observation = robot.get_observation

        def observe_and_stop():
            loop.request_stop("keyboard")
            return get_observation()

        monkeypatch.setattr(robot, "get_observation", observe_and_stop)
    else:
        get_action = runtime.get

        def dequeue_and_stop():
            loop.request_stop("keyboard")
            return get_action()

        monkeypatch.setattr(runtime, "get", dequeue_and_stop)
    try:
        assert loop.run() == "keyboard"
        assert robot.sent == []
        assert robot.stop_calls == 1
        assert runtime.queue.qsize() == 0
    finally:
        runtime.close()


def test_loop_emits_recovered_motion_events_before_control_step() -> None:
    class Trace:
        def __init__(self):
            self.records = []

        def emit(self, event, **fields):
            self.records.append((event, fields))

    class EventRobot(FakeRobot):
        def __init__(self):
            super().__init__()
            self.motion_events = []
            self.loop = None

        def send_action(self, action):
            super().send_action(action)
            self.motion_events.extend(
                [
                    {"event": "franky_motion_failed", "source": "poll", "motion_event_sequence": 1},
                    {
                        "event": "franky_motion_recovery_completed",
                        "source": "poll",
                        "motion_event_sequence": 2,
                    },
                ]
            )
            self.loop.request_stop("test-complete")

        def drain_motion_events(self):
            events, self.motion_events = self.motion_events, []
            return events

    runtime, _ = _rtc()
    runtime.publish({"observation.state": torch.zeros(1, 10)})
    runtime.infer(account_for_latency=False)
    runtime.maybe_start = lambda: None
    robot = EventRobot()
    trace = Trace()
    loop = DeploymentLoop(robot, runtime, task="task", execute=True, trace=trace, session_id=7)
    robot.loop = loop

    assert loop.run() == "test-complete"

    names = [event for event, _ in trace.records]
    assert names.index("franky_motion_failed") < names.index("franky_motion_recovery_completed")
    assert names.index("franky_motion_recovery_completed") < names.index("control_step")
    recovered = next(fields for event, fields in trace.records if event == "franky_motion_recovery_completed")
    assert recovered["session_id"] == 7
    assert recovered["step"] == 0
    assert recovered["source"] == "poll"
    runtime.close()


def test_loop_flushes_motion_events_before_control_failure() -> None:
    class Trace:
        def __init__(self):
            self.records = []

        def emit(self, event, **fields):
            self.records.append((event, fields))

    class FailingEventRobot(FakeRobot):
        def __init__(self):
            super().__init__()
            self.motion_events = []

        def send_action(self, action):
            self.motion_events.append(
                {"event": "franky_motion_recovery_failed", "source": "move", "operation": "state_refresh"}
            )
            raise RuntimeError("Franky remains faulted")

        def drain_motion_events(self):
            events, self.motion_events = self.motion_events, []
            return events

    runtime, _ = _rtc()
    runtime.publish({"observation.state": torch.zeros(1, 10)})
    runtime.infer(account_for_latency=False)
    runtime.maybe_start = lambda: None
    trace = Trace()
    loop = DeploymentLoop(
        FailingEventRobot(),
        runtime,
        task="task",
        execute=True,
        trace=trace,
        session_id=8,
    )

    with pytest.raises(RuntimeError, match="remains faulted"):
        loop.run()

    names = [event for event, _ in trace.records]
    assert names.index("franky_motion_recovery_failed") < names.index("control_failed")
    failed = next(fields for event, fields in trace.records if event == "franky_motion_recovery_failed")
    assert failed == {"session_id": 8, "step": 0, "source": "move", "operation": "state_refresh"}
    runtime.close()


def test_loop_preserves_primary_error_when_robot_stop_also_fails() -> None:
    runtime, _ = _rtc()
    runtime.publish({"observation.state": torch.zeros(1, 10)})
    robot = FakeRobot()
    robot.stop = lambda: (_ for _ in ()).throw(RuntimeError("stop failed"))
    loop = DeploymentLoop(robot, runtime, task="task", execute=True)
    runtime.maybe_start = lambda: None
    with pytest.raises(RuntimeError, match="queue is empty") as caught:
        loop.run()
    assert caught.value.__notes__ == ["Robot stop also failed: stop failed"]


def test_absolute_deadline_recovers_phase_after_short_wakeup_delay() -> None:
    runtime, _ = _rtc()
    runtime.publish({"observation.state": torch.zeros(1, 10)})
    runtime.infer(account_for_latency=False)
    runtime.maybe_start = lambda: None
    clock = LoopClock()
    stop_event = ClockedStopEvent(clock, oversleeps=(0.002,))
    robot = ClockedRobot(clock, stop_event, stop_after=3)
    loop = DeploymentLoop(
        robot,
        runtime,
        task="task",
        fps=30,
        execute=True,
        discontinuity_timeout_s=0.1,
        clock=clock,
    )
    loop._stop = stop_event

    loop.run()

    np.testing.assert_allclose(robot.sent_at, [0.0, 1 / 30 + 0.002, 2 / 30], atol=1e-9)


def test_eighty_ms_gap_discards_missed_actions_without_resynchronizing() -> None:
    runtime, _ = _rtc()
    runtime.publish({"observation.state": torch.zeros(1, 10)})
    runtime.infer(account_for_latency=False)
    runtime.maybe_start = lambda: None
    clock = LoopClock()
    stop_event = ClockedStopEvent(clock)
    robot = ClockedRobot(clock, stop_event, stop_after=2, first_command_stall_s=0.08)
    loop = DeploymentLoop(
        robot, runtime, task="task", fps=30, execute=True,
        discontinuity_timeout_s=0.1, clock=clock,
    )
    loop._stop = stop_event

    loop.run()

    np.testing.assert_array_equal(robot.sent[0], np.arange(10) + 1000)
    np.testing.assert_array_equal(robot.sent[1], np.arange(20, 30) + 1000)
    assert robot.stop_calls == 0


def test_control_discontinuity_stops_and_reprimes_from_fresh_observation() -> None:
    runtime, policy = _rtc(clock=Clock(0.0, 0.0, 1.0, 1.0))
    runtime.publish({"observation.state": torch.zeros(1, 10)})
    runtime.infer(account_for_latency=False)
    runtime.maybe_start = lambda: None
    clock = LoopClock()
    stop_event = ClockedStopEvent(clock)
    robot = ClockedRobot(clock, stop_event, stop_after=2, first_command_stall_s=0.101)
    loop = DeploymentLoop(
        robot,
        runtime,
        task="task",
        fps=30,
        execute=True,
        discontinuity_timeout_s=0.1,
        clock=clock,
    )
    loop._stop = stop_event

    loop.run()

    assert robot.stop_calls == 1
    assert len(policy.calls) == 2
    assert policy.reset_calls == 1
    np.testing.assert_array_equal(robot.sent[0], np.arange(10) + 1000)
    np.testing.assert_array_equal(robot.sent[1], np.arange(10) + 1000)


def test_stop_invalidates_stale_request() -> None:
    runtime, policy = _rtc(clock=Clock(0.0, 0.0))
    runtime.publish({"observation.state": torch.zeros(1, 10)})
    request = runtime._request()
    runtime.stop()
    runtime.infer(request)
    assert runtime.queue.qsize() == 0
    assert policy.calls == []


def test_stop_waits_for_inflight_inference_thread() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingPolicy(FakePolicy):
        def predict_action_chunk(self, batch, **kwargs):
            started.set()
            assert release.wait(timeout=2)
            return super().predict_action_chunk(batch, **kwargs)

    runtime, _ = _rtc(BlockingPolicy(), Clock(0.0, 0.0))
    runtime.publish({"observation.state": torch.zeros(1, 10)})
    runtime.maybe_start()
    assert started.wait(timeout=2)
    stopper = threading.Thread(target=runtime.stop)
    stopper.start()
    stopper.join(timeout=0.05)
    assert stopper.is_alive()
    release.set()
    stopper.join(timeout=2)
    assert not stopper.is_alive()
    assert runtime.queue.qsize() == 0
    runtime.close()


def test_persistent_worker_is_reused_across_refills() -> None:
    inference_started = threading.Event()

    class NotifyingPolicy(FakePolicy):
        def predict_action_chunk(self, batch, **kwargs):
            inference_started.set()
            return super().predict_action_chunk(batch, **kwargs)

    runtime, _ = _rtc(NotifyingPolicy(), time.perf_counter)
    runtime.publish({"observation.state": torch.zeros(1, 10)})
    runtime.maybe_start()
    assert inference_started.wait(timeout=2)
    runtime.stop()
    worker = runtime._worker
    assert worker is not None and worker.is_alive()

    inference_started.clear()
    runtime.publish({"observation.state": torch.zeros(1, 10)})
    runtime.maybe_start()
    assert inference_started.wait(timeout=2)
    runtime.stop()
    assert runtime._worker is worker
    assert worker.is_alive()

    runtime.close()
    assert not worker.is_alive()
