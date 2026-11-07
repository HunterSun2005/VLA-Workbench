from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from scipy.spatial.transform import Rotation

from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
from vla.datacollection.franka_robotiq_rollout.gamepad import (
    decode_gamepad_values,
    gripper_direction_transition,
)
from vla.datacollection.franka_robotiq_rollout.collect import configured_backend
from vla.datacollection.franka_robotiq_rollout.recording import COLLECTOR_HUMAN, COLLECTOR_PI05, COLLECTOR_RLT_ACTOR, episode_source_frame_counts, rollout_state
from vla.datacollection.franka_robotiq_rollout.joy_transport import (
    JoyPacket,
    JoyReceiver,
    decode_joy_packet,
    encode_joy_packet,
)
from vla.datacollection.franka_robotiq_rollout.policy_client import (
    AsyncPolicyClient,
    PolicyResult,
)
import vla.datacollection.franka_robotiq_rollout.policy_client as policy_client_module
from vla.datacollection.franka_robotiq_rollout.safety import (
    FrankaHealthMonitor,
    action_from_joint_command,
    clamp_gripper_action_roundoff,
    clamp_pose_step,
    compose_pose,
    flange_pose_from_tcp,
    gamepad_tracking_twist,
    policy_state_from_raw,
    pose_step_to_twist,
    smooth_pose_target,
    validate_absolute_action,
    validate_rollout_episode,
)
from vla.datacollection.franka_robotiq_single_left import schema
from vla.datacollection.franka_robotiq_single_left.v30_writer import V30DatasetWriter
from vla.cli.build_transition_cache_v2 import _chunk_annotation
from vla.cli.serve_franka_rollout import (
    FrankaRolloutPolicyServer,
    _TrainingProgressSender,
    _load_rollout_rtc_profile,
)
from vla.cli.extract_dataset_features import FEATURE_CONFIG_DIR, load_feature_config
from vla.cli.merge_transition_caches import merge_transition_caches
from vla_rollout_protocol import recv_packet, send_packet
from vla.control.geometry import ACTION_FIELDS, FrankaRolloutSnapshot


def _image_stats(count: int) -> dict[str, np.ndarray]:
    values = {
        key: np.ones((3, 1, 1), dtype=np.float32)
        for key in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
    }
    values["count"] = np.array([count], dtype=np.int64)
    return values


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


def test_rollout_protocol_round_trip() -> None:
    class MemoryConnection:
        def __init__(self) -> None:
            self.data = bytearray()
            self.position = 0

        def sendall(self, value) -> None:
            self.data.extend(value)

        def recv(self, size: int) -> bytes:
            value = bytes(self.data[self.position : self.position + size])
            self.position += len(value)
            return value

    connection = MemoryConnection()
    send_packet(
        connection,
        {"kind": "infer", "request_id": 4},
        {
            "state": np.arange(10, dtype=np.float32),
            "image": np.zeros((8, 8, 3), dtype=np.uint8),
        },
    )
    header, arrays = recv_packet(connection)
    assert header == {"kind": "infer", "request_id": 4, "protocol_version": 2}
    np.testing.assert_array_equal(arrays["state"], np.arange(10, dtype=np.float32))
    assert arrays["image"].shape == (8, 8, 3)


def test_joy_receiver_drains_backlog_and_accepts_bridge_sequence_restart() -> None:
    base = time.monotonic_ns()

    class DatagramQueue:
        def __init__(self):
            self.payloads = [
                encode_joy_packet(JoyPacket(200, base, (0.0,), (0,))),
                encode_joy_packet(JoyPacket(0, base + 1, (0.5,), (1,))),
            ]

        def recv(self, _size):
            if not self.payloads:
                raise BlockingIOError
            return self.payloads.pop(0)

    receiver = object.__new__(JoyReceiver)
    receiver._socket = DatagramQueue()
    receiver._latest = None
    receiver._closed = False

    latest = receiver.poll_latest()

    assert latest is not None
    assert latest.sequence == 0
    assert latest.axes == (0.5,)
    assert receiver._socket.payloads == []


def _mock_rollout_server() -> FrankaRolloutPolicyServer:
    server = object.__new__(FrankaRolloutPolicyServer)
    server.fingerprint = "fingerprint"
    server.task = "Place the blue block on the white paper."
    server.tasks = [server.task]
    server.robot_type = "franka_robotiq"
    server.action_dim = 10
    server.state_dim = 10
    from vla.actions import ActionSpec
    server.action_contract = {"action_spec": ActionSpec().to_dict()}
    server.camera_keys = ["observation.images.base", "observation.images.left_wrist"]
    server.camera_shapes_hwc = {key: [224, 224, 3] for key in server.camera_keys}
    server.state_names = [f"state_{index}" for index in range(10)]
    server.action_names = [f"action_{index}" for index in range(10)]
    server.fps = 30.0
    server.actions_stored_as = "absolute"
    server.use_relative_actions = False
    server.phase_mode = "always_rl"
    server.rtc_enabled = True
    server.rtc_profile_path = Path("/tmp/rtc.yaml")
    server.rtc_execution_horizon = 10
    server.rtc_queue_threshold = 30
    server.phase = "rl"
    server._infer_count = 0
    server.device = "cpu"
    server.preprocessor = object()
    server.postprocessor = object()
    server.use_amp = False
    server.policy = SimpleNamespace(
        reset=lambda: None,
        pop_step_metadata=lambda: None,
        set_rl_mode=lambda: None,
        set_vla_mode=lambda: None,
    )
    server.policy.config = SimpleNamespace(chunk_exec_steps=12, chunk_length=10)
    server.action_stride = 1
    from vla.control.plan_service import PlanGate
    server.plan_gate = PlanGate()
    return server


def test_rollout_server_rejects_legacy_infer_even_with_valid_task():
    server = _mock_rollout_server()
    with pytest.raises(ValueError, match='protocol-v2'):
        server.handle({'kind': 'infer', 'task': server.task}, {})


def test_rollout_server_inference_value_logging_is_opt_in(capsys):
    server = _mock_rollout_server()
    with pytest.raises(ValueError, match='protocol-v2'):
        server.handle({'kind': 'infer'}, {})
    assert capsys.readouterr().out == ''


def test_rollout_server_rejects_legacy_infer_with_reset():
    server = _mock_rollout_server()
    with pytest.raises(ValueError, match='protocol-v2'):
        server.handle({'kind': 'infer', 'reset_episode': True, 'phase': 'vla'}, {})


def test_rollout_server_rejects_legacy_infer_with_rl_phase():
    server = _mock_rollout_server()
    with pytest.raises(ValueError, match='protocol-v2'):
        server.handle({'kind': 'infer', 'phase': 'rl'}, {})


def test_rollout_server_processes_online_episode_and_reports_warmup_progress(
    tmp_path: Path,
) -> None:
    server = _mock_rollout_server()
    dataset_root = tmp_path / "dataset"
    replay = SimpleNamespace(warmup_env_steps=640, active_checkpoint='revision')
    config = SimpleNamespace(warmup_steps=1000, full_episode=False)
    calls = []

    progress_messages = []

    def process_saved_episode(*, episode_id, outcome, progress_callback):
        calls.append((episode_id, outcome))
        progress_callback({"completed_updates": 1, "total_updates": 2})
        return {
            "episode_id": episode_id,
            "transitions": 25,
            "warmup": True,
            "duplicate": False,
            "critical_steps": 120,
        }

    server.online_dataset_root = dataset_root.resolve()
    server.online_replay_root = (tmp_path / "replay").resolve()
    server.online_timeout_s = 60.0
    server.online_session = SimpleNamespace(
        replay=replay,
        config=config,
        process_saved_episode=process_saved_episode,
    )
    stop = dict(session_id='s', generation=1, state='standby', reset_acknowledged=True)
    server.handle({'kind':'progress', **stop}, {})
    response, arrays = server.handle(
        {
            "kind": "episode_saved",
            "execution_context": stop,
            "request_id": -2,
            "task": server.task,
            "checkpoint_fingerprint": server.fingerprint,
            "dataset_root": str(dataset_root),
            "episode_id": 3,
            "outcome": "failure",
        },
        {},
        progress_callback=progress_messages.append,
    )

    assert arrays == {}
    assert calls == [(3, "failure")]
    assert progress_messages == [{"completed_updates": 1, "total_updates": 2}]
    assert response["kind"] == "episode_processed"
    assert response["warmup_env_steps"] == 640
    assert response["warmup_remaining"] == 360
    assert server.contract["online_enabled"] is True


def test_training_progress_sender_ignores_collector_disconnect(monkeypatch) -> None:
    attempts = []

    def disconnected(*args, **kwargs):
        attempts.append((args, kwargs))
        raise BrokenPipeError("collector closed")

    monkeypatch.setattr("vla.cli.serve_franka_rollout.send_packet", disconnected)
    sender = _TrainingProgressSender(SimpleNamespace(), -2)

    sender({"completed_updates": 1, "total_updates": 10})
    sender({"completed_updates": 2, "total_updates": 10})

    assert len(attempts) == 1
    assert sender.available is False


def test_policy_client_streams_training_progress_in_collector(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    client = object.__new__(AsyncPolicyClient)
    client.timeout_s = 1.0
    client.contract = {
        "online_enabled": True,
        "online_timeout_s": 10.0,
        "online_dataset_root": str(tmp_path.resolve()),
        "task": "task",
        "checkpoint_fingerprint": "fingerprint",
    }
    client._wait_for_idle = lambda _timeout: None

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def settimeout(self, _timeout):
            pass

    client._connect = FakeConnection
    responses = iter(
        [
            ({'kind':'progress'}, {}),
            (
                {
                    "kind": "training_progress",
                    "request_id": -2,
                    "completed_updates": 0,
                    "total_updates": 10,
                    "critic_loss": None,
                    "actor_updates": 0,
                    "actor_loss": None,
                },
                {},
            ),
            (
                {
                    "kind": "training_progress",
                    "request_id": -2,
                    "completed_updates": 10,
                    "total_updates": 10,
                    "critic_loss": 0.25,
                    "actor_updates": 5,
                    "actor_loss": -0.5,
                },
                {},
            ),
            (
                {
                    "kind": "episode_processed",
                    "request_id": -2,
                    "transitions": 2,
                },
                {},
            ),
        ]
    )
    monkeypatch.setattr(
        policy_client_module, "send_packet", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        policy_client_module, "recv_packet", lambda _connection: next(responses)
    )

    response = client.process_saved_episode(
        dataset_root=tmp_path,
        execution_context=dict(session_id='s', generation=1, state='standby', reset_acknowledged=True),
        episode_id=3,
        outcome="success",
    )

    output = capsys.readouterr().out
    assert response["kind"] == "episode_processed"
    assert output.count("\n") == 1
    assert output.count("\r") == 2
    assert "10/10 (100.0%)" in output
    assert "actor_updates=5" in output


def test_rollout_server_uses_candidate_plan_adapter() -> None:
    source = (
        Path(__file__).parents[2] / "src/vla/cli/serve_franka_rollout.py"
    ).read_text()
    assert "from vla.adapters.lerobot.plan_inference import infer_plan" in source
    assert "from lerobot.policies.utils import predict_action" not in source
    assert 'parser.add_argument(\n        "--tokenizer-path"' in source
    assert "config.tokenizer_path = args.tokenizer_path" in source


def test_rollout_server_loads_the_ros_rtc_profile() -> None:
    profile = (
        Path(__file__).parents[2]
        / "src/vla/configs/deployment/franka_single_left_pi05_rtc.yaml"
    )
    path, rtc_config, fps, queue_threshold, task = _load_rollout_rtc_profile(profile)

    assert path == profile.resolve()
    assert rtc_config.enabled is True
    assert rtc_config.execution_horizon == 10
    assert rtc_config.max_guidance_weight == pytest.approx(5.0)
    assert rtc_config.prefix_attention_schedule.value == "EXP"
    assert fps == pytest.approx(30.0)
    assert queue_threshold == 28
    assert task == "Place the white lid firmly onto the jig."


def test_rollout_action_validation_and_pose_clamp() -> None:
    action = np.array([0.4, 0.0, 0.3, 1, 0, 0, 0, 1, 0, 0.5], dtype=np.float32)
    decoded = validate_absolute_action(
        action,
        arm_count=1,
        workspace_bounds=[(np.array([0.2, -0.4, 0.1]), np.array([0.8, 0.4, 0.8]))],
    )
    np.testing.assert_allclose(decoded[0][1], np.eye(3), atol=1e-7)
    position, rotation = clamp_pose_step(
        np.zeros(3),
        np.eye(3),
        np.array([1.0, 0.0, 0.0]),
        np.diag([-1.0, -1.0, 1.0]),
        max_linear_step=0.1,
        max_angular_step=0.2,
    )
    assert np.linalg.norm(position) == pytest.approx(0.1)
    assert np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec()) == pytest.approx(
        0.2
    )


@pytest.mark.parametrize(
    "gripper, expected",
    [(1.0013828, 1.0), (1.0233029, 1.0), (-0.001, 0.0)],
)
def test_rollout_clamps_small_policy_gripper_overshoot(
    gripper: float, expected: float
) -> None:
    action = np.array([0.4, 0.0, 0.3, 1, 0, 0, 0, 1, 0, gripper], dtype=np.float32)
    clamped = clamp_gripper_action_roundoff(action, arm_count=1)
    assert clamped[9] == pytest.approx(expected)
    assert action[9] == pytest.approx(gripper)


def test_rollout_rejects_large_policy_gripper_overshoot() -> None:
    action = np.array([0.4, 0.0, 0.3, 1, 0, 0, 0, 1, 0, 1.06], dtype=np.float32)
    with pytest.raises(ValueError, match="beyond tolerance"):
        clamp_gripper_action_roundoff(action, arm_count=1)


def test_rollout_tcp_flange_transform_round_trip() -> None:
    flange_position = np.array([0.4, -0.1, 0.3])
    flange_rotation = Rotation.from_euler("xyz", [0.2, -0.3, 0.4]).as_matrix()
    tool_position = np.array([0.01, -0.02, 0.174])
    tool_rotation = Rotation.from_euler("z", 0.5).as_matrix()
    tcp_position, tcp_rotation = compose_pose(
        flange_position,
        flange_rotation,
        tool_position,
        tool_rotation,
    )
    recovered_position, recovered_rotation = flange_pose_from_tcp(
        tcp_position,
        tcp_rotation,
        tool_position,
        tool_rotation,
    )
    np.testing.assert_allclose(recovered_position, flange_position, atol=1e-9)
    np.testing.assert_allclose(recovered_rotation, flange_rotation, atol=1e-9)


def test_rollout_rtc_pose_tracking_uses_target_window_and_smoothing() -> None:
    current_position = np.zeros(3)
    current_rotation = np.eye(3)
    target_rotation = Rotation.from_rotvec([0.0, 0.0, 1.0]).as_matrix()
    limited_position, limited_rotation = clamp_pose_step(
        current_position,
        current_rotation,
        np.array([1.0, 0.0, 0.0]),
        target_rotation,
        max_linear_step=0.05 * 0.15,
        max_angular_step=0.20 * 0.15,
    )
    smoothed_position, smoothed_rotation = smooth_pose_target(
        current_position,
        current_rotation,
        limited_position,
        limited_rotation,
        alpha=0.25,
    )
    linear, angular = pose_step_to_twist(
        current_position,
        current_rotation,
        smoothed_position,
        smoothed_rotation,
        tracking_tau_s=0.15,
    )

    np.testing.assert_allclose(linear, [0.0375, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(angular, [0.0, 0.0, 0.15], atol=1e-9)


def test_rollout_gamepad_tracking_matches_fine_teleop_gains_and_feedforward() -> None:
    angle = 0.2
    linear, angular = gamepad_tracking_twist(
        np.zeros(3),
        np.eye(3),
        np.array([0.001, 0.0, 0.0]),
        Rotation.from_rotvec([0.0, 0.0, angle]).as_matrix(),
        np.array([0.02, 0.0, 0.0]),
        position_gain=8.0,
        orientation_gain=3.0,
        max_linear_speed=0.04,
        max_angular_speed=0.30,
    )

    np.testing.assert_allclose(linear, [0.028, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(angular, [0.0, 0.0, 0.30], atol=1e-9)


def _robot_health_state(*, rate: float = 1.0, error: bool = False, mode: int = 2):
    return SimpleNamespace(
        current_errors=SimpleNamespace(
            communication_constraints_violation=error,
            cartesian_reflex=False,
        ),
        robot_mode=mode,
        ROBOT_MODE_IDLE=1,
        ROBOT_MODE_MOVE=2,
        control_command_success_rate=rate,
    )


def test_franka_health_monitor_latches_errors_and_sustained_low_success_rate() -> None:
    monitor = FrankaHealthMonitor(0.90, 3)
    monitor.check("left", _robot_health_state(rate=0.89))
    monitor.check("left", _robot_health_state(rate=0.89))
    with pytest.raises(RuntimeError, match="below 0.900 for 3 samples"):
        monitor.check("left", _robot_health_state(rate=0.89))

    with pytest.raises(RuntimeError, match="communication_constraints_violation"):
        monitor.check("right", _robot_health_state(error=True))
    with pytest.raises(RuntimeError, match="not IDLE/MOVE"):
        monitor.check("right", _robot_health_state(mode=4))


@pytest.mark.parametrize("rate", [0.9, np.nextafter(0.9, -np.inf), float(np.float32(0.9))])
def test_franka_health_monitor_accepts_threshold_representations_and_resets(rate) -> None:
    monitor = FrankaHealthMonitor(0.90, 3)
    for _ in range(4):
        monitor.check("left", _robot_health_state(rate=0.89))
        monitor.check("left", _robot_health_state(rate=0.89))
        monitor.check("left", _robot_health_state(rate=rate))


@pytest.mark.parametrize(
    "rate", [0.8999, 0.89, float(np.nextafter(np.float32(0.9), np.float32(0)))],
)
def test_franka_health_monitor_rejects_real_low_rates_with_precise_diagnostics(rate) -> None:
    monitor = FrankaHealthMonitor(0.90, 3)
    for _ in range(2):
        monitor.check("left", _robot_health_state(rate=rate))
        monitor.check("right", _robot_health_state(rate=1.0))
    with pytest.raises(RuntimeError) as caught:
        monitor.check("left", _robot_health_state(rate=rate))
    message = str(caught.value)
    assert f"control_command_success_rate={rate:.17g}" in message
    assert "below 0.900 for 3 samples" in message
    assert "comparison boundary=" in message


def test_rollout_episode_validation_accepts_stationary_failure_but_rejects_bad_gripper() -> (
    None
):
    states = np.zeros((12, len(schema.STATE_FIELDS)), dtype=np.float32)
    states[:, 3] = 1.0
    states[:, 7] = 1.0
    states[:, 9] = 0.085
    actions = np.zeros((12, 10), dtype=np.float32)
    actions[:, :3] = [0.4, 0.0, 0.3]
    actions[:, 3] = 1.0
    actions[:, 7] = 1.0
    validate_rollout_episode(
        states,
        actions,
        arm_count=1,
        workspace_bounds=[(np.array([0.2, -0.4, 0.1]), np.array([0.8, 0.4, 0.8]))],
        maximum_gripper_width=0.085,
    )
    states[0, 9] = 1.0
    with pytest.raises(ValueError, match="measured gripper width"):
        validate_rollout_episode(
            states,
            actions,
            arm_count=1,
            workspace_bounds=[(np.array([0.2, -0.4, 0.1]), np.array([0.8, 0.4, 0.8]))],
            maximum_gripper_width=0.085,
        )


def test_rollout_yaml_profiles_are_single_and_dual() -> None:
    root = Path(__file__).parents[2] / "src/vla/datacollection/franka_robotiq_rollout/configs"
    single = load_rollout_config(root / "single_left.yaml")
    dual = load_rollout_config(root / "dual.yaml")
    assert [arm.name for arm in single.arms] == ["left"]
    assert [arm.gamepad.joy_topic for arm in single.arms] == ["/left/gamepad/joy"]
    assert single.policy_state_dim == single.action_dim == 10
    assert single.backend == "ros2"
    assert dual.backend == "ros2"
    assert single.task == "Place the white lid firmly onto the jig."
    assert single.image_size == (224, 224)
    assert single.cameras[1].serial == "346222073620"
    assert single.min_control_command_success_rate == pytest.approx(0.90)
    assert single.low_success_rate_samples == 3
    assert single.rtc_deployment_config is None
    assert single.max_linear_velocity == pytest.approx(0.04)
    assert single.max_angular_velocity == pytest.approx(0.30)
    assert single.velocity_tracking_tau_s == pytest.approx(0.15)
    assert single.target_smoothing_alpha == pytest.approx(0.25)
    assert single.velocity_smoothing == pytest.approx(0.30)
    assert single.smooth_angular_velocity is True
    assert single.policy_timeout_s == pytest.approx(30.0)
    assert single.control_discontinuity_timeout_s == pytest.approx(0.1)
    assert single.joy_socket_path == Path("/tmp/vla_rollout_joy.sock")
    assert single.gamepad_control_rate == pytest.approx(100.0)
    assert single.gamepad_joy_timeout_s == pytest.approx(0.20)
    assert single.gamepad_state_timeout_s == pytest.approx(0.50)
    assert single.gamepad_position_gain == pytest.approx(8.0)
    assert single.gamepad_orientation_gain == pytest.approx(3.0)
    assert single.gamepad_max_angular_speed == pytest.approx(0.30)
    assert single.gamepad_gripper_trigger_deadzone == pytest.approx(0.05)
    assert single.policy_gripper_speed == pytest.approx(1.0)
    assert single.arms[0].gripper_action_topic == (
        "/left/gripper/robotiq_gripper_controller/gripper_cmd"
    )
    assert single.arms[0].gripper_speed_topic == (
        "/left/gripper/left_gripper_speed_controller/commands"
    )
    assert single.arms[0].gripper_open_position == pytest.approx(0.0)
    assert single.arms[0].gripper_closed_position == pytest.approx(0.8)
    assert single.arms[0].gripper_max_effort == pytest.approx(50.0)
    assert single.arms[0].workspace_min == pytest.approx((0.20, -0.45, 0.05))
    assert single.arms[0].workspace_max == pytest.approx((0.75, 0.45, 0.75))
    assert [arm.name for arm in dual.arms] == ["left", "right"]
    assert [arm.gripper_joint_name for arm in dual.arms] == [
        "robotiq_85_left_knuckle_joint",
        "robotiq_85_left_knuckle_joint",
    ]
    assert [arm.gamepad.joy_topic for arm in dual.arms] == [
        "/left/gamepad/joy",
        "/right/gamepad/joy",
    ]
    assert dual.policy_state_dim == dual.action_dim == 20
    assert [camera.feature_key for camera in dual.cameras] == [
        "observation.images.base",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]
    assert dual.image_size == (224, 224)
    assert [camera.serial for camera in dual.cameras[1:]] == [
        "346222073620",
        "346122070471",
    ]
    feature_config = load_feature_config(
        FEATURE_CONFIG_DIR / "franka_dual_tcp_rot6d_20d.yaml"
    )
    assert len(feature_config.source_names) == 100
    assert len(feature_config.selected_names) == 20
    assert feature_config.selected_names[0] == "left_tcp_position_x"
    assert feature_config.selected_names[10] == "right_tcp_position_x"
    assert feature_config.expected_action_dim == 20


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda raw: raw["dataset"].update(fps=20), "RTC deployment robot config"),
        (
            lambda raw: raw["arms"][0].update(workspace_min=[0.0, -0.45, 0.05]),
            "RTC deployment robot config",
        ),
    ],
)
def test_rollout_rejects_rtc_profile_contract_mismatches(
    tmp_path, mutation, error
) -> None:
    repository = Path(__file__).parents[2]
    source = (
        repository / "src/vla/datacollection/franka_robotiq_rollout/configs/single_left.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["control"]["rtc_deployment_config"] = str(
        repository / "src/vla/configs/deployment/franka_single_left_pi05_rtc.yaml"
    )
    mutation(raw)
    config_path = tmp_path / "rollout.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        load_rollout_config(config_path)


def test_rollout_gamepad_mapping_is_neutral_and_continuous() -> None:
    root = Path(__file__).parents[2] / "src/vla/datacollection/franka_robotiq_rollout/configs"
    config = load_rollout_config(root / "single_left.yaml")
    mapping = config.arms[0].gamepad
    neutral_axes = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    neutral_buttons = [0] * 11
    velocity, x_button, b_button, gripper_motion = decode_gamepad_values(
        neutral_axes, neutral_buttons, mapping, config
    )
    np.testing.assert_allclose(velocity, 0.0)
    assert (x_button, b_button, gripper_motion) == (0.0, 0.0, 0.0)

    command_axes = [0.0, 1.0, -1.0, 0.0, 0.0, 1.0]
    command_buttons = [0] * 11
    command_buttons[mapping.y_button] = 1
    velocity, _, _, gripper_motion = decode_gamepad_values(
        command_axes, command_buttons, mapping, config
    )
    np.testing.assert_allclose(
        velocity, [config.gamepad_xy_speed, 0.0, config.gamepad_z_speed]
    )
    assert gripper_motion == 1.0

    trigger_axes = [0.0, 0.0, 0.8, 0.0, 0.0, 1.0]
    _, _, _, gripper_motion = decode_gamepad_values(
        trigger_axes, neutral_buttons, mapping, config
    )
    assert gripper_motion == pytest.approx((0.1 - 0.05) / 0.95)


def test_rollout_gamepad_gripper_uses_direction_goals_and_measured_stop() -> None:
    direction, goal = gripper_direction_transition(0, -0.5, 0.8)
    assert direction == -1
    assert goal == pytest.approx(0.0)

    direction, goal = gripper_direction_transition(direction, -1.0, 0.4)
    assert direction == -1
    assert goal is None

    direction, goal = gripper_direction_transition(direction, 0.0, 0.37)
    assert direction == 0
    assert goal == pytest.approx(0.37)

    direction, goal = gripper_direction_transition(direction, 0.25, 0.37)
    assert direction == 1
    assert goal == pytest.approx(1.0)


def test_rollout_collector_does_not_import_fine_teleop_node() -> None:
    dispatcher = (
        Path(__file__).parents[2] / "src/vla/datacollection/franka_robotiq_rollout/collect.py"
    ).read_text(encoding="utf-8")
    ros_collector = (
        Path(__file__).parents[2]
        / "src/vla/datacollection/franka_robotiq_rollout/collect_ros.py"
    ).read_text(encoding="utf-8")
    assert "import rclpy" not in dispatcher
    assert "fine_teleop import" not in ros_collector
    assert "self.coordinator.accept" in ros_collector
    assert "self.policy_client.invalidate()" in ros_collector
    assert not (Path(__file__).parents[2]/"src/vla/datacollection/franka_robotiq_rollout/collect_franky.py").exists()


def test_rollout_dispatcher_selects_backends() -> None:
    root = Path(__file__).parents[2] / "src/vla/datacollection/franka_robotiq_rollout/configs"
    assert configured_backend(root / "single_left.yaml") == "ros2"
    assert configured_backend(root / "dual.yaml") == "ros2"


def test_joy_bridge_packet_round_trip_and_validation() -> None:
    packet = JoyPacket(
        sequence=17,
        monotonic_ns=123456,
        axes=(0.25, -1.0, 1.0),
        buttons=(0, 1, 0),
    )
    assert decode_joy_packet(encode_joy_packet(packet)) == packet
    with pytest.raises(ValueError, match="binary"):
        decode_joy_packet(encode_joy_packet(JoyPacket(17, 123456, (0.0,), (2,))))


def test_franky_rollout_state_preserves_29d_prefix_and_appends_joint_dynamics() -> None:
    pose = np.eye(4)
    pose[:3, 3] = [0.4, -0.2, 0.3]
    snapshot = SimpleNamespace(
        pose=pose,
        observed_at_ns=2_000_000_000,
        gripper_width=0.05,
        base_wrench=np.arange(6, dtype=float),
        joint_positions=np.arange(7, dtype=float),
        joint_torques=np.arange(7)+10, external_joint_torques=np.arange(7)+20, joint_velocities=np.arange(7)+30,
    )
    previous = (
        np.array([0.39, -0.2, 0.3]),
        np.eye(3),
        1_000_000_000,
    )
    state = rollout_state(snapshot, previous)
    assert state.shape == (len(schema.STATE_FIELDS),)
    np.testing.assert_allclose(state[:10], [0.4, -0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.05])
    np.testing.assert_allclose(state[10:16], np.arange(6))
    np.testing.assert_allclose(state[16:22], [0.01, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(state[22:29], np.arange(7))
    np.testing.assert_allclose(state[29:36], np.arange(7)+10)
    np.testing.assert_allclose(state[36:43], np.arange(7)+20)
    np.testing.assert_allclose(state[43:50], np.arange(7)+30)




def test_episode_source_frame_counts_are_critical_and_mutually_exclusive() -> None:
    counts = episode_source_frame_counts(
        [0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        [
            COLLECTOR_HUMAN,
            COLLECTOR_PI05,
            COLLECTOR_RLT_ACTOR,
            COLLECTOR_HUMAN,
            COLLECTOR_RLT_ACTOR,
            COLLECTOR_HUMAN,
        ],
    )

    assert counts == {
        "critical_actor": 1,
        "critical_human": 2,
        "critical_vla": 1,
    }












def test_rollout_policy_state_extracts_each_arms_first_10_of_50_dimensions() -> None:
    raw_state = np.arange(2 * len(schema.STATE_FIELDS), dtype=np.float32)

    policy_state = policy_state_from_raw(raw_state, 2)

    np.testing.assert_array_equal(
        policy_state,
        np.concatenate((raw_state[:10], raw_state[len(schema.STATE_FIELDS) :][:10])),
    )


def test_rollout_actual_action_uses_fk_of_published_joint_snapshot() -> None:
    tool_pose = _pose([0.1, 0.0, 0.0])
    robot_state = SimpleNamespace(f_t_ee=tool_pose)

    def forward(joints):
        return np.asarray(joints[:3], dtype=float), np.eye(3)

    first = action_from_joint_command(
        np.arange(7, dtype=float),
        0.2,
        robot_state,
        forward,
    )
    second = action_from_joint_command(
        np.arange(7, dtype=float) + 1.0,
        0.2,
        robot_state,
        forward,
    )

    np.testing.assert_allclose(first[:3], [0.1, 1.0, 2.0])
    np.testing.assert_allclose(second[:3] - first[:3], [1.0, 1.0, 1.0])
    assert first[9] == pytest.approx(0.2)


def test_v30_writer_persists_rollout_annotations(tmp_path: Path) -> None:
    camera = "observation.images.base"
    features = {
        "complementary_info.is_intervention": {
            "dtype": "float32",
            "shape": [1],
            "names": ["is_intervention"],
        },
        "complementary_info.phase": {
            "dtype": "float32",
            "shape": [1],
            "names": ["phase"],
        },
        "complementary_info.collector_policy_id": {
            "dtype": "int64",
            "shape": [1],
            "names": ["collector_policy_id"],
        },
    }
    writer = V30DatasetWriter(
        tmp_path,
        30,
        "rollout task",
        (camera,),
        16,
        16,
        extra_features=features,
    )
    video = writer.video_path(0, camera)
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"placeholder")
    length = 12
    writer.save_episode(
        np.zeros((length, len(schema.STATE_FIELDS)), dtype=np.float32),
        np.zeros((length, 10), dtype=np.float32),
        {camera: _image_stats(length)},
        frame_features={
            "complementary_info.is_intervention": np.zeros(
                (length, 1), dtype=np.float32
            ),
            "complementary_info.phase": np.ones((length, 1), dtype=np.float32),
            "complementary_info.collector_policy_id": np.full(
                (length, 1), 2, dtype=np.int64
            ),
        },
        extra_episode_metadata={
            "episode_success": "failure",
            "rl_intervals": [
                {"start_frame": 0, "end_frame": length, "outcome": "failure"}
            ],
            "human_intervention_intervals": [],
        },
    )
    data = pd.read_parquet(writer.data_path(0))
    assert set(features).issubset(data.columns)
    np.testing.assert_array_equal(
        data["complementary_info.collector_policy_id"].to_numpy(),
        np.full(length, 2),
    )
    assert str(data["complementary_info.phase"].dtype) == "float32"
    assert str(data["complementary_info.collector_policy_id"].dtype) == "int64"
    episodes = pd.read_parquet(tmp_path / "meta/episodes/chunk-000/file-000.parquet")
    assert episodes.loc[0, "episode_success"] == "failure"
    assert episodes.loc[0, "rl_intervals"][0]["outcome"] == "failure"


def test_v30_writer_strict_metadata_rejects_task_and_checkpoint_mixing(
    tmp_path: Path,
) -> None:
    camera = "observation.images.base"
    metadata = {"policy_checkpoint_fingerprint": "checkpoint-a"}
    V30DatasetWriter(
        tmp_path,
        30,
        "blue task",
        (camera,),
        224,
        224,
        info_metadata=metadata,
        strict_metadata=True,
    )
    with pytest.raises(RuntimeError, match="metadata.*does not match"):
        V30DatasetWriter(
            tmp_path,
            30,
            "blue task",
            (camera,),
            224,
            224,
            info_metadata={"policy_checkpoint_fingerprint": "checkpoint-b"},
            strict_metadata=True,
        )
    with pytest.raises(RuntimeError, match="task does not match"):
        V30DatasetWriter(
            tmp_path,
            30,
            "yellow task",
            (camera,),
            224,
            224,
            info_metadata=metadata,
            strict_metadata=True,
        )


def test_chunk_annotations_keep_reference_semantics_and_vote_sources() -> None:
    annotations = {
        index: {
            "collector_policy_id": 0 if index < 5 else 2,
            "is_intervention": float(index < 5),
            "phase": 1.0,
        }
        for index in range(10)
    }
    human = _chunk_annotation(annotations, start_frame=0, chunk_length=10)
    assert human == {"intervention": 1.0, "source": 3, "is_critical": 1.0}
    for item in annotations.values():
        item["collector_policy_id"] = 2
        item["is_intervention"] = 0.0
    rl = _chunk_annotation(annotations, start_frame=0, chunk_length=10)
    assert rl == {"intervention": 0.0, "source": 2, "is_critical": 1.0}


def _write_cache(root: Path, *, value: float, name: str) -> None:
    shard_path = root / "shards/train/episode_000000.pt"
    shard_path.parent.mkdir(parents=True)
    transition = {
        "state_vec": torch.full((3,), value),
        "exec_chunk": torch.full((2, 1), value),
        "ref_chunk": torch.full((2, 1), value),
        "reward_seq": torch.tensor([0.0, 1.0]),
        "next_state_vec": torch.full((3,), value),
        "next_ref_chunk": torch.full((2, 1), value),
        "done": torch.tensor(1.0),
        "intervention": torch.tensor(0.0),
        "actual_steps": torch.tensor(2),
        "source": torch.tensor(0),
        "episode_id": torch.tensor(0),
        "is_critical": torch.tensor(1.0),
    }
    torch.save([transition], shard_path)
    from vla.adapters.lerobot.checkpoint_contract import sha256_file

    manifest = {
        "format_version": 1,
        "status": "complete",
        "contract": {
            "dataset": {
                "repo_id": name,
                "root": str(root),
                "info_sha256": name,
                "fps": 30.0,
                "state_dim": 1,
                "action_dim": 1,
                "state_names": ["state"],
                "action_names": ["action"],
                "camera_keys": ["base"],
                "actions_stored_as": "absolute",
                "action_stats": "absolute",
            },
            "policy": {"action_dim": 1, "proprio_dim": 1, "identity": "same"},
            "cache": {
                "chunk_length": 2,
                "frame_stride": 1,
                "episode_splits": {"train": [0]},
            },
        },
        "completed_episodes": {"train": [0], "val": []},
        "skipped_episodes": {"train": [], "val": []},
        "shards": {
            "train": [
                {
                    "episode_id": 0,
                    "path": "shards/train/episode_000000.pt",
                    "transitions": 1,
                    "sha256": sha256_file(shard_path),
                    "episode_success": True,
                }
            ],
            "val": [],
        },
        "total_transitions": {"train": 1, "val": 0},
    }
    (root / "cache_manifest.json").write_text(json.dumps(manifest))


def test_merge_transition_caches_keeps_source_shards_immutable(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_cache(first, value=1.0, name="first")
    _write_cache(second, value=2.0, name="second")
    output = merge_transition_caches(
        [("demo", first), ("rollout", second)], tmp_path / "merged"
    )
    manifest = json.loads((output / "cache_manifest.json").read_text())
    assert manifest["total_transitions"] == {"train": 2, "val": 0}
    assert [item["source_cache"] for item in manifest["shards"]["train"]] == [
        "demo",
        "rollout",
    ]
    assert (
        torch.load(first / "shards/train/episode_000000.pt", weights_only=False)[0][
            "state_vec"
        ][0].item()
        == 1.0
    )
