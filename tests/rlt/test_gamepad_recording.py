from dataclasses import replace
import json
from pathlib import Path
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from vla.datacollection.start_teleop import gamepad_record as entry
from vla.datacollection.start_teleop.gamepad_recording import (
    CAMERAS, CriticalPhase, EpisodeWorker, IntervalAssembler, RecordingConfig, RecordingStore,
    command_from_debug, dataset_contract, load_recording_config,
)
from vla.robots.franka_robotiq.robot import FrankaRolloutSnapshot
from vla.cli.build_transition_cache_v2 import load_episode_frame_annotations, _chunk_annotation
from vla.core.interfaces import TRANSITION_SOURCE_DEMO
from tests.rlt.test_gamepad_teleop import rig


def snapshot(at=1., **changes):
    pose = np.eye(4)
    pose[:3, 3] = [.4, -.2, .3]
    result = FrankaRolloutSnapshot(
        observed_at_ns=round(at * 1e9), pose=pose, joint_positions=np.zeros(7), base_wrench=np.zeros(6),
        joint_torques=np.arange(7), external_joint_torques=np.arange(7)/2, joint_velocities=np.arange(7)/100,
        gripper_width=.075, gripper_open_fraction=.075 / .085,
        images={key: np.full((224, 224, 3), 50, np.uint8) for key in ("base", "left_wrist")},
        image_timestamps={"base": at - .005, "left_wrist": at - .006},
    )
    return replace(result, **changes)


def command(at=1., linear=(.02, 0, 0), angular=(0, 0, 0), gripper=.75):
    return {"at_ns": round(at * 1e9), "linear": list(linear), "angular": list(angular), "gripper": gripper}


def buttons(rb=False, lb=False):
    result = [0] * 11
    result[4], result[5] = int(lb), int(rb)
    return result


def test_recording_config_preserves_teleop_controls():
    control, recording = load_recording_config(entry.DEFAULT_CONFIG)
    assert control.robot.cameras_enabled and not control.robot.automatic_motion_recovery
    assert recording.critical_button == 5 and control.speed_modes.toggle_button == 4
    assert control.pose_shortcuts.view_button == 6 and control.pose_shortcuts.menu_button == 7
    assert control.mapping.right_stick_rotation_deg == -45
    assert control.speed_modes.high_multiplier == 5 and control.speed_modes.transition_time_s == .3
    assert control.robot.human_linear_velocity_control and control.robot.human_angular_velocity_control


@pytest.mark.parametrize("key,value", [("critical_button", True), ("queue_size", 0), ("min_frames", 11), ("fps", 0)])
def test_recording_config_rejects_invalid_values(key, value):
    with pytest.raises(ValueError):
        RecordingConfig(**{key: value})


@pytest.mark.parametrize("critical_button", [6, 7])
def test_recording_critical_button_cannot_share_pose_shortcuts(tmp_path, critical_button):
    raw = yaml.safe_load(entry.DEFAULT_CONFIG.read_text())
    raw["recording"]["critical_button"] = critical_button
    path = tmp_path / "record.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="must not share"):
        load_recording_config(path)


def test_rb_edges_single_interval_and_reset():
    phase = CriticalPhase(5)
    assert phase.update(buttons(True)) is None  # held at start
    phase.update(buttons(False, True))
    assert "ON" in phase.update(buttons(True, True))
    assert phase.active
    assert phase.update(buttons(True)) is None
    phase.update(buttons())
    assert "OFF" in phase.update(buttons(True))
    phase.update(buttons())
    assert "re-entry rejected" in phase.update(buttons(True))
    assert not phase.active
    phase.reset()
    assert phase.state == "before" and not phase.ready
    with pytest.raises(ValueError, match="missing"):
        phase.update([0] * 5)


def test_interval_action_integrates_subsequent_commands_not_previous_or_next_state():
    assembler = IntervalAssembler(snapshot(), command(), False, 30)
    assembler.accept_command(command(1.01, linear=(.04, 0, 0)))
    assembler.accept_command(command(1.02, linear=(-.02, .01, 0), angular=(0, 0, .3)))
    assembler.accept_command(command(1.03, linear=(0, 0, 0), angular=(0, 0, 0), gripper=.6))
    next_observation = snapshot(1.04)
    next_observation.pose[:3, 3] += .01  # deliberately not the commanded pose
    pending, state, action, phase, end = assembler.observe(next_observation, True)
    np.testing.assert_allclose(action[:3], [.4004, -.1999, .3], atol=1e-7)
    np.testing.assert_allclose(state[:3], [.4, -.2, .3])
    assert action[9] == pytest.approx(.6) and phase == 0 and end == 1040000000
    assert action[4] == pytest.approx(np.sin(.003))
    assert pending.observed_at_ns == 1000000000
    # Phase is attached at interval start, never retroactively changed by RB.
    _, _, stopped, phase, _ = assembler.observe(snapshot(1.08), False)
    np.testing.assert_allclose(stopped[:3], next_observation.pose[:3, 3])
    assert phase == 1


@pytest.mark.parametrize("delta", [.01, .06, -.01])
def test_bad_recording_intervals_are_rejected(delta):
    assembler = IntervalAssembler(snapshot(), command(), False, 30)
    with pytest.raises(ValueError, match="discontinuity"):
        assembler.observe(snapshot(1 + delta), False)


def test_duplicate_camera_invalid_rotation_and_out_of_order_command():
    initial = snapshot()
    assembler = IntervalAssembler(initial, command(), False, 30)
    with pytest.raises(ValueError, match="duplicate"):
        assembler.observe(snapshot(1.04, image_timestamps=initial.image_timestamps), False)
    with pytest.raises(ValueError, match="timestamp"):
        assembler.accept_command(command(.99))
    broken = snapshot()
    broken.pose[0, 0] = 2
    with pytest.raises(ValueError, match="rotation"):
        IntervalAssembler(broken, command(), False, 30)


def test_command_validation_and_watchdog_gap():
    with pytest.raises(ValueError):
        command_from_debug({"human_command_at_ns": 1, "commanded_linear_velocity": [float("nan")] * 3,
                            "commanded_angular_velocity": [0.] * 3, "human_gripper_target": .5})
    assembler = IntervalAssembler(snapshot(), command(), False, 30)
    with pytest.raises(ValueError, match="watchdog"):
        assembler.accept_command(command(1.2))


def feed_episode(worker, count=14):
    for index in range(count):
        start = 1 + index * .04
        for offset in (.01, .02, .03):
            worker.put("command", command(start + offset))
        observed = snapshot(start + .04)
        observed.pose[0, 3] += .02 * (start + .04 - 1)
        worker.put("observation", (observed, 4 <= index + 1 < 10))
    worker.put("finish", None)
    assert worker.done.wait(20), "recording worker failed to finish"


@pytest.fixture
def recorded(tmp_path):
    control, recording = load_recording_config(entry.DEFAULT_CONFIG)
    store = RecordingStore(tmp_path / "raw", control, recording, "test task")
    try:
        worker = store.start(snapshot(), command())
        feed_episode(worker)
        assert worker.error is None
        assert worker.saved_index == 0
        yield store, worker, control, recording
    finally:
        store.close()


def test_real_video_writer_commit_and_trace(recorded):
    store, worker, _, _ = recorded
    frame = pd.read_parquet(store.writer.data_path(0))
    assert len(frame) == worker.frames == 14
    assert np.stack(frame["observation.state"]).shape == (14, 50)
    assert np.stack(frame["action"]).shape == (14, 10)
    assert worker.critical_frames == 6
    episodes = pd.read_parquet(store.root / "meta/episodes/chunk-000/file-000.parquet")
    assert episodes.iloc[0]["episode_success"] == "success"
    assert json.loads(episodes.iloc[0]["critical_intervals"]) == [{"start_frame": 4, "end_frame": 10}]
    trace = [json.loads(line) for line in (store.root / episodes.iloc[0]["command_trace"]).read_text().splitlines()]
    assert trace[0]["kind"] == "seed"
    assert sum(row["kind"] == "command" for row in trace) == 42
    assert sum(row["kind"] == "interval" for row in trace) == 14


@pytest.mark.parametrize("dimension", [10, 22])
def test_feature_extraction_official_loader_and_demo_cache(recorded, tmp_path, dimension):
    from vla.cli import extract_dataset_features as extraction
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    store, _, _, _ = recorded
    result = extraction.extract_dataset_features(
        store.root, extraction.FEATURE_CONFIG_DIR / f"franka_single_left_tcp_rot6d_{dimension}d.yaml",
        tmp_path / f"features-{dimension}", validate=True,
    )
    dataset = LeRobotDataset(repo_id="local/gamepad-test", root=result, video_backend="pyav")
    assert dataset[0]["observation.state"].shape == (dimension,)
    assert dataset[0]["observation.images.base"].shape == (3, 224, 224)
    annotations = load_episode_frame_annotations(dataset, 0, 14)
    for anchor, critical in [(0, 0), (4, 1), (10, 0)]:
        labels = _chunk_annotation(annotations, start_frame=anchor, chunk_length=2)
        assert labels["source"] == TRANSITION_SOURCE_DEMO
        assert labels["is_critical"] == critical and labels["intervention"] == 0


def test_resume_contract_and_uncommitted_quarantine(recorded):
    store, _, control, recording = recorded
    root = store.root
    store.close()
    with pytest.raises(FileExistsError):
        RecordingStore(root, control, recording, "test task")
    with pytest.raises(ValueError, match="contract"):
        RecordingStore(root, control, recording, "another task", resume=True)
    orphan = store.writer.video_path(1, CAMERAS[0])
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"uncommitted video")
    committed_before = store.writer.data_path(0).read_bytes()
    resumed = RecordingStore(root, control, recording, "test task", resume=True)
    try:
        assert resumed.writer.next_episode_index() == 1
        assert not orphan.exists()
        assert store.writer.data_path(0).read_bytes() == committed_before
        assert list(root.glob(".recording_attempts/*/recovery.json"))
    finally:
        resumed.close()


def test_store_lease_and_resume_validation_do_not_create_foreign_dataset(tmp_path):
    control, recording = load_recording_config(entry.DEFAULT_CONFIG)
    with pytest.raises(ValueError, match="existing"):
        RecordingStore(tmp_path / "missing", control, recording, "task", resume=True)
    assert not (tmp_path / "missing").exists()
    store = RecordingStore(tmp_path / "raw", control, recording, "task")
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            RecordingStore(store.root, control, recording, "task", resume=True)
    finally:
        store.close()


def test_cancelled_and_short_episodes_never_commit(tmp_path):
    control, recording = load_recording_config(entry.DEFAULT_CONFIG)
    store = RecordingStore(tmp_path / "raw", control, recording, "task")
    try:
        worker = store.start(snapshot(), command())
        diagnostics = {"recent_states": [{"success_rate": .8999}], "consecutive_low_samples": 3}
        worker.cancel_recording("low success rate", diagnostics)
        assert worker.done.wait(10)
        assert worker.saved_index is None and store.writer.episode_count == 0
        manifest = json.loads((worker.attempt / "attempt.json").read_text())
        assert manifest["status"] == "discarded"
        assert manifest["reason"] == "low success rate"
        assert manifest["controller_fault"] == diagnostics
        store.worker = None
        worker = store.start(snapshot(), command())
        feed_episode(worker, 2)
        assert isinstance(worker.error, ValueError)
        assert store.writer.episode_count == 0
    finally:
        store.close()


def test_save_failure_keeps_uncommitted_diagnostics(tmp_path, monkeypatch):
    control, recording = load_recording_config(entry.DEFAULT_CONFIG)
    store = RecordingStore(tmp_path / "raw", control, recording, "task")
    def fail(*args, **kwargs):
        raise OSError("injected disk error")
    monkeypatch.setattr(store.writer, "save_episode", fail)
    try:
        worker = store.start(snapshot(), command())
        feed_episode(worker)
        assert isinstance(worker.error, OSError)
        assert worker.saved_index is None and store.writer.episode_count == 0
        assert (worker.attempt / "error.json").is_file()
    finally:
        store.close()


def test_queue_full_invalidates_without_waiting(tmp_path, monkeypatch):
    from vla.datacollection.start_teleop import gamepad_recording as recording_module
    control, config = load_recording_config(entry.DEFAULT_CONFIG)
    store = RecordingStore(tmp_path / "raw", control, replace(config, queue_size=1), "task")
    release = threading.Event()
    original = recording_module.EpisodeWorker._run
    def blocked(worker):
        release.wait(5)
        original(worker)
    monkeypatch.setattr(recording_module.EpisodeWorker, "_run", blocked)
    try:
        worker = store.start(snapshot(), command())
        worker.put("command", command(1.01))
        with pytest.raises(RuntimeError, match="queue full"):
            worker.put("command", command(1.02))
        assert worker.cancel.is_set()
        release.set()
        assert worker.done.wait(10)
        detail = json.loads((worker.attempt / "attempt.json").read_text())
        assert "queue full" in detail["reason"]
    finally:
        release.set()
        store.close()


def test_unidentified_phase_only_rollout_still_rejected():
    class Rows:
        column_names = ["complementary_info.phase"]
    dataset = SimpleNamespace(hf_dataset=Rows(), meta=SimpleNamespace(info={}))
    with pytest.raises(ValueError, match="collector_policy_id"):
        load_episode_frame_annotations(dataset, 0, 1)


def test_check_only_never_initializes_dataset_or_connects(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "recording_preflight", lambda *args: None)
    monkeypatch.setattr(entry, "run", lambda *args: pytest.fail("execute called"))
    monkeypatch.setattr(entry, "RecordingStore", lambda *args, **kwargs: pytest.fail("dataset initialized"))
    monkeypatch.setattr("sys.argv", ["record", "--check-only", "--dataset-root", str(tmp_path / "unused")])
    entry.main()
    assert not (tmp_path / "unused").exists()


def test_execute_requires_explicit_dataset_and_task(monkeypatch):
    monkeypatch.setattr("sys.argv", ["record", "--execute"])
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2


class MemoryWorker:
    cancel_recording = EpisodeWorker.cancel_recording

    def __init__(self):
        self.index = self.frames = self.critical_frames = 0
        self.done, self.cancel = threading.Event(), threading.Event()
        self.cancel_details = None
        self.saved_index = self.error = None
        self.success = True
        self.attempt = Path("fake-attempt")
        self.events = []

    def check(self):
        if self.error:
            raise self.error

    def put(self, kind, value):
        self.check()
        self.events.append((kind, value))
        if kind == "observation":
            self.frames += 1


@pytest.fixture
def session_rig(rig, monkeypatch):
    _, recording = load_recording_config(entry.DEFAULT_CONFIG)
    store = SimpleNamespace(worker=None)
    def start(*args):
        store.worker = MemoryWorker()
        return store.worker
    store.start = start
    monkeypatch.setattr(rig.robot, "get_rollout_snapshot", lambda: snapshot(rig.clock[0]))
    monkeypatch.setattr("vla.robots.franka_robotiq.robot.time.monotonic_ns", lambda: round(rig.clock[0] * 1e9))
    session = entry.RecordingSession(rig.config, recording, rig.robot, rig.receiver, store,
                                     clock=lambda: rig.clock[0], report=rig.reports.append)
    rig.app = session
    def advance(dt=.01):
        rig.clock[0] += dt
        rig.receiver.packet = replace(rig.receiver.packet, monotonic_ns=int(rig.clock[0] * 1e9))
        rig.arm.state.time = rig.clock[0]
        session.tick()
    rig.advance = advance
    # Establish fresh state, enable, then submit a zero command before n.
    session.tick()
    rig.advance()
    session.key("s")
    rig.advance()
    session.key("n")
    assert session.worker is not None, rig.reports
    return rig, session


def test_session_100hz_commands_30fps_samples_and_combined_rb_lb(session_rig):
    rig, session = session_rig
    worker = session.worker
    rig.receiver.packet = replace(rig.receiver.packet, axes=(1, 1, 1, -1, 1, 1), buttons=tuple(buttons(True, True)))
    for _ in range(30):
        rig.advance()
    assert session.controller.mode == "active"
    assert session.controller.speed_scale == pytest.approx(5.)
    assert session.phase.active
    commands = [value for kind, value in worker.events if kind == "command"]
    observations = [value for kind, value in worker.events if kind == "observation"]
    assert len(commands) == 30 and 8 <= len(observations) <= 9
    assert np.linalg.norm(commands[-1]["linear"]) > 0
    assert np.linalg.norm(commands[-1]["angular"]) > 0
    assert all(phase for _, phase in observations)
    # Events are observed-before-command at coincident timestamps.
    times = [v.observed_at_ns if kind == "observation" else value["at_ns"]
             for kind, value in worker.events for v in ([value[0]] if kind == "observation" else [None])]
    assert times == sorted(times)


def test_session_save_stops_and_blocks_enable_without_extra_tail_frame(session_rig):
    rig, session = session_rig
    for _ in range(5):
        rig.advance()
    worker = session.worker
    observations = sum(kind == "observation" for kind, _ in worker.events)
    session.key("y")
    assert session.saving and session.controller.mode == "standby"
    assert worker.events[-1][0] == "finish"
    session.key("s")
    for _ in range(5):
        rig.advance()
    assert session.controller.mode == "standby"
    assert sum(kind == "observation" for kind, _ in worker.events) == observations
    worker.saved_index, worker.frames, worker.critical_frames = 0, 14, 5
    worker.done.set()
    rig.advance()
    assert session.worker is None and session.phase.state == "before"
    assert any("critical+Human=5 success" in message for message in rig.reports)


@pytest.mark.parametrize("key", ["d", "x"])
def test_session_pause_discard_does_not_resume_episode(session_rig, key):
    rig, session = session_rig
    worker = session.worker
    session.key(key)
    assert worker.cancel.is_set() and session.saving
    assert session.controller.mode == "standby"
    assert not any(kind == "finish" for kind, _ in worker.events)
    worker.done.set()
    rig.advance()
    session.key("s")
    assert session.worker is None


def test_session_stale_camera_and_control_failure_invalidate(session_rig, monkeypatch):
    rig, session = session_rig
    worker = session.worker
    monkeypatch.setattr(rig.robot, "get_rollout_snapshot", lambda: snapshot(9.))
    for _ in range(5):
        rig.advance()
    # An old observation must not be silently skipped forever.
    assert session.saving and worker.cancel.is_set()


def test_session_control_failure_never_records_unsent_command(session_rig):
    rig, session = session_rig
    worker = session.worker
    rig.arm.motion_error = RuntimeError("fake reflex")
    before = sum(kind == "command" for kind, _ in worker.events)
    rig.advance()
    assert worker.cancel.is_set() and session.controller.mode == "fault"
    assert sum(kind == "command" for kind, _ in worker.events) == before
    assert sum(message.startswith("FAULT:") for message in rig.reports) == 1
    assert "fake reflex" in worker.cancel_details["reason"]
    assert worker.cancel_details["controller_fault"] == session.controller.fault_snapshot


def test_session_low_rate_fault_has_single_stop_and_diagnostics(session_rig, monkeypatch):
    rig, session = session_rig
    worker = session.worker
    stops = []
    original = rig.robot.end_human_control
    def stop():
        stops.append(True)
        original()
    monkeypatch.setattr(rig.robot, "end_human_control", stop)
    for _ in range(12):
        rig.advance()
    rig.arm.state.control_command_success_rate = .8999
    for _ in range(3):
        rig.advance()
    assert session.controller.mode == "fault" and worker.cancel.is_set()
    assert len(stops) == 1
    assert sum(message.startswith("FAULT:") for message in rig.reports) == 1
    assert worker.cancel_details["controller_fault"]["consecutive_low_samples"] == 3
    assert not any(kind == "finish" for kind, _ in worker.events)


def test_session_worker_fault_latches_storage(session_rig):
    rig, session = session_rig
    worker = session.worker
    worker.error = OSError("disk full")
    worker.done.set()
    rig.advance()
    assert session.storage_fault and session.controller.mode == "fault"
    session.key("s")
    assert session.controller.mode != "active"


def test_preview_status_does_not_change_recording_flow(session_rig):
    rig, session = session_rig
    statuses = []
    session.preview = SimpleNamespace(update_status=statuses.append)
    rig.receiver.packet = replace(rig.receiver.packet, buttons=tuple(buttons(True)))
    rig.advance()
    assert statuses[-1]["mode"] == "RECORDING"
    assert statuses[-1]["critical"] == "critical"
    assert statuses[-1]["episode"] == session.worker.index
    assert session.controller.mode == "active"
    session.key("y")
    rig.advance()
    assert statuses[-1]["mode"] == "SAVING/CLEANUP"
    assert session.controller.mode == "standby"


def test_full_recorder_keeps_failure_save_distinct_from_discard(session_rig):
    rig, session = session_rig
    session.key('f')
    assert session.controller.mode == 'standby'
    assert session.worker.events[-1] == ('finish', {'success': False, 'reason': 'operator marked failure'})
    assert not session.worker.cancel.is_set()
