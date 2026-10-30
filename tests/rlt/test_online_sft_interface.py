from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vla.datacollection.franka_robotiq_rollout.collect_franky import (
    CommandSnapshot, EXTRA_FEATURES, FrankyRolloutCollector, _confirm_execute,
    dispatch_event, next_control_deadline,
)
from vla.datacollection.franka_robotiq_rollout.common import keyboard_event
from vla.core.online_contract import TIMING_FEATURES


@pytest.mark.parametrize("key,event", [
    (b"s", "enable"), (b"n", "right"), (b"y", "success"), (b"f", "failure"),
    (b"x", "left"), (b"d", "pause"), (b"r", "recover"), (b"c", "quit"),
    (b"h", "home"),
    (b" ", "intervention"), (b"q", "quit"), (b"\x03", "quit"),
])
def test_sft_keyboard_mapping(key, event):
    assert keyboard_event(key, sft=True) == event
    assert keyboard_event(key.upper(), sft=True) == event


def test_legacy_keyboard_unchanged():
    assert keyboard_event(b"s") == "success"
    assert keyboard_event(b"r") == "phase"
    assert keyboard_event(b"c") == "recover"
    assert keyboard_event(b"n") is None


def test_sft_standby_needs_execute_and_terminal_but_no_arm(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("must not prompt for ARM"))
    _confirm_execute(True, sft=True)
    with pytest.raises(PermissionError, match="execute"):
        _confirm_execute(False, sft=True)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(PermissionError, match="interactive"):
        _confirm_execute(True, sft=True)


def test_dispatch_s_does_not_save_and_quit_does_not_start():
    collector = Mock()
    assert not dispatch_event(collector, keyboard_event(b"s", sft=True))
    collector.enable_teleop.assert_called_once()
    collector.finish_episode.assert_not_called()
    assert dispatch_event(collector, keyboard_event(b"c", sft=True))
    collector.start_episode.assert_not_called()
    dispatch_event(collector, keyboard_event(b"y", sft=True))
    collector.finish_episode.assert_called_once_with("success")


@pytest.mark.parametrize("duration", [.001, .0108, .034, .14])
def test_loop_never_catches_up_after_slow_tick(duration):
    period = 1 / 30
    deadline = next_control_deadline(1, 1 + duration, period)
    assert deadline - 1 >= period - 1e-12
    if duration >= period:
        assert deadline - (1 + duration) == pytest.approx(period)


def collector():
    c = object.__new__(FrankyRolloutCollector)
    c.require_online = c._sft_interface = c._full_episode = True
    c._episode_active = c._faulted = c._intervention = c._priming = False
    c._closed = False
    c._fault_reason = None
    c._pending_event = c._pending_frame = None
    c._segment_id = c._command_sequence = 0
    c._phase = "rl"
    c._states, c._actions = [], []
    c._annotations = {k: [] for k in {**EXTRA_FEATURES, **TIMING_FEATURES}}
    c._frame_features = {**EXTRA_FEATURES, **TIMING_FEATURES}
    c._video_writers = {}
    c._emit = Mock()
    c._preflight = Mock()
    c.robot = Mock()
    c.policy_client = Mock()
    c.policy_client.begin_episode.return_value = {"online_warmup": False}
    c.writer = Mock()
    c.writer.next_episode_index.return_value = 0
    c.config = SimpleNamespace(fps=30, cameras=[])
    c._skip_reasons = Counter()
    c._gamepad_controls = Mock()
    c._gamepad_controls._initial_position = np.array([.4, 0, .3])
    c._gamepad_controls._initial_upright_rotation = np.eye(3)
    c.home_config = SimpleNamespace(
        target_action=(.3, -.35, .36, 1, 0, 0, 0, 1, 0, 1),
        clearance_z=.45,
        position_tolerance_m=.005,
    )
    setup = Mock(mode="standby", reason="")
    setup._neutral.return_value = True
    setup.enable.side_effect = lambda: setattr(setup, "mode", "active")
    setup.pause.side_effect = lambda *a, **kw: setattr(setup, "mode", "fault" if kw.get("fault") else "standby")
    c._setup_teleop = setup
    return c


def test_s_enables_unrecorded_teleop_n_requires_enabled_and_neutral():
    c = collector()
    c.start_episode()
    c.policy_client.begin_episode.assert_not_called()
    c.enable_teleop()
    assert c._setup_teleop.mode == "active" and not c._episode_active
    c._setup_teleop._neutral.return_value = False
    c.start_episode()
    c.policy_client.begin_episode.assert_not_called()
    c._setup_teleop._neutral.return_value = True
    c.start_episode()
    assert c._episode_active and c._priming and c._phase == "rl"
    assert c._setup_teleop.mode == "standby"
    c.policy_client.begin_episode.assert_called_once()
    c._setup_teleop.enable.reset_mock()
    c.enable_teleop()
    c._setup_teleop.enable.assert_not_called()


def test_home_returns_before_rlt_episode_once_teleop_is_enabled():
    c = collector()
    c._setup_teleop.mode = "active"
    c.return_home()
    c._setup_teleop.request_dataset_home_return.assert_called_once()
    position, rotation, clearance, tolerance = c._setup_teleop.request_dataset_home_return.call_args.args
    np.testing.assert_allclose(position, [.3, -.35, .36])
    np.testing.assert_allclose(rotation[:, 2], [0, 0, -1])
    np.testing.assert_allclose(rotation[:, 0], [1, 0, 0])
    assert clearance == .45 and tolerance == .005


def test_home_enables_teleop_from_standby_before_return():
    c = collector()
    c.return_home()
    assert c._setup_teleop.mode == "active"
    c._setup_teleop.request_dataset_home_return.assert_called_once()


def test_home_waits_for_online_boundary_before_human_return():
    c = collector()
    c._episode_active = True
    c._at_boundary = False
    c.return_home()
    assert c._pending_event == ("home", None)
    c._gamepad_controls.request_dataset_home_return.assert_not_called()
    c._pending_event = None
    c._at_boundary = True
    c._intervention = True
    c.return_home()
    c._gamepad_controls.request_dataset_home_return.assert_called_once()


def test_begin_reconciliation_runs_only_after_unrecorded_motion_stops():
    c = collector()
    c.enable_teleop()
    def begin():
        assert c._setup_teleop.mode == "standby"
        return {}
    c.policy_client.begin_episode.side_effect = begin
    c.start_episode()
    assert c._episode_active


def test_unrecorded_gamepad_uses_sft_controller_without_recording():
    c = collector()
    c.enable_teleop()
    c.gamepad_tick()
    c._setup_teleop.tick.assert_called_once()
    assert not c._states
    c.robot.send_human_target.assert_not_called()
    c.policy_client.submit.assert_not_called()
    c.pause_control()
    assert c._setup_teleop.mode == "standby"


def test_discard_returns_to_standby_and_n_cannot_restart_without_s():
    c = collector()
    c.enable_teleop()
    c.start_episode()
    c.pause_control()
    assert not c._episode_active
    assert c._setup_teleop.mode == "standby"
    c.writer.save_episode.assert_not_called()
    c.start_episode()
    assert not c._episode_active


def test_recovery_failure_never_reenables_setup_motion():
    c = collector()
    c.enable_teleop()
    c.policy_client.recover_online.side_effect = RuntimeError("training recovery failed")
    assert not c.recover_fault()
    assert c._faulted and c._setup_teleop.mode != "active"
    c._setup_teleop.tick.reset_mock()
    c.gamepad_tick()
    c._setup_teleop.tick.assert_not_called()


def test_outcome_without_episode_stops_teleop_without_saving():
    c = collector()
    c.enable_teleop()
    c.finish_episode("success")
    assert c._setup_teleop.mode == "standby"
    c.writer.save_episode.assert_not_called()


def test_intervention_does_not_replace_pending_save():
    c = collector()
    c._episode_active = True
    c._at_boundary = False
    c._pending_event = ("finish", "success")
    c.toggle_intervention()
    assert c._pending_event == ("finish", "success")
    c.robot.begin_human_control.assert_not_called()


@pytest.mark.parametrize("fail", [False, True])
def test_save_and_training_leave_motion_disabled(monkeypatch, fail):
    c = collector()
    c.enable_teleop()
    c.start_episode()
    c._at_boundary = True
    c.arm = SimpleNamespace(workspace_min=[0, 0, 0], workspace_max=[1, 1, 1])
    c.config.dataset_root = "unused"
    c._gamepad_controls.profile_metadata = "{}"
    c._command = CommandSnapshot(np.ones(10), 2, 1, 1.001, 1)
    c._priming = False
    for i in range(12):
        c._record_frame(SimpleNamespace(observed_at_ns=1_000_000_000 + i * 33_333_333), np.zeros(29), {})
    monkeypatch.setattr("vla.datacollection.franka_robotiq_rollout.collect_franky.validate_rollout_episode", lambda *a, **kw: None)
    def save(*args, **kwargs):
        assert c._setup_teleop.mode == "standby"
        return 0
    c.writer.save_episode.side_effect = save
    def train(**kwargs):
        assert c._setup_teleop.mode == "standby"
        if fail:
            raise RuntimeError("training error")
        return dict(warmup_env_steps=0, warmup_steps=0, warmup_remaining=0, transitions=5)
    c.policy_client.process_saved_episode.side_effect = train
    c.finish_episode("success")
    assert not c._episode_active
    assert c._faulted is fail
    assert c._setup_teleop.mode == ("fault" if fail else "standby")
    c.writer.save_episode.assert_called_once()
    c._setup_teleop.enable.reset_mock()
    if fail:
        c.enable_teleop()
        c._setup_teleop.enable.assert_not_called()


@pytest.mark.parametrize("human", [False, True])
def test_early_observation_keeps_pending_action_and_does_not_stop(human):
    c = collector()
    c._episode_active = True
    c._intervention = human
    c._command = CommandSnapshot(np.ones(10), 0 if human else 2, 1, 1.001, 1)
    c._record_frame(SimpleNamespace(observed_at_ns=1_000_000_000), np.zeros(29), {})
    c._pending_event = ("finish", "success")
    c.finish_episode = Mock()
    c._tick_timed(SimpleNamespace(observed_at_ns=1_010_800_000), np.ones(29), {})
    assert c._pending_frame == 0 and len(c._states) == 1
    assert c._annotations["complementary_info.action_valid"] == [[0]]
    c.robot.stop.assert_not_called()
    c.finish_episode.assert_not_called()
    c._tick_timed(SimpleNamespace(observed_at_ns=1_033_333_333), np.ones(29), {})
    assert c._annotations["complementary_info.action_valid"][0] == [1]
    c.finish_episode.assert_called_once_with("success")


@pytest.mark.parametrize("next_stamp", [1_060_000_000, 990_000_000])
def test_invalid_recording_interval_discards_instead_of_repriming_human(next_stamp):
    c = collector()
    c._episode_active = c._intervention = True
    c._command = CommandSnapshot(np.ones(10), 0, 1, 1.001, 1)
    c._record_frame(SimpleNamespace(observed_at_ns=1_000_000_000), np.zeros(29), {})
    c._tick_timed(SimpleNamespace(observed_at_ns=next_stamp), np.ones(29), {})
    assert not c._episode_active and c._setup_teleop.mode == "standby"
    c.writer.save_episode.assert_not_called()
    c.robot.end_human_control.assert_called()
