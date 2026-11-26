"""Collector-side hardening with ROS transport and IPC faked (no hardware)."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tests.rlt.test_dual_gamepad_record import clocked_control, collector, ros_module  # noqa: F401


def test_recording_fingerprint_ignores_deployment_scheduling_only(collector, tmp_path):
    c = collector
    cls = type(c)
    base = c._effective_metadata['recording_contract']
    budget = cls(replace(c.config, inference_slots=3, inference_slots_auto_max=5,
                         dataset_root=tmp_path / 'budget'), demonstration=True)
    try:
        assert budget._effective_metadata['recording_contract'] == base
    finally:
        budget.close()
    limited = cls(replace(c.config, gello_max_linear_velocity=.4, dataset_root=tmp_path / 'limit'),
                  demonstration=True)
    try:
        assert limited._effective_metadata['recording_contract'] != base
    finally:
        limited.close()


def test_gripper_goals_use_deadband_and_refresh(collector, ros_module, monkeypatch):
    c = collector
    clock = [10_000_000_000]
    monkeypatch.setattr(ros_module, 'time', NS(monotonic_ns=lambda: clock[0], monotonic=lambda: clock[0] / 1e9))
    c._last_gripper_goal, c._last_gripper_goal_ns = {}, {}
    assert c._gripper_goal_due('left', .5)
    c._last_gripper_goal['left'], c._last_gripper_goal_ns['left'] = .5, clock[0]
    assert not c._gripper_goal_due('left', .5)
    assert not c._gripper_goal_due('left', .505)          # inside deadband, too soon
    assert c._gripper_goal_due('left', .52)               # real change
    clock[0] += int(c.config.gripper_goal_refresh_s * 1e9)
    assert c._gripper_goal_due('left', .505)              # small drift converges after refresh


def _gello(c, far):
    c.config = replace(c.config, input_source='gello')
    for arm in c.config.arms:
        stamp = c._latest[f'{arm.name}.robot_state'][0]
        c._latest[f'{arm.name}.gello_joint'] = (stamp, NS(name=[f'j{i}' for i in range(7)], position=[0.] * 7))
        c._latest[f'{arm.name}.gello_gripper'] = (stamp, NS(data=.5))
        position = np.array([.4 + (.08 if far else .005), 0., .3])
        c.servos[arm.name] = replace_forward(c.servos[arm.name], position)


def replace_forward(servo, position):
    return NS(**{**vars(servo), 'forward_tcp': lambda *args: (position, np.eye(3))})


@pytest.mark.parametrize('far', [True, False])
def test_gello_engages_only_near_measured_tcp(collector, far):
    c = collector
    _gello(c, far)
    alignment = c._gello_alignment()
    assert alignment['left'][0] == pytest.approx(.08 if far else .005)
    c.enable_teleop()
    assert c._teleop_enabled is not far
    if far:
        assert '未对齐' in c._engage_rejection and '8.0 cm' in c._engage_rejection
        assert not c._intervention
    diagnostics = c.input_diagnostics()['left']
    if far:
        assert diagnostics['engage_allowed'] is False


def test_unexpected_tick_error_latches_fault_and_keeps_session(collector, ros_module, monkeypatch):
    c = collector
    clock, tick = clocked_control(c, ros_module, monkeypatch)
    def broken(*args):
        raise TypeError('defect in an input adapter')
    monkeypatch.setattr(c, '_publish_gamepad', broken)
    tick()
    assert c._faulted and 'TypeError' in c._fault_reason
    assert not c._recording and not c._teleop_enabled


def test_tick_timing_counts_lateness_and_missed_slots(collector, ros_module, monkeypatch):
    c = collector
    clock, tick = clocked_control(c, ros_module, monkeypatch)
    tick()
    clock[0] += 2 * 33_333_334 + 5_000_000   # two slots missed, next poll 5 ms late
    tick()
    timing = c.timing_status()
    assert timing['missed_slots'] == 2
    assert timing['lateness_max_ms'] == pytest.approx(5.0, abs=.2)
    assert timing['inference_slots'] is None   # collection has no RTC executor
