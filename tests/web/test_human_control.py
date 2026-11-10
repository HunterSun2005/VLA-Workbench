"""Human positioning and explicit handback using production collector/coordinator.

Only ROS transport, sensors, kinematics and the model client are mocked. No devices.
"""
from dataclasses import replace
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from tests.rlt.test_dual_gamepad_record import collector, ros_module
from vla.control.execution import ExecutionContract, Executor, ExecutionState, PlanResponse
from vla.web.backends import BaseBackend, ROSBackend
from vla.web.config import SessionConfig


@pytest.fixture(params=[('single_left', 'gamepad'), ('dual', 'gamepad'),
                        ('single_left', 'gello'), ('dual', 'gello')])
def deployment(collector, ros_module, monkeypatch, request):
    arms, source = request.param
    count = 2 if arms == 'dual' else 1
    config = replace(collector.config, arms=collector.config.arms[:count], input_source=source,
                     cameras=tuple(cam for cam in collector.config.cameras if count == 2 or cam.name != 'right_wrist'),
                     dataset_root=collector.config.dataset_root.parent/'rollout', subtasks=())
    c = ros_module.RolloutCollector(config, demonstration=True)
    c.demonstration = c._manual_requested = False
    c._web_mode = 'vla'
    now = time.monotonic_ns()
    c._latest = {key: (now, value) for key, (_, value) in collector._latest.items()}
    for arm in c.config.arms:
        c._latest[f'{arm.name}.gello_joint'] = c._latest[f'{arm.name}.measured_joint']
        c._latest[f'{arm.name}.gello_gripper'] = (now, NS(data=.5))
    values = {key: value for key, (_, value) in c._latest.items()}
    images = {cam.feature_key: np.zeros((224, 224, 3), np.uint8) for cam in config.cameras}
    monkeypatch.setattr(c, '_snapshot', lambda **kwargs: (values, images))
    raw = np.zeros(50*count, np.float32)
    for offset in range(0, len(raw), 50):
        raw[offset:offset+10] = [.4, 0., .3, 1., 0., 0., 0., 1., 0., .0425]
    monkeypatch.setattr(c, '_raw_state', lambda *args: raw.copy())
    monkeypatch.setattr(c, '_restore_policy_gripper_speed', Mock())
    c.policy_client = Mock(control_mode='vla', contract={'rtc_enabled': True})
    c.policy_client.set_control_mode.side_effect = lambda mode: setattr(c.policy_client, 'control_mode', mode)
    c.policy_client.poll_latest.return_value = None
    c._execution = Executor(ExecutionContract(config.action_spec, {'vla': 12, 'rlt': 12, 'auto': 12},
                                           'control', 'urdf', 'tcp'),
                          session_id='test-session', model_version='test-model', epoch_ns=now,
                          project=c.coordinator.project)
    b = object.__new__(ROSBackend)
    b.collector = c
    b.config = SessionConfig(workflow='deploy', arms=arms, source=source, handover_path='mock-head')
    b.arms = [arm.name for arm in c.config.arms]
    b.enabled = b._policy_armed = False
    b.mode = 'vla'
    b.record_started = None
    b.check_children = Mock()
    b.executor = NS(spin_once=lambda **kwargs: None)
    b.poses = NS(tick=lambda: None)
    yield b, c, raw, values, images
    c.close()


def command(b, op, **args):
    return BaseBackend.collector_command(b, op, args)


def refresh(c):
    now = time.monotonic_ns()
    c._latest = {key: (now, value) for key, (_, value) in c._latest.items()}


def test_positioning_is_unrecorded_and_recording_keeps_human_owner(deployment):
    b, c, raw, values, images = deployment
    command(b, 'mode', mode='human')
    assert c._manual_requested and not b.enabled and c.coordinator.owner is None
    c.policy_client.set_control_mode.assert_not_called()  # human is not a model inference mode
    command(b, 'enable')
    assert b.enabled and c._teleop_enabled and c._intervention
    assert c.coordinator.owner == c.config.input_source
    assert b.collector_status()['actual_source'] == c.config.input_source
    assert c._execution.state == ExecutionState.HUMAN
    c._record_frame(raw, values, images, time.monotonic_ns())
    assert not c._states and c._journal is None
    generation = c.coordinator.generation
    command(b, 'mode', mode='human')
    assert c.coordinator.generation == generation  # explicit choice is idempotent
    command(b, 'record')
    assert c._recording and c.coordinator.owner == c.config.input_source
    assert c._execution.state == ExecutionState.HUMAN and not c._states
    c._restore_policy_gripper_speed.assert_not_called()
    target = c._raw_anchor(raw)
    c._observe_control()
    c.coordinator.accept(target, source=c.config.input_source, generation=c.coordinator.generation, slot_id=0)
    c._track_control(.01)
    c._record_frame(raw, values, images, time.monotonic_ns())
    assert len(c._states) == 1
    assert c._annotations['complementary_info.is_intervention'] == [[1.]]
    assert c._annotations['command.source'] == [[0]]
    command(b, 'mode', mode='rlt')
    assert c._recording and not c._intervention and not c._manual_requested
    assert c.coordinator.owner == 'policy' and c._execution.state == ExecutionState.PREPARING
    assert c.policy_client.control_mode == 'rlt'
    generation = c.coordinator.generation
    command(b, 'mode', mode='rlt')
    assert c.coordinator.generation == generation


def test_positioning_to_policy_stops_until_explicit_enable_and_record(deployment):
    b, c, *_ = deployment
    command(b, 'mode', mode='human')
    command(b, 'enable')
    command(b, 'mode', mode='vla')
    assert not b.enabled and not c._teleop_enabled and not c._intervention
    assert c.coordinator.owner is None and c._journal is None
    c.policy_client.submit.assert_not_called()
    command(b, 'enable')
    b.tick()  # Readiness must persist while waiting for the user's record command.
    assert b.enabled and b._policy_armed and c.coordinator.owner is None
    command(b, 'record')
    assert c._recording and c.coordinator.owner == 'policy' and not b._policy_armed
    req = c._execution.request(slot=0, anchor=c._raw_anchor(c._raw_state()), observation_id=0,
                             observed_at_ns=c._execution.control_clock.deadline(0), policy='vla')
    old = PlanResponse(req, np.tile(req.anchor, (req.horizon, 1)), ('vla',)*req.horizon,
                       (0,)*req.horizon, np.tile(req.anchor, (req.horizon, 1)))
    command(b, 'mode', mode='human')
    assert c._execution.state == ExecutionState.HUMAN
    assert not c._execution.install(old, time.monotonic_ns())


@pytest.mark.parametrize('blocked', ['_faulted', '_storage_blocked', '_waiting_resume', '_saving', '_processing'])
def test_selection_cannot_resume_blocked_motion(deployment, blocked):
    b, c, *_ = deployment
    command(b, 'mode', mode='human')
    command(b, 'enable')
    command(b, 'record')
    setattr(c, blocked, True)
    generation = c.coordinator.generation
    for mode in ['human', 'vla', 'rlt', 'auto']:
        with pytest.raises(RuntimeError, match='recover/enable'):
            command(b, 'mode', mode=mode)
    assert c.coordinator.generation == generation
    setattr(c, blocked, False)


def test_stale_or_active_input_blocks_initial_manual_enable(deployment):
    b, c, *_ = deployment
    command(b, 'mode', mode='human')
    arm = c.config.arms[-1].name
    key = f'{arm}.gello_joint' if c.config.input_source == 'gello' else f'{arm}.gamepad'
    stamp, value = c._latest[key]
    c._latest[key] = (stamp - int(10e9), value)
    with pytest.raises(RuntimeError, match='enable rejected'):
        command(b, 'enable')
    assert not b.enabled and not c._teleop_enabled and c.coordinator.owner is None
    refresh(c)
    command(b, 'enable')
    assert b.enabled
    if c.config.input_source == 'gamepad':
        c._latest[key][1].axes[1] = 1.
        with pytest.raises(RuntimeError, match='start rejected'):
            command(b, 'record')
        assert not c._recording and c._journal is None


def test_pause_retains_choice_but_selection_does_not_enable(deployment):
    b, c, *_ = deployment
    command(b, 'mode', mode='human')
    command(b, 'enable')
    command(b, 'pause')
    assert not b.enabled and not c._teleop_enabled and c.coordinator.owner is None
    command(b, 'mode', mode='rlt')
    command(b, 'mode', mode='human')
    assert not b.enabled and not c._intervention and c.coordinator.owner is None
    assert b.collector_status()['requested_mode'] == 'human'
    assert b.collector_status()['actual_source'] == 'idle'


def test_controller_reset_and_pending_save_block_selection(deployment):
    b, c, *_ = deployment
    generation = c.coordinator.generation
    for state in ['resetting', 'fault']:
        c.coordinator.mode = state
        with pytest.raises(RuntimeError, match='recover/enable'):
            command(b, 'mode', mode='human')
        c.enable_teleop()
        assert not c._teleop_enabled
    c.coordinator.mode = 'standby'
    c._pending_finish = 'success'
    with pytest.raises(RuntimeError, match='recover/enable'):
        command(b, 'mode', mode='rlt')
    assert c.coordinator.generation == generation
    c._pending_finish = None


def test_direct_handback_before_recording_never_starts_policy(deployment):
    b, c, *_ = deployment
    command(b, 'mode', mode='human')
    command(b, 'enable')
    c.toggle_intervention()
    b.tick()
    assert not b.enabled and not c._teleop_enabled and not c._manual_requested
    assert c.coordinator.owner is None and c._execution.state == ExecutionState.STANDBY
    c.policy_client.submit.assert_not_called()
