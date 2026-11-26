"""Control-chain hardening: inference budget, per-source step limits, worker rearm,
health export and realtime reporting. Fakes only; no hardware or Franky binding."""
import sys
import time
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vla.actions import ActionSpec, matrix_to_rot6d, rot6d_to_matrix
from vla.control.franky import worker as worker_module
from vla.control.franky.ipc import WORKER_IMPLEMENTATION_VERSION, WorkerState
from vla.control.franky.worker import FrankyArm, FrankyWorker
from vla.control.ros import ArmMeasurement, ControlCoordinator
from vla.datacollection.franka_robotiq_rollout.config import (
    DEPLOYMENT_SCHEDULING_FIELDS, RolloutConfig, resolve_inference_slots)


@pytest.mark.parametrize('requested,horizons,stride,expected', [
    ('auto', {'vla': 50, 'auto': 50}, 1, 6),           # VLA-only uses the full auto budget
    ('auto', {'vla': 25, 'rlt': 10, 'auto': 25}, 1, 4),  # RLT H=10 admits d<=5, i.e. 4 + 1 guard
    ('auto', {'vla': 25, 'rlt': 10, 'auto': 25}, 2, 6),  # stride 2 halves the frozen steps
    (4, {'rlt': 10}, 1, 4),
])
def test_inference_budget_resolution(requested, horizons, stride, expected):
    assert resolve_inference_slots(requested, horizons, stride, 1, 6) == expected


def test_explicit_budget_that_breaks_horizon_contract_names_the_limit():
    with pytest.raises(ValueError, match='at most 4 or auto'):
        resolve_inference_slots(6, {'rlt': 10}, 1, 1)
    with pytest.raises(ValueError, match='admits no inference budget'):
        resolve_inference_slots('auto', {'rlt': 3}, 1, 1)


def test_scheduling_fields_exist_on_rollout_config():
    names = {field for field in RolloutConfig.__dataclass_fields__}
    assert set(DEPLOYMENT_SCHEDULING_FIELDS) <= names
    assert RolloutConfig.__dataclass_fields__['inference_slots'].default == 'auto'


def _tcp(position, rotation=np.eye(3), gripper=.5):
    return np.r_[np.asarray(position, float), matrix_to_rot6d(rotation), gripper]


def _coordinator(source, **limits):
    now = [1_000_000_000]
    c = ControlCoordinator(ActionSpec('tcp', ('left',)), {},
                           {'left': (np.array([-1., -1, 0]), np.array([1., 1, 1]))},
                           lambda *a: None, lambda g: None, backend='franky', clock=lambda: now[0],
                           control_hz=30, **limits)
    c.observe('left', ArmMeasurement(np.zeros(7), .5, np.zeros(3), np.eye(3), now[0]))
    c.transition(source)
    c.acknowledge('left', c.generation, 0, np.zeros(7), now[0])
    return c


def test_gello_reference_is_limited_per_slot_and_recorded_as_accepted():
    c = _coordinator('gello', step_limits={'gello': (.6, 2.)})
    c.accept(_tcp([.4, 0, .3]), source='gello', generation=c.generation, slot_id=0)
    accepted = c.accept(_tcp([.5, 0, .3]), source='gello', generation=c.generation, slot_id=1)
    np.testing.assert_allclose(accepted.target[:3], [.4 + .6 / 30, 0, .3])
    np.testing.assert_allclose(c.candidate[:3], [.5, 0, .3])  # raw input kept as candidate
    assert c.last_step_limited and c.limited_steps == 1
    turned = Rotation.from_rotvec([0, 0, 1.]).as_matrix()
    accepted = c.accept(_tcp(accepted.target[:3], turned), source='gello', generation=c.generation, slot_id=2)
    rotation = Rotation.from_matrix(rot6d_to_matrix(accepted.target[3:9]))
    assert rotation.magnitude() == pytest.approx(2. / 30)
    # A new generation has no previous target: never limit against a revoked owner.
    c.transition('gello')
    c.acknowledge('left', c.generation, 0, np.zeros(7), c.clock())
    first = c.accept(_tcp([.9, 0, .3]), source='gello', generation=c.generation, slot_id=3)
    np.testing.assert_allclose(first.target[:3], [.9, 0, .3])


def test_policy_jump_guard_faults_instead_of_clipping_and_respects_stride():
    c = _coordinator('policy', jump_guards={'policy': (.5, 2.5)})
    c.accept(_tcp([.4, 0, .3]), source='policy', generation=c.generation, slot_id=0)
    # A stride-2 plan holds a target for two slots, then moves one model step.
    c.accept(_tcp([.4, 0, .3]), source='policy', generation=c.generation, slot_id=1)
    allowed = .5 * 2 / 30
    accepted = c.accept(_tcp([.4 + .99 * allowed, 0, .3]), source='policy', generation=c.generation, slot_id=2)
    assert accepted.target[0] == pytest.approx(.4 + .99 * allowed)  # never modified
    with pytest.raises(RuntimeError, match='policy target jump'):
        c.accept(_tcp([.6, 0, .3]), source='policy', generation=c.generation, slot_id=3)


class _Affine:
    def __init__(self, matrix):
        matrix = np.asarray(matrix)
        self.translation = matrix[:3, 3]
        self.quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()


class _Tracker:
    instances = 0

    def __init__(self, *args, **kwargs):
        _Tracker.instances += 1
        self.targets, self.is_running = [], True

    def set_target(self, target): self.targets.append(target)
    def set_gains(self, **kwargs): pass
    def stop(self): self.is_running = False


class _Errors:
    joint_velocity_violation = True
    cartesian_reflex = False


def _robot(position=(.4, 0., .3), mode='Move', rate=.97):
    matrix = np.eye(4)
    matrix[:3, 3] = position
    state = NS(O_T_EE=_Affine(matrix), F_T_EE=_Affine(np.eye(4)), q=np.zeros(7), dq=np.zeros(7),
               q_d=np.zeros(7), K_F_ext_hat_K=np.zeros(6), robot_mode=NS(name=mode),
               control_command_success_rate=rate, current_errors=_Errors())
    return NS(state=state, join_motion=lambda: None, recover_from_errors=lambda: None)


def _arm():
    arm = FrankyArm('left', '127.0.0.1')
    arm.franky = NS(Affine=_Affine, CartesianImpedanceTracker=_Tracker)
    arm.robot = _robot()
    return arm


def test_rearm_keeps_the_running_tracker_and_reseeds_from_measured():
    arm = _arm()
    arm.reset(1)
    tracker, created = arm.tracker, _Tracker.instances
    arm.set_target([.42, 0, .3, 0, 0, 0, 1], 1, 0)
    arm.robot.state.O_T_EE = _Affine(np.c_[np.r_[np.eye(3), np.zeros((1, 3))], [.41, 0, .3, 1]])
    assert arm.rearmable()
    arm.rearm(2)
    assert arm.tracker is tracker and _Tracker.instances == created  # no restart
    np.testing.assert_allclose(arm.target_pose[:3], [.41, 0, .3])
    np.testing.assert_allclose(tracker.targets[-1].translation, [.41, 0, .3])
    assert (arm.target_generation, arm.target_sequence) == (2, -1)
    arm.stop()
    assert not arm.rearmable()
    with pytest.raises(RuntimeError, match='full reset'):
        arm.rearm(3)


def test_worker_reset_rearms_only_a_healthy_streaming_owner():
    worker = FrankyWorker({'left': {'robot_ip': 'mock'}}, session_id='s')
    arm = _arm()
    worker.arms = {'left': arm}
    first = worker.request({'kind': 'reset', 'session_id': 's', 'generation': 1, 'request_id': 'r1'})
    assert first['request_status'] == 'applied' and 'rearmed' not in first
    tracker = arm.tracker
    second = worker.request({'kind': 'reset', 'session_id': 's', 'generation': 2, 'request_id': 'r2'})
    assert second.get('rearmed') and arm.tracker is tracker and worker.generation == 2
    worker.request({'kind': 'stop', 'session_id': 's', 'generation': 3, 'request_id': 's3'})
    third = worker.request({'kind': 'reset', 'session_id': 's', 'generation': 4, 'request_id': 'r4'})
    assert 'rearmed' not in third and arm.tracker is not tracker  # standby -> full start
    assert worker.snapshot()['implementation_version'] == WORKER_IMPLEMENTATION_VERSION


def test_worker_exports_libfranka_health_and_settled_success_rate(monkeypatch):
    arm = _arm()
    arm.reset(1)
    state = arm._state()
    assert state['robot_mode'] == 'Move'
    assert state['current_errors'] == ['joint_velocity_violation']
    assert state['control_command_success_rate'] is None  # tracker just started
    arm._tracker_started_at = time.monotonic() - 1
    assert arm._state()['control_command_success_rate'] == pytest.approx(.97)
    parsed = WorkerState.from_dict({**arm._state(), 'fault': None})
    assert parsed.robot_mode == 'Move' and parsed.current_errors == ('joint_velocity_violation',)
    with pytest.raises(ValueError):
        WorkerState.from_dict({**arm._state(), 'control_command_success_rate': 1.5})


def test_realtime_enforce_requires_preempt_rt_before_opening_fci(monkeypatch):
    report = worker_module.kernel_realtime()
    assert set(report) == {'preempt_rt', 'kernel'} and isinstance(report['preempt_rt'], bool)
    franky = ModuleType('franky')
    franky.RealtimeConfig = NS(Enforce='enforce', Ignore='ignore')
    franky.Robot = lambda *a, **k: pytest.fail('FCI opened on a non-RT kernel')
    monkeypatch.setitem(sys.modules, 'franky', franky)
    monkeypatch.setattr(worker_module, 'check_binding', lambda: {})
    monkeypatch.setattr(worker_module, 'HardwareLease', lambda *a: pytest.fail('lease taken'))
    monkeypatch.setattr(worker_module, 'kernel_realtime', lambda: {'preempt_rt': False, 'kernel': 'generic'})
    with pytest.raises(RuntimeError, match='PREEMPT_RT'):
        FrankyArm('left', '127.0.0.1', realtime_config='enforce').open()
    with pytest.raises(ValueError, match='realtime_config'):
        FrankyWorker({'left': {'robot_ip': 'mock'}}, session_id='s', realtime_config='always')
