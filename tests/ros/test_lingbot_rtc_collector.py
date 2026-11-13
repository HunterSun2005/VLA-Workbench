"""Production collector entry points, cached observations and targets only."""
from types import SimpleNamespace
import time
import threading
from collections import Counter

import numpy as np

from vla.actions import ActionSpec
from vla.control.execution import ExecutionState, PlanResponse
from vla.control.lingbot_execution import LingBotRTCContract, LingBotRTCExecutor
from vla.datacollection.franka_robotiq_rollout.collect_ros import RolloutCollector


def test_collector_keeps_snapshot_time_and_fixed_install_boundary():
    epoch = time.monotonic_ns()
    pose = np.array([.4, 0, .3, 1, 0, 0, 0, 1, 0, .5])
    ex = LingBotRTCExecutor(LingBotRTCContract(ActionSpec(), {'vla': 50}, 'c', 'k', 't'),
        session_id='rtc', model_version='m', epoch_ns=epoch)
    ex.enable(pose, episode_id='e')
    node = RolloutCollector.__new__(RolloutCollector)
    node._execution, node._slot_id = ex, 0
    node._intervention = node._waiting_resume = False
    node.coordinator = SimpleNamespace(mode='active')
    node._phase, node._episode_task_index = 'vla', 0
    node.config = SimpleNamespace(tasks=['test'])
    node._raw_anchor = node._policy_state = lambda raw: pose
    node._reset_policy_on_request, node._log_next_policy_request = True, False
    submitted = []
    def submit(**kw):
        submitted.append(kw['plan'])
        return kw['plan'].request_id
    node.policy_client = SimpleNamespace(control_mode='vla', submit=submit)
    observed = epoch-10_000_000
    node._submit_policy(pose, {}, observed_at_ns=observed)
    req, = submitted
    assert req.observed_at_ns == observed and req.replace_slot == 15
    assert req.deadline_ns == observed+14*1_000_000_000//30
    actions = np.tile(pose, (50, 1)); actions[15:, 0] += .001
    response = PlanResponse(req, actions, ('vla',)*50, (0,)*50, actions)
    node.policy_client.poll_latest = lambda: response
    node._slot_id = 2
    node._consume_policy_result()
    assert ex.queue[14][1] == 'system_hold' and ex.queue[15][3] == req.plan_version
    # The collector must not retime an early result to the current slot.
    event = next(e for e in ex.events if e[0] == 'install')
    assert event[1].request.replace_slot == 15


def tick_collector():
    """Production timer path with cached observations, no ROS Node or devices."""
    pose = np.array([.4, 0, .3, 1, 0, 0, 0, 1, 0, .5])
    node = RolloutCollector.__new__(RolloutCollector)
    node.config = SimpleNamespace(fps=30, model_backend='lingbot_v2', cameras=[SimpleNamespace(name='base')],
                                  control_discontinuity_timeout_s=.1)
    node._lock = threading.RLock()
    node._slot_clock = SimpleNamespace(poll=lambda _: 14)
    node._last_control_tick = time.monotonic()-1/30
    node._pending_storage_error = node._pending_submission = node._pending_compliance = None
    node._faulted = node._saving = node._storage_blocked = node._teleop_enabled = False
    node._recording = True
    node._intervention = node._waiting_resume = False
    node._journal = node._pending_phase = node._pending_finish = node._pending_frame = None
    node._last_target_applied_ns = None
    node._pending_worker_targets = {}
    node._gamepad_targets = {}
    node._last_gripper_goal = {}
    node._segment_id = 0
    node._skip_reasons = Counter()
    node._latest = {'camera.base': (time.monotonic_ns(), object())}
    node._last_demo_images = {}
    node._worker_client = SimpleNamespace(target_available=True)
    node.coordinator = SimpleNamespace(mode='active', owner='policy', command_id=1, generation=1,
                                       tick=lambda _: None)
    def transition(owner, **kwargs):
        node.coordinator.mode = 'resetting'
        node.coordinator.command_id += 1
        node.coordinator.generation += 1
    node.coordinator.transition = transition
    node.invalidations, node.stops, node.faults = [], [], []
    node.policy_client = SimpleNamespace(raise_if_failed=lambda: None,
        invalidate=lambda *args: node.invalidations.append(args))
    for name in ('_poll_worker_states', '_consume_submission', '_observe_control', '_online_save_tick',
                 '_drain_execution_events', '_complete_interval', '_validate_robot_health', '_submit_policy',
                 '_consume_policy_result'):
        setattr(node, name, lambda *args, **kwargs: None)
    node._camera_failures = lambda: []
    node._snapshot = lambda **kwargs: ({}, {'base': np.ones((2, 2, 3), np.uint8)})
    node._raw_state = lambda *args: pose.copy()
    node._sensor_frame_snapshot = lambda _: np.array([1, 2, 3])
    node._stop_grippers_at_measured_position = lambda: node.stops.append('stop')
    node._latch_fault = node.faults.append
    node._execution_record = None
    return node, pose


def test_tick_after_rtc_deadline_revokes_submission_without_attaching_a_frame():
    node, pose = tick_collector()
    epoch = time.monotonic_ns()-1_000_000_000
    ex = LingBotRTCExecutor(LingBotRTCContract(ActionSpec(), {'vla': 50}, 'c', 'k', 't'),
        session_id='late-rtc', model_version='model', epoch_ns=epoch)
    ex.enable(pose, episode_id='episode')
    req = ex.request(slot=0, anchor=pose, observation_id=0, observed_at_ns=epoch, policy='vla')
    node._execution = ex
    node._sensor_frame_snapshot = lambda _: (_ for _ in ()).throw(AssertionError('no submitted target to record'))
    generation = ex.generation
    node._tick()
    assert node._pending_submission is None and not node.faults
    assert node.coordinator.mode == 'resetting'
    assert node._waiting_resume and ex.state == ExecutionState.WAITING
    assert ex.generation > generation and not ex.queue and ex.inflight is None
    assert node.invalidations and node.stops == ['stop']
    # Late inference and another timer callback cannot restore motion.
    actions = np.tile(pose, (50, 1))
    assert not ex.install(PlanResponse(req, actions, ('vla',)*50, (0,)*50, actions), time.monotonic_ns())
    # A stop/reset acknowledgement only reaches standby, never active policy.
    node.coordinator.mode = 'standby'
    node._tick()
    assert node._waiting_resume and node.coordinator.mode == 'standby' and not node.faults


def test_tick_attaches_owned_frame_only_to_an_actual_submission():
    node, pose = tick_collector()
    node._publish_policy = lambda *args: setattr(node.coordinator, 'command_id', 2)
    pending = {'frame': None, 'context': {'source': 'system_hold'}}
    node._track_control = lambda _: setattr(node, '_pending_submission', pending)
    node._tick()
    frame = pending['frame']
    assert frame is not None and frame[0] is not pose
    np.testing.assert_array_equal(frame[0], pose)
    np.testing.assert_array_equal(frame[2]['base'], np.ones((2, 2, 3), np.uint8))
    np.testing.assert_array_equal(frame[4], [1, 2, 3])
    assert not node.faults
