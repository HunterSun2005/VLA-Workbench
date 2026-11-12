"""Production collector methods with fixed cached ACKs, never devices/ROS nodes."""
from types import SimpleNamespace
import time

import numpy as np
import pytest

from vla.actions import ActionSpec
from vla.control.execution import ExecutionState, PlanResponse
from vla.control.serial_execution import SerialExecutionContract, SerialExecutor
from vla.datacollection.franka_robotiq_rollout.collect_ros import RolloutCollector


def setup():
    pose = np.array([.4, 0., .3, 1., 0., 0., 0., 1., 0., .5])
    ex = SerialExecutor(SerialExecutionContract(ActionSpec(), 'c', 'k', 't'),
                        session_id='serial', model_version='model', epoch_ns=time.monotonic_ns())
    ex.enable(pose, episode_id='episode')
    req = ex.request(slot=0, anchor=pose, observation_id=0, observed_at_ns=time.monotonic_ns(), policy='vla')
    actions = np.tile(pose, (50, 1))
    result = PlanResponse(req, actions, ('vla',)*50, (0,)*50, actions)
    node = RolloutCollector.__new__(RolloutCollector)
    node._execution = ex
    node._slot_id = 7
    node.policy_client = SimpleNamespace(poll_latest=lambda: result)
    node._consume_policy_result()
    return node, ex


def test_collector_installs_at_current_slot_without_rewriting_request():
    node, ex = setup()
    assert min(ex.queue) == 7
    event = next(e for e in ex.events if e[0] == 'install_serial')
    assert event[1].request.start_slot == 0 and event[2]['start_slot'] == 7
    assert ex.advance(7, time.monotonic_ns()).action_index == 0


@pytest.mark.parametrize('matching', [True, False])
def test_only_correlated_final_worker_ack_allows_next_request(matching):
    node, ex = setup()
    for slot in range(7, 57):
        final = ex.advance(slot, time.monotonic_ns())
    assert not ex.needs_plan(57)
    node.config = SimpleNamespace(control_discontinuity_timeout_s=.1)
    node._worker_session_id = 'worker'
    node.coordinator = SimpleNamespace(generation=9)
    node._journal = None
    node._pending_submission = {'request_id': 'target', 'generation': 9, 'sequence': 42,
                                'slot_id': 56, 'started_ns': time.monotonic_ns(),
                                'context': {'execution': final}, 'frame': None}
    reply = {'request_status': 'applied', 'session_id': 'worker', 'request_id': 'target',
             'generation': 9, 'sequence': 42 if matching else 41, 'slot_id': 56,
             'applied_at_ns': time.monotonic_ns()}
    node._worker_client = SimpleNamespace(result=lambda _: reply)
    if matching:
        node._consume_submission()
        assert ex.state == ExecutionState.WAITING_INFERENCE and ex.needs_plan(57)
    else:
        with pytest.raises(RuntimeError, match='acknowledgement'):
            node._consume_submission()
        assert not ex.needs_plan(57)


def test_serial_request_retains_snapshot_time_not_enqueue_time():
    node, ex = setup()
    pose = np.array([.4, 0., .3, 1., 0., 0., 0., 1., 0., .5])
    ex.enable(pose, episode_id='timestamp-test')
    node._intervention = node._waiting_resume = False
    node.coordinator = SimpleNamespace(mode='active')
    node._phase = 'vla'
    node._episode_task_index = 0
    node.config = SimpleNamespace(tasks=['test'])
    node._raw_anchor = node._policy_state = lambda raw: pose
    node._reset_policy_on_request = True
    node._log_next_policy_request = False
    submitted = []
    def submit(**kwargs):
        submitted.append(kwargs)
        return kwargs['plan'].request_id
    node.policy_client = SimpleNamespace(control_mode='vla', submit=submit)
    observed = time.monotonic_ns()-20_000_000
    node._submit_policy(pose, {}, observed_at_ns=observed)
    assert submitted[0]['plan'].observed_at_ns == observed
    assert submitted[0]['plan'].deadline_ns == observed+5_000_000_000
