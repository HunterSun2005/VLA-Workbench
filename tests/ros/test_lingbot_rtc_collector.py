"""Production collector entry points, cached observations and targets only."""
from types import SimpleNamespace
import time

import numpy as np

from vla.actions import ActionSpec
from vla.control.execution import PlanResponse
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
