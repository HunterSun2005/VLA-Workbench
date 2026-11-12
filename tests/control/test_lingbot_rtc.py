"""RTC timeline, atomic projection and revocation without hardware or model assets."""
from dataclasses import replace

import numpy as np
import pytest

from vla.actions import ActionSpec
from vla.control.execution import ExecutionState, PlanResponse
from vla.control.lingbot_execution import LingBotRTCContract, LingBotRTCExecutor

POSE = np.array([.4, 0., .3, 1., 0., 0., 0., 1., 0., .5123456789])


def setup(project=None):
    ex = LingBotRTCExecutor(LingBotRTCContract(ActionSpec(), {'vla': 50}, 'c', 'k', 't'),
        session_id='rtc', model_version='m', epoch_ns=0, project=project)
    ex.enable(POSE, episode_id='e')
    return ex


def request(ex, slot):
    return ex.request(slot=slot, anchor=POSE, observation_id=slot,
                      observed_at_ns=ex.control_clock.deadline(slot), policy='vla')


def response(req):
    actions = np.tile(POSE, (50, 1))
    actions[:, 0] += .001
    actions[:req.start_idx] = req.prefix
    return PlanResponse(req, actions, ('vla',)*50, (0,)*50, actions)


def test_rtc_runs_old_projected_queue_and_replaces_only_fixed_suffix():
    def project(a):
        a = a.copy(); a[0] += .001
        return a
    ex = setup(project)
    req = request(ex, 0)
    assert req.start_idx == 15 and req.replace_slot == 15
    assert ex.install(response(req), 1)
    for slot in range(20):
        record = ex.advance(slot, ex.control_clock.deadline(slot))
        assert record.source == ('system_hold' if slot < 15 else 'vla')
    req2 = request(ex, 20)
    np.testing.assert_array_equal(req2.prefix, [ex.queue[s][0] for s in range(20, 35)])
    assert request(ex, 20) is None
    old = {s: ex.queue[s] for s in range(20, 35)}
    for slot in range(20, 23):
        assert ex.advance(slot, ex.control_clock.deadline(slot)).plan_version == req.plan_version
    assert ex.install(response(req2), ex.control_clock.deadline(23))
    for slot in range(23, 35):
        assert ex.advance(slot, ex.control_clock.deadline(slot)).plan_version == old[slot][3]
    assert ex.advance(35, ex.control_clock.deadline(35)).plan_version == req2.plan_version


@pytest.mark.parametrize('fault', ['prefix', 'source', 'phase', 'reference', 'context', 'projection'])
def test_invalid_candidate_revokes_everything_and_cannot_retry(fault):
    ex = setup(); req = request(ex, 0); result = response(req)
    if fault == 'prefix':
        actions = result.actions.copy(); actions[0, 0] += .001
        result = replace(result, actions=actions, vla_reference=actions)
    elif fault == 'source':
        result = replace(result, sources=('actor',)*50)
    elif fault == 'phase':
        result = replace(result, phases=(1,)*50)
    elif fault == 'reference':
        result = replace(result, vla_reference=result.actions+.001)
    elif fault == 'context':
        result = replace(result, context={'actor': np.zeros(1)})
    else:
        calls = []
        def bad(a):
            calls.append(1)
            if len(calls) == 10:
                return np.full(10, np.nan)
            return a
        ex.project = bad
    with pytest.raises(ValueError):
        ex.install(result, 1)
    assert ex.state == ExecutionState.WAITING and not ex.queue and ex.inflight is None
    assert not ex.install(response(req), 2) and request(ex, 1) is None


@pytest.mark.parametrize('fault', ['timeout', 'late', 'missing', 'skip', 'takeover', 'storage', 'disconnect'])
def test_discontinuity_requires_explicit_enable(fault):
    ex = setup(); req = request(ex, 0)
    if fault == 'timeout':
        assert ex.advance(0, req.deadline_ns) is None
    elif fault == 'late':
        assert not ex.install(response(req), req.deadline_ns)
    elif fault == 'missing':
        ex.queue.pop(0)
        assert ex.advance(0, 1) is None
    elif fault == 'skip':
        ex.advance(0, 1)
        assert ex.advance(2, 2) is None
    else:
        ex.invalidate(ExecutionState.STORAGE if fault == 'storage' else ExecutionState.WAITING, fault)
    assert not ex.install(response(req), 3)
    assert request(ex, 2) is None
    assert ex.advance(3, 4) is None
    ex.enable(POSE, episode_id='resumed')
    assert request(ex, 4) is not None


def test_old_generation_does_not_revoke_new_inflight():
    ex = setup(); old = request(ex, 0)
    ex.enable(POSE, episode_id='new'); current = request(ex, 1)
    assert not ex.install(response(old), 2)
    assert ex.inflight is current
    assert ex.install(response(current), 3)


@pytest.mark.parametrize('kwargs', [{'inference_slots': 0}, {'guard_slots': True},
    {'inference_slots': 15}, {'inference_slots': 25, 'rtc_max_delay': 40}, {'frame_stride': 2}])
def test_invalid_budget(kwargs):
    with pytest.raises(ValueError):
        LingBotRTCContract(ActionSpec(), {'vla': 50}, 'c', 'k', 't', **kwargs)


@pytest.mark.parametrize('field', ['anchor', 'prefix', 'deadline_ns', 'replace_slot'])
def test_echoed_request_context_cannot_be_changed(field):
    ex = setup(); req = request(ex, 0)
    if field in ('anchor', 'prefix'):
        value = getattr(req, field).copy(); value.flat[0] += .001
        other = replace(req, **{field: value})
    elif field == 'replace_slot':
        other = replace(req, replace_slot=req.replace_slot-1, prefix=req.prefix[:-1])
    else:
        other = replace(req, deadline_ns=req.deadline_ns+1)
    assert not ex.install(response(other), 1)
    assert ex.state == ExecutionState.WAITING and ex.inflight is None and not ex.queue


def test_manual_enable_after_long_pause_starts_new_continuity_window():
    ex = setup(); req = request(ex, 0)
    assert ex.install(response(req), 1)
    assert ex.advance(0, 1) is not None
    ex.invalidate(reason='manual pause')
    assert ex.advance(40, ex.control_clock.deadline(40)) is None
    ex.enable(POSE, episode_id='resumed')
    req = request(ex, 100)
    assert ex.install(response(req), ex.control_clock.deadline(100)+1)
    assert ex.advance(100, ex.control_clock.deadline(100)+2).source == 'system_hold'
    assert ex.advance(101, ex.control_clock.deadline(101)).source == 'system_hold'
    assert ex.advance(103, ex.control_clock.deadline(103)) is None
    assert ex.state == ExecutionState.WAITING
