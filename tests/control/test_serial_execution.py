from dataclasses import replace

import numpy as np
import pytest

from vla.actions import ActionSpec
from vla.control.execution import ExecutionState, PlanResponse
from vla.control.serial_execution import SerialExecutionContract, SerialExecutor


def pose():
    return np.array([.4, 0, .3, 1, 0, 0, 0, 1, 0, .5], dtype=np.float32)


def executor(timeout=5):
    contract = SerialExecutionContract(ActionSpec(), 'control', 'kinematics', 'tool', timeout_s=timeout)
    result = SerialExecutor(contract, session_id='session', model_version='model', epoch_ns=0)
    result.enable(pose(), episode_id='episode')
    return result


def request(ex, slot=0):
    return ex.request(slot=slot, anchor=pose(), observation_id=slot,
                      observed_at_ns=ex.control_clock.deadline(slot), policy='vla')


def response(req):
    actions = np.tile(pose(), (50, 1))
    actions[:, 0] += np.arange(50)*.001
    return PlanResponse(req, actions, ('vla',)*50, (0,)*50, actions)


def test_full_chunk_after_wait_and_ack_then_new_observation():
    ex = executor()
    req = request(ex)
    assert req.start_idx == 0 and req.start_slot == req.replace_slot == 0
    assert request(ex, 1) is None
    for slot in range(6):
        held = ex.advance(slot, ex.control_clock.deadline(slot))
        assert held.source == 'system_hold'
        np.testing.assert_array_equal(held.accepted, pose())
    candidate = response(req)
    assert ex.install(candidate, 200_000_000, start_slot=6)
    assert candidate.request.observed_at_ns == 0
    for index, slot in enumerate(range(6, 56)):
        assert request(ex, slot) is None  # Never prefetch.
        record = ex.advance(slot, ex.control_clock.deadline(slot))
        assert record.action_index == index and record.source == 'vla'
        np.testing.assert_array_equal(record.candidate, candidate.actions[index])
        if index < 49:
            ex.acknowledge(record)
    assert request(ex, 56) is None  # Final action has not been ACKed.
    hold = ex.advance(56, ex.control_clock.deadline(56))
    assert hold.source == 'system_hold'
    np.testing.assert_array_equal(hold.accepted, candidate.actions[-1])
    ex.acknowledge(record)
    assert ex.state == ExecutionState.WAITING_INFERENCE
    next_request = request(ex, 57)
    assert next_request.observation_id == 57 and next_request.request_id > req.request_id
    install_event = next(event for event in ex.events if event[0] == 'install_serial')
    assert install_event[2]['start_slot'] == 6


@pytest.mark.parametrize('state', [ExecutionState.HUMAN, ExecutionState.FAULT,
                                  ExecutionState.STORAGE, ExecutionState.WAITING, ExecutionState.STANDBY])
def test_revocation_cannot_resume_from_a_late_result(state):
    ex = executor()
    candidate = response(request(ex))
    ex.invalidate(state)
    assert not ex.install(candidate, 10, start_slot=1)
    assert ex.state == state and not ex.queue
    assert request(ex, 2) is None


def test_timeout_and_missed_slot_require_manual_resume():
    ex = executor(timeout=.1)
    candidate = response(request(ex))
    assert ex.advance(3, 100_000_000) is None
    assert ex.state == ExecutionState.WAITING
    assert not ex.install(candidate, 100_000_001, start_slot=4)
    ex.enable(pose(), episode_id='new')
    assert ex.install(response(request(ex, 4)), 133_333_334, start_slot=4)
    ex.advance(4, 133_333_334)
    assert ex.advance(6, 200_000_000) is None
    assert ex.state == ExecutionState.WAITING


def test_wrong_context_duplicate_and_bad_actions():
    ex = executor()
    req = request(ex)
    wrong = response(replace(req, observed_at_ns=1))
    assert not ex.install(wrong, 2, start_slot=1)
    assert ex.state == ExecutionState.WAITING
    ex.enable(pose(), episode_id='new')
    req = request(ex, 2)
    good = response(req)
    with pytest.raises(ValueError):
        ex.install(replace(good, sources=('actor',)*50), 100, start_slot=2)
    assert ex.state == ExecutionState.WAITING and not ex.queue
    assert not ex.install(good, 100, start_slot=2)
    ex.enable(pose(), episode_id='manual-resume')
    good = response(request(ex, 2))
    assert ex.install(good, 100, start_slot=2)
    assert not ex.install(good, 101, start_slot=2)
    assert len(ex.queue) == 50


def test_old_final_ack_does_not_finish_new_generation():
    ex = executor()
    ex.install(response(request(ex)), 1, start_slot=0)
    for slot in range(50):
        last = ex.advance(slot, ex.control_clock.deadline(slot))
    ex.invalidate(ExecutionState.FAULT)
    ex.acknowledge(last)
    assert ex.state == ExecutionState.FAULT


def test_contract_requires_supported_dimensions_and_timeout():
    with pytest.raises(ValueError):
        SerialExecutionContract(ActionSpec(arms=('left', 'right')), 'a', 'b', 'c')
    with pytest.raises(ValueError):
        executor(float('nan'))


def test_projection_failure_revokes_entire_chunk():
    ex = executor()
    candidate = response(request(ex))
    def project(action):
        if action[0] > .42:
            raise ValueError('invalid orientation')
        return action
    ex.project = project
    with pytest.raises(ValueError, match='orientation'):
        ex.install(candidate, 1, start_slot=0)
    assert not ex.queue and ex.inflight is None and ex.state == ExecutionState.WAITING
    assert not ex.install(candidate, 2, start_slot=0)
