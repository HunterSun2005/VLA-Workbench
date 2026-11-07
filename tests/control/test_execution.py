from dataclasses import replace
import socket
import numpy as np
import pytest
from vla.actions import ActionSpec
from vla.control.execution import ExecutionContract, Executor, ExecutionState, PlanResponse, FixedDeadline
from vla_rollout_protocol import send_packet, recv_packet


def executor(stride=1, horizon=12):
    contract = ExecutionContract(ActionSpec('joint'), {'vla': horizon, 'actor': horizon},
                                 'control', 'urdf', 'tcp', frame_stride=stride)
    e = Executor(contract, session_id='session', model_version='model', epoch_ns=0)
    e.enable(np.r_[np.zeros(7), .5], episode_id='episode')
    return e


def request(e, slot=0):
    return e.request(slot=slot, anchor=e.hold, observation_id=slot,
                     observed_at_ns=e.control_clock.deadline(slot), policy='actor')


def response(req, source='actor'):
    actions = np.tile(np.r_[np.ones(7)*.1, .5], (req.horizon, 1))
    return PlanResponse(req, actions, (source,)*req.horizon, (1,)*req.horizon, actions)


@pytest.mark.parametrize('stride', [1, 2, 3])
def test_frozen_prefix_actor_lock_and_manual_resume(stride):
    e = executor(stride)
    req = request(e)
    res = response(req)
    assert req.start_idx == e.contract.freeze_steps
    assert e.install(res, 1)
    assert not e.install(res, 2)
    trace = [e.advance(s, e.control_clock.deadline(s)) for s in range(req.horizon*stride)]
    assert all(r.source == 'system_hold' for r in trace[:req.replace_slot])
    assert all(r.source == 'actor' for r in trace[req.replace_slot:])
    np.testing.assert_array_equal(trace[0].accepted, e.hold)
    assert e.advance(req.horizon*stride, e.control_clock.deadline(req.horizon*stride)) is None
    assert e.state == ExecutionState.WAITING
    assert not e.install(res, 3)
    assert request(e, req.horizon*stride) is None


@pytest.mark.parametrize('invalid', ['late', 'generation', 'model', 'plan', 'context', 'takeover'])
def test_invalid_results_cannot_install(invalid):
    e = executor()
    req = request(e)
    now = 1
    if invalid == 'late':
        now = req.deadline_ns
    elif invalid == 'generation':
        req = replace(req, generation=req.generation+1)
    elif invalid == 'model':
        req = replace(req, model_version='other')
    elif invalid == 'plan':
        req = replace(req, plan_version=req.plan_version+1)
    elif invalid == 'context':
        req = replace(req, anchor=req.anchor+.01)
    else:
        e.invalidate(ExecutionState.HUMAN, 'takeover')
    assert not e.install(response(req), now)


def test_actor_next_suffix_connects_at_block_end():
    e = executor()
    req = request(e)
    assert e.install(response(req), 1)
    for slot in range(6):
        e.advance(slot, e.control_clock.deadline(slot))
    assert request(e, 6) is not None
    next_req = e.inflight
    assert next_req.replace_slot == 12
    assert e.install(response(next_req, 'vla'), e.control_clock.deadline(6))
    assert all(e.advance(s, e.control_clock.deadline(s)).source == 'actor' for s in range(6, 12))
    assert e.advance(12, e.control_clock.deadline(12)).source == 'vla'


def test_deadline_skip_idempotence_horizon_and_wire():
    with pytest.raises(ValueError, match='H >= 2d'):
        executor(horizon=9)
    clock = FixedDeadline(0, 30)
    assert clock.poll(0) == 0
    assert clock.poll(100_000_000) == 3
    assert clock.poll(100_000_000) is None
    e = executor()
    req = request(e)
    res = response(req)
    a, b = socket.socketpair()
    try:
        send_packet(a, *res.packet())
        header, arrays = recv_packet(b)
        decoded = PlanResponse.from_packet(header, arrays)
        assert decoded.request.identity == req.identity
        np.testing.assert_array_equal(decoded.actions, res.actions)
    finally:
        a.close(); b.close()
    assert e.install(res, 1)
    row = e.advance(0, 2)
    assert e.advance(0, 3) is row
    with pytest.raises(ValueError):
        row.accepted.setflags(write=True)


def test_auto_mode_freeze_fits_the_shortest_possible_response():
    e = Executor(ExecutionContract(ActionSpec('joint'), {'vla': 50, 'rlt': 10, 'auto': 50},
                                  'c', 'k', 't'), session_id='s', model_version='m', epoch_ns=0)
    e.enable(np.r_[np.zeros(7),.5], episode_id='e')
    def auto(slot):
        return e.request(slot=slot, anchor=e.hold, observation_id=slot,
                         observed_at_ns=e.control_clock.deadline(slot), policy='auto')
    first = auto(0)
    full = response(first)
    short = replace(full, actions=full.actions[:10], sources=full.sources[:10],
                    phases=full.phases[:10], vla_reference=full.vla_reference[:10])
    assert e.install(short, 1)
    e.advance(0,0)
    assert auto(1) is None  # d=9 cannot fit a possible H=10 Actor response.
    for i in range(1,5):
        e.advance(i,e.control_clock.deadline(i))
    follow = auto(5)
    assert follow.start_idx == 5 and follow.replace_slot == 10


@pytest.mark.parametrize('stride', [1,3])
def test_deterministic_replay_of_clock_candidate_and_manual_events(stride):
    def replay():
        e = executor(stride)
        first = request(e)
        candidate = response(first)
        trace = []
        assert e.install(candidate, first.deadline_ns-1)
        assert not e.install(candidate, first.deadline_ns-1)
        # A missed timer deadline never submits historical actions in a burst.
        slots = [0,1,*range(4,first.horizon*stride+2)]
        for slot in slots:
            record = e.advance(slot,e.control_clock.deadline(slot))
            if record is not None:
                trace.append((record.slot_id,record.generation,record.source,record.phase,record.accepted.tolist()))
        assert e.state == ExecutionState.WAITING
        assert not e.install(candidate,e.control_clock.deadline(slots[-1]))
        assert request(e,slots[-1]+1) is None
        e.enable(e.hold,episode_id='episode')
        resumed = request(e,slots[-1]+2)
        assert resumed.generation > first.generation
        from vla.datacollection.episode_journal import jsonable
        return trace,jsonable(e.events)
    first = replay()
    assert first == replay()
    assert 2 not in [r[0] for r in first[0]] and 3 not in [r[0] for r in first[0]]
