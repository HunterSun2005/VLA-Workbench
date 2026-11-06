from types import SimpleNamespace
import numpy as np
import pytest
from vla.actions import ActionSpec
from vla.control.ros import ControlCoordinator, ArmMeasurement


class Servo:
    def __init__(self):
        self.model = SimpleNamespace(lower_limits=np.full(7, -2.), upper_limits=np.full(7, 2.))
        self.calls = 0
        self.resets = 0
    def reset_tracking(self):
        self.resets += 1
    def forward_tcp(self, joints, *tool):
        return joints[:3].copy(), np.eye(3)
    def step(self, joints, position, rotation, *rest):
        self.calls += 1
        return np.r_[position, np.zeros(4)]


def setup(space='tcp', arms=('left',)):
    clock = [1000000000]
    servos = {a: Servo() for a in arms}
    published, resets = [], []
    c = ControlCoordinator(ActionSpec(space, arms), servos,
                           {a: (np.full(3, -.5), np.full(3, .5)) for a in arms},
                           lambda *args: published.append(args), resets.append, clock=lambda: clock[0])
    for arm in arms:
        c.observe(arm, ArmMeasurement(np.zeros(7), .4, np.zeros(3), np.eye(3), clock[0]))
    c.transition('human')
    for arm in arms:
        c.acknowledge(arm, c.generation, 0, np.zeros(7), clock[0])
    return c, clock, published, resets


@pytest.mark.parametrize('space', ['joint', 'tcp'])
@pytest.mark.parametrize('arms', [('left',), ('left', 'right')])
def test_native_teaching_replay_same_joints(space, arms):
    c, clock, published, _ = setup(space, arms)
    targets = c.gello_target({a: np.r_[.2, .3, .4, np.zeros(4)] for a in arms}, {a: .38 for a in arms})
    accepted = c.accept(targets, source='human', generation=c.generation, slot_id=0)
    c.tick()
    recorded = accepted.target.copy()
    teaching = [q.copy() for _, q, *_ in published]
    c.transition('policy')
    for arm in arms:
        c.acknowledge(arm, c.generation, 0, np.zeros(7), clock[0])
    c.accept(recorded, source='policy', generation=c.generation, slot_id=0)
    c.tick()
    for i in range(len(arms)):
        np.testing.assert_array_equal(published[len(arms)+i][1], teaching[i])
    if space == 'joint':
        assert all(s.calls == 0 for s in c.servos.values())
    assert all(p[2] == .38 for p in published)


def test_takeover_reset_timeout_stale_generation_duplicate_and_drop():
    c, clock, published, resets = setup('joint', ('left', 'right'))
    old = c.generation
    target = np.tile(np.r_[np.ones(7)*3, .42], 2)
    command = c.accept(target, source='human', generation=old, slot_id=0)
    np.testing.assert_array_equal(command.target[:7], np.ones(7)*2)
    c.tick()
    clock[0] += 34000000
    assert c.accept(target, source='human', generation=old, slot_id=0) is command
    with pytest.raises(ValueError, match='out of order'):
        c.accept(target, source='human', generation=old, slot_id=-1)
    c.transition('policy')
    assert c.accepted is None
    assert not c.acknowledge('left', old, 0, np.zeros(7), clock[0])
    with pytest.raises(RuntimeError, match='revoked'):
        c.accept(target, source='human', generation=old, slot_id=0)
    c.acknowledge('left', c.generation, 0, np.zeros(7), clock[0])
    assert c.mode == 'resetting'
    clock[0] += 1000000001
    with pytest.raises(RuntimeError, match='timed out'):
        c.tick()
    assert c.mode == 'fault'
    c.acknowledge('right', c.generation, 0, np.zeros(7), clock[0])
    assert c.mode == 'fault'


def test_stale_state_and_publish_failure_reset_all_arms():
    c, clock, published, resets = setup('joint')
    c.accept(np.r_[np.zeros(7), .3], source='human', generation=c.generation, slot_id=0)
    clock[0] += 600000000
    with pytest.raises(RuntimeError, match='stale'):
        c.tick()
    assert c.mode == 'fault' and c.accepted is None and len(resets) == 2
    c, clock, published, resets = setup('joint')
    c.accept(np.r_[np.zeros(7), .3], source='human', generation=c.generation, slot_id=0)
    def broken(*args):
        raise RuntimeError('publisher disconnected')
    c.publish = broken
    with pytest.raises(RuntimeError, match='disconnected'):
        c.tick()
    assert c.mode == 'fault'
