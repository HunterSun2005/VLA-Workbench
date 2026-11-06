"""Production FR3/KDL replay. Run with system Python via scripts/test_ros.sh.

No rclpy initialization, devices, model weights or network are used here.
"""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest

pytest.importorskip('PyKDL', reason='requires isolated ROS Python 3.10 (scripts/test_ros.sh)')
import PyKDL
import pyarrow as pa
import pyarrow.parquet as pq
import xacro
from ament_index_python.packages import get_package_share_directory
from vla.actions import ActionSpec, matrix_to_rot6d
from vla.actions.transforms import DeltaActions, AbsoluteActions
from vla.control.ros import ControlCoordinator, ArmMeasurement
from vla.control.ros.cartesian_servo import CartesianServo
from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
from vla.datacollection.start_teleop.franka_kinematics import build_kinematic_model

Q = np.array([0., -.4, .1, -1.8, .1, 1.4, .5])
TOOL_P = np.array([.015, -.012, .12])
TOOL_R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])


@pytest.fixture(scope='module')
def models():
    share = Path(get_package_share_directory('franka_description'))
    result = {}
    for arm in ('left', 'right'):
        xml = xacro.process_file(str(share/'robots/fr3/fr3.urdf.xacro'),
                                 mappings={'hand': 'false', 'arm_prefix': arm, 'ros2_control': 'false'}).toxml()
        result[arm] = build_kinematic_model(xml, f'{arm}_fr3_link0', f'{arm}_fr3_link8', .05)
        assert result[arm].chain.getNrOfJoints() == 7
        assert np.all(Q > result[arm].lower_limits) and np.all(Q < result[arm].upper_limits)
    return result


def make_rig(models, space, arms, owner):
    config = load_rollout_config(Path(__file__).parents[2]/'src/vla/datacollection/franka_robotiq_rollout/configs/dual.yaml')
    clock, published = [1_000_000_000], []
    servos = {a: CartesianServo(models[a], config) for a in arms}
    c = ControlCoordinator(ActionSpec(space, arms), servos,
        {a: (np.array([-.8, -.8, -.1]), np.array([.8, .8, 1.2])) for a in arms},
        lambda a, q, g, *_: published.append((a, q.copy(), g)), lambda _: None, clock=lambda: clock[0])
    observe(c, clock, 0)
    reset(c, owner)
    return c, clock, published


def observe(c, clock, tick):
    clock[0] = 1_000_000_000 + tick * 10_000_000
    for i, arm in enumerate(c.spec.arms):
        q = Q + .003 * np.sin(tick*.09 + np.arange(7) + i)
        c.observe(arm, ArmMeasurement(q, .45, TOOL_P, TOOL_R, clock[0]))


def reset(c, owner):
    c.transition(owner)
    for arm in c.spec.arms:
        c.acknowledge(arm, c.generation, 0, Q, c.clock())


def target(c, tick, source):
    goals, grips, tcp = {}, {}, {}
    for i, arm in enumerate(c.spec.arms):
        q = c._measurements[arm].joints
        goals[arm] = q + .025*np.sin(tick*.12 + np.arange(7) + i)
        grips[arm] = .2 if tick % 40 < 20 else .7
        p, r = c.servos[arm].forward_tcp(goals[arm], TOOL_P, TOOL_R)
        tcp[arm] = p, r, grips[arm]
    return c.gello_target(goals, grips) if source == 'gello' else c.gamepad_target(tcp, 1/30)


def anchor(c):
    parts = []
    for arm in c.spec.arms:
        m = c._measurements[arm]
        p, r = c.servos[arm].forward_tcp(m.joints, TOOL_P, TOOL_R)
        parts.append(np.r_[m.joints if c.spec.space == 'joint' else np.r_[p, matrix_to_rot6d(r)], m.gripper])
    return np.concatenate(parts)


@pytest.mark.parametrize('space', ['joint', 'tcp'])
@pytest.mark.parametrize('arms', [('left',), ('left', 'right')])
@pytest.mark.parametrize('source', ['gello', 'gamepad'])
def test_production_teaching_parquet_replay(models, tmp_path, space, arms, source):
    c, clock, expected = make_rig(models, space, arms, 'human')
    records = []
    for tick in range(150):
        observe(c, clock, tick)
        if tick == 90:
            reset(c, 'human')
        if tick % 10 in (0, 4, 7):
            receipt = c.accept(target(c, tick, source), source='human', generation=c.generation)
            assert receipt is not None
            records.append((tick, receipt.target.copy(), anchor(c)))
        c.tick(.01)
    path = tmp_path/'labels.parquet'
    pq.write_table(pa.table({'tick': [r[0] for r in records],
        'action': pa.array([r[1].astype(np.float32).tolist() for r in records], type=pa.list_(pa.float32(), c.spec.dim)),
        'anchor': pa.array([r[2].tolist() for r in records], type=pa.list_(pa.float64(), c.spec.dim))}), path)
    rows = {r['tick']: r for r in pq.read_table(path).to_pylist()}
    for representation in ('absolute', 'chunk_relative'):
        replay, replay_clock, actual = make_rig(models, space, arms, 'policy')
        training_spec = replace(c.spec, representation=representation)
        # Five-action chunks: every future action uses the same measured origin.
        decoded = {}
        ticks = list(rows)
        for start in range(0, len(ticks), 5):
            block = ticks[start:start+5]
            original = np.array([rows[t]['action'] for t in block])
            origin = np.array(rows[block[0]]['anchor'])
            model_values = DeltaActions(training_spec)(original, origin).astype(np.float32)
            absolute = AbsoluteActions(training_spec)(model_values, origin)
            decoded.update(zip(block, absolute))
        for tick in range(150):
            observe(replay, replay_clock, tick)
            if tick == 90:
                reset(replay, 'policy')
            if tick in rows:
                replay.accept(decoded[tick], source='policy', generation=replay.generation)
            replay.tick(.01)
        assert len(actual) == len(expected) == 150*len(arms)
        np.testing.assert_allclose([x[1] for x in actual], [x[1] for x in expected], atol=1e-5, rtol=0)
        np.testing.assert_allclose([x[2] for x in actual], [x[2] for x in expected], atol=1e-6, rtol=0)
    assert 'torch' not in sys.modules


def test_fk_tool_composition_clipping_and_finite_ik(models):
    c, clock, published = make_rig(models, 'tcp', ('left',), 'human')
    servo = c.servos['left']
    p, r = servo.forward(Q)
    tcp_p, tcp_r = servo.forward_tcp(Q, TOOL_P, TOOL_R)
    np.testing.assert_allclose(tcp_p, p+r@TOOL_P)
    np.testing.assert_allclose(tcp_r, r@TOOL_R)
    receipt = c.accept(np.r_[[2., 0., .4], matrix_to_rot6d(tcp_r), .5], source='human', generation=c.generation)
    assert receipt.target[0] == .8
    c.tick()
    q = published[-1][1]
    assert np.isfinite(q).all()
    assert np.all(q >= models['left'].lower_limits) and np.all(q <= models['left'].upper_limits)
    joint, _, out = make_rig(models, 'joint', ('left',), 'human')
    joint.accept(np.r_[np.full(7, 100.), .5], source='human', generation=joint.generation)
    joint.tick()
    np.testing.assert_array_equal(out[-1][1], models['left'].upper_limits)


def test_negative_control_detects_changed_history_and_wrong_anchor(models):
    a, ca, out_a = make_rig(models, 'tcp', ('left',), 'human')
    b, cb, out_b = make_rig(models, 'tcp', ('left',), 'policy')
    for tick in range(30):
        observe(a, ca, tick); observe(b, cb, tick)
        goal = target(a, tick, 'gello')
        if tick % 10 in (0, 4, 7):
            a.accept(goal, source='human', generation=a.generation)
            b.accept(goal, source='policy', generation=b.generation)
        a.tick(); b.tick()
    assert np.linalg.norm(a.servos['left']._last_linear_velocity) > 1e-6
    b.servos['left'].reset_tracking()
    a.tick(); b.tick()
    assert np.max(np.abs(out_a[-1][1]-out_b[-1][1])) > 1e-5
    spec = replace(a.spec, representation='chunk_relative')
    origin = anchor(a)
    delta = DeltaActions(spec)(a.accepted.target, origin)
    wrong = origin.copy(); wrong[0] += .03
    restored = AbsoluteActions(spec)(delta, wrong)
    assert abs(restored[0]-a.accepted.target[0]) > .02
