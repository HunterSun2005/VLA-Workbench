"""TCP tracking regressions with real KDL and an offline lag model; no ROS nodes."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.ros.test_native_replay import models, make_rig, Q, TOOL_P, TOOL_R
from vla.control.ros import ArmMeasurement


def test_sensor_noise_at_tracking_limit_does_not_fault_or_jump(models):
    c, _, _ = make_rig(models, 'tcp', ('left',), 'gamepad')
    s = c.servos['left']
    p, r = s.forward_tcp(Q, TOOL_P, TOOL_R)
    target = r @ Rotation.from_rotvec([.5, 0., 0.]).as_matrix()
    for _ in range(300):
        q = s.step(Q, p, target, TOOL_P, TOOL_R, .01)
    # A stopped arm reaches the following-error limiter, then a fresh sensor
    # sample differs by only 0.006 degrees. This must not be a hard fault.
    index = np.argmax(np.abs(q-Q))
    assert abs(q[index]-Q[index]) == pytest.approx(s.config.max_joint_speed*s.config.velocity_tracking_tau_s)
    measured = Q.copy()
    measured[index] -= np.sign(q[index]-Q[index]) * .0001
    previous = q.copy()
    q = s.step(measured, p, target, TOOL_P, TOOL_R, .01)
    assert np.max(np.abs(q-previous)) <= s.config.max_joint_speed*.01 + 1e-12
    assert abs(q[index]-measured[index]) <= abs(previous[index]-measured[index]) + 1e-12

    # A real excursion beyond the guard band still revokes the owner. The
    # diagnostic must identify the affected joint and measured/commanded values.
    measured[index] = q[index] - np.sign(q[index]-Q[index]) * .056
    c.observe('left', ArmMeasurement(measured, .45, TOOL_P, TOOL_R, c.clock()))
    c.accept(c.gamepad_target({'left': (p, target, .45)}, .01), source='gamepad',
             generation=c.generation, slot_id=0)
    with pytest.raises(RuntimeError, match='tracking error.*measured=') as error:
        c.tick(.01)
    assert s.model.joint_names[index] in str(error.value)
    assert c.mode == 'fault' and c.accepted is None and c.owner is None


@pytest.mark.parametrize('tool_length', [.174, .30])
@pytest.mark.parametrize('lag_s', [0., .08])
@pytest.mark.parametrize('space', ['tcp', 'joint'])
def test_pitch_rotates_about_tcp_without_translation(models, tool_length, lag_s, space):
    c, clock, out = make_rig(models, space, ('left',), 'gamepad')
    s = c.servos['left']
    tool_p = np.array([0., 0., tool_length])
    tool_r = Rotation.from_euler('z', -45, degrees=True).as_matrix()
    q = Q.copy()
    p, r = s.forward_tcp(q, tool_p, tool_r)
    rotation = r.copy()
    drift = []
    for tick in range(300):
        clock[0] += 10_000_000
        c.observe('left', ArmMeasurement(q, .45, tool_p, tool_r, clock[0]))
        rotation = rotation @ Rotation.from_rotvec([0., .003, 0.]).as_matrix()
        candidate = c.gamepad_target({'left': (p, rotation, .45)}, .01)
        if tick % 10 in (0, 4, 7):
            c.accept(candidate,
                     source='gamepad', generation=c.generation, slot_id=tick)
        c.tick(.01)
        q += min(1., .01/max(lag_s, .01)) * (out[-1][1]-q)
        actual_p, actual_r = s.forward_tcp(q, tool_p, tool_r)
        drift.append(np.linalg.norm(actual_p-p))
    # Damped IK is approximate: allow 1.5 mm over this 0.9-rad sweep,
    # including a tool longer than the field configuration and plant delay.
    assert max(drift) < .0015
    assert Rotation.from_matrix(r.T@actual_r).magnitude() > .6


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('speed', [.02, .04])
@pytest.mark.parametrize('space', ['tcp', 'joint'])
def test_xy_motion_with_tracking_delay_remains_directional(models, axis, speed, space):
    c, clock, out = make_rig(models, space, ('left',), 'gamepad')
    s = c.servos['left']
    q = Q.copy()
    p, r = s.forward_tcp(q, TOOL_P, TOOL_R)
    goal = p.copy()
    for tick in range(300):
        clock[0] += 10_000_000
        c.observe('left', ArmMeasurement(q, .45, TOOL_P, TOOL_R, clock[0]))
        goal[axis] += speed*.01
        candidate = c.gamepad_target({'left': (goal, r, .45)}, .01)
        if tick % 10 in (0, 4, 7):
            c.accept(candidate,
                     source='gamepad', generation=c.generation, slot_id=tick)
        c.tick(.01)
        q += .1*(out[-1][1]-q)  # 100 ms first-order tracking lag
    actual, actual_r = s.forward_tcp(q, TOOL_P, TOOL_R)
    delta = actual-p
    assert delta[axis] > speed*2.5
    assert np.linalg.norm(np.delete(delta, axis)) < .001
    assert Rotation.from_matrix(r.T@actual_r).magnitude() < .005
