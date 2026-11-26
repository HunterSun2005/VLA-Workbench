from dataclasses import asdict
from types import SimpleNamespace as NS
import time
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from vla.control.franky.velocity_control import (
    VelocityControl, tracking_twist, smooth_pose, validate_context)
from vla.control.franky.worker import FrankyArm

IDENTITY = np.array([0., 0, 0, 0, 0, 0, 1])
CONTEXT = {'source': 'policy', 'dt': 1/30}


def fake_arm():
    class Robot:
        is_in_control = False
        def __init__(self):
            self.moves = []
            self.state = NS(time=NS(to_sec=lambda: time.monotonic()), O_T_EE=np.eye(4), F_T_EE=np.eye(4), q=np.zeros(7), dq=np.zeros(7))
        def poll_motion(self): pass
        def move(self, motion, **kw):
            self.moves.append((motion, kw))
            self.is_in_control = motion[0] != 'stop'
        def join_motion(self, timeout=None): return True
        def stop(self): self.is_in_control = False
        def recover_from_errors(self): pass
    arm = FrankyArm('left', 'mock')
    arm.robot = Robot()
    arm.franky = NS(Twist=lambda v,w: np.r_[v,w], RobotVelocity=lambda x:x,
                   Duration=lambda x:x, CartesianVelocityMotion=lambda v,*args:('move',v),
                   CartesianVelocityStopMotion=lambda:('stop',))
    return arm


@pytest.mark.parametrize('vector', [[.0001,0,0],[.04,-.03,.02],[1.,2.,-3.]])
def test_policy_twist_matches_old_unsmoothed_limit_and_tau(vector):
    target = IDENTITY.copy(); target[:3] = vector
    result = tracking_twist(IDENTITY, target, VelocityControl())
    distance = np.linalg.norm(vector)
    expected = np.asarray(vector)*min(1., .05*.15/distance)/.15
    if distance <= .0005: expected[:] = 0
    np.testing.assert_allclose(result[:3], expected)
    assert np.linalg.norm(result[:3]) <= .05+1e-12


def test_rotation_shortest_path_and_smoothing_are_on_so3():
    a = IDENTITY.copy(); b = IDENTITY.copy()
    a[3:] = Rotation.from_euler('z',179,degrees=True).as_quat()
    b[3:] = Rotation.from_euler('z',-179,degrees=True).as_quat()
    result = tracking_twist(a,b,VelocityControl())
    assert 0 < result[5] <= .2
    midpoint = smooth_pose(a,b,.5)
    assert Rotation.from_quat(midpoint[3:]).magnitude() == pytest.approx(np.pi)
    b[3:] *= -1
    np.testing.assert_allclose(tracking_twist(a,b,VelocityControl()),result)


@pytest.mark.parametrize('distance,expected',[(.0004,0.),(.0006,.004),(.002,.002/.15),(.1,.05)])
def test_common_braking_reaches_saved_pose_tolerance(distance, expected):
    target = IDENTITY.copy(); target[0] = distance
    assert tracking_twist(IDENTITY, target, VelocityControl())[0] == pytest.approx(expected)


def test_reset_is_nonmoving_ack_requires_first_sdk_submission_and_stop_clears():
    arm=fake_arm()
    with pytest.raises(RuntimeError,match='reset'):arm.set_target(IDENTITY.tolist(),1,1,CONTEXT)
    arm.reset(1)
    assert arm.servo_ready and not arm.robot.moves
    target=IDENTITY.copy();target[0]=.01
    assert arm.set_target(target.tolist(),1,1,CONTEXT) is None
    assert not arm.robot.moves
    event=arm.servo_once()
    assert event['sequence']==1 and event['applied_at_ns']>=event['observed_at_ns']
    assert len(arm.robot.moves)==1
    for _ in range(5):arm.servo_once()
    assert len(arm.robot.moves)==6 and arm.target_sequence==1
    assert arm.robot.moves[0][1]==dict(asynchronous=True,limit_rate=False,cutoff_frequency=100.)
    arm.stop()
    assert not arm.servo_ready and arm.target_pose is None
    assert arm.servo_once() is None


def test_defaults_and_strict_configuration():
    values=asdict(VelocityControl())
    assert not values['target_smoothing_enabled'] and not values['velocity_smoothing_enabled']
    for bad in ({'translational_stiffness':1000},{'tracking_integral_gain':.1},
                {'max_linear_speed':True},{'target_smoothing_enabled':1},
                {'max_linear_speed':.17},{'tracking_tau_s':float('nan')}):
        with pytest.raises(ValueError):VelocityControl.from_mapping(bad)
    with pytest.raises(ValueError):validate_context({'source':'policy','dt':.1})


@pytest.mark.parametrize('target_smoothing,velocity_smoothing', [(False,False),(True,False),(False,True),(True,True)])
def test_all_sources_have_identical_references_and_twists_with_smoothing(monkeypatch, target_smoothing, velocity_smoothing):
    import vla.control.franky.worker as module
    clock = [1_000_000_000]
    monkeypatch.setattr(module.time, 'monotonic_ns', lambda:clock[0])
    traces = []
    for source in ('gamepad','gello','policy','pose'):
        arm = fake_arm()
        arm.velocity_control = VelocityControl(target_smoothing_enabled=target_smoothing,
                                               velocity_smoothing_enabled=velocity_smoothing)
        arm.reset(1)
        trace = []
        for slot, (x, angle) in enumerate([(0.,0.),(.005,.04),(.02,.2),(.0004,.003),(0.,0.)]):
            target = IDENTITY.copy(); target[0] = x
            target[3:] = Rotation.from_rotvec([0, angle, 0]).as_quat()
            clock[0] = 1_000_000_000 + slot * 30_000_000
            arm.set_target(target.tolist(),1,slot,{'source':source,'dt':1/30})
            if slot == 1:
                assert arm.target_pose[0] == pytest.approx(.005*(.75 if target_smoothing else 1.))
            for tick in range(3):
                clock[0] = 1_000_000_000 + slot*30_000_000 + tick*10_000_000
                event = arm.servo_once()
                assert event['source'] == source
                np.testing.assert_array_equal(event['absolute_target'], target)
                trace.append(np.r_[event['reference_tcp'],event['twist']])
            if slot == 1:
                expected = tracking_twist(IDENTITY, arm.target_pose, arm.velocity_control)
                if velocity_smoothing:
                    # Three 10 ms EMA updates from an initially zero velocity.
                    expected *= 1 - .3**.9
                np.testing.assert_allclose(event['twist'], expected)
        traces.append(np.array(trace))
        arm.stop()
        assert arm.target_pose is None and not arm.servo_ready
    for trace in traces[1:]:
        np.testing.assert_array_equal(trace, traces[0])


def test_source_context_is_provenance_only_and_rejects_old_control_masks():
    for source in ('gamepad','gello','policy','pose'):
        assert validate_context({'source':source,'dt':1/30})['source'] == source
    with pytest.raises(ValueError, match='only source and dt'):
        validate_context({'source':'gamepad','dt':1/30,'linear_active':[False]*3})
