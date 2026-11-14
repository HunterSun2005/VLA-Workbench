from dataclasses import asdict
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from vla.control.franky.velocity_control import (
    VelocityControl, tracking_twist, smooth_pose, return_twist, validate_context)
from vla.control.franky.worker import FrankyArm

IDENTITY = np.array([0., 0, 0, 0, 0, 0, 1])
CONTEXT = {'source': 'policy', 'dt': 1/30}


def fake_arm():
    class Robot:
        is_in_control = False
        def __init__(self):
            self.moves = []
            self.state = NS(O_T_EE=np.eye(4), F_T_EE=np.eye(4), q=np.zeros(7), dq=np.zeros(7))
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
    result = tracking_twist(IDENTITY, target, CONTEXT, VelocityControl())
    distance = np.linalg.norm(vector)
    expected = np.asarray(vector)*min(1., .05*.15/distance)/.15
    if distance <= .0008: expected[:] = 0
    np.testing.assert_allclose(result[:3], expected)
    assert np.linalg.norm(result[:3]) <= .05+1e-12


def test_rotation_shortest_path_and_smoothing_are_on_so3():
    a = IDENTITY.copy(); b = IDENTITY.copy()
    a[3:] = Rotation.from_euler('z',179,degrees=True).as_quat()
    b[3:] = Rotation.from_euler('z',-179,degrees=True).as_quat()
    result = tracking_twist(a,b,CONTEXT,VelocityControl())
    assert 0 < result[5] <= .2
    midpoint = smooth_pose(a,b,.5)
    assert Rotation.from_quat(midpoint[3:]).magnitude() == pytest.approx(np.pi)
    b[3:] *= -1
    np.testing.assert_allclose(tracking_twist(a,b,CONTEXT,VelocityControl()),result)


def test_manual_release_never_chases_accumulated_error():
    target = IDENTITY.copy(); target[:3]=[.01,.02,.03]
    target[3:] = Rotation.from_rotvec([.1,.2,.3]).as_quat()
    context = dict(CONTEXT, source='gamepad', linear_active=[True,False,False],
                   angular_active=[False,False,True], rotation_basis=np.eye(3).tolist())
    result = tracking_twist(IDENTITY,target,validate_context(context),VelocityControl())
    assert result[1] == result[2] == result[3] == result[4] == 0
    assert result[5] == pytest.approx(.3)
    context.update(linear_active=[False]*3,angular_active=[False]*3)
    np.testing.assert_array_equal(tracking_twist(IDENTITY,target,context,VelocityControl()),np.zeros(6))


def test_released_axis_error_does_not_slow_active_axis():
    target = IDENTITY.copy(); target[:3] = [.001, .1, -.1]
    context = dict(CONTEXT, source='gamepad', linear_active=[True,False,False],
                   angular_active=[False]*3, rotation_basis=np.eye(3).tolist())
    result = tracking_twist(IDENTITY, target, context, VelocityControl())
    np.testing.assert_allclose(result, [.03, 0, 0, 0, 0, 0])


@pytest.mark.parametrize('distance,cap',[(.1,.03),(.01,.005),(.002,.001)])
def test_saved_return_distance_braking(distance,cap):
    target=IDENTITY.copy(); target[0]=distance
    assert return_twist(IDENTITY,target)[0] == pytest.approx(cap)
    np.testing.assert_array_equal(return_twist(target,target),np.zeros(6))


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
def test_smoothing_switches_are_independent_and_manual_input_bypasses_them(target_smoothing, velocity_smoothing):
    arm = fake_arm()
    arm.velocity_control = VelocityControl(target_smoothing_enabled=target_smoothing,
                                           velocity_smoothing_enabled=velocity_smoothing)
    arm.reset(1)
    arm.set_target(IDENTITY.tolist(), 1, 1, CONTEXT)
    arm.servo_once()
    target = IDENTITY.copy(); target[0] = .005
    arm.set_target(target.tolist(), 1, 2, CONTEXT)
    assert arm.target_pose[0] == pytest.approx(.005 * (.75 if target_smoothing else 1.))
    expected = tracking_twist(IDENTITY, arm.target_pose, CONTEXT, arm.velocity_control)
    # A deterministic 30 Hz equivalent interval also checks the EMA exponent.
    arm.last_servo_ns = None
    event = arm.servo_once()
    np.testing.assert_allclose(event['twist'], expected * (.7 if velocity_smoothing else 1.))
    manual = dict(CONTEXT, source='gamepad', linear_active=[True,False,False],
                  angular_active=[False]*3, rotation_basis=np.eye(3).tolist())
    arm.set_target(target.tolist(), 1, 3, manual)
    np.testing.assert_allclose(arm.target_pose, target)
    np.testing.assert_allclose(arm.servo_once()['twist'], [.15,0,0,0,0,0])
