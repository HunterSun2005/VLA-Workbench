"""Reference geometry and limits, without robot hardware or a control thread."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from vla.control.pose_reference import PoseReference


@pytest.mark.parametrize('linear,angular', [(.1,.45), (.012,.09), (.006,.03)])
def test_synchronized_line_shortest_rotation_and_endpoint(linear, angular):
    start = np.array([.35,-.11,.2])
    goal = np.array([.53,.02,.28])
    r0 = Rotation.from_euler('xyz', [.4,-.2,.9]).as_matrix()
    # A large rotation specified beyond pi must take the shortest arc.
    r1 = Rotation.from_rotvec(np.array([1.,2.,-1.])*2.).as_matrix() @ r0
    vector = Rotation.from_matrix(r1 @ r0.T).as_rotvec()
    plan = PoseReference(start,r0,goal,r1,linear_speed=linear,angular_speed=angular)
    previous_p,previous_r = start,r0
    increments = []
    while not plan.done:
        p,r = plan.step(1/30)
        s = (p-start)/(goal-start)
        np.testing.assert_allclose(s,plan.progress,atol=1e-13)
        np.testing.assert_allclose(Rotation.from_matrix(r @ r0.T).as_rotvec(),s[0]*vector,atol=1e-12)
        increment = np.linalg.norm(p-previous_p)
        increments.append(increment)
        assert increment <= linear/30+1e-12
        assert Rotation.from_matrix(r @ previous_r.T).magnitude() <= angular/30+1e-12
        previous_p,previous_r = p,r
    np.testing.assert_array_equal(p,goal)
    np.testing.assert_array_equal(r,r1)
    # Smooth departure and arrival, with no tolerance-sized terminal jump.
    assert increments[0] < max(increments)*.01
    assert increments[-1] < max(increments)*.01


@pytest.mark.parametrize('kind',['stationary','translation','rotation'])
def test_zero_length_components_share_finite_duration(kind):
    p0 = np.zeros(3); r0 = np.eye(3)
    p1 = [.1,0,0] if kind=='translation' else p0
    r1 = Rotation.from_euler('x',1).as_matrix() if kind=='rotation' else r0
    plan = PoseReference(p0,r0,p1,r1)
    assert np.isfinite(plan.duration) and plan.duration >= .3
    while not plan.done:
        p,r = plan.step(1/30)
    np.testing.assert_array_equal(p,p1)
    np.testing.assert_array_equal(r,r1)


def test_yaw_zero_degenerate_projection_is_deterministic_and_preserves_tilt():
    from vla.control.geometry import reset_tcp_heading, upright_tcp_rotation
    # Make tool X vertical, including a nonzero horizontal heading of tool Y.
    rotation=Rotation.from_euler('z',.8).as_matrix() @ Rotation.from_euler('y',np.pi/2).as_matrix()
    fixed=np.diag([1.,-1.,-1.])
    result=reset_tcp_heading(rotation,fixed)
    np.testing.assert_allclose(result[2,:],rotation[2,:],atol=1e-12)
    np.testing.assert_allclose(upright_tcp_rotation(result),fixed,atol=1e-12)
    np.testing.assert_array_equal(result,reset_tcp_heading(rotation,fixed))
