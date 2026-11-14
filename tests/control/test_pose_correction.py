from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vla.control.franky.pose_correction import PoseErrorCompensator
from vla.control.franky.worker import Compliance
from vla.control.franky.worker import FrankyArm


DT = 1/30
IDENTITY = np.array([0., 0., 0., 0., 0., 0., 1.])


def plant(gain):
    """First-order impedance plant with a constant load-induced equilibrium bias."""
    c = Compliance(tracking_integral_gain=gain)
    compensator = PoseErrorCompensator()
    goal = np.r_[[.4, -.2, .3], Rotation.from_euler('xyz', [.3, -.2, .5]).as_quat()]
    bias_p = np.array([.004, -.003, .002])
    bias_r = Rotation.from_rotvec([.04, -.03, .02])
    measured = goal.copy()
    measured[:3] -= bias_p
    measured[3:] = (bias_r.inv() * Rotation.from_quat(goal[3:])).as_quat()
    for _ in range(30*30):
        reference = compensator.update(goal, measured, DT, c)
        equilibrium_p = reference[:3]-bias_p
        equilibrium_r = bias_r.inv() * Rotation.from_quat(reference[3:])
        measured[:3] += .15*(equilibrium_p-measured[:3])
        delta = (equilibrium_r * Rotation.from_quat(measured[3:]).inv()).as_rotvec()
        measured[3:] = (Rotation.from_rotvec(.15*delta) * Rotation.from_quat(measured[3:])).as_quat()
    return np.linalg.norm(goal[:3]-measured[:3]), (
        Rotation.from_quat(goal[3:]) * Rotation.from_quat(measured[3:]).inv()).magnitude()


def test_static_load_bias_converges_with_absolute_pose_input():
    uncorrected, corrected = plant(0), plant(.5)
    assert uncorrected[0] > .005 and uncorrected[1] > np.deg2rad(3)
    assert corrected[0] < .0006 and corrected[1] < np.deg2rad(.35)


def test_disabled_is_exact_and_does_not_mutate_inputs():
    goal = np.r_[[.1, .2, .3], Rotation.from_rotvec([.3, -.8, .2]).as_quat()]
    original = goal.copy()
    comp = PoseErrorCompensator()
    for _ in range(50):
        np.testing.assert_array_equal(comp.update(goal, IDENTITY, DT, Compliance()), original)
    np.testing.assert_array_equal(goal, original)
    assert not comp.offset.any()


def test_unreachable_target_has_bounded_rate_and_offset_and_unwinds_on_new_target():
    c = Compliance(tracking_integral_gain=1., translational_clip=.05, rotational_clip=.3,
                   tracking_max_offset_m=.001, tracking_max_offset_rad=.01)
    comp = PoseErrorCompensator()
    goal = np.r_[[.01, .01, .01], Rotation.from_rotvec([.1, -.1, .1]).as_quat()]
    for _ in range(900):
        previous = comp.offset.copy()
        comp.update(goal, IDENTITY, DT, c)
        assert np.linalg.norm(comp.offset[:3]-previous[:3]) <= c.tracking_max_linear_rate*DT+1e-12
        assert np.linalg.norm(comp.offset[3:]-previous[3:]) <= c.tracking_max_angular_rate*DT+1e-12
        assert np.linalg.norm(comp.offset[:3]) <= .001+1e-12
        assert np.linalg.norm(comp.offset[3:]) <= .01+1e-12
    assert comp.status == 'saturated'
    before = comp.offset.copy()
    # Sign reversal never reuses a growing integral from the old target.
    comp.update(IDENTITY, goal, DT, c)
    assert np.linalg.norm(comp.offset[:3]) < np.linalg.norm(before[:3])
    for i in range(100):
        moving = IDENTITY.copy(); moving[0] = i*.0001
        comp.update(moving, moving, DT, c)
    np.testing.assert_allclose(comp.offset, 0., atol=1e-12)
    comp.reset()
    assert comp.previous is None and comp.status == 'disabled'


def test_inner_clip_prevents_windup_and_large_errors_do_not_integrate():
    c = Compliance(tracking_integral_gain=.5)
    comp = PoseErrorCompensator()
    goal = np.r_[[.02, 0, 0], Rotation.from_rotvec([.2, 0, 0]).as_quat()]
    for _ in range(300):
        comp.update(goal, IDENTITY, DT, c)
    np.testing.assert_array_equal(comp.offset, np.zeros(6))
    assert comp.status == 'saturated'
    goal[0] = .04
    comp.update(goal, IDENTITY, DT, c)
    assert comp.status == 'outside_capture_range'
    np.testing.assert_array_equal(comp.offset, np.zeros(6))


def test_disable_unwinds_and_uses_so3_shortest_path():
    c = Compliance(tracking_integral_gain=.5)
    goal = IDENTITY.copy(); goal[0] = .004
    goal[3:] = -Rotation.from_rotvec([0, 0, .04]).as_quat()
    comp = PoseErrorCompensator()
    for _ in range(45):
        comp.update(goal, IDENTITY, DT, c)
    assert comp.offset[0] > 0 and comp.offset[5] > 0
    for _ in range(90):
        comp.update(goal, IDENTITY, DT, replace(c, tracking_integral_gain=0.))
    np.testing.assert_array_equal(comp.offset, np.zeros(6))
    np.testing.assert_array_equal(comp.update(goal, IDENTITY, DT, Compliance()), goal)


@pytest.mark.parametrize('values', [
    {'tracking_integral_gain': True}, {'tracking_integral_gain': float('nan')},
    {'tracking_integral_gain': -1}, {'tracking_integral_gain': 1.01},
    {'tracking_max_offset_m': .021}, {'tracking_max_offset_rad': .36},
    {'tracking_max_linear_rate': .011}, {'tracking_max_angular_rate': .11},
    {'tracking_max_linear_rate': 0.},
])
def test_invalid_tracking_parameters_rejected(values):
    with pytest.raises(ValueError):
        Compliance.from_mapping(values)


@pytest.fixture
def simulated_arm(monkeypatch):
    from vla.control.franky import worker
    clock = [1.]
    monkeypatch.setattr(worker.time, 'monotonic', lambda: clock[0])

    class Affine:
        def __init__(self, matrix):
            self.translation = np.asarray(matrix)[:3, 3]
            self.quaternion = Rotation.from_matrix(np.asarray(matrix)[:3, :3]).as_quat()

    class Tracker:
        def __init__(self, *args, **kwargs): self.is_running = True
        def set_target(self, pose): self.reference = pose
        def stop(self): self.is_running = False

    raw = SimpleNamespace(O_T_EE=Affine(np.eye(4)), F_T_EE=Affine(np.eye(4)), q=np.zeros(7),
                          dq=np.zeros(7), time=SimpleNamespace(to_sec=lambda: clock[0]))
    robot = SimpleNamespace(state=raw, join_motion=lambda: None, recover_from_errors=lambda: None)
    arm = FrankyArm('left', 'mock', compliance=Compliance(tracking_integral_gain=.5))
    arm.franky = SimpleNamespace(Affine=Affine, CartesianImpedanceTracker=Tracker)
    arm.robot = robot
    arm.reset(1)
    return arm, clock


def test_accepted_slots_state_freshness_parameter_rebuild_and_failed_stop_clear_correction(simulated_arm):
    arm, clock = simulated_arm
    goal = IDENTITY.copy(); goal[0] = .004
    for i in range(40):
        clock[0] += DT
        arm.set_target(goal, 1, i)
    assert arm.correction.offset[0] > 0
    np.testing.assert_array_equal(arm.nominal_pose, goal)
    # Long scheduling gaps never integrate multiple missed control slots.
    previous = arm.correction.offset.copy()
    clock[0] += 2.
    arm.set_target(goal, 1, 41)
    assert np.linalg.norm(arm.correction.offset[:3]-previous[:3]) <= .002*DT+1e-12
    old = arm.tracker
    arm.update_compliance({'tracking_integral_gain': .3})
    assert not old.is_running and arm.tracker is not old
    assert not arm.correction.offset.any()
    # Stalled feedback fails before another correction/target can be submitted.
    arm.robot.state.time = SimpleNamespace(to_sec=lambda: clock[0])
    arm._state()
    stamp = clock[0]
    arm.robot.state.time = SimpleNamespace(to_sec=lambda: stamp)
    clock[0] += 1.
    with pytest.raises(RuntimeError, match='timestamp stalled'):
        arm.set_target(goal, 1, 42)
    arm.correction.offset[0] = .001
    def failed_stop(): raise RuntimeError('injected stop failure')
    arm.tracker.stop = failed_stop
    with pytest.raises(RuntimeError, match='injected stop failure'):
        arm.stop()
    assert arm.tracker is None and arm.target_pose is None and arm.nominal_pose is None
    assert not arm.correction.offset.any()


def test_new_paths_respect_final_reference_step_and_correction_bounds(simulated_arm):
    arm, clock = simulated_arm
    arm.compliance = replace(arm.compliance, max_step=.0001, max_step_rad=.001)
    goal = IDENTITY.copy(); goal[:3] = [.003, -.002, .002]
    goal[3:] = Rotation.from_rotvec([.03, -.02, .02]).as_quat()
    for i in range(100):
        clock[0] += DT
        arm.set_target(goal, 1, i)
    assert np.linalg.norm(arm.correction.offset) > 0
    rng = np.random.default_rng(13)
    for i in range(200):
        goal[:3] = rng.uniform(-.02, .02, 3)
        goal[3:] = Rotation.from_rotvec(rng.uniform(-.2, .2, 3)).as_quat()
        previous = arm.target_pose.copy()
        clock[0] += DT
        arm.set_target(goal, 1, 100+i)
        assert np.linalg.norm(arm.target_pose[:3]-previous[:3]) <= .0001+1e-12
        assert (Rotation.from_quat(arm.target_pose[3:]) * Rotation.from_quat(previous[3:]).inv()).magnitude() <= .001+1e-12
        assert np.linalg.norm(arm.correction.offset[:3]) <= arm.compliance.tracking_max_offset_m+1e-12
        assert np.linalg.norm(arm.correction.offset[3:]) <= arm.compliance.tracking_max_offset_rad+1e-12
