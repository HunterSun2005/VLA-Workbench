"""Shortcut semantics, bounded references and cancellation; no device connections."""
from dataclasses import replace
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.ros.test_teleop_inputs import rig
from vla.datacollection.franka_robotiq_rollout.gamepad_layout import axis_buttons


def make_shortcut_rig(monkeypatch, profile='single_left'):
    import vla.datacollection.franka_robotiq_rollout.collect_ros as module
    c = rig(profile)
    values = {k:v[1] for k,v in c._latest.items()}
    initial = Rotation.from_rotvec([0,0,-.6]).as_matrix() @ np.diag([1.,-1.,-1.])
    for name in c._initial_pose:
        c._initial_pose[name] = (np.array([.35,.1,.25]), initial.copy())
    clock = [time.monotonic_ns()]
    monkeypatch.setattr(module, 'time', NS(monotonic_ns=lambda:clock[0], monotonic=lambda:clock[0]/1e9))
    def tick(follow=False):
        if follow:
            for name,t in c._gamepad_targets.items():
                values[f'{name}.robot_state'].o_t_ee = c._worker_pose([*t.position,*Rotation.from_matrix(t.rotation).as_quat()])
        for name in c._gamepad_targets:
            c._latest[f'{name}.gamepad'] = (clock[0], values[f'{name}.gamepad'])
        c._publish_gamepad(values,1/30)
        clock[0] += 33_333_334
    return c, values, clock, tick


@pytest.mark.parametrize('profile',['single_left','dual'])
@pytest.mark.parametrize('button,label',[(6,'initial pose'),(7,'upright'),(10,'heading')])
def test_raw_shortcuts_preserve_original_geometry_and_are_bounded(monkeypatch,profile,button,label):
    c,values,clock,tick = make_shortcut_rig(monkeypatch,profile)
    t = c._gamepad_targets['left']
    p,r = t.position.copy(),t.rotation.copy()
    values['left.gamepad'].buttons[button]=1
    tick()
    goal=t.pose_return
    assert goal.label==label
    if button==6:
        np.testing.assert_allclose(goal.position,c._initial_pose['left'][0])
        np.testing.assert_allclose(goal.rotation,np.diag([1.,-1.,-1.]))
    else:
        np.testing.assert_array_equal(goal.position,p)
        np.testing.assert_array_equal(t.position,p)
        if button==7:
            np.testing.assert_allclose(goal.rotation,np.diag([1.,-1.,-1.]))
        else:
            yaw = np.arctan2(r[1,0],r[0,0])
            delta = Rotation.from_rotvec([0,0,-yaw]).as_matrix()
            np.testing.assert_allclose(goal.rotation,delta@r,atol=1e-12)
            # Base-Z rotation preserves tool tilt; no change of TCP XYZ.
            np.testing.assert_allclose(goal.rotation[2,:],r[2,:],atol=1e-12)
    assert np.linalg.norm(t.position-p)<=c.config.max_linear_velocity/30+1e-12
    assert Rotation.from_matrix(t.rotation@r.T).magnitude()<=.3/30+1e-12
    # Same held button does not renew the deadline or recapture a moving pose.
    deadline=goal.deadline
    for _ in range(5):
        tick(follow=True)
        assert t.pose_return is goal and goal.deadline==deadline
        if button!=6:
            np.testing.assert_array_equal(t.position,p)
    if profile=='dual':
        assert c._gamepad_targets['right'].pose_return is None
        np.testing.assert_allclose(c._gamepad_targets['right'].position,[.4,0,.3])


@pytest.mark.parametrize('buttons,label',[([6,7],'initial pose'),([6,7,10],'heading')])
def test_simultaneous_shortcut_precedence(monkeypatch,buttons,label):
    c,v,_,tick=make_shortcut_rig(monkeypatch)
    for button in buttons:
        v['left.gamepad'].buttons[button]=1
    tick()
    assert c._gamepad_targets['left'].pose_return.label==label


@pytest.mark.parametrize('motion',['stick','opposing_buttons','opposing_triggers'])
def test_motion_blocks_shortcut_and_cancels_existing_return(monkeypatch,motion):
    c,v,_,tick=make_shortcut_rig(monkeypatch)
    joy=v['left.gamepad'];t=c._gamepad_targets['left']
    joy.buttons[6]=1;tick()
    assert t.pose_return is not None
    joy.buttons[6]=0
    if motion=='stick':
        joy.axes[1]=1.
    elif motion=='opposing_buttons':
        joy.buttons[0]=joy.buttons[3]=1
    else:
        joy.axes[2]=joy.axes[5]=-1.
    tick()
    assert t.pose_return is None and 'cancelled' in t.shortcut_status
    # A fresh shortcut press while motion is held must be ignored too.
    joy.buttons[7]=1;tick()
    assert t.pose_return is None and 'ignored' in t.shortcut_status


def test_shortcut_completion_is_measured_and_held_button_does_not_restart(monkeypatch):
    c,v,_,tick=make_shortcut_rig(monkeypatch)
    t=c._gamepad_targets['left']
    v['left.gamepad'].buttons[6]=1
    tick();goal=t.pose_return
    for _ in range(800):
        before=t.position.copy();r=t.rotation.copy()
        tick(follow=True)
        assert np.linalg.norm(t.position-before)<=c.config.max_linear_velocity/30+1e-12
        assert Rotation.from_matrix(t.rotation@r.T).magnitude()<=.3/30+1e-12
        if t.pose_return is None:
            break
    assert t.shortcut_status=='completed: initial pose'
    assert np.linalg.norm(t.position-goal.position)<=.0005
    assert Rotation.from_matrix(t.rotation@goal.rotation.T).magnitude()<=np.deg2rad(2)
    p,r=t.position.copy(),t.rotation.copy()
    for _ in range(3):
        tick(follow=True)
        assert t.pose_return is None
        np.testing.assert_array_equal(t.position,p)
        np.testing.assert_array_equal(t.rotation,r)


def test_input_hold_cancels_return_and_reconnect_does_not_resume_it(monkeypatch):
    c,v,clock,tick=make_shortcut_rig(monkeypatch)
    t=c._gamepad_targets['left'];v['left.gamepad'].buttons[6]=1
    tick()
    clock[0]+=110_000_000
    c._publish_gamepad(v,1/30)  # Keep the old Joy stamp.
    assert t.pose_return is None and t.input_hold
    p,r=t.position.copy(),t.rotation.copy()
    tick()  # Fresh packet still holding View is not a new press.
    assert t.pose_return is None
    np.testing.assert_array_equal(t.position,p)
    np.testing.assert_array_equal(t.rotation,r)


def test_stationary_measured_robot_cannot_complete_reference_only_and_times_out(monkeypatch):
    c,v,clock,tick=make_shortcut_rig(monkeypatch)
    t=c._gamepad_targets['left'];v['left.gamepad'].buttons[6]=1;tick()
    goal=t.pose_return
    t.position,t.rotation=goal.position.copy(),goal.rotation.copy()
    tick()
    assert t.pose_return is goal  # Desired reached, measured still far away.
    clock[0]=int((goal.deadline+.01)*1e9)
    with pytest.raises(RuntimeError,match='return timed out'):
        tick()


def test_explicit_sdl_layout_maps_back_start_and_right_stick_click(monkeypatch):
    c,v,_,tick=make_shortcut_rig(monkeypatch)
    arm=c.config.arms[0]
    mapping=replace(arm.gamepad,input_layout='standard_sdl',**axis_buttons('standard_sdl'))
    c.config=replace(c.config,arms=(replace(arm,gamepad=mapping),))
    assert (mapping.reset_pose_button,mapping.upright_button,mapping.yaw_reset_button)==(4,6,8)
    v['left.gamepad'].axes=[0.]*6
    v['left.gamepad'].buttons[8]=1;tick()
    assert c._gamepad_targets['left'].pose_return.label=='heading'


@pytest.mark.parametrize('profile', ['single_left', 'dual'])
def test_shortcuts_submit_absolute_tcp_through_coordinator_and_real_tracker(monkeypatch, profile):
    import json
    from tests.control.test_franky_process import process_worker, request
    from vla.control.ros import ArmMeasurement
    c, values, _, tick = make_shortcut_rig(monkeypatch, profile)
    names = tuple(a.name for a in c.config.arms)
    with process_worker(names) as (root, _, _):
        (root/'capture-targets').touch()
        c.coordinator.request_reset = lambda gen: request(root, 'reset', gen)
        generation = c.coordinator.transition('gamepad')
        for name in names:
            now = time.monotonic_ns()
            c.coordinator.observe(name, ArmMeasurement(np.zeros(7), .5, np.zeros(3), np.eye(3), now))
            c.coordinator.acknowledge(name, generation, 0, np.zeros(7), now)
        pending, replies = {}, []
        def publish(name, pose, gripper, gen, command_id):
            pending[name] = {'pose': pose.tolist()}
            if set(pending) == set(names):
                reply = request(root, 'target', gen, sequence=command_id, slot_id=command_id,
                                timestamp_ns=time.monotonic_ns(), arms=dict(pending))
                assert reply['request_status'] == 'applied' and not reply['fault']
                replies.append(reply)
                pending.clear()
        c.coordinator.publish = publish
        slot = 0
        def submit():
            nonlocal slot
            for name in names:
                c.coordinator.observe(name, ArmMeasurement(np.zeros(7), .5, np.zeros(3), np.eye(3), time.monotonic_ns()))
            tick()
            c.coordinator.accept(c._input_action, source='gamepad', generation=generation, slot_id=slot)
            c.coordinator.tick(1/30)
            slot += 1
        # Allow worker step limits to reach the starting reference.
        for _ in range(40):
            submit()
        for button in (6, 7, 10):
            values['left.gamepad'].buttons = [0]*21
            submit()
            values['left.gamepad'].buttons[button] = 1
            submit()
            assert c._gamepad_targets['left'].pose_return is not None
            actual = [json.loads(line) for line in (root/'targets.jsonl').read_text().splitlines()][-len(names):]
            for name, capture in zip(names, actual):
                target = c._gamepad_targets[name]
                np.testing.assert_allclose(capture['translation'], target.position, atol=1e-6)
                np.testing.assert_allclose(Rotation.from_quat(capture['quaternion']).as_matrix(), target.rotation, atol=1e-6)
                np.testing.assert_allclose(replies[-1]['references'][name][:3], target.position, atol=1e-6)
        assert len(replies) == slot  # Exactly one request for each accepted slot.
        request(root, 'stop', generation+1)


def test_control_boundary_cancels_shortcut_without_resuming(monkeypatch):
    c, v, _, tick = make_shortcut_rig(monkeypatch)
    v['left.gamepad'].buttons[6] = 1
    tick()
    t = c._gamepad_targets['left']
    assert t.pose_return is not None
    c._pending_compliance = c._execution = c._pending_frame = c._journal = None
    c._pending_worker_targets, c._last_gripper_goal = {}, {}
    c._segment_id = 0
    c._control_boundary(reason='operator pause')
    assert t.pose_return is None and c.coordinator.owner is None
    assert t.shortcut_status == 'cancelled: control boundary'
    tick()  # Holding the key through a boundary does not create a new edge.
    assert t.pose_return is None


def test_each_manual_enable_recaptures_initial_pose(monkeypatch):
    from vla.control.geometry import upright_tcp_rotation
    c, v, clock, _ = make_shortcut_rig(monkeypatch)
    c._faulted = c._storage_blocked = c._waiting_resume = c._saving = c._processing = False
    c._pending_finish = None
    c.demonstration = False
    c._teleop_enabled = True
    c.servos = {'left': NS(model=NS(joint_names=v['left.measured_joint'].name))}
    c._stop_grippers_at_measured_position = c._reset_servo_tracking = c._invalidate_policy = lambda: None
    c._control_boundary = lambda *a, **kw: None
    for p, r in (([.4, 0., .3], [.3, -.4, .8]), ([.45, .02, .35], [.2, .1, -.3])):
        c._intervention = False
        measured = Rotation.from_euler('xyz', r).as_matrix()
        v['left.robot_state'].o_t_ee = c._worker_pose([*p, *Rotation.from_matrix(measured).as_quat()])
        v['left.robot_state'].f_t_ee = c._worker_pose([0, 0, 0, 0, 0, 0, 1])
        c.toggle_intervention()
        assert c._intervention
        np.testing.assert_allclose(c._initial_pose['left'][0], p)
        np.testing.assert_allclose(c._initial_pose['left'][1], np.diag([1.,-1.,-1.]))


@pytest.mark.parametrize('button', [6, 7, 10])
def test_mount_offset_stays_out_of_shortcut_action(monkeypatch, button):
    c, v, _, tick = make_shortcut_rig(monkeypatch)
    mount = Rotation.from_euler('z', -45, degrees=True).as_matrix()
    t = c._gamepad_targets['left']
    t.hardware_to_logical_rotation = mount
    v['left.gamepad'].buttons[button] = 1
    tick()
    goal = t.pose_return
    if button != 10:
        np.testing.assert_allclose(goal.rotation, np.diag([1, -1, -1]))
    else:
        assert np.arctan2(goal.rotation[1, 0], goal.rotation[0, 0]) == pytest.approx(0, abs=1e-12)
    physical_goal = goal.rotation @ mount
    # Downward shortcut corresponds to hardware yaw +45, logical action yaw 0.
    if button != 10:
        assert np.arctan2(physical_goal[1, 0], physical_goal[0, 0]) == pytest.approx(np.pi/4)


def test_gamepad_rotation_and_release_use_the_same_logical_control_basis(monkeypatch):
    from vla.datacollection.franka_robotiq_rollout.gamepad import decode_cartesian_gamepad_values
    c, v, _, tick = make_shortcut_rig(monkeypatch)
    mount = Rotation.from_euler('z', -45, degrees=True).as_matrix()
    t = c._gamepad_targets['left']
    hardware = t.rotation.copy()
    t.rotation = hardware @ mount.T
    t.hardware_to_logical_rotation = mount
    v['left.robot_state'].o_t_ee = c._worker_pose([*t.position, *Rotation.from_matrix(t.rotation).as_quat()])
    v['left.gamepad'].axes[4] = .8
    decoded = decode_cartesian_gamepad_values(v['left.gamepad'].axes, v['left.gamepad'].buttons,
                                              c.config.arms[0].gamepad, c.config)
    expected_increment = hardware @ decoded.angular_velocity_tool / 30
    before = t.rotation.copy()
    tick()
    np.testing.assert_allclose(Rotation.from_matrix(t.rotation @ before.T).as_rotvec(), expected_increment, atol=1e-12)
    v['left.gamepad'].axes[4] = 0.
    tick()
    np.testing.assert_allclose(t.rotation, before, atol=1e-12)
