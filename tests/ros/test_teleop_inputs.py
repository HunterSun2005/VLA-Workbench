"""Production input integration and coordinator, independent real-wheel fake-Robot process."""
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
from vla.datacollection.franka_robotiq_rollout.collect_ros import RolloutCollector, GamepadTarget
from vla.datacollection.franka_robotiq_rollout.gamepad import decode_cartesian_gamepad_values, gamepad_is_neutral
from vla.datacollection.franka_robotiq_rollout.gamepad_layout import axis_buttons, node_for_layout
from vla.control.ros import ControlCoordinator, ArmMeasurement
from tests.control.test_franky_process import process_worker, request

ROOT = Path(__file__).resolve().parents[2]


def rig(profile):
    c = RolloutCollector.__new__(RolloutCollector)
    c.config = load_rollout_config(ROOT/f'src/vla/datacollection/franka_robotiq_rollout/configs/{profile}.yaml')
    c._lock = threading.RLock()
    c._gamepad_targets, c._previous_buttons, c._initial_pose = {}, {}, {}
    c._high_speed, c._speed_scale = defaultdict(bool), defaultdict(lambda:1.)
    c._recording = False
    c._latest, c._input_diagnostics = {}, {}
    c._update_gamepad_gripper = lambda a,t,g,m: setattr(t,'gripper', np.clip(t.gripper+m*.01,0,1))
    c._publish_gripper_speed = lambda *args: None
    c.coordinator = ControlCoordinator(c.config.action_spec, {},
        {a.name:(np.array(a.workspace_min),np.array(a.workspace_max)) for a in c.config.arms},
        lambda *a: None, lambda *a: None, backend='franky')
    for a in c.config.arms:
        measured = Rotation.from_euler('xyz',[.3,-.4,.8]).as_matrix()
        ref = measured.copy()
        p = np.array([.4,0,.3])
        c._gamepad_targets[a.name] = GamepadTarget(p.copy(),ref.copy(),.5,np.zeros(7),np.zeros(3),np.eye(3))
        from vla.control.geometry import upright_tcp_rotation
        c._initial_pose[a.name] = (p.copy(),upright_tcp_rotation(ref))
        robot = NS(robot_time_s=time.monotonic(), o_t_ee=c._worker_pose([*p, *Rotation.from_matrix(measured).as_quat()]))
        now = time.monotonic_ns()
        for key,value in [('robot_state',robot),('gamepad',NS(axes=[0.,0.,1.,0.,0.,1.],buttons=[0]*21)),
                          ('gripper_state',NS(name=[a.gripper_joint_name],position=[.4])),
                          ('measured_joint',NS(name=[f'fr3_joint{i}' for i in range(1,8)],position=[0.]*7))]:
            c._latest[f'{a.name}.{key}']=(now,value)
        c.coordinator.observe(a.name, ArmMeasurement(np.zeros(7), .5, np.zeros(3),np.eye(3), now))
    return c


def drive(c, values, dt):
    for a in c.config.arms:
        c._latest[f'{a.name}.gamepad'] = (time.monotonic_ns(), values[f'{a.name}.gamepad'])
        values[f'{a.name}.robot_state'].robot_time_s += dt
    return c._publish_gamepad(values, dt)


@pytest.mark.parametrize('profile',['single_left','dual'])
@pytest.mark.parametrize('axis',[0,1,3,4])
@pytest.mark.parametrize('sign',[-1,1])
def test_each_stick_direction_and_actual_tool_rotation(profile,axis,sign):
    c=rig(profile)
    values={k:v[1] for k,v in c._latest.items()}
    arm=c.config.arms[0]; target=c._gamepad_targets[arm.name]
    p,r=target.position.copy(),target.rotation.copy()
    values[f'{arm.name}.gamepad'].axes[axis]=sign
    assert drive(c,values,.01)
    if axis<2:
        delta=target.position-p
        assert delta[1-axis]*sign>0
        np.testing.assert_allclose(np.delete(delta,1-axis),0,atol=1e-12)
        np.testing.assert_allclose(target.rotation,r,atol=1e-12)
    else:
        # Original single-arm calibrated directions, independent of the decoder.
        angular = np.array([-1., -1., 0.] if axis == 3 else [1., -1., 0.])
        angular *= sign * c.config.gamepad_tilt_speed / np.sqrt(2)
        measured=Rotation.from_euler('xyz',[.3,-.4,.8]).as_matrix()
        expected=Rotation.from_rotvec(measured@angular*.01).as_matrix()@r
        np.testing.assert_allclose(target.rotation,expected,atol=1e-12)
        assert Rotation.from_matrix(target.rotation@r.T).magnitude()>0
        np.testing.assert_allclose(target.position,p,atol=1e-12)
    held_p,held_r=target.position.copy(),target.rotation.copy()
    values[f'{arm.name}.gamepad'].axes[axis]=0
    drive(c,values,.01)
    np.testing.assert_allclose(target.position,p,atol=1e-12)
    np.testing.assert_allclose(target.rotation,r,atol=1e-12)
    # A later measured disturbance must not drag the held reference.
    drive(c,values,.01)
    np.testing.assert_allclose(target.position,p,atol=1e-12)
    np.testing.assert_allclose(target.rotation,r,atol=1e-12)


def test_standard_buttons_raw_profile_and_workspace():
    c=rig('single_left'); a=c.config.arms[0]; m=a.gamepad
    values={k:v[1] for k,v in c._latest.items()}; joy=values['left.gamepad']
    assert node_for_layout(m.input_layout)=='joy_node'
    assert gamepad_is_neutral(joy.axes,joy.buttons,m,c.config)
    standard=replace(m, input_layout='standard_sdl', **axis_buttons('standard_sdl'))
    assert node_for_layout(standard.input_layout)=='game_controller_node'
    assert gamepad_is_neutral([0]*6,[0]*21,standard,c.config)
    # Logical shoulder controls no longer overlap SDL Back/Start buttons.
    joy.buttons[m.speed_button]=1;drive(c,values,.01)
    assert c._high_speed['left']
    joy.buttons[m.speed_button]=0;joy.buttons[m.upright_button]=1
    drive(c,values,.01)
    np.testing.assert_allclose(c._gamepad_targets['left'].pose_return.rotation[:,2],[0,0,-1],atol=1e-12)
    joy.buttons=[0]*21;joy.buttons[m.reset_pose_button]=1;drive(c,values,.01)
    np.testing.assert_allclose(c._gamepad_targets['left'].pose_return.rotation,np.diag([1.,-1.,-1.]))
    c._gamepad_targets['left'].pose_return = None
    joy.buttons=[0]*21;joy.axes[1]=1
    values['left.robot_state'].o_t_ee.pose.position.x=a.workspace_max[0]
    c._gamepad_targets['left'].position[0]=a.workspace_max[0]
    drive(c,values,.01)
    assert c._gamepad_targets['left'].position[0]>a.workspace_max[0]
    assert c.input_diagnostics()['left']['workspace_limits_enabled'] is False
    joy.axes[1]=0;joy.axes[2]=-1
    drive(c,values,.01)
    assert c._gamepad_targets['left'].gripper>.5


@pytest.mark.parametrize('profile',['single_left','dual'])
@pytest.mark.parametrize('source',['gamepad','gello','policy'])
def test_input_through_coordinator_to_real_tracker_fake_process(profile,source):
    c=rig(profile); names=tuple(a.name for a in c.config.arms)
    with process_worker(names) as (root,child,argv):
        (root/'capture-targets').touch()
        c.coordinator.request_reset=lambda gen: request(root,'reset',gen)
        gen=c.coordinator.transition(source)
        pending={};sequence=0
        def publish(name,pose,gripper,generation,command_id):
            nonlocal sequence
            pending[name]={'pose':pose.tolist()}
            if set(pending)==set(names):
                sequence+=1
                response=request(root,'target',generation,sequence=sequence,timestamp_ns=time.monotonic_ns(),arms=dict(pending))
                assert not response.get('fault'),response
                pending.clear()
        c.coordinator.publish=publish
        for a in c.config.arms:
            now=time.monotonic_ns()
            c.coordinator.observe(a.name,ArmMeasurement(np.zeros(7),.5,np.zeros(3),np.eye(3),now))
            assert c.coordinator.acknowledge(a.name,gen,0,np.zeros(7),now)
        values={k:v[1] for k,v in c._latest.items()}
        values['left.gamepad'].axes[1]=1
        drive(c,values,.01)
        action=c._input_action
        if source=='gello':
            c.coordinator.servos={name:NS(reset_tracking=lambda:None, forward_tcp=lambda q,tp,tr:(np.array([.41,.02,.3]),np.eye(3))) for name in names}
            action=c.coordinator.gello_target({n:np.zeros(7) for n in names},{n:.5 for n in names})
        for slot in range(32):
            c.coordinator.accept(action,source=source,generation=gen,slot_id=slot)
            c.coordinator.tick(1/30)
        captures=[json.loads(line) for line in (root/'targets.jsonl').read_text().splitlines()]
        assert len(captures)>=32*len(names)
        for index in range(len(names)):
            expected=action[index*c.config.action_spec.width:][:3]
            np.testing.assert_allclose(captures[-len(names)+index]['translation'],expected,atol=1e-6)
        if source == 'gamepad':
            # Verify the real wheel receives all four physical stick directions,
            # using independent expected translations/rotations, not the decoder.
            measured = Rotation.from_euler('xyz', [.3, -.4, .8]).as_matrix()
            slot = 32
            for axis in (0,1,3,4):
                for sign in (-1, 1):
                    for a in c.config.arms:
                        values[f'{a.name}.gamepad'].axes = [0.,0.,1.,0.,0.,1.]
                        values[f'{a.name}.gamepad'].axes[axis] = sign
                    expected = {}
                    for name in names:
                        t = c._gamepad_targets[name]
                        t.position, t.rotation = np.array([.4, 0., .3]), measured.copy()
                        t.linear_active = t.angular_active = (False, False, False)
                        p, r = t.position.copy(), t.rotation.copy()
                        if axis < 2:
                            p[1-axis] += sign * c.config.gamepad_xy_speed / 30
                        else:
                            angular = np.array([-1., -1., 0.] if axis == 3 else [1., -1., 0.])
                            angular *= sign * c.config.gamepad_tilt_speed / np.sqrt(2)
                            r = Rotation.from_rotvec(measured @ angular / 30).as_matrix() @ r
                        expected[name] = p, r
                        now = time.monotonic_ns()
                        c.coordinator.observe(name, ArmMeasurement(np.zeros(7), .5, np.zeros(3), np.eye(3), now))
                    drive(c,values, 1/30)
                    c.coordinator.accept(c._input_action, source=source, generation=gen, slot_id=slot)
                    c.coordinator.tick(1/30)
                    slot += 1
                    actual = [json.loads(line) for line in (root/'targets.jsonl').read_text().splitlines()][-len(names):]
                    for name, capture in zip(names, actual):
                        np.testing.assert_allclose(capture['translation'], expected[name][0], atol=1e-6)
                        np.testing.assert_allclose(Rotation.from_quat(capture['quaternion']).as_matrix(),
                                                   expected[name][1], atol=1e-6)
                    for name in names:
                        values[f'{name}.gamepad'].axes = [0.,0.,1.,0.,0.,1.]
                    drive(c, values, 1/30)
                    c.coordinator.accept(c._input_action, source=source, generation=gen, slot_id=slot)
                    c.coordinator.tick(1/30)
                    slot += 1
                    actual = [json.loads(line) for line in (root/'targets.jsonl').read_text().splitlines()][-len(names):]
                    for capture in actual:
                        np.testing.assert_allclose(capture['translation'], [.4,0,.3], atol=1e-6)
                        np.testing.assert_allclose(Rotation.from_quat(capture['quaternion']).as_matrix(),
                                                   measured, atol=1e-6)
        stopped=request(root,'stop',gen+1)
        assert not any(s['servo_ready'] for s in stopped['states'])
        stale=request(root,'target',gen,sequence=999,timestamp_ns=time.monotonic_ns(),arms={n:{'pose':[.4,0,.3,0,0,0,1]} for n in names})
        assert stale['request_status']=='ignored'


def test_standard_rejects_raw_trigger_shape_and_all_button_roles():
    c=rig('single_left');a=c.config.arms[0];m=a.gamepad
    standard=replace(m, input_layout='standard_sdl', **axis_buttons('standard_sdl'))
    with pytest.raises(ValueError,match='standard_sdl'):
        gamepad_is_neutral([0,0,1,0,0,1],[0]*21,standard,c.config)
    with pytest.raises(ValueError,match='invalid gamepad'):
        decode_cartesian_gamepad_values([0,1.1,0,0,0,0],[0]*21,m,c.config)
    for button,linear_axis,sign in [(m.a_button,2,-1),(m.y_button,2,1)]:
        buttons=[0]*21;buttons[button]=1
        decoded=decode_cartesian_gamepad_values([0,0,1,0,0,1],buttons,m,c.config)
        assert decoded.linear_velocity_base[linear_axis]*sign>0
        np.testing.assert_allclose(decoded.angular_velocity_tool,0)
    for button,sign in [(m.x_button,-1),(m.b_button,1)]:
        buttons=[0]*21;buttons[button]=1
        decoded=decode_cartesian_gamepad_values([0,0,1,0,0,1],buttons,m,c.config)
        assert decoded.angular_velocity_tool[2]*sign>0
        np.testing.assert_allclose(decoded.linear_velocity_base,0)
    values={k:v[1] for k,v in c._latest.items()}
    c._recording=True
    toggles=[];c.toggle_phase=lambda:toggles.append(True)
    values['left.gamepad'].buttons[m.phase_button]=1
    drive(c,values,.01);drive(c,values,.01)
    assert toggles==[True]


def test_partial_translation_release_follows_until_settled_and_preserves_other_axes():
    c = rig('single_left')
    values = {k:v[1] for k,v in c._latest.items()}
    joy, t = values['left.gamepad'], c._gamepad_targets['left']
    joy.axes[0] = joy.axes[1] = 1.
    drive(c, values, 1/30)
    t.position += [.02, .03, .01]  # Simulated following error, including a held Z reference.
    before = t.position.copy()
    joy.axes[1] = 0.
    drive(c, values, 1/30)
    np.testing.assert_allclose(t.position, [.4, before[1]+c.config.gamepad_xy_speed/30, before[2]], atol=1e-12)
    # During braking, only the released X channel follows the moving measurement.
    values['left.robot_state'].o_t_ee.pose.position.x = .39
    drive(c, values, 1/30)
    assert t.position[0] == pytest.approx(.39)


@pytest.mark.parametrize('profile', ['single_left', 'dual'])
@pytest.mark.parametrize('motion', ['xyz', 'tilt', 'twist'])
@pytest.mark.parametrize('fast', [False, True])
@pytest.mark.parametrize('smoothing', [(False, False), (True, False), (False, True), (True, True)])
def test_release_with_moving_delayed_plant(profile, motion, fast, smoothing, monkeypatch):
    """30 Hz production frontend + 100 Hz servo + delayed inertial plant.

    The fixed-state acceptance harness cannot expose a stale release latch.
    This plant carries velocity beyond release and makes that regression visible.
    """
    from collections import deque
    from tests.control.test_velocity_control import fake_arm
    from vla.control.franky.velocity_control import VelocityControl
    import vla.control.franky.worker as worker_module
    import vla.datacollection.franka_robotiq_rollout.collect_ros as collector_module
    c = rig(profile)
    values = {k:v[1] for k,v in c._latest.items()}
    clock = [0.]
    fake_time = NS(monotonic=lambda:clock[0], monotonic_ns=lambda:int(clock[0]*1e9))
    monkeypatch.setattr(collector_module, 'time', fake_time)
    monkeypatch.setattr(worker_module, 'time', fake_time)
    plants = {}
    for arm in c.config.arms:
        name = arm.name
        servo = fake_arm()
        servo.velocity_control = VelocityControl(target_smoothing_enabled=smoothing[0],
                                                velocity_smoothing_enabled=smoothing[1])
        servo.robot.state.time = NS(to_sec=lambda:clock[0])
        servo.reset(1)
        t = c._gamepad_targets[name]
        plants[name] = dict(servo=servo, p=t.position.copy(), r=t.rotation.copy(), v=np.zeros(6),
                            delay=deque([np.zeros(6) for _ in range(6)]), trace=[], latched=None)
        c._high_speed[name] = fast
    step = 1/300
    for tick in range(1350):  # 2s motion, then 2.5s to brake and observe the hold.
        clock[0] = tick*step
        if tick % 10 == 0:
            for arm in c.config.arms:
                name, m = arm.name, arm.gamepad
                plant = plants[name]
                robot, joy = values[f'{name}.robot_state'], values[f'{name}.gamepad']
                robot.robot_time_s = clock[0]
                robot.o_t_ee = c._worker_pose([*plant['p'], *Rotation.from_matrix(plant['r']).as_quat()])
                c._latest[f'{name}.gamepad'] = (int(clock[0]*1e9), joy)
                if motion == 'xyz':
                    joy.axes[1] = float(tick < 600)
                elif motion == 'tilt':
                    joy.axes[3] = float(tick < 600)
                else:
                    joy.buttons[m.x_button] = int(tick < 600)
            c._publish_gamepad(values, 1/30)
            for name, plant in plants.items():
                t = c._gamepad_targets[name]
                pose = np.r_[t.position, Rotation.from_matrix(t.rotation).as_quat()]
                plant['servo'].set_target(pose.tolist(), 1, tick//10, {'source':'gamepad', 'dt':1/30})
                if tick > 600 and not t.release.braking.any() and plant['latched'] is None:
                    plant['latched'] = len(plant['trace'])
        for plant in plants.values():
            servo = plant['servo']
            servo.robot.state.O_T_EE[:3, 3] = plant['p']
            servo.robot.state.O_T_EE[:3, :3] = plant['r']
            if tick % 3 == 0:
                servo.servo_once()
            plant['delay'].append(servo.last_twist.copy())
            demanded = plant['delay'].popleft()
            plant['v'] += (demanded-plant['v'])*(1-np.exp(-step/.12))
            plant['p'] += plant['v'][:3]*step
            plant['r'] = Rotation.from_rotvec(plant['v'][3:]*step).as_matrix() @ plant['r']
            plant['trace'].append((plant['p'].copy(), plant['r'].copy()))
    for plant in plants.values():
        assert plant['latched'] is not None
        p, r = plant['trace'][plant['latched']]
        held = plant['trace'][plant['latched']:]
        assert max(np.linalg.norm(q-p) for q, _ in held) <= .0005
        assert max(Rotation.from_matrix(s @ r.T).magnitude() for _, s in held) <= .01
        # No sustained return to the pose sampled at release, even before latching.
        after = plant['trace'][600:]
        if motion == 'xyz':
            assert max(q[0] for q, _ in after)-after[-1][0][0] < .0005
        else:
            r0 = after[0][1]
            angles = [Rotation.from_matrix(s @ r0.T).magnitude() for _, s in after]
            assert max(angles)-angles[-1] < .01


@pytest.mark.parametrize('profile', ['single_left', 'dual'])
@pytest.mark.parametrize('mount_deg', [0, -45])
def test_xyz_motion_keeps_absolute_orientation_despite_measured_drift(profile, mount_deg):
    """Physical tracking error must not become a new neutral-stick goal."""
    from vla.actions import matrix_to_rot6d
    c = rig(profile)
    values = {k:v[1] for k,v in c._latest.items()}
    initial = {name:(t.position.copy(), t.rotation.copy()) for name,t in c._gamepad_targets.items()}
    for t in c._gamepad_targets.values():
        t.hardware_to_logical_rotation = Rotation.from_euler('z', mount_deg, degrees=True).as_matrix()
    for tick in range(120):
        for arm in c.config.arms:
            name = arm.name
            p, reference = initial[name]
            # Tilt feedback while XYZ moves, reverses and releases. No right
            # stick or twist button is pressed; keep the nonidentity goal.
            measured = Rotation.from_rotvec(np.array([.08, -.1, .05])*np.sin(tick/12)).as_matrix() @ reference
            values[f'{name}.robot_state'].o_t_ee = c._worker_pose([*p, *Rotation.from_matrix(measured).as_quat()])
            joy = values[f'{name}.gamepad']
            joy.axes[0] = float(np.sign(np.sin(tick/15)))
            joy.axes[1] = float(np.sign(np.cos(tick/15)))
            joy.buttons[arm.gamepad.y_button] = int(tick < 30)
            joy.buttons[arm.gamepad.a_button] = int(30 <= tick < 60)
        drive(c, values, 1/30)
        projected = c.coordinator.project(c._input_action)
        for index, arm in enumerate(c.config.arms):
            measured = Rotation.from_quat([values[f'{arm.name}.robot_state'].o_t_ee.pose.orientation.x, values[f'{arm.name}.robot_state'].o_t_ee.pose.orientation.y, values[f'{arm.name}.robot_state'].o_t_ee.pose.orientation.z, values[f'{arm.name}.robot_state'].o_t_ee.pose.orientation.w]).as_matrix()
            reference = initial[arm.name][1]
            expected = matrix_to_rot6d(reference)
            np.testing.assert_allclose(c._gamepad_targets[arm.name].rotation, reference, atol=1e-12)
            offset = index*c.config.action_spec.width
            np.testing.assert_allclose(projected[offset+3:offset+9], expected, atol=1e-7)


def test_partial_angular_release_uses_calibrated_axis_with_following_error():
    c = rig('single_left')
    values = {k:v[1] for k,v in c._latest.items()}
    joy, t = values['left.gamepad'], c._gamepad_targets['left']
    measured = Rotation.from_euler('xyz', [.3, -.4, .8]).as_matrix()
    # Independent -45 degree columns: vertical, horizontal, tool twist.
    basis = measured @ np.array([[1,1,0],[-1,1,0],[0,0,np.sqrt(2)]])/np.sqrt(2)
    joy.axes[3] = -1.; joy.axes[4] = 1.
    drive(c, values, 1/30)
    t.rotation = Rotation.from_rotvec(basis @ [.08, .05, .03]).as_matrix() @ measured
    p = t.position.copy()
    joy.axes[4] = 0.  # Release vertical tilt; keep horizontal and the prior twist reference.
    drive(c, values, 1/30)
    held = Rotation.from_rotvec(basis @ [0., .05, .03]).as_matrix() @ measured
    expected = Rotation.from_rotvec(basis[:,1]*c.config.gamepad_tilt_speed/30).as_matrix() @ held
    np.testing.assert_allclose(t.rotation, expected, atol=1e-12)
    np.testing.assert_array_equal(t.position, p)
    # With stationary feedback the released component stays zero while horizontal input continues.
    components = basis.T @ Rotation.from_matrix(expected @ measured.T).as_rotvec()
    components[0] = 0.
    expected = Rotation.from_rotvec(basis @ components).as_matrix() @ measured
    drive(c, values, 1/30)
    expected = Rotation.from_rotvec(basis[:,1]*c.config.gamepad_tilt_speed/30).as_matrix() @ expected
    np.testing.assert_allclose(t.rotation, expected, atol=1e-12)


@pytest.mark.parametrize('axis', [3, 4])
def test_fast_is_five_times_translation_and_twice_right_stick_with_same_gripper(axis):
    normal, fast = rig('single_left'), rig('single_left')
    for c in (normal, fast):
        values = {k:v[1] for k,v in c._latest.items()}
        joy = values['left.gamepad']
        joy.axes[1] = joy.axes[axis] = 1.; joy.axes[2] = -1.
        if c is fast:
            c._high_speed['left'] = True
        p = c._gamepad_targets['left'].position.copy()
        for _ in range(9):
            drive(c, values, 1/30)
        c._test_before = c._gamepad_targets['left'].position.copy()
        c._test_rotation_before = c._gamepad_targets['left'].rotation.copy()
        drive(c, values, 1/30)
    nt, ft = normal._gamepad_targets['left'], fast._gamepad_targets['left']
    assert fast._speed_scale['left'] == pytest.approx(5.)
    np.testing.assert_allclose(ft.position-fast._test_before, 5*(nt.position-normal._test_before), atol=1e-12)
    normal_step = Rotation.from_matrix(nt.rotation @ normal._test_rotation_before.T).as_rotvec()
    fast_step = Rotation.from_matrix(ft.rotation @ fast._test_rotation_before.T).as_rotvec()
    np.testing.assert_allclose(fast_step, 2*normal_step, atol=1e-12)
    assert ft.gripper == pytest.approx(nt.gripper)


def test_input_gap_holds_once_then_fresh_input_resumes_without_catchup(monkeypatch):
    import vla.datacollection.franka_robotiq_rollout.collect_ros as module
    c = rig('single_left')
    values = {k:v[1] for k,v in c._latest.items()}
    clock = [time.monotonic_ns()]
    monkeypatch.setattr(module, 'time', NS(monotonic_ns=lambda:clock[0]))
    joy, t = values['left.gamepad'], c._gamepad_targets['left']
    joy.axes[1] = joy.axes[4] = 1.
    c._latest['left.gamepad'] = (clock[0], joy)
    c._publish_gamepad(values, 1/30)
    clock[0] += 110_000_000
    c._publish_gamepad(values, 1/30)
    assert t.input_hold
    p, r = t.position.copy(), t.rotation.copy()
    np.testing.assert_allclose(p, [.4,0,.3])
    values['left.robot_state'].o_t_ee.pose.position.x = .39
    clock[0] += 70_000_000
    c._publish_gamepad(values, 1/30)
    np.testing.assert_array_equal(t.position, [.4,0,.3])
    np.testing.assert_allclose(t.rotation, r, atol=1e-12)
    c._latest['left.gamepad'] = (clock[0], joy)
    c._publish_gamepad(values, 1/30)
    assert not t.input_hold
    assert t.position[0] == pytest.approx(.39+c.config.gamepad_xy_speed/30)
    clock[0] += 300_000_001
    with pytest.raises(RuntimeError, match='stale left.gamepad'):
        c._publish_gamepad(values, 1/30)


def test_workspace_disabled_gamepad_and_all_source_projection_keep_finite_checks():
    from vla.actions import matrix_to_rot6d
    from vla.datacollection.franka_robotiq_rollout.safety import validate_absolute_action
    c=rig('single_left')
    c.config=replace(c.config,workspace_limits_enabled=False)
    c.coordinator.workspace_limits_enabled=False
    a=c.config.arms[0]
    values={k:v[1] for k,v in c._latest.items()}
    t=c._gamepad_targets['left']
    t.position[0]=a.workspace_max[0]
    values['left.robot_state'].o_t_ee.pose.position.x=a.workspace_max[0]
    values['left.gamepad'].axes[1]=1.
    drive(c,values,1/30)
    assert t.position[0]>a.workspace_max[0]
    assert c.input_diagnostics()['left']['workspace_limits_enabled'] is False
    action=np.r_[[1.2,0,.3],matrix_to_rot6d(t.rotation),.5]
    np.testing.assert_allclose(c.coordinator.project(action),action)
    bounds=[(np.array(a.workspace_min),np.array(a.workspace_max))]
    validate_absolute_action(action,arm_count=1,workspace_bounds=bounds,workspace_limits_enabled=False)
    validate_absolute_action(action,arm_count=1,workspace_bounds=bounds)
    for bad in (np.r_[np.nan,action[1:]], np.r_[action[:3],np.zeros(6),.5]):
        with pytest.raises(ValueError):
            c.coordinator.project(bad)
        with pytest.raises(ValueError):
            validate_absolute_action(bad,arm_count=1,workspace_bounds=bounds,workspace_limits_enabled=False)
