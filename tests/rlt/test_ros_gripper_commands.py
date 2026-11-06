"""Command/state separation and recorder heartbeat without ROS nodes or hardware."""
from concurrent.futures import Future
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from tests.rlt.test_dual_gamepad_record import ros_module
from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
from vla.datacollection.franka_robotiq_dual.gamepad_record import DEFAULT_CONFIG
from vla.datacollection.gripper_contract import gello_recording_metadata
from vla.datacollection.franka_robotiq_single_left.v30_writer import V30DatasetWriter


class Goal:
    def __init__(self):
        self.command = NS(position=0., max_effort=0.)


class Messages:
    def __init__(self):
        self.values = []

    def publish(self, msg):
        self.values.append(msg.data)


def joint_message():
    return NS(header=NS())


def pose_message():
    return NS(pose=NS(position=NS(x=0., y=0., z=0.), orientation=NS(x=0., y=0., z=0., w=1.)))


@pytest.fixture
def ros_transport(ros_module, monkeypatch):
    monkeypatch.setattr(ros_module, 'GripperCommand', NS(Goal=Goal))
    monkeypatch.setattr(ros_module, 'Float32', NS)
    monkeypatch.setattr(ros_module, 'Float64MultiArray', NS)
    monkeypatch.setattr(ros_module, 'JointState', joint_message)
    monkeypatch.setattr(sys.modules['vla.control.ros.transport'], 'JointState', joint_message)
    return ros_module


@pytest.mark.parametrize('arm_name', ['left', 'right'])
def test_gamepad_rows_record_submitted_goals_and_latched_hold(ros_transport, arm_name):
    m = ros_transport
    arm = next(a for a in load_rollout_config(DEFAULT_CONFIG).arms if a.name == arm_name)
    c = object.__new__(m.RolloutCollector)
    c.config = NS(gamepad_gripper_speed=.4, arms=(arm,))
    goals, faults = [], []
    c._gripper_action_clients = {arm_name: NS(send_goal_async=lambda goal: (goals.append(goal.command.position) or Future()))}
    c._gripper_goal_futures = set()
    c._gripper_speed_publishers = {arm_name: Messages()}
    c._gripper_publishers = {arm_name: Messages()}
    c._joint_publishers = {arm_name: NS(publish=lambda _: None)}
    c.get_clock = lambda: NS(now=lambda: NS(to_msg=lambda: None))
    c.servos = {arm_name: NS(model=NS(joint_names=[f'j{i}' for i in range(7)]))}
    c.coordinator = NS(owner='gamepad')
    c._last_gripper_goal = {}
    c._publish_sequence = 0
    c._latch_fault = faults.append
    target = NS(gripper=.5, gripper_direction=0)
    for motion, measured, expected, count in [(1., .2, 1., 1), (1., .4, 1., 1),
                                              (0., .37, .37, 2), (0., .41, .37, 2),
                                              (-1., .6, 0., 3)]:
        c._update_gamepad_gripper(arm, target, measured, motion)
        c._coordinated_publish(arm_name, np.zeros(7), target.gripper, 1, 1)
        assert target.gripper == expected and len(goals) == count
        assert c._gripper_publishers[arm_name].values[-1] == expected
    # Policy's intermediate aperture passes through the identical adapter unchanged.
    c.coordinator.owner = 'policy'
    c._coordinated_publish(arm_name, np.zeros(7), .73, 2, 1)
    assert goals[-1] == pytest.approx(arm.gripper_closed_position + .73*(arm.gripper_open_position-arm.gripper_closed_position))
    rejected = Future(); rejected.set_result(NS(accepted=False))
    c._gripper_goal_response(arm_name, rejected)
    failed = Future(); failed.set_exception(RuntimeError('execution failed'))
    c._gripper_goal_result(arm_name, failed)
    assert len(faults) == 2


@pytest.fixture
def fine_module(ros_transport, monkeypatch):
    for name, attrs in {
        'rclpy.signals': {'SignalHandlerOptions': NS(NO=0)},
        'std_srvs.srv': {'SetBool': object},
    }.items():
        module = ModuleType(name); module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys.modules['rclpy.executors'], 'ExternalShutdownException', RuntimeError, raising=False)
    monkeypatch.setattr(sys.modules['std_msgs.msg'], 'Float32', NS)
    monkeypatch.setattr(sys.modules['std_msgs.msg'], 'Float64MultiArray', NS)
    monkeypatch.setattr(sys.modules['control_msgs.action'], 'GripperCommand', NS(Goal=Goal))
    name = 'vla.datacollection.start_teleop._test_fine_gripper'
    path = Path(__file__).parents[2] / 'src/vla/datacollection/start_teleop/fine_teleop.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module); spec.loader.exec_module(module)
    return module


def test_fine_goal_heartbeat_does_not_redrive_or_follow_measured_state(fine_module, monkeypatch):
    m = fine_module
    clock = [1.]
    monkeypatch.setattr(m.time, 'monotonic', lambda: clock[0])
    node = object.__new__(m.FineTeleopNode)
    node.managed = node.managed_enabled = True
    node.managed_fault = ''
    node.raw_gello_joints = None
    node.command_joints = None
    node.command_gripper, node.measured_gripper, node.raw_gripper = .8, .3, .8
    node.measured_gripper_time = node.raw_gripper_time = 1.
    node.source_timeout, node.gripper_speed, node.gripper_return_speed = .5, .4, .5
    node.gripper_direction, node.return_gripper_goal = 0, None
    node.gripper_open_position, node.gripper_closed_position, node.gripper_max_effort = 0., .8, 50.
    node.gripper_publisher, node.gripper_action_command_publisher = Messages(), Messages()
    node.gripper_drive_publisher, node.gripper_speed_publisher = Messages(), Messages()
    node._gripper_goal_futures = set()
    goals, errors = [], []
    def send(goal):
        goals.append(goal.command.position)
        return Future()
    node.gripper_action_client = NS(send_goal_async=send)
    node.get_logger = lambda: NS(info=lambda _: None, error=errors.append)
    node.mode = m.Mode.GELLO
    node._publish_commands(clock[0])
    assert node.gripper_drive_publisher.values == [.8]
    # The real fine-entry path submits this measured hold before enabling fine.
    node._send_gripper_goal(.3)
    node.mode = m.Mode.FINE
    node._update_fine_gripper(1.)
    assert node.command_gripper == 1.
    timestamps = []
    for _ in range(20):
        clock[0] += .01
        node._measured_gripper_callback(NS(name=['robotiq_85_left_knuckle_joint'], position=[.48]))
        assert node.measured_gripper == pytest.approx(.4)
        assert node.command_gripper == 1.
        node._update_fine_gripper(1.)
        node._publish_commands(clock[0]); timestamps.append(clock[0])
    assert len(goals) == 2
    assert max(np.diff(timestamps)) < .1
    assert node.gripper_action_command_publisher.values[-20:] == [1.] * 20
    node._update_fine_gripper(0.)
    assert node.command_gripper == pytest.approx(.4)
    node.measured_gripper = .46
    node._update_fine_gripper(0.)
    node._publish_commands(clock[0])
    assert len(goals) == 3 and node.gripper_action_command_publisher.values[-1] == pytest.approx(.4)
    node.raw_gripper_time = clock[0]
    assert node._request_mode(False)[0]
    assert node.mode == m.Mode.RETURNING
    node._publish_commands(clock[0])
    assert node.gripper_action_command_publisher.values[-1] == .8 and len(goals) == 4
    assert node.gripper_drive_publisher.values == [.8]  # No duplicate drive during fine/return.
    node.mode = m.Mode.GELLO
    node._update_gello_mode(clock[0]); node._publish_commands(clock[0])
    assert node.gripper_drive_publisher.values == [.8, .8]
    before = len(node.gripper_action_command_publisher.values)
    node.managed_enabled = False
    node._publish_commands(clock[0])
    node.managed_enabled = True; node.managed_fault = 'fault'
    node._publish_commands(clock[0])
    node.managed_fault = ''; clock[0] += 1.
    node._publish_commands(clock[0])
    assert len(node.gripper_action_command_publisher.values) == before
    node.mode = m.Mode.FINE
    before_goal = node.command_gripper
    def fail(_): raise RuntimeError('submission failed')
    node.gripper_action_client.send_goal_async = fail
    with pytest.raises(RuntimeError, match='submission failed'):
        node._update_fine_gripper(-1.)
    assert node.command_gripper == before_goal and node.gripper_direction == 0
    rejected = Future(); rejected.set_result(NS(accepted=False))
    node._gripper_goal_response(rejected)
    assert any('rejected' in error for error in errors)


@pytest.mark.parametrize('arms', [('left',), ('left', 'right')])
def test_new_gripper_contract_rejects_old_resume_before_recovery(tmp_path, arms):
    fields = [f'{arm}_{i}' for arm in arms for i in range(50)]
    actions = [f'{arm}_{i}' for arm in arms for i in range(10)]
    args = (tmp_path, 30, 'task', ('observation.images.base',), 16, 16)
    kwargs = dict(state_fields=fields, action_fields=actions)
    V30DatasetWriter(*args, **kwargs)
    recovery = tmp_path / 'data' / 'uncommitted.tmp'
    recovery.parent.mkdir(exist_ok=True); recovery.write_text('retain')
    metadata = gello_recording_metadata({arm: f'/{arm}/teleop/gripper_action_width_percent' for arm in arms})
    with pytest.raises(RuntimeError, match='gripper_recording_version'):
        V30DatasetWriter(*args, **kwargs, info_metadata=metadata, strict_metadata=True)
    assert recovery.read_text() == 'retain'
    fresh = (tmp_path / 'new', *args[1:])
    V30DatasetWriter(*fresh, **kwargs, info_metadata=metadata, strict_metadata=True)
    reopened = V30DatasetWriter(*fresh, **kwargs, info_metadata=metadata, strict_metadata=True)
    assert reopened.info['gripper_command_topics'] == metadata['gripper_command_topics']


def test_collector_preflight_requires_drive_consumer_not_recording_listener(ros_transport):
    c = object.__new__(ros_transport.RolloutCollector)
    c.config = load_rollout_config(DEFAULT_CONFIG)
    c._camera_failures = lambda: {}
    c.count_publishers = lambda _: 1
    missing = set()
    c.count_subscribers = lambda topic: 0 if topic in missing else 1
    c._gripper_action_clients = {a.name: NS(server_is_ready=lambda: True) for a in c.config.arms}
    c.policy_client = None
    c._snapshot = lambda **_: None
    missing.update(a.command_gripper_topic for a in c.config.arms)
    assert c._preflight_problems() == []
    c._gripper_action_clients[c.config.arms[0].name].server_is_ready = lambda: False
    problems = c._preflight_problems()
    assert len(problems) == 1 and 'gripper' in problems[0]


@pytest.mark.parametrize("managed", [False, True], ids=["terminal", "web"])
def test_recorder_topic_override_and_live_driver_checks(ros_transport, monkeypatch, managed):
    # Execute real recorder construction, replacing only I/O and device boundaries.
    monkeypatch.setitem(sys.modules, 'pyrealsense2', ModuleType('pyrealsense2'))
    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[2] / 'src/vla/datacollection' / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module
    single = load('vla.datacollection.franka_robotiq_single_left.auto_record', 'franka_robotiq_single_left/auto_record.py')
    dual = load('vla.datacollection.franka_robotiq_dual._test_recorder', 'franka_robotiq_dual/auto_record.py')
    topics = {a: f'/{a}/teleop/gripper_action_width_percent' for a in ('left', 'right')} if managed else {}
    monkeypatch.setattr(dual, 'V30DatasetWriter', lambda *args, **kw: NS(info=kw['info_metadata'], episode_count=0))
    cls = dual.DualLeRobotRecorder
    monkeypatch.setattr(cls, '_validate_existing_dataset', lambda _: None)
    monkeypatch.setattr(cls, '_load_robot_description', lambda *_: '')
    monkeypatch.setattr(dual, 'CommandPoseKinematics', lambda *_: None)
    monkeypatch.setattr(dual, 'QoSProfile', lambda **_: None)
    monkeypatch.setattr(dual, 'HistoryPolicy', NS(KEEP_LAST=1))
    monkeypatch.setattr(dual, 'ReliabilityPolicy', NS(BEST_EFFORT=1))
    subscribed = []
    monkeypatch.setattr(cls, 'create_subscription', lambda self, kind, topic, *args: subscribed.append(topic), raising=False)
    monkeypatch.setattr(dual, 'DirectRealSenseCapture', lambda *_: NS(start=lambda: None))
    recorder = cls(gripper_command_topics=topics)
    right = recorder.arms[1].gripper_command_topic
    assert right == (topics['right'] if managed else dual.schema.ARMS[1].gripper_command_topic)
    assert right in subscribed and recorder.dataset.info['gripper_command_topics']['right'] == right
    from vla.datacollection.gripper_contract import gripper_consumer_requirement
    counts = {}
    for arm in recorder.arms:
        counts[arm.command_joint_topic] = 2
        topic, minimum = gripper_consumer_requirement(arm.gripper_command_topic)
        counts[topic] = minimum
    recorder.count_subscribers = lambda topic: counts.get(topic, 0)
    assert recorder.unconsumed_command_topics() == []
    counts[gripper_consumer_requirement(right)[0]] -= 1
    assert recorder.unconsumed_command_topics() == [gripper_consumer_requirement(right)[0]]
    left = object.__new__(single.DirectLeRobotRecorder)
    left.count_subscribers = recorder.count_subscribers
    assert left.unconsumed_command_topics() == []
    counts[gripper_consumer_requirement(single.schema.TOPIC_GRIPPER_COMMAND)[0]] = 0
    assert left.unconsumed_command_topics() == [gripper_consumer_requirement(single.schema.TOPIC_GRIPPER_COMMAND)[0]]
