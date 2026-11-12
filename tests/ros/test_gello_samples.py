"""Source GELLO adapter with fake serial samples; never construct a device/Node."""
import importlib
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest
from sensor_msgs.msg import JointState
from rclpy.time import Time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'ros_ws/src/franka_gello_state_publisher'))
from franka_gello_state_publisher.dynamixel.driver import DynamixelDriver
from franka_gello_state_publisher.gello_hardware import GelloHardware
from franka_gello_state_publisher.gello_publisher import GelloPublisher
from vla.datacollection.franka_robotiq_rollout.collect_ros import RolloutCollector


def fake_driver():
    driver = DynamixelDriver.__new__(DynamixelDriver)
    driver._sample_lock = threading.Lock()
    driver._sample_sequence = 0
    driver._joint_sample = None
    driver._pulses_per_revolution = 4096
    return driver


def test_serial_failure_does_not_renew_cache(monkeypatch):
    driver = fake_driver()
    driver._cache_joint_sample(np.arange(8), 100)
    original = driver.get_joint_sample()
    stop = threading.Event()
    driver._stop_thread = stop
    def broken(_):
        stop.set()
        raise RuntimeError('serial disconnected')
    driver.read_value_by_name = broken
    driver._joint_polling_loop()
    after = driver.get_joint_sample()
    np.testing.assert_array_equal(after[0], original[0])
    assert after[1:] == original[1:] == (100, 1)
    after[0][:] = 42
    assert driver.get_joint_sample()[0][0] == 0


def test_hardware_only_processes_each_sample_once():
    hardware = GelloHardware.__new__(GelloHardware)
    hardware._driver = fake_driver()
    hardware._num_arm_joints = 7
    hardware.process_arm_joint_positions = lambda v: v
    hardware.process_gripper_position = lambda v: float(v)
    assert hardware.get_position_sample() is None
    hardware._driver._cache_joint_sample(np.arange(8), 100)
    assert hardware.get_position_sample()[2:] == (100, 1)
    assert hardware.get_position_sample() is None
    hardware._driver._cache_joint_sample(np.arange(8)+1, 200)
    assert hardware.get_position_sample()[2:] == (200, 2)


def test_publisher_keeps_capture_time_and_synchronized_gripper():
    published, grips = [], []
    node = GelloPublisher.__new__(GelloPublisher)
    now = time.monotonic_ns()
    node.get_clock = lambda: NS(now=lambda: Time(nanoseconds=10_000_000_000))
    samples = iter([(np.arange(7, dtype=float), .4, now-20_000_000, 1), None,
                    (np.arange(7, dtype=float), .8, now-1_000_000_000, 2)])
    node.gello_hardware = NS(get_position_sample=lambda: next(samples))
    node.arm_joint_publisher = NS(publish=published.append)
    node.gripper_joint_publisher = NS(publish=grips.append)
    for _ in range(3): node.publish_joint_jog()
    assert len(published) == len(grips) == 1
    msg = published[0]
    stamp = msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
    assert 9_970_000_000 < stamp <= 9_980_000_000
    assert msg.name[-1] == 'gello_gripper' and msg.position[-1] == .4


def test_collector_rejects_old_duplicate_future_and_unsynchronized_samples():
    c = RolloutCollector.__new__(RolloutCollector)
    c._lock = threading.RLock()
    c.config = NS(gello_sample_timeout_s=.2)
    c._input_stamps, c._input_diagnostics, c._latest, c._sensor_sequences = {}, {}, {}, {}
    now = 10_000_000_000
    c.get_clock = lambda: NS(now=lambda: Time(nanoseconds=now))
    msg = JointState()
    msg.header.stamp = Time(nanoseconds=now-20_000_000).to_msg()
    msg.name = [f'fr3_joint{i}' for i in range(1,8)] + ['gello_gripper']
    msg.position = [0.]*7+[.4]
    c._store_gello_sample('left.gello_joint', msg)
    captured = c._latest['left.gello_joint'][0]
    assert c._latest['left.gello_gripper'][0] == captured
    assert 15e6 < time.monotonic_ns()-captured < 40e6
    for stamp in (now-20_000_000, now-30_000_000, now-300_000_000, now+1):
        msg.header.stamp = Time(nanoseconds=stamp).to_msg()
        c._store_gello_sample('left.gello_joint', msg)
        assert c._latest['left.gello_joint'][0] == captured
    msg.header.stamp = Time(nanoseconds=now).to_msg()
    msg.name = msg.name[:-1]; msg.position = msg.position[:-1]
    c._store_gello_sample('left.gello_joint', msg)
    assert c._latest['left.gello_joint'][0] == captured
