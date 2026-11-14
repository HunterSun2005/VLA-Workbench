"""Subprocess acceptance adapter: real wheel tracker, fake Robot, Unix IPC only."""
import argparse
import functools
import json
from pathlib import Path
import signal
import socket
import time
from types import SimpleNamespace

import numpy as np
import franky
from vla.control.franky.binding import check_binding
from vla.control.franky import worker
from vla.core.hardware_lease import HardwareLease

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--socket-path', type=Path, required=True)
args = parser.parse_args()
config = json.loads(args.config.read_text())
root = args.config.parent
(root/'binding.json').write_text(json.dumps(check_binding()))
original_connect = socket.socket.connect

def connect(self, address):
    if self.family != socket.AF_UNIX:
        raise AssertionError('Hardware/network connection forbidden in mock acceptance')
    return original_connect(self, address)
socket.socket.connect = connect

class Robot:
    def __init__(self, ip, **kwargs):
        if (root/f'fail-open-{ip}').exists():
            raise RuntimeError('injected open failure')
        self.ip = ip
        self.active = False
        self.pose = franky.Affine(np.eye(4))
        self.stamp = time.monotonic()

    @property
    def current_pose(self):
        return SimpleNamespace(end_effector_pose=self.pose)

    @property
    def state(self):
        if (root/f'fail-state-{self.ip}').exists():
            raise RuntimeError('injected state failure')
        if not (root/'stale-state').exists():
            self.stamp = time.monotonic()
        tool = config.get('tool_pose', [0, 0, 0, 0, 0, 0, 1])
        if (root/'hardware-tool.json').exists():
            tool = json.loads((root/'hardware-tool.json').read_text())
        tool_affine = worker._pose_to_affine(franky, np.asarray(tool))
        return SimpleNamespace(O_T_EE=self.pose, O_T_EE_d=self.pose, F_T_EE=tool_affine,
                               q=np.zeros(7), dq=np.zeros(7), time=SimpleNamespace(to_sec=lambda: self.stamp))

    @property
    def is_in_control(self):
        return self.active

    def move(self, motion, asynchronous, limit_rate=False, cutoff_frequency=100.):
        assert asynchronous and not limit_rate and cutoff_frequency == 100.
        assert isinstance(motion, (franky.CartesianVelocityMotion, franky.CartesianVelocityStopMotion))
        if isinstance(motion, franky.CartesianVelocityStopMotion):
            self.active = False
            self.event('stop')
        else:
            self.active = True
            self.event('move')

    def event(self, value):
        with (root/'robot-events.jsonl').open('a') as f:
            f.write(json.dumps([self.ip, value])+'\n')

    def stop(self):
        self.active = False
        self.event('stop')

    def join_motion(self, timeout=None):
        if (root/f'fail-join-{self.ip}').exists():
            raise RuntimeError('injected stopped motion error')
    def poll_motion(self):
        if (root/f'fail-tracker-{self.ip}').exists():
            raise RuntimeError('injected asynchronous tracker error')
    def recover_from_errors(self):
        self.event('recover')
        (root/f'fail-join-{self.ip}').unlink(missing_ok=True)
    def set_collision_behavior(self, *args): pass

# Exercise real 2.0.0 motion constructors with a fixed-state Robot boundary.
real_servo = worker.FrankyArm.servo_once
def capture_target(self):
    result = real_servo(self)
    if result and (root/'follow-targets').exists():
        self.robot.pose = worker._pose_to_affine(franky, self._frame().to_hardware(self.target_pose))
    if result and (root/'capture-targets').exists():
        pose = self._frame().to_hardware(self.target_pose)
        with (root/'targets.jsonl').open('a') as stream:
            stream.write(json.dumps({'translation': pose[:3].tolist(), 'quaternion': pose[3:].tolist()})+'\n')
    return result
worker.FrankyArm.servo_once = capture_target
franky.Robot = Robot
worker.HardwareLease = functools.partial(HardwareLease, directory=root/'leases')
instance = worker.FrankyWorker(config['arms'], session_id=config['session_id'],
                              target_timeout_s=config.get('target_timeout_s', .1),
                              state_timeout_s=config.get('state_timeout_s', .5))
server = worker.FrankyWorkerServer(instance, args.socket_path)
signal.signal(signal.SIGTERM, lambda *args: server.close())
signal.signal(signal.SIGINT, lambda *args: server.close())
server.serve()
