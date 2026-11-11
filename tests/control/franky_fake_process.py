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
        self.current_pose = SimpleNamespace(end_effector_pose=self.pose)
        self.stamp = time.monotonic()

    @property
    def state(self):
        if (root/f'fail-state-{self.ip}').exists():
            raise RuntimeError('injected state failure')
        if not (root/'stale-state').exists():
            self.stamp = time.monotonic()
        return SimpleNamespace(O_T_EE=self.pose, O_T_EE_d=self.pose,
                               q=np.zeros(7), dq=np.zeros(7), time=SimpleNamespace(to_sec=lambda: self.stamp))

    @property
    def is_in_control(self):
        return self.active and not (root/f'fail-tracker-{self.ip}').exists()

    def move(self, motion, asynchronous):
        assert isinstance(motion, franky.CartesianImpedanceTrackingMotion) and asynchronous
        self.active = True
        self.event('move')

    def event(self, value):
        with (root/'robot-events.jsonl').open('a') as f:
            f.write(json.dumps([self.ip, value])+'\n')

    def stop(self):
        self.active = False
        self.event('stop')

    def join_motion(self):
        if (root/f'fail-join-{self.ip}').exists():
            raise RuntimeError('injected stopped motion error')
    def poll_motion(self):
        if (root/f'fail-tracker-{self.ip}').exists():
            raise RuntimeError('injected asynchronous tracker error')
    def recover_from_errors(self):
        self.event('recover')
        (root/f'fail-join-{self.ip}').unlink(missing_ok=True)
    def set_collision_behavior(self, *args): pass

franky.Robot = Robot
worker.HardwareLease = functools.partial(HardwareLease, directory=root/'leases')
instance = worker.FrankyWorker(config['arms'], session_id=config['session_id'],
                              target_timeout_s=config.get('target_timeout_s', .1),
                              state_timeout_s=config.get('state_timeout_s', .5))
server = worker.FrankyWorkerServer(instance, args.socket_path)
signal.signal(signal.SIGTERM, lambda *args: server.close())
signal.signal(signal.SIGINT, lambda *args: server.close())
server.serve()
