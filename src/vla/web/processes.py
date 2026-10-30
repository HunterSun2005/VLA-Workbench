"""Owned child processes and explicit, isolated interpreter selection."""
from __future__ import annotations
import os
from pathlib import Path
import signal
import subprocess
import time

HARDWARE_ENV = frozenset({'LEFT_ROBOT_IP','RIGHT_ROBOT_IP','LEFT_ROBOTIQ_PORT','RIGHT_ROBOTIQ_PORT',
    'LEFT_GELLO_SERIAL_ID','RIGHT_GELLO_SERIAL_ID','BASE_CAMERA_SERIAL','LEFT_WRIST_CAMERA_SERIAL',
    'RIGHT_WRIST_CAMERA_SERIAL','GELLO_CONFIG','ROBOT_CONFIG','GRIPPER_CONFIG'})


def hardware_environment(values):
    if not isinstance(values, dict) or set(values) - HARDWARE_ENV:
        raise ValueError('unknown ROS hardware setting')
    # Existing launcher uses shell interpolation internally; accept only literal identifiers/paths.
    import re
    if any(not isinstance(v, str) or not re.fullmatch(r'[A-Za-z0-9_./:@+-]+', v) for v in values.values()):
        raise ValueError('hardware values must be literal paths/identifiers without shell syntax')
    return values


def clean_environment(root, python, *, ros=False, workspace=None):
    env = dict(os.environ)
    for key in ('PYTHONHOME','PYTHONPATH','AMENT_PREFIX_PATH','COLCON_PREFIX_PATH','CMAKE_PREFIX_PATH',
                'LD_LIBRARY_PATH','ROS_DISTRO','ROS_VERSION','ROS_PYTHON_VERSION','CONDA_PREFIX','CONDA_DEFAULT_ENV'):
        env.pop(key, None)
    env.update(VLA_ROOT=str(root), VLA_PYTHON=str(python), PYTHONNOUSERSITE='1',
               PYTHONPATH=str(Path(__file__).resolve().parents[2]),
               PATH=f'{Path(python).parent}:/usr/bin:/bin:/usr/sbin:/sbin',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env


def interpreter_command(python, args, *, ros=False, root=None, workspace=None):
    if not ros:
        return [str(python), *args]
    root = Path(root)
    # Only fixed shell source expressions; external paths are positional arguments.
    script = ('source /opt/ros/humble/setup.bash; source "$1/install/local_setup.bash"; '
              'export PYTHONPATH="$2/ros_ws/python:$2/ros_ws/python_deps:$3:${PYTHONPATH:-}"; '
              'export LD_LIBRARY_PATH="/opt/openrobots/lib:${LD_LIBRARY_PATH:-}"; '
              'export CMAKE_PREFIX_PATH="/opt/openrobots:${CMAKE_PREFIX_PATH:-}"; '
              'shift 3; exec /usr/bin/python3 "$@"')
    return ['/bin/bash','--noprofile','--norc','-ec',script,'vla-ros',
            str(workspace or root/'ros_ws'),str(root),str(root/'src'),*args]


class Children:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.items = []

    def start(self, name, argv, env):
        with (self.directory / f'{name}.log').open('ab') as log:
            process = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.items.append((name, process))
        return process

    def check(self):
        for name, process in self.items:
            if process.poll() is not None:
                raise RuntimeError(f'{name} exited ({process.returncode}); see {name}.log')

    def close(self):
        for name, process in reversed(self.items):
            if process.poll() is not None:
                continue
            # Launchers clean their own process groups; signal their parent first.
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
