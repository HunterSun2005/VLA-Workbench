import socket
import threading
import time
from pathlib import Path

import numpy as np

from vla.actions import ActionSpec, matrix_to_rot6d
from vla.control.franky import Compliance, FrankyWorkerClient
from vla.control.franky.ipc import decode_packet, encode_packet
from vla.control.franky.worker import FrankyWorkerServer
from vla.control.franky.worker import FrankyArm
from vla.control.franky.worker import FrankyWorker
from vla.control.ros import ArmMeasurement, ControlCoordinator


def test_compliance_defaults_and_safety_bounds():
    value = Compliance().validate()
    assert value.translational_stiffness == 1000
    assert value.translational_clip == .008
    assert Compliance.from_mapping({'rotational_stiffness': 40}).rotational_stiffness == 40
    try:
        Compliance.from_mapping({'translational_stiffness': 1201})
    except ValueError as error:
        assert 'translational_stiffness' in str(error)
    else:
        raise AssertionError('unsafe stiffness was accepted')


class _FakeWorker:
    def __init__(self):
        self.opened = False
        self.closed = False
        self.requests = []

    def open(self):
        self.opened = True

    def request(self, message):
        self.requests.append(dict(message))
        return {'kind': 'state', 'states': [{
            'arm': 'left', 'observed_at_ns': 1,
            'tcp_pose': [0, 0, 0, 0, 0, 0, 1],
            'joints': [0] * 7, 'joint_velocity': [0] * 7,
        }]}

    def snapshot(self):
        return self.request({'kind': 'health'})

    def close(self):
        self.closed = True


def test_worker_ipc_client_receives_state(tmp_path):
    path = tmp_path / 'worker.sock'
    fake = _FakeWorker()
    server = FrankyWorkerServer(fake, path)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    deadline = time.monotonic() + 2
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    client = FrankyWorkerClient(path, session_id='session')
    deadline = time.monotonic() + 2
    while not client.states() and time.monotonic() < deadline:
        time.sleep(.01)
    assert client.states()['left'].joints == (0.0,) * 7
    client.target('session', 3, 4, {'left': {'pose': [0, 0, 0, 0, 0, 0, 1]}})
    time.sleep(.05)
    client.close()
    server.close()
    thread.join(timeout=1)
    assert any(item['kind'] == 'target' for item in fake.requests)


def test_franky_coordinator_submits_tcp_pose_without_ik():
    published = []
    resets = []
    c = ControlCoordinator(
        ActionSpec('tcp', ('left',)), {},
        {'left': (np.array([-.5, -.5, 0]), np.array([.5, .5, 1]))},
        lambda *args: published.append(args), resets.append,
        backend='franky', clock=lambda: 1_000_000_000,
    )
    c.observe('left', ArmMeasurement(np.zeros(7), .5, np.zeros(3), np.eye(3), 1_000_000_000))
    c.transition('human')
    c.acknowledge('left', c.generation, 0, np.zeros(7), 1_000_000_000)
    target = np.r_[np.array([.1, .2, .3]), matrix_to_rot6d(np.eye(3)), .5]
    c.accept(target, source='human', generation=c.generation, slot_id=0)
    c.tick(.01)
    assert published[0][0] == 'left'
    np.testing.assert_allclose(published[0][1], [.1, .2, .3, 0, 0, 0, 1])


def test_fake_cartesian_tracker_gains_rebuild_and_recover():
    class Affine:
        def __init__(self, matrix):
            self.translation = np.asarray(matrix)[:3, 3]
            self.quaternion = np.array([0., 0., 0., 1.])

    class Tracker:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            self.targets = []
            self.is_running = True
            self.gains = []
        def set_target(self, target): self.targets.append(target)
        def set_gains(self, **kwargs): self.gains.append(kwargs)
        def stop(self): self.is_running = False

    class Robot:
        def __init__(self):
            self.state = type('State', (), {
                'O_T_EE': Affine(np.eye(4)), 'q': np.zeros(7), 'dq': np.zeros(7),
                'q_d': np.zeros(7), 'K_F_ext_hat_K': np.zeros(6)})()
        def join_motion(self): pass
        def recover_from_errors(self): pass

    fake = type('Franky', (), {'Affine': Affine, 'CartesianImpedanceTracker': Tracker})
    arm = FrankyArm('left', '127.0.0.1')
    arm.franky, arm.robot = fake, Robot()
    arm._ensure_tracker()
    arm.set_target([.01, 0, 0, 0, 0, 0, 1], 2, 3)
    assert len(arm.tracker.targets) == 1
    arm.update_compliance({'translational_stiffness': 900})
    assert arm.tracker.gains[-1]['translational_stiffness'] == 900
    arm.update_compliance({'translational_clip': .01})
    assert arm.tracker is not None and arm.tracker.kwargs['translational_error_clip'][0] == .01
    arm.fault = 'tracker failed'
    try:
        arm.reset(4)
    except RuntimeError as error:
        assert 'tracker failed' in str(error)
    else:
        raise AssertionError('faulted tracker reset was accepted')
    arm.recover()
    assert arm.fault is None
    arm.close()


def test_worker_drops_stale_targets_and_requires_atomic_dual_arm_packet():
    class Arm:
        def __init__(self, name):
            self.name = name
            self.calls = []
            self.fault = None
            self.compliance = Compliance()
            self.binding_fingerprint = {}

        def set_target(self, pose, generation, sequence):
            self.calls.append((pose, generation, sequence))

        def reset(self, generation):
            pass

        def stop(self):
            pass

        def recover(self):
            pass

        def update_compliance(self, values):
            self.compliance = Compliance.from_mapping(values, self.compliance)

        def _state(self):
            return {
                'arm': self.name,
                'observed_at_ns': 1,
                'tcp_pose': [0, 0, 0, 0, 0, 0, 1],
                'joints': [0] * 7,
                'joint_velocity': [0] * 7,
            }

    worker = FrankyWorker.__new__(FrankyWorker)
    worker.session_id = 's'
    worker.arms = {'left': Arm('left'), 'right': Arm('right')}
    worker._lock = threading.RLock()
    worker.generation = -1
    worker.sequence = -1
    worker.timestamp_ns = -1
    worker.fault = None
    worker.binding_fingerprint = {}
    pose = [0, 0, 0, 0, 0, 0, 1]
    base = {'kind': 'target', 'session_id': 's', 'generation': 1,
            'sequence': 1, 'timestamp_ns': 10,
            'arms': {'left': {'pose': pose}, 'right': {'pose': pose}}}
    worker.request(base)
    assert [call[2] for call in worker.arms['left'].calls] == [1]
    assert [call[2] for call in worker.arms['right'].calls] == [1]
    stale = dict(base, sequence=0, timestamp_ns=11)
    worker.request(stale)
    assert len(worker.arms['left'].calls) == 1
    incomplete = dict(base, sequence=2, timestamp_ns=12,
                      arms={'left': {'pose': pose}})
    try:
        worker.request(incomplete)
    except ValueError as error:
        assert 'every configured' in str(error)
    else:
        raise AssertionError('incomplete dual-arm target was accepted')
