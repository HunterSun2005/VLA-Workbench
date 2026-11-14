import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from vla.actions import ActionSpec, matrix_to_rot6d
from vla.control.franky import VelocityControl, FrankyWorkerClient
from vla.control.franky.ipc import decode_packet, encode_packet
from vla.control.franky.worker import FrankyWorkerServer
from vla.control.franky.worker import FrankyArm
from vla.control.franky.worker import FrankyWorker
from vla.control.ros import ArmMeasurement, ControlCoordinator


class _FakeWorker:
    state_timeout_s = .5

    def __init__(self):
        self.opened = False
        self.closed = False
        self.requests = []

    def open(self):
        self.opened = True

    def request(self, message):
        self.requests.append(dict(message))
        return {'kind': 'state', 'session_id': message.get('session_id'),
                'request_id': str(message.get('request_id')), 'generation': message.get('generation', -1),
                'sequence': message.get('sequence'), 'slot_id': message.get('slot_id'),
                'applied_at_ns': time.monotonic_ns(),
                'references': {name: value['pose'] for name, value in message.get('arms', {}).items()},
                'request_status': 'applied', 'states': [{
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
    client.target('session', 3, 4, {'left': {'pose': [0, 0, 0, 0, 0, 0, 1], 'context': {'source':'policy','dt':1/30}}}, slot_id=0)
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


def test_worker_drops_stale_targets_and_requires_atomic_dual_arm_packet():
    from tests.control.test_velocity_control import fake_arm
    worker = FrankyWorker({'left': {'robot_ip':'mock'}, 'right': {'robot_ip':'mock2'}},session_id='s')
    worker.arms = {name:fake_arm() for name in ('left','right')}
    for name,arm in worker.arms.items():
        arm.name=name
    worker._watchdog=threading.Thread(target=worker._watch,daemon=True)
    worker._watchdog.start()
    pose = [0, 0, 0, 0, 0, 0, 1]
    worker.request({'kind': 'reset', 'session_id': 's', 'generation': 1, 'request_id': 'reset'})
    base = {'kind': 'target', 'session_id': 's', 'generation': 1, 'request_id': 'target1',
            'sequence': 1, 'slot_id': 0, 'timestamp_ns': time.monotonic_ns(),
            'arms': {'left': {'pose': pose, 'context': {'source':'policy','dt':1/30}}, 'right': {'pose': pose, 'context': {'source':'policy','dt':1/30}}}}
    worker.request(base)
    assert worker.arms['left'].target_sequence == 1
    assert worker.arms['right'].target_sequence == 1
    stale = dict(base, sequence=0, timestamp_ns=time.monotonic_ns(), request_id='stale')
    worker.request(stale)
    assert worker.arms['left'].target_sequence == 1
    incomplete = dict(base, sequence=2, slot_id=1, timestamp_ns=time.monotonic_ns(), request_id='incomplete',
                      arms={'left': {'pose': pose}})
    result = worker.request(incomplete)
    assert 'every configured' in result['error']
    assert worker.fault
    worker.close()


def test_lifecycle_fence_cancels_dequeued_target(tmp_path):
    """A target waiting for the IPC mutex must not slip past stop."""
    class Client(FrankyWorkerClient):
        def _request(self, message):
            sent.append(message['kind'])
            return {'kind': 'state', 'session_id': message['session_id'],
                    'request_id': message['request_id'], 'generation': message['generation'],
                    'request_status': 'applied', 'states': []}
    sent = []
    client = Client(tmp_path/'unused')
    try:
        with client._request_lock:
            client.target('s', 1, 1, {'left': {'pose': [0, 0, 0, 0, 0, 0, 1], 'context': {'source':'policy','dt':1/30}}}, slot_id=0)
            deadline = time.monotonic()+2
            while time.monotonic() < deadline:
                with client._lock:
                    if not client._queue:
                        break
                time.sleep(.001)
            else:
                raise AssertionError('IPC thread did not dequeue target')
            client.stop('s', 2)
        deadline = time.monotonic()+2
        while not sent and time.monotonic() < deadline:
            time.sleep(.001)
        assert sent == ['stop']
    finally:
        client.close()


def test_packet_deadline_is_shared_by_header_and_body(monkeypatch):
    from vla.control.franky import ipc
    clock = [0.]
    monkeypatch.setattr(ipc.time, 'monotonic', lambda: clock[0])
    packet = encode_packet({'kind': 'health'})

    class DelayedSocket:
        def settimeout(self, timeout):
            self.timeout = timeout

        def recv(self, size):
            if self.timeout < .06:
                clock[0] += self.timeout
                raise TimeoutError('timed out')
            clock[0] += .06
            return packet[:4] if size == 4 else packet[4:]

    with pytest.raises(TimeoutError):
        decode_packet(DelayedSocket(), deadline=.1)
    assert clock[0] == pytest.approx(.1)


def test_client_transport_error_identifies_request_and_stage(tmp_path):
    client = FrankyWorkerClient(tmp_path/'missing.sock')
    try:
        with pytest.raises(RuntimeError, match='Franky IPC health connect failed.*request_id=probe'):
            client.request_now({'kind': 'health', 'session_id': 's', 'generation': -1, 'request_id': 'probe'})
        assert 'FileNotFoundError' in client.fault
        assert str(tmp_path/'missing.sock') in client.fault
    finally:
        client.close()


def test_full_audit_buffer_stops_before_another_submission_and_retains_evidence():
    from tests.control.test_velocity_control import fake_arm
    worker = FrankyWorker({'left': {'robot_ip':'mock'}}, session_id='s')
    worker.arms = {'left': fake_arm()}
    worker.request(dict(kind='reset',session_id='s',generation=1,request_id='reset'))
    for _ in range(4096):
        worker._record_servo({'generation':1, 'sequence':0})
    result = worker.request(dict(kind='target',session_id='s',generation=1,request_id='full',
        sequence=1,slot_id=0,timestamp_ns=time.monotonic_ns(),
        arms={'left':{'pose':[0,0,0,0,0,0,1],'context':{'source':'policy','dt':1/30}}}))
    assert 'audit buffer full' in result['fault']
    assert not worker.arms['left'].robot.moves and not worker.arms['left'].servo_ready
    assert len(worker._audit) == 4096
    result = worker.request(dict(kind='health',session_id='s',generation=1,request_id='drain',audit_ack=128))
    assert result['fault'] and len(worker._audit) == 4096-128
    assert result['servo_events'][0]['audit_id'] == 129
    assert not worker.arms['left'].servo_ready
    worker.close()
