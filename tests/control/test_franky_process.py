"""Real wheel in an isolated process, fake Robot only. No device APIs are opened."""
from contextlib import contextmanager
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import uuid

import pytest

from vla.control.franky.ipc import encode_packet, decode_packet, FrankyWorkerClient
from vla.core.franky_runtime import worker_environment, worker_python
from vla.core.hardware_lease import HardwareLease

ROOT = Path(__file__).resolve().parents[2]
FAKE = Path(__file__).with_name('franky_fake_process.py')
CONTEXT = {'source': 'policy', 'dt': 1/30}
POSE = [0.01, 0., 0., 0., 0., 0., 1.]


def wait_for(function, timeout=5):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        result = function()
        if result:
            return result
        time.sleep(.01)
    raise AssertionError('condition timed out')


@contextmanager
def process_worker(arms=('left',), target_timeout=3., state_timeout=3., tool_pose=None, workspace_bounds=None):
    python = worker_python(ROOT)
    if not python.is_file():
        pytest.skip('persistent Franky wheel environment not installed')
    with tempfile.TemporaryDirectory(prefix='vfw-') as temporary:
        root = Path(temporary)
        cfg = {'session_id': 's', 'arms': {name: {'robot_ip': name} for name in arms},
               'target_timeout_s': target_timeout, 'state_timeout_s': state_timeout}
        if tool_pose is not None:
            cfg['tool_pose'] = tool_pose
        if workspace_bounds is not None:
            for arm in cfg['arms'].values():
                arm['workspace_bounds'] = workspace_bounds
        config = root/'config.json'
        config.write_text(json.dumps(cfg))
        log = (root/'process.log').open('w+')
        argv = [str(python), str(FAKE), '--config', str(config), '--socket-path', str(root/'w.sock')]
        child = subprocess.Popen(argv, env=worker_environment(ROOT, python), stdout=log, stderr=log)
        try:
            def ready():
                if child.poll() is not None:
                    log.seek(0)
                    raise AssertionError(log.read())
                return (root/'w.sock').exists()
            wait_for(ready, timeout=30)
            yield root, child, argv
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                    pytest.fail('worker did not shut down promptly')
            log.close()


def request(root, kind='health', generation=-1, **kwargs):
    if kind == 'target':
        kwargs['arms'] = {name: dict(value, context=value.get('context', CONTEXT)) for name, value in kwargs['arms'].items()}
        kwargs.setdefault('slot_id', kwargs.get('sequence', 0))
    packet = {'session_id': 's', 'kind': kind, 'generation': generation,
              'request_id': uuid.uuid4().hex, **kwargs}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        sock.connect(str(root/'w.sock'))
        sock.sendall(encode_packet(packet))
        return decode_packet(sock)


def target(root, generation=1, sequence=1, arms=('left',), **kw):
    return request(root, 'target', generation, sequence=sequence,
                   timestamp_ns=time.monotonic_ns(), arms={a: {'pose': POSE} for a in arms}, **kw)


@pytest.mark.parametrize('arms', [('left',), ('left', 'right')])
def test_real_tracker_lifecycle_and_explicit_recovery(arms):
    with process_worker(arms) as (root, child, _):
        ready = request(root)
        assert ready['binding_fingerprint']['libfranka'] == '0.21.2'
        assert not ready['fault'] and all(v['state_valid'] for v in ready['states'])
        assert request(root, 'reset', 1)['request_status'] == 'applied'
        assert not target(root, arms=arms)['fault']
        for seq in range(2, 6):
            assert not target(root, sequence=seq, arms=arms)['fault']
        gains = request(root, 'velocity_control', 1, values={'max_linear_speed': .04})
        assert gains['request_status'] == 'applied' and gains['velocity_control']['max_linear_speed'] == .04
        assert request(root, 'velocity_control', 1, values={'tracking_tau_s': .2})['request_status'] == 'applied'
        assert request(root, 'stop', 2)['request_status'] == 'applied'
        assert target(root, generation=1, sequence=6, arms=arms)['request_status'] == 'ignored'
        assert not any(s['servo_ready'] for s in request(root)['states'])
        assert request(root, 'reset', 3)['request_status'] == 'applied'
        assert not target(root, generation=3, sequence=1, arms=arms)['fault']
        (root/f'fail-tracker-{arms[-1]}').touch()
        wait_for(lambda: request(root)['fault'])
        events = [json.loads(line) for line in (root/'robot-events.jsonl').read_text().splitlines()]
        assert all([arm, 'stop'] in events for arm in arms)
        assert not any(event == 'recover' for _, event in events)  # No implicit recovery.
        (root/f'fail-tracker-{arms[-1]}').unlink()
        assert request(root, 'reset', 4)['fault']
        assert request(root)['fault']
        assert not request(root, 'recover', 5)['fault']
        assert not any(s['servo_ready'] for s in request(root)['states'])
        assert not request(root, 'reset', 6)['fault']
        assert not target(root, generation=6, arms=arms)['fault']
        child.terminate(); child.wait(timeout=5)
        assert not (root/'w.sock').exists()
        for arm in arms:
            with HardwareLease(arm, directory=root/'leases'):
                pass


def test_absolute_target_receipts_frames_and_speed_limits_without_xyz_bounds():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from vla.control.franky.tcp_frame import TCPFrame
    tool = [0, 0, .174, *Rotation.from_euler('z', -np.pi/4).as_quat()]
    frame = TCPFrame(tool)
    with process_worker(tool_pose=tool) as (root, _, _):
        request(root, 'reset', 1)
        goal = frame.to_logical(np.array([1.2, 0, 0, 0, 0, 0, 1.]))
        result = request(root, 'target', 1, sequence=1, timestamp_ns=time.monotonic_ns(),
                         arms={'left': {'pose': goal.tolist()}})
        assert not result['fault']
        state = result['states'][0]
        np.testing.assert_allclose(state['nominal_tcp_pose'], goal)
        np.testing.assert_allclose(state['reference_tcp_pose'], goal)
        np.testing.assert_allclose(state['hardware_reference_tcp_pose'], frame.to_hardware(goal), atol=1e-12)
        assert state['commanded_twist'][0] == pytest.approx(.05)
        event = result['servo_events'][0]
        assert event['sequence'] == 1 and event['slot_id'] == 1
        assert event['applied_at_ns'] >= event['observed_at_ns']
        remaining = request(root, audit_ack=event['audit_id'])['servo_events']
        assert all(item['audit_id'] > event['audit_id'] for item in remaining)
        stopped = request(root, 'stop', 2)['states'][0]
        assert not stopped['servo_ready'] and stopped['reference_tcp_pose'] == []


def test_watchdog_expires_despite_health_and_duplicate_traffic():
    with process_worker(target_timeout=.08) as (root, _, _):
        request(root, 'reset', 1)
        target(root)
        start = time.monotonic()
        while time.monotonic()-start < .15:
            request(root)
            target(root, sequence=1)  # Duplicate targets do not renew target time.
            time.sleep(.01)
        assert 'expired' in request(root)['fault']
        assert target(root, sequence=2)['fault']


@pytest.mark.parametrize('disconnect', [False, True])
def test_partial_packet_and_dual_arm_state_failure_stop_all(disconnect):
    with process_worker(('left', 'right')) as (root, _, _):
        request(root, 'reset', 1); target(root, arms=('left', 'right'))
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(str(root/'w.sock'))
            sock.sendall(b'\x00\x00')
            if not disconnect:
                time.sleep(.08)
        assert request(root)['fault']
        request(root, 'recover', 2); request(root, 'reset', 3)
        target(root, generation=3, arms=('left', 'right'))
        (root/'fail-state-right').touch()
        wait_for(lambda: request(root)['fault'])
        assert not any(s.get('servo_ready') for s in request(root)['states'])


def test_duplicate_worker_cannot_steal_socket_or_leases():
    with process_worker(('left', 'right')) as (root, _, argv):
        stat = (root/'w.sock').stat()
        other = subprocess.run(argv, env=worker_environment(ROOT, argv[0]), capture_output=True, timeout=5)
        assert other.returncode != 0 and b'Hardware already owned' in other.stderr
        assert (root/'w.sock').stat().st_ino == stat.st_ino
        assert not request(root)['fault']


def test_wrong_generation_future_time_and_duplicate_request():
    with process_worker() as (root, _, _):
        assert target(root)['fault']  # No reset handshake.
        request(root, 'recover', 1); request(root, 'reset', 2)
        packet = dict(sequence=1, timestamp_ns=time.monotonic_ns(), arms={'left': {'pose': POSE}}, request_id='same')
        first = request(root, 'target', 2, **packet)
        assert request(root, 'target', 2, **packet) == first
        assert request(root, 'target', 2, sequence=2, timestamp_ns=time.monotonic_ns()+10**9,
                       arms={'left': {'pose': POSE}})['fault']


def test_client_fault_latches_until_recover_and_control_requires_reset():
    with process_worker(target_timeout=.08) as (root, _, _):
        client = FrankyWorkerClient(root/'w.sock', session_id='s')
        try:
            wait_for(client.states)
            client.reset('s', 1)
            wait_for(lambda: client.states()['left'].generation == 1)
            client.target('s', 1, 1, {'left': {'pose': POSE, 'context': CONTEXT}}, slot_id=0)
            wait_for(lambda: client.states()['left'].servo_ready)
            wait_for(lambda: client.fault)
            request(root, 'recover', 2)  # Another channel's success cannot clear client's latch.
            time.sleep(.15)
            assert client.fault
            response = client.request_now({'kind': 'recover', 'session_id': 's', 'generation': 3,
                                           'request_id': uuid.uuid4().hex})
            assert not client.fault and not response['fault']
            assert not any(s.servo_ready for s in client.states().values())
        finally:
            client.close()


def test_idle_client_refreshes_real_worker_without_enabling_motion():
    with process_worker(target_timeout=.1, state_timeout=.5) as (root, _, _):
        client = FrankyWorkerClient(root/'w.sock', session_id='s')
        try:
            wait_for(client.states)
            samples = []
            end = time.monotonic() + .6
            while time.monotonic() < end:
                state = client.states()['left']
                samples.append((time.monotonic_ns() - state.observed_at_ns) / 1e9)
                assert not state.servo_ready
                time.sleep(.005)
            assert max(samples) < .1, max(samples)
            assert not client.fault
            log = root/'robot-events.jsonl'
            events = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            assert not any(event in ('move', 'recover') for _, event in events)
        finally:
            client.close()


def test_idle_client_loss_and_stalled_measurement_are_faults():
    with process_worker(state_timeout=.12) as (root, _, _):
        request(root)
        time.sleep(.2)
        assert 'heartbeat' in request(root)['fault']
        request(root, 'recover', 1); request(root, 'reset', 2)
        (root/'stale-state').touch()
        seq = 0
        end = time.monotonic()+.3
        while time.monotonic() < end:
            seq += 1
            result = target(root, generation=2, sequence=seq)
            if result['fault']:
                break
            time.sleep(.01)
        assert 'timestamp stalled' in result['fault']


def test_dual_arm_velocity_control_failure_keeps_last_confirmed_values():
    with process_worker(('left', 'right')) as (root, _, _):
        request(root, 'reset', 1); target(root, arms=('left', 'right'))
        (root/'fail-join-right').touch()
        # A second-arm stop failure must not commit new global parameters.
        result = request(root, 'velocity_control', 1, values={'tracking_tau_s': .2})
        assert result['fault']
        assert result['velocity_control']['tracking_tau_s'] == .15


def test_open_failure_releases_first_arm():
    with process_worker(('left', 'right')) as (root, child, argv):
        child.terminate(); child.wait(timeout=5)
        (root/'fail-open-right').touch()
        other = subprocess.run(argv, env=worker_environment(ROOT, argv[0]), capture_output=True, timeout=10)
        assert other.returncode != 0
        for arm in ('left', 'right'):
            with HardwareLease(arm, directory=root/'leases'):
                pass
        assert not (root/'w.sock').exists()


def test_explicit_recovery_handles_join_error_after_controller_stopped():
    with process_worker() as (root, _, _):
        request(root, 'reset', 1); target(root)
        (root/'fail-join-left').touch()
        assert request(root, 'stop', 2)['fault']
        assert not any(s.get('servo_ready') for s in request(root)['states'])
        result = request(root, 'recover', 3)
        assert not result['fault'] and result['request_status'] == 'applied'
        assert not any(s.get('servo_ready') for s in result['states'])
        request(root, 'reset', 4)
        assert not target(root, generation=4)['fault']


def test_dual_worker_30hz_with_production_watchdog_deadlines():
    with process_worker(('left', 'right'), target_timeout=.1, state_timeout=.5) as (root, _, _):
        request(root, 'reset', 1)
        assert not target(root, arms=('left', 'right'))['fault']
        start = time.monotonic()
        for sequence in range(2, 62):
            deadline = start + (sequence-1)/30
            time.sleep(max(0, deadline-time.monotonic()))
            result = target(root, sequence=sequence, arms=('left', 'right'))
            assert not result['fault']
            assert all(s['sequence'] == sequence and s['servo_ready'] for s in result['states'])
        assert not request(root, 'stop', 2)['fault']


def test_idle_health_connection_tolerates_scheduler_delay_within_state_deadline():
    with process_worker(target_timeout=.1, state_timeout=.5) as (root, _, _):
        assert not request(root)['fault']
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            sock.connect(str(root/'w.sock'))
            # Simulate the ROS thread holding the GIL after connect returns.
            time.sleep(.12)
            sock.sendall(encode_packet({'kind': 'health', 'session_id': 's',
                                      'generation': -1, 'request_id': 'delayed-health'}))
            result = decode_packet(sock)
        assert not result['fault']
        assert not any(s['servo_ready'] for s in result['states'])


def test_waiting_for_health_frame_does_not_extend_target_watchdog():
    with process_worker(target_timeout=.1, state_timeout=.5) as (root, _, _):
        request(root, 'reset', 1)
        assert not target(root)['fault']
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            sock.connect(str(root/'w.sock'))
            time.sleep(.18)
            sock.sendall(encode_packet({'kind': 'health', 'session_id': 's',
                                      'generation': 1, 'request_id': 'late-health'}))
            result = decode_packet(sock)
        assert 'target stream expired' in result['fault']
        assert not any(s['servo_ready'] for s in result['states'])
        assert target(root, sequence=2)['fault']  # No implicit resumption.


def test_incomplete_frame_exceeding_state_deadline_faults_and_logs_source():
    with process_worker(state_timeout=.12) as (root, _, _):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            sock.connect(str(root/'w.sock'))
            sock.sendall(b'\x00\x00')
            assert sock.recv(1) == b''  # Server expires the incomplete header.
        result = request(root)
        assert 'IPC receive' in result['fault'] and 'TimeoutError' in result['fault']
        assert 'Franky worker fault session=s' in (root/'process.log').read_text()
