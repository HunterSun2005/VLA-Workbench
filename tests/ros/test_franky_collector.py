"""Real ROS collector methods with cached worker replies; no Node/device creation."""
from types import SimpleNamespace
import time

from vla.datacollection.franka_robotiq_rollout.collect_ros import RolloutCollector
from vla.control.franky.worker import Compliance
from dataclasses import asdict


def collector():
    node = RolloutCollector.__new__(RolloutCollector)
    values = asdict(Compliance())
    results, requests, faults = {}, [], []
    node.config = SimpleNamespace(backend='franky', franky_compliance=values, gamepad_state_timeout_s=.5)
    node._pending_compliance = None
    node._worker_session_id = 'session'
    node.coordinator = SimpleNamespace(generation=3, mode='standby')
    node._worker_client = SimpleNamespace(
        fault=None, compliance_values=lambda: values.copy(), result=results.get,
        compliance=lambda *args: requests.append(args), states=lambda: {},
        stop=lambda *args: requests.append(('stop', *args)),
        reset=lambda *args: requests.append(('reset', *args)))
    node._latch_fault = faults.append
    return node, values, results, requests, faults


def test_compliance_needs_matching_request_ack_even_if_values_already_match():
    node, values, results, requests, faults = collector()
    node.update_compliance({'translational_stiffness': 1000})
    request_id = requests[-1][2]
    node._poll_worker_states()
    assert node.compliance_status()['pending']
    results['unrelated'] = {'request_status': 'applied'}
    node._poll_worker_states()
    assert node.compliance_status()['pending']
    results[request_id] = {'request_status': 'applied'}
    node._poll_worker_states()
    assert not node.compliance_status()['pending'] and not faults


def test_compliance_timeout_and_rejection_latch_fault_keep_effective_values():
    for response in (None, {'request_status': 'failed', 'error': 'second arm failed'}):
        node, values, results, requests, faults = collector()
        node.update_compliance({'translational_stiffness': 900})
        if response is None:
            node._pending_compliance['started'] = time.monotonic()-1
        else:
            results[requests[-1][2]] = response
        node._poll_worker_states()
        assert faults and not node.compliance_status()['pending']
        assert node.compliance_status()['values']['translational_stiffness'] == 1000


def test_fault_boundary_stops_worker_without_rearming():
    node, _, _, requests, _ = collector()
    node.coordinator.mode = 'fault'
    node._request_controller_reset(4)
    assert requests == [('stop', 'session', 4)]


def test_close_releases_worker_before_storage_and_cameras():
    import threading
    node, _, _, _, _ = collector()
    events = []
    node._lock = threading.RLock()
    node._closed = False
    node._stop_grippers_at_measured_position = lambda: events.append('gripper-stop')
    node._control_boundary = lambda **k: events.append('revoke')
    node._worker_shutdown = lambda: events.append('worker-exited')
    node._save_worker = SimpleNamespace(join=lambda: events.append('storage-wait'))
    node._journal = None
    node.policy_client = None
    node._cameras = {'camera': SimpleNamespace(close=lambda: events.append('camera-close'))}
    node.close()
    assert events == ['gripper-stop', 'revoke', 'worker-exited', 'storage-wait', 'camera-close']
