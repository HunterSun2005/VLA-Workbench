"""Real ROS collector methods with cached worker replies; no Node/device creation."""
from types import SimpleNamespace
from pathlib import Path
import time
import threading
import numpy as np
import pytest

from vla.datacollection.franka_robotiq_rollout.collect_ros import RolloutCollector
from vla.control.franky.worker import Compliance
from dataclasses import asdict
from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
from vla.control.ros.cartesian_servo import CartesianServo


def collector():
    node = RolloutCollector.__new__(RolloutCollector)
    values = asdict(Compliance())
    results, requests, faults = {}, [], []
    node.config = SimpleNamespace(backend='franky', franky_compliance=values, gamepad_state_timeout_s=.5)
    node._lock = threading.RLock()
    node._faulted = False
    node._journal = None
    node._pending_worker_targets = {}
    node._pending_compliance = None
    node._worker_session_id = 'session'
    node.coordinator = SimpleNamespace(generation=3, mode='standby', command_id=1)
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
    results[request_id] = {'request_status': 'applied', 'request_id': request_id, 'generation': 3, 'compliance': values}
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


@pytest.mark.parametrize('profile', ['single_left', 'dual'])
def test_franky_loads_real_namespaced_kinematics_without_ros_arm_nodes(monkeypatch, profile):
    monkeypatch.delenv('FRANKA_URDF', raising=False)
    root = Path(__file__).resolve().parents[2]
    node = RolloutCollector.__new__(RolloutCollector)
    node.config = load_rollout_config(root/f'src/vla/datacollection/franka_robotiq_rollout/configs/{profile}.yaml')
    node._kinematics_identity = {}
    node.create_client = lambda *a: pytest.fail('Franky FK must not query a ROS arm node')
    for arm in node.config.arms:
        model = node._load_kinematic_model(arm)
        assert model.chain.getNrOfJoints() == 7
        assert model.joint_names == tuple(f'{arm.name}_fr3_joint{i}' for i in range(1, 8))
        assert len(node._kinematics_identity[arm.name]) == 64
        q = np.array([0., -.4, .1, -1.8, .1, 1.4, .5])
        assert np.all(model.lower_limits < q) and np.all(q < model.upper_limits)
        p, r = CartesianServo(model, node.config).forward_tcp(q, np.zeros(3), np.eye(3))
        assert np.isfinite(p).all() and np.isfinite(r).all()
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)


def test_explicit_franky_urdf_is_used_and_missing_override_is_rejected(monkeypatch, tmp_path):
    import xacro
    from ament_index_python.packages import get_package_share_directory
    node = RolloutCollector.__new__(RolloutCollector)
    node.config = SimpleNamespace(backend='franky', joint_limit_margin=.05)
    node._kinematics_identity = {}
    arm = SimpleNamespace(name='left', base_link='custom_fr3_link0', tip_link='custom_fr3_link8')
    source = Path(get_package_share_directory('franka_description'))/'robots/fr3/fr3.urdf.xacro'
    path = tmp_path/'custom.urdf'
    path.write_text(xacro.process_file(str(source), mappings={
        'hand': 'false', 'arm_prefix': 'custom', 'ros2_control': 'false'}).toxml())
    monkeypatch.setenv('FRANKA_URDF', str(path))
    assert node._load_kinematic_model(arm).joint_names[0] == 'custom_fr3_joint1'
    monkeypatch.setenv('FRANKA_URDF', str(tmp_path/'missing.urdf'))
    with pytest.raises(FileNotFoundError):
        node._load_kinematic_model(arm)


def test_late_compliance_ack_cannot_resume_a_new_generation():
    node, values, results, requests, faults = collector()
    node.update_compliance({'rotational_stiffness': 60}, request_id='web-command')
    assert node.compliance_status()['request']['request_id'] == 'web-command'
    node.coordinator.generation += 1
    results['web-command']={'request_id':'web-command','generation':3,'request_status':'applied','compliance':values}
    node._poll_worker_states()
    assert node.compliance_status()['request']['status']=='cancelled'
    assert not node.compliance_status()['pending'] and not faults


def test_tuning_freezes_input_and_waits_for_fresh_acceptance():
    node, _, _, _, _ = collector()
    node.update_compliance({'rotational_stiffness': 60})
    node._pending_storage_error = None
    node._tracking_clock = SimpleNamespace(poll=lambda _:0)
    node._last_gamepad_tick = time.monotonic()-.01
    node._observe_control = node._online_save_tick = lambda:None
    node._publish_gamepad = lambda *a: pytest.fail('integrated input during tuning')
    node._track_control = lambda *a: pytest.fail('submitted target during tuning')
    node._gamepad_tick()
    node._pending_compliance = None
    node.coordinator.tick = lambda *a: pytest.fail('resumed an old accepted target')
    RolloutCollector._track_control(node,.01)
