"""Production backend startup with a real wheel worker and mocked ROS peripherals."""
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace

import pytest

from vla.web.backends import make_backend, ROSBackend
from vla.web.config import SessionConfig, preflight
from vla.web.hardware import preset
from vla.web.processes import Children
from vla.core.hardware_lease import HardwareLease
from vla.core.franky_runtime import check_worker
from tests.control.test_franky_process import FAKE, ROOT, wait_for


@pytest.mark.parametrize('arms', ['single_left', 'dual'])
@pytest.mark.parametrize('source', ['gello', 'gamepad'])
@pytest.mark.parametrize('workflow', ['collect', 'deploy'])
def test_backend_ready_identity_lease_and_shutdown_order(monkeypatch, arms, source, workflow, failure=None):
    if not (ROOT/'runtime/franky/current/bin/python').is_file():
        pytest.skip('persistent Franky environment missing')
    with tempfile.TemporaryDirectory(prefix='vfb-') as temporary:
        directory = Path(temporary)
        events = []
        hardware = preset(arms)
        for arm in ('LEFT', 'RIGHT'):
            hardware[f'{arm}_ROBOT_IP'] = arm.lower()
            hardware[f'{arm}_ROBOTIQ_PORT'] = str(directory/f'{arm}-gripper')
        path = directory/'hardware.json'; path.write_text(json.dumps(hardware))
        cfg = SessionConfig(workflow=workflow, arms=arms, source=source, task='mock',
                            ros_config=str(path), dataset_root=str(directory/'data'))
        # Lease implementation is real; its namespace is isolated from devices.
        import vla.core.hardware_lease as leases
        original_lease = leases.HardwareLease
        monkeypatch.setattr(leases, 'HardwareLease', lambda ip, port=None: original_lease(ip, port, directory=directory/'leases'))
        # A peripheral process whose shutdown asserts worker has exited first.
        children = []
        class Peripheral:
            returncode = None
            def poll(self): return self.returncode
            def send_signal(self, _):
                assert all(p.poll() is not None for p in children)
                events.append('peripheral-stop'); self.returncode = 0
            def wait(self, timeout=None): return self.returncode
        start = Children.start
        def start_mock(self, name, argv, env):
            if name == 'franky-worker':
                argv = [argv[0], str(FAKE), *argv[3:]]
                result = start(self, name, argv, env)
                children.append(result); events.append('worker-start')
                return result
            if name == 'hardware':
                assert env['FRANKY_OWNER'] == '1'
                (directory/'hardware.ready').touch()
                events.append('peripheral-start')
                result = Peripheral(); self.items.append((name, result)); return result
            raise AssertionError(f'unexpected process {name}')
        monkeypatch.setattr(Children, 'start', start_mock)
        monkeypatch.setattr(ROSBackend, '_start_gello', lambda *a: events.append('gello-ready'))
        rclpy = ModuleType('rclpy'); rclpy.init = lambda **k: None; rclpy.shutdown = lambda: None
        executors = ModuleType('rclpy.executors')
        executors.SingleThreadedExecutor = lambda: SimpleNamespace(add_node=lambda node: None, shutdown=lambda: None)
        monkeypatch.setitem(sys.modules, 'rclpy', rclpy)
        monkeypatch.setitem(sys.modules, 'rclpy.executors', executors)
        collector_module = ModuleType('vla.datacollection.franka_robotiq_rollout.collect_ros')
        class Collector:
            require_online = False
            def __init__(self, config, *, demonstration, worker_client, worker_session_id, worker_shutdown):
                self.config, self.client = config, worker_client
                self.session_id = worker_session_id
                self.worker_shutdown = worker_shutdown
                assert demonstration == (workflow == 'collect')
                assert worker_session_id == directory.name
                assert set(worker_client.states()) == {a.name for a in config.arms}
                assert worker_client.binding_fingerprint()['libfranka'] == '0.21.2'
                if failure == 'collector':
                    raise RuntimeError('injected collector initialization failure')
                events.append('collector-ready')
            def set_control_mode(self, mode): pass
            def destroy_node(self): pass
            def close(self):
                self.client.request_now({'kind': 'stop', 'session_id': self.session_id,
                                         'generation': 1, 'request_id': 'collector-close'})
                self.worker_shutdown()
                assert all(p.poll() is not None for p in children)
                events.append('collector-stop')
        collector_module.RolloutCollector = Collector
        monkeypatch.setitem(sys.modules, collector_module.__name__, collector_module)
        backend = None
        try:
            if failure:
                with pytest.raises(RuntimeError, match='cancelled|injected collector'):
                    make_backend(cfg, directory, ROOT,
                                 cancelled=lambda: failure == 'cancel' and bool(children))
                assert all(child.poll() is not None for child in children)
                for arm in ('left', 'right') if arms == 'dual' else ('left',):
                    with original_lease(arm, hardware[f'{arm.upper()}_ROBOTIQ_PORT'], directory=directory/'leases'):
                        pass
                return
            backend = make_backend(cfg, directory, ROOT)
            assert events.index('worker-start') < events.index('peripheral-start') < events.index('collector-ready')
            for arm in backend.arms:
                with pytest.raises(RuntimeError, match='Hardware already owned'):
                    original_lease(arm, directory=directory/'leases')
            backend.close()
            assert events.index('collector-stop') < events.index('peripheral-stop')
            assert all(child.poll() == 0 for child in children)
            for arm in backend.arms:
                with original_lease(arm, hardware[f'{arm.upper()}_ROBOTIQ_PORT'], directory=directory/'leases'):
                    pass
        finally:
            if backend is not None:
                backend.close()
            for child in children:
                if child.poll() is None:
                    child.terminate(); child.wait(timeout=5)


@pytest.mark.parametrize('failure', ['cancel', 'collector'])
def test_startup_failure_reaps_worker_and_releases_leases(monkeypatch, failure):
    test_backend_ready_identity_lease_and_shutdown_order(monkeypatch, 'dual', 'gamepad', 'collect', failure)


def test_preflight_reports_missing_worker_before_session(tmp_path, monkeypatch):
    monkeypatch.setenv('FRANKY_PYTHON', str(tmp_path/'missing-worker'))
    result = preflight(SessionConfig(task='test'), root=tmp_path)
    assert not result['ok']
    assert any('Franky Python unavailable' in error for error in result['errors'])
    assert not (tmp_path/'runtime').exists()


def test_backend_failed_preflight_never_starts_children(monkeypatch, tmp_path):
    from vla.core import franky_runtime
    def fail(*a, **k): raise RuntimeError('binding import failed')
    monkeypatch.setattr(franky_runtime, 'check_worker', fail)
    monkeypatch.setattr(Children, 'start', lambda *a, **k: pytest.fail('hardware/process started before preflight'))
    # Real ROS imports are unnecessary for a dependency failure.
    rclpy = ModuleType('rclpy')
    executors = ModuleType('rclpy.executors'); executors.SingleThreadedExecutor = object
    monkeypatch.setitem(sys.modules, 'rclpy', rclpy)
    monkeypatch.setitem(sys.modules, 'rclpy.executors', executors)
    path = tmp_path/'hardware.json'; path.write_text(json.dumps(preset('single_left')))
    with pytest.raises(RuntimeError, match='binding import failed'):
        make_backend(SessionConfig(task='test', ros_config=str(path)), tmp_path, ROOT)
