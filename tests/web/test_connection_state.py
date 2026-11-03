"""Connection lifecycle and bounded diagnostics without starting devices."""
import json
import time
from types import SimpleNamespace

import pytest

from vla.web.manager import Manager
from vla.web.processes import Children


def test_endpoint_startup_grace_recovery_and_real_fault(tmp_path, monkeypatch):
    manager = Manager(tmp_path)
    directory = tmp_path / 'session'; directory.mkdir()
    record = {'session_id': 'test', 'alive': True, 'directory': str(directory),
              'config': {}, 'created_at': time.time()}
    monkeypatch.setattr(manager, 'active', lambda: record)
    def missing(*args, **kwargs):
        raise FileNotFoundError(2, 'No such file or directory')
    monkeypatch.setattr('vla.web.manager.request', missing)
    state = manager.snapshot()
    assert state['connection_state'] == 'starting'
    assert 'error' not in state['status']
    assert 'Errno' not in state['connection_message']
    record['created_at'] -= 11
    assert manager.snapshot()['connection_state'] == 'delayed'
    fault = {'status': {'state': 'fault', 'error': "[Errno 2] No such file: checkpoint/config.json"}}
    monkeypatch.setattr('vla.web.manager.request', lambda *a, **k: fault)
    state = manager.snapshot()
    assert state['connection_state'] == 'connected'
    assert state['status']['error'] == fault['status']['error']
    assert not state['connection_message']
    monkeypatch.setattr('vla.web.manager.request', missing)
    assert manager.snapshot()['connection_state'] == 'reconnecting'
    monkeypatch.setattr('vla.web.manager.request', lambda *a, **k: {'status': {'state': 'standby'}})
    assert manager.snapshot()['connection_state'] == 'connected'


def test_exited_runner_without_final_keeps_actual_diagnostic(tmp_path, monkeypatch):
    manager = Manager(tmp_path)
    directory = tmp_path / 'session'; directory.mkdir()
    (directory / 'runner.log').write_text('ValueError: invalid schema\n')
    record = {'session_id': 'test', 'alive': False, 'directory': str(directory), 'config': {}}
    monkeypatch.setattr(manager, 'active', lambda: record)
    state = manager.snapshot()
    assert state['status']['final'] and state['connection_state'] == 'stopped'
    assert 'invalid schema' in state['status']['error']
    (directory / 'final.json').write_text(json.dumps({'status': {'state': 'stopped', 'final': True}}))
    assert 'error' not in manager.snapshot()['status']
    monkeypatch.setattr(manager, 'active', lambda: None)
    assert manager.snapshot()['connection_state'] == 'idle'


def test_child_failure_diagnostic_is_bounded(tmp_path):
    (tmp_path / 'hardware.log').write_text('x' * 20000 + '\nERROR: device busy\nShutdown complete\n')
    children = Children(tmp_path)
    children.items = [('hardware', SimpleNamespace(poll=lambda: 1, returncode=1))]
    with pytest.raises(RuntimeError) as caught:
        children.check()
    message = str(caught.value)
    assert 'hardware exited (1)' in message and 'device busy' in message
    assert str(tmp_path / 'hardware.log') in message and len(message) < 2000
