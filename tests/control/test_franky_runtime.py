from pathlib import Path
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vla.core.franky_runtime import worker_python, check_worker, worker_environment, semantic_binding

ROOT = Path(__file__).resolve().parents[2]


def test_interpreter_precedence_preserves_symlink(tmp_path, monkeypatch):
    monkeypatch.delenv('FRANKY_PYTHON', raising=False)
    assert worker_python(tmp_path) == tmp_path/'runtime/franky/current/bin/python'
    link = tmp_path/'python'
    link.symlink_to(sys.executable)
    monkeypatch.setenv('FRANKY_PYTHON', str(link))
    assert worker_python(tmp_path) == link
    assert worker_python(tmp_path, '/explicit/python') == Path('/explicit/python')


def test_preflight_isolated_and_missing_environment(tmp_path, monkeypatch):
    for key in ('LD_PRELOAD', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'VIRTUAL_ENV', 'CONDA_PREFIX'):
        monkeypatch.setenv(key, '/unwanted')
    env = worker_environment(tmp_path, tmp_path/'python')
    assert env['PYTHONNOUSERSITE'] == '1'
    assert all(key not in env for key in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'VIRTUAL_ENV', 'CONDA_PREFIX'))
    with pytest.raises(RuntimeError, match='install_franky_worker'):
        check_worker(tmp_path, tmp_path/'absent')


def test_preflight_process_failures(monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=139, stderr='native crash'))
    with pytest.raises(RuntimeError, match='139'):
        check_worker(ROOT, sys.executable)
    def timeout(*a, **k): raise subprocess.TimeoutExpired('mock', 1)
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(RuntimeError, match='preflight failed'):
        check_worker(ROOT, sys.executable)


def test_persistent_wheel_preflight():
    if not (ROOT/'runtime/franky/current/bin/python').is_file():
        pytest.skip('persistent Franky environment not installed')
    python, value = check_worker(ROOT)
    assert value['libfranka'] == '0.21.2' and value['numpy'] == '1.26.4'
    assert '/runtime/franky/' in value['binding'] and '/tmp/' not in value['binding']
    moved = dict(value, binding='/another/path', tracker='/another/tracker', libfranka_library='/other/lib')
    assert semantic_binding(value) == semantic_binding(moved)
    result = subprocess.run([str(python), '-c', 'import sys,site,json; print(json.dumps([sys.prefix,sys.base_prefix,site.ENABLE_USER_SITE,sys.path]))'],
                            env=worker_environment(ROOT, python), capture_output=True, text=True, check=True)
    prefix, base, user, paths = json.loads(result.stdout)
    assert prefix != base and not user
    assert not any('/dist-packages' in p or '/envs/vla/' in p for p in paths)
