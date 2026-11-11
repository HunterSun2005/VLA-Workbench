"""Hardware-free interpreter selection and worker preflight shared by all front ends."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


def worker_python(root, configured=None):
    value = configured or os.environ.get('FRANKY_PYTHON') or Path(root)/'runtime/franky/current/bin/python'
    # Resolving bin/python symlinks can turn a venv into the system interpreter.
    return Path(os.path.abspath(Path(value).expanduser()))


def worker_environment(root, python):
    from vla.web.processes import clean_environment
    env = clean_environment(root, python)
    env.pop('LD_PRELOAD', None)
    env.pop('VIRTUAL_ENV', None)
    return env


def check_worker(root, configured=None, *, timeout=120.):
    python = worker_python(root, configured)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError(f'Franky Python unavailable: {python}; run /usr/bin/python3 scripts/install_franky_worker.py')
    try:
        result = subprocess.run([str(python), '-m', 'vla.control.franky.worker_main', '--check-only'],
                                env=worker_environment(root, python), capture_output=True, text=True,
                                timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f'Franky preflight failed: {python}: {error}') from error
    if result.returncode:
        raise RuntimeError(f'Franky preflight failed ({result.returncode}): {python}\n{result.stderr[-3000:]}')
    try:
        value = json.loads(result.stdout)
        if value['libfranka'] != '0.19.0' or value['affine_check'] != 'passed':
            raise ValueError('wrong library or failed Affine check')
        for key in ('binding_sha256', 'tracker_sha256', 'libfranka_library_sha256'):
            if len(value[key]) != 64:
                raise ValueError('missing binary fingerprint')
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f'Invalid Franky preflight result from {python}: {error}') from error
    return python, value


def semantic_binding(value):
    """Installation paths are diagnostics, never part of a recording identity."""
    return {key: value[key] for key in ('franky', 'libfranka', 'numpy', 'binding_sha256',
                                      'tracker_sha256', 'libfranka_library_sha256')}
