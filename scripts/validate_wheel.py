#!/usr/bin/env python3
"""Build a clean wheel and check external imports without installing anything."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    output = Path(tempfile.mkdtemp(prefix='vla-wheel-validation-'))
    source = output / 'source'
    source.mkdir()
    for name in ('pyproject.toml', 'README.md', 'LICENSE'):
        shutil.copy2(root / name, source / name)
    paths = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard', 'src'], cwd=root
    ).decode().split('\0')
    for name in filter(None, paths):
        original = root / name
        if original.is_file():
            destination = source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', str(source), '--no-deps',
                    '--no-build-isolation', '-w', str(output)], check=True, cwd=output)
    wheel, = output.glob('*.whl')
    unpacked = output / 'unpacked'
    with zipfile.ZipFile(wheel) as archive:
        expected = {str(p.relative_to(source / 'src')) for p in (source / 'src').rglob('*.py')}
        actual = {p for p in archive.namelist() if p.endswith('.py')}
        if actual != expected:
            raise RuntimeError(f'wheel/source mismatch: missing={expected-actual}, extra={actual-expected}')
        archive.extractall(unpacked)
    env = dict(os.environ, PYTHONPATH=str(unpacked), PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    checks = {
        '/usr/bin/python3': '''
import sys
from vla.control.execution import Executor, ExecutionContract
from vla.control.plan_service import PlanGate
from vla_rollout_protocol import protocol
from vla.datacollection.franka_robotiq_single_left.auto_record import KeyboardEvents
from vla.datacollection.franka_robotiq_dual.auto_record import main
assert not any(name in sys.modules for name in ('torch', 'rclpy', 'pyrealsense2'))
''',
        sys.executable: '''
import sys
from vla.adapters.lerobot.plan_inference import infer_plan
from vla.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
from vla.core.losses import actor_loss, critic_loss
from vla.core.execution_replay import validate_execution_transition
from vla.cli.recover_episode import main
assert 'rclpy' not in sys.modules
''',
    }
    for interpreter, code in checks.items():
        subprocess.run([interpreter, '-c', code + '''
import os
from pathlib import Path
import vla.control.execution as execution
assert Path(execution.__file__).is_relative_to(os.environ['PYTHONPATH'])
'''], check=True, cwd=output, env=env)
    print(json.dumps({'wheel': str(wheel), 'python_files': len(expected),
                      'external_imports': 'passed', 'installed': False}, indent=2))


if __name__ == '__main__':
    main()
