#!/usr/bin/env python3
"""Run isolated offline acceptance checks and retain a source-bound JSON/Markdown report."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ('model', 'ros', 'wheel', 'timing')
UNVERIFIED = [
    'Real PI0.5/GPU behavior unless explicitly exercised by resource-backed tests',
    'Real cameras, synchronization and recording throughput',
    'Physical robot/gripper continuity, tracking and task success',
    'Combined camera/encoding/inference load during the timing harness',
]


def git(root, *args):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(root), *args])


def source_identity(root):
    digest = hashlib.sha256()
    names = set(git(root, 'ls-files', '-z', '--cached', '--others', '--exclude-standard').split(b'\0'))
    for name in sorted(names - {b''}):
        path = root / os.fsdecode(name)
        digest.update(len(name).to_bytes(8, 'big') + name)
        if path.is_symlink():
            content = os.fsencode(os.readlink(path))
        elif path.is_file():
            content = path.read_bytes()
        else:
            content = b'<deleted>'
        digest.update(len(content).to_bytes(8, 'big') + content)
    status = git(root, 'status', '--porcelain=v1', '--untracked-files=all').decode()
    return {'commit': git(root, 'rev-parse', 'HEAD').decode().strip(),
            'branch': git(root, 'branch', '--show-current').decode().strip(),
            'dirty': bool(status), 'status': status, 'tree_sha256': digest.hexdigest()}


def junit_result(path):
    document = ET.parse(path).getroot()
    counts = dict(passed=0, skipped=0, failed=0, errors=0)
    skipped, failures = [], []
    for case in document.iter('testcase'):
        name = '.'.join(filter(None, (case.get('classname'), case.get('name'))))
        status = next((key for key in ('error', 'failure', 'skipped') if case.find(key) is not None), None)
        if status is None:
            counts['passed'] += 1
        else:
            node = case.find(status)
            counts[{'error': 'errors', 'failure': 'failed', 'skipped': 'skipped'}[status]] += 1
            item = {'test': name, 'reason': node.get('message') or (node.text or '').strip()}
            (skipped if status == 'skipped' else failures).append(item)
    if not sum(counts.values()):
        raise ValueError('test report contains no test cases')
    return {'counts': counts, 'skipped_cases': skipped, 'failures': failures}


def stop_child(process):
    # Only this invocation's process group, never existing services or sessions.
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)  # Let mock ROS harnesses close their own child groups.
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def run_check(command, *, root, env, log, timeout_s):
    started = time.monotonic()
    result = {'command': command, 'log': log.name, 'status': 'failed'}
    with log.open('w') as output:
        process = None
        try:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            result['exit_code'] = process.wait(timeout=timeout_s)
            result['status'] = 'passed' if result['exit_code'] == 0 else 'failed'
        except subprocess.TimeoutExpired:
            result.update(status='timed_out', reason=f'exceeded {timeout_s}s')
            stop_child(process)
        except KeyboardInterrupt:
            if process is not None:
                stop_child(process)
            result.update(status='interrupted', reason='validation interrupted')
        except OSError as error:
            result.update(status='failed', reason=str(error))
    result['elapsed_s'] = round(time.monotonic()-started, 3)
    return result


def finalize_check(name, result, output):
    if result['status'] != 'passed' and name != 'timing':
        if name in ('model', 'ros') and (output/f'{name}.xml').is_file():
            try:
                result.update(junit_result(output/f'{name}.xml'))
                result['junit'] = f'{name}.xml'
            except (OSError, ValueError, ET.ParseError):
                pass  # Preserve a timeout/interruption even if JUnit is incomplete.
        return result
    try:
        if name in ('model', 'ros'):
            result.update(junit_result(output/f'{name}.xml'))
            result['junit'] = f'{name}.xml'
            counts = result['counts']
            if counts['failed'] or counts['errors']:
                result['status'] = 'failed'
            elif counts['skipped']:
                result['status'] = 'passed_with_skips' if counts['passed'] else 'skipped'
        elif name == 'wheel':
            result['details'] = json.loads((output/'wheel/result.json').read_text())
            if result['details']['external_imports'] != 'passed':
                raise ValueError('wheel import verification did not pass')
        else:
            lines = (output/result['log']).read_text().splitlines()
            report_path = next(Path(line[7:]) for line in reversed(lines) if line.startswith('REPORT='))
            details = json.loads(report_path.read_text())
            (output/'timing.json').write_text(json.dumps(details, indent=2)+'\n')
            result.update(details=details, metrics='timing.json')
            if not details.get('frequency_target_passed') or details.get('error'):
                raise ValueError('timing acceptance did not pass')
    except (OSError, ValueError, KeyError, StopIteration, ET.ParseError) as error:
        if result['status'] == 'passed':
            result['status'] = 'failed'
        result['reason'] = f'invalid/missing evidence: {error}'
    return result


def write_report(output, report):
    statuses = [value['status'] for value in report['checks'].values()]
    report['status'] = ('failed' if report.get('source_changed') or any(
        status in ('failed', 'timed_out', 'interrupted') for status in statuses)
        else 'passed_with_gaps' if report.get('finished_at') else 'running')
    temp = output/'report.json.tmp'
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    temp.replace(output/'report.json')
    source = report['source']
    lines = ['# Offline acceptance', '', f"Result: **{report['status']}**", '',
             f"Commit: `{source['commit']}`; branch: `{source['branch']}`; dirty: `{source['dirty']}`", '',
             f"Source SHA256: `{source['tree_sha256']}`", '',
             '| Check | Result | Passed / skipped / failed / errors | Seconds | Evidence |',
             '| --- | --- | --- | --- | --- |']
    for name, value in report['checks'].items():
        counts = value.get('counts')
        count_text = ' / '.join(str(counts[key]) for key in ('passed','skipped','failed','errors')) if counts else '—'
        evidence = f"[{value['log']}]({value['log']})" if value.get('log') else '—'
        lines.append(f"| {name} | {value['status']} | {count_text} | {value.get('elapsed_s', '—')} | {evidence} |")
    lines += ['', '## Gaps and failures', '']
    if report.get('source_changed'):
        lines.append('- Source changed during verification; results do not validate a single revision.')
    for name, value in report['checks'].items():
        if value.get('reason'):
            lines.append(f"- {name}: {value['reason']}")
        for item in value.get('skipped_cases', []) + value.get('failures', []):
            reason = item['reason'].replace('\n', ' ')
            lines.append(f"- {name}: `{item['test']}` — {reason}")
    lines += ['', '## Outside acceptance scope', '', *('- '+item for item in UNVERIFIED), '']
    (output/'report.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-python', default=os.environ.get('VLA_PYTHON', sys.executable))
    parser.add_argument('--checks', nargs='+', choices=CHECKS, default=['model', 'ros', 'wheel'])
    parser.add_argument('--output', type=Path, help='New artifact directory; defaults to artifacts/validation/RUN')
    parser.add_argument('--timeout', type=float, default=900, help='Deadline per selected check in seconds')
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    source = source_identity(ROOT)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+source['commit'][:8]+'-'+uuid.uuid4().hex[:6]
    output = (args.output or ROOT/'artifacts/validation'/run_id).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, VLA_PYTHON=args.model_python, VLA_ROOT=str(ROOT),
               HF_HOME=str(output/'huggingface'), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONNOUSERSITE='1')
    for key in ('PYTHONHOME', 'PYTHONPATH', 'PYTHONOPTIMIZE', 'PYTEST_ADDOPTS', 'LD_LIBRARY_PATH',
                'AMENT_PREFIX_PATH', 'COLCON_PREFIX_PATH', 'CMAKE_PREFIX_PATH', 'ROS_DISTRO',
                'ROS_VERSION', 'ROS_PYTHON_VERSION'):
        env.pop(key, None)
    commands = {
        'model': ['bash', 'scripts/test.sh', '--ignore=tests/ros', '-q', '-p', 'no:cacheprovider', '--tb=short', '-rs',
                  '--junitxml='+str(output/'model.xml')],
        'ros': ['bash', 'scripts/test_ros.sh', 'tests/ros', '-q', '--junitxml='+str(output/'ros.xml')],
        'wheel': [args.model_python, 'scripts/validate_wheel.py', '--output', str(output/'wheel')],
        'timing': ['bash', 'scripts/validate_ros_control.sh', '--duration', '600'],
    }
    report = {'schema_version': 1, 'run_id': run_id, 'started_at': datetime.now(timezone.utc).isoformat(),
              'source': source, 'model_python': args.model_python, 'scope': 'offline; no physical devices',
              'outside_scope': UNVERIFIED, 'checks': {name: {'status': 'not_run', 'reason': 'not selected'} for name in CHECKS}}
    for name in args.checks:
        report['checks'][name]['reason'] = 'not started'
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    write_report(output, report)
    for name in dict.fromkeys(args.checks):
        print(f'[{name}] starting; log: {output/(name+".log")}', flush=True)
        result = run_check(commands[name], root=ROOT, env=env, log=output/f'{name}.log', timeout_s=args.timeout)
        report['checks'][name] = finalize_check(name, result, output)
        write_report(output, report)
        print(f"[{name}] {report['checks'][name]['status']}", flush=True)
        if result['status'] == 'interrupted':
            break
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    report['source_after'] = source_identity(ROOT)
    report['source_changed'] = report['source_after'] != source
    write_report(output, report)
    print(f'REPORT={output / "report.json"}', flush=True)
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    sys.exit(main())
