#!/usr/bin/env python3
"""Inspect Git/runtime locks; optionally remove a conservatively checked stale index lock.

Does not stop processes, start sessions, alter dependency lockfiles or unlink flock files.
Run on the host with its complete /proc namespace and no concurrent Git commands during repair.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
import time


def signature(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
        raise RuntimeError('refusing symlink, non-regular, hard-linked or non-owned lock')
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def process_blockers(identity):
    """Fail closed on inaccessible same-user processes; never rely on a stored PID."""
    blockers = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            name = (process / 'comm').read_text().strip()
            if name == 'git' or name.startswith('git-'):
                blockers.append(f'Git process {process.name} ({name})')
            for fd in (process / 'fd').iterdir():
                try:
                    info = fd.stat()
                    if (info.st_dev, info.st_ino) == identity[:2]:
                        blockers.append(f'open descriptor in process {process.name}')
                except FileNotFoundError:
                    continue  # Descriptor closed during the scan.
        except (FileNotFoundError, ProcessLookupError):
            continue  # Process exited during the scan.
        except PermissionError:
            blockers.append(f'cannot inspect process {process.name}')
    return sorted(set(blockers))


def clear_stale_index(path):
    initial = signature(path)
    age = time.time() - max(initial[3:]) / 1e9
    if age < 60:
        raise RuntimeError('index lock is recent (less than 60 seconds); leave it in place')
    blockers = process_blockers(initial)
    if blockers:
        raise RuntimeError('; '.join(blockers))
    if signature(path) != initial:
        raise RuntimeError('index lock changed during inspection; leave it in place')
    # This is a Git O_EXCL transaction marker, never a runtime flock inode.
    # No automatic retries: a changing repository needs another explicit inspection.
    path.unlink()


def runtime_status(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return {'path': str(path), 'kind': 'runtime', 'state': 'absent'}
    with os.fdopen(fd, 'r') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise RuntimeError('runtime lock is not a regular file')
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            state = 'held'
        else:
            state = 'idle_persistent_file'
        # Contents are diagnostic history, not authority to remove a file.
        return {'path': str(path), 'kind': 'runtime', 'state': state,
                'owner_metadata': stream.read(4096)}


def git_path(root, *args):
    value = subprocess.check_output(['git', '--no-optional-locks', '-C', str(root),
                                     'rev-parse', *args], text=True).strip()
    path = Path(value)
    return path if path.is_absolute() else root / path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--runtime-lock', type=Path, action='append', default=[])
    parser.add_argument('--clear-stale-index', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    index = git_path(root, '--git-path', 'index.lock')
    report = []
    if args.clear_stale_index and index.exists():
        try:
            clear_stale_index(index)
        except (OSError, RuntimeError) as error:
            parser.exit(1, f'Refused index cleanup: {error}\n')
        report.append({'path': str(index), 'kind': 'git', 'state': 'removed_stale_index'})
    directories = {git_path(root, '--git-dir'), git_path(root, '--git-common-dir')}
    locks = {path for directory in directories for path in directory.glob('*.lock')}
    locks.update(path for directory in directories for path in (directory / 'refs').rglob('*.lock'))
    for path in sorted(locks | {index}):
        try:
            identity = signature(path)
            report.append({'path': str(path), 'kind': 'git', 'state': 'present',
                           'age_seconds': round(time.time() - max(identity[3:]) / 1e9, 1),
                           'blockers': process_blockers(identity)})
        except FileNotFoundError:
            report.append({'path': str(path), 'kind': 'git', 'state': 'absent'})
        except (OSError, RuntimeError) as error:
            report.append({'path': str(path), 'kind': 'git', 'state': 'unknown', 'error': str(error)})
    runtime = set(args.runtime_lock)
    runtime.update((root / 'runtime/web').glob('*.lock'))
    for name in ('datasets', 'outputs'):
        runtime.update((root / name).rglob('.*.rlt-writer.lock'))
    runtime.update(Path('/tmp/vla-hardware-locks').glob('*.lock'))
    runtime.update(Path('/tmp').glob(f'vla-validation-domain-*-{os.getuid()}.lock'))
    for path in sorted(runtime):
        try:
            report.append(runtime_status(path))
        except (OSError, RuntimeError) as error:
            report.append({'path': str(path), 'kind': 'runtime', 'state': 'unknown', 'error': str(error)})
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
