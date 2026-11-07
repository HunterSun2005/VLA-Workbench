"""Lock ownership and crash recovery using temporary files, never hardware."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import check_locks
from vla.core.hardware_lease import HardwareLease
from vla.core.lease_metadata import release_owned_streams
from vla.core.writer_lock import WriterLease


def acquire(kind, root):
    if kind == 'writer':
        return WriterLease(root / 'data')
    return HardwareLease('test-only', '/test-only/serial', directory=root)


@pytest.mark.parametrize('kind', ['writer', 'hardware'])
def test_normal_release_clears_owner_but_preserves_inode(tmp_path, kind):
    lease = acquire(kind, tmp_path)
    paths = list(tmp_path.glob('*.lock')) + list(tmp_path.glob('.*.lock'))
    paths = set(paths)
    assert paths
    inodes = {p: p.stat().st_ino for p in paths}
    for path in paths:
        assert json.loads(path.read_text())['pid'] == os.getpid()
        assert check_locks.runtime_status(path)['state'] == 'held'
    with pytest.raises(RuntimeError, match='owned|owns'):
        acquire(kind, tmp_path)
    lease.close()
    lease.close()
    for path in paths:
        assert path.stat().st_ino == inodes[path]
        assert path.read_text() == ''
        assert check_locks.runtime_status(path)['state'] == 'idle_persistent_file'
    with acquire(kind, tmp_path):
        assert all(p.stat().st_ino == inodes[p] for p in paths)


@pytest.mark.parametrize('kind', ['writer', 'hardware'])
def test_killed_owner_releases_kernel_lock_without_deleting_file(tmp_path, kind):
    code = '''
import sys
from pathlib import Path
from vla.core.writer_lock import WriterLease
from vla.core.hardware_lease import HardwareLease
root = Path(sys.argv[2])
lease = (WriterLease(root/'data') if sys.argv[1] == 'writer' else
         HardwareLease('test-only', '/test-only/serial', directory=root))
print('ready', flush=True)
sys.stdin.read()
'''
    child = subprocess.Popen([sys.executable, '-c', code, kind, str(tmp_path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        import select
        assert select.select([child.stdout], [], [], 10)[0]
        assert child.stdout.readline().strip() == 'ready'
        with pytest.raises(RuntimeError):
            acquire(kind, tmp_path)
        child.kill()
        child.wait(timeout=10)
        paths = set(tmp_path.glob('*.lock')) | set(tmp_path.glob('.*.lock'))
        assert paths
        for path in paths:
            assert json.loads(path.read_text())['pid'] == child.pid
            assert check_locks.runtime_status(path)['state'] == 'idle_persistent_file'
        with acquire(kind, tmp_path):
            pass
        assert all(p.read_text() == '' for p in paths)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
        child.stdin.close()
        child.stdout.close()


def test_cleanup_error_still_releases_all_descriptors():
    class Broken(io.StringIO):
        def truncate(self, *args):
            raise OSError('injected disk error')
    first, second = io.StringIO('owner'), Broken('owner')
    streams = [first, second]
    with pytest.raises(OSError, match='disk error'):
        release_owned_streams(streams)
    assert not streams and first.closed and second.closed


def test_partial_acquisition_does_not_clear_other_owner(tmp_path):
    first, second = tmp_path / 'a', tmp_path / 'b'
    with WriterLease(second):
        owner = (tmp_path / '.b.rlt-writer.lock').read_text()
        with pytest.raises(RuntimeError):
            WriterLease(first, second)
        assert (tmp_path / '.a.rlt-writer.lock').read_text() == ''
        assert (tmp_path / '.b.rlt-writer.lock').read_text() == owner
        with WriterLease(first):
            pass


@pytest.mark.parametrize('reason', ['recent', 'busy', 'changed', 'symlink', 'hardlink', 'stale'])
def test_stale_index_cleanup_is_conservative(tmp_path, monkeypatch, reason):
    path = tmp_path / 'index.lock'
    path.write_text('transaction')
    now = check_locks.time.time()
    monkeypatch.setattr(check_locks.time, 'time', lambda: now + (0 if reason == 'recent' else 300))
    def blockers(_):
        if reason == 'changed':
            path.write_text('new transaction')
        return ['Git process still active'] if reason == 'busy' else []
    monkeypatch.setattr(check_locks, 'process_blockers', blockers)
    if reason == 'symlink':
        target = tmp_path / 'target'
        path.rename(target)
        path.symlink_to(target)
    if reason == 'hardlink':
        os.link(path, tmp_path / 'other')
    if reason == 'stale':
        check_locks.clear_stale_index(path)
        assert not path.exists()
    else:
        with pytest.raises(RuntimeError):
            check_locks.clear_stale_index(path)
        assert path.exists()


def test_process_scan_finds_open_file(tmp_path):
    path = tmp_path / 'index.lock'
    path.write_text('')
    with path.open() as stream:
        assert stream.fileno() >= 0
        blockers = check_locks.process_blockers(check_locks.signature(path))
    assert f'open descriptor in process {os.getpid()}' in blockers


def test_process_scan_reports_incomplete_inspection(tmp_path, monkeypatch):
    original = Path.iterdir
    def restricted(path):
        if str(path) == f'/proc/{os.getpid()}/fd':
            raise PermissionError('restricted process')
        return original(path)
    monkeypatch.setattr(Path, 'iterdir', restricted)
    path = tmp_path / 'index.lock'
    path.write_text('')
    assert f'cannot inspect process {os.getpid()}' in check_locks.process_blockers(check_locks.signature(path))


def test_cli_only_repairs_index_and_leaves_dependency_runtime_files(tmp_path, monkeypatch, capsys):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    runtime = tmp_path / 'runtime/web/manager.lock'
    runtime.parent.mkdir(parents=True)
    runtime.write_text('')
    dependency = tmp_path / 'requirements.lock'
    dependency.write_text('pinned requirements')
    index = tmp_path / '.git/index.lock'
    index.write_text('stale')
    now = check_locks.time.time()
    monkeypatch.setattr(check_locks.time, 'time', lambda: now + 300)
    monkeypatch.setattr(check_locks, 'process_blockers', lambda _: [])
    check_locks.main(['--root', str(tmp_path)])
    assert index.exists()  # Default is diagnosis only.
    capsys.readouterr()
    check_locks.main(['--root', str(tmp_path), '--clear-stale-index'])
    report = json.loads(capsys.readouterr().out)
    assert any(r['state'] == 'removed_stale_index' for r in report)
    assert not index.exists() and runtime.exists()
    assert dependency.read_text() == 'pinned requirements'
