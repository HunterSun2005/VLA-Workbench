"""Real disposable subprocesses exercise blocked pipes, stderr and deadlines."""
import sys
import time

import numpy as np
import pytest

from vla.core.media_process import MediaProcessError, StreamingEncoder, run_media


@pytest.mark.parametrize('operation', ['write', 'close'])
def test_stuck_encoder_times_out_and_is_reaped(operation):
    encoder = StreamingEncoder([sys.executable, '-c', 'import time; time.sleep(60)'], timeout_s=.15)
    started = time.monotonic()
    with pytest.raises(MediaProcessError, match='timed out'):
        if operation == 'write':
            encoder.write(b'x' * 1_000_000)  # Exceeds pipe capacity; child never reads.
        else:
            encoder.close()
    assert time.monotonic()-started < 3
    assert encoder.process.poll() is not None
    assert encoder.process.stdin.closed and encoder.process.stderr.closed
    encoder.abort()


def test_encoder_drains_large_stderr_and_retains_bounded_diagnostics():
    code = "import sys; sys.stderr.write('x'*1000000); sys.stderr.flush(); sys.stdin.buffer.read()"
    encoder = StreamingEncoder([sys.executable, '-c', code], timeout_s=3)
    encoder.write(b'y' * 1_000_000)
    encoder.close()
    encoder.close()
    assert encoder.process.returncode == 0
    assert len(encoder.stderr) == 65536


def test_encoder_failure_reports_stderr_and_closes_pipes():
    encoder = StreamingEncoder([sys.executable, '-c',
                                "import sys; sys.stderr.write('bad codec'); sys.exit(7)"], timeout_s=3)
    with pytest.raises(MediaProcessError, match='bad codec'):
        encoder.close()
    assert encoder.process.returncode == 7


def test_media_command_timeout_is_bounded():
    with pytest.raises(MediaProcessError, match='timed out'):
        run_media([sys.executable, '-c', 'import time; time.sleep(60)'], timeout_s=.1)


def test_encoder_timeout_preserves_journal_tail_and_can_retry(tmp_path, monkeypatch):
    from vla.datacollection.episode_journal import EpisodeJournal, StorageBlocked
    from vla.datacollection.franka_robotiq_rollout import common
    real_encoder = common.StreamingEncoder
    children = []
    def stalled(*args, **kwargs):
        child = real_encoder([sys.executable, '-c', 'import time; time.sleep(60)'], timeout_s=.1)
        children.append(child)
        return child
    monkeypatch.setattr(common, 'StreamingEncoder', stalled)
    journal = EpisodeJournal(tmp_path/'stage', identity={}, fps=1)
    journal.submit({'action': np.array([.1])}, {'observation.images.base': np.zeros((16,16,3), np.uint8)})
    journal.thread.join(3)
    assert not journal.thread.is_alive()
    with pytest.raises(StorageBlocked, match='timed out'):
        journal.close()
    assert len(journal.frames) == 1 and not list(journal.path.glob('segment-*.json'))
    with pytest.raises(StorageBlocked, match='timed out'):
        journal.retry()  # A repeated failure must remain retryable.
    assert journal.error is not None
    monkeypatch.setattr(common, 'StreamingEncoder', real_encoder)
    journal.retry()
    result = journal.close()
    assert result.frames == 1 and result.complete
    assert all(child.process.poll() is not None for child in children)
