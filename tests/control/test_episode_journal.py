from threading import Event
import json
import numpy as np
import pytest
from vla.datacollection.episode_journal import EpisodeJournal, StorageBlocked, recover_staging
from vla.actions.windows import window_ends, window_indices


def frame(i):
    return {'action': np.array([i, .5]), 'slot_id': np.array([i])}


def test_durable_prefix_and_idempotent_close(tmp_path):
    journal = EpisodeJournal(tmp_path/'stage', identity={'episode': 'x'}, fps=3)
    journal.event(('request', {'prefix': np.ones((2, 2))}))
    for i in range(7):
        journal.submit(frame(i), {})
    first = journal.close()
    assert first.complete and first.frames == 7 and len(first.segments) == 3
    assert journal.close() == first
    (journal.path/'complete.json').unlink()
    last = journal.path/first.segments[-1]['arrays']
    last.write_bytes(b'crash tail')
    recovered = recover_staging(journal.path)
    assert recovered.frames == 6 and recovered.incomplete_tail


def test_bounded_queue_never_blocks_control_or_drops_silently(tmp_path, monkeypatch):
    entered, release = Event(), Event()
    original = EpisodeJournal._flush
    def flush(self):
        entered.set()
        assert release.wait(2)
        original(self)
    monkeypatch.setattr(EpisodeJournal, '_flush', flush)
    journal = EpisodeJournal(tmp_path/'stage', identity={}, fps=1, capacity=1)
    journal.submit(frame(0), {})
    assert entered.wait(2)
    journal.submit(frame(1), {})
    with pytest.raises(StorageBlocked, match='overflow'):
        journal.submit(frame(2), {})
    release.set()
    assert journal.close().frames == 2


def test_disk_failure_preserves_and_retries_exact_tail(tmp_path, monkeypatch):
    original = EpisodeJournal._flush
    def fail(self):
        raise OSError(28, 'disk full')
    monkeypatch.setattr(EpisodeJournal, '_flush', fail)
    journal = EpisodeJournal(tmp_path/'stage', identity={}, fps=2)
    for i in range(2):
        journal.submit(frame(i), {})
    journal.thread.join(2)
    with pytest.raises(StorageBlocked, match='disk full'):
        journal.close()
    assert journal.path.exists() and len(journal.frames) == 2
    monkeypatch.setattr(EpisodeJournal, '_flush', original)
    journal.retry()
    assert journal.close().frames == 2


def test_windows_stop_at_unsampled_invalid_slot_and_segment():
    valid = np.ones(8, bool)
    valid[1] = False
    ends = window_ends(valid, slots=[0,1,2,3,5,6,7,8], segments=[0,0,0,0,0,0,1,1])
    assert window_indices(ends, 0, 4, 2).tolist() == [0]
    assert window_indices(ends, 2, 4, 2).tolist() == [2]
    assert window_indices(ends, 4, 4).tolist() == [4,5]


def test_video_segments_have_exact_frame_mapping(tmp_path):
    journal = EpisodeJournal(tmp_path/'stage', identity={}, fps=3)
    for i in range(7):
        journal.submit(frame(i), {'observation.images.base': np.full((16, 16, 3), i*20, np.uint8)})
    recovered = journal.close()
    assert recovered.frames == 7 and [s['frames'] for s in recovered.segments] == [3,3,1]
