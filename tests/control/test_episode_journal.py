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


@pytest.mark.parametrize('damage', ['json', 'schema', 'missing', 'gap', 'checksum', 'npz',
                                   'events', 'layout', 'unhashed', 'traversal', 'symlink'])
def test_corrupt_middle_segment_keeps_only_verified_prefix(tmp_path, damage):
    from vla.core.online_contract import file_digest
    journal = EpisodeJournal(tmp_path/'stage', identity={}, fps=1)
    for i in range(3):
        journal.submit(frame(i), {})
    journal.close()
    path = journal.path/'segment-000001.json'
    value = json.loads(path.read_text())
    payload = journal.path/value['arrays']
    if damage == 'json':
        path.write_text('{')
    elif damage == 'missing':
        payload.unlink()
    elif damage == 'gap':
        path.unlink()
    elif damage == 'checksum':
        payload.write_bytes(b'corrupt payload')
    elif damage == 'symlink':
        target = tmp_path/'outside.npz'
        payload.rename(target)
        payload.symlink_to(target)
    else:
        if damage == 'schema':
            value['schema_version'] = 4
        elif damage == 'npz':
            payload.write_bytes(b'not an archive')
            value['files'][payload.name] = file_digest(payload)
        elif damage == 'events':
            event = journal.path/value['events']
            event.write_text('{')
            value['files'][event.name] = file_digest(event)
        elif damage == 'layout':
            np.savez(payload, other=np.zeros((1,3)))
            value['files'][payload.name] = file_digest(payload)
        elif damage == 'unhashed':
            value['files'].pop(value['arrays'])
        elif damage == 'traversal':
            value['arrays'] = '../outside.npz'
            value['files'] = {name: '' for name in (value['arrays'], value['events'])}
        path.write_text(json.dumps(value))
    before = {p.name: p.read_bytes() for p in journal.path.iterdir() if p.is_file()}
    recovered = recover_staging(journal.path)
    assert recovered.frames == 1 and len(recovered.segments) == 1
    assert not recovered.complete and recovered.incomplete_tail
    assert recovered.issues[0].file == ('segment-000002.json' if damage == 'gap' else path.name)
    assert before == {p.name: p.read_bytes() for p in journal.path.iterdir() if p.is_file()}


@pytest.mark.parametrize('marker', ['complete.json', 'events-tail-000002.json'])
def test_corrupt_completion_preserves_all_verified_frames(tmp_path, marker):
    journal = EpisodeJournal(tmp_path/'stage', identity={}, fps=1)
    for i in range(2):
        journal.submit(frame(i), {})
    journal.close()
    (journal.path/marker).write_text('{')
    recovered = recover_staging(journal.path)
    assert recovered.frames == 2 and recovered.incomplete_tail and not recovered.complete
    assert recovered.issues[0].file == marker


def test_video_probe_timeout_reports_unverified_tail(tmp_path, monkeypatch):
    import vla.datacollection.episode_journal as module
    from vla.core.media_process import MediaProcessError
    journal = EpisodeJournal(tmp_path/'stage', identity={}, fps=1)
    for i in range(2):
        journal.submit(frame(i), {'observation.images.base': np.zeros((16,16,3), np.uint8)})
    journal.close()
    original = module.video_frames
    def probe(path):
        if path.name.startswith('segment-000001'):
            raise MediaProcessError('ffprobe timed out')
        return original(path)
    monkeypatch.setattr(module, 'video_frames', probe)
    recovered = recover_staging(journal.path)
    assert recovered.frames == 1 and 'timed out' in recovered.issues[0].reason
