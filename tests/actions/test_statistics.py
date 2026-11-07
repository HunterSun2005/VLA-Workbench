"""Exact statistics equivalence and temporary-storage failure boundaries."""
import json
import errno

import numpy as np
import pytest

from vla.actions import ActionSpec, DeltaActions, SamplingSpec, compute_statistics
from vla.actions import statistics
from tests.actions.test_contract import sample


def expanded_reference(episodes, spec, sampling):
    values = []
    for targets, anchors, valid in episodes:
        for start in range(0, len(targets), sampling.start_stride):
            if valid[start]:
                for index in range(start, min(len(targets), start+sampling.chunk_length*sampling.frame_stride),
                                   sampling.frame_stride):
                    if not valid[start:index+1].all():
                        break
                    if valid[index]:
                        values.append(DeltaActions(spec)(targets[index], anchors[start]))
    values = np.asarray(values)
    return {'count': len(values), 'min': values.min(0), 'max': values.max(0),
            'mean': values.mean(0), 'std': values.std(0),
            'q01': np.quantile(values, .01, axis=0), 'q99': np.quantile(values, .99, axis=0)}


@pytest.mark.parametrize('space', ['joint', 'tcp'])
@pytest.mark.parametrize('arms', [('left',), ('left', 'right')])
@pytest.mark.parametrize('representation', ['absolute', 'chunk_relative'])
def test_exact_statistics_match_expanded_chunks(tmp_path, monkeypatch, space, arms, representation):
    spec = ActionSpec(space, arms, representation)
    sampling = SamplingSpec(11, frame_stride=2, start_stride=3)
    episodes = []
    for length in (13, 17):
        targets, anchors = sample(spec, length), sample(spec, length)[::-1].copy()
        targets[:, spec.width-1::spec.width] = .4  # Constant dimensions and repeated quantiles.
        valid = np.ones(length, dtype=bool)
        valid[[2, 3, 7]] = False
        episodes.append((targets, anchors, valid))
    monkeypatch.setattr(statistics, '_TRANSFORM_ROWS', 2)  # A chunk spans several write batches.
    actual = compute_statistics(iter(episodes), spec, sampling, 'dataset', temporary_directory=tmp_path)
    expected = expanded_reference(episodes, spec, sampling)
    assert actual['count'] == expected['count']
    for key in ('min', 'max', 'q01', 'q99'):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-14)
    for key in ('mean', 'std'):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-14)
    statistics.validate_statistics(actual, spec, sampling, 'dataset')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('failure', ['empty', 'invalid', 'interrupted', 'disk_full'])
def test_statistics_closes_temporary_files_on_failure(tmp_path, monkeypatch, failure):
    opened = []
    original = statistics.TemporaryFile
    def tracked(**kwargs):
        stream = original(**kwargs)
        opened.append(stream)
        if failure == 'disk_full':
            class FullDisk:
                def __enter__(self):
                    return self
                def __exit__(self, *_):
                    stream.close()
                def write(self, _):
                    raise OSError(errno.ENOSPC, 'injected full temporary disk')
            return FullDisk()
        return stream
    monkeypatch.setattr(statistics, 'TemporaryFile', tracked)
    def episodes():
        if failure == 'empty':
            return
        values = np.zeros((1, 8))
        if failure == 'invalid':
            values[0, 0] = np.nan
        yield values, np.zeros_like(values), [True]
        raise OSError('source read failed')
    with pytest.raises((ValueError, OSError)) as error:
        compute_statistics(episodes(), ActionSpec('joint'), SamplingSpec(3), 'data', temporary_directory=tmp_path)
    if failure == 'disk_full':
        assert error.value.errno == errno.ENOSPC
    assert opened and all(stream.closed for stream in opened)
    assert not list(tmp_path.iterdir())


def test_projected_episode_reads_preserve_shuffled_shards_and_segments(tmp_path, monkeypatch):
    import pandas as pd
    import pyarrow.parquet as pq
    from vla.actions.dataset import dataset_statistics

    spec = ActionSpec('joint', representation='chunk_relative')
    sampling = SamplingSpec(4, frame_stride=2, start_stride=2)
    values = sample(spec, 16)
    rows = pd.DataFrame({'episode_index': [0]*8+[1]*8, 'frame_index': list(range(8))*2,
        'action': list(values), 'observation.anchor': list(values[::-1]),
        'complementary_info.segment_id': [[0]]*5+[[1]]*3+[[0]]*4+[[1]]*4,
        'complementary_info.action_valid': [[int(i not in (2, 7, 9))] for i in range(16)],
        'observation.state': [[0.]*100]*16, 'observation.images.base': ['unused']*16})
    rows['complementary_info.supervision_valid'] = [[1]]*len(rows)
    rows['complementary_info.observation_valid'] = [[1]]*len(rows)
    rows['complementary_info.slot_id'] = [[i] for i in range(len(rows))]
    (tmp_path/'meta').mkdir()
    (tmp_path/'data').mkdir()
    (tmp_path/'meta/info.json').write_text(json.dumps({'action_spec': ActionSpec('joint').to_dict(), 'recording_schema_version': 5}))
    rows.iloc[::2].iloc[::-1].to_parquet(tmp_path/'data/a.parquet', index=False)
    rows.iloc[1::2].iloc[::-1].to_parquet(tmp_path/'data/b.parquet', index=False)
    original = pq.read_table
    reads = []
    def projected(path, **kwargs):
        assert 'observation.state' not in kwargs['columns']
        assert 'observation.images.base' not in kwargs['columns']
        reads.append(kwargs['filters'])
        return original(path, **kwargs)
    monkeypatch.setattr(pq, 'read_table', projected)
    actual = dataset_statistics(tmp_path, spec, sampling, temporary_directory=tmp_path)
    episodes = []
    rows['_segment'] = np.stack(rows['complementary_info.segment_id']).reshape(-1)
    for _, group in rows.groupby(['episode_index', '_segment'], sort=False):
        episodes.append((np.stack(group['action']), np.stack(group['observation.anchor']),
                         np.stack(group['complementary_info.action_valid']).reshape(-1).astype(bool)))
    expected = expanded_reference(episodes, spec, sampling)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-14)
    assert reads == [[('episode_index', '=', i)] for i in (0, 0, 1, 1)]
