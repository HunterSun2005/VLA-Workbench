import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from vla.actions import (ActionSpec, ObservationSnapshot, ChunkContext, DeltaActions, AbsoluteActions,
                         matrix_to_rot6d, SamplingSpec, compute_statistics, validate_statistics,
                         QuantileNormalizer)
from vla.actions.chunks import ChunkCodec, AbsoluteChunkQueue


def sample(spec, n=7):
    rng = np.random.default_rng(12)
    value = rng.normal(size=(n, spec.dim))
    for off in range(0, spec.dim, spec.width):
        value[:, off+spec.width-1] = rng.uniform(0, 1, n)
        if spec.space == 'tcp':
            value[:, off+3:off+9] = matrix_to_rot6d(Rotation.random(n, random_state=rng).as_matrix())
    return value


@pytest.mark.parametrize('space', ['joint', 'tcp'])
@pytest.mark.parametrize('arms', [('left',), ('left', 'right')])
@pytest.mark.parametrize('representation', ['absolute', 'chunk_relative'])
def test_training_inference_roundtrip(space, arms, representation):
    spec = ActionSpec(space, arms, representation)
    absolute = sample(spec)
    anchor = absolute[2].copy()
    obs = ObservationSnapshot(dict(zip(reversed(spec.measured_fields), reversed(anchor))), 12)
    context = ChunkContext.capture(spec, obs, 0, 1)
    delta = DeltaActions(spec)(absolute, context.anchor)
    np.testing.assert_allclose(AbsoluteActions(spec)(delta, context.anchor), absolute, atol=1e-12)
    np.testing.assert_array_equal(delta[:, spec.width-1::spec.width], absolute[:, spec.width-1::spec.width])
    batched = DeltaActions(spec)(np.stack([absolute, absolute]), np.stack([anchor, anchor]))
    np.testing.assert_allclose(batched[1], delta)
    with pytest.raises(ValueError):
        context.anchor.setflags(write=True)
    with pytest.raises(ValueError, match='missing raw'):
        spec.anchor_from_fields({'unrelated_model_input': 3})


def test_noncommuting_rotations_use_base_frame():
    spec = ActionSpec(representation='chunk_relative')
    value = sample(spec, 2)
    target = Rotation.from_euler('x', .7).as_matrix()
    origin = Rotation.from_euler('z', .8).as_matrix()
    value[0, 3:9], value[1, 3:9] = matrix_to_rot6d(target), matrix_to_rot6d(origin)
    result = DeltaActions(spec)(value[:1], value[1])
    np.testing.assert_allclose(result[0, 3:9], matrix_to_rot6d(target @ origin.T))
    with pytest.raises(ValueError, match='degenerate'):
        DeltaActions(spec)(np.zeros((2, 10)), value[1])


@pytest.mark.parametrize('length,stride', [(1, 1), (4, 2), (12, 3)])
def test_statistics_masks_episode_boundaries_and_cache(length, stride):
    spec = ActionSpec('joint', representation='chunk_relative')
    targets = sample(spec)
    anchors = sample(spec) * .5
    valid = np.array([1, 1, 0, 1, 1, 1, 1], dtype=bool)
    sampling = SamplingSpec(length, stride)
    episodes = [(targets[:4], anchors[:4], valid[:4]), (targets[4:], anchors[4:], valid[4:])]
    stats = compute_statistics(episodes, spec, sampling, 'dataset-a')
    expected = []
    for t, a, v in episodes:
        for start in np.flatnonzero(v):
            for i in range(start, min(len(t), start+length*stride), stride):
                if v[i]:
                    expected.append(DeltaActions(spec)(t[i], a[start]))
    np.testing.assert_allclose(stats['q99'], np.quantile(expected, .99, axis=0))
    validate_statistics(stats, spec, sampling, 'dataset-a')
    with pytest.raises(ValueError, match='recompute'):
        validate_statistics(stats, spec, SamplingSpec(length+1, stride), 'dataset-a')


def test_latched_requests_rtc_reanchor_and_stale_generation():
    spec = ActionSpec('tcp', representation='chunk_relative')
    absolute = sample(spec)
    stats = compute_statistics([(absolute, absolute, np.ones(7, bool))], spec, SamplingSpec(4), 'x')
    norm = QuantileNormalizer(stats)
    first = ChunkCodec(ChunkContext(spec, absolute[0], 0, 0, 1), norm)
    second = ChunkCodec(ChunkContext(spec, absolute[3], 1, 0, 2), norm)
    queue = AbsoluteChunkQueue()
    assert queue.install(first, first.encode(absolute), consumed=2)
    expected = absolute[2:]
    np.testing.assert_allclose(queue.snapshot(), expected, atol=1e-12)
    np.testing.assert_allclose(second.decode(second.prefix(queue.snapshot())), expected, atol=1e-12)
    assert not np.allclose(first.encode(expected), second.prefix(expected))
    assert queue.install(second, second.encode(expected))
    assert not queue.install(first, first.encode(absolute))
    queue.invalidate()
    assert not queue.install(second, second.encode(expected))
    assert queue.pop() is None
    replay = first.replay(absolute, absolute, next_context=second.context, next_absolute=expected)
    np.testing.assert_allclose(second.decode(replay['next']), expected, atol=1e-12)


def test_checkpoint_and_derived_data_contracts(tmp_path):
    from vla.actions.metadata import (RecordingContract, validate_mixture, save_model_contract,
                                      load_model_contract, require_model_contract)
    spec = ActionSpec('joint')
    native = RecordingContract(spec, 'robot', 'tcp', 'urdf', 'control').to_dict()
    derived = RecordingContract(spec, 'robot', 'tcp', 'urdf', 'control', 'tcp_submitted_joints').to_dict()
    with pytest.raises(ValueError, match='explicit'):
        validate_mixture([derived])
    sources = validate_mixture([native, derived], allow_derived=True)
    targets = sample(spec)
    sampling = SamplingSpec(4)
    stats = compute_statistics([(targets, targets, np.ones(7, bool))], spec, sampling, 'x')
    saved = save_model_contract(tmp_path/'action_contract.json', spec=spec, sampling=sampling,
                                statistics=stats, data_fingerprint='x', sources=sources)
    actual_spec, _, actual = load_model_contract(tmp_path/'action_contract.json')
    assert actual_spec == spec
    require_model_contract(saved, actual)
    with pytest.raises(ValueError, match='mismatch'):
        require_model_contract(saved, {**actual, 'sources': []})
