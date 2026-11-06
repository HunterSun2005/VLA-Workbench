"""Statistics over exactly the valid samples seen by chunk training."""
from dataclasses import asdict, dataclass
from contextlib import ExitStack
from tempfile import TemporaryFile
import numpy as np
from .spec import ActionSpec, fingerprint
from .transforms import DeltaActions


@dataclass(frozen=True)
class SamplingSpec:
    chunk_length: int
    frame_stride: int = 1
    start_stride: int = 1

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in asdict(self).values()):
            raise ValueError('chunk length and sampling strides must be positive integers')


def statistics_contract(spec: ActionSpec, sampling: SamplingSpec, data_fingerprint: str):
    if not data_fingerprint:
        raise ValueError('statistics require a dataset content fingerprint')
    return {'version': 1, 'action_spec': spec.to_dict(), 'sampling': asdict(sampling),
            'data_fingerprint': data_fingerprint, 'padding': 'exclude', 'quantiles': [0.01, 0.99]}


_TRANSFORM_ROWS = 8192


def _training_samples(episodes, spec, sampling):
    transform = DeltaActions(spec)
    for targets, anchors, valid in episodes:
        targets, anchors, valid = np.asarray(targets), np.asarray(anchors), np.asarray(valid, dtype=bool)
        if targets.ndim != 2 or targets.shape != anchors.shape or valid.shape != (len(targets),):
            raise ValueError('episode targets, anchors and mask do not align')
        for start in range(0, len(targets), sampling.start_stride):
            if not valid[start]:
                continue
            # Bound even a very long requested horizon; do not construct padding.
            length = min(sampling.chunk_length, (len(targets)-1-start)//sampling.frame_stride+1)
            for offset in range(0, length, _TRANSFORM_ROWS):
                indices = start + np.arange(offset, min(offset+_TRANSFORM_ROWS, length)) * sampling.frame_stride
                indices = indices[valid[indices]]
                if len(indices):
                    yield transform(targets[indices], anchors[start])


def compute_statistics(episodes, spec, sampling, data_fingerprint, *, temporary_directory=None):
    """Episodes yield (absolute targets [T,D], raw measured anchors [T,D], valid [T]).

    Iterate each episode independently; never use replicated padding in quantiles.
    A valid target contributes once for every training chunk in which it occurs.

    Spool float64 samples by dimension, then reduce one dimension at a time.
    Quantiles remain exact NumPy linear quantiles, with no reservoir/sketch.
    For M expanded samples, working RAM is O(M), not O(M*D); std may allocate
    one extra column. Temporary disk is 8*M*D bytes. Files close on all exits.
    temporary_directory defaults to tempfile's directory (including TMPDIR).
    """
    contract = statistics_contract(spec, sampling, data_fingerprint)
    result = {name: [] for name in ('min', 'max', 'mean', 'std', 'q01', 'q99')}
    count = 0
    with ExitStack() as stack:
        columns = [stack.enter_context(TemporaryFile(dir=temporary_directory, buffering=65536))
                   for _ in range(spec.dim)]
        for chunk in _training_samples(episodes, spec, sampling):
            for column, stream in enumerate(columns):
                stream.write(chunk[:, column].tobytes())
            count += len(chunk)
        if not count:
            raise ValueError('no valid training actions')
        for stream in columns:
            stream.flush()
            stream.seek(0)
            values = np.fromfile(stream, dtype=np.float64, count=count)
            if len(values) != count:
                raise OSError('incomplete temporary action statistics column')
            for name, value in (('min', values.min()), ('max', values.max()),
                                ('mean', values.mean()), ('std', values.std())):
                result[name].append(float(value))
            low, high = np.quantile(values, [.01, .99], overwrite_input=True)
            result['q01'].append(float(low))
            result['q99'].append(float(high))
            del values
    return {'contract': contract, 'digest': fingerprint(contract), 'count': count, **result}


def validate_statistics(stats, spec, sampling, data_fingerprint):
    expected = statistics_contract(spec, sampling, data_fingerprint)
    if stats.get('contract') != expected or stats.get('digest') != fingerprint(expected):
        raise ValueError('action statistics contract mismatch; recompute for this dataset and chunk sampler')


class QuantileNormalizer:
    def __init__(self, stats):
        self.low, self.high = np.asarray(stats['q01']), np.asarray(stats['q99'])
        if self.low.shape != self.high.shape or not np.isfinite([self.low, self.high]).all() or np.any(self.high < self.low):
            raise ValueError('invalid action quantiles')
        self.scale = self.high - self.low + 1e-8

    def normalize(self, actions):
        return 2 * (np.asarray(actions) - self.low) / self.scale - 1

    def unnormalize(self, actions):
        return (np.asarray(actions) + 1) * self.scale / 2 + self.low
