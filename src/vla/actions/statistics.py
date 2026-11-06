"""Statistics over exactly the valid samples seen by chunk training."""
from dataclasses import asdict, dataclass
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


def compute_statistics(episodes, spec, sampling, data_fingerprint):
    """Episodes yield (absolute targets [T,D], raw measured anchors [T,D], valid [T]).

    Iterate each episode independently; never use replicated padding in quantiles.
    A valid target contributes once for every training chunk in which it occurs.
    """
    samples = []
    for targets, anchors, valid in episodes:
        targets, anchors, valid = np.asarray(targets), np.asarray(anchors), np.asarray(valid, dtype=bool)
        if targets.ndim != 2 or targets.shape != anchors.shape or valid.shape != (len(targets),):
            raise ValueError('episode targets, anchors and mask do not align')
        for start in range(0, len(targets), sampling.start_stride):
            if not valid[start]:
                continue
            indices = start + np.arange(sampling.chunk_length) * sampling.frame_stride
            indices = indices[indices < len(targets)]
            indices = indices[valid[indices]]
            if len(indices):
                samples.append(DeltaActions(spec)(targets[indices], anchors[start]))
    if not samples:
        raise ValueError('no valid training actions')
    values = np.concatenate(samples)
    contract = statistics_contract(spec, sampling, data_fingerprint)
    return {'contract': contract, 'digest': fingerprint(contract), 'count': len(values),
            **{name: value.tolist() for name, value in {
                'min': values.min(0), 'max': values.max(0), 'mean': values.mean(0),
                'std': values.std(0), 'q01': np.quantile(values, .01, axis=0),
                'q99': np.quantile(values, .99, axis=0)}.items()}}


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
