#!/usr/bin/env python3
"""Synthetic-only peak RSS comparison. Run each implementation in a fresh process."""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from vla.actions import ActionSpec, DeltaActions, SamplingSpec, compute_statistics


def expanded_statistics(targets, anchors, spec, sampling):
    # Previous implementation, intentionally retaining every overlapping chunk.
    samples = []
    for start in range(len(targets)):
        indices = start + np.arange(sampling.chunk_length)
        indices = indices[indices < len(targets)]
        samples.append(DeltaActions(spec)(targets[indices], anchors[start]))
    values = np.concatenate(samples)
    return {'count': len(values), 'min': values.min(0).tolist(), 'max': values.max(0).tolist(),
            'mean': values.mean(0).tolist(), 'std': values.std(0).tolist(),
            'q01': np.quantile(values, .01, axis=0).tolist(),
            'q99': np.quantile(values, .99, axis=0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation', choices=['expanded', 'columns'], required=True)
    parser.add_argument('--frames', type=int, default=12000)
    parser.add_argument('--chunk-length', type=int, default=100)
    parser.add_argument('--temporary-directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1 or args.chunk_length < 1:
        parser.error('frames and chunk length must be positive')
    spec = ActionSpec('joint', ('left', 'right'), 'chunk_relative')
    sampling = SamplingSpec(args.chunk_length)
    rng = np.random.default_rng(20261106)
    targets = rng.normal(size=(args.frames, spec.dim))
    anchors = rng.normal(size=targets.shape)
    targets[:, 7::8] = rng.uniform(size=(args.frames, 2))
    anchors[:, 7::8] = rng.uniform(size=(args.frames, 2))
    start = time.perf_counter()
    if args.implementation == 'expanded':
        stats = expanded_statistics(targets, anchors, spec, sampling)
    else:
        stats = compute_statistics([(targets, anchors, np.ones(args.frames, bool))], spec,
                                   sampling, 'synthetic-benchmark', temporary_directory=args.temporary_directory)
    result = {'implementation': args.implementation, 'frames': args.frames,
              'chunk_length': args.chunk_length, 'dimensions': spec.dim,
              'elapsed_s': time.perf_counter()-start,
              'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'temporary_bytes': 8*stats['count']*spec.dim if args.implementation == 'columns' else 0,
              'statistics': stats}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'statistics'}))


if __name__ == '__main__':
    main()
