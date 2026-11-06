"""Build training metadata without changing absolute parquet records."""
import argparse
from dataclasses import replace
from pathlib import Path
from vla.actions import ActionSpec, SamplingSpec
from vla.actions.dataset import dataset_statistics
from vla.actions.metadata import save_model_contract, validate_mixture
from vla.dataset_metadata import load_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--representation', choices=['absolute', 'chunk_relative'], required=True)
    parser.add_argument('--chunk-length', type=int, default=50)
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--allow-derived', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; choose a new contract path')
    info = load_info(args.dataset)
    source = info.get('recording_contract') or info.get('processing', {}).get('source_contract', {}).get('recording_contract')
    if not source or source.get('version') != 1 or source.get('backend') != 'ros2_joint_impedance':
        raise ValueError('missing native ROS action contract; legacy data needs explicit verified conversion')
    spec = replace(ActionSpec.from_dict(info['action_spec']), representation=args.representation)
    sampling = SamplingSpec(args.chunk_length, args.frame_stride)
    sources = validate_mixture([source], allow_derived=args.allow_derived)
    stats = dataset_statistics(args.dataset, spec, sampling)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    value = save_model_contract(args.output, spec=spec, sampling=sampling, statistics=stats,
                        data_fingerprint=stats['contract']['data_fingerprint'], sources=sources)
    import json
    value["state_fields"] = info["features"]["observation.state"]["names"]
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    print(args.output)


if __name__ == '__main__':
    main()
