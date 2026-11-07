"""Inspect or publish a durable staging prefix while collection is stopped."""
import argparse
import json
from pathlib import Path
import numpy as np
from vla.dataset_metadata import load_info
from vla.datacollection.episode_journal import recover_staging, materialize, publish_audit
from vla.datacollection.franka_robotiq_single_left.v30_writer import V30DatasetWriter
from vla.core.online_contract import durable_json


def recover_episode(root, staging, *, outcome=None):
    if outcome is None:
        return _recover_episode(root, staging, outcome=None)
    from vla.core.writer_lock import WriterLease
    with WriterLease(root):
        return _recover_episode(root, staging, outcome=outcome)


def _recover_episode(root, staging, *, outcome=None):
    root, staging = Path(root).resolve(), Path(staging).resolve()
    if staging.parent != root/'staging':
        raise ValueError('staging directory must belong to this dataset')
    recovery = recover_staging(staging)
    if outcome is None:
        return {'durable_frames': recovery.frames, 'complete': recovery.complete,
                'incomplete_tail': recovery.incomplete_tail}
    if outcome not in {'success', 'failure'}:
        raise ValueError('explicit episode outcome required')
    info = load_info(root)
    if info.get('recording_schema_version') != 5:
        raise ValueError('only new-format episodes can be recovered')
    header = json.loads((staging/'episode.json').read_text())
    if (staging/'published.json').is_file():
        return json.loads((staging/'published.json').read_text())
    features = info['features']
    cameras = tuple(k for k, v in features.items() if v['dtype'] == 'video')
    height, width = features[cameras[0]]['shape'][:2]
    standard = {'observation.state', 'action', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'}
    extras = {k: v for k, v in features.items() if k not in standard and k not in cameras}
    writer = V30DatasetWriter(root, info['fps'], header['task'], cameras, height, width,
                             state_fields=tuple(features['observation.state']['names']),
                             action_fields=tuple(features['action']['names']), extra_features=extras,
                             robot_type=info['robot_type'], allow_multiple_tasks=True)
    rows = writer._read_episode_rows()
    matches = rows[rows['journal_id'] == header['journal_id']] if 'journal_id' in rows else []
    if len(matches):
        writer._rebuild_metadata()
        result = {'episode_index': int(matches.iloc[0]['episode_index'])}
        durable_json(staging/'published.json', result)
        return result
    columns, image_stats, recovery = materialize(staging, writer, allow_incomplete=True)
    metadata = {k: header[k] for k in ('journal_id', 'run_uuid', 'dataset_uuid', 'episode_uuid',
                                      'online_warmup', 'policy_revision') if k in header}
    metadata.update(episode_success=outcome, incomplete_tail=recovery.incomplete_tail)
    # No synthetic endpoint, terminal marker, image, action or reward is created.
    publish_audit(staging, root/'execution'/f'episode-{writer.next_episode_index():06d}',
                  episode_index=writer.next_episode_index())
    episode = writer.save_episode(columns['observation.state'], columns['action'], image_stats,
                                  frame_features={k: columns[k] for k in extras},
                                  extra_episode_metadata=metadata, task=header['task'])
    result = {'episode_index': episode, 'durable_frames': recovery.frames,
              'incomplete_tail': recovery.incomplete_tail}
    durable_json(staging/'published.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--staging', type=Path, required=True)
    parser.add_argument('--outcome', choices=['success', 'failure'], help='Omit for read-only inspection')
    args = parser.parse_args()
    print(json.dumps(recover_episode(args.dataset, args.staging, outcome=args.outcome), indent=2))


if __name__ == '__main__':
    main()
