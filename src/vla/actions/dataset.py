"""Offline action export and sampling, without model or ROS imports."""
from dataclasses import replace
import hashlib
from pathlib import Path
import numpy as np
from .spec import ActionSpec
from .statistics import compute_statistics


def dataset_fingerprint(root):
    digest = hashlib.sha256()
    root = Path(root)
    paths = [root/'meta/info.json', *([root/'meta/vla_metadata.json'] if (root/'meta/vla_metadata.json').is_file() else []), *sorted((root/'data').rglob('*.parquet'))]
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        with path.open('rb') as stream:
            while chunk := stream.read(8*1024*1024):
                digest.update(chunk)
    return digest.hexdigest()


def anchors_from_state(states, names, spec, *, gripper_unit):
    if len(names) != len(set(names)):
        raise ValueError('ambiguous measured field names')
    measured = dict(zip(names, np.asarray(states).T))
    if gripper_unit not in ('metres', 'open_fraction'):
        raise ValueError('measured gripper unit must be declared')
    if gripper_unit == 'metres':
        for name in spec.measured_fields:
            if name.endswith('gripper_position') and name in measured:
                measured[name] = np.clip(measured[name] / .085, 0, 1)
    return spec.anchor_from_fields(measured)


def export_actions(table, source_spec, target_space, *, allow_derived=False):
    """Conversion uses recorded commands only; never invent joints with IK or measurements."""
    target_spec = replace(source_spec, space=target_space, representation='absolute')
    if source_spec.space == target_space:
        return np.stack(table['action']), target_spec, None
    if not allow_derived:
        raise ValueError('cross-space export requires allow_derived and records derivation provenance')
    field, derivation = (('command.joint', 'tcp_submitted_joints') if target_space == 'joint'
                         else ('command.fk_tcp', 'joint_command_fk'))
    if field not in table:
        raise ValueError(f'cannot reconstruct original commands: missing {field}')
    actions = np.stack(table[field])
    if actions.shape != (len(table), target_spec.dim) or not np.isfinite(actions).all():
        raise ValueError('invalid auxiliary absolute commands')
    return actions, target_spec, derivation


def dataset_statistics(root, spec, sampling, *, temporary_directory=None):
    import pyarrow.parquet as pq
    from vla.dataset_metadata import load_info
    info = load_info(root)
    recorded = ActionSpec.from_dict(info['action_spec'])
    spec.require_compatible(recorded, training=False)
    def episodes():
        import pandas as pd
        paths = sorted((Path(root)/'data').rglob('*.parquet'))
        if not paths:
            raise ValueError('dataset has no action rows')
        # Index only episode IDs, then load one episode's required columns.
        # Shard boundaries must not become chunk/segment boundaries.
        by_episode, schemas = {}, {}
        for path in paths:
            with pq.ParquetFile(path) as parquet:
                schemas[path] = set(parquet.schema_arrow.names)
                if 'observation.anchor' not in schemas[path]:
                    raise ValueError('dataset lacks independent raw observation.anchor; re-export from full measured state')
                episode_ids = set()
                for batch in parquet.iter_batches(columns=['episode_index'], batch_size=65536):
                    episode_ids.update(batch.column(0).unique().to_pylist())
                for episode in episode_ids:
                    by_episode.setdefault(episode, []).append(path)
        available = set().union(*schemas.values())
        order = 'frame_index' if 'frame_index' in available else 'index' if 'index' in available else None
        columns = ['episode_index', 'action', 'observation.anchor',
                   'complementary_info.segment_id', 'complementary_info.action_valid']
        if order:
            columns.append(order)
        for episode in sorted(by_episode):
            table = pd.concat([pq.read_table(path, columns=[c for c in columns if c in schemas[path]],
                                            filters=[('episode_index', '=', episode)]).to_pandas()
                               for path in by_episode[episode]], ignore_index=True)
            if order and order in table:
                table = table.sort_values(order, kind='stable')
            if 'complementary_info.segment_id' in table:
                table['_segment'] = np.stack(table['complementary_info.segment_id']).reshape(-1)
            else:
                table['_segment'] = 0
            for _, group in table.groupby('_segment', sort=False):
                actions = np.stack(group['action'])
                anchors = np.stack(group['observation.anchor'])
                valid = (np.stack(group['complementary_info.action_valid']).reshape(-1).astype(bool)
                         if 'complementary_info.action_valid' in group else np.ones(len(group), bool))
                yield actions, anchors, valid
    return compute_statistics(episodes(), spec, sampling, dataset_fingerprint(root),
                              temporary_directory=temporary_directory)
