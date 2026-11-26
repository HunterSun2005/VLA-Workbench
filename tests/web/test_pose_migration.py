from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from scripts.migrate_saved_tcp_poses import migrate
from vla.web.poses import PoseStore
from vla.web.pose_control import compatible, pose_value


@pytest.fixture
def migration(tmp_path):
    store = PoseStore(tmp_path)
    mount = np.array([[2**-.5, 2**-.5, 0], [-2**-.5, 2**-.5, 0], [0, 0, 1]])
    for name, x in [('抓放红色物块初始位置', .61), ('插头上方初始位置', .56)]:
        store.add(name, pose_value('left', 'robot', 'base/TCP:flange_aligned_tcp_v1',
                                  [x, -.2, .15], np.diag([1, -1, -1]), [0, 0, .174], mount))
    metadata = tmp_path/'metadata.json'
    definition = dict(position=[0, 0, .174], rotation=np.eye(3).tolist(),
                      hardware_flange_to_tcp_pose=[0, 0, .174, 0, 0, 0, 1],
                      orientation_frame='flange_aligned_tcp_v1', base='base')
    metadata.write_text(json.dumps({'tool_definition': {'left': definition}}))
    kwargs = dict(metadata=metadata, device='robot', arm='left',
                  record_ids=[r['id'] for r in store.list()],
                  expected_sha256=hashlib.sha256(store.path.read_bytes()).hexdigest())
    return store, kwargs, definition


def test_migration_preserves_logical_pose_and_names_without_double_rotation(migration):
    store, kwargs, _ = migration
    before, original = store.list(), store.path.read_bytes()
    result = migrate(store.path.parents[2], **kwargs)
    assert len(result['changed']) == 2
    from pathlib import Path
    assert Path(result['backup']).read_bytes() == original
    after = store.list()
    for old, new in zip(before, after):
        expected = deepcopy(old)
        expected['pose']['tool']['rotation'] = np.eye(3).tolist()
        assert new == expected
        current = deepcopy(new['pose'])
        compatible(new['pose'], current)
        with pytest.raises(ValueError):
            compatible(old['pose'], current)
    stable = store.path.read_bytes()
    again = migrate(store.path.parents[2], **dict(kwargs, expected_sha256=result['sha256']))
    assert again['changed'] == [] and again['backup'] is None
    assert store.path.read_bytes() == stable


@pytest.mark.parametrize('problem', ['hash', 'device', 'frame', 'origin', 'rotation', 'missing_id'])
def test_migration_refuses_unreviewed_or_incompatible_records(migration, problem):
    store, kwargs, _ = migration
    records = store.list()
    if problem == 'hash':
        kwargs['expected_sha256'] = '0'*64
    elif problem == 'device':
        kwargs['device'] = 'another-robot'
    elif problem == 'missing_id':
        kwargs['record_ids'].append('deleted-record')
    else:
        pose = records[0]['pose']
        if problem == 'frame':
            pose['frame'] = 'base/EE'
        elif problem == 'origin':
            pose['tool']['position'][2] = .2
        else:
            pose['tool']['rotation'] = np.diag([-1, -1, 1]).tolist()
        with store.locked():
            store._write(records)
        kwargs['expected_sha256'] = hashlib.sha256(store.path.read_bytes()).hexdigest()
    original = store.path.read_bytes()
    with pytest.raises(ValueError):
        migrate(store.path.parents[2], **kwargs)
    assert store.path.read_bytes() == original


def test_failed_write_preserves_original_and_backup(migration, monkeypatch):
    store, kwargs, _ = migration
    original = store.path.read_bytes()
    monkeypatch.setattr('vla.web.poses.os.replace', lambda *_: (_ for _ in ()).throw(OSError('disk failure')))
    with pytest.raises(OSError):
        migrate(store.path.parents[2], **kwargs)
    assert store.path.read_bytes() == original
    backups = list((store.path.parent/'pose-backups').glob('*/poses.json'))
    assert len(backups) == 1 and backups[0].read_bytes() == original
