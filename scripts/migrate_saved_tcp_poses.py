#!/usr/bin/env python3
"""Explicit offline rebinding of already-logical saved poses to the current tool.

This is not a runtime compatibility path. It only accepts the known -45 degree
mount metadata (or an already migrated record), the same physical TCP origin,
and an explicitly named device, frame and set of records.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import uuid

from vla.web.poses import PoseStore, validate_pose


def migrate(root, *, metadata, device, arm, record_ids, expected_sha256):
    store = PoseStore(root)
    definition = json.loads(Path(metadata).read_text())['tool_definition'][arm]
    if definition['orientation_frame'] != 'flange_aligned_tcp_v1':
        raise ValueError('metadata must describe the current logical TCP frame')
    identity = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    tool = dict(position=definition['position'], rotation=identity)
    if definition['rotation'] != identity:
        raise ValueError('this migration requires the verified identity tool rotation')
    hardware = definition['hardware_flange_to_tcp_pose']
    if len(hardware) != 7 or any(abs(a-b) > 1e-7 for a,b in zip(hardware, [*tool['position'], 0, 0, 0, 1])):
        raise ValueError('hardware and logical tool definitions disagree')
    frame = definition['base']+'/TCP:flange_aligned_tcp_v1'
    old_mount = [[2**-.5, 2**-.5, 0.], [-2**-.5, 2**-.5, 0.], [0., 0., 1.]]
    selected = set(record_ids)
    if not selected or len(selected) != len(record_ids):
        raise ValueError('select distinct pose IDs explicitly')
    with store.locked():
        original = store.path.read_bytes()
        if hashlib.sha256(original).hexdigest() != expected_sha256:
            raise ValueError('pose file changed since review; refusing to overwrite')
        records = store._read()
        if not selected <= {r['id'] for r in records}:
            raise ValueError('a selected pose no longer exists')
        updated = deepcopy(records)
        changed = []
        for record in updated:
            if record['id'] not in selected:
                continue
            pose = record['pose']
            if (pose['arm'], pose['device'], pose['frame']) != (arm, device, frame):
                raise ValueError('saved pose device/arm/logical frame mismatch')
            if any(abs(a-b) > 1e-7 for a,b in zip(pose['tool']['position'], tool['position'])):
                raise ValueError('physical TCP origin changed; rebinding is not valid')
            rotation = pose['tool']['rotation']
            if rotation == identity:
                continue
            if any(abs(rotation[i][j]-old_mount[i][j]) > 1e-7 for i in range(3) for j in range(3)):
                raise ValueError('unexpected old tool rotation')
            pose['tool'] = deepcopy(tool)
            validate_pose(pose)
            changed.append(dict(id=record['id'], note=record['note']))
        for record in updated:
            validate_pose(record['pose'])
        if not changed:
            return dict(changed=[], backup=None, sha256=expected_sha256)
        # Respect cooperating Web writers and detect unexpected noncooperating edits.
        if store.path.read_bytes() != original:
            raise ValueError('pose file changed during migration')
        backup_dir = store.path.parent/'pose-backups'/uuid.uuid4().hex
        backup_dir.mkdir(parents=True, exist_ok=False)
        backup = backup_dir/'poses.json'
        with backup.open('xb') as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
        (backup_dir/'tool-metadata.json').write_text(json.dumps(definition, indent=2)+'\n')
        store._write(updated)
        result = dict(changed=changed, backup=str(backup),
                      sha256=hashlib.sha256(store.path.read_bytes()).hexdigest())
        (backup_dir/'migration.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--arm', choices=('left', 'right'), required=True)
    parser.add_argument('--pose-id', action='append', required=True)
    parser.add_argument('--expected-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.root, metadata=args.metadata, device=args.device, arm=args.arm,
                             record_ids=args.pose_id, expected_sha256=args.expected_sha256),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
