"""Persistent measured TCP poses. No device, model or ROS imports."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
import uuid


def note_text(note):
    if not isinstance(note, str) or not note.strip() or len(note) > 200:
        raise ValueError('请填写 1–200 字的姿态备注')
    return note.strip()


def validate_pose(pose):
    """Validate an exact schema, so a record can never contain a gripper target."""
    if not isinstance(pose, dict) or set(pose) != {'arm', 'device', 'frame', 'position', 'rotation', 'tool'}:
        raise ValueError('invalid pose fields')
    if pose['arm'] not in ('left', 'right'):
        raise ValueError('invalid arm')
    for field in ('device', 'frame'):
        if not isinstance(pose[field], str) or not 0 < len(pose[field]) <= 200:
            raise ValueError('invalid pose identity')
    def vector(value, n):
        if not isinstance(value, list) or len(value) != n or any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
            raise ValueError('invalid pose values')
    def rotation(value):
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError('invalid rotation')
        for row in value:
            vector(row, 3)
        for i in range(3):
            for j in range(3):
                if abs(sum(value[k][i]*value[k][j] for k in range(3)) - (i == j)) > 1e-5:
                    raise ValueError('invalid rotation')
        a,b,c = value
        det = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
        if abs(det-1) > 1e-5:
            raise ValueError('invalid rotation')
    vector(pose['position'], 3)
    rotation(pose['rotation'])
    tool = pose['tool']
    if not isinstance(tool, dict) or set(tool) != {'position', 'rotation'}:
        raise ValueError('invalid TCP definition')
    vector(tool['position'], 3)
    rotation(tool['rotation'])
    return pose


class PoseStore:
    def __init__(self, root, *, demo=False):
        self.path = Path(root)/'runtime/web'/('poses-demo.json' if demo else 'poses.json')

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _read(self):
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text())
        if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data.get('records'), list) or len(data['records']) > 500:
            raise ValueError('姿态库格式错误；请保留原文件并检查 ' + str(self.path))
        seen = set()
        for record in data['records']:
            uuid.UUID(record['id'])
            if record['id'] in seen:
                raise ValueError('duplicate pose record')
            seen.add(record['id'])
            note_text(record['note'])
            validate_pose(record['pose'])
        return data['records']

    def _write(self, records):
        name = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=self.path.parent, delete=False) as out:
                name = out.name
                json.dump({'version': 1, 'records': records}, out, ensure_ascii=False, allow_nan=False, indent=2)
                out.flush()
                os.fsync(out.fileno())
            os.replace(name, self.path)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)

    def list(self):
        with self.locked():
            return self._read()

    def get(self, record_id):
        return next((r for r in self.list() if r['id'] == record_id), None) or self._missing()

    @staticmethod
    def _missing():
        raise ValueError('姿态记录不存在或已被删除')

    def add(self, note, pose):
        record = {'id': str(uuid.uuid4()), 'note': note_text(note),
                  'created_at': datetime.now(timezone.utc).isoformat(), 'pose': validate_pose(pose)}
        with self.locked():
            records = self._read()
            if len(records) >= 500:
                raise ValueError('姿态库已达 500 条，请先删除不再使用的记录')
            self._write([record, *records])
        return record

    def edit(self, record_id, note=None):
        with self.locked():
            records = self._read()
            record = next((r for r in records if r['id'] == record_id), None)
            if record is None:
                self._missing()
            if note is None:
                records.remove(record)
            else:
                record['note'] = note_text(note)
            self._write(records)
        return {'ok': True}
