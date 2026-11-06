"""Offline processing drafts and detached jobs, independent of the hardware runner."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from .manager import process_identity
from .processes import clean_environment


def save_json(path, value):
    from vla.datacollection.franka_robotiq_single_left.v30_writer import atomic_json
    atomic_json(path, value)


@contextmanager
def operation_guard(root):
    directory = Path(root)/'runtime/web'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'manager.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def active_job(root):
    path = Path(root)/'runtime/web/processing/active.json'
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    directory = path.parent/'jobs'/record['job_id']
    status_path = directory/'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'state':'starting'}
    alive = process_identity(record['pid']) == record['identity']
    if not alive and status['state'] in ('starting','running'):
        # A process can fail before creating its status file. Never automatically restart it.
        status = {**status, 'state':'failed', 'error':'转换进程已退出，请查看任务日志后重新提交'}
    return {**record, **status, 'alive':alive}


def reject_processing_job(root):
    job = active_job(root)
    if job and job['alive']:
        raise RuntimeError('数据转换正在运行，请完成或取消转换后启动硬件会话')


def field_descriptor(name):
    arm = 'left' if name.startswith('left_') else 'right' if name.startswith('right_') else 'single'
    base = name.removeprefix('left_').removeprefix('right_')
    groups = [('external_joint_torque_', '外部关节力矩', 'N·m'), ('joint_torque_', '关节力矩', 'N·m'),
              ('joint_velocity_', '关节速度', 'rad/s'), ('joint_position_', '关节位置', 'rad'),
              ('base_tcp_angular_velocity_', 'TCP 角速度', 'rad/s'), ('base_tcp_linear_velocity_', 'TCP 线速度', 'm/s'),
              ('base_tcp_force_', 'TCP 外力', 'N'), ('base_tcp_torque_', 'TCP 外力矩', 'N·m'),
              ('tcp_position_', 'TCP 位置', 'm'), ('tcp_rot6d_', 'TCP 姿态', '无量纲'),
              ('gripper_position', '夹爪', 'm')]
    for prefix, group, unit in groups:
        if base.startswith(prefix):
            return {'name':name, 'arm':arm, 'group':group, 'unit':unit}
    return {'name':name, 'arm':arm, 'group':'其他', 'unit':'未声明'}


class ProcessingService:
    def __init__(self, manager):
        self.manager = manager
        self.root = manager.root
        self.directory = self.root/'runtime/web/processing'
        (self.directory/'drafts').mkdir(parents=True, exist_ok=True)
        (self.directory/'jobs').mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.preview_slots = threading.BoundedSemaphore(4)
        self.indices = OrderedDict()
        self.jpeg_cache = OrderedDict()
        self.checked_at = {}

    def path(self, value):
        if not isinstance(value,str) or not value.strip() or '\x00' in value or len(value)>4096:
            raise ValueError('需要有效数据集路径')
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.root/path).resolve()

    def hardware(self):
        # A demo Web server may share a root with an independently running real session.
        if not self.manager.demo:
            return self.manager.active()
        from .manager import Manager
        return Manager(self.root, demo=False).active()

    def available(self):
        results = []
        for stage in ('raw','rollouts'):
            base = self.root/'datasets'/stage
            # Do not descend into datasets, videos or arbitrary symlinked trees.
            for parent, dirs, files in os.walk(base, followlinks=False):
                path = Path(parent)
                if (path/'meta/info.json').is_file():
                    try:
                        info = json.loads((path/'meta/info.json').read_text())
                        self.ensure_idle_source(path.resolve())
                        if int(info.get('total_episodes', 0)) > 0:
                            results.append(str(path.relative_to(self.root)))
                    except (OSError, ValueError, TypeError):
                        pass
                    dirs[:] = []
                else:
                    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('videos','data','meta')]
                if len(results) >= 500:
                    return results
        return results

    def ensure_idle_source(self, path):
        record = self.hardware()
        if record and record['alive'] and not record.get('demo'):
            from .config import SessionConfig
            if SessionConfig.parse(record['config']).dataset_path(self.root) == path:
                raise ValueError('当前采集／部署数据集正在使用，请结束会话后打开')

    def _draft_path(self, dataset_id):
        if not isinstance(dataset_id,str) or len(dataset_id)!=32 or any(c not in '0123456789abcdef' for c in dataset_id):
            raise ValueError('无效数据集 ID')
        return self.directory/'drafts'/f'{dataset_id}.json'

    def draft(self, dataset_id):
        path = self._draft_path(dataset_id)
        if not path.is_file():
            raise ValueError('数据集尚未打开')
        return json.loads(path.read_text())

    def _index(self, dataset_id):
        from vla.datacollection.processing import DatasetIndex
        draft = self.draft(dataset_id)
        path = Path(draft['source_root'])
        self.ensure_idle_source(path)
        if dataset_id not in self.indices:
            self.indices[dataset_id] = DatasetIndex(path)
        index = self.indices[dataset_id]
        if time.monotonic() - self.checked_at.get(dataset_id, 0) >= 1:
            index.check()
            self.checked_at[dataset_id] = time.monotonic()
        if index.fingerprint != draft['source_fingerprint']:
            raise ValueError('源数据已变化，请重新加载并确认选择')
        self.indices.move_to_end(dataset_id)
        while len(self.indices)>4:
            self.indices.popitem(last=False)
        return index, draft

    def open(self, value):
        from vla.datacollection.processing import DatasetIndex, read_lease
        path = self.path(value)
        self.ensure_idle_source(path)
        if not path.is_dir():
            raise ValueError('数据集目录不存在')
        with self.lock, read_lease(path):
            index = DatasetIndex(path)
            dataset_id = hashlib.sha256(str(path).encode()).hexdigest()[:32]
            draft_path = self._draft_path(dataset_id)
            draft = json.loads(draft_path.read_text()) if draft_path.exists() else None
            changed = draft is not None and draft['source_fingerprint'] != index.fingerprint
            if draft is None or changed:
                draft = {'version':1, 'dataset_id':dataset_id, 'source_root':str(path),
                         'source_fingerprint':index.fingerprint, 'state_fields':index.names,
                         'episodes':list(range(len(index.rows))), 'revision':0,
                         'output_root':f'datasets/sft/{path.name}_{len(index.names)}d'}
                save_json(draft_path, draft)
            self.indices[dataset_id] = index
            while len(self.indices)>4:
                self.indices.popitem(last=False)
            self.jpeg_cache.clear()
            save_json(self.directory/'last.json', {'dataset_id':dataset_id})
            return self._summary(index,draft,changed)

    def _summary(self,index,draft,changed=False):
        return {'draft':draft, 'source_changed':changed, 'total_episodes':len(index.rows),
                'total_frames':index.info['total_frames'], 'fps':index.fps,
                'action_spec':index.info.get('action_spec'), 'robot_type':index.info.get('robot_type'), 'cameras':index.cameras,
                'fields':[field_descriptor(n) for n in index.names],
                'action_names':index.features['action'].get('names') or [],
                'action_units':[('0–1' if n.endswith('gripper') else 'm' if 'tcp_position_' in n
                                 else '无量纲' if 'rot6d_' in n else '未声明')
                                for n in (index.features['action'].get('names') or [])],
                'presets':self._presets(index.names)}

    @staticmethod
    def _presets(names):
        from vla.datacollection.franka_robotiq_single_left.schema import (
            TCP_STATE_FIELDS, LEGACY_STATE_FIELDS, JOINT_TORQUE_FIELDS,
            EXTERNAL_JOINT_TORQUE_FIELDS, JOINT_VELOCITY_FIELDS,
        )
        choices = {'全部':names, 'TCP 10D':TCP_STATE_FIELDS, 'TCP＋力与速度 22D':LEGACY_STATE_FIELDS[:22],
                   '原标准 29D':LEGACY_STATE_FIELDS,
                   '力矩与关节速度 31D':TCP_STATE_FIELDS+JOINT_TORQUE_FIELDS+EXTERNAL_JOINT_TORQUE_FIELDS+JOINT_VELOCITY_FIELDS,
                   '双臂 TCP 20D':[f'{arm}_{n}' for arm in ('left','right') for n in TCP_STATE_FIELDS]}
        return {k:v for k,v in choices.items() if set(v).issubset(names)}

    def summary(self, dataset_id):
        with self.lock:
            index,draft = self._index(dataset_id)
            return self._summary(index,draft)

    def last(self):
        path = self.directory/'last.json'
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        return {**record, 'source_root':self.draft(record['dataset_id'])['source_root']}

    def update(self, dataset_id, value):
        from vla.datacollection.processing import selection
        with self.lock:
            index,draft = self._index(dataset_id)
            if not isinstance(value,dict) or set(value)-{'revision','state_fields','episodes','output_root','action_space','allow_derived','preview_representation','preview_anchor_frame'}:
                raise ValueError('无效处理草稿')
            if type(value.get('revision')) is not int or value['revision'] != draft['revision']:
                raise RuntimeError('另一页面已修改草稿，请重新加载')
            candidate = {**draft, **value}
            # Drafts may temporarily select nothing; exports may not.
            fields,eps = candidate['state_fields'],candidate['episodes']
            selection(index,{**candidate,'state_fields':fields or [index.names[0]],'episodes':eps or [0]})
            if not isinstance(fields,list) or not isinstance(eps,list):
                raise ValueError('选择必须为列表')
            if candidate.get('action_space') not in (None, 'joint', 'tcp'):
                raise ValueError('invalid export action space')
            if type(candidate.get('allow_derived', False)) is not bool:
                raise ValueError('derived data consent must be explicit')
            if candidate.get('preview_representation', 'absolute') not in ('absolute', 'chunk_relative'):
                raise ValueError('invalid preview representation')
            if type(candidate.get('preview_anchor_frame', 0)) is not int or candidate.get('preview_anchor_frame', 0) < 0:
                raise ValueError('invalid preview chunk start')
            self.path(candidate['output_root'])
            candidate.update(state_fields=[n for n in index.names if n in fields],episodes=sorted(eps),revision=draft['revision']+1)
            save_json(self._draft_path(dataset_id),candidate)
            return candidate

    def episodes(self, dataset_id, offset=0, limit=25):
        from vla.datacollection.processing import scalar
        if offset<0 or not 1<=limit<=100:
            raise ValueError('分页范围无效')
        with self.lock:
            index,draft = self._index(dataset_id)
            rows = []
            for ep in range(offset,min(offset+limit,len(index.rows))):
                row = index.rows.iloc[ep]
                rows.append({'episode':ep,'length':int(row['length']),'duration':int(row['length'])/index.fps,
                             'tasks':scalar(row.get('tasks',[])), 'outcome':scalar(row.get('episode_success')),
                             'kept':ep in draft['episodes']})
            return {'episodes':rows,'total':len(index.rows)}

    @contextmanager
    def preview_slot(self):
        if not self.preview_slots.acquire(blocking=False):
            raise RuntimeError('预览忙，请稍后重试')
        try:
            yield
        finally:
            self.preview_slots.release()

    def frame(self,dataset_id,episode,frame):
        from vla.datacollection.processing import read_lease
        with self.preview_slot(), self.lock:
            index,draft = self._index(dataset_id)
            with read_lease(index.root):
                row = index.frame(episode,frame)
                if draft.get('preview_representation') == 'chunk_relative':
                    import numpy as np
                    from dataclasses import replace
                    from vla.actions import ActionSpec, DeltaActions
                    spec = replace(ActionSpec.from_dict(index.info['action_spec']), representation='chunk_relative')
                    anchor_frame = draft.get('preview_anchor_frame', 0)
                    if frame < anchor_frame:
                        raise ValueError('预览帧不能早于 chunk 起点')
                    anchor = index.frame(episode,anchor_frame).get('observation.anchor')
                    if anchor is None:
                        raise ValueError('缺少独立实测锚点，无法预览相对动作')
                    row['preview.chunk_relative_action'] = DeltaActions(spec)(np.asarray(row['action']), np.asarray(anchor)).tolist()
                    row['preview.anchor_frame'] = anchor_frame
                return row

    def window(self,dataset_id,episode,frame,seconds=10):
        from vla.datacollection.processing import read_lease
        with self.preview_slot(), self.lock:
            index,_ = self._index(dataset_id)
            with read_lease(index.root):
                return index.window(episode,frame,seconds)

    def jpeg(self,dataset_id,episode,frame,camera):
        from vla.datacollection.processing import read_lease
        with self.preview_slot(), self.lock:
            index,_ = self._index(dataset_id)
            key = (dataset_id,index.fingerprint,episode,frame,camera)
            with read_lease(index.root):
                if key not in self.jpeg_cache:
                    self.jpeg_cache[key] = index.jpeg(episode,frame,camera)
                self.jpeg_cache.move_to_end(key)
                while len(self.jpeg_cache)>24:
                    self.jpeg_cache.popitem(last=False)
                return self.jpeg_cache[key]

    def status(self):
        return active_job(self.root)

    def start(self,dataset_id,revision):
        from vla.datacollection.processing import selection
        from vla.cli.extract_dataset_features import _validate_paths
        with self.lock, operation_guard(self.root):
            record = self.hardware()
            if record and record['alive'] and not record.get('demo'):
                raise RuntimeError('请停止采集／部署会话后再转换')
            reject_processing_job(self.root)
            index,draft = self._index(dataset_id)
            if type(revision) is not int or revision != draft['revision']:
                raise RuntimeError('草稿已变化，请重新加载')
            selection(index,draft)
            output = self.path(draft['output_root'])
            _validate_paths(index.root,output)
            job_id = uuid.uuid4().hex
            directory = self.directory/'jobs'/job_id
            directory.mkdir()
            save_json(directory/'selection.json',draft)
            save_json(directory/'status.json',{'state':'starting','stage':'等待预检'})
            argv = [sys.executable,'-m','vla.web.processing_worker','--root',str(self.root),
                    '--job',job_id,'--output',str(output)]
            env = clean_environment(self.root,sys.executable)
            env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            with (directory/'worker.log').open('ab') as log:
                process = subprocess.Popen(argv,env=env,cwd=self.root,stdin=subprocess.DEVNULL,
                                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            identity = process_identity(process.pid)
            if identity is None:
                raise RuntimeError('转换进程启动失败')
            record = {'job_id':job_id,'pid':process.pid,'identity':identity,'dataset_id':dataset_id,
                      'output_root':str(output),'started_at':time.time()}
            save_json(self.directory/'active.json',record)
            threading.Thread(target=process.wait,daemon=True).start()
            return record

    def cancel(self,job_id):
        with operation_guard(self.root):
            record = self.status()
            if not record or job_id != record['job_id']:
                raise ValueError('任务不匹配')
            if record['alive']:
                (self.directory/'jobs'/job_id/'cancel').touch()
            return self.status()
