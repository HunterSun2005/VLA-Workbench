"""Read-only VLA dataset browsing and immutable whole-episode exports (no Torch)."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from vla.dataset_metadata import load_info, write_metadata_sidecar
from vla.core.writer_lock import WriterLease
from vla.datacollection.franka_robotiq_single_left.v30_writer import (
    DATA_PATH, VIDEO_PATH, EPISODES_PATH, atomic_json, feature_stats,
    aggregate_feature_stats, _numeric_array, _jsonable, _fixed_array,
)


@contextmanager
def read_lease(root):
    """Shared with cooperating collectors' WriterLease; never modifies dataset contents."""
    root = Path(root).resolve()
    with root.with_name(f'.{root.name}.rlt-writer.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('数据集正在写入，请结束采集后再打开') from None
        yield


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'数据集引用文件缺失或越界: {relative}')
    return path


def scalar(value):
    if isinstance(value, dict):
        return {str(k): scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [scalar(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class DatasetIndex:
    """Only metadata is retained; parquet/video reads are per requested episode/frame."""
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.info = load_info(self.root)
        if self.info.get('codebase_version') != 'v3.0':
            raise ValueError('仅支持 LeRobot v3 数据集')
        self.features = self.info['features']
        self.names = self.features['observation.state'].get('names')
        if (not isinstance(self.names, list) or not self.names
                or any(not isinstance(n, str) or not n for n in self.names)
                or len(set(self.names)) != len(self.names)
                or self.features['observation.state']['shape'] != [len(self.names)]):
            raise ValueError('状态需要唯一的字段名称和一致的一维 shape')
        self.cameras = [k for k, v in self.features.items() if v.get('dtype') == 'video']
        if not self.cameras or any(v.get('dtype') == 'image' for v in self.features.values()):
            raise ValueError('首版仅支持 VLA 的独立视频回合格式')
        self.fps = float(self.info['fps'])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('invalid FPS')
        paths = sorted((self.root/'meta/episodes').glob('*/*.parquet'))
        if not paths:
            raise ValueError('没有已保存回合')
        self.rows = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True).sort_values('episode_index').reset_index(drop=True)
        if self.rows['episode_index'].tolist() != list(range(len(self.rows))):
            raise ValueError('episode_index 必须从零连续编号')
        if (len(self.rows) != self.info['total_episodes'] or not len(self.rows)
                or int(self.rows['length'].sum()) != self.info['total_frames']):
            raise ValueError('回合元数据与总计不一致')
        self.files = set(paths)
        self.files.update(p for p in (self.root/'meta').glob('*.json'))
        self.files.add(inside(self.root, 'meta/tasks.parquet'))
        used = set()
        for ep, row in self.rows.iterrows():
            if int(row['length']) <= 0:
                raise ValueError('空回合不受支持')
            for camera in [None, *self.cameras]:
                path = self.file(ep, camera)
                if path in used:
                    raise ValueError('仅支持每回合独立 parquet／视频文件；不支持共享文件布局')
                used.add(path)
                if camera and abs(float(row[f'videos/{camera}/from_timestamp'])) > 1e-6:
                    raise ValueError('仅支持每回合从零开始的视频')
            trace = row.get('command_trace')
            if isinstance(trace, str) and trace:
                self.files.add(inside(self.root, trace))
        self.files.update(used)
        self.fingerprint = self.signature()

    def file(self, episode, camera=None):
        if type(episode) is not int or not 0 <= episode < len(self.rows):
            raise ValueError('回合编号越界')
        row = self.rows.iloc[episode]
        prefix = f'videos/{camera}' if camera else 'data'
        if camera is not None and camera not in self.cameras:
            raise ValueError('未知相机')
        template = self.info['video_path' if camera else 'data_path']
        return inside(self.root, template.format(video_key=camera,
            chunk_index=int(row[f'{prefix}/chunk_index']), file_index=int(row[f'{prefix}/file_index'])))

    def signature(self):
        entries = []
        for p in sorted(self.files):
            st = p.stat()
            digest = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_relative_to(self.root/'meta') else None
            entries.append((str(p.relative_to(self.root)), st.st_size, st.st_mtime_ns, digest))
        return hashlib.sha256(json.dumps(entries).encode()).hexdigest()

    def check(self):
        if self.signature() != self.fingerprint:
            raise ValueError('源数据已变化，请重新加载并确认选择')

    def frame(self, episode, frame):
        length = int(self.rows.iloc[episode]['length']) if type(episode) is int and 0 <= episode < len(self.rows) else 0
        if type(frame) is not int or not 0 <= frame < length:
            raise ValueError('帧编号越界')
        parquet = pq.ParquetFile(self.file(episode))
        if parquet.metadata.num_rows != length:
            raise ValueError('parquet 帧数与回合元数据不一致')
        offset = frame
        for group in range(parquet.num_row_groups):
            size = parquet.metadata.row_group(group).num_rows
            if offset < size:
                row = parquet.read_row_group(group).slice(offset, 1).to_pylist()[0]
                if row['episode_index'] != episode or row['frame_index'] != frame:
                    raise ValueError('帧索引与元数据不一致')
                return scalar(row)
            offset -= size
        raise ValueError('找不到帧')

    def jpeg(self, episode, frame, camera):
        import av
        import cv2
        row = self.frame(episode, frame)
        target = float(row['timestamp'])
        with av.open(str(self.file(episode, camera))) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            container.seek(max(0, int(target / stream.time_base)), stream=stream, backward=True)
            for decoded in container.decode(stream):
                if decoded.time is None:
                    continue
                delta = decoded.time - target
                if abs(delta) <= .51 / self.fps:
                    image = decoded.to_ndarray(format='bgr24')
                    h, w = image.shape[:2]
                    scale = min(1., 640/w, 480/h)
                    if scale < 1:
                        image = cv2.resize(image, (round(w*scale), round(h*scale)))
                    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ok:
                        raise ValueError('JPEG 编码失败')
                    return encoded.tobytes()
                if delta > .51 / self.fps:
                    break
        raise ValueError(f'相机 {camera} 缺少时间 {target:.3f}s 对应帧')


def selection(index, value):
    if not isinstance(value, dict) or value.get('version') != 1:
        raise ValueError('需要 version=1 的处理选择文件')
    if value.get('source_fingerprint') != index.fingerprint:
        raise ValueError('源数据已变化，请重新加载并确认选择')
    fields, episodes = value.get('state_fields'), value.get('episodes')
    if (not isinstance(fields, list) or not fields or any(not isinstance(n, str) for n in fields)
            or len(set(fields)) != len(fields) or not set(fields).issubset(index.names)):
        raise ValueError('至少选择一个有效且不重复的状态字段')
    if (not isinstance(episodes, list) or not episodes or any(type(e) is not int for e in episodes)
            or len(set(episodes)) != len(episodes) or not set(episodes).issubset(range(len(index.rows)))):
        raise ValueError('至少保留一个有效且不重复的回合')
    return [n for n in index.names if n in fields], sorted(episodes)


def _exact_stats(values, count, checkpoint=lambda: None):
    # Quantiles need raw values: process one column at a time, not a second full matrix.
    parts = []
    for i in range(values.shape[1]):
        checkpoint()
        parts.append(feature_stats(np.asarray(values[:count, i])))
    return {k: parts[0][k] if k == 'count' else np.concatenate([p[k] for p in parts]) for k in parts[0]}


def export_selection(source, output, chosen, *, progress=None, cancelled=None, validate=True):
    """Write selected complete episodes and named state columns, never edit the source."""
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    from vla.cli.extract_dataset_features import _validate_paths
    _validate_paths(source, output)
    progress = progress or (lambda *_: None)
    def checkpoint(stage, done=0, total=0):
        if cancelled and cancelled():
            raise InterruptedError('转换已取消')
        progress(stage, done, total)
    with read_lease(source), WriterLease(output):
        index = DatasetIndex(source)
        names, episodes = selection(index, chosen)
        if index.info.get('evo_rlt', {}).get('actions_stored_as') != 'absolute':
            raise ValueError('转换需要明确标记为 absolute 的原始动作数据')
        columns = [index.names.index(n) for n in names]
        output.parent.mkdir(parents=True, exist_ok=True)
        parent = Path(tempfile.mkdtemp(prefix=f'.{output.name}.processing-', dir=output.parent))
        staging = parent/'dataset'
        staging.mkdir()
        try:
            info = deepcopy(index.info)
            info['features']['observation.state'].update(shape=[len(names)], names=names)
            action_dim = info['features']['action']['shape'][0]
            total = sum(int(index.rows.iloc[e]['length']) for e in episodes)
            states = np.memmap(parent/'states.bin', mode='w+', dtype='float32', shape=(total, len(names)))
            actions = np.memmap(parent/'actions.bin', mode='w+', dtype='float32', shape=(total, action_dim))
            rows, offset, action_count, mapping, provenance = [], 0, 0, [], []
            image_stats = {camera: [] for camera in index.cameras}
            tasks = pd.read_parquet(source/'meta/tasks.parquet')
            task_ids = {}
            for new_ep, old_ep in enumerate(episodes):
                checkpoint('转换回合', new_ep, len(episodes))
                table = pd.read_parquet(index.file(old_ep))
                length = len(table)
                if (length != int(index.rows.iloc[old_ep]['length'])
                        or not np.array_equal(table['frame_index'], np.arange(length))
                        or not (table['episode_index'] == old_ep).all()):
                    raise ValueError(f'回合 {old_ep} 帧数或索引不一致')
                raw = np.stack(table['observation.state'].to_numpy()).reshape(length, -1)
                action = np.stack(table['action'].to_numpy()).reshape(length, -1)
                if raw.shape != (length, len(index.names)) or action.shape != (length, action_dim):
                    raise ValueError('状态／动作维度与字段元数据不符')
                selected = raw[:, columns].astype('float32')
                if not np.isfinite(selected).all() or not np.isfinite(action).all():
                    raise ValueError('状态／动作包含非有限值')
                valid = (np.stack(table['complementary_info.action_valid'].to_numpy()).reshape(-1).astype(bool)
                         if 'complementary_info.action_valid' in table else np.ones(length, dtype=bool))
                if not valid.any():
                    raise ValueError(f'回合 {old_ep} 没有有效动作')
                table['observation.state'] = list(selected)
                table['episode_index'] = np.int64(new_ep)
                table['index'] = np.arange(offset, offset+length, dtype='int64')
                for value in table['task_index'].unique():
                    old_id = int(value)
                    if old_id not in task_ids:
                        task_ids[old_id] = len(task_ids)
                table['task_index'] = table['task_index'].map(task_ids).astype('int64')
                chunk, file = divmod(new_ep, int(info['chunks_size']))
                data_path = staging/DATA_PATH.format(chunk_index=chunk, file_index=file)
                data_path.parent.mkdir(parents=True, exist_ok=True)
                arrow = pa.Table.from_pandas(table, preserve_index=False)
                state_pos = arrow.column_names.index('observation.state')
                arrow = arrow.set_column(state_pos, 'observation.state', _fixed_array(selected, 'float32'))
                pq.write_table(arrow, data_path)
                row = index.rows.iloc[old_ep].to_dict()
                original = {'source_episode': old_ep}
                for key in ('online_files', 'run_uuid', 'dataset_uuid', 'episode_uuid'):
                    if key in row:
                        original[key] = scalar(row.pop(key))
                provenance.append(original)
                row.update(episode_index=new_ep, dataset_from_index=offset, dataset_to_index=offset+length,
                           **{'data/chunk_index':chunk, 'data/file_index':file})
                for camera in index.cameras:
                    checkpoint('复制视频', new_ep, len(episodes))
                    src = index.file(old_ep, camera)
                    dest = staging/VIDEO_PATH.format(video_key=camera, chunk_index=chunk, file_index=file)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    # A real copy ensures later independent edits cannot mutate raw assets.
                    with src.open('rb') as inp, dest.open('wb') as out:
                        while block := inp.read(4*1024*1024):
                            checkpoint('复制视频', new_ep, len(episodes))
                            out.write(block)
                    row[f'videos/{camera}/chunk_index'], row[f'videos/{camera}/file_index'] = chunk, file
                    prefix = f'stats/{camera}/'
                    stats = {k[len(prefix):]: _numeric_array(v) for k, v in row.items() if k.startswith(prefix)}
                    if not {'min','max','mean','std','count','q01','q10','q50','q90','q99'}.issubset(stats):
                        raise ValueError(f'{camera} 缺少回合图像统计')
                    image_stats[camera].append(stats)
                trace = row.get('command_trace')
                if isinstance(trace, str) and trace:
                    target = Path('traces')/f'episode-{new_ep:06d}.jsonl'
                    (staging/target).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(inside(source, trace), staging/target)
                    row['command_trace'] = str(target)
                for key, values in [('observation.state', selected), ('action', action[valid])]:
                    row.update({f'stats/{key}/{k}': v.tolist() for k, v in feature_stats(values).items()})
                states[offset:offset+length] = selected
                actions[action_count:action_count+int(valid.sum())] = action[valid]
                action_count += int(valid.sum())
                rows.append(row)
                mapping.append({'source_episode': old_ep, 'output_episode':new_ep})
                offset += length
            checkpoint('重算统计', len(episodes), len(episodes))
            stats_checkpoint = lambda: checkpoint('重算统计', len(episodes), len(episodes))
            stats = {'observation.state':_exact_stats(states,total,stats_checkpoint),
                     'action':_exact_stats(actions,action_count,stats_checkpoint)}
            stats.update({camera:aggregate_feature_stats(values) for camera, values in image_stats.items()})
            del states, actions
            # Reindex referenced task IDs while preserving their text.
            tasks = tasks[tasks['task_index'].isin(task_ids)]
            if set(int(v) for v in tasks['task_index']) != set(task_ids):
                raise ValueError('任务索引缺少对应文本')
            tasks['task_index'] = tasks['task_index'].map(task_ids).astype('int64')
            tasks = tasks.sort_values('task_index')
            ep_path = staging/EPISODES_PATH
            ep_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(ep_path, index=False)
            tasks.to_parquet(staging/'meta/tasks.parquet')
            info.update(total_episodes=len(episodes), total_frames=total, total_tasks=len(tasks),
                        splits={'train':f'0:{len(episodes)}'}, data_path=DATA_PATH, video_path=VIDEO_PATH)
            previous_processing = info.get('processing', {})
            source_contract = deepcopy(previous_processing.get('source_contract', {}))
            source_contract.update({k:info.pop(k) for k in ('recording_contract','run_uuid','dataset_uuid','online_run','online_contract',
                                   'online_rlt','online_replay_root','online_warmup_steps') if k in info})
            info.setdefault('evo_rlt', {}).update(action_stats='absolute', actions_stored_as='absolute', action_chunk_size=50)
            info['processing'] = {'version':1, 'source_root':str(source), 'source_fingerprint':index.fingerprint,
                'state_fields':names, 'kept_episodes':episodes,
                'excluded_episodes':sorted(set(range(len(index.rows)))-set(episodes)),
                'episode_mapping':mapping, 'task_mapping':task_ids, 'source_contract':source_contract, 'source_episode_metadata':provenance}
            atomic_json(staging/'meta/info.json', _jsonable(info))
            write_metadata_sidecar(staging, _jsonable(info))
            atomic_json(staging/'meta/stats.json', _jsonable(stats))
            atomic_json(staging/'meta/processing_selection.json', chosen)
            checkpoint('校验输出', len(episodes), len(episodes))
            if validate:
                _validate_export(staging, lambda: checkpoint('校验输出', len(episodes), len(episodes)))
            index.check()
            checkpoint('发布输出', len(episodes), len(episodes))
            if output.exists():
                raise FileExistsError('输出目录已存在')
            os.replace(staging, output)
            return output
        finally:
            shutil.rmtree(parent, ignore_errors=True)


def _validate_export(root, checkpoint=lambda: None):
    import av
    index = DatasetIndex(root)
    for ep in range(len(index.rows)):
        length = int(index.rows.iloc[ep]['length'])
        timestamps = pd.read_parquet(index.file(ep), columns=['timestamp'])['timestamp'].to_numpy()
        if len(timestamps) != length or not np.isfinite(timestamps).all() or (np.diff(timestamps) <= 0).any():
            raise ValueError('输出时间戳或帧数不一致')
        for camera in index.cameras:
            with av.open(str(index.file(ep,camera))) as container:
                stream = container.streams.video[0]
                stream.thread_count = 1
                count = 0
                for count, frame in enumerate(container.decode(stream), 1):
                    checkpoint()
                    if list((frame.height, frame.width, 3)) != index.features[camera]['shape']:
                        raise ValueError('视频尺寸与元数据不一致')
                    if count > length or frame.time is None or abs(frame.time-timestamps[count-1]) > .51/index.fps:
                        raise ValueError('视频与数据帧不同步')
                if count != length:
                    raise ValueError('视频帧数与数据帧数不一致')
