"""Bounded nonblocking collection with independently committed one-second segments.

Only the worker encodes or writes. A segment manifest is published after every
file is durable; episode publication is a separate commit. Failed tails remain
in staging and are never promoted to training samples by recovery.
"""
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
import json
import os
import queue
import shutil
import subprocess
import threading
import numpy as np
from vla.core.online_contract import durable_json, file_digest, fsync_directory


class StorageBlocked(RuntimeError):
    pass


def freeze(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('object arrays are not journal records')
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, Mapping):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value


def jsonable(value):
    if is_dataclass(value):
        return {f.name: jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def video_frames(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                             '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(path)],
                            check=True, capture_output=True, text=True)
    return int(json.loads(result.stdout)['streams'][0]['nb_read_frames'])


@dataclass(frozen=True)
class Recovery:
    frames: int
    segments: tuple
    complete: bool
    incomplete_tail: bool


def recover_staging(path):
    path = Path(path)
    header = json.loads((path/'episode.json').read_text())
    if header.get('schema_version') != 5:
        raise ValueError('unsupported episode journal schema')
    segments, frames = [], 0
    for manifest in sorted(path.glob('segment-[0-9][0-9][0-9][0-9][0-9][0-9].json')):
        value = json.loads(manifest.read_text())
        if value['index'] != len(segments) or value['start_frame'] != frames:
            break
        if any(not (path/name).is_file() or file_digest(path/name) != digest
               for name, digest in value['files'].items()):
            break
        with np.load(path/value['arrays'], allow_pickle=False) as data:
            if any(len(data[key]) != value['frames'] for key in data.files):
                break
        if any(video_frames(path/name) != value['frames'] for name in value['videos'].values()):
            break
        segments.append(value)
        frames += value['frames']
    complete = False
    if (path/'complete.json').is_file():
        completion = json.loads((path/'complete.json').read_text())
        complete = (completion['frames'] == frames and completion['segments'] == len(segments))
    return Recovery(frames, tuple(segments), complete, not complete)


class EpisodeJournal:
    def __init__(self, path, *, identity, fps=30, capacity=90):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError('staging episode already exists; recover it explicitly')
        self.path.mkdir(parents=True)
        self.fps = fps
        durable_json(self.path/'episode.json', {'schema_version': 5, 'fps': fps, **identity})
        self.queue = queue.Queue(capacity)
        self.error = None
        self.closed = False
        self.frames = []
        self.events = []
        self.count = self.segment = 0
        self.thread = threading.Thread(target=self._worker, name='episode-journal', daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise StorageBlocked(f'episode persistence blocked: {self.error}') from self.error

    def submit(self, columns, images):
        self._put(('frame', freeze(columns), freeze(images)))

    def event(self, event):
        self._put(('event', freeze(event)))

    def _put(self, record):
        self.check()
        if self.closed:
            raise StorageBlocked('episode journal is closed')
        try:
            self.queue.put_nowait(record)
        except queue.Full as error:
            # The caller must stop control and retain its current record for retry.
            raise StorageBlocked('episode persistence queue overflow') from error

    def _worker(self):
        try:
            while True:
                record = self.queue.get()
                if record is None:
                    self._flush()
                    durable_json(self.path/'complete.json', {'frames': self.count, 'segments': self.segment})
                    return
                if record[0] == 'event':
                    self.events.append(record[1])
                else:
                    self.frames.append(record[1:])
                    if len(self.frames) == self.fps:
                        self._flush()
        except BaseException as error:
            self.error = error

    def _flush(self):
        if not self.frames:
            if self.events:
                durable_json(self.path/f'events-tail-{self.segment:06d}.json',
                             {'schema_version': 5, 'events': jsonable(self.events)})
                self.events.clear()
            return
        stem = f'segment-{self.segment:06d}'
        arrays_path = self.path/f'{stem}.npz'
        columns = {key: np.stack([frame[0][key] for frame in self.frames]) for key in self.frames[0][0]}
        with arrays_path.open('wb') as stream:
            np.savez(stream, **columns)
            stream.flush()
            os.fsync(stream.fileno())
        videos = {}
        from vla.datacollection.franka_robotiq_rollout.common import FfmpegVideoWriter
        for key in self.frames[0][1]:
            frames = [frame[1][key] for frame in self.frames]
            name = f'{stem}-{key}.mp4'
            writer = FfmpegVideoWriter(self.path/name, width=frames[0].shape[1],
                                       height=frames[0].shape[0], fps=self.fps)
            try:
                for image in frames:
                    writer.write(image)
                writer.close()
            except BaseException:
                writer.abort()
                raise
            if video_frames(self.path/name) != len(self.frames):
                raise StorageBlocked('encoded frame count mismatch')
            with (self.path/name).open('rb') as stream:
                os.fsync(stream.fileno())
            videos[key] = name
        event_path = self.path/f'{stem}-events.json'
        durable_json(event_path, {'schema_version': 5, 'events': jsonable(self.events)})
        paths = [arrays_path, event_path, *(self.path/name for name in videos.values())]
        manifest = {'schema_version': 5, 'index': self.segment, 'start_frame': self.count,
                    'frames': len(self.frames), 'arrays': arrays_path.name, 'videos': videos,
                    'events': event_path.name, 'files': {p.name: file_digest(p) for p in paths}}
        durable_json(self.path/f'{stem}.json', manifest)
        self.count += len(self.frames)
        self.segment += 1
        self.frames.clear()
        self.events.clear()

    def close(self):
        if not self.closed:
            self.closed = True
            while self.thread.is_alive():
                self.check()
                try:
                    self.queue.put(None, timeout=.05)
                    break
                except queue.Full:
                    continue
            self.thread.join()
        self.check()
        return recover_staging(self.path)

    def retry(self):
        """Stopped control only: retry the exact failed segment and queued tail."""
        if self.thread.is_alive():
            raise RuntimeError('journal worker is still active')
        self.error = None
        self.closed = False
        self._flush()
        self.thread = threading.Thread(target=self._worker, name='episode-journal-retry', daemon=True)
        self.thread.start()


def materialize(path, writer, *, allow_incomplete=False):
    """Offline/stopped-control recovery. Returns exact durable rows and image stats."""
    path = Path(path)
    recovery = recover_staging(path)
    if not recovery.complete and not allow_incomplete:
        raise StorageBlocked('staging episode has an incomplete tail')
    if not recovery.frames:
        raise StorageBlocked('staging episode has no complete data prefix')
    columns = {}
    for segment in recovery.segments:
        with np.load(path/segment['arrays'], allow_pickle=False) as data:
            for key in data.files:
                columns.setdefault(key, []).append(data[key].copy())
    columns = {key: np.concatenate(values) for key, values in columns.items()}
    identity = json.loads((path/'episode.json').read_text())
    episode = writer.journal_episode_index(identity.get('journal_id'))
    image_stats = {}
    from vla.datacollection.franka_robotiq_rollout.common import ImageRunningStats
    for key in recovery.segments[0]['videos']:
        # Encode segments independently; stream-copy concatenation performs no model/control work.
        listing = path/f'concat-{key}.txt'
        listing.write_text(''.join("file '"+s['videos'][key]+"'\n" for s in recovery.segments))
        target = writer.video_path(episode, key)
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'concat', '-safe', '1',
                        '-i', str(listing), '-c', 'copy', str(target)], check=True, capture_output=True)
        if video_frames(target) != recovery.frames:
            raise StorageBlocked('final video/data frame mapping mismatch')
        import cv2
        capture = cv2.VideoCapture(str(target))
        stats = ImageRunningStats()
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                stats.update(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        image_stats[key] = stats.as_dict()
    return columns, image_stats, recovery


def publish_audit(path, destination, *, episode_index):
    """Keep the authoritative request/plan/commit records alongside the episode."""
    path, destination = Path(path), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    recovery = recover_staging(path)
    names = [segment['events'] for segment in recovery.segments]
    if recovery.complete:
        names.extend(p.name for p in sorted(path.glob('events-tail-*.json')))
    for name in names:
        shutil.copyfile(path/name, destination/name)
        with (destination/name).open('rb') as stream:
            os.fsync(stream.fileno())
    header = json.loads((path/'episode.json').read_text())
    durable_json(destination/'manifest.json', {'schema_version': 5, 'episode_index': episode_index,
                                              'execution_contract': header.get('execution_contract'),
                                              'files': {name: file_digest(destination/name) for name in names}})
    fsync_directory(destination)
