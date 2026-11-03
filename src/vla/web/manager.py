"""Reattach to independently owned sessions; never restart hardware on reconnect."""
from __future__ import annotations
from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid
from .config import SessionConfig, preflight
from .processes import clean_environment, interpreter_command, process_failure
from .protocol import request


def process_identity(pid):
    try:
        # Linux /proc start time distinguishes PID reuse.
        parts = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if parts[0] == 'Z' else parts[19]
    except (OSError, IndexError):
        return None


class Manager:
    def __init__(self, root, *, demo=False):
        self.root = Path(root).resolve()
        self.directory = self.root/'runtime/web'
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.demo = demo
        self.lock = threading.Lock()
        self._connected_session = None
        self.manifest = self.directory/('demo-active.json' if demo else 'active.json')

    def active(self):
        if not self.manifest.exists():
            return None
        record = json.loads(self.manifest.read_text())
        live = process_identity(record['pid']) == record['identity']
        return {**record, 'alive': live}

    def create(self, data):
        config = SessionConfig.parse(data)
        result = preflight(config, demo=self.demo, root=self.root)
        if not result['ok']:
            raise ValueError('; '.join(result['errors']))
        with self.lock, (self.directory/'manager.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not self.demo:
                from .processing_service import reject_processing_job
                reject_processing_job(self.root)
            active = self.active()
            if active and active['alive']:
                raise RuntimeError('a session is already running')
            sid = uuid.uuid4().hex
            directory = self.directory/sid
            directory.mkdir(mode=0o700)
            runner_config = replace(config, dataset_root=result['dataset_root'])
            if config.ros and not self.demo:
                from .hardware import snapshot
                runner_config = replace(runner_config, ros_config=snapshot(self.root, config.ros_config, directory))
            (directory/'config.json').write_text(json.dumps(asdict(runner_config), ensure_ascii=False, indent=2))
            args = ['-m','vla.web.runner','--directory',str(directory),'--root',str(self.root)]
            if self.demo:
                args.append('--demo')
            ros = config.ros and not self.demo
            python = '/usr/bin/python3' if ros else config.model_python
            argv = interpreter_command(python, args, ros=ros, root=self.root,
                                       workspace=config.ros_workspace or None)
            env = clean_environment(self.root, python, ros=ros)
            with (directory/'runner.log').open('ab') as log:
                process = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True, cwd=self.root)
            identity = process_identity(process.pid)
            if identity is None:
                raise RuntimeError(process_failure('runner', process.poll(), directory/'runner.log'))
            record = {'session_id': sid, 'pid': process.pid, 'identity': identity,
                      'directory': str(directory), 'config': asdict(config), 'demo': self.demo, 'created_at': time.time()}
            temporary = self.manifest.with_suffix('.tmp')
            temporary.write_text(json.dumps(record, ensure_ascii=False))
            temporary.replace(self.manifest)
            # Reap only our child, without tying its lifetime to the web server.
            threading.Thread(target=process.wait, daemon=True).start()
            return record

    def snapshot(self, since=-1):
        """Read cached session state; represent transport lifecycle separately from faults."""
        record = self.active()
        base = {'active': record, 'samples': [], 'commands': [], 'cameras': {}, 'now': time.monotonic()}
        if not record:
            return {**base, 'status': {'state': 'unavailable'}, 'connection_state': 'idle'}
        sid, directory = record['session_id'], Path(record['directory'])
        base.update(session_id=sid, config=record['config'])
        if not record['alive']:
            final = directory/'final.json'
            if final.is_file():
                result = json.loads(final.read_text())
                return {**base, **result, 'connection_state': 'stopped'}
            return {**base, 'connection_state': 'stopped', 'status': {
                'state': 'fault', 'final': True, 'enabled': False,
                'error': process_failure('runner', '已退出', directory/'runner.log')}}
        try:
            result = request(directory/'control.sock', {'kind': 'state', 'session_id': sid, 'since': since})
        except (OSError, RuntimeError, ValueError) as error:
            try:
                created = record.get('created_at')
                if created is None:
                    created = (directory/'config.json').stat().st_mtime
            except OSError:
                created = 0
            waiting = isinstance(error, (FileNotFoundError, ConnectionRefusedError))
            starting = waiting and self._connected_session != sid and 0 <= time.time()-created < 10
            state = 'starting' if starting else 'reconnecting' if self._connected_session == sid else 'delayed'
            message = ('正在建立会话连接' if starting else
                       f'会话连接暂时中断，正在重连。日志：{directory / "runner.log"}'
                       if state == 'reconnecting' else
                       f'会话连接建立延迟，请检查日志：{directory / "runner.log"}')
            if not waiting:
                message += f'\n通信诊断：{error}'
            return {**base, 'status': {'state': 'starting' if starting else 'unavailable'},
                    'connection_state': state, 'connection_message': message}
        self._connected_session = sid
        return {**base, **result, 'connection_state': 'connected', 'connection_message': ''}

    def rpc(self, kind, **values):
        active = self.active()
        if not active:
            raise RuntimeError('no session')
        if not active['alive']:
            final = Path(active['directory'])/'final.json'
            if kind == 'state' and final.exists():
                return json.loads(final.read_text())
            raise RuntimeError('runner stopped; see runner.log')
        if values.get('session_id', active['session_id']) != active['session_id']:
            raise ValueError('stale session')
        return request(Path(active['directory'])/'control.sock',
                       {'kind': kind, 'session_id': active['session_id'], **values})
