"""Detached session state: bounded telemetry, serialized commands, latest JPEGs."""
from __future__ import annotations
import base64
from collections import OrderedDict, deque
import json
from pathlib import Path
import queue
import socketserver
import threading
import time
import uuid
from .protocol import COMMANDS, receive, send


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


class Runtime:
    def __init__(self, session_id, config, directory, backend=None):
        self.id, self.config, self.directory = session_id, config, Path(directory)
        self.backend = backend
        self.commands = queue.Queue(16)
        self.results = OrderedDict()
        self.lock = threading.Lock()
        self.revision = 0
        self.seq = 0
        self.samples = deque(maxlen=6000)
        self.status = {'state': 'starting', 'enabled': False, 'recording': False, 'available_commands':['stop'], 'updated_at':time.monotonic()}
        self.frames = {}
        self.frame_rates = {}
        self.frame_interest = {}
        self.stopped = threading.Event()
        self.cancel_start = threading.Event()
        self.completed = False
        self.pending_completion = None
        self.last_sample = 0.
        self.monitor_error = None
        self._threads = []
        self.server = None

    def rpc(self, message):
        if message.get('session_id') != self.id:
            raise ValueError('wrong session identity')
        kind = message.get('kind')
        with self.lock:
            if kind == 'state':
                since = int(message.get('since', -1))
                samples = [v for v in self.samples if v['seq'] > since][-100:]
                return {'session_id': self.id, 'revision': self.revision, 'seq': self.seq,
                        'status': self.status, 'samples': samples,
                        'commands': list(self.results.values())[-16:], 'now': time.monotonic(),
                        'monitor_error': self.monitor_error,
                        'cameras': {r: {'captured_at': f[0], 'encoded_at': f[2], 'fps': self.frame_rates.get(r)} for r,f in self.frames.items()}}
            if kind == 'camera':
                role = message.get('role')
                if role not in self.config.cameras:
                    raise ValueError('unknown camera')
                self.frame_interest[role] = time.monotonic()
                frame = self.frames.get(role)
                return {'captured_at': frame[0] if frame else 0,
                        'jpeg': base64.b64encode(frame[1]).decode() if frame else None}
            if kind != 'command':
                raise ValueError('unknown local request')
            cid = message.get('command_id')
            uuid.UUID(cid)
            if cid in self.results:
                return self.results[cid]
            if self.completed or self.stopped.is_set():
                raise RuntimeError('session has ended')
            if type(message.get('revision')) is not int or message['revision'] != self.revision:
                raise RuntimeError('state changed; refresh before submitting')
            if message.get('operation') not in COMMANDS or not isinstance(message.get('args', {}), dict):
                raise ValueError('unknown command')
            if any(v['status'] == 'accepted' for v in self.results.values()):
                raise RuntimeError('another command is pending')
            if self.backend is None and message['operation'] == 'stop':
                self.cancel_start.set()
            self.commands.put_nowait(message)
            self.revision += 1
            record = {'command_id': cid, 'status': 'accepted', 'operation': message['operation'], 'revision': self.revision}
            self.results[cid] = record
            while len(self.results) > 256:
                self.results.popitem(last=False)
            return record

    def poll_command(self):
        if self.pending_completion is not None:
            cid, check = self.pending_completion
            try:
                if not check():
                    return
                result = {**self.results[cid], 'status': 'completed'}
            except Exception as error:
                result = {**self.results[cid], 'status': 'failed', 'error': str(error)}
            with self.lock:
                self.results[cid] = result
            self.pending_completion = None
            return
        try:
            message = self.commands.get_nowait()
        except queue.Empty:
            return
        cid, op = message['command_id'], message['operation']
        try:
            args = message.get('args', {})
            if set(args) - {'mode', 'outcome'} or args.get('outcome', 'success') not in ('success', 'failure'):
                raise ValueError('invalid operation arguments')
            if op == 'save':
                with self.lock:
                    self.status = {**self.status, 'state':'saving', 'saving':True, 'updated_at':time.monotonic()}
            completion = self.backend.command(op, args)
            if callable(completion):
                self.pending_completion = (cid, completion)
                return
            result = {**self.results[cid], 'status': 'completed'}
            if op == 'stop':
                self.completed = True
        except Exception as error:
            result = {**self.results[cid], 'status': 'failed', 'error': str(error)}
        with self.lock:
            self.results[cid] = result

    def publish(self):
        now = time.monotonic()
        if now - self.last_sample < .05:
            return
        self.last_sample = now
        try:
            sample = self.backend.sample()
            values = sample.pop('values', {})
            stamp = sample.pop('sample_time', now)
            with self.lock:
                self.seq += 1
                self.samples.append({'seq': self.seq, 'time': stamp, 'values': values})
                if self.completed or self.stopped.is_set():
                    return
                if any(r['status']=='accepted' and r['operation']=='save' for r in self.results.values()):
                    sample.update(state='saving',saving=True)
                self.status = {**sample, 'updated_at': now, 'session_id': self.id}
                self.monitor_error = None
        except Exception as error:
            self.monitor_error = str(error)

    def _monitor(self):
        # Backend sample() reads owner caches only. Even failed/slow telemetry is
        # independent of the control/ROS event loop.
        while not self.stopped.wait(.05):
            if self.backend is not None and not self.completed:
                self.publish()

    def _encode(self):
        import cv2
        while not self.stopped.wait(.1):
            with self.lock:
                interested = {r for r,t in self.frame_interest.items() if time.monotonic()-t < 2}
            if not interested or self.backend is None:
                continue
            try:
                images = self.backend.images()
                for role in interested:
                    if role not in images:
                        continue
                    stamp, image = images[role]
                    old = self.frames.get(role)
                    if old and old[0] == stamp:
                        continue
                    h,w = image.shape[:2]
                    scale = min(1., 640/w, 480/h)
                    if scale < 1:
                        image = cv2.resize(image, (int(w*scale), int(h*scale)))
                    ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                               [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        with self.lock:
                            if old and stamp>old[0]:
                                rate=1/(stamp-old[0])
                                self.frame_rates[role]=.8*self.frame_rates.get(role,rate)+.2*rate
                            self.frames[role] = (stamp, encoded.tobytes(), time.monotonic())
            except Exception as error:
                self.monitor_error = str(error)

    def start(self):
        runtime = self
        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.settimeout(2)
                try:
                    response = runtime.rpc(receive(self.request))
                    send(self.request, plain(response))
                except Exception as error:
                    try:
                        send(self.request, {'error': str(error)})
                    except OSError:
                        pass
        class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True
            request_queue_size = 16
            slots = threading.BoundedSemaphore(16)
            def process_request(self, request, address):
                if not self.slots.acquire(blocking=False):
                    self.shutdown_request(request)
                    return
                try:
                    super().process_request(request,address)
                except BaseException:
                    self.slots.release()
                    raise
            def process_request_thread(self, request, address):
                try:
                    super().process_request_thread(request,address)
                finally:
                    self.slots.release()
        self.server = Server(str(self.directory/'control.sock'), Handler)
        (self.directory/'control.sock').chmod(0o600)
        for target in (self.server.serve_forever, self._encode, self._monitor):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)

    def close(self):
        self.stopped.set()
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        for thread in self._threads:
            thread.join(timeout=2)
        (self.directory/'control.sock').unlink(missing_ok=True)
