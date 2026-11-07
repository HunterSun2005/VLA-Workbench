"""Concurrent control messages around a single model worker; no execution cursor."""
from contextlib import contextmanager
from threading import Lock, RLock


class PlanGate:
    def __init__(self):
        self.lock = RLock()
        self.worker = Lock()
        self.generations = {}
        self.requests = {}
        self.progress = {}

    def invalidate(self, session, generation):
        with self.lock:
            current = self.generations.get(session, -1)
            if generation < current:
                raise ValueError('stale invalidation generation')
            self.generations[session] = generation

    def current(self, request):
        with self.lock:
            return self.generations.get(request.session_id) == request.generation

    @contextmanager
    def planning(self, request):
        if not self.worker.acquire(blocking=False):
            raise RuntimeError('one inference or training operation is already in flight')
        try:
            with self.lock:
                session = request.session_id
                if request.generation < self.generations.get(session, -1):
                    raise ValueError('revoked generation')
                if request.request_id <= self.requests.get(session, -1):
                    raise ValueError('duplicate or out of order request')
                self.generations[session] = request.generation
                self.requests[session] = request.request_id
            yield
        finally:
            self.worker.release()


def control_message(server, header):
    kind = header.get('kind')
    if kind not in {'invalidate', 'progress'}:
        return None
    gate = server.plan_gate
    if kind == 'invalidate':
        gate.invalidate(header['session_id'], int(header['generation']))
        return {'kind': kind, 'request_id': header.get('request_id', -1)}, {}
    if kind == 'progress':
        with gate.lock:
            gate.progress[header['session_id']] = dict(header)
        return {'kind': kind, 'request_id': header.get('request_id', -1)}, {}
    return None
