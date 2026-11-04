"""Detached process; only its main loop invokes device/session commands."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import signal
import time
import traceback
from .config import SessionConfig
from .runtime import Runtime, plain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    config = SessionConfig.parse(json.loads((args.directory/'config.json').read_text()))
    runtime = Runtime(args.directory.name, config, args.directory)
    backend = None
    interrupted = False
    def stop(*_):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (args.root/'runtime/web'/('demo.lock' if args.demo else 'hardware.lock')).open('a') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            runtime.start()
            if args.demo:
                from .demo import DemoBackend
                backend = DemoBackend(config, args.directory, args.root)
            else:
                from .backends import make_backend
                backend = make_backend(config, args.directory, args.root, cancelled=runtime.cancel_start.is_set)
            runtime.backend = backend
            while not runtime.completed and not interrupted:
                started = time.monotonic()
                runtime.poll_command()
                if not runtime.completed:
                    backend.tick()
                time.sleep(max(0., .005-(time.monotonic()-started)))
            runtime.status = {**runtime.status, 'state': 'stopped', 'enabled': False}
        except BaseException as error:
            traceback.print_exc()
            runtime.status = {**runtime.status, 'state': 'stopped' if runtime.cancel_start.is_set() else 'fault', 'enabled': False, 'error': str(error)}
        finally:
            runtime.stopped.set()
            try:
                if backend is not None:
                    backend.close()
            except Exception as error:
                traceback.print_exc()
                runtime.status = {**runtime.status, 'state': 'fault', 'error': str(error)}
            for cid, command in list(runtime.results.items()):
                if command['status'] == 'accepted':
                    runtime.results[cid] = {**command, 'status':'completed' if command['operation']=='stop' and runtime.cancel_start.is_set() else 'failed',
                                            'error':None if runtime.cancel_start.is_set() else 'session ended before completion'}
            if backend is not None and hasattr(backend, 'poses'):
                runtime.status = {**runtime.status, 'pose_operation': backend.poses.status}
            result = {'session_id': runtime.id, 'revision': runtime.revision, 'seq': runtime.seq,
                      'status': {**runtime.status, 'final': True}, 'samples': [], 'commands': list(runtime.results.values()),
                      'now': time.monotonic(), 'cameras': {}}
            (args.directory/'final.json').write_text(json.dumps(plain(result), allow_nan=False))
            runtime.close()


if __name__ == '__main__':
    main()
