"""Measure independent MJPEG frames in a synthetic panel; never uses hardware."""
import argparse
from collections import deque
import json
from pathlib import Path
import statistics
import sys
import threading
import time
import uuid

import httpx
import psutil
from websockets.sync.client import connect


def multipart_frames(chunks):
    """Parse Content-Length framing even when HTTP chunks split headers/JPEGs."""
    buffer = bytearray()
    for chunk in chunks:
        buffer.extend(chunk)
        if len(buffer) > 4 * 1024 * 1024:
            raise ValueError('MJPEG buffer exceeded 4 MiB')
        while True:
            end = buffer.find(b'\r\n\r\n')
            if end < 0:
                break
            lines = bytes(buffer[:end]).split(b'\r\n')
            headers = {key.strip().lower(): value.strip() for line in lines if b':' in line
                       for key, value in [line.split(b':', 1)]}
            length = int(headers[b'content-length'])
            if not 0 < length <= 2 * 1024 * 1024:
                raise ValueError('invalid MJPEG Content-Length')
            frame_end = end + 4 + length
            if len(buffer) < frame_end:
                break
            jpeg = bytes(buffer[end+4:frame_end])
            del buffer[:frame_end]
            yield headers, jpeg


def percentiles(values):
    values = sorted(values)
    return {'p50_ms': statistics.median(values) * 1000,
            'p95_ms': values[min(len(values)-1, int(len(values)*.95))] * 1000} if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--web-pid', type=int, required=True)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--workflow', choices=('collect', 'deploy'), default='collect')
    parser.add_argument('--output', type=Path, default=Path('runtime/web/soak.json'))
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    base = f'http://127.0.0.1:{args.port}'
    client = httpx.Client(timeout=5, trust_env=False)
    boot = client.get(base + '/api/bootstrap').json()
    if not boot.get('demo'):
        raise SystemExit('Refusing: server must be started with --demo')
    if boot.get('active') and boot['active']['alive']:
        raise SystemExit('Stop existing demo session before soak')
    token = boot['token']
    headers = {'X-VLA-Token': token}
    data = {'workflow': args.workflow, 'arms': 'dual', 'source': 'gamepad',
            'task': 'Web simulated bimanual soak', 'mode': 'auto' if args.workflow == 'deploy' else 'vla',
            'model_python': sys.executable}
    response = client.post(base + '/api/sessions', json=data, headers=headers)
    response.raise_for_status()
    active = response.json()
    sid = active['session_id']
    runner, web = psutil.Process(active['pid']), psutil.Process(args.web_pid)
    stop = threading.Event()
    lock = threading.Lock()
    roles = ('base', 'left_wrist', 'right_wrist')
    counts = {role: 0 for role in roles}
    duplicates = {role: 0 for role in roles}
    ages = {role: deque(maxlen=100000) for role in roles}
    encode_ages = {role: deque(maxlen=100000) for role in roles}
    receive_times = {role: [None, None] for role in roles}
    stream_errors = {}
    threads = []

    def command(operation):
        state = client.get(base + '/api/state').json()
        cid = str(uuid.uuid4())
        response = client.post(base + '/api/commands', json={'session_id': sid,
            'revision': state['revision'], 'command_id': cid, 'operation': operation, 'args': {}}, headers=headers)
        response.raise_for_status()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = client.get(base + '/api/state').json()
            result = next((item for item in state.get('commands', []) if item['command_id'] == cid), None)
            if result and result['status'] != 'accepted':
                if result['status'] != 'completed':
                    raise RuntimeError(result)
                return
            time.sleep(.05)
        raise TimeoutError(operation)

    def camera(role):
        previous = None
        try:
            with httpx.Client(timeout=httpx.Timeout(5, read=2), trust_env=False) as connection:
                with connection.stream('GET', base + '/api/cameras/' + role) as stream:
                    stream.raise_for_status()
                    for frame_headers, jpeg in multipart_frames(stream.iter_bytes()):
                        now = time.monotonic()
                        if stop.is_set():
                            return
                        captured = float(frame_headers[b'x-captured-at'])
                        encoded = float(frame_headers[b'x-encoded-at'])
                        if not jpeg.startswith(b'\xff\xd8') or not jpeg.endswith(b'\xff\xd9'):
                            raise ValueError('invalid JPEG framing')
                        with lock:
                            if captured == previous:
                                duplicates[role] += 1
                                continue
                            previous = captured
                            counts[role] += 1
                            receive_times[role][0] = receive_times[role][0] or now
                            receive_times[role][1] = now
                            ages[role].append(max(0, now-captured))
                            encode_ages[role].append(max(0, encoded-captured))
        except Exception as error:
            if not stop.is_set():
                with lock:
                    stream_errors[role] = str(error)

    try:
        for _ in range(100):
            state = client.get(base + '/api/state').json()
            if state.get('status', {}).get('state') == 'standby':
                break
            time.sleep(.1)
        else:
            raise RuntimeError(state)
        for operation in ('enable', 'record'):
            command(operation)
        started = last = time.monotonic()
        for role in roles:
            thread = threading.Thread(target=camera, args=(role,), daemon=True)
            threads.append(thread)
            thread.start()
        metrics, latencies = [], deque(maxlen=100000)
        messages = samples = 0
        runner.cpu_percent(); web.cpu_percent()
        with connect(f'ws://127.0.0.1:{args.port}/api/live', subprotocols=['vla', token], origin=base, proxy=None) as ws:
            while time.monotonic()-started < args.seconds:
                message = json.loads(ws.recv(timeout=10))
                messages += 1
                batch = message.get('samples', [])
                samples += len(batch)
                latencies.extend(max(0, message['now']-sample['time']) for sample in batch)
                now = time.monotonic()
                if now-last >= 10:
                    with lock:
                        row = {'elapsed_s': round(now-started, 2),
                            'runner_rss_mb': runner.memory_info().rss/2**20,
                            'web_rss_mb': web.memory_info().rss/2**20,
                            'runner_cpu_percent': runner.cpu_percent(), 'web_cpu_percent': web.cpu_percent(),
                            'frames': dict(counts), 'seq': message.get('seq')}
                    metrics.append(row)
                    last = now
                    print(json.dumps(row), flush=True)
        duration = time.monotonic()-started
        stop.set()
        for thread in threads:
            thread.join(timeout=3)
        with lock:
            frame_metrics = {role: {'frames': counts[role], 'duplicates': duplicates[role],
                'received_fps': ((counts[role]-1)/(receive_times[role][1]-receive_times[role][0])
                    if counts[role] > 1 else 0),
                'received_age': percentiles(ages[role]), 'encode_age': percentiles(encode_ages[role])}
                for role in roles}
        result = {'workflow': args.workflow, 'target_preview_fps': 30 if args.workflow == 'collect' else 10,
            'duration_s': duration, 'messages': messages, 'samples': samples,
            'websocket_hz': messages/duration, 'telemetry_hz': samples/duration,
            'cameras': frame_metrics, 'telemetry_age': percentiles(latencies),
            'metrics': metrics, 'stream_errors': stream_errors, 'session_id': sid,
            'age_clock': 'same-host monotonic; HTTP receive time, not browser presentation time'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f'DONE {args.output}', flush=True)
        if stream_errors or any(counts[role] < 2 for role in roles):
            raise RuntimeError('camera streams incomplete; see result')
    finally:
        stop.set()
        # Only this synthetic session is owned by this script.
        try:
            state = client.get(base + '/api/state').json()
            if state.get('session_id') == sid and not state.get('status', {}).get('final'):
                if state.get('status', {}).get('recording'):
                    command('discard')
                command('stop')
        finally:
            client.close()


if __name__ == '__main__':
    main()
