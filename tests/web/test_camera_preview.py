"""Preview timing and independent failure recovery; no physical devices."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vla.web.config import SessionConfig
from vla.web.demo import DemoBackend
from vla.web.runtime import Runtime
from vla.robots.franka_robotiq.camera import RealSenseCapture


@pytest.mark.parametrize('workflow,expected', [('collect', 30), ('deploy', 10)])
def test_jittered_source_reaches_preview_rate_without_duplicate_encoding(tmp_path, monkeypatch, workflow, expected):
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr('vla.web.runtime.time.monotonic', lambda: clock.now)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    cfg = SessionConfig(task='test', workflow=workflow, arms='dual')
    def images():
        index = int((clock.now-100)*30)
        stamp = 100 + index/30 - (.001 if index % 2 else 0)
        return {role: (stamp, image) for role in cfg.cameras}
    runtime = Runtime('test', cfg, tmp_path, SimpleNamespace(images=images))
    unique = {role: set() for role in cfg.cameras}
    for tick in range(expected * 4 * 6):
        clock.now = 100 + tick/(expected*4)
        runtime.frame_interest = {role: clock.now for role in cfg.cameras}
        runtime._encode_once(clock.now)
        for role, frame in runtime.frames.items():
            unique[role].add(frame[0])
    for role in cfg.cameras:
        assert expected*5.8 <= len(unique[role]) <= expected*6.1
        assert runtime._preview_rate(role, clock.now) == pytest.approx(expected, abs=1)
        assert len(runtime.frame_times[role]) <= 120
    response = runtime.rpc({'kind': 'camera', 'role': 'base', 'session_id': 'test'})
    assert response['preview_fps'] == expected
    assert response['captured_at'] <= response['encoded_at']
    assert response['jpeg']
    clock.now += .6
    assert runtime._preview_rate('base', clock.now) is None


def test_camera_and_telemetry_errors_recover_independently(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr('vla.web.runtime.time.monotonic', lambda: clock.now)
    cfg = SessionConfig(task='test', arms='dual')
    backend = SimpleNamespace(telemetry_bad=True, camera_bad=True, calls=0)
    def sample():
        if backend.telemetry_bad:
            raise RuntimeError('telemetry unavailable')
        return {'state': 'standby'}
    def images():
        backend.calls += 1
        return {role: (clock.now, np.zeros((0, 0, 3) if role == 'right_wrist' and backend.camera_bad
                                         else (8, 8, 3), dtype=np.uint8)) for role in cfg.cameras}
    backend.sample, backend.images = sample, images
    runtime = Runtime('test', cfg, tmp_path, backend)
    runtime.frame_interest = {role: clock.now for role in cfg.cameras}
    runtime.publish(); runtime._encode_once(clock.now)
    assert set(runtime.monitor_errors) == {'telemetry', 'camera.right_wrist'}
    assert set(runtime.frames) == {'base', 'left_wrist'}
    backend.telemetry_bad = False; clock.now += .1; runtime.publish()
    assert set(runtime.monitor_errors) == {'camera.right_wrist'}
    backend.camera_bad = False; clock.now += .1; runtime._encode_once(clock.now)
    assert not runtime.monitor_errors and runtime.monitor_error is None
    clock.now += .6
    runtime._encode_once(clock.now)  # Backend supplies fresh frames for every role.
    assert not runtime.monitor_errors
    old_calls = backend.calls
    clock.now += 3
    assert not runtime._encode_once(clock.now)
    assert backend.calls == old_calls and not runtime.frame_times


def test_missing_camera_marks_only_that_view_stale(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr('vla.web.runtime.time.monotonic', lambda: clock.now)
    cfg = SessionConfig(task='test')
    runtime = Runtime('test', cfg, tmp_path,
        SimpleNamespace(images=lambda: {'base': (clock.now, np.zeros((8, 8, 3), dtype=np.uint8))}))
    runtime.frame_interest = {role: clock.now for role in cfg.cameras}
    runtime._encode_once(clock.now)
    assert set(runtime.monitor_errors) == {'camera.left_wrist'}
    assert runtime.frames['base'][0] == clock.now


def test_latest_preview_is_independent_of_policy_synchronization():
    capture = RealSenseCapture({'base': 'fake', 'left_wrist': 'fake2'}, width=640, height=480,
        fps=30, image_size=224, timeout_s=.5, max_pair_skew_s=.02, max_frame_age_s=.1)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    capture._frames['base'].append((10., image))
    capture._frames['left_wrist'].append((1., image))
    capture._errors['left_wrist'] = RuntimeError('offline')
    result = capture.latest_frames()
    assert result['base'][1] is image and result['base'][0] == 10.
    assert result['left_wrist'][0] == 1.
    with pytest.raises(RuntimeError, match='offline'):
        capture.snapshot()  # Policy health checks remain strict.


@pytest.mark.parametrize('workflow,rate', [('collect', 30), ('deploy', 10)])
def test_demo_uses_absolute_frame_clock(tmp_path, monkeypatch, workflow, rate):
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr('vla.web.demo.time.monotonic', lambda: clock.now)
    backend = DemoBackend(SessionConfig(task='test', workflow=workflow), tmp_path)
    timestamps = []
    for frame in range(12):
        clock.now = 100 + (frame+.05)/rate
        timestamps.append(backend.images()['base'][0])
    assert len(set(timestamps)) == 12
    assert timestamps[-1]-timestamps[0] == pytest.approx(11/rate)


def test_soak_parses_split_headers_and_jpeg_payloads():
    spec = spec_from_file_location('camera_soak', Path(__file__).resolve().parents[2] / 'scripts/web_demo_soak.py')
    module = module_from_spec(spec); spec.loader.exec_module(module)
    jpeg = b'\xff\xd8jpeg\xff\xd9'
    packet = (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpeg)).encode() +
              b'\r\nX-Captured-At: 100.1\r\nX-Encoded-At: 100.2\r\n\r\n' + jpeg + b'\r\n')
    body = packet * 3
    frames = list(module.multipart_frames(body[i:i+3] for i in range(0, len(body), 3)))
    assert len(frames) == 3
    assert all(headers[b'x-captured-at'] == b'100.1' and data == jpeg for headers, data in frames)
