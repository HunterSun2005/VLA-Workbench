import mmap
import struct
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from vla.datacollection.start_teleop import gamepad_preview as preview
from vla.datacollection.start_teleop import gamepad_record as entry
from vla.robots.franka_robotiq.robot import FrankaRobotiqRobot


def images():
    return {"base": np.full((224, 224, 3), [255, 0, 0], np.uint8),
            "left_wrist": np.full((224, 224, 3), [0, 255, 0], np.uint8)}


def test_mailbox_latest_frame_roundtrip_and_independent_copies():
    with mmap.mmap(-1, preview.SLOT_SIZE) as slot:
        assert preview.read_slot(slot) is None
        original = images()
        preview.write_slot(slot, original, {"frames": 1, "critical": "before"})
        original["base"][:] = 0
        frames, status = preview.read_slot(slot)
        assert status["frames"] == 1
        np.testing.assert_array_equal(frames[0, 0, 0], [255, 0, 0])
        # Many writes with no reader never queue up or wait for the display.
        for count in range(50):
            preview.write_slot(slot, original, {"frames": count})
        frames, status = preview.read_slot(slot)
        assert status == {"frames": 49}
        np.testing.assert_array_equal(frames[0], original["base"])
        struct.pack_into("<Q", slot, 0, 101)
        assert preview.read_slot(slot) is None  # in-progress frame is skipped


def test_mailbox_invalid_payload_is_not_displayed():
    with mmap.mmap(-1, preview.SLOT_SIZE) as slot:
        struct.pack_into("<Q", slot, 0, 2)
        struct.pack_into("<I", slot, 8, preview.HEADER_SIZE)
        assert preview.read_slot(slot) is None
        with pytest.raises(ValueError, match="too long"):
            preview.write_slot(slot, images(), {"text": "a" * preview.HEADER_SIZE})


def test_render_rgb_order_stale_warning_and_no_mutation():
    pair = np.stack(list(images().values()))
    original = pair.copy()
    rendered = preview.render_frame(pair, {"captured_at": 1., "critical": "critical"}, now=1.1)
    assert rendered.shape == (560, 896, 3)
    np.testing.assert_array_equal(rendered[100, 100], [0, 0, 255])
    np.testing.assert_array_equal(rendered[100, 600], [0, 255, 0])
    stale = preview.render_frame(pair, {"captured_at": 1.}, now=2.)
    np.testing.assert_array_equal(stale[270, 0], [0, 0, 100])
    np.testing.assert_array_equal(pair, original)


@pytest.mark.parametrize("key", ["<Escape>", "<Key-q>", "WM_DELETE_WINDOW"])
def test_gui_close_only_ends_display_loop(key):
    calls = []
    class Root:
        def __init__(self):
            calls.append("open")
            self.callbacks, self.scheduled = {}, []
        def title(self, *args):
            pass
        def resizable(self, *args):
            pass
        def protocol(self, key, callback):
            self.callbacks[key] = callback
        bind = protocol
        def after(self, delay, callback):
            self.scheduled.append(callback)
        def mainloop(self):
            self.scheduled.pop(0)()
            self.callbacks[key]()
        def destroy(self):
            calls.append("closed")
    class Label:
        def __init__(self, *args):
            pass
        def pack(self):
            pass
        def configure(self, **kwargs):
            calls.append("show")
    gui = SimpleNamespace(Tk=Root, Label=Label)
    image_tk = SimpleNamespace(PhotoImage=lambda *args, **kw: object())
    with mmap.mmap(-1, preview.SLOT_SIZE) as slot:
        preview.display_loop(slot, gui, image_tk)
    assert calls == ["open", "show", "closed"]


@pytest.mark.parametrize("enabled,display", [(False, ":fake"), (True, "")])
def test_disabled_or_headless_never_starts_gui_or_reads_camera(tmp_path, monkeypatch, enabled, display):
    monkeypatch.setenv("DISPLAY", display)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(preview.subprocess, "Popen", lambda *a, **kw: pytest.fail("GUI process started"))
    with preview.CameraPreview(tmp_path, lambda: pytest.fail("camera accessed"), enabled=enabled, report=lambda _: None) as viewer:
        viewer.update_status({"frames": 10})
    assert not list(tmp_path.iterdir())


class FakeProcess:
    def __init__(self, *args, **kwargs):
        self.returncode = None
        self.arguments, self.options = args, kwargs

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


def test_feeder_uses_existing_frames_in_background_and_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":fake")
    monkeypatch.setattr(preview.subprocess, "Popen", FakeProcess)
    ready = threading.Event()
    write = preview.write_slot
    def publish(*args):
        write(*args)
        ready.set()
    monkeypatch.setattr(preview, "write_slot", publish)
    threads = []
    def source():
        threads.append(threading.current_thread().name)
        return SimpleNamespace(images=images(), timestamps={"base": 1., "left_wrist": 1.01})
    with preview.CameraPreview(tmp_path, source, report=lambda _: None) as viewer:
        assert ready.wait(2)
        frame, status = preview.read_slot(viewer._slot)
        assert status["captured_at"] == 1.
        assert frame.shape == preview.SHAPE
        assert set(threads) == {"gamepad-camera-preview"}
        assert viewer._process.options["start_new_session"] is True
    assert not viewer._thread.is_alive() and viewer._process.poll() is not None


def test_camera_read_failure_marks_preview_stale_without_propagating(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":fake")
    monkeypatch.setattr(preview.subprocess, "Popen", FakeProcess)
    ready = threading.Event()
    write = preview.write_slot
    def publish(*args):
        write(*args)
        ready.set()
    monkeypatch.setattr(preview, "write_slot", publish)
    def source():
        raise TimeoutError("fake stale camera")
    with preview.CameraPreview(tmp_path, source, report=lambda _: None) as viewer:
        assert ready.wait(2)
        _, status = preview.read_slot(viewer._slot)
        assert status["captured_at"] == 0
        assert status["camera_error"] == "fake stale camera"


@pytest.mark.parametrize("returncode", [0, -6])
def test_closed_or_crashed_child_does_not_raise_into_control(tmp_path, monkeypatch, returncode):
    monkeypatch.setenv("DISPLAY", ":fake")
    process = FakeProcess()
    process.returncode = returncode
    monkeypatch.setattr(preview.subprocess, "Popen", lambda *a, **kw: process)
    messages = []
    with preview.CameraPreview(tmp_path, lambda: pytest.fail("camera read after GUI exit"), report=messages.append) as viewer:
        viewer._thread.join(2)
        viewer.update_status({"mode": "RECORDING"})
        assert viewer.status["mode"] == "RECORDING"
    assert any("recording continues" in message for message in messages)


def test_gui_spawn_failure_is_optional(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":fake")
    def fail(*args, **kwargs):
        raise OSError("fake spawn failure")
    monkeypatch.setattr(preview.subprocess, "Popen", fail)
    messages = []
    with preview.CameraPreview(tmp_path, lambda: None, report=messages.append):
        pass
    assert any("recording remains available" in message for message in messages)


def test_camera_buffer_api_never_touches_arm_or_connects():
    expected = object()
    class Camera:
        def snapshot(self):
            return expected
        def connect(self):
            pytest.fail("camera connected again")
    robot = SimpleNamespace(_camera=Camera())
    assert FrankaRobotiqRobot.get_camera_preview_snapshot(robot) is expected


@pytest.mark.parametrize("extra,enabled", [([], True), (["--no-preview"], False)])
def test_cli_preview_default_and_headless_override(tmp_path, monkeypatch, extra, enabled):
    monkeypatch.setattr(entry, "recording_preflight", lambda *a: None)
    calls = []
    monkeypatch.setattr(entry, "run", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr("sys.argv", ["record", "--execute", "--dataset-root", str(tmp_path), "--task", "test", *extra])
    entry.main()
    assert calls == [{"preview": enabled}]
