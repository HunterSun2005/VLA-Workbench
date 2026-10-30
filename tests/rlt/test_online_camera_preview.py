from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vla.datacollection.franka_robotiq_rollout import collect_franky as entry
from vla.datacollection.start_teleop import gamepad_preview


@pytest.mark.parametrize("mode", ["STANDBY", "TELEOP", "PRIMING", "RLT REC", "HUMAN REC", "SAVING/TRAINING", "FAULT"])
def test_preview_status_is_bounded_and_does_not_access_hardware(mode):
    c = object.__new__(entry.FrankyRolloutCollector)
    c.preview = Mock()
    c.robot = Mock()
    c.writer = Mock()
    c._faulted = mode == "FAULT"
    c._processing = mode == "SAVING/TRAINING"
    c._episode_active = mode in {"PRIMING", "RLT REC", "HUMAN REC", "SAVING/TRAINING"}
    c._intervention = mode == "HUMAN REC"
    c._priming = mode == "PRIMING"
    c._phase, c._full_episode = "rl", True
    c._preview_episode, c._states = 12, [None] * 42
    c._setup_teleop = SimpleNamespace(mode="active" if mode == "TELEOP" else "standby", high_speed=False)
    c._gamepad_controls = SimpleNamespace(high_speed=True)
    c.update_preview()
    status = c.preview.update_status.call_args.args[0]
    assert status["mode"] == mode
    assert status["frames"] == 42 and status["episode"] == 12
    assert status["full_episode"] is True
    assert status["speed"] == ("HIGH" if c._episode_active else "LOW")
    assert not c.robot.mock_calls and not c.writer.mock_calls


@pytest.mark.parametrize("disabled,fail", [(False, False), (True, False), (False, True)])
def test_collector_preview_lifecycle_uses_existing_buffer_and_closes_first(tmp_path, monkeypatch, disabled, fail):
    events = []
    robot = Mock()
    robot.connect.side_effect = lambda: events.append("connect")
    collector = Mock(_gamepad_controls=None, _full_episode=True)
    collector.close.side_effect = lambda: events.append("collector_close")
    viewer = Mock()
    def start():
        events.append("preview_start")
        if fail:
            raise RuntimeError("GUI unavailable")
    viewer.start.side_effect = start
    viewer.close.side_effect = lambda: events.append("preview_close")
    preview_factory = Mock(return_value=viewer)
    monkeypatch.setattr(gamepad_preview, "CameraPreview", preview_factory)
    monkeypatch.setattr(entry, "FrankaRobotiqRobot", Mock(return_value=robot))
    monkeypatch.setattr(entry, "FrankyRolloutCollector", Mock(return_value=collector))
    monkeypatch.setattr(entry, "JsonlDebugTrace", Mock())
    monkeypatch.setattr(entry, "_configure_affinity", Mock())
    monkeypatch.setattr(entry, "_confirm_execute", Mock())
    monkeypatch.setattr(entry, "KeyboardEvents", Mock(return_value=SimpleNamespace(consume=lambda: ["quit"], close=lambda: None)))
    config = Path(__file__).parents[2] / "src/vla/datacollection/franka_robotiq_rollout/configs/single_left.yaml"
    monkeypatch.setattr("sys.argv", ["collect", "--config", str(config), "--execute", "--online",
                                    "--dataset-root", str(tmp_path), *(["--no-preview"] if disabled else [])])
    entry.main()
    robot.prepare_for_control.assert_not_called()
    if disabled:
        preview_factory.assert_not_called()
        assert events == ["connect", "collector_close"]
    else:
        assert preview_factory.call_args.args[1] == robot.get_camera_preview_snapshot
        assert collector.preview is viewer
        assert events == ["connect", "preview_start", "preview_close", "collector_close"]
    robot.get_camera_preview_snapshot.assert_not_called()  # No camera reads on the main thread.


def test_full_episode_overlay_does_not_request_critical_marking(monkeypatch):
    import cv2
    import numpy as np
    labels = []
    put_text = cv2.putText
    def capture(image, text, *args, **kwargs):
        labels.append(text)
        return put_text(image, text, *args, **kwargs)
    monkeypatch.setattr(cv2, "putText", capture)
    frame = gamepad_preview.render_frame(np.zeros(gamepad_preview.SHAPE, np.uint8),
        {"mode": "RLT REC", "full_episode": True, "captured_at": 1}, now=1)
    assert frame.shape == (560, 896, 3)
    assert any("RLT FULL" in text for text in labels)
    assert not any("critical" in text for text in labels)
