from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from vla.deployment.franka_pi05_rtc import (
    LEGACY_PADDING_CAMERAS,
    PHYSICAL_CAMERAS,
    JsonlDebugTrace,
    MockFrankaRobotiqRobot,
    policy_observation,
    validate_checkpoint,
)
from vla.control.geometry import ACTION_FIELDS, STATE_FIELDS


def _checkpoint(tmp_path, **updates):
    root = tmp_path / "checkpoint"
    root.mkdir()
    inputs = {
        key: {"type": "VISUAL", "shape": [3, 224, 224]}
        for key in PHYSICAL_CAMERAS
    }
    config = {
        "type": "pi05", "input_features": {**inputs, "observation.state": {"shape": [10]}},
        "output_features": {"action": {"shape": [10]}}, "use_relative_actions": False,
        "chunk_size": 50, "action_feature_names": list(ACTION_FIELDS),
    }
    config.update(updates)
    (root / "config.json").write_text(json.dumps(config))
    (root / "policy_preprocessor.json").write_text(json.dumps({"steps": [{
        "registry_name": "tokenizer_processor", "config": {"tokenizer_name": "cached-tokenizer"}
    }]}))
    (root / "policy_postprocessor.json").write_text('{"steps": []}')
    stats = {
        "action.min": torch.tensor([0.4, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        "action.max": torch.tensor([0.7, 0.2, 0.6, 0.0, 1.0, 0.2, -1.0, 0.0, 0.2, 1.0]),
        "action.q01": torch.tensor([0.42, -0.18, 0.32, 0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0]),
        "action.q99": torch.tensor([0.68, 0.18, 0.58, 0.0, 0.9, 0.1, -0.9, 0.0, 0.1, 1.0]),
    }
    save_file(stats, root / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    (root / "model.safetensors").write_bytes(b"weights")
    return root


@pytest.mark.parametrize('space', ['joint', 'tcp'])
@pytest.mark.parametrize('representation', ['absolute', 'chunk_relative'])
def test_checkpoint_requires_versioned_native_action_contract(monkeypatch, tmp_path, space, representation):
    from vla.actions import ActionSpec
    from vla.actions.statistics import SamplingSpec, compute_statistics
    monkeypatch.setattr('huggingface_hub.snapshot_download', lambda *a, **k: '/cached/tokenizer')
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', lambda *a, **k: object())
    spec = ActionSpec(space=space, representation=representation)
    root = _checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError):
        validate_checkpoint(root)
    raw = np.zeros((2, spec.dim))
    if space == 'tcp':
        raw[:, 3:9] = [1, 0, 0, 0, 1, 0]
    stats = compute_statistics([(raw, raw, np.ones(2, dtype=bool))], spec, SamplingSpec(50), 'fixture')
    config = json.loads((root/'config.json').read_text())
    config['action_contract'] = {'version': 2, 'action_spec': spec.to_dict(), 'statistics': stats,
                                 'state_fields': list(STATE_FIELDS)}
    config['output_features']['action']['shape'] = [spec.dim]
    (root/'config.json').write_text(json.dumps(config))
    assert validate_checkpoint(root).tokenizer == '/cached/tokenizer'
    config['use_relative_actions'] = True
    (root/'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='project relative processor'):
        validate_checkpoint(root)


def test_policy_observation_only_supplies_two_real_cameras() -> None:
    raw = {name: index / 10 for index, name in enumerate(STATE_FIELDS)}
    raw.update({name: np.zeros((224, 224, 3), dtype=np.uint8) for name in ("base", "left_wrist")})
    observation = policy_observation(raw, "task")
    assert set(observation) == {"observation.state", "task", *PHYSICAL_CAMERAS}
    assert not set(LEGACY_PADDING_CAMERAS) & set(observation)


def test_jsonl_debug_trace_writes_structured_events(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    trace = JsonlDebugTrace(path)
    trace.emit("control_step", step=7, predicted_xyz=[0.1, 0.2, 0.3], predicted_gripper=0.5)
    trace.close()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["event"] == "control_step"
    assert records[0]["step"] == 7
    assert records[0]["predicted_xyz"] == [0.1, 0.2, 0.3]
    assert records[-1]["event"] == "trace_closed"
    assert records[-1]["dropped_records"] == 0








def test_mock_hardware_provides_complete_observation_and_never_executes() -> None:
    robot = MockFrankaRobotiqRobot((0.2, -0.45, 0.05), (0.75, 0.45, 0.75))
    robot.connect()
    observation = policy_observation(robot.get_observation(), "task")
    np.testing.assert_allclose(observation["observation.state"][:3], [0.475, 0.0, 0.4])
    assert observation["observation.state"][9] == pytest.approx(0.0425)
    assert set(PHYSICAL_CAMERAS) <= set(observation)
    with pytest.raises(AssertionError, match="never execute"):
        robot.send_action(np.zeros(10))
    robot.disconnect()
    assert not robot.is_connected














def test_legacy_execution_loops_are_not_exposed():
    import vla.deployment.franka_pi05_rtc as deployment
    assert not hasattr(deployment, 'Pi05RTC')
    assert not hasattr(deployment, 'DeploymentLoop')
