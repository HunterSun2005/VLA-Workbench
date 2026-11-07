from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vla.control.geometry import (
    ACTION_FIELDS,
    STATE_FIELDS,
    _rotation_angle,
    rot6d_to_matrix,
)


log = logging.getLogger(__name__)


PHYSICAL_CAMERAS = ("observation.images.base", "observation.images.left_wrist")
LEGACY_PADDING_CAMERAS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)


class JsonlDebugTrace:
    """Non-blocking JSONL trace writer for hardware deployment diagnostics."""

    def __init__(self, path: str | Path, max_pending: int = 4096) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max_pending)
        self._dropped = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._writer, name="franka-debug-trace", daemon=True)
        self._thread.start()

    def emit(self, event: str, **fields: Any) -> None:
        record = {"event": event, "wall_time": time.time(), "monotonic": time.perf_counter(), **fields}
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped += 1

    def _writer(self) -> None:
        try:
            with self.path.open("a", encoding="utf-8", buffering=1) as stream:
                while True:
                    record = self._queue.get()
                    if record is None:
                        break
                    stream.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
        except BaseException as error:
            self._error = error

    def close(self) -> None:
        if not self._thread.is_alive():
            return
        self.emit("trace_closed", dropped_records=self._dropped)
        try:
            self._queue.put(None, timeout=5.0)
        except queue.Full:
            log.warning("Debug trace queue did not drain within 5 seconds: %s", self.path)
            return
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            log.warning("Debug trace writer did not stop within 5 seconds: %s", self.path)
        if self._error is not None:
            log.warning("Debug trace writer failed: %s", self._error)


@dataclass(frozen=True)
class CheckpointContract:
    path: Path
    config: dict[str, Any]
    tokenizer: str


def _feature_shape(config: dict[str, Any], group: str, key: str) -> tuple[int, ...]:
    try:
        return tuple(config[group][key]["shape"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"checkpoint is missing {group}.{key}.shape") from error


def _tokenizer_name(checkpoint: Path) -> str:
    processor = json.loads((checkpoint / "policy_preprocessor.json").read_text(encoding="utf-8"))
    for step in processor.get("steps", []):
        if step.get("registry_name") == "tokenizer_processor":
            name = step.get("config", {}).get("tokenizer_name")
            if isinstance(name, str) and name:
                return name
    raise ValueError("checkpoint preprocessor has no tokenizer_processor tokenizer_name")


def _validate_action_unnormalizer(checkpoint: Path) -> None:
    from safetensors.torch import load_file

    path = checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is missing action unnormalizer stats: {path}")
    tensors = load_file(path)
    try:
        minimum = tensors["action.min"].detach().cpu().numpy()
        maximum = tensors["action.max"].detach().cpu().numpy()
        q01 = tensors["action.q01"].detach().cpu().numpy()
        q99 = tensors["action.q99"].detach().cpu().numpy()
    except KeyError as error:
        raise ValueError(f"action unnormalizer is missing {error.args[0]!r}: {path}") from error
    if any(value.shape != (10,) for value in (minimum, maximum, q01, q99)):
        raise ValueError(f"action unnormalizer stats must be 10D: {path}")
    translation_span = float(np.linalg.norm(maximum[:3] - minimum[:3]))
    rot6d_span = float(np.linalg.norm(maximum[3:9] - minimum[3:9]))
    quantile_span = float(np.linalg.norm(q99[:9] - q01[:9]))
    if translation_span < 5e-3 and rot6d_span < 1e-2 and quantile_span < 1e-2:
        raise ValueError(
            "checkpoint action unnormalizer has degenerate constant TCP stats; "
            "the checkpoint was trained on invalid actions and cannot control the arm"
        )


def validate_checkpoint(checkpoint: str | Path, tokenizer_path: str | None = None) -> CheckpointContract:
    root = Path(checkpoint).expanduser().resolve()
    for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json", "model.safetensors"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"checkpoint is missing {name}: {root}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    from vla.actions import ActionSpec
    from vla.actions.metadata import checkpoint_contract
    contract = checkpoint_contract(root)
    spec = ActionSpec.from_dict(contract['action_spec'])
    sampling = contract['statistics']['contract']['sampling']
    expected_cameras = ('observation.images.base', 'observation.images.left_wrist') + (
        ('observation.images.right_wrist',) if len(spec.arms) == 2 else ())
    visual_features = tuple(key for key, feature in config.get('input_features', {}).items()
                            if feature.get('type') == 'VISUAL')
    checks = {
        'PI05 policy': config.get('type') in ('pi05', 'vla_pi05'),
        'state fields': _feature_shape(config, 'input_features', 'observation.state') == (len(contract['state_fields']),),
        'action dimensions': _feature_shape(config, 'output_features', 'action') == (spec.dim,),
        'project relative processor': config.get('use_relative_actions') is False,
        'chunk sampling': config.get('chunk_size') == sampling['chunk_length'],
        'physical cameras': visual_features == expected_cameras,
        'no empty cameras': config.get('empty_cameras', 0) == 0,
        'camera shapes': all(_feature_shape(config, 'input_features', key) == (3, 224, 224)
                             for key in expected_cameras),
    }
    if failed := [name for name, valid in checks.items() if not valid]:
        raise ValueError(f'checkpoint is incompatible with ROS deployment: {failed}')
    tokenizer = tokenizer_path or _tokenizer_name(root)
    tokenizer_root = Path(tokenizer).expanduser()
    if tokenizer_root.is_dir():
        tokenizer = str(tokenizer_root.resolve())
    else:
        from huggingface_hub import snapshot_download

        try:
            tokenizer = snapshot_download(tokenizer, local_files_only=True)
        except Exception as error:
            raise FileNotFoundError(
                f"tokenizer {tokenizer!r} is neither an explicit local snapshot "
                "nor available in offline cache"
            ) from error
    from transformers import AutoTokenizer

    try:
        AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
    except Exception as error:
        raise FileNotFoundError(f"could not load tokenizer from local snapshot {tokenizer!r}") from error
    return CheckpointContract(root, config, tokenizer)


def policy_observation(robot_observation: dict[str, Any], task: str) -> dict[str, Any]:
    state = np.asarray([robot_observation[name] for name in STATE_FIELDS], dtype=np.float32)
    if state.shape != (10,) or not np.isfinite(state).all():
        raise ValueError("robot observation did not produce a finite 10D state")
    result: dict[str, Any] = {"observation.state": state, "task": task}
    for role in ("base", "left_wrist"):
        image = np.asarray(robot_observation[role])
        if image.shape != (224, 224, 3) or image.dtype != np.uint8:
            raise ValueError(f"{role} must be 224x224 HWC uint8 RGB")
        result[f"observation.images.{role}"] = image
    return result


class MockFrankaRobotiqRobot:
    """Deterministic hardware replacement for a full disarmed policy smoke test."""

    def __init__(self, workspace_min: tuple[float, ...], workspace_max: tuple[float, ...]) -> None:
        center = (np.asarray(workspace_min) + np.asarray(workspace_max)) / 2.0
        self._state = np.concatenate((center, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0425]))
        self.is_connected = False
        self.stop_calls = 0

    def connect(self) -> None:
        self.is_connected = True

    def check_health(self) -> None:
        if not self.is_connected:
            raise RuntimeError("mock robot is disconnected")

    def get_observation(self) -> dict[str, Any]:
        self.check_health()
        observation: dict[str, Any] = {
            name: float(value) for name, value in zip(STATE_FIELDS, self._state)
        }
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        image[..., 1] = 64
        observation["base"] = image
        observation["left_wrist"] = np.flip(image, axis=1).copy()
        return observation

    def send_action(self, action) -> None:
        raise AssertionError("mock deployment must never execute actions")

    def stop(self) -> None:
        self.stop_calls += 1

    def disconnect(self) -> None:
        self.stop()
        self.is_connected = False
