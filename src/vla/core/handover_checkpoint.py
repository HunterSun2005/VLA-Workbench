from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from vla.adapters.lerobot.checkpoint_contract import assert_same_fingerprint
from vla.core.phase_controller import HandoverClassifier


HANDOVER_FORMAT_VERSION = 1
HANDOVER_CONFIG_NAME = "handover_config.json"
HANDOVER_MODEL_NAME = "model.safetensors"
HANDOVER_METRICS_NAME = "metrics.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def save_handover_checkpoint(
    output_dir: str | Path,
    *,
    classifier: HandoverClassifier,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"handover output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        state = {
            key: value.detach().contiguous().cpu()
            for key, value in classifier.state_dict().items()
        }
        save_file(state, temporary / HANDOVER_MODEL_NAME)
        (temporary / HANDOVER_CONFIG_NAME).write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / HANDOVER_METRICS_NAME).write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except Exception:
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
        raise
    return output


def load_handover_classifier(
    checkpoint_dir: str | Path,
    *,
    expected_input_dim: int,
    expected_rl_token_contract: dict[str, Any],
    device: str | torch.device,
) -> tuple[HandoverClassifier, dict[str, Any]]:
    root = Path(checkpoint_dir).expanduser().resolve()
    config_path = root / HANDOVER_CONFIG_NAME
    model_path = root / HANDOVER_MODEL_NAME
    if not config_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"incomplete handover checkpoint: {root}")
    config = _load_json(config_path)
    if config.get("format_version") != HANDOVER_FORMAT_VERSION:
        raise ValueError(
            f"unsupported handover format {config.get('format_version')!r}; "
            f"expected {HANDOVER_FORMAT_VERSION}"
        )
    if config.get("input_kind") != "z_rl":
        raise ValueError(f"handover checkpoint input must be 'z_rl': {root}")
    input_dim = int(config.get("input_dim", -1))
    if input_dim != expected_input_dim:
        raise ValueError(
            f"handover input mismatch: checkpoint={input_dim}, policy={expected_input_dim}"
        )
    assert_same_fingerprint(
        config.get("rl_token_contract"),
        expected_rl_token_contract,
        label="Handover RL Token",
    )
    hidden_dim = int(config.get("hidden_dim", 0))
    if hidden_dim <= 0:
        raise ValueError(f"invalid handover hidden_dim={hidden_dim}")
    classifier = HandoverClassifier(input_dim=input_dim, hidden_dim=hidden_dim)
    classifier.load_state_dict(load_file(model_path, device=str(device)), strict=True)
    classifier.to(device).eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    return classifier, config
