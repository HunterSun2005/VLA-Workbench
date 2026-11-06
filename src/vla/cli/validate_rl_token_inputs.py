"""Validate a LeRobot dataset against the SFT checkpoint used by RL Token."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vla.adapters.lerobot.checkpoint_contract import fingerprint_sft_checkpoint, sha256_file
from vla.adapters.lerobot.policies.cache_manifest import atomic_write_json
from vla.cli.common import configure_logging

logger = configure_logging(__name__)


@dataclass(frozen=True)
class RLTokenInputSummary:
    dataset_root: Path
    sft_path: Path
    state_dim: int
    action_dim: int
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    camera_keys: tuple[str, ...]
    chunk_size: int
    use_relative_actions: bool
    action_stats: str | None
    action_chunk_size: int | None
    actions_stored_as: str | None
    tokenizer_path: str
    sft_checkpoint_contract: dict[str, Any]
    dataset_info_sha256: str
    dataset_stats_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check dataset/SFT compatibility before loading the RL Token model."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--sft-path", required=True)
    parser.add_argument(
        "--dataset-repo-id",
        default="local/rl_token_preflight",
        help="Local identifier passed to LeRobot metadata loading.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional local tokenizer override. The SFT processor value is used by default.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path for a machine-readable validated input contract.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if path.name == "info.json" and path.parent.name == "meta":
        from vla.dataset_metadata import load_info
        return load_info(path.parent.parent)
    if not path.is_file():
        raise FileNotFoundError(f"required file does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _feature_dim(features: dict[str, Any], key: str, label: str) -> int:
    feature = features.get(key)
    shape = feature.get("shape") if isinstance(feature, dict) else None
    if not isinstance(shape, list) or len(shape) != 1 or not isinstance(shape[0], int):
        raise ValueError(f"{label} {key!r} must have a one-dimensional shape, got {shape}")
    return int(shape[0])


def _feature_names(features: dict[str, Any], key: str, expected_dim: int) -> tuple[str, ...]:
    names = features[key].get("names")
    if not isinstance(names, list) or len(names) != expected_dim or not all(
        isinstance(name, str) and name for name in names
    ):
        raise ValueError(f"dataset feature {key!r} must have {expected_dim} ordered names")
    return tuple(names)


def _camera_keys(features: dict[str, Any]) -> tuple[str, ...]:
    prefix = "observation.images."
    return tuple(
        key.removeprefix(prefix)
        for key, feature in features.items()
        if key.startswith(prefix) and isinstance(feature, dict) and feature.get("dtype") in {"video", "image"}
    )


def _policy_camera_keys(features: dict[str, Any]) -> tuple[str, ...]:
    prefix = "observation.images."
    return tuple(key.removeprefix(prefix) for key in features if key.startswith(prefix))


def _processor_steps(processor: dict[str, Any]) -> list[dict[str, Any]]:
    steps = processor.get("steps")
    if not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps):
        raise ValueError("SFT policy_preprocessor.json has no valid steps list")
    return steps


def _tokenizer_from_processor(processor: dict[str, Any]) -> str:
    for step in _processor_steps(processor):
        if step.get("registry_name") == "tokenizer_processor":
            tokenizer = step.get("config", {}).get("tokenizer_name")
            if isinstance(tokenizer, str) and tokenizer:
                return tokenizer
    raise ValueError("SFT preprocessor has no tokenizer_processor with tokenizer_name")


def _relative_from_processor(processor: dict[str, Any]) -> bool | None:
    for step in _processor_steps(processor):
        if step.get("registry_name") in {"delta_actions_processor", "relative_actions_processor"}:
            return bool(step.get("config", {}).get("enabled", False))
    return None


def _validate_stats(stats: dict[str, Any], key: str, expected_dim: int) -> None:
    feature_stats = stats.get(key)
    if not isinstance(feature_stats, dict):
        raise ValueError(f"dataset stats are missing {key!r}")
    for quantile in ("q01", "q99"):
        value = feature_stats.get(quantile)
        if not isinstance(value, list) or len(value) != expected_dim:
            raise ValueError(
                f"dataset {key}.{quantile} must have {expected_dim} values, got "
                f"{len(value) if isinstance(value, list) else None}"
            )


def validate_rl_token_inputs(
    dataset_root: str | Path,
    sft_path: str | Path,
    *,
    dataset_repo_id: str = "local/rl_token_preflight",
    tokenizer_path: str | None = None,
) -> RLTokenInputSummary:
    dataset_root = Path(dataset_root).expanduser().resolve()
    sft_path = Path(sft_path).expanduser().resolve()
    info = _load_json(dataset_root / "meta/info.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"RL Token training requires a LeRobot v3.0 dataset, got {info.get('codebase_version')!r}"
        )
    dataset_features = info.get("features")
    if not isinstance(dataset_features, dict):
        raise ValueError("dataset info.json has no features mapping")
    state_dim = _feature_dim(dataset_features, "observation.state", "dataset feature")
    action_dim = _feature_dim(dataset_features, "action", "dataset feature")
    state_names = _feature_names(dataset_features, "observation.state", state_dim)
    action_names = _feature_names(dataset_features, "action", action_dim)
    cameras = _camera_keys(dataset_features)
    if not cameras:
        raise ValueError("dataset contains no observation.images.* features")

    config = _load_json(sft_path / "config.json")
    if config.get("type") != "pi05":
        raise ValueError(f"SFT checkpoint must be a pi05 policy, got type={config.get('type')!r}")
    sft_inputs = config.get("input_features")
    sft_outputs = config.get("output_features")
    if not isinstance(sft_inputs, dict) or not isinstance(sft_outputs, dict):
        raise ValueError("SFT config is missing input_features/output_features")
    sft_state_dim = _feature_dim(sft_inputs, "observation.state", "SFT input feature")
    sft_action_dim = _feature_dim(sft_outputs, "action", "SFT output feature")
    sft_cameras = _policy_camera_keys(sft_inputs)
    mismatches = []
    if state_dim != sft_state_dim:
        mismatches.append(f"state_dim dataset={state_dim}, SFT={sft_state_dim}")
    if action_dim != sft_action_dim:
        mismatches.append(f"action_dim dataset={action_dim}, SFT={sft_action_dim}")
    if cameras != sft_cameras:
        mismatches.append(f"camera order dataset={list(cameras)}, SFT={list(sft_cameras)}")
    if mismatches:
        raise ValueError("dataset/SFT schema mismatch: " + "; ".join(mismatches))

    chunk_size = int(config.get("chunk_size", 0))
    if chunk_size <= 0:
        raise ValueError(f"SFT chunk_size must be positive, got {chunk_size}")
    if state_dim > int(config.get("max_state_dim", 32)):
        raise ValueError(f"state_dim {state_dim} exceeds SFT max_state_dim")
    if action_dim > int(config.get("max_action_dim", 32)):
        raise ValueError(f"action_dim {action_dim} exceeds SFT max_action_dim")

    preprocessor = _load_json(sft_path / "policy_preprocessor.json")
    _load_json(sft_path / "policy_postprocessor.json")
    saved_tokenizer = _tokenizer_from_processor(preprocessor)
    resolved_tokenizer = tokenizer_path or saved_tokenizer
    tokenizer_candidate = Path(resolved_tokenizer).expanduser()
    if (tokenizer_path is not None or Path(saved_tokenizer).is_absolute()) and not tokenizer_candidate.exists():
        raise FileNotFoundError(f"tokenizer path does not exist: {tokenizer_candidate}")

    action_contract = config.get('action_contract')
    if action_contract:
        from vla.actions import ActionSpec
        from vla.actions.metadata import checkpoint_contract, validate_mixture
        contract = checkpoint_contract(sft_path)
        spec = ActionSpec.from_dict(contract['action_spec'])
        spec.require_compatible(ActionSpec.from_dict(info['action_spec']), training=False)
        if 'observation.anchor' not in dataset_features:
            raise ValueError('RLT training requires independent raw observation.anchor')
        if info.get('actions_stored_as', info.get('evo_rlt', {}).get('actions_stored_as')) != 'absolute':
            raise ValueError('RLT training requires absolute disk actions')
        sampling = contract['statistics']['contract']['sampling']
        if sampling['chunk_length'] != chunk_size:
            raise ValueError('RLT checkpoint chunk sampler mismatch')
        recording = info.get('recording_contract') or info.get('processing', {}).get('source_contract', {}).get('recording_contract')
        if recording:
            validate_mixture([recording], allow_derived=bool(contract.get('sources', {}).get('allow_derived', False)))
        use_relative_actions = spec.representation == 'chunk_relative'
        actions_stored_as = 'absolute'
        expected_stats = 'relative' if use_relative_actions else 'absolute'
    else:
        processor_relative = _relative_from_processor(preprocessor)
        config_relative = config.get("use_relative_actions")
        use_relative_actions = bool(config_relative) if config_relative is not None else bool(processor_relative)
        if processor_relative is not None and processor_relative != use_relative_actions:
            raise ValueError(
                "SFT config and preprocessor disagree on use_relative_actions: "
                f"config={use_relative_actions}, processor={processor_relative}"
            )
        vla_metadata = info.get("evo_rlt", {})
        if not isinstance(vla_metadata, dict):
            raise ValueError("dataset vla metadata must be a mapping")
        action_stats = vla_metadata.get("action_stats")
        expected_stats = "relative" if use_relative_actions else "absolute"
        if action_stats is not None and action_stats != expected_stats:
            raise ValueError(
                f"dataset action_stats={action_stats!r} does not match SFT use_relative_actions="
                f"{use_relative_actions}; expected {expected_stats!r}"
            )
        if action_stats is None:
            logger.warning(
                "dataset has no evo_rlt.action_stats marker; cannot prove %s stats match the SFT processor",
                expected_stats,
            )

        action_chunk_size = vla_metadata.get("action_chunk_size")
        if action_chunk_size is not None:
            if not isinstance(action_chunk_size, int) or action_chunk_size != chunk_size:
                raise ValueError(
                    f"dataset action_chunk_size={action_chunk_size!r} does not match "
                    f"SFT chunk_size={chunk_size}"
                )
        else:
            logger.warning(
                "dataset has no evo_rlt.action_chunk_size marker; cannot prove its action stats "
                "use the SFT chunk size"
            )

        actions_stored_as = vla_metadata.get("actions_stored_as")
        if actions_stored_as is not None and actions_stored_as != "absolute":
            raise ValueError(
                f"dataset actions_stored_as={actions_stored_as!r}; RL Token expects absolute "
                "actions in parquet and applies relative conversion in the SFT processor"
            )
        if actions_stored_as is None:
            logger.warning(
                "dataset has no evo_rlt.actions_stored_as marker; cannot prove parquet actions are absolute"
            )


    stats = _load_json(dataset_root / "meta/stats.json")
    _validate_stats(stats, "observation.state", state_dim)
    _validate_stats(stats, "action", action_dim)

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(repo_id=dataset_repo_id, root=dataset_root, revision="v3.0")
    if metadata.total_frames <= 0 or metadata.total_episodes <= 0:
        raise ValueError("dataset has no frames or episodes")

    return RLTokenInputSummary(
        dataset_root=dataset_root,
        sft_path=sft_path,
        state_dim=state_dim,
        action_dim=action_dim,
        state_names=state_names,
        action_names=action_names,
        camera_keys=cameras,
        chunk_size=chunk_size,
        use_relative_actions=use_relative_actions,
        action_stats=action_stats,
        action_chunk_size=action_chunk_size,
        actions_stored_as=actions_stored_as,
        tokenizer_path=resolved_tokenizer,
        sft_checkpoint_contract=fingerprint_sft_checkpoint(sft_path),
        dataset_info_sha256=sha256_file(dataset_root / "meta/info.json"),
        dataset_stats_sha256=sha256_file(dataset_root / "meta/stats.json"),
    )


def summary_to_dict(summary: RLTokenInputSummary) -> dict[str, Any]:
    return {
        "dataset_root": str(summary.dataset_root),
        "sft_path": str(summary.sft_path),
        "state_dim": summary.state_dim,
        "action_dim": summary.action_dim,
        "state_names": list(summary.state_names),
        "action_names": list(summary.action_names),
        "camera_keys": list(summary.camera_keys),
        "chunk_size": summary.chunk_size,
        "use_relative_actions": summary.use_relative_actions,
        "action_stats": summary.action_stats,
        "action_chunk_size": summary.action_chunk_size,
        "actions_stored_as": summary.actions_stored_as,
        "tokenizer_path": summary.tokenizer_path,
        "sft_checkpoint_contract": summary.sft_checkpoint_contract,
        "dataset_info_sha256": summary.dataset_info_sha256,
        "dataset_stats_sha256": summary.dataset_stats_sha256,
    }


def main() -> None:
    args = parse_args()
    summary = validate_rl_token_inputs(
        args.dataset_root,
        args.sft_path,
        dataset_repo_id=args.dataset_repo_id,
        tokenizer_path=args.tokenizer_path,
    )
    if args.json_output is not None:
        atomic_write_json(Path(args.json_output).expanduser().resolve(), summary_to_dict(summary))
    logger.info(
        "RL Token inputs valid: state=%d action=%d cameras=%s chunk=%d relative=%s tokenizer=%s",
        summary.state_dim,
        summary.action_dim,
        list(summary.camera_keys),
        summary.chunk_size,
        summary.use_relative_actions,
        summary.tokenizer_path,
    )
