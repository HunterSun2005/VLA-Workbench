"""Train single/dual Franka PI0.5 with inputs rebuilt from the LeRobot dataset."""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from functools import wraps
from pathlib import Path
from typing import Any

from vla.diagnostics.franka_sft_split import (
    load_split_manifest,
    split_manifest_digest,
    training_schedule,
    validate_split_dataset_contract,
    write_json_atomic,
)


INPUT_FEATURES_PREFIX = "--policy.input_features="
EMPTY_CAMERAS_PREFIX = "--policy.empty_cameras="
SPLIT_MANIFEST_OPTION = "--split-manifest"
SUPPORTED_ACTION_DIMS = (8, 10, 16, 20)


def _add_average_meter(tracker: Any, key: str, display_name: str, value: float) -> None:
    from lerobot.utils.logging_utils import AverageMeter

    if key not in tracker.metrics:
        tracker.metrics[key] = AverageMeter(display_name, ":.4f")
    tracker.metrics[key].update(value)


def record_pi05_loss_metrics(
    train_tracker: Any,
    output_dict: dict[str, Any] | None,
    action_names: Sequence[str],
) -> dict[str, Any]:
    """Move PI0.5 batch outputs into scalar, window-averaged metrics."""
    if output_dict is None:
        raise ValueError("PI0.5 forward returned no loss metrics")

    output = dict(output_dict)
    loss_per_dim = output.pop("loss_per_dim", None)
    if loss_per_dim is None:
        raise ValueError("PI0.5 forward output is missing loss_per_dim")
    action_dim = len(action_names)
    if action_dim not in SUPPORTED_ACTION_DIMS or len(loss_per_dim) != action_dim:
        raise ValueError(
            "Franka PI0.5 loss metrics require exactly 10 or 20 named action dimensions; "
            f"got names={len(action_names)}, losses={len(loss_per_dim)}"
        )
    if len(set(action_names)) != action_dim or not all(
        isinstance(name, str) and name for name in action_names
    ):
        raise ValueError("PI0.5 action feature names must be unique non-empty strings")

    values = [float(value) for value in loss_per_dim]
    batch_loss = float(output.pop("loss"))
    if not all(math.isfinite(value) for value in (*values, batch_loss)):
        raise ValueError("PI0.5 loss metrics contain NaN or infinity")

    for index, (name, value) in enumerate(zip(action_names, values, strict=True)):
        _add_average_meter(
            train_tracker,
            f"flow_loss/dim/{name}",
            f"fl_d{index}",
            value,
        )

    width = 8 if action_dim in (8, 16) else 10
    per_arm = [
        {"position": sum(values[i:i+3]) / 3,
         "rotation_6d": sum(values[i+3:i+9]) / 6,
         "gripper": values[i+9]} if width == 10 else
        {"joint": sum(values[i:i+7])/7, "gripper": values[i+7]}
        for i in range(0, action_dim, width)
    ]
    groups = {key: sum(arm[key] for arm in per_arm) / len(per_arm) for key in per_arm[0]}
    if action_dim == 2 * width:
        groups.update({f"{side}/{key}": value
                       for side, arm in zip(("left", "right"), per_arm, strict=True)
                       for key, value in arm.items()})
    for name, value in groups.items():
        _add_average_meter(train_tracker, f"flow_loss/group/{name}", f"fl_{name}", value)

    output["batch_loss"] = batch_loss
    return output


def install_pi05_loss_metrics_patch(lerobot_train_module: Any) -> None:
    """Patch LeRobot's update hook in-process without changing site-packages."""
    original = lerobot_train_module.update_policy
    if getattr(original, "_vla_pi05_loss_metrics", False):
        return

    @wraps(original)
    def update_policy_with_metrics(*args: Any, **kwargs: Any):
        train_tracker, output_dict = original(*args, **kwargs)
        policy = args[1] if len(args) > 1 else kwargs["policy"]
        accelerator = kwargs.get("accelerator")
        if accelerator is None and len(args) > 5:
            accelerator = args[5]
        unwrapped = accelerator.unwrap_model(policy) if accelerator is not None else policy
        policy_type = getattr(getattr(unwrapped, "config", None), "type", None)
        if policy_type not in ("pi05", "vla_pi05"):
            raise ValueError(f"Franka PI0.5 training entrypoint received policy type {policy_type!r}")
        action_names = getattr(unwrapped.config, "action_feature_names", None)
        if action_names is None:
            raise ValueError("PI0.5 policy config is missing action_feature_names")
        return train_tracker, record_pi05_loss_metrics(train_tracker, output_dict, action_names)

    update_policy_with_metrics._vla_pi05_loss_metrics = True
    lerobot_train_module.update_policy = update_policy_with_metrics


def rebuild_input_features_args(args: Sequence[str]) -> list[str]:
    """Clear pretrained camera keys so LeRobot infers exactly the dataset inputs."""
    filtered = [
        arg
        for arg in args
        if not arg.startswith((INPUT_FEATURES_PREFIX, EMPTY_CAMERAS_PREFIX))
    ]
    filtered.extend((INPUT_FEATURES_PREFIX + "null", EMPTY_CAMERAS_PREFIX + "0"))
    return filtered


def consume_cli_option(args: Sequence[str], name: str) -> tuple[list[str], str | None]:
    """Remove one custom option before forwarding arguments to Draccus."""
    filtered: list[str] = []
    value: str | None = None
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == name:
            if value is not None:
                raise ValueError(f"{name} may only be provided once")
            if index + 1 >= len(args) or str(args[index + 1]).startswith("--"):
                raise ValueError(f"{name} requires a path")
            value = str(args[index + 1])
            index += 2
            continue
        prefix = name + "="
        if arg.startswith(prefix):
            if value is not None:
                raise ValueError(f"{name} may only be provided once")
            value = arg.removeprefix(prefix)
            if not value:
                raise ValueError(f"{name} requires a path")
            index += 1
            continue
        filtered.append(arg)
        index += 1
    return filtered, value


def _argument_value(args: Sequence[str], name: str) -> str | None:
    prefix = name + "="
    matches = [arg.removeprefix(prefix) for arg in args if arg.startswith(prefix)]
    if len(matches) > 1:
        raise ValueError(f"{name} may only be provided once")
    return matches[0] if matches else None


def validate_split_training_args(
    args: Sequence[str], manifest_path: Path
) -> tuple[dict[str, Any], Path]:
    manifest = load_split_manifest(manifest_path)
    dataset_repo_id = _argument_value(args, "--dataset.repo_id")
    dataset_root = _argument_value(args, "--dataset.root")
    output_dir = _argument_value(args, "--output_dir")
    if dataset_repo_id is None or dataset_root is None or output_dir is None:
        raise ValueError(
            "split-manifest training requires explicit --dataset.repo_id, --dataset.root, and --output_dir"
        )
    validate_split_dataset_contract(
        manifest,
        "train",
        dataset_repo_id=dataset_repo_id,
        dataset_root=Path(dataset_root),
    )
    if any(arg.startswith("--dataset.episodes=") for arg in args):
        raise ValueError("materialized split training must not also set --dataset.episodes")
    relative = _argument_value(args, "--policy.use_relative_actions")
    if relative and relative.lower() != "false":
        raise ValueError("use the VLA action contract for chunk_relative; generic rot6d subtraction is unsupported")
    return manifest, Path(output_dir).expanduser().resolve()


def write_training_provenance(
    output_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    forwarded_args: Sequence[str],
) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"training completed without creating output_dir: {output_dir}")
    provenance = {
        "schema_version": 1,
        "kind": "franka_pi05_sft_training_provenance",
        "split_manifest": str(Path(manifest_path).expanduser().resolve()),
        "split_manifest_sha256": split_manifest_digest(manifest),
        "source": manifest["source"],
        "train_split": manifest["splits"]["train"],
        "validation_split": manifest["splits"]["val"],
        "recommended_schedule": training_schedule(manifest),
        "forwarded_arguments": list(forwarded_args),
    }
    path = output_dir / "sft_run_provenance.json"
    write_json_atomic(path, provenance)
    return path


def main() -> None:
    from vla.adapters.lerobot import register

    register()
    forwarded_args, split_manifest_value = consume_cli_option(sys.argv[1:], SPLIT_MANIFEST_OPTION)
    forwarded_args = rebuild_input_features_args(forwarded_args)
    split_contract = None
    output_dir = None
    split_manifest_path = None
    if split_manifest_value is not None:
        split_manifest_path = Path(split_manifest_value).expanduser().resolve()
        split_contract, output_dir = validate_split_training_args(forwarded_args, split_manifest_path)
    if not any(arg.startswith("--dataset.eval_split=") for arg in forwarded_args):
        forwarded_args.append("--dataset.eval_split=0")
    elif "--dataset.eval_split=0" not in forwarded_args and "--dataset.eval_split=0.0" not in forwarded_args:
        raise ValueError("Franka SFT uses external splits; set dataset.eval_split=0")
    sys.argv[1:] = forwarded_args

    from lerobot.scripts import lerobot_train

    from vla.adapters.lerobot.action_training import install_action_training
    install_action_training(lerobot_train)
    install_pi05_loss_metrics_patch(lerobot_train)
    lerobot_train.main()
    if split_contract is not None and output_dir is not None and split_manifest_path is not None:
        write_training_provenance(output_dir, split_manifest_path, split_contract, forwarded_args)


if __name__ == "__main__":
    main()
