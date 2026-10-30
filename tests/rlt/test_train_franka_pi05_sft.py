from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vla.cli.train_franka_pi05_sft import (
    install_pi05_loss_metrics_patch,
    consume_cli_option,
    rebuild_input_features_args,
    record_pi05_loss_metrics,
    validate_split_training_args,
    write_training_provenance,
)


def _output(loss: float, offset: float = 0.0) -> dict:
    return {"loss": loss, "loss_per_dim": [offset + index / 100 for index in range(10)]}


def _action_names() -> list[str]:
    return [f"action_{index}" for index in range(10)]


def test_pi05_loss_metrics_are_window_averaged_without_loss_overwrite() -> None:
    tracker = SimpleNamespace(metrics={})

    first = record_pi05_loss_metrics(tracker, _output(0.5, 0.0), _action_names())
    second = record_pi05_loss_metrics(tracker, _output(0.7, 0.2), _action_names())

    assert first == {"batch_loss": 0.5}
    assert second == {"batch_loss": 0.7}
    assert tracker.metrics["flow_loss/dim/action_0"].avg == pytest.approx(0.1)
    assert tracker.metrics["flow_loss/group/position"].avg == pytest.approx(0.11)
    assert tracker.metrics["flow_loss/group/rotation_6d"].avg == pytest.approx(0.155)
    assert tracker.metrics["flow_loss/group/gripper"].avg == pytest.approx(0.19)


def test_pi05_loss_metrics_reject_nonfinite_and_dimension_mismatch() -> None:
    tracker = SimpleNamespace(metrics={})
    with pytest.raises(ValueError, match="exactly 10"):
        record_pi05_loss_metrics(tracker, {"loss": 1.0, "loss_per_dim": [1.0]}, _action_names())
    with pytest.raises(ValueError, match="NaN or infinity"):
        record_pi05_loss_metrics(tracker, _output(float("nan")), _action_names())


def test_pi05_update_patch_is_idempotent_and_uses_unwrapped_config() -> None:
    tracker = SimpleNamespace(metrics={})
    policy = SimpleNamespace(config=SimpleNamespace(type="pi05", action_feature_names=_action_names()))
    accelerator = SimpleNamespace(unwrap_model=lambda value: value)
    module = SimpleNamespace(update_policy=lambda *args, **kwargs: (tracker, _output(0.5)))

    install_pi05_loss_metrics_patch(module)
    patched = module.update_policy
    install_pi05_loss_metrics_patch(module)

    result_tracker, output = module.update_policy(
        tracker, policy, {}, None, 1.0, accelerator=accelerator
    )
    assert module.update_policy is patched
    assert result_tracker is tracker
    assert output == {"batch_loss": 0.5}


def test_rebuild_input_features_clears_pretrained_camera_mapping() -> None:
    args = rebuild_input_features_args(
        [
            "--dataset.root=/dataset",
            '--policy.input_features={"observation.images.base":{"type":"VISUAL"}}',
            "--policy.empty_cameras=3",
            "--policy.path=/model",
        ]
    )
    assert args == [
        "--dataset.root=/dataset",
        "--policy.path=/model",
        "--policy.input_features=null",
        "--policy.empty_cameras=0",
    ]


def _split_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "split_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "franka_pi05_sft_split",
                "source": {"repo_id": "local/source", "root": str(tmp_path / "source")},
                "splits": {
                    "train": {
                        "repo_id": "local/source_train",
                        "root": str(tmp_path / "split/train"),
                        "original_episode_ids": [0, 1],
                        "total_frames": 100,
                    },
                    "val": {
                        "repo_id": "local/source_val",
                        "root": str(tmp_path / "split/val"),
                        "original_episode_ids": [2],
                        "total_frames": 50,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_split_manifest_option_is_consumed_and_training_contract_is_strict(tmp_path: Path) -> None:
    manifest_path = _split_manifest(tmp_path)
    args, value = consume_cli_option(
        ["--policy.path=/model", f"--split-manifest={manifest_path}"], "--split-manifest"
    )
    assert args == ["--policy.path=/model"]
    assert value == str(manifest_path)

    forwarded = [
        "--dataset.repo_id=local/source_train",
        f"--dataset.root={tmp_path / 'split/train'}",
        "--policy.use_relative_actions=false",
        f"--output_dir={tmp_path / 'output'}",
    ]
    manifest, output_dir = validate_split_training_args(forwarded, manifest_path)
    assert output_dir == (tmp_path / "output").resolve()
    (tmp_path / "output").mkdir()
    provenance = write_training_provenance(output_dir, manifest_path, manifest, forwarded)
    assert json.loads(provenance.read_text())["train_split"]["repo_id"] == "local/source_train"

    with pytest.raises(ValueError, match="root does not match"):
        validate_split_training_args(
            [arg.replace("split/train", "wrong") if "dataset.root" in arg else arg for arg in forwarded],
            manifest_path,
        )


def test_train_entrypoint_forces_dataset_inference(monkeypatch) -> None:
    import vla.adapters.lerobot as adapter
    import lerobot.scripts.lerobot_train as lerobot_train
    from vla.cli import train_franka_pi05_sft

    calls = []
    monkeypatch.setattr(adapter, "register", lambda: calls.append("register"))
    monkeypatch.setattr(
        lerobot_train,
        "main",
        lambda: calls.append(tuple(sys.argv[1:])),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vla-train-franka-pi05-sft",
            "--policy.path=/model",
            "--policy.input_features={}",
        ],
    )

    train_franka_pi05_sft.main()

    assert calls == [
        "register",
        (
            "--policy.path=/model",
            "--policy.input_features=null",
            "--policy.empty_cameras=0",
        ),
    ]


def test_null_override_clears_real_pi05_base_camera_keys() -> None:
    from pathlib import Path

    base = Path("models/pi05_base")
    if not (base / "config.json").is_file():
        pytest.skip("local PI0.5 base checkpoint is unavailable")

    import lerobot.policies.pi05.configuration_pi05  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig

    args = rebuild_input_features_args(["--policy.path=models/pi05_base"])
    overrides = [
        "--" + arg.removeprefix("--policy.")
        for arg in args
        if arg.startswith(("--policy.input_features=", "--policy.empty_cameras="))
    ]
    config = PreTrainedConfig.from_pretrained(base, cli_overrides=overrides)

    assert config.input_features is None
    assert config.empty_cameras == 0
