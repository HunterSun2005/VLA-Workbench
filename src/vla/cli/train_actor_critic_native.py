"""Run LeRobot's native trainer with the Chunk Actor-Critic policy registered."""

from __future__ import annotations

import functools
import inspect
import json
from pathlib import Path
import sys


def install_cache_guard(trainer, *, accept_upgrade=False) -> None:
    original = getattr(trainer.make_dataset, "_vla_original", trainer.make_dataset)

    @functools.wraps(original)
    def make_dataset(cfg):
        from vla.adapters.lerobot.policies.configuration_rlt_ac import ChunkACPolicyConfig
        from vla.adapters.lerobot.policies.cache_manifest import load_cache_manifest
        from vla.core.training_contract import validate_cache_upgrade, validate_training_cache
        from vla.core.online_contract import durable_json

        if isinstance(cfg.policy, ChunkACPolicyConfig):
            manifest_path = Path(cfg.dataset.repo_id).expanduser().resolve() / "cache_manifest.json"
            manifest = load_cache_manifest(manifest_path)
            if manifest.get("status") != "complete":
                raise ValueError("A/C training requires a complete transition cache")
            current = manifest["contract"]
            validate_training_cache(current)
            previous = cfg.policy.cache_contract
            checkpoint = cfg.policy.pretrained_path
            if checkpoint:
                config_path = Path(checkpoint) / "config.json"
                if not config_path.is_file():
                    raise ValueError("A/C resume requires a local checkpoint with its cache contract")
                previous = json.loads(config_path.read_text()).get("cache_contract")
                if previous is None:
                    raise ValueError("checkpoint has no verifiable cache contract; cannot upgrade safely")
            if previous is not None and previous != current:
                if not accept_upgrade:
                    raise ValueError("use --accept-training-semantics-upgrade with a rebuilt cache")
                validate_cache_upgrade(previous, current)
                if checkpoint:
                    source, destination = Path(checkpoint).resolve(), Path(cfg.output_dir).resolve()
                    if source.is_relative_to(destination) or destination.is_relative_to(source):
                        raise ValueError("training semantics upgrade requires a separate output directory")
                report = {"checkpoint": str(checkpoint), "previous_contract": previous, "contract": current}
                report_path = Path(cfg.output_dir) / "training_semantics_upgrade.json"
                if report_path.exists():
                    if json.loads(report_path.read_text()) != report:
                        raise ValueError("output directory contains a different training upgrade")
                else:
                    durable_json(report_path, report)
            cfg.policy.cache_contract = current
            cfg.policy.cache_manifest_path = str(manifest_path)
            cfg.policy._apply_cache_contract()
        return original(cfg)

    make_dataset._vla_original = original
    trainer.make_dataset = make_dataset


def install_ac_update_hook(trainer) -> None:
    original = trainer.update_policy
    if getattr(original, "_vla_ac_update", False):
        return
    signature = inspect.signature(original)

    @functools.wraps(original)
    def update_policy(*args, **kwargs):
        from vla.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy

        arguments = signature.bind(*args, **kwargs).arguments
        accelerator = arguments["accelerator"]
        policy = accelerator.unwrap_model(arguments["policy"], keep_fp32_wrapper=True)
        result = original(*args, **kwargs)
        if isinstance(policy, ChunkACPolicy) and not accelerator.optimizer_step_was_skipped:
            policy.on_optimizer_step()
            if result[1] is not None:
                result[1]["critic_step"] = int(policy._critic_step.item())
        return result

    update_policy._vla_ac_update = True
    trainer.update_policy = update_policy


def main() -> None:
    from vla.adapters.lerobot import register

    register()

    from lerobot.scripts import lerobot_train

    install_ac_update_hook(lerobot_train)
    flag = "--accept-training-semantics-upgrade"
    accept_upgrade = flag in sys.argv
    sys.argv[:] = [arg for arg in sys.argv if arg != flag]
    install_cache_guard(lerobot_train, accept_upgrade=accept_upgrade)
    lerobot_train.main()


if __name__ == "__main__":
    main()
