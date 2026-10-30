"""Validate a manifest-based transition cache before AC model loading."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vla.adapters.lerobot.checkpoint_contract import (
    assert_same_fingerprint,
    fingerprint_rl_token_checkpoint,
    fingerprint_sft_checkpoint,
    sha256_file,
)
from vla.adapters.lerobot.policies.cache_manifest import (
    atomic_write_json,
    load_cache_manifest,
)
from vla.cli.common import configure_logging

logger = configure_logging(__name__)


@dataclass(frozen=True)
class CacheValidationSummary:
    cache_dir: Path
    state_dim: int
    action_dim: int
    chunk_length: int
    fps: float
    train_transitions: int
    val_transitions: int
    episodes: int
    successful_episodes: int
    failed_episodes: int
    intervention_transitions: int
    source_counts: dict[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--sft-path", default=None, help="Optional relocated SFT checkpoint.")
    parser.add_argument(
        "--rl-token-path", default=None, help="Optional relocated RL Token checkpoint."
    )
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def _check_tensor(
    transition: dict[str, Any],
    key: str,
    shape: tuple[int, ...],
    *,
    shard_path: Path,
    index: int,
) -> torch.Tensor:
    value = transition.get(key)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(
            f"{shard_path} transition {index} {key} must have shape {shape}, got "
            f"{tuple(value.shape) if isinstance(value, torch.Tensor) else type(value)}"
        )
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"{shard_path} transition {index} {key} contains non-finite values")
    return value


def _check_optional_scalar(
    transition: dict[str, Any],
    key: str,
    default: float | int,
    *,
    shard_path: Path,
    index: int,
) -> torch.Tensor:
    if key not in transition:
        return torch.tensor(default)
    return _check_tensor(transition, key, (), shard_path=shard_path, index=index)


def validate_transition_cache(
    cache_dir: str | Path,
    *,
    sft_path: str | Path | None = None,
    rl_token_path: str | Path | None = None,
    verify_checkpoints: bool = True,
) -> CacheValidationSummary:
    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest = load_cache_manifest(cache_dir)
    if manifest.get("status") != "complete":
        raise ValueError("transition cache manifest is not complete")
    contract = manifest["contract"]
    dataset = contract["dataset"]
    policy = contract["policy"]
    cache = contract["cache"]
    action_dim = int(policy["action_dim"])
    proprio_dim = int(policy["proprio_dim"])
    rl_token_dim = int(policy["rl_token_dim"])
    chunk_length = int(cache["chunk_length"])
    state_dim = rl_token_dim + proprio_dim
    split_counts = {"train": 0, "val": 0}
    episode_count = 0
    successful_episodes = 0
    failed_episodes = 0
    intervention_transitions = 0
    source_counts: dict[int, int] = {}

    for split in ("train", "val"):
        for shard in manifest.get("shards", {}).get(split, []):
            shard_path = cache_dir / shard["path"]
            if not shard_path.is_file():
                raise FileNotFoundError(f"transition shard does not exist: {shard_path}")
            if sha256_file(shard_path) != shard.get("sha256"):
                raise ValueError(f"transition shard fingerprint mismatch: {shard_path}")
            transitions = torch.load(shard_path, weights_only=False, map_location="cpu")
            if not isinstance(transitions, list) or len(transitions) != int(shard["transitions"]):
                raise ValueError(f"transition shard count mismatch: {shard_path}")
            done_count = 0
            for index, transition in enumerate(transitions):
                if not isinstance(transition, dict):
                    raise ValueError(f"{shard_path} transition {index} is not a mapping")
                _check_tensor(transition, "state_vec", (state_dim,), shard_path=shard_path, index=index)
                _check_tensor(transition, "next_state_vec", (state_dim,), shard_path=shard_path, index=index)
                for key in ("exec_chunk", "ref_chunk", "next_ref_chunk"):
                    _check_tensor(
                        transition,
                        key,
                        (chunk_length, action_dim),
                        shard_path=shard_path,
                        index=index,
                    )
                reward = _check_tensor(
                    transition,
                    "reward_seq",
                    (chunk_length,),
                    shard_path=shard_path,
                    index=index,
                )
                done = _check_tensor(transition, "done", (), shard_path=shard_path, index=index)
                actual_steps = _check_tensor(
                    transition, "actual_steps", (), shard_path=shard_path, index=index
                )
                intervention = _check_optional_scalar(
                    transition, "intervention", 0.0, shard_path=shard_path, index=index
                )
                source = _check_optional_scalar(
                    transition, "source", 0, shard_path=shard_path, index=index
                )
                _check_optional_scalar(
                    transition, "is_critical", 1.0, shard_path=shard_path, index=index
                )
                if int(actual_steps.item()) != chunk_length:
                    raise ValueError(f"{shard_path} transition {index} has invalid actual_steps")
                if float(done.item()) not in (0.0, 1.0):
                    raise ValueError(f"{shard_path} transition {index} has invalid done value")
                done_count += int(done.item())
                intervention_transitions += int(float(intervention.item()) > 0.5)
                source_id = int(source.item())
                source_counts[source_id] = source_counts.get(source_id, 0) + 1
                if not done.item() and reward.abs().sum().item() != 0:
                    raise ValueError(f"{shard_path} non-terminal transition {index} has reward")
            if done_count != 1:
                raise ValueError(f"{shard_path} must contain exactly one terminal transition")
            terminal = transitions[-1]
            if terminal["done"].item() != 1:
                raise ValueError(f"{shard_path} terminal transition must be last")
            episode_success = shard.get("episode_success")
            terminal_reward = terminal["reward_seq"]
            if episode_success is True and terminal_reward[-1].item() != 1:
                raise ValueError(f"{shard_path} successful episode has no terminal reward")
            if episode_success is False and terminal_reward.abs().sum().item() != 0:
                raise ValueError(f"{shard_path} failed episode has nonzero reward")
            successful_episodes += int(episode_success is True)
            failed_episodes += int(episode_success is False)
            split_counts[split] += len(transitions)
            episode_count += 1

    if split_counts["train"] <= 0:
        raise ValueError("transition cache has no train transitions")

    if verify_checkpoints:
        actual_sft_path = Path(sft_path or policy["sft_path"]).expanduser().resolve()
        actual_rl_path = Path(rl_token_path or policy["rl_token_path"]).expanduser().resolve()
        assert_same_fingerprint(
            policy.get("sft_contract"),
            fingerprint_sft_checkpoint(actual_sft_path),
            label="SFT",
        )
        assert_same_fingerprint(
            policy.get("rl_token_contract"),
            fingerprint_rl_token_checkpoint(actual_rl_path),
            label="RL Token",
        )

    return CacheValidationSummary(
        cache_dir=cache_dir,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_length=chunk_length,
        fps=float(dataset["fps"]),
        train_transitions=split_counts["train"],
        val_transitions=split_counts["val"],
        episodes=episode_count,
        successful_episodes=successful_episodes,
        failed_episodes=failed_episodes,
        intervention_transitions=intervention_transitions,
        source_counts=source_counts,
    )


def summary_to_dict(summary: CacheValidationSummary) -> dict[str, Any]:
    return {
        "cache_dir": str(summary.cache_dir),
        "state_dim": summary.state_dim,
        "action_dim": summary.action_dim,
        "chunk_length": summary.chunk_length,
        "fps": summary.fps,
        "train_transitions": summary.train_transitions,
        "val_transitions": summary.val_transitions,
        "episodes": summary.episodes,
        "successful_episodes": summary.successful_episodes,
        "failed_episodes": summary.failed_episodes,
        "intervention_transitions": summary.intervention_transitions,
        "source_counts": summary.source_counts,
    }


def main() -> None:
    args = parse_args()
    summary = validate_transition_cache(
        args.cache_dir,
        sft_path=args.sft_path,
        rl_token_path=args.rl_token_path,
    )
    if args.json_output:
        atomic_write_json(Path(args.json_output).expanduser().resolve(), summary_to_dict(summary))
    logger.info(
        "transition cache valid: state=%d action=%d chunk=%d fps=%.3f train=%d val=%d "
        "episodes=%d success=%d failure=%d intervention=%d sources=%s",
        summary.state_dim,
        summary.action_dim,
        summary.chunk_length,
        summary.fps,
        summary.train_transitions,
        summary.val_transitions,
        summary.episodes,
        summary.successful_episodes,
        summary.failed_episodes,
        summary.intervention_transitions,
        summary.source_counts,
    )


if __name__ == "__main__":
    main()
