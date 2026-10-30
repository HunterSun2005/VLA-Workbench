"""Build a resumable z_rl-only cache for training the handover classifier."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vla.adapters.lerobot.checkpoint_contract import sha256_file
from vla.adapters.lerobot.policies.cache_manifest import (
    CACHE_MANIFEST_NAME,
    atomic_write_json,
)
from vla.adapters.lerobot.policies.processor_rlt_token import (
    make_rlt_token_pre_post_processors,
)
from vla.cli.build_transition_cache_v2 import (
    FRAME_PHASE_KEY,
    _episode_value,
    _load_json,
    _load_policy,
    _prepare_manifest,
    split_episode_ids,
)
from vla.core.phase_annotations import (
    load_phase_annotations,
    overlay_episode_phase_annotations,
)


HANDOVER_CACHE_KIND = "handover_z_rl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--rl-token-policy-path", required=True)
    parser.add_argument(
        "--sft-path",
        default=None,
        help="Optional relocated SFT checkpoint; content must match the RL Token contract.",
    )
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--phase-annotations",
        default=None,
        help="Optional reviewed phase sidecar; overrides dataset complementary_info.phase.",
    )
    parser.add_argument("--task-instruction", default=None)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=0,
        help="Call torch.cuda.empty_cache every N batches; 0 disables it.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _phase_scalar(value: Any, *, frame_index: int) -> float:
    if isinstance(value, Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"phase at frame {frame_index} must contain one value")
        value = value[0]
    result = float(value)
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"non-finite phase at frame {frame_index}")
    return float(result > 0.5)


def load_episode_phase_labels(
    dataset: LeRobotDataset,
    episode_start: int,
    episode_stop: int,
) -> dict[int, dict[str, float]] | None:
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is None or FRAME_PHASE_KEY not in set(getattr(hf_dataset, "column_names", [])):
        return None
    selected = hf_dataset.select(range(episode_start, episode_stop))
    return {
        frame_index: {
            "phase": _phase_scalar(selected[FRAME_PHASE_KEY][local_index], frame_index=frame_index)
        }
        for local_index, frame_index in enumerate(range(episode_start, episode_stop))
    }


def _encode_episode(
    *,
    policy: Any,
    preprocessor: Any,
    dataset: LeRobotDataset,
    frame_indices: list[int],
    episode_start: int,
    episode_id: int,
    frame_annotations: dict[int, dict[str, float]],
    batch_size: int,
    num_workers: int,
    device: str,
    empty_cache_every: int,
    task_override: str | None,
) -> list[dict[str, Tensor]]:
    loader = DataLoader(
        Subset(dataset, frame_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=False,
    )
    samples: list[dict[str, Tensor]] = []
    offset = 0
    for batch_index, batch in enumerate(loader):
        batch_count = int(batch["observation.state"].shape[0])
        raw_indices = frame_indices[offset : offset + batch_count]
        offset += batch_count
        if task_override is not None:
            batch["task"] = [task_override] * batch_count
        elif "task" not in batch:
            raise ValueError("dataset batch has no task text; pass --task-instruction")
        processed = preprocessor(batch)
        with torch.no_grad():
            z_rl = policy.encode_observation(processed).detach().to(dtype=torch.float32).cpu()
        if z_rl.shape != (batch_count, policy.config.rl_token_dim):
            raise ValueError(
                f"RL Token output must have shape {(batch_count, policy.config.rl_token_dim)}, "
                f"got {tuple(z_rl.shape)}"
            )
        for local_index, raw_index in enumerate(raw_indices):
            annotation = frame_annotations.get(raw_index)
            if annotation is None or "phase" not in annotation:
                raise ValueError(f"missing phase label for episode {episode_id} frame {raw_index}")
            samples.append(
                {
                    "z_rl": z_rl[local_index],
                    "is_critical": torch.tensor(float(annotation["phase"])),
                    "episode_id": torch.tensor(episode_id, dtype=torch.int64),
                    "frame_index": torch.tensor(raw_index - episode_start, dtype=torch.int64),
                }
            )
        if (
            device.startswith("cuda")
            and empty_cache_every > 0
            and (batch_index + 1) % empty_cache_every == 0
        ):
            torch.cuda.empty_cache()
    if offset != len(frame_indices):
        raise ValueError(f"encoded {offset} handover frames, expected {len(frame_indices)}")
    return samples


def _atomic_torch_save(value: Any, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _build_contract(
    *,
    args: argparse.Namespace,
    config: Any,
    dataset_info: dict[str, Any],
    episode_splits: dict[str, list[int]],
    sft_contract: dict[str, Any],
    rl_token_contract: dict[str, Any],
    phase_annotations_path: pathlib.Path | None,
) -> dict[str, Any]:
    features = dataset_info["features"]
    state = features["observation.state"]
    cameras = [
        key.removeprefix("observation.images.")
        for key in features
        if key.startswith("observation.images.")
    ]
    if int(state["shape"][0]) != config.proprio_dim:
        raise ValueError(
            f"dataset state_dim={state['shape'][0]} does not match RL Token {config.proprio_dim}"
        )
    if cameras != list(config.camera_keys):
        raise ValueError(
            f"dataset camera order {cameras} does not match RL Token {list(config.camera_keys)}"
        )
    dataset_root = pathlib.Path(args.dataset_root).expanduser().resolve()
    return {
        "dataset": {
            "repo_id": args.dataset_repo_id,
            "root": str(dataset_root),
            "info_sha256": sha256_file(dataset_root / "meta/info.json"),
            "fps": float(dataset_info["fps"]),
            "state_dim": int(state["shape"][0]),
            "state_names": state.get("names"),
            "camera_keys": cameras,
            "features": features,
        },
        "policy": {
            "sft_path": str(pathlib.Path(config.vla_pretrained_path).expanduser().resolve()),
            "rl_token_path": str(pathlib.Path(args.rl_token_policy_path).expanduser().resolve()),
            "sft_contract": sft_contract,
            "rl_token_contract": rl_token_contract,
            "rl_token_dim": config.rl_token_dim,
            "rl_token_num_rl_tokens": config.rl_token_num_rl_tokens,
            "token_pool_size": config.token_pool_size,
            "image_only": config.image_only,
            "active_camera_indices": config.active_camera_indices,
            "num_per_camera": config.num_per_camera,
        },
        "cache": {
            "kind": HANDOVER_CACHE_KIND,
            "extractor": "pi05_prefix_only_v1",
            "frame_stride": args.frame_stride,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "task_instruction_override": args.task_instruction,
            "phase_annotations": (
                {
                    "path": str(phase_annotations_path),
                    "sha256": sha256_file(phase_annotations_path),
                }
                if phase_annotations_path is not None
                else None
            ),
            "phase_source": (
                "phase_annotations_sidecar"
                if phase_annotations_path is not None
                else FRAME_PHASE_KEY
            ),
            "episode_splits": episode_splits,
        },
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    args = parse_args()
    if args.frame_stride <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("frame-stride and batch-size must be positive; num-workers cannot be negative")
    if args.empty_cache_every < 0:
        raise ValueError("empty-cache-every cannot be negative")
    if not 0 < args.train_ratio < 1:
        raise ValueError("handover cache requires train-ratio in (0, 1)")

    _log(f"load RLTokenPolicy from {args.rl_token_policy_path}")
    policy, sft_contract, rl_token_contract = _load_policy(args)
    config = policy.config
    preprocessor, _ = make_rlt_token_pre_post_processors(config=config)

    dataset_root = pathlib.Path(args.dataset_root).expanduser().resolve()
    dataset_info = _load_json(dataset_root / "meta/info.json")
    if dataset_info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"handover cache requires a LeRobot v3.0 dataset, got "
            f"{dataset_info.get('codebase_version')!r}"
        )
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=dataset_root,
        video_backend=args.video_backend,
    )
    episode_count = dataset.num_episodes
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("max-episodes must be positive")
        episode_count = min(episode_count, args.max_episodes)
    if episode_count < 2:
        raise ValueError("handover cache requires at least two episodes for train/val splits")
    episode_ids = list(range(episode_count))
    train_ids, val_ids = split_episode_ids(episode_ids, args.train_ratio, args.seed)
    episode_splits = {"train": train_ids, "val": val_ids}

    phase_annotations_path = (
        pathlib.Path(args.phase_annotations).expanduser().resolve()
        if args.phase_annotations is not None
        else None
    )
    phase_annotations = None
    if phase_annotations_path is not None:
        episode_lengths = {
            int(_episode_value(dataset.meta.episodes, "episode_index", episode_id)): int(
                _episode_value(dataset.meta.episodes, "length", episode_id)
            )
            for episode_id in range(dataset.num_episodes)
        }
        phase_annotations = load_phase_annotations(
            phase_annotations_path,
            expected_info_sha256=sha256_file(dataset_root / "meta/info.json"),
            episode_lengths=episode_lengths,
        )

    contract = _build_contract(
        args=args,
        config=config,
        dataset_info=dataset_info,
        episode_splits=episode_splits,
        sft_contract=sft_contract,
        rl_token_contract=rl_token_contract,
        phase_annotations_path=phase_annotations_path,
    )
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _prepare_manifest(output_dir, contract, args.resume)
    started = time.time()

    for split, ids in episode_splits.items():
        finished = set(manifest["completed_episodes"][split])
        for position, episode_id in enumerate(ids, start=1):
            if episode_id in finished:
                _log(f"[{split}] skip completed episode {episode_id}")
                continue
            episodes = dataset.meta.episodes
            episode_start = int(_episode_value(episodes, "dataset_from_index", episode_id))
            episode_stop = int(_episode_value(episodes, "dataset_to_index", episode_id))
            frame_annotations = load_episode_phase_labels(dataset, episode_start, episode_stop)
            if phase_annotations is not None:
                frame_annotations = overlay_episode_phase_annotations(
                    frame_annotations,
                    phase_annotations,
                    episode_id=episode_id,
                    episode_start=episode_start,
                    episode_stop=episode_stop,
                )
            if frame_annotations is None:
                raise ValueError(
                    f"episode {episode_id} has no phase labels; pass --phase-annotations or "
                    f"record {FRAME_PHASE_KEY}"
                )
            frame_indices = list(range(episode_start, episode_stop, args.frame_stride))
            _log(
                f"[{split}] episode {position}/{len(ids)} id={episode_id} "
                f"frames={episode_stop - episode_start} samples={len(frame_indices)}"
            )
            samples = _encode_episode(
                policy=policy,
                preprocessor=preprocessor,
                dataset=dataset,
                frame_indices=frame_indices,
                episode_start=episode_start,
                episode_id=episode_id,
                frame_annotations=frame_annotations,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                empty_cache_every=args.empty_cache_every,
                task_override=args.task_instruction,
            )
            negative = sum(float(sample["is_critical"]) == 0.0 for sample in samples)
            positive = len(samples) - negative
            relative_path = pathlib.Path("shards") / split / f"episode_{episode_id:06d}.pt"
            _atomic_torch_save(samples, output_dir / relative_path)
            manifest["completed_episodes"][split].append(episode_id)
            manifest["shards"][split].append(
                {
                    "episode_id": episode_id,
                    "path": relative_path.as_posix(),
                    "transitions": len(samples),
                    "examples": len(samples),
                    "class_counts": {"negative": negative, "positive": positive},
                    "sha256": sha256_file(output_dir / relative_path),
                }
            )
            atomic_write_json(output_dir / CACHE_MANIFEST_NAME, manifest)

    if not manifest["shards"]["train"] or not manifest["shards"]["val"]:
        raise ValueError("handover cache requires non-empty train and val splits")
    class_counts = {
        split: {
            label: sum(int(shard["class_counts"][label]) for shard in manifest["shards"][split])
            for label in ("negative", "positive")
        }
        for split in ("train", "val")
    }
    for split, counts in class_counts.items():
        if counts["negative"] == 0 or counts["positive"] == 0:
            raise ValueError(f"handover {split} split requires both classes; found {counts}")
    manifest["class_counts"] = class_counts
    manifest["total_examples"] = {
        split: sum(int(shard["examples"]) for shard in manifest["shards"][split])
        for split in ("train", "val")
    }
    manifest["status"] = "complete"
    atomic_write_json(output_dir / CACHE_MANIFEST_NAME, manifest)
    _log(
        f"done in {time.time() - started:.1f}s: examples={manifest['total_examples']} "
        f"classes={class_counts}"
    )


if __name__ == "__main__":
    main()
