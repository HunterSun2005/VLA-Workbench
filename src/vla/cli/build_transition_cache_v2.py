"""Build a resumable, contract-checked chunk-transition cache."""

from __future__ import annotations

import argparse
import pathlib
import random
import sys
import time
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vla.adapters.lerobot.checkpoint_contract import (
    assert_same_fingerprint,
    fingerprint_rl_token_checkpoint,
    fingerprint_sft_checkpoint,
    sha256_file,
)
from vla.adapters.lerobot.offline_dataset import build_overlap_frame_indices
from vla.adapters.lerobot.policies.action_modifier import PrefixOutputCapture
from vla.adapters.lerobot.policies.cache_manifest import (
    CACHE_MANIFEST_NAME,
    assert_manifest_contract,
    atomic_write_json,
    load_cache_manifest,
    new_cache_manifest,
)
from vla.adapters.lerobot.policies.configuration_rlt_token import RLTokenConfig
from vla.adapters.lerobot.policies.modeling_rlt_token import RLTokenPolicy
from vla.adapters.lerobot.policies.processor_rlt_token import make_rlt_token_pre_post_processors
from vla.core.rewards import build_reward_seq
from vla.core.training_contract import REFERENCE_POLICY, TRAINING_SEMANTICS_VERSION, supervision_reference
from vla.core.interfaces import (
    TRANSITION_SOURCE_DEMO,
    TRANSITION_SOURCE_HUMAN_OVERRIDE,
    TRANSITION_SOURCE_RL_AUTONOMOUS,
    TRANSITION_SOURCE_WARMUP_VLA,
)
from vla.core.phase_annotations import (
    load_phase_annotations,
    overlay_episode_phase_annotations,
)


FRAME_INTERVENTION_KEY = "complementary_info.is_intervention"
FRAME_PHASE_KEY = "complementary_info.phase"
FRAME_COLLECTOR_KEY = "complementary_info.collector_policy_id"
COLLECTOR_HUMAN = 0
COLLECTOR_PI05 = 1
COLLECTOR_RLT_ACTOR = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dataset-repo-id", required=True)
    parser.add_argument("--demo-dataset-root", required=True)
    parser.add_argument("--rl-token-policy-path", required=True)
    parser.add_argument(
        "--sft-path",
        "--vla-pretrained-path",
        dest="sft_path",
        default=None,
        help="Optional relocated SFT checkpoint. Its content must match the RL Token contract.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional relocated tokenizer recorded by the SFT preprocessor.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--phase-annotations",
        default=None,
        help=(
            "Optional phase_annotations.json produced by vla-annotate-phases. "
            "Its reviewed labels override complementary_info.phase."
        ),
    )
    parser.add_argument(
        "--task-instruction",
        default=None,
        help="Optional task override. Dataset task text is preserved when omitted.",
    )
    parser.add_argument("--chunk-length", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-episodes", type=int, default=None)
    # TorchCodec is the supported video decoder; pyav routes through the
    # deprecated torchvision.io video APIs.
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--tolerance-s", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--empty-cache-every", type=int, default=4)
    parser.add_argument(
        "--default-episode-success",
        choices=("success", "failure"),
        default=None,
        help="Explicit fallback for datasets without episode_success metadata.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    if path.name == "info.json" and path.parent.name == "meta":
        from vla.dataset_metadata import load_info
        return load_info(path.parent.parent)
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _episode_value(episodes: Any, key: str, episode_id: int) -> Any:
    column_names = getattr(episodes, "column_names", None)
    if column_names is not None and key not in column_names:
        raise KeyError(key)
    try:
        column = episodes[key]
    except (KeyError, TypeError) as error:
        raise KeyError(key) from error
    if hasattr(column, "iloc"):
        return column.iloc[episode_id]
    return column[episode_id]


def _parse_episode_success(value: Any, episode_id: int) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "success":
            return True
        if normalized == "failure":
            return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"unrecognized episode_success for episode {episode_id}: {value!r}")


def resolve_episode_success(
    episodes: Any,
    episode_id: int,
    default: str | None,
) -> bool:
    try:
        value = _episode_value(episodes, "episode_success", episode_id)
    except KeyError:
        if default is None:
            raise ValueError(
                "dataset has no episode_success metadata; relabel it or pass "
                "--default-episode-success explicitly"
            ) from None
        value = default
    return _parse_episode_success(value, episode_id)


def split_episode_ids(
    episode_ids: list[int],
    train_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0 < train_ratio <= 1:
        raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")
    shuffled = list(episode_ids)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) == 1 and train_ratio != 1:
        raise ValueError("a one-episode dataset requires --train-ratio=1.0")
    if train_ratio == 1 or not shuffled:
        return shuffled, []
    n_train = min(max(int(train_ratio * len(shuffled)), 1), len(shuffled) - 1)
    return shuffled[:n_train], shuffled[n_train:]


def validate_chunk_settings(chunk_length: int, stride: int, horizon: int) -> None:
    if not 1 <= chunk_length <= horizon:
        raise ValueError(
            f"chunk_length must be between 1 and PI0.5 chunk_size={horizon}, got {chunk_length}"
        )
    if stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {stride}")
    if chunk_length % stride != 0:
        raise ValueError(
            f"chunk_length={chunk_length} must be divisible by frame_stride={stride}"
        )


def _encoded_to_transitions(
    encoded: dict[int, dict[str, Tensor]],
    frame_indices: list[int],
    episode_last_frame: int,
    chunk_length: int,
    episode_success: bool,
    episode_id: int,
    frame_annotations: dict[int, dict[str, float | int]] | None = None,
) -> list[dict[str, Tensor]]:
    transitions: list[dict[str, Tensor]] = []
    for frame in frame_indices:
        next_frame = frame + chunk_length
        if next_frame > episode_last_frame:
            continue
        if next_frame not in encoded:
            raise ValueError(f"missing x_(t+C) anchor for frame {frame} -> {next_frame}")
        current = encoded[frame]
        following = encoded[next_frame]
        terminal = next_frame == episode_last_frame
        annotation = _chunk_annotation(
            frame_annotations,
            start_frame=frame,
            chunk_length=chunk_length,
        )
        transitions.append(
            {
                "state_vec": current["state_vec"],
                "exec_chunk": current["exec_chunk"],
                "ref_chunk": supervision_reference(
                    current["exec_chunk"], current["ref_chunk"], frame_annotations, frame,
                ),
                "reward_seq": build_reward_seq(
                    chunk_length=chunk_length,
                    is_terminal_chunk=terminal,
                    episode_success=episode_success,
                ),
                "next_state_vec": following["state_vec"],
                "next_ref_chunk": following["ref_chunk"],
                "done": torch.tensor(float(terminal)),
                "intervention": torch.tensor(annotation["intervention"]),
                "actual_steps": torch.tensor(chunk_length, dtype=torch.int64),
                "source": torch.tensor(annotation["source"], dtype=torch.int64),
                "episode_id": torch.tensor(episode_id, dtype=torch.int64),
                "is_critical": torch.tensor(annotation["is_critical"]),
            }
        )
    return transitions


def _chunk_annotation(
    frame_annotations: dict[int, dict[str, float | int]] | None,
    *,
    start_frame: int,
    chunk_length: int,
) -> dict[str, float | int]:
    if not frame_annotations:
        return {
            "intervention": 0.0,
            "source": TRANSITION_SOURCE_DEMO,
            "is_critical": 1.0,
        }

    window = [
        frame_annotations.get(index, {})
        for index in range(start_frame, start_frame + chunk_length)
    ]
    has_rollout_source = any(
        "collector_policy_id" in item or "is_intervention" in item
        for item in window
    )
    if not has_rollout_source:
        human_dominant = False
        source = TRANSITION_SOURCE_DEMO
    else:
        collector_codes = [
            int(item.get("collector_policy_id", COLLECTOR_PI05))
            for item in window
        ]
        human_votes = sum(
            bool(float(item.get("is_intervention", 0.0)) > 0.5)
            or int(item.get("collector_policy_id", COLLECTOR_PI05)) == COLLECTOR_HUMAN
            for item in window
        )
        human_dominant = human_votes >= len(window) - human_votes
        if human_dominant:
            source = TRANSITION_SOURCE_HUMAN_OVERRIDE
        else:
            policy_codes = [code for code in collector_codes if code != COLLECTOR_HUMAN]
            dominant = max(
                set(policy_codes or [COLLECTOR_PI05]),
                key=lambda code: ((policy_codes or [COLLECTOR_PI05]).count(code), code),
            )
            source = (
                TRANSITION_SOURCE_RL_AUTONOMOUS
                if dominant == COLLECTOR_RLT_ACTOR
                else TRANSITION_SOURCE_WARMUP_VLA
            )
    anchor = frame_annotations.get(start_frame, {})
    return {
        "intervention": float(human_dominant),
        "source": source,
        "is_critical": float(float(anchor.get("phase", 1.0)) > 0.5),
    }


def _annotation_scalar(value: Any, *, key: str, frame_index: int) -> float | int:
    if isinstance(value, Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{key} at frame {frame_index} must contain one value")
        value = value[0]
    if key == FRAME_COLLECTOR_KEY:
        result = int(value)
        if result not in {COLLECTOR_HUMAN, COLLECTOR_PI05, COLLECTOR_RLT_ACTOR}:
            raise ValueError(f"unknown collector_policy_id={result} at frame {frame_index}")
        return result
    result = float(value)
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"non-finite {key} at frame {frame_index}")
    return result


def load_episode_frame_annotations(
    dataset: LeRobotDataset,
    episode_start: int,
    episode_stop: int,
) -> dict[int, dict[str, float | int]] | None:
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is None:
        return None
    columns = set(getattr(hf_dataset, "column_names", []))
    available = columns.intersection({FRAME_INTERVENTION_KEY, FRAME_PHASE_KEY, FRAME_COLLECTOR_KEY})
    if not available:
        return None
    from vla.dataset_metadata import dataset_info
    info = dataset_info(dataset)
    contract = info.get("recording_contract", {})
    if not contract and info.get("processing", {}).get("version") == 1:
        # Offline exports retain demonstration provenance without pretending to be
        # resumable collection/online runs under the original writer contract.
        contract = info["processing"].get("source_contract", {}).get("recording_contract", {})
    phase_only_demo = (
        available == {FRAME_PHASE_KEY}
        and info.get("collection_kind") == "human_demonstration"
        and info.get("gamepad_recording_version") == 1
        and contract.get("version") == 1
        and contract.get("source") == "demonstration"
        and contract.get("collection_kind") == "human_demonstration"
        and (
            contract.get("action_encoding") == "interval_integrated_base_velocity_absolute_tcp_rot6d_v1"
            or (info.get("robot_type") == "franka_robotiq_dual"
                and contract.get("action_encoding") == "controller_joint_command_fk_absolute_tcp_rot6d_v1")
        )
    )
    if FRAME_COLLECTOR_KEY not in available and not phase_only_demo:
        raise ValueError(
            f"rollout annotations require {FRAME_COLLECTOR_KEY!r} when complementary fields are present"
        )
    selected = hf_dataset.select(range(episode_start, episode_stop))
    annotations: dict[int, dict[str, float | int]] = {}
    names = {
        FRAME_INTERVENTION_KEY: "is_intervention",
        FRAME_PHASE_KEY: "phase",
        FRAME_COLLECTOR_KEY: "collector_policy_id",
    }
    for local_index, frame_index in enumerate(range(episode_start, episode_stop)):
        annotations[frame_index] = {
            names[key]: _annotation_scalar(selected[key][local_index], key=key, frame_index=frame_index)
            for key in available
        }
        if phase_only_demo and annotations[frame_index]["phase"] not in (0.0, 1.0):
            raise ValueError("demonstration phase labels must be binary")
    return annotations


def _encode_episode(
    *,
    pi05: Any,
    rl_token: Any,
    preprocessor: Any,
    capture: PrefixOutputCapture,
    dataset: LeRobotDataset,
    frame_indices: list[int],
    episode_last_frame: int,
    chunk_length: int,
    action_dim: int,
    proprio_dim: int,
    batch_size: int,
    num_workers: int,
    device: str,
    empty_cache_every: int,
    task_override: str | None,
    episode_success: bool,
    episode_id: int,
    frame_annotations: dict[int, dict[str, float | int]] | None = None,
) -> list[dict[str, Tensor]]:
    if not frame_indices:
        return []
    loader = DataLoader(
        Subset(dataset, frame_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=False,
    )
    encoded: dict[int, dict[str, Tensor]] = {}
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
        if "action" not in processed:
            raise KeyError("SFT preprocessor output has no action chunk")
        with torch.no_grad():
            vla_chunk = pi05.predict_action_chunk(processed)
            prefix = capture.consume()
            latent = rl_token.encode(prefix.to(torch.float32))
        proprio = processed["observation.state"][:, :proprio_dim].detach().cpu()
        state_vec = torch.cat([latent.detach().cpu(), proprio], dim=-1)
        exec_chunk = processed["action"][:, :chunk_length, :action_dim].detach().cpu()
        ref_chunk = vla_chunk[:, :chunk_length, :action_dim].detach().cpu()
        if exec_chunk.shape[1:] != ref_chunk.shape[1:]:
            raise ValueError(
                f"expert/reference chunk shape mismatch: {tuple(exec_chunk.shape)} vs "
                f"{tuple(ref_chunk.shape)}"
            )
        for local_index, raw_index in enumerate(raw_indices):
            encoded[raw_index] = {
                "state_vec": state_vec[local_index],
                "exec_chunk": exec_chunk[local_index],
                "ref_chunk": ref_chunk[local_index],
            }
        if (
            device.startswith("cuda")
            and empty_cache_every > 0
            and (batch_index + 1) % empty_cache_every == 0
        ):
            torch.cuda.empty_cache()
    if offset != len(frame_indices):
        raise ValueError(f"encoded {offset} anchors, expected {len(frame_indices)}")
    return _encoded_to_transitions(
        encoded,
        frame_indices,
        episode_last_frame,
        chunk_length,
        episode_success,
        episode_id,
        frame_annotations,
    )


def _atomic_torch_save(value: Any, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _load_policy(args: argparse.Namespace) -> tuple[RLTokenPolicy, dict[str, Any], dict[str, Any]]:
    RLTokenConfig.ensure_registered()
    config = RLTokenConfig.from_pretrained(args.rl_token_policy_path)
    config.device = args.device
    if args.sft_path is not None:
        config.vla_pretrained_path = str(pathlib.Path(args.sft_path).expanduser().resolve())
    if args.tokenizer_path is not None:
        config.tokenizer_path = args.tokenizer_path
    if config.sft_checkpoint_contract is None:
        raise ValueError("RL Token checkpoint has no saved SFT content fingerprint")
    policy = RLTokenPolicy.from_pretrained(
        args.rl_token_policy_path,
        config=config,
    ).to(args.device).eval()
    sft_contract = getattr(policy, "_validated_sft_contract", None)
    if sft_contract is None:
        sft_contract = fingerprint_sft_checkpoint(config.vla_pretrained_path)
        assert_same_fingerprint(config.sft_checkpoint_contract, sft_contract, label="SFT")
    rl_token_contract = fingerprint_rl_token_checkpoint(args.rl_token_policy_path)
    return policy, sft_contract, rl_token_contract


def _build_contract(
    *,
    args: argparse.Namespace,
    config: RLTokenConfig,
    dataset_info: dict[str, Any],
    episode_splits: dict[str, list[int]],
    sft_contract: dict[str, Any],
    rl_token_contract: dict[str, Any],
    phase_annotations_path: pathlib.Path | None,
) -> dict[str, Any]:
    features = dataset_info["features"]
    state = features["observation.state"]
    action = features["action"]
    cameras = [
        key.removeprefix("observation.images.")
        for key in features
        if key.startswith("observation.images.")
    ]
    vla = dataset_info.get("evo_rlt", {})
    if int(state["shape"][0]) != config.proprio_dim:
        raise ValueError(
            f"dataset state_dim={state['shape'][0]} does not match RL Token {config.proprio_dim}"
        )
    if int(action["shape"][0]) != config.action_dim:
        raise ValueError(
            f"dataset action_dim={action['shape'][0]} does not match RL Token {config.action_dim}"
        )
    if cameras != list(config.camera_keys):
        raise ValueError(
            f"dataset camera order {cameras} does not match RL Token {list(config.camera_keys)}"
        )
    if vla.get("actions_stored_as") != "absolute":
        raise ValueError(
            "transition cache requires evo_rlt.actions_stored_as='absolute' in dataset metadata"
        )
    sft_config = _load_json(pathlib.Path(config.vla_pretrained_path) / "config.json")
    use_relative_actions = bool(sft_config.get("use_relative_actions", False))
    sft_preprocessor = _load_json(
        pathlib.Path(config.vla_pretrained_path) / "policy_preprocessor.json"
    )
    relative_processor = next(
        (
            {"registry_name": step.get("registry_name"), "config": step.get("config", {})}
            for step in sft_preprocessor.get("steps", [])
            if step.get("registry_name")
            in {"delta_actions_processor", "relative_actions_processor"}
        ),
        None,
    )
    expected_stats = "relative" if use_relative_actions else "absolute"
    if vla.get("action_stats") != expected_stats:
        raise ValueError(
            f"dataset action_stats={vla.get('action_stats')!r} does not match SFT "
            f"use_relative_actions={use_relative_actions}"
        )
    info_path = pathlib.Path(args.demo_dataset_root).expanduser().resolve() / "meta/info.json"
    return {
        "dataset": {
            "repo_id": args.demo_dataset_repo_id,
            "root": str(pathlib.Path(args.demo_dataset_root).expanduser().resolve()),
            "info_sha256": sha256_file(info_path),
            "fps": float(dataset_info["fps"]),
            "state_dim": int(state["shape"][0]),
            "action_dim": int(action["shape"][0]),
            "state_names": state.get("names"),
            "action_names": action.get("names"),
            "camera_keys": cameras,
            "features": features,
            "actions_stored_as": vla.get("actions_stored_as"),
            "action_stats": vla.get("action_stats"),
        },
        "policy": {
            "sft_path": str(pathlib.Path(config.vla_pretrained_path).expanduser().resolve()),
            "rl_token_path": str(pathlib.Path(args.rl_token_policy_path).expanduser().resolve()),
            "sft_contract": sft_contract,
            "rl_token_contract": rl_token_contract,
            "pi05_chunk_size": config.chunk_size,
            "action_dim": config.action_dim,
            "proprio_dim": config.proprio_dim,
            "camera_keys": list(config.camera_keys),
            "rl_token_dim": config.rl_token_dim,
            "rl_token_num_rl_tokens": config.rl_token_num_rl_tokens,
            "token_pool_size": config.token_pool_size,
            "image_only": config.image_only,
            "active_camera_indices": config.active_camera_indices,
            "num_per_camera": config.num_per_camera,
            "use_relative_actions": use_relative_actions,
            "relative_action_processor": relative_processor,
            "action_space": "sft_preprocessed",
        },
        "cache": {
            "training_semantics_version": TRAINING_SEMANTICS_VERSION,
            "chunk_length": args.chunk_length,
            "frame_stride": args.frame_stride,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "task_instruction_override": args.task_instruction,
            "default_episode_success": args.default_episode_success,
            "phase_annotations": (
                {
                    "path": str(phase_annotations_path),
                    "sha256": sha256_file(phase_annotations_path),
                }
                if phase_annotations_path is not None
                else None
            ),
            "annotation_policy": {
                "ref_chunk": REFERENCE_POLICY,
                "exec_chunk": "dataset_action",
                "source_vote": "majority_human_wins_ties",
                "critical_phase": "anchor_frame",
                "critical_phase_source": (
                    "phase_annotations_sidecar"
                    if phase_annotations_path is not None
                    else "dataset_frame_or_demo_default"
                ),
            },
            "episode_splits": episode_splits,
        },
    }


def _prepare_manifest(
    output_dir: pathlib.Path,
    contract: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    path = output_dir / CACHE_MANIFEST_NAME
    if path.exists():
        if not resume:
            raise FileExistsError(f"cache manifest already exists; pass --resume or use a new directory: {path}")
        manifest = load_cache_manifest(path)
        assert_manifest_contract(manifest, contract)
        for split in ("train", "val"):
            completed = set(manifest["completed_episodes"][split])
            shard_episodes = {int(shard["episode_id"]) for shard in manifest["shards"][split]}
            if completed != shard_episodes:
                raise ValueError(f"manifest completed/shard episode mismatch for split {split}")
            for shard in manifest["shards"][split]:
                shard_path = output_dir / shard["path"]
                if not shard_path.is_file():
                    raise FileNotFoundError(f"completed transition shard is missing: {shard_path}")
                expected_sha = shard.get("sha256")
                if expected_sha is not None and sha256_file(shard_path) != expected_sha:
                    raise ValueError(f"transition shard fingerprint mismatch: {shard_path}")
        return manifest
    if resume:
        raise FileNotFoundError(f"cannot resume without an existing cache manifest: {path}")
    manifest = new_cache_manifest(contract)
    atomic_write_json(path, manifest)
    return manifest


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    args = parse_args()
    if args.empty_cache_every < 0:
        raise ValueError("empty_cache_every cannot be negative")

    dataset_root = pathlib.Path(args.demo_dataset_root).expanduser().resolve()
    dataset_info = _load_json(dataset_root / "meta/info.json")
    from vla.core.online_contract import reject_unaligned_offline_dataset

    reject_unaligned_offline_dataset(dataset_info)
    _log(f"load RLTokenPolicy from {args.rl_token_policy_path}")
    policy, sft_contract, rl_token_contract = _load_policy(args)
    config = policy.config
    validate_chunk_settings(args.chunk_length, args.frame_stride, config.chunk_size)

    if dataset_info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"transition cache requires a LeRobot v3.0 dataset, got "
            f"{dataset_info.get('codebase_version')!r}"
        )
    fps = float(dataset_info["fps"])
    if fps <= 0:
        raise ValueError(f"dataset FPS must be positive, got {fps}")
    delta = {"action": [index / fps for index in range(config.chunk_size)]}
    dataset = LeRobotDataset(
        repo_id=args.demo_dataset_repo_id,
        root=dataset_root,
        delta_timestamps=delta,
        tolerance_s=args.tolerance_s,
        video_backend=args.video_backend,
    )
    count = dataset.num_episodes
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        count = min(count, args.max_episodes)
    episode_ids = list(range(count))
    train_ids, val_ids = split_episode_ids(episode_ids, args.train_ratio, args.seed)
    episode_splits = {"train": train_ids, "val": val_ids}
    phase_annotations_path = (
        pathlib.Path(args.phase_annotations).expanduser().resolve()
        if args.phase_annotations is not None
        else None
    )
    phase_annotations = None
    if phase_annotations_path is not None:
        if not phase_annotations_path.is_file():
            raise FileNotFoundError(f"phase annotations are missing: {phase_annotations_path}")
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

    preprocessor, _ = make_rlt_token_pre_post_processors(config=config)
    capture = PrefixOutputCapture(
        token_pool_size=config.token_pool_size,
        image_only=config.image_only,
        num_image_tokens=policy._num_image_tokens,
        num_per_camera=config.num_per_camera,
        active_camera_indices=config.active_camera_indices,
    )
    capture.attach(policy._pi05)
    started = time.time()
    try:
        for split, ids in episode_splits.items():
            finished = set(manifest["completed_episodes"][split])
            skipped = set(manifest["skipped_episodes"][split])
            for position, episode_id in enumerate(ids, start=1):
                if episode_id in finished or episode_id in skipped:
                    _log(f"[{split}] skip completed episode {episode_id}")
                    continue
                episodes = dataset.meta.episodes
                episode_start = int(_episode_value(episodes, "dataset_from_index", episode_id))
                episode_stop = int(_episode_value(episodes, "dataset_to_index", episode_id))
                episode_success = resolve_episode_success(
                    episodes,
                    episode_id,
                    args.default_episode_success,
                )
                frame_annotations = load_episode_frame_annotations(
                    dataset,
                    episode_start,
                    episode_stop,
                )
                if phase_annotations is not None:
                    frame_annotations = overlay_episode_phase_annotations(
                        frame_annotations,
                        phase_annotations,
                        episode_id=episode_id,
                        episode_start=episode_start,
                        episode_stop=episode_stop,
                    )
                frame_indices = build_overlap_frame_indices(
                    episode_start=episode_start,
                    episode_stop=episode_stop,
                    chunk_length=args.chunk_length,
                    stride=args.frame_stride,
                )
                _log(
                    f"[{split}] episode {position}/{len(ids)} id={episode_id} "
                    f"frames={episode_stop - episode_start} anchors={len(frame_indices)}"
                )
                transitions = _encode_episode(
                    pi05=policy._pi05,
                    rl_token=policy.rl_token,
                    preprocessor=preprocessor,
                    capture=capture,
                    dataset=dataset,
                    frame_indices=frame_indices,
                    episode_last_frame=episode_stop - 1,
                    chunk_length=args.chunk_length,
                    action_dim=config.action_dim,
                    proprio_dim=config.proprio_dim,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=args.device,
                    empty_cache_every=args.empty_cache_every,
                    task_override=args.task_instruction,
                    episode_success=episode_success,
                    episode_id=episode_id,
                    frame_annotations=frame_annotations,
                )
                if not transitions:
                    manifest["skipped_episodes"][split].append(episode_id)
                else:
                    relative_path = pathlib.Path("shards") / split / f"episode_{episode_id:06d}.pt"
                    _atomic_torch_save(transitions, output_dir / relative_path)
                    manifest["completed_episodes"][split].append(episode_id)
                    manifest["shards"][split].append(
                        {
                            "episode_id": episode_id,
                            "path": relative_path.as_posix(),
                            "transitions": len(transitions),
                            "sha256": sha256_file(output_dir / relative_path),
                            "episode_success": episode_success,
                        }
                    )
                atomic_write_json(output_dir / CACHE_MANIFEST_NAME, manifest)
    finally:
        capture.detach()

    if not manifest["shards"]["train"]:
        raise ValueError("transition cache contains no train transitions")
    manifest["status"] = "complete"
    manifest["total_transitions"] = {
        split: sum(item["transitions"] for item in manifest["shards"][split])
        for split in ("train", "val")
    }
    atomic_write_json(output_dir / CACHE_MANIFEST_NAME, manifest)
    _log(f"done in {time.time() - started:.1f}s: {manifest['total_transitions']}")


if __name__ == "__main__":
    main()
