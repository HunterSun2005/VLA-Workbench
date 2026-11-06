from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
import uuid
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from vla.adapters.lerobot.policies.action_modifier import PrefixOutputCapture
from vla.core.actor import ChunkActor
from vla.core.critic import TwinCritic
from vla.core.interfaces import (
    TRANSITION_SOURCE_HUMAN_OVERRIDE,
    TRANSITION_SOURCE_RL_AUTONOMOUS,
    TRANSITION_SOURCE_WARMUP_VLA,
)
from vla.core.losses import critic_loss
from vla.core.online_replay import PersistentReplayBuffer, ReplayTransitionRecord
from vla.core.online_contract import (
    MAX_STEP_PERIODS, MIN_STEP_PERIODS, RECORDING_SCHEMA_VERSION, TIMING_FIELDS,
    bind_run, file_digest, fsync_directory, load_run, model_digest,
)
from vla.core.utils import soft_update
from vla.core.training_contract import TRAINING_SEMANTICS_VERSION, supervision_reference
from vla.core.writer_lock import WriterLease, cleanup_on_init_error

log = logging.getLogger(__name__)

FRAME_INTERVENTION_KEY = "complementary_info.is_intervention"
FRAME_PHASE_KEY = "complementary_info.phase"
FRAME_COLLECTOR_KEY = "complementary_info.collector_policy_id"
COLLECTOR_HUMAN = 0
COLLECTOR_RLT_ACTOR = 2


class _OnlineTrainingProgress:
    def __init__(
        self,
        total: int,
        *,
        stream: Any = None,
        refresh_interval_s: float = 0.1,
        callback: Callable[[dict[str, float | int | str | None]], None] | None = None,
    ) -> None:
        self.total = int(total)
        self.stream = sys.stdout if stream is None else stream
        self.refresh_interval_s = float(refresh_interval_s)
        self.callback = callback
        self.last_rendered_at = float("-inf")
        self.last_width = 0
        self.rendered = False
        self.closed = False

    def update(
        self,
        completed: int,
        *,
        critic_loss_value: float | None,
        actor_updates: int,
        actor_loss_value: float | None,
    ) -> None:
        if self.closed or self.total <= 0:
            return
        now = time.monotonic()
        completed = min(max(int(completed), 0), self.total)
        if (
            completed not in {0, 1, self.total}
            and now - self.last_rendered_at < self.refresh_interval_s
        ):
            return
        fraction = completed / self.total
        filled = min(24, int(24 * fraction))
        bar = "=" * filled + "-" * (24 - filled)
        critic_text = "--" if critic_loss_value is None else f"{critic_loss_value:.4f}"
        actor_text = "--" if actor_loss_value is None else f"{actor_loss_value:.4f}"
        line = (
            f"Online training [{bar}] {completed}/{self.total} "
            f"({100.0 * fraction:5.1f}%) critic_loss={critic_text} "
            f"actor_updates={actor_updates} actor_loss={actor_text}"
        )
        self.stream.write("\r" + line.ljust(self.last_width))
        self.stream.flush()
        self.last_width = max(self.last_width, len(line))
        self.last_rendered_at = now
        self.rendered = True
        if self.callback is not None:
            self.callback(
                {
                    "completed_updates": completed,
                    "total_updates": self.total,
                    "critic_loss": critic_loss_value,
                    "actor_updates": int(actor_updates),
                    "actor_loss": actor_loss_value,
                }
            )
        if completed == self.total:
            self.close()

    def close(self) -> None:
        if self.closed:
            return
        if self.rendered:
            self.stream.write("\n")
            self.stream.flush()
        self.closed = True


@dataclass
class OnlineRLTTrainingConfig:
    enable: bool = False
    replay_root: str = ""
    replay_capacity: int = 100_000
    warmup_steps: int = 1_000
    utd_ratio: int = 5
    frame_stride: int = 2
    batch_size: int = 32
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    seed: int = 42
    # Batch VLA/RL Token encoding across sampled episode frames. Larger values
    # keep the GPU busier while preserving the transition semantics.
    encoding_batch_size: int = 16
    encoding_num_workers: int = 0
    video_backend: str = "torchcodec"
    tolerance_s: float = 0.04
    inherit_actor_critic: bool = False
    full_episode: bool = False

    def validate(self, *, chunk_length: int, actor_update_interval: int) -> None:
        if not self.replay_root:
            raise ValueError(
                "`online.replay_root` is required when online RLT is enabled"
            )
        if self.replay_capacity <= 0:
            raise ValueError("`online.replay_capacity` must be positive")
        if self.full_episode and (not self.inherit_actor_critic or self.warmup_steps != 0):
            raise ValueError("full-episode online RLT requires inherited A/C and warmup_steps=0")
        if not self.full_episode and self.warmup_steps <= 0:
            raise ValueError("`online.warmup_steps` must be positive")
        if self.utd_ratio != 5:
            raise ValueError("paper-aligned online RLT requires `online.utd_ratio=5`")
        if self.frame_stride != 2:
            raise ValueError(
                "paper-aligned online RLT requires `online.frame_stride=2`"
            )
        if chunk_length <= 0 or chunk_length % self.frame_stride:
            raise ValueError(
                "chunk length must be positive and divisible by the online frame stride"
            )
        if actor_update_interval != 2:
            raise ValueError(
                "paper-aligned online RLT requires one Actor update per two Critic updates"
            )
        if self.batch_size <= 0 or self.encoding_batch_size <= 0:
            raise ValueError("online batch sizes must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("online replay capacity must be at least batch_size")
        if self.encoding_num_workers < 0:
            raise ValueError("`online.encoding_num_workers` cannot be negative")
        if self.actor_lr <= 0 or self.critic_lr <= 0:
            raise ValueError("online learning rates must be positive")
        if self.grad_clip_norm < 0:
            raise ValueError("`online.grad_clip_norm` cannot be negative")


@dataclass(frozen=True)
class CriticalTransitionSpec:
    anchor_frame: int
    next_frame: int
    actual_steps: int
    terminal: bool


def build_critical_transition_specs(
    annotations: dict[int, dict[str, float | int]],
    *,
    episode_start: int,
    episode_stop: int,
    chunk_length: int,
    stride: int,
    fps: float | None = None,
) -> list[CriticalTransitionSpec]:
    """Build overlapping C-step anchors, plus one padded terminal anchor per interval."""
    if annotations and "action_valid" in next(iter(annotations.values())):
        return _timed_critical_specs(annotations, episode_start, episode_stop, chunk_length, stride, fps)
    critical = [
        frame
        for frame in range(episode_start, episode_stop)
        if float(annotations.get(frame, {}).get("phase", 0.0)) > 0.5
    ]
    intervals: list[tuple[int, int]] = []
    for frame in critical:
        if not intervals or frame != intervals[-1][1]:
            intervals.append((frame, frame + 1))
        else:
            intervals[-1] = (intervals[-1][0], frame + 1)

    specs: list[CriticalTransitionSpec] = []
    for start, stop in intervals:
        anchor = start
        while anchor + chunk_length < stop:
            specs.append(
                CriticalTransitionSpec(
                    anchor, anchor + chunk_length, chunk_length, False
                )
            )
            anchor += stride
        if anchor < stop:
            actual_steps = min(chunk_length, stop - anchor)
            specs.append(CriticalTransitionSpec(anchor, stop - 1, actual_steps, True))
    return specs


def _timed_critical_specs(annotations, start, stop, chunk_length, stride, fps):
    if fps is None or fps <= 0:
        raise ValueError("timed online transitions require a positive fps")
    intervals: list[list[int]] = []
    entered = exited = False
    previous_segment = -1
    previous_observed = -1
    for frame in range(start, stop):
        row = annotations[frame]
        segment = int(row["segment_id"])
        if row["action_valid"] not in {0, 1} or row["segment_terminal"] not in {0, 1}:
            raise ValueError("action_valid and segment_terminal must be binary")
        if row["segment_terminal"] and row["action_valid"]:
            raise ValueError("terminal observation cannot also be an action")
        if row["observed_at_ns"] <= previous_observed:
            raise ValueError("observation times must increase within an episode")
        previous_observed = row["observed_at_ns"]
        if segment < previous_segment:
            raise ValueError("segment ids must be monotonic")
        previous_segment = segment
        if not row["action_valid"]:
            continue
        critical = float(row["phase"]) > 0.5
        if critical and exited:
            raise ValueError("online RLT requires one contiguous critical phase per episode")
        if not critical:
            exited |= entered
            continue
        entered = True
        if frame + 1 >= stop:
            raise ValueError("valid action has no post-action observation")
        following = annotations[frame + 1]
        elapsed = (following["observed_at_ns"] - row["observed_at_ns"]) / 1e9
        if (
            following["segment_id"] != segment
            or not MIN_STEP_PERIODS / fps <= elapsed <= MAX_STEP_PERIODS / fps
            or not row["observed_at_ns"] <= row["command_at_ns"] < following["observed_at_ns"]
        ):
            raise ValueError(f"invalid action/observation timing at frame {frame}")
        if not intervals or frame != intervals[-1][-1] + 1 or annotations[intervals[-1][-1]]["segment_id"] != segment:
            intervals.append([])
        intervals[-1].append(frame)
    specs = []
    for frames in intervals:
        endpoint = frames[-1] + 1
        terminal = bool(annotations[endpoint]["segment_terminal"])
        anchor = frames[0]
        while anchor + chunk_length <= endpoint:
            done = terminal and anchor + chunk_length == endpoint
            specs.append(CriticalTransitionSpec(anchor, anchor + chunk_length, chunk_length, done))
            if done:
                break
            anchor += stride
        else:
            if terminal and anchor < endpoint:
                specs.append(CriticalTransitionSpec(anchor, endpoint, endpoint - anchor, True))
    return specs


def _scalar(value: Any, *, integer: bool = False) -> float | int:
    if isinstance(value, Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("online frame annotations must be scalar")
        value = value[0]
    return int(value) if integer else float(value)


def load_frame_annotations(
    dataset: Any, episode_start: int, episode_stop: int
) -> dict[int, dict[str, float | int]]:
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is None:
        raise ValueError(
            "online RLT requires a readable LeRobot hf_dataset after save_episode"
        )
    columns = set(getattr(hf_dataset, "column_names", []))
    required = {FRAME_INTERVENTION_KEY, FRAME_PHASE_KEY, FRAME_COLLECTOR_KEY}
    missing = required - columns
    if missing:
        raise ValueError(
            f"online RLT dataset is missing frame annotations: {sorted(missing)}"
        )
    selected = hf_dataset.select(range(episode_start, episode_stop))
    annotations: dict[int, dict[str, float | int]] = {}
    for local, frame in enumerate(range(episode_start, episode_stop)):
        collector = _scalar(selected[FRAME_COLLECTOR_KEY][local], integer=True)
        if collector not in {0, 1, 2}:
            raise ValueError(
                f"unknown collector_policy_id={collector} at frame {frame}"
            )
        annotations[frame] = {
            "is_intervention": _scalar(selected[FRAME_INTERVENTION_KEY][local]),
            "phase": _scalar(selected[FRAME_PHASE_KEY][local]),
            "collector_policy_id": collector,
        }
        present = {name for name in TIMING_FIELDS if f"complementary_info.{name}" in columns}
        if present and present != set(TIMING_FIELDS):
            raise ValueError("incomplete online frame timing fields")
        for name in present:
            annotations[frame][name] = _scalar(selected[f"complementary_info.{name}"][local], integer=True)
    return annotations


class _TimedActionDataset(torch.utils.data.Dataset):
    """Assemble explicit same-segment action rows; video timestamps stay unchanged."""

    def __init__(self, dataset, annotations, chunk_length, action_dim):
        self.dataset, self.annotations = dataset, annotations
        self.chunk_length, self.action_dim = chunk_length, action_dim

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        actions = torch.zeros(self.chunk_length, self.action_dim)
        segment = self.annotations[index]["segment_id"]
        for offset in range(self.chunk_length):
            row = self.annotations.get(index + offset)
            if row is None or not row["action_valid"] or row["segment_id"] != segment:
                break
            raw = self.dataset.hf_dataset[index + offset]["action"]
            actions[offset] = torch.as_tensor(raw)[:self.action_dim]
        item["action"] = actions
        return item


def _transition_source(
    annotations: dict[int, dict[str, float | int]],
    spec: CriticalTransitionSpec,
    *,
    warmup: bool,
) -> tuple[int, float]:
    window = [
        annotations[spec.anchor_frame + offset] for offset in range(spec.actual_steps)
    ]
    human = sum(
        float(item["is_intervention"]) > 0.5
        or int(item["collector_policy_id"]) == COLLECTOR_HUMAN
        for item in window
    )
    if human:
        return TRANSITION_SOURCE_HUMAN_OVERRIDE, 1.0
    if warmup:
        return TRANSITION_SOURCE_WARMUP_VLA, 0.0
    return TRANSITION_SOURCE_RL_AUTONOMOUS, 0.0


def resolve_state_indices(
    dataset: Any,
    state_names: tuple[str, ...] | None,
    proprio_dim: int,
) -> tuple[int, ...] | None:
    if state_names is None:
        return None
    if len(state_names) != proprio_dim or len(set(state_names)) != len(state_names):
        raise ValueError(
            "online target state names must be unique and match proprio_dim: "
            f"names={len(state_names)} proprio_dim={proprio_dim}"
        )
    features = getattr(getattr(dataset, "meta", None), "features", {})
    state_feature = features.get("observation.state", {})
    source_names = tuple(state_feature.get("names") or ())
    if len(source_names) != len(set(source_names)):
        raise ValueError("online dataset observation.state names must be unique")
    missing = [name for name in state_names if name not in source_names]
    if missing:
        raise ValueError(
            "online dataset observation.state is missing checkpoint fields: "
            f"{missing}"
        )
    return tuple(source_names.index(name) for name in state_names)


def encode_saved_episode(
    *,
    dataset: Any,
    episode_id: int,
    episode_start: int,
    episode_stop: int,
    episode_success: bool,
    warmup: bool,
    pi05: Any,
    rl_token: Any,
    preprocessor: Any,
    capture: PrefixOutputCapture,
    chunk_length: int,
    stride: int,
    action_dim: int,
    proprio_dim: int,
    batch_size: int,
    num_workers: int,
    device: str,
    task_instruction: str | None,
    state_indices: tuple[int, ...] | None = None,
    fps: float | None = None,
    require_timing: bool = False,
    require_full_episode: bool = False,
) -> tuple[list[ReplayTransitionRecord], int]:
    annotations = load_frame_annotations(dataset, episode_start, episode_stop)
    if require_full_episode and (warmup or any(
        float(row["phase"]) != 1.0
        or (bool(row.get("action_valid", 1)) and int(row["collector_policy_id"]) not in {COLLECTOR_HUMAN, COLLECTOR_RLT_ACTOR})
        for row in annotations.values()
    )):
        raise ValueError("full-episode RLT requires phase=1 and Actor/Human commands throughout")
    if require_timing and any("action_valid" not in row for row in annotations.values()):
        raise ValueError("legacy online episode has no causal timing contract")
    specs = build_critical_transition_specs(
        annotations,
        episode_start=episode_start,
        episode_stop=episode_stop,
        chunk_length=chunk_length,
        stride=stride,
        fps=fps,
    )
    if sum(spec.terminal for spec in specs) > 1:
        raise ValueError(
            "online RLT requires one contiguous critical phase per episode"
        )
    warmup_steps = sum(
        float(annotations[frame]["phase"]) > 0.5 and bool(annotations[frame].get("action_valid", 1))
        for frame in range(episode_start, episode_stop)
    )
    if not specs:
        return [], int(warmup_steps)

    if require_timing:
        dataset = _TimedActionDataset(dataset, annotations, chunk_length, action_dim)

    encode_indices = sorted(
        {value for spec in specs for value in (spec.anchor_frame, spec.next_frame)}
    )
    loader = DataLoader(
        Subset(dataset, encode_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=False,
    )
    encoded: dict[int, dict[str, Tensor]] = {}
    offset = 0
    for batch in loader:
        raw_state = batch["observation.state"]
        count = int(raw_state.shape[0])
        raw_indices = encode_indices[offset : offset + count]
        offset += count
        if state_indices is not None:
            if len(state_indices) != proprio_dim or not state_indices:
                raise ValueError("online state indices must match proprio_dim")
            if min(state_indices) < 0 or max(state_indices) >= raw_state.shape[-1]:
                raise ValueError(
                    "online state index is outside the dataset state dimension: "
                    f"indices={state_indices} state_dim={raw_state.shape[-1]}"
                )
            batch["observation.state"] = raw_state[..., list(state_indices)]
        elif raw_state.shape[-1] != proprio_dim:
            raise ValueError(
                "online dataset state dimension does not match policy proprio_dim and "
                "no state field mapping was provided: "
                f"dataset={raw_state.shape[-1]} policy={proprio_dim}"
            )
        if task_instruction is not None:
            batch["task"] = [task_instruction] * count
        elif "task" not in batch:
            raise ValueError("online dataset batch has no task text")
        processed = preprocessor(batch)
        if "action" not in processed:
            raise KeyError("online SFT preprocessor output has no action chunk")
        with torch.no_grad():
            vla_chunk = pi05.predict_action_chunk(processed)
            prefix = capture.consume()
            latent = rl_token.encode(prefix.to(torch.float32))
        state = torch.cat(
            [
                latent.detach().cpu(),
                processed["observation.state"][:, :proprio_dim].detach().cpu(),
            ],
            dim=-1,
        )
        actions = processed["action"][:, :chunk_length, :action_dim].detach().cpu()
        refs = vla_chunk[:, :chunk_length, :action_dim].detach().cpu()
        if actions.shape[1:] != refs.shape[1:] or actions.shape[1] != chunk_length:
            raise ValueError(
                f"online action/reference shape mismatch: {tuple(actions.shape)} vs {tuple(refs.shape)}"
            )
        for local, raw_index in enumerate(raw_indices):
            encoded[raw_index] = {
                "state_vec": state[local],
                "exec_chunk": actions[local],
                "ref_chunk": refs[local],
            }
            if 'observation.anchor' in batch:
                encoded[raw_index]['action_anchor'] = batch['observation.anchor'][local].detach().cpu().clone()
                encoded[raw_index]['absolute_exec_chunk'] = batch['action'][local, :chunk_length, :action_dim].detach().cpu().clone()

    if offset != len(encode_indices):
        raise ValueError(
            f"encoded {offset} online anchors, expected {len(encode_indices)}"
        )

    records: list[ReplayTransitionRecord] = []
    for spec in specs:
        current = encoded[spec.anchor_frame]
        following = encoded[spec.next_frame]
        exec_chunk = current["exec_chunk"].clone()
        ref_chunk = supervision_reference(
            exec_chunk, current["ref_chunk"], annotations, spec.anchor_frame, spec.actual_steps,
        )
        if spec.actual_steps < chunk_length:
            exec_chunk[spec.actual_steps :].zero_()
            ref_chunk[spec.actual_steps :].zero_()
        reward = torch.zeros(chunk_length, dtype=torch.float32)
        if spec.terminal and episode_success:
            reward[spec.actual_steps - 1] = 1.0
        source, intervention = _transition_source(annotations, spec, warmup=warmup)
        records.append(
            ReplayTransitionRecord(
                anchor_frame=spec.anchor_frame,
                transition={
                    "state_vec": current["state_vec"],
                    "exec_chunk": exec_chunk,
                    "ref_chunk": ref_chunk,
                    "reward_seq": reward,
                    "next_state_vec": following["state_vec"],
                    "next_ref_chunk": following["ref_chunk"],
                **({'action_anchor': current['action_anchor'], 'next_action_anchor': following['action_anchor'],
                    'absolute_exec_chunk': current['absolute_exec_chunk']} if 'action_anchor' in current else {}),
                    "done": torch.tensor(float(spec.terminal)),
                    "intervention": torch.tensor(intervention),
                    "actual_steps": torch.tensor(spec.actual_steps, dtype=torch.int64),
                    "source": torch.tensor(source, dtype=torch.int64),
                    "episode_id": torch.tensor(episode_id, dtype=torch.int64),
                    "is_critical": torch.tensor(1.0),
                },
            )
        )
    return records, int(warmup_steps)


class OnlineUpdateScheduler:
    def __init__(
        self,
        *,
        policy: Any,
        replay: PersistentReplayBuffer,
        config: OnlineRLTTrainingConfig,
    ) -> None:
        self.policy = policy
        self.replay = replay
        self.config = config
        self.device = torch.device(policy.config.device)
        self.actor_optimizer = torch.optim.AdamW(
            policy.actor.parameters(),
            lr=config.actor_lr,
            weight_decay=config.weight_decay,
        )
        self.critic_optimizer = torch.optim.AdamW(
            policy.critic.parameters(),
            lr=config.critic_lr,
            weight_decay=config.weight_decay,
        )
        self.consumed_credits = replay.consumed_credits
        self.critic_updates = replay.critic_updates
        self.actor_updates = replay.actor_updates

    def _batch_to_device(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        tx = self.policy._coerce_batch(batch)
        return {key: value.to(self.device) for key, value in tx.items()}

    def drain(
        self,
        progress_callback: (
            Callable[[dict[str, float | int | str | None]], None] | None
        ) = None,
    ) -> dict[str, float | int]:
        if getattr(self, "faulted", False):
            raise RuntimeError("online scheduler faulted; restore a complete checkpoint before training")
        critic_total = 0.0
        actor_total = 0.0
        actor_count = 0
        initial = self.consumed_credits
        pending = self.replay.produced_credits - initial
        progress = _OnlineTrainingProgress(pending, callback=progress_callback)
        latest_actor_loss: float | None = None
        self.policy.train()
        if pending > 0:
            progress.update(
                0,
                critic_loss_value=None,
                actor_updates=0,
                actor_loss_value=None,
            )
        try:
            while self.consumed_credits < self.replay.produced_credits:
                batch_size = min(self.config.batch_size, len(self.replay))
                if batch_size == 0:
                    raise RuntimeError(
                        "replay credits exist but the online replay is empty"
                    )
                tx = self._batch_to_device(self.replay.sample(batch_size))
                self.critic_optimizer.zero_grad(set_to_none=True)
                value = critic_loss(
                    self.policy.critic,
                    self.policy.target_critic,
                    self.policy.actor,
                    tx,
                    gamma=self.policy.config.gamma,
                    C=self.policy.config.chunk_length,
                    target_q_clip=self.policy.config.target_q_clip,
                )
                if not torch.isfinite(value).all():
                    raise RuntimeError("non-finite online Critic loss")
                value.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.critic.parameters(), self.config.grad_clip_norm or float("inf"),
                    error_if_nonfinite=True,
                )
                self.critic_optimizer.step()
                soft_update(
                    self.policy.target_critic,
                    self.policy.critic,
                    self.policy.config.tau,
                )
                latest_critic_loss = float(value.detach())
                critic_total += latest_critic_loss

                if (self.critic_updates + 1) % 2 == 0:
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_value = self.policy._actor_loss_without_critic_grads(tx)
                    if not torch.isfinite(actor_value).all():
                        raise RuntimeError("non-finite online Actor loss")
                    actor_value.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.actor.parameters(), self.config.grad_clip_norm or float("inf"),
                        error_if_nonfinite=True,
                    )
                    self.actor_optimizer.step()
                    self.actor_updates += 1
                    actor_count += 1
                    latest_actor_loss = float(actor_value.detach())
                    actor_total += latest_actor_loss
                self.consumed_credits += 1
                self.critic_updates += 1
                self.policy._critic_step.fill_(self.critic_updates)
                progress.update(
                    self.consumed_credits - initial,
                    critic_loss_value=latest_critic_loss,
                    actor_updates=actor_count,
                    actor_loss_value=latest_actor_loss,
                )
        except BaseException:
            self.faulted = True
            raise
        finally:
            progress.close()
            self.policy.eval()
        updates = self.consumed_credits - initial
        self.policy.eval()
        return {
            "critic_updates": updates,
            "actor_updates": actor_count,
            "mean_critic_loss": critic_total / max(updates, 1),
            "mean_actor_loss": actor_total / max(actor_count, 1),
        }


class OnlineRLTSession:
    """Sequential single-GPU online RLT: collect, encode, update, checkpoint."""

    @cleanup_on_init_error
    def __init__(
        self,
        *,
        policy: Any,
        dataset: Any,
        fps: float,
        task_instruction: str,
        config: OnlineRLTTrainingConfig,
        resume: bool,
        state_feature_names: tuple[str, ...] | None = None,
        runtime_contract: dict[str, Any] | None = None,
    ) -> None:
        from vla.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
        from vla.adapters.lerobot.policies.processor_rlt_token import (
            make_rlt_token_pre_post_processors,
        )

        if not isinstance(policy, ChunkACPolicy):
            raise TypeError("online RLT requires ChunkACPolicy")
        config.validate(
            chunk_length=policy.config.chunk_length,
            actor_update_interval=policy.config.actor_update_interval,
        )
        self.policy = policy
        self.dataset = dataset
        self.fps = float(fps)
        self.task_instruction = task_instruction
        self.state_feature_names = (
            None if state_feature_names is None else tuple(state_feature_names)
        )
        self.config = config
        self.root = Path(config.replay_root).expanduser().resolve()
        self.status = "RECOVERING"
        self.fault_reason: str | None = None
        training_contract = {key: value for key, value in asdict(config).items() if key != "replay_root"}
        if not config.full_episode:
            training_contract.pop("full_episode")
        # Keep existing scratch-run contracts byte-for-byte compatible.
        if not config.inherit_actor_critic:
            training_contract.pop("inherit_actor_critic")
        elif not (runtime_contract or {}).get("checkpoint_fingerprint"):
            raise ValueError("inherited Actor/Critic requires a source checkpoint fingerprint")
        if config.inherit_actor_critic:
            for module in (policy.actor, policy.critic, policy.target_critic):
                if any(not torch.isfinite(value).all() for value in module.state_dict().values()):
                    raise ValueError("source Actor/Critic checkpoint contains non-finite weights")
        contract = {
            "training_semantics_version": TRAINING_SEMANTICS_VERSION,
            "recording_schema_version": RECORDING_SCHEMA_VERSION,
            "task": task_instruction,
            "state_feature_names": self.state_feature_names,
            "training": training_contract,
            "policy": asdict(policy.config),
            "runtime": runtime_contract or {},
            "step_period_bounds": [MIN_STEP_PERIODS, MAX_STEP_PERIODS],
            "rl_token_sha256": model_digest(policy.config.rl_token_pretrained_path),
            "vla_sha256": model_digest(policy.config.vla_pretrained_path),
            "fps": self.fps,
            "chunk_length": policy.config.chunk_length,
            "stride": config.frame_stride,
            "action_dim": policy.config.action_dim,
            "proprio_dim": policy.config.proprio_dim,
            "rl_token_dim": policy.config.rl_token_dim,
            "gamma": policy.config.gamma,
            "beta": policy.config.beta,
            "tau": policy.config.tau,
            "utd_ratio": config.utd_ratio,
            "actor_update_interval": policy.config.actor_update_interval,
            "rl_token_pretrained_path": str(policy.config.rl_token_pretrained_path),
            "vla_pretrained_path": str(policy.config.vla_pretrained_path),
        }
        self._writer_lease = WriterLease(dataset.root, self.root)
        if not resume and (self.root / "replay.sqlite3").exists():
            raise FileExistsError("online replay already exists; use a new dataset/replay pair")
        self.run = bind_run(Path(dataset.root), contract, resume=resume)
        self.replay = PersistentReplayBuffer(
            self.root,
            capacity=config.replay_capacity,
            contract=self.run,
            resume=resume,
            seed=config.seed,
            writer_lease=self._writer_lease,
        )
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        if not resume or self.replay.active_checkpoint is None:
            if config.inherit_actor_critic:
                # Transfer network weights, not offline optimizer/update history.
                policy._critic_step.zero_()
                policy.modifier = None
                policy.target_critic.eval()
                for parameter in policy.target_critic.parameters():
                    parameter.requires_grad_(False)
            else:
                self._initialize_actor_critic_from_scratch()
        self.scheduler = OnlineUpdateScheduler(
            policy=policy, replay=self.replay, config=config
        )
        if self.replay.active_checkpoint is None:
            if self.replay.produced_credits:
                raise ValueError("replay has credits but no complete initial checkpoint")
            self._checkpoint_scheduler_progress()

        rl_token_policy = policy._ensure_rl_token_policy()
        self.rl_token_policy = rl_token_policy
        self.preprocessor, _ = make_rlt_token_pre_post_processors(
            config=rl_token_policy.config
        )
        self.capture = PrefixOutputCapture(
            token_pool_size=policy.config.token_pool_size,
            image_only=policy.config.image_only,
            num_image_tokens=policy._compute_num_image_tokens(),
            num_per_camera=policy.config.num_per_camera,
            active_camera_indices=policy.config.active_camera_indices,
        )
        self.capture.attach(rl_token_policy._pi05)
        self._dataset_reader: Any | None = None
        self._dataset_episode_count = -1
        self._dataset_requested_episode_id: int | None = None
        self._episode_warmup = self.is_warmup
        self.set_episode_mode()
        self.recover()

    def require_ready(self) -> None:
        if self.status != "READY":
            raise RuntimeError(f"online training is {self.status}: {self.fault_reason or 'not ready'}")

    def _fault(self, error: BaseException) -> None:
        self.status = "FAULTED"
        self.fault_reason = f"{type(error).__name__}: {error}"
        self.policy.eval()

    def _saved_rows(self) -> list[dict[str, Any]]:
        import pandas as pd
        from vla.datacollection.franka_robotiq_single_left.v30_writer import EPISODES_PATH

        if load_run(self.dataset.root) != self.run:
            raise ValueError("online dataset identity/contract changed")
        path = Path(self.dataset.root) / EPISODES_PATH
        rows = pd.read_parquet(path).sort_values("episode_index").to_dict("records") if path.is_file() else []
        keys = set()
        for index, row in enumerate(rows):
            if int(row["episode_index"]) != index:
                raise ValueError("online dataset episodes are not contiguous")
            for field in ("run_uuid", "dataset_uuid"):
                if row.get(field) != self.run[field]:
                    raise ValueError(f"episode {index} has mismatched {field}")
            uuid.UUID(row["episode_uuid"])
            key = self._episode_key(row)
            if key in keys:
                raise ValueError("duplicate online episode UUID")
            keys.add(key)
            stored_digest = self.replay.episode_digest(key)
            if self.replay.has_episode(key) and stored_digest != self._episode_digest(row):
                raise ValueError(f"episode {index} metadata changed after Replay commit")
            if row.get("episode_success") not in {"success", "failure"}:
                raise ValueError(f"episode {index} has no outcome")
            files = json.loads(row["online_files"])
            if not files:
                raise ValueError(f"episode {index} has no committed file checksums")
            for relative, expected in files.items():
                target = (Path(self.dataset.root) / relative).resolve()
                if not target.is_relative_to(Path(self.dataset.root).resolve()):
                    raise ValueError("episode file escapes dataset root")
                if not target.is_file() or not self._verify_file(target, expected):
                    raise ValueError(f"episode {index} missing or changed file: {relative}")
        if self.replay.episode_keys() - keys:
            raise ValueError("Replay references missing Dataset episodes")
        return rows

    def _verify_file(self, path: Path, expected: str) -> bool:
        # Rehash on recovery or any filesystem change, not every new episode.
        cache = getattr(self, "_verified_files", None)
        if cache is None:
            self._verified_files = cache = {}
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if cache.get(path) == (signature, expected):
            return True
        if file_digest(path) != expected:
            return False
        after = path.stat()
        if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            return False
        cache[path] = (signature, expected)
        return True

    def _episode_key(self, row: dict[str, Any]) -> str:
        return f"{self.run['dataset_uuid']}:{row['episode_uuid']}"

    @staticmethod
    def _episode_digest(row: dict[str, Any]) -> str:
        fields = ("run_uuid", "dataset_uuid", "episode_uuid", "episode_index", "episode_success", "online_warmup", "policy_revision", "online_files")
        return hashlib.sha256(json.dumps({key: row[key] for key in fields}, sort_keys=True).encode()).hexdigest()

    def _drain_pending(self, progress_callback=None) -> dict[str, Any]:
        metrics = {}
        can_train = self.replay.actor_enabled or (
            self.replay.trainable_env_steps >= self.config.warmup_steps
            and len(self.replay) >= self.config.batch_size
        )
        if self.config.full_episode:
            can_train = len(self.replay) >= self.config.batch_size
        if can_train:
            if self.scheduler.consumed_credits < self.replay.produced_credits:
                metrics = self.scheduler.drain(progress_callback=progress_callback)
            if progress_callback:
                progress_callback({"stage": "checkpoint"})
            if metrics or (not self.replay.actor_enabled and self.scheduler.actor_updates > 0):
                self._checkpoint_scheduler_progress(actor_enabled=self.scheduler.actor_updates > 0)
        self.set_episode_mode()
        return metrics

    def _reconcile(self, progress_callback=None) -> None:
        rows = self._saved_rows()
        self._drain_pending(progress_callback)
        for row in rows:
            if not self.replay.has_episode(self._episode_key(row)):
                self._process_saved_row(row, progress_callback)

    def begin_episode(self) -> dict[str, Any]:
        self.require_ready()
        self.status = "PROCESSING"
        try:
            self.policy.quiesce_for_online_training()
            self._reconcile()
            self.status = "READY"
            return {
                "run_uuid": self.run["run_uuid"],
                "dataset_uuid": self.run["dataset_uuid"],
                "episode_uuid": str(uuid.uuid4()),
                "online_warmup": self.is_warmup,
                "trainable_env_steps": self.replay.trainable_env_steps,
                "warmup_steps": self.config.warmup_steps,
                "policy_revision": self.replay.active_checkpoint,
            }
        except BaseException as error:
            self._fault(error)
            raise

    def recover(self, progress_callback=None) -> dict[str, Any]:
        self.status = "RECOVERING"
        self._verified_files = {}
        try:
            if progress_callback:
                progress_callback({"stage": "recovery verification"})
            self.policy.quiesce_for_online_training()
            self.replay.reload()
            contract = self.run["contract"]
            for prefix in ("rl_token", "vla"):
                if f"{prefix}_sha256" in contract and model_digest(contract[f"{prefix}_pretrained_path"]) != contract[f"{prefix}_sha256"]:
                    raise ValueError(f"{prefix} model content changed; recovery refused")
            self._saved_rows()
            if not self.replay.active_checkpoint:
                raise ValueError("no complete checkpoint available for recovery")
            self._restore_checkpoint(self.replay.active_checkpoint)
            self.policy.reset()
            self._reconcile(progress_callback)
            self.policy.eval()
            self.fault_reason = None
            self.status = "READY"
        except Exception as error:
            self._fault(error)
            log.exception("online recovery failed; inference remains disabled")
        return {"online_status": self.status, "online_fault": self.fault_reason}

    @property
    def is_warmup(self) -> bool:
        return not self.config.full_episode and not self.replay.actor_enabled

    def _initialize_actor_critic_from_scratch(self) -> None:
        cfg = self.policy.config
        state_dim = cfg.rl_token_dim + cfg.proprio_dim
        chunk_dim = cfg.chunk_length * cfg.action_dim
        self.policy.actor = ChunkActor(
            state_dim=state_dim,
            chunk_dim=chunk_dim,
            hidden_dim=cfg.actor_hidden_dim,
            num_layers=cfg.actor_num_layers,
            fixed_std=cfg.actor_fixed_std,
            ref_dropout_p=cfg.actor_ref_dropout_p,
            activation=cfg.actor_activation,
            layer_norm=cfg.actor_layer_norm,
            residual=cfg.actor_residual,
        ).to(cfg.device)
        self.policy.critic = TwinCritic(
            state_dim=state_dim,
            chunk_dim=chunk_dim,
            hidden_dim=cfg.critic_hidden_dim,
            num_layers=cfg.critic_num_layers,
            activation=cfg.critic_activation,
            layer_norm=cfg.critic_layer_norm,
            residual=cfg.critic_residual,
        ).to(cfg.device)
        self.policy.target_critic = copy.deepcopy(self.policy.critic).eval()
        for parameter in self.policy.target_critic.parameters():
            parameter.requires_grad_(False)
        self.policy._critic_step.zero_()
        self.policy.modifier = None

    def set_episode_mode(self) -> bool:
        self._episode_warmup = self.is_warmup
        self.policy.set_online_warmup(self._episode_warmup)
        return self._episode_warmup

    def _fresh_dataset(self, *, episode_id: int | None = None) -> Any:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from vla.datacollection.franka_robotiq_single_left.v30_writer import repair_committed_metadata

        # Reuse the reader between episodes. The writer appends metadata after
        # each commit, so refresh only when the requested episode is not visible.
        requested = self._dataset_requested_episode_id if episode_id is None else episode_id
        if self._dataset_reader is not None and (
            requested is None or requested < self._dataset_episode_count
        ):
            return self._dataset_reader
        repair_committed_metadata(Path(self.dataset.root))
        reader = LeRobotDataset(
            repo_id=self.dataset.repo_id,
            root=self.dataset.root,
            tolerance_s=self.config.tolerance_s,
            video_backend=self.config.video_backend,
        )
        self._dataset_reader = reader
        self._dataset_episode_count = int(reader.num_episodes)
        return reader

    @staticmethod
    def _episode_value(episodes: Any, key: str, episode_id: int) -> Any:
        column = episodes[key]
        return (
            column.iloc[episode_id] if hasattr(column, "iloc") else column[episode_id]
        )

    def process_saved_episode(
        self,
        *,
        episode_id: int,
        outcome: str,
        progress_callback: (
            Callable[[dict[str, float | int | str | None]], None] | None
        ) = None,
    ) -> dict[str, Any]:
        self.require_ready()
        if outcome not in {"success", "failure"}:
            raise ValueError("online RLT requires an explicit success/failure outcome")
        self.status = "PROCESSING"
        try:
            self.policy.quiesce_for_online_training()
            rows = self._saved_rows()
            if episode_id < 0 or episode_id >= len(rows):
                raise ValueError("not a committed Dataset episode")
            row = rows[episode_id]
            if row["episode_success"] != outcome:
                raise ValueError("episode outcome differs from durable Dataset outcome")
            duplicate = self.replay.has_episode(self._episode_key(row))
            metrics = self._drain_pending(progress_callback)
            for candidate in rows:
                if not self.replay.has_episode(self._episode_key(candidate)):
                    result = self._process_saved_row(candidate, progress_callback)
                    if int(candidate["episode_index"]) == episode_id:
                        metrics.update(result)
            self.status = "READY"
            return {"episode_id": episode_id, "transitions": 0, **metrics, "duplicate": duplicate}
        except BaseException as error:
            self._fault(error)
            raise

    def _process_saved_row(self, row: dict[str, Any], progress_callback=None) -> dict[str, Any]:
        episode_id = int(row["episode_index"])
        if progress_callback:
            progress_callback({"stage": "encoding", "episode_id": episode_id})
        outcome = row["episode_success"]
        warmup = bool(row["online_warmup"])
        if warmup != self.is_warmup:
            raise ValueError("saved episode warmup mode disagrees with ordered replay history")
        if row["policy_revision"] != self.replay.active_checkpoint:
            # A recovery checkpoint can have a different name with identical progress.
            if Path(row["policy_revision"]).name != row["policy_revision"]:
                raise ValueError("invalid collection policy revision")
            revision = self.root / "checkpoints" / row["policy_revision"] / "training_state.pt"
            metadata = json.loads((revision.parent / "metadata.json").read_text())
            if metadata.get("sha256", {}).get("training_state.pt") != file_digest(revision):
                raise ValueError("collection policy checkpoint checksum mismatch")
            state = torch.load(revision, map_location="cpu", weights_only=False)
            if int(state["consumed_credits"]) != self.scheduler.consumed_credits:
                raise ValueError("saved episode was collected with a different policy revision")
        episode_key = self._episode_key(row)
        self._dataset_requested_episode_id = episode_id
        readable = self._fresh_dataset()
        state_indices = resolve_state_indices(
            readable,
            self.state_feature_names,
            self.policy.config.proprio_dim,
        )
        episodes = readable.meta.episodes
        start = int(self._episode_value(episodes, "dataset_from_index", episode_id))
        stop = int(self._episode_value(episodes, "dataset_to_index", episode_id))
        records, critical_steps = encode_saved_episode(
            dataset=readable,
            episode_id=episode_id,
            episode_start=start,
            episode_stop=stop,
            episode_success=outcome == "success",
            warmup=warmup,
            pi05=self.rl_token_policy._pi05,
            rl_token=self.rl_token_policy.rl_token,
            preprocessor=self.preprocessor,
            capture=self.capture,
            chunk_length=self.policy.config.chunk_length,
            stride=self.config.frame_stride,
            action_dim=self.policy.config.action_dim,
            proprio_dim=self.policy.config.proprio_dim,
            batch_size=self.config.encoding_batch_size,
            num_workers=self.config.encoding_num_workers,
            device=self.policy.config.device,
            task_instruction=self.task_instruction,
            state_indices=state_indices,
            fps=self.fps,
            require_timing=True,
            require_full_episode=self.config.full_episode,
        )
        committed = self.replay.commit_episode(
            episode_key=episode_key,
            episode_id=episode_id,
            outcome=outcome,
            warmup=warmup,
            warmup_env_steps=critical_steps,
            records=records,
            utd_ratio=self.config.utd_ratio,
            checkpoint=self._save_checkpoint(),
            episode_digest=self._episode_digest(row),
        )
        metrics: dict[str, Any] = {
            "episode_id": episode_id,
            "duplicate": False,
            "transitions": committed,
            "critical_steps": critical_steps,
            "trainable_env_steps": self.replay.trainable_env_steps,
            "warmup": warmup,
        }
        metrics.update(self._drain_pending(progress_callback))
        log.info("online RLT episode complete: %s", metrics)
        return metrics

    def _checkpoint_scheduler_progress(self, *, actor_enabled: bool | None = None) -> None:
        if getattr(self.scheduler, "faulted", False):
            raise RuntimeError("cannot checkpoint a partially updated model")
        checkpoint = self._save_checkpoint(actor_enabled=actor_enabled)
        self.replay.checkpoint_progress(
            checkpoint=checkpoint,
            consumed_credits=self.scheduler.consumed_credits,
            critic_updates=self.scheduler.critic_updates,
            actor_updates=self.scheduler.actor_updates,
            actor_enabled=actor_enabled,
        )

    def _model_state(self) -> dict[str, Tensor]:
        tensors: dict[str, Tensor] = {}
        for prefix, module in (
            ("actor", self.policy.actor),
            ("critic", self.policy.critic),
            ("target_critic", self.policy.target_critic),
        ):
            tensors.update(
                {
                    f"{prefix}.{name}": value.detach().contiguous().cpu()
                    for name, value in module.state_dict().items()
                }
            )
        tensors["critic_step"] = self.policy._critic_step.detach().contiguous().cpu()
        return tensors

    def _save_checkpoint(self, *, actor_enabled: bool | None = None) -> str:
        checkpoints = self.root / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        name = f"checkpoint-{time.time_ns()}"
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=checkpoints))
        save_file(self._model_state(), temporary / "model.safetensors")
        torch.save(
            {
                "training_semantics_version": TRAINING_SEMANTICS_VERSION,
                "actor_enabled": self.replay.actor_enabled if actor_enabled is None else actor_enabled,
                "actor_optimizer": self.scheduler.actor_optimizer.state_dict(),
                "critic_optimizer": self.scheduler.critic_optimizer.state_dict(),
                "consumed_credits": self.scheduler.consumed_credits,
                "critic_updates": self.scheduler.critic_updates,
                "actor_updates": self.scheduler.actor_updates,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "python_rng_state": random.getstate(),
                "replay_rng_state": self.replay.get_rng_state(),
            },
            temporary / "training_state.pt",
        )
        (temporary / "metadata.json").write_text(
            json.dumps({
                "format_version": 2, "complete": True,
                "sha256": {name: file_digest(temporary / name) for name in ("model.safetensors", "training_state.pt")},
            }, sort_keys=True),
            encoding="utf-8",
        )
        for path in temporary.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        fsync_directory(temporary)
        os.replace(temporary, checkpoints / name)
        fsync_directory(checkpoints)
        return name

    @staticmethod
    def _module_state(tensors: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
        marker = prefix + "."
        return {
            key.removeprefix(marker): value
            for key, value in tensors.items()
            if key.startswith(marker)
        }

    def _restore_checkpoint(self, name: str) -> None:
        path = self.root / "checkpoints" / name
        if Path(name).name != name:
            raise ValueError("invalid checkpoint name")
        if not (path / "metadata.json").is_file():
            raise FileNotFoundError(f"online checkpoint is incomplete: {path}")
        metadata = json.loads((path / "metadata.json").read_text())
        if metadata.get("format_version") != 2 or metadata.get("complete") is not True:
            raise ValueError("unsupported or incomplete online checkpoint")
        for filename in ("model.safetensors", "training_state.pt"):
            if metadata.get("sha256", {}).get(filename) != file_digest(path / filename):
                raise ValueError(f"online checkpoint checksum mismatch: {filename}")
        tensors = load_file(
            path / "model.safetensors", device=str(self.scheduler.device)
        )
        self.policy.actor.load_state_dict(
            self._module_state(tensors, "actor"), strict=True
        )
        self.policy.critic.load_state_dict(
            self._module_state(tensors, "critic"), strict=True
        )
        self.policy.target_critic.load_state_dict(
            self._module_state(tensors, "target_critic"), strict=True
        )
        self.policy._critic_step.copy_(
            tensors["critic_step"].to(self.policy._critic_step.device)
        )
        state = torch.load(
            path / "training_state.pt", map_location="cpu", weights_only=False
        )
        if state.get("training_semantics_version") != TRAINING_SEMANTICS_VERSION:
            raise ValueError("checkpoint training semantics require explicit upgrade")
        if state.get("actor_enabled") != self.replay.actor_enabled:
            raise ValueError("checkpoint/replay Actor readiness mismatch")
        counters = (
            state["consumed_credits"],
            state["critic_updates"],
            state["actor_updates"],
        )
        expected = (
            self.replay.consumed_credits,
            self.replay.critic_updates,
            self.replay.actor_updates,
        )
        if counters != expected:
            raise ValueError(
                f"checkpoint/replay progress mismatch: checkpoint={counters}, replay={expected}"
            )
        if counters[2] != counters[1] // 2 or counters[0] != counters[1]:
            raise ValueError("checkpoint has incomplete Actor/Critic update counts")
        self.scheduler.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.scheduler.critic_optimizer.load_state_dict(state["critic_optimizer"])
        torch.set_rng_state(state["torch_rng_state"])
        if state["cuda_rng_state"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        random.setstate(state["python_rng_state"])
        self.replay.set_rng_state(state["replay_rng_state"])
        self.scheduler.consumed_credits, self.scheduler.critic_updates, self.scheduler.actor_updates = counters
        self.scheduler.faulted = False
        self.policy.eval()

    def close(self) -> None:
        try:
            if getattr(self, "capture", None) is not None:
                self.capture.detach()
        finally:
            try:
                if getattr(self, "replay", None) is not None:
                    self.replay.close()
            finally:
                if getattr(self, "_writer_lease", None) is not None:
                    self._writer_lease.close()
