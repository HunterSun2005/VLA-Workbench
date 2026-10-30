from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from vla.core.writer_lock import WriterLease, cleanup_on_init_error


ONLINE_REPLAY_FORMAT_VERSION = 3


@dataclass(frozen=True)
class ReplayTransitionRecord:
    anchor_frame: int
    transition: dict[str, torch.Tensor]


class PersistentReplayBuffer:
    """Crash-recoverable replay storage with an in-memory sampling mirror."""

    @cleanup_on_init_error
    def __init__(
        self,
        root: str | Path,
        *,
        capacity: int,
        contract: dict[str, Any],
        resume: bool,
        seed: int,
        writer_lease: WriterLease | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.root = Path(root).expanduser().resolve()
        publication = contract.get("upgrade_publication")
        if publication is not None and (
            str(self.root) != publication["replay_root"]
            or not (self.root.parent / "upgrade_report.json").is_file()
        ):
            raise ValueError("unpublished or relocated training upgrade")
        self._owned_lease = WriterLease(self.root) if writer_lease is None else None
        (writer_lease or self._owned_lease).require(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "replay.sqlite3"
        existed = self.path.exists()
        if existed and not resume:
            raise FileExistsError(
                f"online replay already exists; enable resume or use a new directory: {self.path}"
            )
        if resume and not existed:
            raise FileNotFoundError(f"cannot resume without online replay: {self.path}")

        self.capacity = int(capacity)
        self._rng = random.Random(seed)
        self._connection = sqlite3.connect(self.path)
        if existed:
            if int(self._get_state("format_version")) != ONLINE_REPLAY_FORMAT_VERSION:
                raise ValueError("unsupported online replay format; run upgrade_rlt_training explicitly")
        else:
            self._create_schema()
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        canonical_contract = json.dumps(contract, sort_keys=True, separators=(",", ":"))
        if existed:
            stored = self._get_state("contract")
            if stored != canonical_contract:
                raise ValueError("online replay contract does not match this run")
            stored_capacity = int(self._get_state("capacity"))
            if stored_capacity != self.capacity:
                raise ValueError(
                    f"online replay capacity mismatch: stored={stored_capacity}, configured={self.capacity}"
                )
        else:
            with self._connection:
                self._set_state("format_version", str(ONLINE_REPLAY_FORMAT_VERSION))
                self._set_state("contract", canonical_contract)
                self._set_state("capacity", str(self.capacity))
                self._set_state("produced_credits", "0")
                self._set_state("consumed_credits", "0")
                self._set_state("warmup_env_steps", "0")
                self._set_state("trainable_env_steps", "0")
                self._set_state("actor_enabled", "0")
                self._set_state("critic_updates", "0")
                self._set_state("actor_updates", "0")
                self._set_state("active_checkpoint", "")
        if int(self._get_state("format_version")) != ONLINE_REPLAY_FORMAT_VERSION:
            raise ValueError("unsupported online replay format")
        self._entries = self._load_entries()
        self._rebuild_tensor_storage()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS episodes (
                episode_key TEXT PRIMARY KEY,
                episode_id INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                warmup INTEGER NOT NULL,
                transition_count INTEGER NOT NULL,
                episode_digest TEXT
            );
            CREATE TABLE IF NOT EXISTS transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_key TEXT NOT NULL,
                anchor_frame INTEGER NOT NULL,
                payload BLOB NOT NULL,
                UNIQUE(episode_key, anchor_frame)
            );
            CREATE INDEX IF NOT EXISTS transitions_episode_idx
                ON transitions(episode_key);
            """
        )
        self._connection.commit()

    def _get_state(self, key: str) -> str:
        row = self._connection.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        if row is None:
            raise ValueError(f"online replay is missing state key {key!r}")
        return str(row[0])

    def _set_state(self, key: str, value: str) -> None:
        self._connection.execute(
            "INSERT INTO state(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    @staticmethod
    def _serialize(transition: dict[str, torch.Tensor]) -> bytes:
        tensors = {
            key: value.detach().contiguous().cpu()
            for key, value in transition.items()
            if isinstance(value, torch.Tensor)
        }
        if set(tensors) != set(transition):
            raise TypeError("online replay transitions may only contain tensors")
        return save_safetensors(tensors)

    @staticmethod
    def _deserialize(payload: bytes) -> dict[str, torch.Tensor]:
        return load_safetensors(payload)

    def _load_entries(self) -> list[dict[str, torch.Tensor]]:
        rows = self._connection.execute(
            "SELECT payload FROM transitions ORDER BY id ASC"
        ).fetchall()
        return [self._deserialize(row[0]) for row in rows]

    def _rebuild_tensor_storage(self) -> None:
        """Keep replay samples in contiguous per-field tensors for fast batching."""
        if not self._entries:
            self._storage: dict[str, torch.Tensor] = {}
            return
        keys = tuple(self._entries[0])
        key_set = set(keys)
        if any(set(entry) != key_set for entry in self._entries[1:]):
            raise ValueError("replay entries do not share the same tensor keys")
        self._storage = {
            key: torch.stack([entry[key] for entry in self._entries])
            for key in keys
        }
        # Keep the compatibility list, but make each entry a view of storage.
        self._entries = [
            {key: value[index] for key, value in self._storage.items()}
            for index in range(len(self._entries))
        ]

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def produced_credits(self) -> int:
        return int(self._get_state("produced_credits"))

    @property
    def consumed_credits(self) -> int:
        return int(self._get_state("consumed_credits"))

    @property
    def warmup_env_steps(self) -> int:
        return int(self._get_state("warmup_env_steps"))

    @property
    def trainable_env_steps(self) -> int:
        return int(self._get_state("trainable_env_steps"))

    @property
    def actor_enabled(self) -> bool:
        return bool(int(self._get_state("actor_enabled")))

    @property
    def critic_updates(self) -> int:
        return int(self._get_state("critic_updates"))

    @property
    def actor_updates(self) -> int:
        return int(self._get_state("actor_updates"))

    @property
    def active_checkpoint(self) -> str | None:
        value = self._get_state("active_checkpoint")
        return value or None

    def has_episode(self, episode_key: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM episodes WHERE episode_key = ?", (episode_key,)
        ).fetchone()
        return row is not None

    def commit_episode(
        self,
        *,
        episode_key: str,
        episode_id: int,
        outcome: str,
        warmup: bool,
        warmup_env_steps: int,
        records: list[ReplayTransitionRecord],
        utd_ratio: int,
        checkpoint: str | None = None,
        episode_digest: str | None = None,
    ) -> int:
        if outcome not in {"success", "failure"}:
            raise ValueError(f"invalid episode outcome: {outcome!r}")
        if warmup_env_steps < 0:
            raise ValueError("warmup_env_steps cannot be negative")
        if utd_ratio <= 0:
            raise ValueError("utd_ratio must be positive")
        payloads = [
            (
                record.anchor_frame,
                self._serialize(record.transition),
                {key: value.detach().contiguous().cpu() for key, value in record.transition.items()},
            )
            for record in records
        ]
        anchors = [anchor for anchor, _, _ in payloads]
        if len(anchors) != len(set(anchors)):
            raise ValueError(f"episode {episode_key!r} contains duplicate replay anchors")
        covered = set()
        for record in records:
            steps = record.transition["actual_steps"]
            length = int(steps.item())
            if (length <= 0 or length != steps.item()
                    or length > record.transition["exec_chunk"].shape[0]
                    or record.anchor_frame < 0):
                raise ValueError("invalid replay actual_steps")
            covered.update(range(record.anchor_frame, record.anchor_frame + length))

        connection = self._connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            if connection.execute(
                "SELECT 1 FROM episodes WHERE episode_key = ?", (episode_key,)
            ).fetchone():
                connection.rollback()
                return 0
            connection.executemany(
                "INSERT INTO transitions(episode_key, anchor_frame, payload) VALUES(?, ?, ?)",
                [(episode_key, anchor, payload) for anchor, payload, _ in payloads],
            )
            connection.execute(
                "INSERT INTO episodes(episode_key, episode_id, outcome, warmup, transition_count, episode_digest) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (episode_key, episode_id, outcome, int(warmup), len(payloads), episode_digest),
            )
            produced = self.produced_credits + len(payloads) * utd_ratio
            warmup_total = self.warmup_env_steps + (warmup_env_steps if warmup else 0)
            self._set_state("produced_credits", str(produced))
            self._set_state("warmup_env_steps", str(warmup_total))
            self._set_state("trainable_env_steps", str(self.trainable_env_steps + len(covered)))
            if checkpoint is not None:
                self._set_state("active_checkpoint", checkpoint)
            count = int(connection.execute("SELECT COUNT(*) FROM transitions").fetchone()[0])
            excess = max(0, count - self.capacity)
            if excess:
                connection.execute(
                    "DELETE FROM transitions WHERE id IN "
                    "(SELECT id FROM transitions ORDER BY id ASC LIMIT ?)",
                    (excess,),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

        new_entries = [transition for _, _, transition in payloads]
        combined = self._entries + new_entries
        if len(combined) > self.capacity:
            combined = combined[-self.capacity:]
        self._entries = combined
        self._rebuild_tensor_storage()
        return len(payloads)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self._entries) < batch_size:
            raise ValueError(
                f"replay has {len(self._entries)} transitions, need batch_size={batch_size}"
            )
        indices = self._rng.sample(range(len(self._entries)), batch_size)
        index = torch.tensor(indices, dtype=torch.long)
        return {key: value.index_select(0, index) for key, value in self._storage.items()}

    def get_rng_state(self) -> object:
        return self._rng.getstate()

    def set_rng_state(self, state: object) -> None:
        self._rng.setstate(state)

    def checkpoint_progress(
        self,
        *,
        checkpoint: str,
        consumed_credits: int,
        critic_updates: int,
        actor_updates: int,
        actor_enabled: bool | None = None,
    ) -> None:
        if not checkpoint:
            raise ValueError("checkpoint name cannot be empty")
        if consumed_credits < self.consumed_credits:
            raise ValueError("consumed credits cannot move backwards")
        if consumed_credits > self.produced_credits:
            raise ValueError("consumed credits cannot exceed produced credits")
        if critic_updates != consumed_credits:
            raise ValueError("one critic update must consume exactly one replay credit")
        if critic_updates < self.critic_updates or actor_updates < self.actor_updates:
            raise ValueError("online update counters cannot move backwards")
        if actor_updates != critic_updates // 2:
            raise ValueError("actor updates must equal critic updates // 2")
        enabled = self.actor_enabled if actor_enabled is None else actor_enabled
        if self.actor_enabled and not enabled:
            raise ValueError("Actor cannot return to warmup within a training version")
        if enabled and actor_updates < 1:
            raise ValueError("Actor cannot be enabled before a successful Actor update")
        if enabled and not self.actor_enabled and consumed_credits != self.produced_credits:
            raise ValueError("first Actor activation requires all pending credits to be completed")
        with self._connection:
            self._set_state("active_checkpoint", checkpoint)
            self._set_state("consumed_credits", str(consumed_credits))
            self._set_state("critic_updates", str(critic_updates))
            self._set_state("actor_updates", str(actor_updates))
            self._set_state("actor_enabled", str(int(enabled)))

    def close(self) -> None:
        try:
            if getattr(self, "_connection", None) is not None:
                self._connection.close()
                self._connection = None
        finally:
            if getattr(self, "_owned_lease", None) is not None:
                self._owned_lease.close()

    def episode_keys(self) -> set[str]:
        return {str(row[0]) for row in self._connection.execute("SELECT episode_key FROM episodes")}

    def episode_digest(self, key: str) -> str | None:
        row = self._connection.execute("SELECT episode_digest FROM episodes WHERE episode_key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def reload(self) -> None:
        """Discard an interrupted transaction and rebuild the sampling mirror."""
        self._connection.rollback()
        self._entries = self._load_entries()
        self._rebuild_tensor_storage()
