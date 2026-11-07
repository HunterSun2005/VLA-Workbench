from __future__ import annotations

import bisect
import pathlib
import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import Dataset

from vla.adapters.lerobot.checkpoint_contract import sha256_file
from vla.adapters.lerobot.policies.cache_manifest import (
    CACHE_MANIFEST_NAME,
    load_cache_manifest,
)


@dataclass
class _EpisodeIndexShim:
    dataset_from_index: list[int]
    dataset_to_index: list[int]

    def __getitem__(self, key: str):
        if key not in ("dataset_from_index", "dataset_to_index"):
            raise KeyError(key)
        return getattr(self, key)


@dataclass
class _MetaShim:
    stats: dict[str, dict[str, torch.Tensor]]
    episodes: _EpisodeIndexShim
    fps: float
    features: dict = field(default_factory=dict)
    has_language_columns: bool = False

    @property
    def camera_keys(self):
        return [key for key in self.features if key.startswith("observation.images.")]


class ChunkTransitionDataset(Dataset):
    """Lazy dataset over manifest-indexed episode shards.

    Execution caches require training semantics v3; handover-only caches carry labels.
    """

    def __init__(self, cache_dir: str | pathlib.Path, split: str = "train", *, handover_only=False):
        self.handover_only = handover_only
        self.cache_dir = pathlib.Path(cache_dir).expanduser().resolve()
        self.split = split
        self._legacy_transitions: list[dict[str, torch.Tensor]] | None = None
        self._shards: list[dict[str, Any]] = []
        self._ends: list[int] = []
        # RandomSampler interleaves episodes. A one-shard cache makes almost
        # every sample deserialize another episode from disk.
        self._cached_shards: dict[int, list[dict[str, torch.Tensor]]] = {}

        manifest_path = self.cache_dir / CACHE_MANIFEST_NAME
        if manifest_path.is_file():
            self._init_manifest_cache(manifest_path)
        else:
            raise ValueError('legacy Replay is not supported; build a new execution cache')
        # Local transition indices already form contiguous episode ranges.
        self.episodes = list(range(self.num_episodes))
        self.absolute_to_relative_idx = None

    def _init_manifest_cache(self, manifest_path: pathlib.Path) -> None:
        manifest = load_cache_manifest(manifest_path)
        if manifest.get("status") != "complete":
            raise ValueError(f"transition cache is not complete: {manifest_path}")
        shards = manifest.get("shards", {}).get(self.split)
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"transition cache split {self.split!r} is empty")
        from vla.core.training_contract import validate_training_cache
        if self.handover_only:
            from vla.cli.build_handover_cache import HANDOVER_CACHE_KIND
            if manifest['contract']['cache'].get('kind') != HANDOVER_CACHE_KIND:
                raise ValueError('expected handover label cache')
        else:
            validate_training_cache(manifest['contract'])
        self.manifest = manifest
        self._shards = shards
        total = 0
        starts: list[int] = []
        ends: list[int] = []
        for shard in shards:
            count = int(shard["transitions"])
            if count <= 0:
                raise ValueError(f"invalid shard transition count: {shard}")
            path = self.cache_dir / shard["path"]
            if not path.is_file():
                raise FileNotFoundError(f"transition shard does not exist: {path}")
            expected_sha = shard.get("sha256")
            if expected_sha is not None and sha256_file(path) != expected_sha:
                raise ValueError(f"transition shard fingerprint mismatch: {path}")
            starts.append(total)
            total += count
            ends.append(total)
        self._ends = ends
        self.num_frames = total
        self.num_episodes = len(shards)
        dataset_contract = manifest["contract"]["dataset"]
        self.fps = float(dataset_contract["fps"])
        raw_features = dataset_contract.get("features", {})
        self.features = raw_features
        self.meta = _MetaShim(
            stats={},
            episodes=_EpisodeIndexShim(
                dataset_from_index=starts,
                dataset_to_index=ends,
            ),
            fps=self.fps,
            features=raw_features,
        )

    def _load_shard(self, shard_index: int) -> list[dict[str, torch.Tensor]]:
        cached = self._cached_shards.get(shard_index)
        if cached is None:
            path = self.cache_dir / self._shards[shard_index]["path"]
            value = torch.load(path, weights_only=False, map_location="cpu")
            if not isinstance(value, list) or len(value) != int(
                self._shards[shard_index]["transitions"]
            ):
                raise ValueError(f"transition shard content does not match manifest: {path}")
            from vla.core.execution_replay import validate_execution_transition
            if not self.handover_only:
                for transition in value:
                    validate_execution_transition(transition)
            self._cached_shards[shard_index] = value
            cached = value
        return cached

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += self.num_frames
        if not 0 <= index < self.num_frames:
            raise IndexError(index)
        if self._legacy_transitions is not None:
            return self._legacy_transitions[index]
        shard_index = bisect.bisect_right(self._ends, index)
        shard_start = 0 if shard_index == 0 else self._ends[shard_index - 1]
        return self._load_shard(shard_index)[index - shard_start]
