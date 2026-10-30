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


@dataclass
class _MetaShim:
    stats: dict[str, dict[str, torch.Tensor]]
    episodes: _EpisodeIndexShim
    fps: float
    features: dict = field(default_factory=dict)


class ChunkTransitionDataset(Dataset):
    """Lazy dataset over manifest-indexed episode shards.

    Legacy ``chunk_transitions_{split}.pt`` caches remain readable, but new
    training must use the manifest format to carry robot/checkpoint contracts.
    """

    def __init__(self, cache_dir: str | pathlib.Path, split: str = "train"):
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
            self._init_legacy_cache()

    def _init_manifest_cache(self, manifest_path: pathlib.Path) -> None:
        manifest = load_cache_manifest(manifest_path)
        if manifest.get("status") != "complete":
            raise ValueError(f"transition cache is not complete: {manifest_path}")
        shards = manifest.get("shards", {}).get(self.split)
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"transition cache split {self.split!r} is empty")
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

    def _init_legacy_cache(self) -> None:
        path = self.cache_dir / f"chunk_transitions_{self.split}.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"no {CACHE_MANIFEST_NAME} or legacy transition cache at {path}"
            )
        warnings.warn(
            f"loading legacy transition cache without a schema/checkpoint contract: {path}",
            stacklevel=2,
        )
        transitions = torch.load(path, weights_only=False, map_location="cpu")
        if not isinstance(transitions, list) or not transitions:
            raise ValueError(f"empty or invalid legacy cache at {path}")
        self.manifest = None
        self._legacy_transitions = transitions
        self.num_frames = len(transitions)
        self.num_episodes = 1
        self.fps = 30.0
        sample = transitions[0]
        self.features = {
            key: {"shape": tuple(value.shape), "dtype": str(value.dtype)}
            for key, value in sample.items()
            if isinstance(value, torch.Tensor)
        }
        self.meta = _MetaShim(
            stats={},
            episodes=_EpisodeIndexShim([0], [self.num_frames]),
            fps=self.fps,
            features={},
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
