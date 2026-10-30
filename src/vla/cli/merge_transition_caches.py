"""Merge compatible manifest transition caches without rewriting shard tensors."""

from __future__ import annotations

import argparse
import copy
import os
import pathlib
import shutil
from typing import Any

from vla.adapters.lerobot.checkpoint_contract import sha256_file
from vla.adapters.lerobot.policies.cache_manifest import (
    CACHE_MANIFEST_NAME,
    atomic_write_json,
    load_cache_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        action="append",
        required=True,
        help="NAME=PATH; pass at least two times",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _parse_cache(value: str) -> tuple[str, pathlib.Path]:
    if "=" not in value:
        raise ValueError(f"cache must use NAME=PATH syntax: {value!r}")
    name, raw_path = value.split("=", 1)
    if not name or "/" in name or "\\" in name:
        raise ValueError(f"invalid cache name: {name!r}")
    return name, pathlib.Path(raw_path).expanduser().resolve()


def _dataset_schema(dataset: dict[str, Any]) -> dict[str, Any]:
    return {
        key: dataset.get(key)
        for key in (
            "fps",
            "state_dim",
            "action_dim",
            "state_names",
            "action_names",
            "camera_keys",
            "actions_stored_as",
            "action_stats",
        )
    }


def _cache_settings(cache: dict[str, Any]) -> dict[str, Any]:
    ignored = {"episode_splits", "train_ratio", "seed", "default_episode_success"}
    return {key: value for key, value in cache.items() if key not in ignored}


def _assert_compatible(base: dict[str, Any], candidate: dict[str, Any], name: str) -> None:
    if base["policy"] != candidate["policy"]:
        raise ValueError(f"cache {name!r} uses a different policy/checkpoint contract")
    if _dataset_schema(base["dataset"]) != _dataset_schema(candidate["dataset"]):
        raise ValueError(f"cache {name!r} uses an incompatible dataset schema")
    if _cache_settings(base["cache"]) != _cache_settings(candidate["cache"]):
        raise ValueError(f"cache {name!r} uses incompatible chunk/build settings")


def _link_or_copy(source: pathlib.Path, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def merge_transition_caches(
    sources: list[tuple[str, pathlib.Path]],
    output_dir: str | pathlib.Path,
) -> pathlib.Path:
    if len(sources) < 2:
        raise ValueError("at least two caches are required")
    names = [name for name, _ in sources]
    if len(names) != len(set(names)):
        raise ValueError("cache names must be unique")
    output = pathlib.Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    manifests = [(name, root, load_cache_manifest(root)) for name, root in sources]
    for name, _, manifest in manifests:
        if manifest.get("status") != "complete":
            raise ValueError(f"cache {name!r} is not complete")
    base_contract = manifests[0][2]["contract"]
    for name, _, manifest in manifests[1:]:
        _assert_compatible(base_contract, manifest["contract"], name)

    contract = copy.deepcopy(base_contract)
    contract["dataset_sources"] = [
        {
            "name": name,
            **{
                key: manifest["contract"]["dataset"].get(key)
                for key in ("repo_id", "root", "info_sha256")
            },
        }
        for name, _, manifest in manifests
    ]
    contract["cache"].pop("episode_splits", None)
    contract["cache"]["merged_sources"] = names
    merged: dict[str, Any] = {
        "format_version": 1,
        "status": "complete",
        "contract": contract,
        "completed_episodes": {"train": [], "val": []},
        "skipped_episodes": {"train": [], "val": []},
        "shards": {"train": [], "val": []},
        "total_transitions": {"train": 0, "val": 0},
    }
    output.mkdir(parents=True)
    try:
        for source_index, (name, root, manifest) in enumerate(manifests):
            for split in ("train", "val"):
                for shard_index, shard in enumerate(manifest["shards"].get(split, [])):
                    source_path = root / shard["path"]
                    if not source_path.is_file() or sha256_file(source_path) != shard.get("sha256"):
                        raise ValueError(f"source shard is missing or changed: {source_path}")
                    relative_path = (
                        pathlib.Path("shards")
                        / split
                        / f"{source_index:02d}_{name}_{shard_index:06d}.pt"
                    )
                    destination = output / relative_path
                    _link_or_copy(source_path, destination)
                    merged_shard = copy.deepcopy(shard)
                    merged_shard.update(
                        {
                            "path": relative_path.as_posix(),
                            "source_cache": name,
                            "source_episode_id": shard.get("episode_id"),
                            "episode_id": len(merged["shards"][split]),
                            "sha256": sha256_file(destination),
                        }
                    )
                    merged["shards"][split].append(merged_shard)
                    merged["completed_episodes"][split].append(merged_shard["episode_id"])
                    merged["total_transitions"][split] += int(shard["transitions"])
        if not merged["shards"]["train"]:
            raise ValueError("merged cache has no training shards")
        atomic_write_json(output / CACHE_MANIFEST_NAME, merged)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return output


def main() -> None:
    args = parse_args()
    result = merge_transition_caches(
        [_parse_cache(value) for value in args.cache],
        args.output_dir,
    )
    print(f"Merged transition cache written to {result}", flush=True)


if __name__ == "__main__":
    main()
