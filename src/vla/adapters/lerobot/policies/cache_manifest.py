"""Manifest helpers shared by transition-cache building and AC training."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


CACHE_FORMAT_VERSION = 1
CACHE_MANIFEST_NAME = "cache_manifest.json"


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_cache_manifest(path_or_dir: str | Path) -> dict[str, Any]:
    path = Path(path_or_dir).expanduser().resolve()
    if path.is_dir():
        path = path / CACHE_MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"transition cache manifest does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"transition cache manifest must be a JSON object: {path}")
    if value.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"unsupported transition cache format {value.get('format_version')!r}; "
            f"expected {CACHE_FORMAT_VERSION}"
        )
    if not isinstance(value.get("contract"), dict):
        raise ValueError("transition cache manifest has no contract mapping")
    return value


def new_cache_manifest(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "status": "building",
        "contract": contract,
        "completed_episodes": {"train": [], "val": []},
        "skipped_episodes": {"train": [], "val": []},
        "shards": {"train": [], "val": []},
    }


def assert_manifest_contract(manifest: dict[str, Any], expected: dict[str, Any]) -> None:
    if manifest.get("contract") != expected:
        raise ValueError(
            "existing transition cache manifest does not match this build; "
            "use a new output directory or restore the original arguments/checkpoints"
        )
