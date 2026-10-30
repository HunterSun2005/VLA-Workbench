"""Durable identity and timing contract shared by collection and online training."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid
from typing import Any

RECORDING_SCHEMA_VERSION = 4
RUN_FORMAT_VERSION = 2
RUN_MANIFEST = "meta/online_run.json"
TIMING_FIELDS = {
    "observed_at_ns": "int64",
    "command_at_ns": "int64",
    "command_id": "int64",
    "segment_id": "int64",
    "action_valid": "int64",
    "segment_terminal": "int64",
}
TIMING_FEATURES = {
    f"complementary_info.{name}": {"dtype": dtype, "shape": [1], "names": [name]}
    for name, dtype in TIMING_FIELDS.items()
}
MIN_STEP_PERIODS = 0.5
MAX_STEP_PERIODS = 1.5


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_digest(path: str | Path) -> str:
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"online model resource is missing: {root}")
    files = [root] if root.is_file() else sorted(
        item for item in root.rglob("*")
        if item.is_file() and not any(part.startswith(".") for part in item.relative_to(root).parts)
    )
    if not files:
        raise ValueError(f"online model resource is empty: {root}")
    digest = hashlib.sha256()
    for item in files:
        digest.update((item.name if root.is_file() else item.relative_to(root).as_posix()).encode())
        digest.update(file_digest(item).encode())
    return digest.hexdigest()


def load_run(root: str | Path) -> dict[str, Any]:
    path = Path(root) / RUN_MANIFEST
    if not path.is_file():
        raise ValueError("legacy or unidentified online dataset; use a new dataset/replay run")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format_version") != RUN_FORMAT_VERSION:
        raise ValueError("unsupported online run format; old runs are read-only")
    for key in ("run_uuid", "dataset_uuid"):
        uuid.UUID(value[key])
    publication = value.get("upgrade_publication")
    if publication is not None:
        target = Path(root).expanduser().resolve()
        if str(target) != publication["dataset_root"]:
            raise ValueError("unpublished or relocated training upgrade; use the published output pair")
        report = target.parent / "upgrade_report.json"
        if not report.is_file() or not json.loads(report.read_text()).get("complete"):
            raise ValueError("training upgrade has not been published completely")
    return value


def bind_run(root: Path, contract: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    canonical = json.loads(json.dumps(contract, sort_keys=True, default=str))
    if resume:
        value = load_run(root)
        old = value["contract"]
        differences = sorted(key for key in old.keys() | canonical.keys() if old.get(key) != canonical.get(key))
        if differences:
            raise ValueError(f"online run contract mismatch: {', '.join(differences)}")
        return value
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"new online run requires an empty dataset: {root}")
    value = {
        "format_version": RUN_FORMAT_VERSION,
        "recording_schema_version": RECORDING_SCHEMA_VERSION,
        "run_uuid": str(uuid.uuid4()),
        "dataset_uuid": str(uuid.uuid4()),
        "contract": canonical,
    }
    durable_json(root / RUN_MANIFEST, value)
    return value


def reject_unaligned_offline_dataset(info: dict[str, Any]) -> None:
    if int(info.get("recording_schema_version", 0)) >= RECORDING_SCHEMA_VERSION:
        raise ValueError("schema-4 rollout requires segment-aware online encoding; this offline tool is not supported")
