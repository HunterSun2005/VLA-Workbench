"""Content fingerprints for the local checkpoints used by the RLT pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SFT_REQUIRED_FILES = (
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_digest(files: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fingerprint_directory(
    root: str | Path,
    *,
    kind: str,
    required_files: Iterable[str] = (),
    include_safetensors: bool = True,
    require_safetensors: bool = False,
) -> dict[str, Any]:
    """Fingerprint checkpoint metadata and every top-level safetensors shard."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{kind} checkpoint directory does not exist: {root}")

    paths: set[Path] = set()
    for name in required_files:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"{kind} checkpoint is missing required file: {path}")
        paths.add(path)
    if include_safetensors:
        safetensors_paths = [path for path in root.glob("*.safetensors") if path.is_file()]
        if require_safetensors and not safetensors_paths:
            raise FileNotFoundError(f"{kind} checkpoint has no top-level safetensors weights: {root}")
        paths.update(safetensors_paths)

    files = {
        path.relative_to(root).as_posix(): {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in sorted(paths)
    }
    return {
        "kind": kind,
        "path": str(root),
        "digest": _contract_digest(files),
        "files": files,
    }


def fingerprint_sft_checkpoint(root: str | Path) -> dict[str, Any]:
    return fingerprint_directory(
        root,
        kind="pi05_sft",
        required_files=SFT_REQUIRED_FILES,
        include_safetensors=True,
        require_safetensors=True,
    )


def fingerprint_rl_token_checkpoint(root: str | Path) -> dict[str, Any]:
    return fingerprint_directory(
        root,
        kind="rlt_token",
        required_files=("config.json", "model.safetensors"),
        include_safetensors=True,
        require_safetensors=True,
    )


def assert_same_fingerprint(
    expected: dict[str, Any] | None,
    actual: dict[str, Any],
    *,
    label: str,
) -> None:
    if not expected:
        raise ValueError(f"{label} checkpoint has no saved content fingerprint")
    if expected.get("digest") != actual.get("digest"):
        raise ValueError(
            f"{label} checkpoint fingerprint mismatch: saved={expected.get('digest')!r}, "
            f"actual={actual.get('digest')!r}"
        )
