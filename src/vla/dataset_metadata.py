"""VLA metadata survives LeRobot's typed DatasetInfo serialization."""
import json
import os
from pathlib import Path

STANDARD_FIELDS = frozenset("codebase_version fps features total_episodes total_frames total_tasks chunks_size data_files_size_in_mb video_files_size_in_mb data_path video_path robot_type splits tools".split())


def load_info(root):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    sidecar = root / "meta/vla_metadata.json"
    if sidecar.is_file():
        extras = json.loads(sidecar.read_text(encoding="utf-8"))
        if STANDARD_FIELDS.intersection(extras):
            raise ValueError("VLA sidecar must not override standard dataset fields")
        for key, value in extras.items():
            if key in info and info[key] != value:
                raise ValueError(f"Dataset metadata conflict: {key}")
        info.update(extras)
    return info


def write_metadata_sidecar(root, info):
    path = Path(root) / "meta/vla_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({k: v for k, v in info.items() if k not in STANDARD_FIELDS}, indent=2) + "\n")
    os.replace(temporary, path)


def dataset_info(dataset):
    root = getattr(dataset, "root", None)
    if root is not None and (Path(root) / "meta/info.json").is_file():
        return load_info(root)
    info = getattr(getattr(dataset, "meta", None), "info", {})
    return info.to_dict() if hasattr(info, "to_dict") else info
