"""Dispatch rollout collection to the configured hardware backend."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def configured_backend(path: str | Path) -> str:
    config_path = Path(path).expanduser().resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"rollout config must be a YAML mapping: {config_path}")
    backend = str(value.get("backend", "ros2"))
    if backend != "ros2":
        raise ValueError("Legacy Franky backend removed; use an explicit ROS2 rollout configuration")
    return backend


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()
    configured_backend(args.config)
    from vla.cli.ros_session import main as backend_main
    import sys
    backend_main(["--workflow", "deploy", *sys.argv[1:]])


if __name__ == "__main__":
    main()
