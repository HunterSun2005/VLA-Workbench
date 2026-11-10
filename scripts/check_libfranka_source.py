#!/usr/bin/python3
"""Refuse stale ROS sources before a build; never modify a third-party tree."""
from pathlib import Path
import re
import sys

import yaml


def check_source(workspace: Path) -> None:
    manifest = yaml.safe_load((workspace / "dependencies.repos").read_text())
    relative = "src/third_party/libfranka"
    expected = manifest["repositories"][relative]["version"]
    source = workspace / relative
    stamp = source / ".vla-revision"
    if not stamp.is_file() or stamp.read_text().strip() != expected:
        raise RuntimeError(
            f"libfranka source does not match dependencies.repos ({expected}). "
            "Prepare a fresh workspace from the pinned manifest; existing checkouts are not overwritten.")
    cmake = (source / "CMakeLists.txt").read_text()
    if not re.search(r"set\(libfranka_VERSION\s+0\.18\.0\s*\)", cmake):
        raise RuntimeError("ROS build requires libfranka 0.18.0 source")


if __name__ == "__main__":
    try:
        check_source(Path(__file__).resolve().parents[1] / "ros_ws")
    except (OSError, RuntimeError) as error:
        sys.exit(str(error))
