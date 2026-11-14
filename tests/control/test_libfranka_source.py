"""Building ROS must not silently reuse the previous libfranka source pin."""
import importlib.util
from pathlib import Path

import pytest
import yaml

from vla.control.franky.binding import REQUIRED_LIBFRANKA_VERSION


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_libfranka_source", ROOT / "scripts/check_libfranka_source.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pinned_source_and_worker_version_agree(tmp_path):
    manifest = (ROOT / "ros_ws/dependencies.repos").read_text()
    (tmp_path / "dependencies.repos").write_text(manifest)
    relative = "src/third_party/libfranka"
    source = tmp_path / relative
    source.mkdir(parents=True)
    revision = yaml.safe_load(manifest)["repositories"][relative]["version"]
    assert revision == "9f9304ec0ac897eff3219a67f612b959948535e2"
    (source / ".vla-revision").write_text(revision)
    cmake = source / "CMakeLists.txt"
    cmake.write_text(f"set(libfranka_VERSION {REQUIRED_LIBFRANKA_VERSION})\n")
    module.check_source(tmp_path)
    cmake.write_text("set(libfranka_VERSION 0.20.4)\n")
    with pytest.raises(RuntimeError, match="requires libfranka 0.21.2"):
        module.check_source(tmp_path)
    (source / ".vla-revision").write_text("95b406f47c97cedfbe426c04d8fb7f83a9885e1c")
    with pytest.raises(RuntimeError, match="does not match"):
        module.check_source(tmp_path)
