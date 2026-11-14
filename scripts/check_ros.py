#!/usr/bin/python3
"""Read-only ROS dependency/import check; does not create nodes or connect devices."""
import importlib
from pathlib import Path
import sys
import json
import re
import subprocess
from ament_index_python.packages import get_package_prefix

root = Path(__file__).resolve().parent.parent
from vla.core.ros_workspace import ros_workspace
workspace = ros_workspace(root)
for name in ("vla.datacollection.franka_robotiq_rollout.collect_ros", "vla.datacollection.franka_robotiq_dual.auto_record", "vla.datacollection.franka_robotiq_dual.gamepad_record", "vla.datacollection.franka_robotiq_single_left.auto_record", "vla.core.writer_lock"):
    importlib.import_module(name)
assert "torch" not in sys.modules
assert sys.version_info[:2] == (3, 10)
for name in ("dynamixel_sdk", "serial", "numpy", "cv2", "pandas", "pyarrow", "pyrealsense2"):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(root / "ros_ws/python_deps"), (name, module.__file__)
packages = {name: get_package_prefix(name) for name in ("franka_hardware", "franka_fr3_arm_controllers", "franka_gripper_manager", "franka_gello_state_publisher", "robotiq_driver", "vla_bringup")}
assert all(Path(path).is_relative_to(workspace / "install") for path in packages.values())
from vla.control.franky.binding import REQUIRED_LIBFRANKA_VERSION

links = {}
expected = workspace / "install/libfranka/lib" / f"libfranka.so.{REQUIRED_LIBFRANKA_VERSION}"
assert expected.is_file(), f"Missing rebuilt libfranka: {expected}"
for package in ("franka_hardware", "franka_semantic_components",
                "franka_fr3_arm_controllers", "franka_robot_state_broadcaster"):
    library = workspace / "install" / package / "lib" / f"lib{package}.so"
    output = subprocess.check_output(["ldd", str(library)], text=True)
    assert "not found" not in output, (library, output)
    resolved = re.findall(r"^\s*libfranka\.so\.[\d.]+\s+=>\s+(\S+)", output, re.MULTILINE)
    assert len(resolved) == 1 and Path(resolved[0]).resolve() == expected.resolve(), (library, output)
    links[package] = str(Path(resolved[0]).resolve())
print(json.dumps({"python": sys.executable, "packages": packages, "collection_imports": "passed",
                  "libfranka": REQUIRED_LIBFRANKA_VERSION, "libfranka_links": links,
                  "torch_loaded": False, "hardware_connection": "not_attempted"}, indent=2))
