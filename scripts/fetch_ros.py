#!/usr/bin/python3
"""Fetch immutable upstream source archives listed in dependencies.repos."""
from pathlib import Path
import io, tarfile, tempfile, urllib.request, shutil, subprocess
import yaml
import argparse
import hashlib

repository_root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace', type=Path)
parser.add_argument('--print-workspace', action='store_true')
args = parser.parse_args()
manifest_path = repository_root/'ros_ws/dependencies.repos'
identity = hashlib.sha256(manifest_path.read_bytes() + (repository_root/'ros_ws/patches/robotiq-registers.patch').read_bytes()).hexdigest()[:20]
root = args.workspace or repository_root/'runtime/ros/workspaces'/identity
root = root.resolve()
if args.print_workspace:
    print(root)
    raise SystemExit(0)
if not root.is_relative_to(repository_root/'runtime/ros/workspaces'):
    raise ValueError('fresh ROS sources must use runtime/ros/workspaces')
root.mkdir(parents=True, exist_ok=True)
(root/'dependencies.repos').write_bytes(manifest_path.read_bytes())
manifest = yaml.safe_load(manifest_path.read_text())["repositories"]
# Owned packages remain live source inputs; third-party sources are fresh archives.
for package in (repository_root/'ros_ws/src').iterdir():
    if package.name == 'third_party' or not package.is_dir():
        continue
    target = root/'src'/package.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(package, target_is_directory=True)
print(f'Workspace: {root}', flush=True)
for relative, entry in manifest.items():
    target = root / relative
    revision = entry["version"]
    stamp = target / ".vla-revision"
    if stamp.exists():
        if stamp.read_text().strip() != revision:
            raise RuntimeError(f"Revision mismatch: {target}")
        continue
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f"Refusing to overwrite unverified source: {target}")
    repository = entry["url"].removeprefix("https://github.com/").removesuffix(".git")
    url = f"https://codeload.github.com/{repository}/tar.gz/{revision}"
    print(f"Fetching {repository}@{revision}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    with tempfile.TemporaryDirectory(prefix="vla-ros-source-") as temporary:
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            members = archive.getmembers()
            archive.extractall(temporary, members=[m for m in members if not m.issym() and not m.islnk()], filter="data")
            for member in members:
                if member.issym():
                    destination = Path(temporary) / member.name
                    original = (destination.parent / member.linkname).resolve()
                    if not original.is_relative_to(Path(temporary).resolve()):
                        raise ValueError(f"Unsafe upstream symlink: {member.name}")
                    if original.is_file():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(original, destination)
        source, = Path(temporary).iterdir()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True)
    stamp.write_text(revision + "\n")
for patch_name, source_name in (("robotiq-registers.patch", "ros2_robotiq_gripper"),):
    patch = repository_root / "ros_ws/patches" / patch_name
    driver = root / "src/third_party" / source_name
    check = subprocess.run(["git", "apply", "--check", str(patch)], cwd=driver, capture_output=True)
    if check.returncode == 0:
        subprocess.run(["git", "apply", str(patch)], cwd=driver, check=True)
    else:
        subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=driver, check=True)
# Use the distro description/controllers; build only the patched upstream driver.
for name in ("robotiq_description", "robotiq_controllers", "robotiq_hardware_tests"):
    package = root / "src/third_party/ros2_robotiq_gripper" / name
    if package.exists():
        (package / "COLCON_IGNORE").touch()

# Only its Python SDK is used; do not build unrelated ROS1/ROS2 examples.
(root / "src/third_party/dynamixel_sdk/COLCON_IGNORE").touch()
