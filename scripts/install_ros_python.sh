#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec env -u PYTHONPATH /usr/bin/python3 -m pip install --no-build-isolation --no-deps --upgrade --target "$root/ros_ws/python_deps" "$root/ros_ws/src/third_party/dynamixel_sdk/python" -r "$root/ros_ws/requirements-python.lock" "$@"
