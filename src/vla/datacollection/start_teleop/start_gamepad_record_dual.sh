#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="${VLA_ROOT:-$PWD}"
# Load the same ROS2 workspaces as start_rollout_duo.sh before executing.
exec env PYTHONPATH="$(cd "$SCRIPT_DIR/../../.." && pwd):${PYTHONPATH:-}" \
  /usr/bin/python3 -m vla.datacollection.franka_robotiq_dual.gamepad_record "$@"
