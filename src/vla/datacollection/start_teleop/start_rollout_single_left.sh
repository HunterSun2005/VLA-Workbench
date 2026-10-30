#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Franky owns the arm, gripper, and cameras. This process only bridges ROS Joy.
export CONTROL_OWNER=rollout
REPOSITORY_ROOT="${VLA_ROOT:-$PWD}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
LEFT_JOY_DEVICE_ID="${LEFT_JOY_DEVICE_ID:-0}"
JOY_AUTOREPEAT_RATE="${JOY_AUTOREPEAT_RATE:-100.0}"
JOY_SOCKET_PATH="${JOY_SOCKET_PATH:-/tmp/vla_franka_single_left_joy.sock}"

[[ -f "$ROS_SETUP" ]] || {
  printf 'Missing ROS setup: %s\n' "$ROS_SETUP" >&2
  exit 1
}
[[ "$LEFT_JOY_DEVICE_ID" =~ ^[0-9]+$ ]] || {
  printf 'LEFT_JOY_DEVICE_ID must be a non-negative integer\n' >&2
  exit 2
}

# ROS environment hooks legitimately probe variables that may not exist yet.
set +u
source "$ROS_SETUP"
set -u
export PYTHONPATH="$(cd "$SCRIPT_DIR/../../.." && pwd):${PYTHONPATH:-}"

JOY_PID=""
BRIDGE_PID=""
cleanup() {
  trap - INT TERM EXIT
  [[ -n "$BRIDGE_PID" ]] && kill -INT "$BRIDGE_PID" 2>/dev/null || true
  [[ -n "$JOY_PID" ]] && kill -INT "$JOY_PID" 2>/dev/null || true
  [[ -n "$BRIDGE_PID" ]] && wait "$BRIDGE_PID" 2>/dev/null || true
  [[ -n "$JOY_PID" ]] && wait "$JOY_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

ros2 run joy joy_node --ros-args \
  -r __node:=left_rollout_joy \
  -r joy:=/left/gamepad/joy \
  -p device_id:="$LEFT_JOY_DEVICE_ID" \
  -p deadzone:=0.0 \
  -p autorepeat_rate:="$JOY_AUTOREPEAT_RATE" &
JOY_PID=$!

/usr/bin/python3 -m vla.datacollection.franka_robotiq_rollout.joy_bridge \
  --topic /left/gamepad/joy \
  --socket-path "$JOY_SOCKET_PATH" &
BRIDGE_PID=$!

printf 'Joy-only rollout sidecar ready; socket=%s\n' "$JOY_SOCKET_PATH"
wait -n "$JOY_PID" "$BRIDGE_PID"
