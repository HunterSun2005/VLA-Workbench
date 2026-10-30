#!/usr/bin/env bash
set -Eeuo pipefail

# PI0.5/RTC inference with immediate gamepad human takeover.
# The policy server must already be running on the configured Unix socket.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${VLA_ROOT:-$PWD}"
MODEL_PYTHON="${MODEL_PYTHON:-${CONDA_PREFIX:-$HOME/miniconda3/envs/vla}/bin/python}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
CONFIG="$SCRIPT_DIR/../franka_robotiq_rollout/configs/single_left_sft_takeover.yaml"
POLICY_SOCKET="/tmp/vla_franka_single_left.sock"
JOY_DEVICE_ID="${LEFT_JOY_DEVICE_ID:-0}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/raw/gamepad_takeover_live}"

usage() {
  printf '%s\n' "Usage: $0 [--check-only] [--execute] [--resume] [--dataset-root PATH]" \
    "Policy server must already be running; Space toggles human takeover." \
    "s enables teleop standby, n starts inference, Space takes/returns control."
}
EXECUTE=0; RESUME=0; CHECK_ONLY=0
while (($#)); do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --resume) RESUME=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -x "$MODEL_PYTHON" ]] || { printf 'Missing Python: %s\n' "$MODEL_PYTHON" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { printf 'Missing rollout config: %s\n' "$CONFIG" >&2; exit 1; }
[[ -f "$ROS_SETUP" ]] || { printf 'Missing ROS setup: %s\n' "$ROS_SETUP" >&2; exit 1; }
[[ "$JOY_DEVICE_ID" =~ ^[0-9]+$ ]] || { printf 'LEFT_JOY_DEVICE_ID must be numeric\n' >&2; exit 2; }
JOY_DEVICE="/dev/input/js${JOY_DEVICE_ID}"
if ((CHECK_ONLY)); then
  printf 'check-only: python=%s config=%s joy_device=%s dataset=%s\n' "$MODEL_PYTHON" "$CONFIG" "$JOY_DEVICE" "$DATASET_ROOT"
  if [[ ! -e "$JOY_DEVICE" ]]; then
    printf 'WARNING: gamepad device is missing: %s (connect it or set LEFT_JOY_DEVICE_ID)\n' "$JOY_DEVICE"
  fi
  exit 0
fi
((EXECUTE)) || { printf 'Refusing to connect without --execute\n' >&2; exit 2; }
[[ -e "$JOY_DEVICE" ]] || {
  printf 'Gamepad device not found: %s\n' "$JOY_DEVICE" >&2
  printf 'Connect the controller and check /dev/input/js*, or set LEFT_JOY_DEVICE_ID.\n' >&2
  exit 4
}
[[ -S "$POLICY_SOCKET" ]] || {
  printf 'Policy server socket is missing: %s\n' "$POLICY_SOCKET" >&2
  printf 'Start vla.cli.serve_franka_sft first and wait for READY.\n' >&2
  exit 5
}
if [[ -d "$DATASET_ROOT" && "$(find "$DATASET_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" != "" && "$RESUME" != 1 ]]; then
  printf 'Dataset is non-empty; use --resume or a new --dataset-root: %s\n' "$DATASET_ROOT" >&2; exit 3
fi
JOY_PID=""; BRIDGE_PID=""
cleanup() {
  trap - INT TERM EXIT
  [[ -n "$BRIDGE_PID" ]] && kill -INT "$BRIDGE_PID" 2>/dev/null || true
  [[ -n "$JOY_PID" ]] && kill -INT "$JOY_PID" 2>/dev/null || true
  [[ -n "$BRIDGE_PID" ]] && wait "$BRIDGE_PID" 2>/dev/null || true
  [[ -n "$JOY_PID" ]] && wait "$JOY_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT
source "$REPO_ROOT/scripts/ros_env.sh"
set -u
export PYTHONPATH="$(cd "$SCRIPT_DIR/../../.." && pwd):${PYTHONPATH:-}"
ros2 run joy joy_node --ros-args -r __node:=left_rollout_joy -r joy:=/left/gamepad/joy \
  -p device_id:="$JOY_DEVICE_ID" -p deadzone:=0.0 -p autorepeat_rate:=100.0 & JOY_PID=$!
/usr/bin/python3 -m vla.datacollection.franka_robotiq_rollout.joy_bridge \
  --topic /left/gamepad/joy --socket-path /tmp/vla_franka_single_left_joy.sock & BRIDGE_PID=$!
printf 'Policy inference + gamepad takeover ready. Space toggles human control; c/Esc exits.\n'
ARGS=(--config "$CONFIG" --dataset-root "$DATASET_ROOT" --execute)
((RESUME)) || true
# Keep the collector in the foreground so it retains the user's TTY for
# --execute and keyboard events (s/n/Space/y/x/r/c/Esc). The shell remains
# alive long enough for the EXIT trap to stop the Joy sidecars.
env -u PYTHONPATH -u LD_LIBRARY_PATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH PYTHONPATH="$(cd "$SCRIPT_DIR/../../.." && pwd)" \
  "$MODEL_PYTHON" -u -m vla.datacollection.franka_robotiq_rollout.collect "${ARGS[@]}"
