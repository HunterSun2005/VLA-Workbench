#!/usr/bin/env bash
set -euo pipefail

include_ros2=false
if [[ "${1:-}" == "--include-ros2" ]]; then
  include_ros2=true
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--include-ros2]" >&2
  exit 2
fi

patterns=(
  "vla-deploy-franka-pi05-rtc"
  "vla.cli.deploy_franka_pi05_rtc"
)
if $include_ros2; then
  patterns+=(
    "franka_ros2_control_node"
    "franka_robot_state_broadcaster"
    "robotiq.*(driver|node)"
  )
fi

matches_pattern() {
  local command=$1 pattern
  for pattern in "${patterns[@]}"; do
    if [[ $command =~ $pattern ]]; then
      return 0
    fi
  done
  return 1
}

find_targets() {
  local pid uid command
  while read -r pid; do
    [[ -n $pid && $pid -ne $$ && -r /proc/$pid/status ]] || continue
    uid=$(awk '/^Uid:/ {print $2}' "/proc/$pid/status")
    [[ $uid -eq $(id -u) ]] || continue
    command=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    matches_pattern "$command" && printf '%s\n' "$pid"
  done < <(pgrep -u "$(id -u)" -f 'vla-deploy-franka-pi05-rtc|vla\.cli\.deploy_franka_pi05_rtc|franka_ros2_control_node|franka_robot_state_broadcaster|robotiq' || true)
}

wait_for_exit() {
  local timeout=$1 pid alive
  local deadline=$((SECONDS + timeout))
  while (( SECONDS < deadline )); do
    alive=false
    for pid in "${targets[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=true
        break
      fi
    done
    $alive || return 0
    sleep 0.2
  done
  return 1
}

mapfile -t targets < <(find_targets | sort -nu)
if (( ${#targets[@]} == 0 )); then
  echo "No matching Franka PI0.5 RTC processes found."
else
  echo "Stopping matching processes:"
  for pid in "${targets[@]}"; do
    printf '  PID %-8s %s\n' "$pid" "$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  done

  kill -INT "${targets[@]}" 2>/dev/null || true
  if ! wait_for_exit 8; then
    echo "SIGINT timeout; sending SIGTERM."
    kill -TERM "${targets[@]}" 2>/dev/null || true
    if ! wait_for_exit 3; then
      echo "SIGTERM timeout; sending SIGKILL."
      kill -KILL "${targets[@]}" 2>/dev/null || true
      wait_for_exit 2 || true
    fi
  fi
fi

gripper=/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DABC0KC0-if00-port0
echo "Remaining relevant device holders:"
holders_found=false
if command -v fuser >/dev/null 2>&1; then
  if [[ -e $gripper ]]; then
    holder_output=$(fuser "$gripper" 2>/dev/null || true)
    if [[ -n $holder_output ]]; then
      echo "  Robotiq serial: $holder_output"
      holders_found=true
    fi
  fi
  for device in /dev/video*; do
    [[ -e $device ]] || continue
    holder_output=$(fuser "$device" 2>/dev/null || true)
    if [[ -n $holder_output ]]; then
      echo "  $device: $holder_output"
      holders_found=true
    fi
  done
fi
$holders_found || echo "  none"

if $include_ros2; then
  echo "ROS2 Franka/Robotiq process matching was enabled."
else
  echo "ROS2 processes were not touched; pass --include-ros2 to include known Franka/Robotiq controllers."
fi
