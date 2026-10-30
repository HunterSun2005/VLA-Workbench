#!/usr/bin/env bash
set -Eeuo pipefail

# Stop the FR3 GELLO/gamepad teleoperation stack and remove its runtime files.
# Usage: clean_teleop.sh [--dry-run] [--keep-daemon] [--tmux]

STATE_DIR="${STATE_DIR:-/tmp/vla_fr3_fine_teleop_${UID}}"
ROS2_WS="${ROS2_WS:-/home/amax/workspace/ros2}"
FRANKA_ROS2_WS="${FRANKA_ROS2_WS:-/home/amax/franka_ros2_ws}"
ROBOTIQ_ROS2_SETUP="${ROBOTIQ_ROS2_SETUP:-$ROS2_WS/install/robotiq_underlay/setup.bash}"
WAIT_SEC="${WAIT_SEC:-30}"
TERM_WAIT_SEC="${TERM_WAIT_SEC:-5}"
LAUNCHER_WAIT_SEC="${LAUNCHER_WAIT_SEC:-120}"
ARM_SHUTDOWN_TIMEOUT_SEC="${ARM_SHUTDOWN_TIMEOUT_SEC:-10}"
DRY_RUN=0
STOP_ROS_DAEMON="${STOP_ROS_DAEMON:-1}"
KILL_TMUX=0

usage() {
  printf 'Usage: %s [--dry-run] [--keep-daemon] [--tmux]\n' "$(basename "$0")"
}

log() {
  printf '[%s] %s\n' "$(date +"%Y-%m-%d %H:%M:%S")" "$*"
}

ros_env_cmd() {
  printf 'source %q/install/setup.bash; ' "$FRANKA_ROS2_WS"
  printf 'source %q; ' "$ROBOTIQ_ROS2_SETUP"
  printf 'source %q/install/setup.bash; ' "$ROS2_WS"
}

while (($#)); do
  case "$1" in
    --dry-run|-n) DRY_RUN=1 ;;
    --keep-daemon) STOP_ROS_DAEMON=0 ;;
    --tmux) KILL_TMUX=1 ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

declare -a PATTERNS=(
  "datacollection/start_teleop/start_teleop_(duo|single_left)\\.sh"
  "datacollection/start_teleop/start_rollout_(duo|single_left)\\.sh"
  "datacollection/start_teleop/start_teleop\\.sh"
  "start_fr3_teleop\\.sh"
  "start_fr3_left_teleop\\.sh"
  "start_fr3_duo_teleop\\.sh"
  "fine_teleop\\.py"
  "gello_duo_fine\\.launch\\.py"
  "fr3_duo_teleop\\.launch\\.py"
  "robotiq_duo_teleop\\.launch\\.py"
  "ros2 run joy (joy_node|game_controller_node)"
  "franka_gello_state_publisher"
  "gello_publisher"
  "franka_fr3_arm_controllers"
  "franka_gripper_manager"
  "robotiq_gripper"
  "controller_manager/ros2_control_node.*__ns:=/(left|right)(/gripper)?"
  "controller_manager/spawner .*(joint_state_broadcaster|franka_robot_state_broadcaster|joint_impedance_controller|robotiq)"
  "robot_state_publisher/robot_state_publisher.*__ns:=/(left|right)(/gripper)?"
  "realsense2_camera.*camera_(base|left_wrist|right_wrist)"
)

declare -A TRACKED_PIDS=()
declare -A TRACKED_GROUPS=()

matches_pattern() {
  local command="$1"
  local pattern
  for pattern in "${PATTERNS[@]}"; do
    [[ "$command" =~ $pattern ]] && return 0
  done
  return 1
}

track_pid() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  [[ "$pid" != "$$" && "$pid" != "${PPID:-}" ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  TRACKED_PIDS["$pid"]=1

  local pgid sid
  read -r pgid sid < <(ps -p "$pid" -o pgid=,sid= 2>/dev/null || true)
  pgid="${pgid//[[:space:]]/}"
  sid="${sid//[[:space:]]/}"
  if [[ -n "$pgid" && "$pgid" == "$sid" && "$pgid" != "1" ]]; then
    TRACKED_GROUPS["$pgid"]=1
  fi
}

collect_state_pids() {
  if [[ -f "$STATE_DIR/launcher.pid" ]]; then
    track_pid "$(tr -cd '0-9' <"$STATE_DIR/launcher.pid")"
  fi
  if [[ -f "$STATE_DIR/task_pids" ]]; then
    local pid name
    while read -r pid name; do
      track_pid "${pid:-}"
    done <"$STATE_DIR/task_pids"
  fi
  return 0
}

collect_pattern_pids() {
  local pid ppid pgid sid state command
  while read -r pid ppid pgid sid state command; do
    [[ -n "${pid:-}" && -n "${command:-}" ]] || continue
    [[ "$state" != Z* ]] || continue
    [[ "$pid" != "$$" && "$pid" != "${PPID:-}" ]] || continue
    [[ "$command" != *"clean_teleop.sh"* ]] || continue
    matches_pattern "$command" && track_pid "$pid"
  done < <(ps -eo pid=,ppid=,pgid=,sid=,stat=,args=)
  return 0
}

collect_descendants() {
  local changed=1
  local pid ppid
  while ((changed)); do
    changed=0
    while read -r pid ppid; do
      [[ -n "${pid:-}" && -n "${ppid:-}" ]] || continue
      if [[ -n "${TRACKED_PIDS[$ppid]+x}" && -z "${TRACKED_PIDS[$pid]+x}" ]]; then
        track_pid "$pid"
        changed=1
      fi
    done < <(ps -eo pid=,ppid=)
  done
  return 0
}

refresh_targets() {
  collect_state_pids
  collect_pattern_pids
  collect_descendants
  return 0
}

print_targets() {
  local pid
  for pid in "${!TRACKED_PIDS[@]}"; do
    ps -p "$pid" -o pid=,ppid=,pgid=,stat=,args= 2>/dev/null || true
  done | sort -n
}

signal_targets() {
  local signal="$1"
  local current_pgid pgid pid
  current_pgid="$(ps -p "$$" -o pgid= | tr -d '[:space:]')"

  for pgid in "${!TRACKED_GROUPS[@]}"; do
    [[ "$pgid" != "$current_pgid" ]] || continue
    if ((DRY_RUN)); then
      log "DRY-RUN: would send $signal to process group $pgid"
    else
      kill "-$signal" -- "-$pgid" 2>/dev/null || true
    fi
  done

  for pid in "${!TRACKED_PIDS[@]}"; do
    [[ "$pid" != "$$" && "$pid" != "${PPID:-}" ]] || continue
    pgid="$(ps -p "$pid" -o pgid= 2>/dev/null | tr -d '[:space:]' || true)"
    [[ -z "$pgid" || -z "${TRACKED_GROUPS[$pgid]+x}" ]] || continue
    if ((DRY_RUN)); then
      log "DRY-RUN: would send $signal to process $pid"
    else
      kill "-$signal" "$pid" 2>/dev/null || true
    fi
  done
}

alive_target_count() {
  local count=0 pid
  for pid in "${!TRACKED_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      count=$((count + 1))
    fi
  done
  printf '%s\n' "$count"
}

launcher_pid() {
  [[ -f "$STATE_DIR/launcher.pid" ]] || return 1
  local pid
  pid="$(tr -cd '0-9' <"$STATE_DIR/launcher.pid")"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null || return 1
  printf '%s\n' "$pid"
}

request_launcher_shutdown() {
  local pid
  pid="$(launcher_pid)" || return 1
  if ((DRY_RUN)); then
    log "DRY-RUN: would send SIGINT to teleop launcher $pid"
    return 0
  fi

  log "Requesting graceful shutdown from teleop launcher pid=$pid"
  kill -INT "$pid" 2>/dev/null || return 1
  local deadline=$((SECONDS + LAUNCHER_WAIT_SEC))
  while ((SECONDS < deadline)); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.2
  done
  log "Launcher is still alive after ${LAUNCHER_WAIT_SEC}s; falling back to direct cleanup."
  return 1
}

prepare_arm_shutdown() {
  ((DRY_RUN == 0)) || {
    log "DRY-RUN: would deactivate arm controllers and Franka hardware"
    return 0
  }

  local services
  services="$(timeout 3s bash -lc "$(ros_env_cmd) ros2 service list" 2>/dev/null || true)"
  local side
  for side in right left; do
    grep -Fxq "/$side/controller_manager/switch_controller" <<<"$services" || continue
    log "Deactivating $side arm controllers and hardware"
    timeout "${ARM_SHUTDOWN_TIMEOUT_SEC}s" bash -lc \
      "$(ros_env_cmd) ros2 control switch_controllers --deactivate joint_impedance_controller franka_robot_state_broadcaster joint_state_broadcaster --switch-timeout 5 -c '/$side/controller_manager'" \
      >/dev/null 2>&1 || log "WARNING: failed to deactivate all $side arm controllers"
    timeout "${ARM_SHUTDOWN_TIMEOUT_SEC}s" bash -lc \
      "$(ros_env_cmd) ros2 control set_hardware_component_state '${side}_FrankaHardwareInterface' inactive -c '/$side/controller_manager'" \
      >/dev/null 2>&1 || log "WARNING: failed to deactivate $side Franka hardware"
  done
}

remove_runtime_files() {
  [[ -f "$STATE_DIR/runtime_files" ]] || return 0
  local path basename
  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    basename="$(basename "$path")"
    if [[ "$basename" != runtime_fine_duo_* ]]; then
      log "WARNING: refusing to remove unexpected runtime path: $path"
      continue
    fi
    if ((DRY_RUN)); then
      log "DRY-RUN: would remove $path"
    elif [[ -f "$path" ]]; then
      rm -f -- "$path"
    fi
  done <"$STATE_DIR/runtime_files"
}

cleanup_state_dir() {
  ((DRY_RUN == 0)) || return 0
  rm -f -- "$STATE_DIR/launcher.pid" "$STATE_DIR/task_pids" "$STATE_DIR/runtime_files"
  rmdir "$STATE_DIR" 2>/dev/null || true
}

kill_tmux_sessions() {
  ((KILL_TMUX)) || return 0
  command -v tmux >/dev/null 2>&1 || return 0
  local session
  for session in single duo fr3_single fr3_duo fr3_fine; do
    tmux has-session -t "$session" 2>/dev/null || continue
    if ((DRY_RUN)); then
      log "DRY-RUN: would kill tmux session $session"
    else
      tmux kill-session -t "$session"
    fi
  done
}

stop_ros_daemon() {
  ((STOP_ROS_DAEMON)) || return 0
  if ((DRY_RUN)); then
    log "DRY-RUN: would stop the ROS 2 daemon"
    return 0
  fi
  if command -v ros2 >/dev/null 2>&1; then
    ros2 daemon stop >/dev/null 2>&1 || true
  elif [[ -f /opt/ros/humble/setup.bash ]]; then
    bash -c 'source /opt/ros/humble/setup.bash; ros2 daemon stop' >/dev/null 2>&1 || true
  fi
}

main() {
  kill_tmux_sessions

  if request_launcher_shutdown; then
    if ((DRY_RUN == 0)); then
      cleanup_state_dir
      stop_ros_daemon
      log "Cleanup complete."
    fi
    return 0
  fi

  refresh_targets

  if ((${#TRACKED_PIDS[@]} == 0)); then
    log "No FR3/GELLO/gamepad teleoperation processes found."
  else
    log "Matched processes:"
    print_targets
    prepare_arm_shutdown
    signal_targets INT

    if ((DRY_RUN == 0)); then
      local deadline=$((SECONDS + WAIT_SEC))
      while ((SECONDS < deadline)); do
        [[ "$(alive_target_count)" == 0 ]] && break
        sleep 0.2
      done
      if [[ "$(alive_target_count)" != 0 ]]; then
        log "Processes still alive after SIGINT; sending SIGTERM."
        signal_targets TERM
        deadline=$((SECONDS + TERM_WAIT_SEC))
        while ((SECONDS < deadline)); do
          [[ "$(alive_target_count)" == 0 ]] && break
          sleep 0.2
        done
        if [[ "$(alive_target_count)" != 0 ]]; then
          log "Processes still alive after ${TERM_WAIT_SEC}s; sending SIGKILL."
          signal_targets KILL
          sleep 0.5
        fi
      fi
    fi
  fi

  remove_runtime_files
  cleanup_state_dir
  stop_ros_daemon
  log "Cleanup complete."
}

main "$@"
