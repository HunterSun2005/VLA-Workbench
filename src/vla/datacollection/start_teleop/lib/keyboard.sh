#!/usr/bin/env bash
# Internal start_teleop.sh functions; source after launcher settings are initialized.
# Uses the launcher globals and sibling lib functions; does not start hardware on source.

request_fine_mode() {
  local enabled="$1"
  local label="$2"
  local mode_signal="USR2"
  [[ "$enabled" == "true" ]] && mode_signal="USR1"

  local index pid=""
  for index in "${!TASK_NAMES[@]}"; do
    if [[ "${TASK_NAMES[$index]}" == "left_fine_teleop" ]]; then
      pid="${TASK_PIDS[$index]}"
      break
    fi
  done

  log "$label"
  if [[ -z "$pid" ]] || ! task_is_alive "$pid"; then
    log "WARNING: left_fine_teleop is unavailable; mode request ignored"
    return
  fi
  if ! kill "-$mode_signal" -- "$pid" 2>/dev/null; then
    log "WARNING: failed to signal left_fine_teleop pid=$pid"
  fi
}

monitor_with_keyboard() {
  log "Controls: s=fine mode, d=return to GELLO, Esc/c=stop"
  while true; do
    local key=""
    if IFS= read -rsn1 -t "$KEYBOARD_POLL_SEC" key; then
      case "${key,,}" in
        s) request_fine_mode true "Requesting fine gamepad mode..." ;;
        d) request_fine_mode false "Returning left arm and gripper to GELLO..." ;;
        c|$'\e')
          log "Exit requested from keyboard."
          return 0
          ;;
      esac
    fi
    assert_tasks_alive
  done
}

monitor_rollout_hardware() {
  log "Rollout hardware is ready; use the collector terminal for episode controls. Esc/c stops hardware."
  while true; do
    local key=""
    if IFS= read -rsn1 -t "$KEYBOARD_POLL_SEC" key; then
      case "${key,,}" in
        c|$'\e')
          log "Exit requested from keyboard."
          return 0
          ;;
      esac
    fi
    assert_tasks_alive
  done
}
