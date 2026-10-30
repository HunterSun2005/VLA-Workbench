#!/usr/bin/env bash
# Internal start_teleop.sh functions; source after launcher settings are initialized.
# Uses the launcher globals and sibling lib functions; does not start hardware on source.

ensure_single_instance() {
  local pid_file="$STATE_DIR/launcher.pid"
  if [[ -f "$pid_file" ]]; then
    local old_pid
    old_pid="$(tr -cd '0-9' <"$pid_file")"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      die "FR3 $CONTROL_OWNER launcher is already running with pid=$old_pid"
    fi
  fi

  mkdir -p "$STATE_DIR" "$LOG_DIR"
  printf '%s\n' "$$" >"$pid_file"
  : >"$STATE_DIR/task_pids"
  : >"$STATE_DIR/runtime_files"
}

add_task() {
  TASK_NAMES+=("$1")
  TASK_COMMANDS+=("$2")
  TASK_PIDS+=("")
  TASK_INT_SCOPES+=("${3:-pid}")
  TASK_LOG_OFFSETS+=("0")
}

start_task() {
  local index="$1"
  local name="${TASK_NAMES[$index]}"
  local log_file="$LOG_DIR/${name}.log"
  log "Starting $name"
  if [[ -f "$log_file" ]]; then
    TASK_LOG_OFFSETS[$index]="$(wc -c <"$log_file")"
  else
    TASK_LOG_OFFSETS[$index]="0"
  fi
  # Non-interactive shells start asynchronous children with SIGINT ignored.
  # Reset it before exec so ROS/Python can install their graceful handlers.
  setsid env --default-signal=INT,QUIT bash -lc "${TASK_COMMANDS[$index]}" \
    >>"$log_file" 2>&1 &
  TASK_PIDS[$index]="$!"
  printf '%s %s\n' "$!" "$name" >>"$STATE_DIR/task_pids"
  log "$name pid=$! log=$log_file"
}

task_is_alive() {
  local pid="$1"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  local state
  state="$(ps -p "$pid" -o stat= 2>/dev/null | awk '{print $1}')"
  [[ -n "$state" && "$state" != Z* ]]
}

assert_tasks_alive() {
  local index
  for index in "${!TASK_NAMES[@]}"; do
    task_is_alive "${TASK_PIDS[$index]}" ||
      die "Task ${TASK_NAMES[$index]} exited; see $LOG_DIR/${TASK_NAMES[$index]}.log"
  done
}

stop_task() {
  local index="$1"
  local pid="${TASK_PIDS[$index]}"
  [[ -n "$pid" ]] || return 0

  if task_is_alive "$pid"; then
    log "Stopping ${TASK_NAMES[$index]} pid=$pid with SIGINT"
    # Signal only the ros2/launch parent first. It owns the shutdown order for
    # its native children and avoids concurrent driver teardown.
    local int_target="$pid"
    if [[ "${TASK_INT_SCOPES[$index]}" == "group" ]]; then
      # ros2 run waits for its executable but does not forward a PID-only
      # SIGINT, so both wrapper and executable need the group signal.
      int_target="-$pid"
    fi
    kill -INT -- "$int_target" 2>/dev/null || true
    if ! wait_for_task_exit "$pid" "$TASK_INT_TIMEOUT_SEC"; then
      log "${TASK_NAMES[$index]} did not exit after ${TASK_INT_TIMEOUT_SEC}s; sending SIGTERM to process group $pid"
      kill -TERM -- "-$pid" 2>/dev/null || true
      if ! wait_for_task_exit "$pid" "$TASK_TERM_TIMEOUT_SEC"; then
        log "${TASK_NAMES[$index]} did not exit after SIGTERM; sending SIGKILL to process group $pid"
        kill -KILL -- "-$pid" 2>/dev/null || true
        wait_for_task_exit "$pid" 2 || true
      fi
    fi
  fi

  record_task_exit "$index"
}

wait_for_task_exit() {
  local pid="$1"
  local timeout_sec="$2"
  local deadline=$((SECONDS + timeout_sec))
  while ((SECONDS < deadline)); do
    task_is_alive "$pid" || return 0
    sleep 0.2
  done
  ! task_is_alive "$pid"
}

record_task_exit() {
  local index="$1"
  local pid="${TASK_PIDS[$index]}"
  local log_file="$LOG_DIR/${TASK_NAMES[$index]}.log"
  local log_offset="${TASK_LOG_OFFSETS[$index]}"
  local status=0
  local nested_segfault=0
  [[ -n "$pid" ]] || return 0

  if wait "$pid" 2>/dev/null; then
    status=0
  else
    status=$?
  fi
  TASK_PIDS[$index]=""

  if [[ -f "$log_file" ]] && grep -Eiq 'segmentation fault|exit code -11' \
    < <(tail -c "+$((log_offset + 1))" "$log_file" 2>/dev/null); then
    nested_segfault=1
    log "ERROR: task ${TASK_NAMES[$index]} reported a child-process segmentation fault; see $log_file"
  fi

  case "$status" in
    0) ((nested_segfault)) || log "Task ${TASK_NAMES[$index]} exited cleanly" ;;
    130|143) log "Task ${TASK_NAMES[$index]} stopped by signal (status=$status)" ;;
    137) log "WARNING: task ${TASK_NAMES[$index]} required SIGKILL (status=$status)" ;;
    139)
      ((nested_segfault)) || log "ERROR: task ${TASK_NAMES[$index]} exited with segmentation fault (status=139); see $log_file"
      ;;
    *) log "WARNING: task ${TASK_NAMES[$index]} exited with status=$status; see $log_file" ;;
  esac
}

cleanup_runtime_files() {
  local path
  for path in "${RUNTIME_FILES[@]}"; do
    [[ -f "$path" ]] && rm -f -- "$path"
  done
  rm -f -- "$STATE_DIR/launcher.pid" "$STATE_DIR/task_pids" "$STATE_DIR/runtime_files"
  rmdir "$STATE_DIR" 2>/dev/null || true
}

shutdown_all() {
  ((SHUTTING_DOWN == 0)) || return 0
  SHUTTING_DOWN=1
  log "Shutting down teleoperation stack..."
  local index
  for ((index=${#TASK_NAMES[@]} - 1; index >= 0; index--)); do
    if [[ "${TASK_NAMES[$index]}" == "fr3_${TELEOP_MODE}_controllers" ]] && \
      task_is_alive "${TASK_PIDS[$index]}"; then
      prepare_arm_shutdown
    fi
    stop_task "$index"
  done
  cleanup_runtime_files
  log "Shutdown complete."
}

on_exit() {
  local status=$?
  shutdown_all
  exit "$status"
}
