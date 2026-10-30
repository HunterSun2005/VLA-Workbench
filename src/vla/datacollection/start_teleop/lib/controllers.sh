#!/usr/bin/env bash
# Internal start_teleop.sh functions; source after launcher settings are initialized.
# Uses the launcher globals and sibling lib functions; does not start hardware on source.

prepare_arm_shutdown() {
  local services
  services="$(timeout 3s bash --noprofile --norc -c "$(ros_env_cmd) ros2 service list" 2>/dev/null || true)"
  local side side_index
  for ((side_index=${#ACTIVE_SIDES[@]} - 1; side_index >= 0; side_index--)); do
    side="${ACTIVE_SIDES[$side_index]}"
    if ! grep -Fxq "/$side/controller_manager/switch_controller" <<<"$services"; then
      log "WARNING: /$side/controller_manager is unavailable; skipping ordered controller shutdown"
      continue
    fi

    log "Deactivating $side arm controllers before driver shutdown"
    if ! timeout "${ARM_SHUTDOWN_TIMEOUT_SEC}s" bash --noprofile --norc -c \
      "$(ros_env_cmd) ros2 control switch_controllers --deactivate joint_impedance_controller franka_robot_state_broadcaster joint_state_broadcaster -c '/$side/controller_manager'" \
      >>"$LOG_DIR/shutdown_arms.log" 2>&1; then
      log "WARNING: failed to deactivate all $side arm controllers; see $LOG_DIR/shutdown_arms.log"
    fi

  done
}

wait_for_service() {
  local service="$1"
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SEC))
  log "Waiting for service $service ..."
  while ((SECONDS < deadline)); do
    assert_tasks_alive
    if bash --noprofile --norc -c "$(ros_env_cmd) ros2 service list" 2>/dev/null | grep -Fxq "$service"; then
      log "Service available: $service"
      return 0
    fi
    sleep 1
  done
  die "Timed out waiting for service $service"
}

wait_for_action() {
  local action="$1"
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SEC))
  log "Waiting for action $action ..."
  while ((SECONDS < deadline)); do
    assert_tasks_alive
    if bash --noprofile --norc -c "$(ros_env_cmd) ros2 action list" 2>/dev/null | grep -Fxq "$action"; then
      log "Action available: $action"
      return 0
    fi
    sleep 1
  done
  die "Timed out waiting for action $action"
}

wait_for_gripper_speed_controller() {
  local side="$1"
  local manager="/$side/gripper/controller_manager"
  local controller="${side}_gripper_speed_controller"
  LAST_CONTROLLER_STATUS=""
  if LAST_CONTROLLER_STATUS="$(timeout "$((CONTROLLER_ACTIVE_TIMEOUT_SEC + 5))s" bash --noprofile --norc -c \
    "$(ros_env_cmd) /usr/bin/python3 '$CONTROLLER_STATE_CHECK' --manager '$manager' --controller '$controller' --timeout-s '$CONTROLLER_ACTIVE_TIMEOUT_SEC'" \
    2>&1)"; then
    return 0
  fi
  return 1
}

report_gripper_controllers() {
  if [[ -z "$LAST_CONTROLLER_STATUS" ]]; then
    log "Controller manager returned an empty controller list."
    return
  fi
  log "Controller manager reported:"
  while IFS= read -r line; do
    log "  $line"
  done <<<"$LAST_CONTROLLER_STATUS"
}

start_gripper_speed_controller() {
  local side="$1"
  local manager="/$side/gripper/controller_manager"
  local controller="${side}_gripper_speed_controller"
  local log_file="$LOG_DIR/${controller}.log"
  local interface="robotiq_85_left_knuckle_joint/set_gripper_max_velocity"
  local interfaces

  interfaces="$(timeout 10s bash --noprofile --norc -c \
    "$(ros_env_cmd) ros2 control list_hardware_interfaces -c '$manager'" 2>/dev/null || true)"
  grep -Fq "$interface" <<<"$interfaces" ||
    die "$side Robotiq driver does not expose required interface: $interface"

  log "Starting $side Robotiq continuous speed controller"
  timeout "${STARTUP_TIMEOUT_SEC}s" bash --noprofile --norc -c \
    "$(ros_env_cmd) ros2 run controller_manager spawner '$controller' -c '$manager' -t forward_command_controller/ForwardCommandController -p '$GRIPPER_SPEED_CONTROLLER_CONFIG' --controller-manager-timeout '$STARTUP_TIMEOUT_SEC' --switch-timeout 10" \
    >>"$log_file" 2>&1 ||
    die "Failed to start $side gripper speed controller; see $log_file"

  if wait_for_gripper_speed_controller "$side"; then
    log "$side gripper speed controller is active."
    return
  fi

  log "Controller has not reported active; requesting explicit activation."
  if ! timeout 15s bash --noprofile --norc -c \
    "$(ros_env_cmd) ros2 control switch_controllers --activate '$controller' -c '$manager'" \
    >>"$log_file" 2>&1; then
    log "WARNING: explicit controller activation returned an error; checking final state."
  fi
  if wait_for_gripper_speed_controller "$side"; then
    log "$side gripper speed controller is active."
    return
  fi

  report_gripper_controllers
  die "$side gripper speed controller did not become active; see $log_file"
}

apply_collision_behavior() {
  local namespace="$1"
  local service="/$namespace/service_server/set_full_collision_behavior"
  local type="franka_msgs/srv/SetFullCollisionBehavior"
  local request='{
  lower_torque_thresholds_acceleration: [50.0, 50.0, 50.0, 40.0, 40.0, 40.0, 40.0],
  upper_torque_thresholds_acceleration: [100.0, 100.0, 100.0, 80.0, 80.0, 40.0, 40.0],
  lower_torque_thresholds_nominal: [50.0, 50.0, 50.0, 30.0, 40.0, 40.0, 40.0],
  upper_torque_thresholds_nominal: [100.0, 100.0, 100.0, 80.0, 80.0, 40.0, 40.0],
  lower_force_thresholds_acceleration: [50.0, 50.0, 50.0, 30.0, 30.0, 30.0],
  upper_force_thresholds_acceleration: [100.0, 100.0, 100.0, 30.0, 30.0, 30.0],
  lower_force_thresholds_nominal: [50.0, 50.0, 50.0, 30.0, 30.0, 30.0],
  upper_force_thresholds_nominal: [100.0, 100.0, 100.0, 30.0, 30.0, 30.0]
}'
  wait_for_service "$service"
  timeout "${SERVICE_CALL_TIMEOUT_SEC}s" bash --noprofile --norc -c \
    "$(ros_env_cmd) ros2 service call '$service' '$type' \"$request\"" \
    >>"$LOG_DIR/collision_${namespace}.log" 2>&1 ||
    die "Failed to configure collision behavior for $namespace"
  log "Collision behavior configured for $namespace."
}
