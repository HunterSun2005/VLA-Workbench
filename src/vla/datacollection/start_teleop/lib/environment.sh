#!/usr/bin/env bash
# Internal start_teleop.sh functions; source after launcher settings are initialized.
# Uses the launcher globals and sibling lib functions; does not start hardware on source.

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

is_enabled() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_file() {
  [[ -f "$1" ]] || die "Missing required file: $1"
}

ros_env_cmd() {
  printf 'source %q/install/setup.bash; ' "$FRANKA_ROS2_WS"
  printf 'source %q; ' "$ROBOTIQ_ROS2_SETUP"
  printf 'source %q/install/setup.bash; ' "$ROS2_WS"
  printf 'export LD_LIBRARY_PATH=%q:${LD_LIBRARY_PATH:-}; ' "$OPENROBOTS_LIB_DIR"
}

package_share() {
  local package="$1"
  bash -lc "$(ros_env_cmd) ros2 pkg prefix '$package'" |
    tail -n 1 | awk "{print \$1 \"/share/$package\"}"
}

config_source_path() {
  local package="$1"
  local config="$2"
  local source_path="$ROS2_WS/src/$package/config/$config"
  if [[ -f "$source_path" ]]; then
    printf '%s\n' "$source_path"
  else
    printf '%s/config/%s\n' "$(package_share "$package")" "$config"
  fi
}

validate_environment() {
  require_file "$ROS2_WS/install/setup.bash"
  require_file "$FRANKA_ROS2_WS/install/setup.bash"
  require_file "$ROBOTIQ_ROS2_SETUP"
  [[ -d "$OPENROBOTS_LIB_DIR" ]] ||
    die "Missing robotpkg library directory: $OPENROBOTS_LIB_DIR"
  require_file "$FRANKA_HARDWARE_LIBRARY"
  require_file "$PREPARE_CONFIGS"
  require_file "$CONTROLLER_STATE_CHECK"
  require_file "$ARM_TELEOP_LAUNCH"
  require_file "$GRIPPER_TELEOP_LAUNCH"
  if [[ "$CONTROL_OWNER" == "teleop" ]]; then
    require_file "$FINE_TELEOP_NODE"
    require_file "$GELLO_FINE_LAUNCH"
  fi

  local package
  local -a packages=(
    franka_hardware
    franka_robot_state_broadcaster
    franka_fr3_arm_controllers
    franka_gripper_manager
    robotiq_driver
    robotiq_controllers
    joy
  )
  if [[ "$CONTROL_OWNER" == "teleop" ]]; then
    packages+=(franka_gello_state_publisher forward_command_controller std_srvs)
  fi
  for package in "${packages[@]}"; do
    bash -lc "$(ros_env_cmd) ros2 pkg prefix '$package' >/dev/null 2>&1" ||
      die "Missing ROS package: $package"
  done

  bash -lc "$(ros_env_cmd) /usr/bin/python3 -c 'import PyKDL, controller_manager_msgs.srv, numpy, rclpy, urdf_parser_py.urdf'" ||
    die "FR3 control Python dependencies are unavailable"
  if [[ "$CONTROL_OWNER" == "teleop" ]]; then
    bash -lc "$(ros_env_cmd) /usr/bin/python3 '$FINE_TELEOP_NODE' --self-test" ||
      die "Fine teleop self-test failed"
  fi

  local hardware_library_q missing_libraries
  printf -v hardware_library_q '%q' "$FRANKA_HARDWARE_LIBRARY"
  missing_libraries="$(
    bash -lc "$(ros_env_cmd) ldd $hardware_library_q" |
      awk '$2 == "=>" && $3 == "not" && $4 == "found" {print $1}'
  )"
  if [[ -n "$missing_libraries" ]]; then
    die "Franka hardware library has unresolved dependencies: ${missing_libraries//$'\n'/, }"
  fi

  if ! is_enabled "$CHECK_ONLY"; then
    local existing_nodes
    existing_nodes="$(bash -lc "$(ros_env_cmd) ros2 node list" 2>/dev/null || true)"
    if grep -Eq '^/(left|right)/(controller_manager|gello_publisher)(/|$)' <<<"$existing_nodes"; then
      die "An FR3/GELLO stack is already running; run $SCRIPT_DIR/clean_teleop.sh first"
    fi
  fi

  if ! is_enabled "$CHECK_ONLY" && [[ ! -t 0 ]]; then
    die "An interactive terminal is required for keyboard shutdown"
  fi

  if [[ "$CONTROL_OWNER" == "rollout" ]]; then
    local side device_id
    [[ "$LEFT_JOY_DEVICE_ID" =~ ^[0-9]+$ ]] ||
      die "LEFT_JOY_DEVICE_ID must be a non-negative integer"
    if [[ "$TELEOP_MODE" == "duo" ]]; then
      [[ "$RIGHT_JOY_DEVICE_ID" =~ ^[0-9]+$ ]] ||
        die "RIGHT_JOY_DEVICE_ID must be a non-negative integer"
      if [[ "$LEFT_JOY_DEVICE_ID" == "$RIGHT_JOY_DEVICE_ID" ]]; then
        die "Dual rollout requires different LEFT_JOY_DEVICE_ID and RIGHT_JOY_DEVICE_ID values"
      fi
    fi
    for side in "${ACTIVE_SIDES[@]}"; do
      if [[ "$side" == "left" ]]; then
        device_id="$LEFT_JOY_DEVICE_ID"
      else
        device_id="$RIGHT_JOY_DEVICE_ID"
      fi
      [[ -e "/dev/input/js$device_id" ]] ||
        log "WARNING: $side gamepad /dev/input/js$device_id is missing."
    done
  elif [[ ! -e /dev/input/js0 ]]; then
    log "WARNING: /dev/input/js0 is missing; joy_node will not start successfully."
  fi
}

add_runtime_file() {
  RUNTIME_FILES+=("$1")
  printf '%s\n' "$1" >>"$STATE_DIR/runtime_files"
}

prepare_runtime_configs() {
  local gello_source="" arm_source gripper_source
  local arm_share gripper_share suffix
  if [[ "$CONTROL_OWNER" == "teleop" ]]; then
    gello_source="$(config_source_path franka_gello_state_publisher "$GELLO_CONFIG")"
    require_file "$gello_source"
  fi
  arm_source="$(config_source_path franka_fr3_arm_controllers "$ROBOT_CONFIG")"
  gripper_source="$(config_source_path franka_gripper_manager "$GRIPPER_CONFIG")"
  require_file "$arm_source"
  require_file "$gripper_source"

  arm_share="$(package_share franka_fr3_arm_controllers)"
  gripper_share="$(package_share franka_gripper_manager)"
  suffix="runtime_${CONTROL_OWNER}_${TELEOP_MODE}_${USER:-user}_$$"
  ARM_RUNTIME_CONFIG="$arm_share/config/${suffix}_arms.yaml"
  GRIPPER_RUNTIME_CONFIG="$gripper_share/config/${suffix}_grippers.yaml"
  GRIPPER_SPEED_CONTROLLER_CONFIG="$STATE_DIR/${suffix}_gripper_speed_controller.yaml"
  if [[ "$CONTROL_OWNER" == "teleop" ]]; then
    GELLO_RUNTIME_CONFIG="$STATE_DIR/${suffix}_gello.yaml"
    add_runtime_file "$GELLO_RUNTIME_CONFIG"
  fi
  add_runtime_file "$ARM_RUNTIME_CONFIG"
  add_runtime_file "$GRIPPER_RUNTIME_CONFIG"
  add_runtime_file "$GRIPPER_SPEED_CONTROLLER_CONFIG"

  local -a config_args=(
    --mode "$TELEOP_MODE"
    --control-owner "$CONTROL_OWNER"
    --arm-source "$arm_source"
    --gripper-source "$gripper_source"
    --arm-output "$ARM_RUNTIME_CONFIG"
    --gripper-output "$GRIPPER_RUNTIME_CONFIG"
    --gripper-speed-output "$GRIPPER_SPEED_CONTROLLER_CONFIG"
    --left-gello "$LEFT_GELLO_SERIAL_ID"
    --right-gello "$RIGHT_GELLO_SERIAL_ID"
    --left-ip "$LEFT_ROBOT_IP"
    --right-ip "$RIGHT_ROBOT_IP"
    --left-gripper "$LEFT_ROBOTIQ_PORT"
    --right-gripper "$RIGHT_ROBOTIQ_PORT"
  )
  if [[ "$CONTROL_OWNER" == "teleop" ]]; then
    config_args+=(
      --gello-source "$gello_source"
      --gello-output "$GELLO_RUNTIME_CONFIG"
    )
  fi
  /usr/bin/python3 "$PREPARE_CONFIGS" "${config_args[@]}"
  log "Control mode: $MODE_LABEL $CONTROL_OWNER"
  if [[ "$CONTROL_OWNER" == "teleop" ]]; then
    log "Runtime GELLO config: $GELLO_RUNTIME_CONFIG"
  fi
  log "Runtime arm config: $(basename "$ARM_RUNTIME_CONFIG")"
  log "Runtime gripper config: $(basename "$GRIPPER_RUNTIME_CONFIG")"
  log "Runtime left gripper speed controller config: $GRIPPER_SPEED_CONTROLLER_CONFIG"
}
