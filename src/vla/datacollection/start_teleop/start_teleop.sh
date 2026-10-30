#!/usr/bin/env bash
set -Eeuo pipefail

# FR3 GELLO teleoperation backend for the explicit single-left/duo entrypoints.
#
# Controls:
#   s  enter left-arm fine mode
#   d  return left arm/gripper smoothly to GELLO, then restore GELLO control
#   Esc/c  stop the complete stack cleanly

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TELEOP_MODE="${TELEOP_MODE:-duo}"
CONTROL_OWNER="${CONTROL_OWNER:-teleop}"
case "$CONTROL_OWNER" in
  teleop|rollout) ;;
  *)
    printf 'Unsupported CONTROL_OWNER: %s (expected teleop or rollout)\n' "$CONTROL_OWNER" >&2
    exit 2
    ;;
esac
case "$TELEOP_MODE" in
  duo)
    MODE_LABEL="dual"
    EXPECTED_GELLO_NAMESPACES="left,right"
    ACTIVE_SIDES=(left right)
    ;;
  single_left)
    MODE_LABEL="single-left"
    EXPECTED_GELLO_NAMESPACES="left"
    ACTIVE_SIDES=(left)
    ;;
  *)
    printf 'Unsupported TELEOP_MODE: %s (expected duo or single_left)\n' "$TELEOP_MODE" >&2
    exit 2
    ;;
esac

ROS2_WS="${ROS2_WS:-${VLA_ROOT:-$PWD}/ros_ws}"
FRANKA_ROS2_WS="${FRANKA_ROS2_WS:-$ROS2_WS}"
ROBOTIQ_ROS2_SETUP="${ROBOTIQ_ROS2_SETUP:-$ROS2_WS/install/local_setup.bash}"
OPENROBOTS_LIB_DIR="${OPENROBOTS_LIB_DIR:-/opt/openrobots/lib}"
FRANKA_HARDWARE_LIBRARY="${FRANKA_HARDWARE_LIBRARY:-$FRANKA_ROS2_WS/install/franka_hardware/lib/libfranka_hardware.so}"
LOG_DIR="${LOG_DIR:-$ROS2_WS/log/${TELEOP_MODE}_${CONTROL_OWNER}}"
STATE_DIR="${STATE_DIR:-${VLA_ROOT:-$PWD}/runtime/fr3_fine_teleop_${UID}}"

FINE_TELEOP_NODE="$SCRIPT_DIR/fine_teleop.py"
PREPARE_CONFIGS="$SCRIPT_DIR/prepare_runtime_configs.py"
CONTROLLER_STATE_CHECK="$SCRIPT_DIR/check_controller_state.py"
GELLO_FINE_LAUNCH="$SCRIPT_DIR/gello_duo_fine.launch.py"
ARM_TELEOP_LAUNCH="$SCRIPT_DIR/fr3_duo_teleop.launch.py"
GRIPPER_TELEOP_LAUNCH="$SCRIPT_DIR/robotiq_duo_teleop.launch.py"

GELLO_CONFIG="${GELLO_CONFIG:-franka_gello_duo.yaml}"
ROBOT_CONFIG="${ROBOT_CONFIG:-example_fr3_duo_config.yaml}"
GRIPPER_CONFIG="${GRIPPER_CONFIG:-example_fr3_duo_config_robotiq.yaml}"

LEFT_GELLO_SERIAL_ID="${LEFT_GELLO_SERIAL_ID:-}"
RIGHT_GELLO_SERIAL_ID="${RIGHT_GELLO_SERIAL_ID:-}"
LEFT_ROBOT_IP="${LEFT_ROBOT_IP:-}"
RIGHT_ROBOT_IP="${RIGHT_ROBOT_IP:-}"
LEFT_ROBOTIQ_PORT="${LEFT_ROBOTIQ_PORT:-}"
RIGHT_ROBOTIQ_PORT="${RIGHT_ROBOTIQ_PORT:-}"

JOY_DEVICE_NAME="${JOY_DEVICE_NAME:-Xbox Series X Controller}"
JOY_AUTOREPEAT_RATE="${JOY_AUTOREPEAT_RATE:-60.0}"
LEFT_JOY_DEVICE_ID="${LEFT_JOY_DEVICE_ID:-0}"
RIGHT_JOY_DEVICE_ID="${RIGHT_JOY_DEVICE_ID:-1}"
LEFT_STICK_X_SIGN="${LEFT_STICK_X_SIGN:-1.0}"
LEFT_STICK_Y_SIGN="${LEFT_STICK_Y_SIGN:-1.0}"
FINE_CONTROL_RATE="${FINE_CONTROL_RATE:-100.0}"
FINE_POSITION_GAIN="${FINE_POSITION_GAIN:-8.0}"
FINE_MAX_JOINT_SPEED="${FINE_MAX_JOINT_SPEED:-0.30}"
FINE_JOINT7_SPEED="${FINE_JOINT7_SPEED:-0.30}"
FINE_JOINT7_SIGN="${FINE_JOINT7_SIGN:--1.0}"
FINE_XY_SPEED="${FINE_XY_SPEED:-0.02}"
FINE_Z_SPEED="${FINE_Z_SPEED:-0.01}"
FINE_GRIPPER_SPEED="${FINE_GRIPPER_SPEED:-0.40}"
RETURN_JOINT_SPEED="${RETURN_JOINT_SPEED:-0.20}"
RETURN_GRIPPER_SPEED="${RETURN_GRIPPER_SPEED:-1.0}"
KEYBOARD_POLL_SEC="${KEYBOARD_POLL_SEC:-0.02}"

START_CAMERAS="${START_CAMERAS:-0}"
SET_FULL_COLLISION_BEHAVIOR="${SET_FULL_COLLISION_BEHAVIOR:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"
BASE_CAMERA_SERIAL="${BASE_CAMERA_SERIAL:-260722303785}"
LEFT_WRIST_CAMERA_SERIAL="${LEFT_WRIST_CAMERA_SERIAL:-${WRIST_CAMERA_SERIAL:-346222073620}}"
RIGHT_WRIST_CAMERA_SERIAL="${RIGHT_WRIST_CAMERA_SERIAL:-${SIDE_CAMERA_SERIAL:-346122070471}}"

STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-90}"
SERVICE_CALL_TIMEOUT_SEC="${SERVICE_CALL_TIMEOUT_SEC:-10}"
TASK_INT_TIMEOUT_SEC="${TASK_INT_TIMEOUT_SEC:-20}"
TASK_TERM_TIMEOUT_SEC="${TASK_TERM_TIMEOUT_SEC:-5}"
ARM_SHUTDOWN_TIMEOUT_SEC="${ARM_SHUTDOWN_TIMEOUT_SEC:-10}"
CONTROLLER_ACTIVE_TIMEOUT_SEC="${CONTROLLER_ACTIVE_TIMEOUT_SEC:-10}"

declare -a TASK_NAMES=()
declare -a TASK_COMMANDS=()
declare -a TASK_PIDS=()
declare -a TASK_INT_SCOPES=()
declare -a TASK_LOG_OFFSETS=()
declare -a RUNTIME_FILES=()

SHUTTING_DOWN=0
GELLO_RUNTIME_CONFIG=""
ARM_RUNTIME_CONFIG=""
GRIPPER_RUNTIME_CONFIG=""
GRIPPER_SPEED_CONTROLLER_CONFIG=""
LAST_CONTROLLER_STATUS=""

# Internal helpers share the launcher settings above. Keep SCRIPT_DIR at the
# public entrypoint so launch files, Python nodes and config paths stay stable.
source "$SCRIPT_DIR/lib/environment.sh"
source "$SCRIPT_DIR/lib/processes.sh"
source "$SCRIPT_DIR/lib/controllers.sh"
source "$SCRIPT_DIR/lib/keyboard.sh"

main() {
  trap on_exit EXIT
  trap 'log "Ctrl-C exit is disabled; press Esc or c for an orderly shutdown."' INT
  trap 'exit 143' TERM

  ensure_single_instance
  validate_environment
  prepare_runtime_configs

  if [[ "$CONTROL_OWNER" == "teleop" ]]; then
    bash --noprofile --norc -c "$(ros_env_cmd) /usr/bin/python3 '$GELLO_FINE_LAUNCH' --check --config-path '$GELLO_RUNTIME_CONFIG' --expected-namespaces '$EXPECTED_GELLO_NAMESPACES'" ||
      die "GELLO fine launch configuration check failed"
  fi
  bash --noprofile --norc -c "$(ros_env_cmd) ros2 launch '$ARM_TELEOP_LAUNCH' --show-args >/dev/null" ||
    die "FR3 teleop topic remap launch check failed"
  bash --noprofile --norc -c "$(ros_env_cmd) ros2 launch '$GRIPPER_TELEOP_LAUNCH' --show-args >/dev/null" ||
    die "Robotiq teleop topic remap launch check failed"

  if is_enabled "$CHECK_ONLY"; then
    log "Configuration and $CONTROL_OWNER self-check passed; hardware was not started."
    return 0
  fi

  local environment gello_pythonpath
  environment="$(ros_env_cmd)"
  gello_pythonpath="$ROS2_WS/build/franka_gello_state_publisher:$ROS2_WS/src/franka_gello_state_publisher"

  if [[ "$CONTROL_OWNER" == "teleop" ]]; then
    add_task "gello_${TELEOP_MODE}_fine" \
      "${environment} export PYTHONPATH=${gello_pythonpath}:\${PYTHONPATH:-}; exec /usr/bin/python3 '$GELLO_FINE_LAUNCH' --config-path '$GELLO_RUNTIME_CONFIG'"
    add_task "xbox_joy" \
      "${environment} exec ros2 run joy joy_node --ros-args -r joy:=/gamepad/joy -p device_name:='$JOY_DEVICE_NAME' -p deadzone:=0.0 -p autorepeat_rate:=$JOY_AUTOREPEAT_RATE" \
      "group"
  else
    local side device_id
    for side in "${ACTIVE_SIDES[@]}"; do
      if [[ "$side" == "left" ]]; then
        device_id="$LEFT_JOY_DEVICE_ID"
      else
        device_id="$RIGHT_JOY_DEVICE_ID"
      fi
      add_task "${side}_rollout_gamepad" \
        "${environment} exec ros2 run joy joy_node --ros-args -r __node:=${side}_rollout_joy -r joy:=/${side}/gamepad/joy -p device_id:=$device_id -p deadzone:=0.0 -p autorepeat_rate:=$JOY_AUTOREPEAT_RATE" \
        "group"
    done
  fi

  if is_enabled "$START_CAMERAS"; then
    add_task "camera_base" \
      "${environment} exec ros2 launch realsense2_camera rs_launch.py camera_name:=camera_base serial_no:=_${BASE_CAMERA_SERIAL} enable_depth:=false rgb_camera.color_profile:=640x480x30"
    add_task "camera_left_wrist" \
      "${environment} exec ros2 launch realsense2_camera rs_launch.py camera_name:=camera_left_wrist serial_no:=_${LEFT_WRIST_CAMERA_SERIAL} enable_depth:=false rgb_camera.color_profile:=640x480x30"
    if [[ "$TELEOP_MODE" == "duo" ]]; then
      add_task "camera_right_wrist" \
        "${environment} exec ros2 launch realsense2_camera rs_launch.py camera_name:=camera_right_wrist serial_no:=_${RIGHT_WRIST_CAMERA_SERIAL} enable_depth:=false rgb_camera.color_profile:=640x480x30"
    fi
  fi

  local index
  for index in "${!TASK_NAMES[@]}"; do
    start_task "$index"
  done

  add_task "robotiq_${TELEOP_MODE}" \
    "${environment} exec ros2 launch '$GRIPPER_TELEOP_LAUNCH' config_file:='$GRIPPER_RUNTIME_CONFIG' control_owner:=$CONTROL_OWNER"
  start_task "$((${#TASK_NAMES[@]} - 1))"
  local side
  for side in "${ACTIVE_SIDES[@]}"; do
    wait_for_action "/$side/gripper/robotiq_gripper_controller/gripper_cmd"
  done
  if [[ "$CONTROL_OWNER" == "teleop" ]]; then
    start_gripper_speed_controller left
  else
    for side in "${ACTIVE_SIDES[@]}"; do
      start_gripper_speed_controller "$side"
    done
  fi

  add_task "fr3_${TELEOP_MODE}_controllers" \
    "${environment} exec ros2 launch '$ARM_TELEOP_LAUNCH' robot_config_file:='$ARM_RUNTIME_CONFIG' control_owner:=$CONTROL_OWNER"
  start_task "$((${#TASK_NAMES[@]} - 1))"

  if is_enabled "$SET_FULL_COLLISION_BEHAVIOR"; then
    for side in "${ACTIVE_SIDES[@]}"; do
      apply_collision_behavior "$side"
    done
  fi

  if [[ "$CONTROL_OWNER" == "teleop" ]]; then
    add_task "left_fine_teleop" \
      "${environment} exec /usr/bin/python3 '$FINE_TELEOP_NODE' --ros-args -r __ns:=/left -p control_rate:=$FINE_CONTROL_RATE -p position_gain:=$FINE_POSITION_GAIN -p max_joint_speed:=$FINE_MAX_JOINT_SPEED -p joint7_speed:=$FINE_JOINT7_SPEED -p joint7_button_sign:=$FINE_JOINT7_SIGN -p left_stick_x_sign:=$LEFT_STICK_X_SIGN -p left_stick_y_sign:=$LEFT_STICK_Y_SIGN -p max_xy_speed:=$FINE_XY_SPEED -p max_z_speed:=$FINE_Z_SPEED -p gripper_speed:=$FINE_GRIPPER_SPEED -p return_joint_speed:=$RETURN_JOINT_SPEED -p gripper_return_speed:=$RETURN_GRIPPER_SPEED"
    start_task "$((${#TASK_NAMES[@]} - 1))"
    wait_for_service "/left/left_fine_teleop/set_fine_mode"
  fi

  if [[ "$CONTROL_OWNER" == "rollout" ]]; then
    log "$MODE_LABEL FR3 rollout hardware stack is ready; collector owns all arm commands."
  elif [[ "$TELEOP_MODE" == "duo" ]]; then
    log "Dual FR3 teleoperation stack is ready. Right arm remains under GELLO control."
  else
    log "Single-left FR3 teleoperation stack is ready."
  fi
  log "Logs: $LOG_DIR"
  if [[ -n "${VLA_WEB_READY_FILE:-}" ]]; then
    printf '%s\n' "$$" >"$VLA_WEB_READY_FILE"
  fi
  if [[ "$CONTROL_OWNER" == "rollout" ]]; then
    monitor_rollout_hardware
  else
    monitor_with_keyboard
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
