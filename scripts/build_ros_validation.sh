#!/usr/bin/env bash
# Dedicated mock overlay; never replace the managed workspace used by sessions.
set -eo pipefail
export VLA_ROOT="${VLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VALIDATION_ROOT="${VLA_VALIDATION_ROOT:-/tmp/vla-ros-validation-$UID}"
[[ "$VALIDATION_ROOT" == /tmp/* ]] || { echo 'Validation overlay must be under /tmp'; exit 2; }
source "$VLA_ROOT/scripts/ros_env.sh"
cd "$VLA_ROS_WORKSPACE"
exec colcon --log-base "$VALIDATION_ROOT/log" build --build-base "$VALIDATION_ROOT/build" \
  --install-base "$VALIDATION_ROOT/install" --packages-select franka_fr3_arm_controllers vla_control_validation \
  --parallel-workers 2 --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3 "$@"
