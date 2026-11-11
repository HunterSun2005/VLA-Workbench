#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/ros_env.sh"
/usr/bin/python3 "$VLA_ROOT/scripts/check_libfranka_source.py"
cd "$VLA_ROOT/ros_ws"
exec colcon build --base-paths src --packages-up-to franka_fr3_arm_controllers franka_gripper_manager franka_gello_state_publisher franka_hardware franka_robot_state_broadcaster franka_description vla_bringup --parallel-workers 2 --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3 "$@"
