#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/ros_env.sh"
/usr/bin/python3 "$VLA_ROOT/scripts/check_libfranka_source.py"
cd "$VLA_ROS_WORKSPACE"
colcon build --base-paths src --packages-up-to franka_fr3_arm_controllers franka_gripper_manager franka_gello_state_publisher franka_hardware franka_robot_state_broadcaster franka_description vla_bringup --parallel-workers 2 --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3 "$@"

source "$VLA_ROS_WORKSPACE/install/local_setup.bash"
/usr/bin/python3 "$VLA_ROOT/scripts/check_ros.py"
/usr/bin/python3 - <<'ACTIVATE'
import os
from pathlib import Path
root = Path(os.environ['VLA_ROOT'])
workspace = Path(os.environ['VLA_ROS_WORKSPACE'])
pointer = root/'runtime/ros/current'
candidate = pointer.with_name(f'.current-{os.getpid()}')
candidate.symlink_to(workspace, target_is_directory=True)
candidate.replace(pointer)
ACTIVATE
