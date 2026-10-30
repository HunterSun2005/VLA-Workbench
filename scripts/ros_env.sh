#!/usr/bin/env bash
# Source this in a dedicated ROS shell, never in the model process.
export VLA_ROOT="${VLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
unset PYTHONHOME PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH LD_LIBRARY_PATH
export PATH=/usr/bin:/bin:/usr/sbin:/sbin PYTHONNOUSERSITE=1
set +u
source /opt/ros/humble/setup.bash
if [[ -f "$VLA_ROOT/ros_ws/install/local_setup.bash" ]]; then
  source "$VLA_ROOT/ros_ws/install/local_setup.bash"
fi
export PYTHONPATH="$VLA_ROOT/ros_ws/python:$VLA_ROOT/ros_ws/python_deps:$VLA_ROOT/src:${PYTHONPATH:-}"
export CMAKE_PREFIX_PATH="/opt/openrobots:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="/opt/openrobots/lib:${LD_LIBRARY_PATH:-}"
export ROS_HOME="${VLA_ROOT}/runtime/ros"
export ROS_LOG_DIR="${ROS_HOME}/log"
