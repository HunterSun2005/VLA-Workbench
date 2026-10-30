#!/usr/bin/env bash
# Source after activating the vla Conda environment, or set VLA_PYTHON explicitly.
export VLA_ROOT="${VLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export VLA_PYTHON="${VLA_PYTHON:-${CONDA_PREFIX:?Activate the vla environment first}/bin/python}"
unset PYTHONHOME PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH LD_LIBRARY_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VLA_ROOT/src"
export PATH="$(dirname "$VLA_PYTHON"):/usr/local/bin:/usr/bin:/bin"
export HF_HOME="${HF_HOME:-$VLA_ROOT/runtime/huggingface}"
