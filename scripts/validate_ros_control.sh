#!/usr/bin/env bash
set -eo pipefail
export VLA_ROOT="${VLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$VLA_ROOT/scripts/ros_env.sh"
VALIDATION_ROOT="${VLA_VALIDATION_ROOT:-/tmp/vla-ros-validation-$UID}"
[[ "$VALIDATION_ROOT" == /tmp/* ]] || exit 2
source "$VALIDATION_ROOT/install/local_setup.bash"
export OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec /usr/bin/python3 "$VLA_ROOT/scripts/validate_ros_control.py" "$@"
