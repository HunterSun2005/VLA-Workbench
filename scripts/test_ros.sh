#!/usr/bin/env bash
# Python tests using the real ROS/KDL interpreter; no hardware launch.
set -eo pipefail
export VLA_ROOT="${VLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$VLA_ROOT/scripts/ros_env.sh"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1
cd "$VLA_ROOT"
if [[ $# -eq 0 ]]; then set -- tests/ros; fi
exec /usr/bin/python3 -m pytest -c /dev/null -p no:cacheprovider "$@"
