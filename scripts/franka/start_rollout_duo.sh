#!/usr/bin/env bash
set -euo pipefail
VLA_ROOT="${VLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export VLA_ROOT
exec bash "$VLA_ROOT/src/vla/datacollection/start_teleop/start_rollout_duo.sh" "$@"
