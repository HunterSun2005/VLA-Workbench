#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TELEOP_MODE=duo
export CONTROL_OWNER=rollout
export START_CAMERAS=0
exec "$SCRIPT_DIR/start_teleop.sh" "$@"
