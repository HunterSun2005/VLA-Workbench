#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TELEOP_MODE=single_left
exec "$SCRIPT_DIR/start_teleop.sh" "$@"
