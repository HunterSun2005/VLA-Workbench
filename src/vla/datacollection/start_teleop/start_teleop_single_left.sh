#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="${VLA_ROOT:-$PWD}"
MODEL_PYTHON="${VLA_PYTHON:-${CONDA_PREFIX:-/data0/miniconda3/envs/vla}/bin/python}"
extra=()
[[ "${CHECK_ONLY:-0}" == 1 ]] && extra+=(--check-only)
exec env PYTHONPATH="$REPOSITORY_ROOT/src" "$MODEL_PYTHON" -m vla.cli.ros_session --arms single_left --source gello "${extra[@]}" "$@"
