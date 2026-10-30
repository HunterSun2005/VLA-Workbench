#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="${VLA_ROOT:-$PWD}"
MODEL_PYTHON="${VLA_PYTHON:-${CONDA_PREFIX:-$HOME/miniconda3/envs/vla}/bin/python}"
[[ -x "$MODEL_PYTHON" ]] || { printf 'Missing Python: %s\n' "$MODEL_PYTHON" >&2; exit 1; }
exec env -u PYTHONHOME -u LD_LIBRARY_PATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH \
  PYTHONPATH="$(cd "$SCRIPT_DIR/../../.." && pwd)" PYTHONNOUSERSITE=1 \
  HF_HOME="${HF_HOME:-/tmp/vla-gamepad-hf-cache}" CUDA_VISIBLE_DEVICES='' \
  "$MODEL_PYTHON" -m vla.datacollection.start_teleop.gamepad_record "$@"
