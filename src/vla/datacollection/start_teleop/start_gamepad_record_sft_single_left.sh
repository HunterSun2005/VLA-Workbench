#!/usr/bin/env bash
set -Eeuo pipefail

# New single-left SFT collection run. This wrapper keeps the established
# gamepad_record implementation and only supplies a fresh dataset location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="${VLA_ROOT:-$PWD}"
MODEL_PYTHON="${VLA_PYTHON:-${CONDA_PREFIX:-$HOME/miniconda3/envs/vla}/bin/python}"
DATASET_ROOT="${SFT_DATASET_ROOT:-$REPOSITORY_ROOT/datasets/raw/sft_$(date +%Y%m%d_%H%M%S)_single_left_raw}"
TASK="${SFT_TASK:-}"

[[ -x "$MODEL_PYTHON" ]] || {
  printf 'Missing Python: %s\n' "$MODEL_PYTHON" >&2
  exit 1
}
[[ -n "$TASK" ]] || {
  printf 'Set SFT_TASK to the complete task prompt before starting collection.\n' >&2
  printf 'Example: SFT_TASK="Place the white lid firmly onto the jig." %s --execute\n' \
    "$0" >&2
  exit 2
}

exec env -u PYTHONHOME -u LD_LIBRARY_PATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH \
  PYTHONPATH="$(cd "$SCRIPT_DIR/../../.." && pwd)" PYTHONNOUSERSITE=1 \
  HF_HOME="${HF_HOME:-/tmp/vla-gamepad-hf-cache}" CUDA_VISIBLE_DEVICES='' \
  "$MODEL_PYTHON" -m vla.datacollection.start_teleop.gamepad_record \
  --dataset-root "$DATASET_ROOT" --task "$TASK" "$@"
