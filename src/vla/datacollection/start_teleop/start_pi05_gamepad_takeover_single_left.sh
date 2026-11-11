#!/usr/bin/env bash
# Compatibility entry point; the shared ROS backend owns all Joy/peripheral processes.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VLA_ROOT="${VLA_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
export VLA_PYTHON="${VLA_PYTHON:-${MODEL_PYTHON:-${CONDA_PREFIX:-$HOME/miniconda3/envs/vla}/bin/python}}"
source "$VLA_ROOT/scripts/model_env.sh"
CONFIG="$SCRIPT_DIR/../franka_robotiq_rollout/configs/single_left_sft_takeover.yaml"
exec "$VLA_PYTHON" -u -m vla.cli.ros_session \
  --workflow deploy --source gamepad --policy-kind sft --config "$CONFIG" \
  --dataset-root "${DATASET_ROOT:-$VLA_ROOT/datasets/raw/gamepad_takeover_live}" "$@"
