#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${VLA_PYTHON:-${CONDA_PREFIX:-$HOME/miniconda3/envs/vla}/bin/python}
split_root=${SPLIT_ROOT:-$repo_root/datasets/sft/franka_single_left_split}
split_manifest="$split_root/split_manifest.json"
log_root="$repo_root/outputs/franka_pi05_sft_screen_logs"

export HF_HOME=${HF_HOME:-$repo_root/cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo_root/src"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [[ ! -x $python_bin ]]; then
  echo "Python environment is unavailable: $python_bin" >&2
  exit 1
fi
if [[ ! -s $split_manifest ]]; then
  echo "Split manifest is unavailable: $split_manifest" >&2
  exit 1
fi

read -r train_frames steps_per_epoch screen_steps warmup_steps < <(
  "$python_bin" - "$split_manifest" <<'PY'
import json
import math
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
train_frames = int(manifest["splits"]["train"]["total_frames"])
steps_per_epoch = math.ceil(train_frames / 2)
screen_steps = math.ceil(steps_per_epoch / 2)
warmup_steps = math.ceil(steps_per_epoch * 0.05)
print(train_frames, steps_per_epoch, screen_steps, warmup_steps)
PY
)
printf -v checkpoint_step '%06d' "$screen_steps"

screen_a_output="$repo_root/outputs/franka_pi05_sft_screen_a"
screen_b_output="$repo_root/outputs/franka_pi05_sft_screen_b"
screen_c_output="$repo_root/outputs/franka_pi05_sft_screen_c"
for output_dir in "$screen_a_output" "$screen_b_output" "$screen_c_output"; do
  if [[ -e $output_dir ]]; then
    echo "Refusing to overwrite an existing screen output: $output_dir" >&2
    exit 1
  fi
done

tokenizer_cache="$HF_HOME/hub/models--google--paligemma-3b-pt-224"
if [[ ! -s $tokenizer_cache/refs/main ]]; then
  echo "PaliGemma tokenizer cache mapping is unavailable under $HF_HOME" >&2
  exit 1
fi
"$python_bin" -c \
  "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224')"

mkdir -p "$log_root"

mild_tfs='{"brightness":{"weight":1,"type":"ColorJitter","kwargs":{"brightness":[0.9,1.1]}},"contrast":{"weight":1,"type":"ColorJitter","kwargs":{"contrast":[0.9,1.1]}},"saturation":{"weight":1,"type":"ColorJitter","kwargs":{"saturation":[0.9,1.1]}},"hue":{"weight":0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1,"type":"SharpnessJitter","kwargs":{"sharpness":[0.8,1.2]}},"affine":{"weight":0,"type":"RandomAffine","kwargs":{"degrees":[-5,5],"translate":[0.05,0.05]}}}'

common_args=(
  --split-manifest="$split_manifest"
  --dataset.repo_id=local/franka_single_left_sft_10d_absolute_task3_train
  --dataset.root="$split_root/train"
  --policy.path="$repo_root/models/pi05_base"
  --policy.device=cuda
  --policy.dtype=bfloat16
  --policy.use_relative_actions=false
  --policy.push_to_hub=false
  --policy.scheduler_warmup_steps="$warmup_steps"
  --policy.scheduler_decay_steps="$screen_steps"
  --batch_size=2
  --num_workers=4
  --seed=1000
  --steps="$screen_steps"
  --save_freq="$screen_steps"
  --env_eval_freq=0
  --dataset.eval_split=0
  --tolerance_s=0.04
  --wandb.enable=true
  --wandb.project=vla-franka
  --wandb.disable_artifact=true
)

augmentation_args=(
  --dataset.image_transforms.enable=true
  --dataset.image_transforms.max_num_transforms=1
  --dataset.image_transforms.tfs="$mild_tfs"
)

validate_screen() {
  local name=$1
  local output_dir=$2
  local checkpoint="$output_dir/checkpoints/$checkpoint_step/pretrained_model"
  local required_outputs=(
    "$output_dir/sft_run_provenance.json"
    "$checkpoint/config.json"
    "$checkpoint/model.safetensors"
    "$checkpoint/policy_preprocessor.json"
    "$checkpoint/policy_postprocessor.json"
    "$checkpoint/train_config.json"
  )
  for path in "${required_outputs[@]}"; do
    if [[ ! -s $path ]]; then
      echo "Screen $name completed without required output: $path" >&2
      return 1
    fi
  done
  echo "$(date --iso-8601=seconds) screen $name output validated: $checkpoint"
}

run_screen() {
  local name=$1
  local output_dir=$2
  shift 2
  local job_name="franka_pi05_sft_screen_${name,,}"
  local log_path="$log_root/screen_${name,,}.log"

  echo "$(date --iso-8601=seconds) starting screen $name on CUDA device $CUDA_VISIBLE_DEVICES"
  if ! "$python_bin" -m vla.cli.train_franka_pi05_sft \
    "${common_args[@]}" \
    "$@" \
    --output_dir="$output_dir" \
    --job_name="$job_name" 2>&1 | tee "$log_path"; then
    echo "Screen $name failed; subsequent screens will not start. Log: $log_path" >&2
    return 1
  fi
  validate_screen "$name" "$output_dir"
}

echo "SFT screens: train_frames=$train_frames steps_per_epoch=$steps_per_epoch"
echo "SFT screens: steps=$screen_steps warmup=$warmup_steps checkpoint=$checkpoint_step"

cd "$repo_root"
run_screen A "$screen_a_output" \
  --dataset.image_transforms.enable=false \
  --policy.freeze_vision_encoder=false
run_screen B "$screen_b_output" \
  "${augmentation_args[@]}" \
  --policy.freeze_vision_encoder=false
run_screen C "$screen_c_output" \
  "${augmentation_args[@]}" \
  --policy.freeze_vision_encoder=true

echo "$(date --iso-8601=seconds) screens A, B, and C completed successfully"
