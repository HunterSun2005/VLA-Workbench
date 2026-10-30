#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/model_env.sh"
cd "$VLA_ROOT"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
export PYTHONPATH="$VLA_ROOT/src:$VLA_ROOT"
exec "$VLA_PYTHON" -m pytest "$@"
