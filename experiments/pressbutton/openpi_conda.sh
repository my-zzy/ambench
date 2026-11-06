#!/usr/bin/env bash
set -euo pipefail

# Explicit server settings. Change these for another dedicated environment/GPU allocation.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_PREFIX="$HOME/.conda/envs/ambench_openpi"
WORK_DIR="$PROJECT_ROOT/outputs/openpi"
GPUS="0,1"
MEMORY_FRACTION=0.85

cd "$PROJECT_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export OPENPI_DATA_HOME="$WORK_DIR/cache/openpi"
export HF_HOME="$WORK_DIR/cache/huggingface"
export HF_LEROBOT_HOME="$WORK_DIR/datasets"
export XDG_CACHE_HOME="$WORK_DIR/cache"
export JAX_COMPILATION_CACHE_DIR="$WORK_DIR/cache/jax"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEMORY_FRACTION"
export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy=
export PYTHONPATH="$PROJECT_ROOT/source/ambench_learn:$PROJECT_ROOT/ext/openpi/src:$PROJECT_ROOT/ext/openpi/packages/openpi-client/src"
unset JAX_PLATFORMS LD_LIBRARY_PATH CUDA_HOME CUDA_PATH

exec python -m "ambench_learn.policies.pi.${1:?Pass train, infer, serve, compute_norm_stats, or runtime}" "${@:2}"
