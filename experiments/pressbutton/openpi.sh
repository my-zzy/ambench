#!/usr/bin/env bash
set -euo pipefail

# Edit these explicit settings for another workstation. No shared env is modified.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$PROJECT_ROOT/outputs/openpi"
GPU=0
MEMORY_FRACTION=0.85

cd "$PROJECT_ROOT"
source "$WORK_DIR/.venv/bin/activate"
export OPENPI_DATA_HOME="$WORK_DIR/cache/openpi"
export HF_HOME="$WORK_DIR/cache/huggingface"
export HF_LEROBOT_HOME="$WORK_DIR/datasets"
export XDG_CACHE_HOME="$WORK_DIR/cache"
export JAX_COMPILATION_CACHE_DIR="$WORK_DIR/cache/jax"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEMORY_FRACTION"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONPATH="$PROJECT_ROOT/source/ambench_learn:$PROJECT_ROOT/ext/openpi/src:$PROJECT_ROOT/ext/openpi/packages/openpi-client/src"
unset JAX_PLATFORMS LD_LIBRARY_PATH CUDA_HOME CUDA_PATH

exec python -m "ambench_learn.policies.pi.${1:?Pass train, infer, serve, compute_norm_stats, or runtime}" "${@:2}"
