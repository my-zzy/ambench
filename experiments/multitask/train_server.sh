#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_PREFIX="$HOME/.conda/envs/ambench_openpi"
WORK_DIR="$PROJECT_ROOT/outputs/openpi"
DATASET="my-zzy/ambench_contact3_30hz_v1"
EXP_NAME="contact3_full_20k_v1"
GPUS="0,1"
STEPS=20000
BATCH_SIZE=32

cd "$PROJECT_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export CUDA_VISIBLE_DEVICES="$GPUS"
export OPENPI_DATA_HOME="$WORK_DIR/cache/openpi" HF_HOME="$WORK_DIR/cache/huggingface"
export HF_LEROBOT_HOME="$WORK_DIR/datasets" XDG_CACHE_HOME="$WORK_DIR/cache"
export JAX_COMPILATION_CACHE_DIR="$WORK_DIR/cache/jax"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export PYTHONPATH="$PROJECT_ROOT/source/ambench_learn:$PROJECT_ROOT/ext/openpi/src:$PROJECT_ROOT/ext/openpi/packages/openpi-client/src"
export HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy=
unset JAX_PLATFORMS LD_LIBRARY_PATH CUDA_HOME CUDA_PATH
exec python -m ambench_learn.policies.pi.train \
  --variant multitask --finetune-mode full --repo-id "$DATASET" --work-dir "$WORK_DIR" \
  --pretrained-checkpoint "$WORK_DIR/pretrained/pi0_base" \
  --exp-name "$EXP_NAME" --steps "$STEPS" --batch-size "$BATCH_SIZE" --fsdp-devices 2 --num-workers 8 \
  --run-dir "$WORK_DIR/multitask/runs/$EXP_NAME"
