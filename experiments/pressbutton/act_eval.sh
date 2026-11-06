#!/usr/bin/env bash
set -euo pipefail
# Run from the AM-Bench repository root after: conda activate ambench
CHECKPOINT="${1:?Pass an ACT checkpoint directory}"
RUN_DIR="${2:?Pass an output directory}"
export PYTHONPATH="$PWD/outputs/act_runtime/python_deps:$PWD/source/ambench_learn:$PWD/source/ambench"
export HF_HOME="$RUN_DIR/cache/huggingface"
export HF_HUB_OFFLINE=1
export TORCH_HOME="$RUN_DIR/cache/torch"
export MPLCONFIGDIR="$RUN_DIR/cache/matplotlib"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
for BATCH_INDEX in 0 1 2 3 4; do
  ROLLOUT_SEED=$((10000 + 4 * BATCH_INDEX))
  python -u -m ambench_learn.policies.act.eval \
    --task PressButton-Am-FAHexa-Abs-PID-Direct-v0 \
    --checkpoint "$CHECKPOINT" \
    --num-envs 4 --num-rollouts 4 --seed 10000 --rollout-seed-base "$ROLLOUT_SEED" \
    --policy-target-hz 30 --n-action-steps 8 \
    --scene-video-size 640 360 --save-video --video-camera-names ee_camera scene_camera \
    --progress-every 600 --output-dir "$RUN_DIR/eval/batch_$BATCH_INDEX" \
    --headless --device cuda:0
done
