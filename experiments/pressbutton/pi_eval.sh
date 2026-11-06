#!/usr/bin/env bash
set -eo pipefail

# Run from the repository root with the simulator environment already activated.
PROJECT_ROOT="$(pwd)"
RUN_DIR="${1:?Pass an output directory}"
PORT="${2:?Pass the localhost policy server port}"
mkdir -p "$RUN_DIR"
export PYTHONPATH="$PROJECT_ROOT/ext/openpi/packages/openpi-client/src:$PROJECT_ROOT/source/ambench_learn"
export MPLCONFIGDIR="$RUN_DIR/cache/matplotlib"
export JAX_PLATFORMS=cpu
export ACADOS_SOURCE_DIR="$PROJECT_ROOT/ext/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$ACADOS_SOURCE_DIR/build:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
for BATCH_INDEX in 0 1 2 3 4; do
  ROLLOUT_SEED=$((10000 + 4 * BATCH_INDEX))
  python -u -m ambench_learn.policies.pi.eval \
    --task PressButton-Am-FAHexa-Abs-PID-Direct-v0 \
    --host 127.0.0.1 --port "$PORT" --prompt 'press the button' \
    --policy-id pi0_lora_pressbutton_10000 \
    --num-envs 4 --num-rollouts 4 --seed 10000 --rollout-seed-base "$ROLLOUT_SEED" \
    --policy-target-hz 30 --n-action-steps 8 \
    --scene-video-size 640 360 --record-inference-timing --save-video --video-camera-names ee_camera scene_camera \
    --progress-every 600 --output-dir "$RUN_DIR/eval/batch_$BATCH_INDEX" \
    --headless --device cuda:0 > "$RUN_DIR/eval_batch_$BATCH_INDEX.log" 2>&1
done
