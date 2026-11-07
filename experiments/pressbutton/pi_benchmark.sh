#!/usr/bin/env bash
set -eo pipefail

# Activate the simulator environment and run from the repository root.
# Run once to warm caches, then repeat into distinct output directories.
PROJECT_ROOT="$(pwd)"
OUTPUT_DIR="${1:?Pass a new batch output directory}"
PORT="${2:?Pass the localhost policy server port}"
SEED=30000
NUM_ENVS=4
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
export PYTHONPATH="$PROJECT_ROOT/ext/openpi/packages/openpi-client/src:$PROJECT_ROOT/source/ambench_learn:$PROJECT_ROOT/source/ambench:$PROJECT_ROOT/outputs/act_runtime/python_deps"
export ACADOS_SOURCE_DIR="$PROJECT_ROOT/ext/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$ACADOS_SOURCE_DIR/build:$LD_LIBRARY_PATH"
export MPLCONFIGDIR="$OUTPUT_DIR/cache/matplotlib"
export HF_HOME="$PROJECT_ROOT/outputs/openpi/cache/huggingface" HF_HUB_OFFLINE=1
export HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy=
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1

test ! -e "$OUTPUT_DIR"
exec python -u -m ambench_learn.policies.pi.eval \
  --task PressButton-Am-FAHexa-Abs-PID-Direct-v0 \
  --host 127.0.0.1 --port "$PORT" --prompt 'press the button' \
  --policy-id pi0_full_pressbutton_corrected_20000_batch32 \
  --num-envs "$NUM_ENVS" --num-rollouts "$NUM_ENVS" --seed "$SEED" --rollout-seed-base "$SEED" \
  --policy-target-hz 30 --n-action-steps 8 \
  --scene-video-size 640 360 --record-inference-timing --record-runtime-timing \
  --save-video --video-camera-names ee_camera scene_camera \
  --progress-every 600 --output-dir "$OUTPUT_DIR" \
  --headless --device cuda:0
