#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_PREFIX="$HOME/.conda/envs/ambench_sim"
GPU="${1:?Pass one idle simulator GPU index}"
shift
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU" JAX_PLATFORMS=cpu
export ISAACLAB_PATH="$HOME/.conda/envs/ambench_sim_sources/IsaacLab"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/source/ambench:$PROJECT_ROOT/source/ambench_learn"
export HF_HOME="$PROJECT_ROOT/outputs/openpi/cache/huggingface" HF_HUB_OFFLINE=1
export ACADOS_SOURCE_DIR="$PROJECT_ROOT/ext/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$ACADOS_SOURCE_DIR/build:$LD_LIBRARY_PATH"
TORCH_GOMP=("$CONDA_PREFIX"/lib/python3.11/site-packages/torch/lib/libgomp*.so.1)
export LD_PRELOAD="${TORCH_GOMP[0]}"
export LC_ALL=C.UTF-8 OMNI_KIT_ACCEPT_EULA=YES
export MESA_GL_VERSION_OVERRIDE=4.6 MESA_GLSL_VERSION_OVERRIDE=460
export TMPDIR="/tmp/ambench-sim-$(id -u)"
mkdir -p -m 700 "$TMPDIR"
test -O "$TMPDIR"
export HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy=
exec xvfb-run -a -s '-screen 0 1280x1024x24' \
  python -u scripts/data/collect_multitask_demos.py "$@"
