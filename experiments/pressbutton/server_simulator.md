# Isolated server simulator for the PressButton benchmark

Use a separate simulator environment alongside the OpenPI training/serving
environment. The measured simulator stack is Python 3.11.15, Isaac Sim 5.1.0.0,
Isaac Lab 0.54.4, PyTorch 2.7.0+cu128, NumPy 1.26.0, and JAX 0.4.28 on CPU
for inverse kinematics. The policy server keeps its own JAX/CUDA dependencies.

## Prepare the environment

An existing compatible simulator environment can be copied to avoid downloading
Isaac Sim and PyTorch again. Review the donor path and available storage first.
`--copy` gives the new environment independent files. The donor environment is
not an installation target. Conda must have access to the donor's cached Conda
packages or their download URLs; an empty offline cache cannot finish a clone.

```bash
SIM_PREFIX="$HOME/.conda/envs/ambench_sim"
DONOR_PREFIX="$HOME/.conda/envs/agile_zzy"
LAB_SOURCE="$HOME/.conda/envs/ambench_sim_sources/IsaacLab"
conda create --prefix "$SIM_PREFIX" --clone "$DONOR_PREFIX" --copy -y
git clone --no-checkout --filter=blob:none https://github.com/isaac-sim/IsaacLab.git "$LAB_SOURCE"
git -C "$LAB_SOURCE" checkout b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SIM_PREFIX"
```

From the AM-Bench repository root, initialize the pinned controller dependencies
and install the matching runtime additions. The requirements file records the
evaluation subset used in this experiment; it is not a complete independent
Isaac Sim distribution or a lock for every optional LeRobot hardware package.

```bash
git submodule update --init --recursive ext/pyroki ext/acados
uv pip install --python "$CONDA_PREFIX/bin/python" --no-deps \
  -r experiments/pressbutton/simulator-requirements.txt
uv pip install --python "$CONDA_PREFIX/bin/python" hatchling==1.27.0 editables==0.5
uv pip install --python "$CONDA_PREFIX/bin/python" --no-deps --no-build-isolation \
  -e "$LAB_SOURCE/source/isaaclab" -e "$LAB_SOURCE/source/isaaclab_assets" \
  -e "$LAB_SOURCE/source/isaaclab_tasks" -e "$LAB_SOURCE/source/isaaclab_rl" \
  -e "$LAB_SOURCE/source/isaaclab_mimic" -e "$LAB_SOURCE/source/isaaclab_contrib" \
  -e ext/pyroki -e ext/acados/interfaces/acados_template \
  -e source/ambench -e source/ambench_learn -e ext/openpi/packages/openpi-client
cmake -S ext/acados -B ext/acados/build -DACADOS_WITH_QPOASES=ON
cmake --build ext/acados/build --target install --parallel 4
JAX_PLATFORMS=cpu python -c 'import torch, jax, pyroki, acados_template, av; from lerobot.datasets.lerobot_dataset import LeRobotDataset'
```

If creating a simulator environment from scratch, follow the
[Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/setup/installation/pip_installation.html)
and use the versions above before installing these additions.

## Headless rendering on the tested server

This server needs a private Xvfb display for Isaac Sim's OpenGL interoperability
startup. Its Mesa context advertises OpenGL 4.5 by default; the launch requires
the process-local overrides below. Camera rendering still uses the NVIDIA
Vulkan/RTX device. Confirm the active GPU in the simulator log and with
`nvidia-smi`. These settings do not change system drivers or another display.
The initial failure and remedy are consistent with the
[reported headless GLX issue](https://github.com/isaac-sim/IsaacLab/issues/2573).

The shared `/tmp/isaaclab/logs` directory was not writable by this user, so each
user needs a private temporary directory. Keep its logs for diagnosis.

```bash
# Simulator environment already activated, policy served on localhost port 8019.
nvidia-smi
export ISAACLAB_PATH="$LAB_SOURCE"
TORCH_GOMP=("$CONDA_PREFIX"/lib/python3.11/site-packages/torch/lib/libgomp*.so.1)
export LD_PRELOAD="${TORCH_GOMP[0]}"
export LC_ALL=C.UTF-8 OMNI_KIT_ACCEPT_EULA=YES
export MESA_GL_VERSION_OVERRIDE=4.6 MESA_GLSL_VERSION_OVERRIDE=460
export TMPDIR="/tmp/ambench-sim-$(id -u)"
mkdir -p -m 700 "$TMPDIR"
test -O "$TMPDIR"
xvfb-run -a -s '-screen 0 1280x1024x24' \
  bash experiments/pressbutton/pi_benchmark.sh outputs/openpi/benchmark/server_warmup 8019
```

The benchmark recipe selects simulator GPU 0. Serve the unchanged checkpoint on
a different idle GPU, using the existing OpenPI environment. Repeat the command
with fresh output directories for timed runs. Download the completed videos and
reports after evaluation; model weights can remain on the server.

Generated setup logs and the measured local/server comparison live under
`outputs/openpi/server_sim_setup/` and
`outputs/openpi/runs/20261107_server_local_comparison/`. They remain untracked.
