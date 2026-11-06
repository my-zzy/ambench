# Reconstructed local experiment milestones

These recipes recover the reusable code developed in the ignored local runs.
Commits are being created now, in milestone order; they do not backdate the work
or rewrite the upstream history. Original runs remain unchanged under `outputs/`.
Generated datasets, checkpoints, videos, logs, and environment installations stay
ignored. The commands below must be run from the repository root.

## 1. Scripted expert recordings — November 1, 2026

Sources: `outputs/expert_demos/20261101_144142/run_suite.py` and `run_task.py`.
The tracked versions take an explicit output directory, run one simulator process
at a time, preserve completed tasks, and stop each task after its first success
or three native-length episodes. They save videos, numeric trajectories, episode
metrics, a run manifest, and a Markdown summary. Incomplete output is never
silently overwritten. The optional horizon override is a supplemental experiment,
not a replacement for the native benchmark result.

Activate your existing Isaac Lab environment first, then inspect GPU usage:

```bash
source ../IsaacLab/env_isaaclab/bin/activate
nvidia-smi
python scripts/data/run_expert_suite.py --output outputs/expert_demos/reproduction
# Restrict the suite when checking the recorder:
python scripts/data/run_expert_suite.py --output outputs/expert_demos/pressbutton_check --tasks PressButton
```

The recorded task registry, expert, controller, and termination logic remain the
repository implementations. Publication PDF builders are presentation tooling and
are not dependencies of data collection, learning, or evaluation.

Reconstruction validation: Python syntax and suite CLI checks. The simulator
suite has not been rerun as part of this code migration.

## 2. PressButton ACT baseline — November 2, 2026

Sources: `outputs/act_pressbutton/20261102_1603/`. Collection adds explicit seed
and camera-rate arguments to the existing recorder. The ACT evaluator now allows
separate scene-startup and reset seeds, and an external-video resolution override.
Defaults retain existing behavior. Wrist-camera observations remain 384 × 384.

The historical environment used Python 3.11, Isaac Sim 5.1.0, Isaac Lab 0.54.4,
PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128, LeRobot 0.4.4, transformers 4.57.6,
datasets 4.0.0, and PyAV 15.1.0. Use the standard installer and ACT dependencies
in your isolated simulator environment; no `python_deps` folder from a run is
required by these recipes. Recreate the historical extra packages with the
tracked installer below; it places `diffusers==0.35.2` and `pyserial==3.5` under
`outputs/act_runtime/python_deps` without changing the activated environment.
The ACT launchers select that reproducible overlay. These versions describe the measured environment,
not a request to replace an existing shared environment.

```bash
bash scripts/setup/install_act_runtime.sh
export PYTHONPATH="$PWD/outputs/act_runtime/python_deps:$PWD/source/ambench_learn:$PWD/source/ambench"
python scripts/data/record_demos_scripted.py \
  --task PressButton-Am-FAHexa-Abs-PID-Direct-v0 --num_envs 4 \
  --num_demos 100 --seed 100 --camera_fps 30 --env_length_s 20 \
  --dataset_root outputs/pressbutton/demos \
  --state_keys ee_pos ee_quat gripper_width --task_prompt 'press the button' \
  --repo_id my-zzy/pressbutton_act_train --headless --device cuda:0
# Use the finalized .../demo-.../lerobot path printed by collection:
bash experiments/pressbutton/act_train.sh <canonical-dataset> outputs/pressbutton/act
bash experiments/pressbutton/act_eval.sh outputs/pressbutton/act/train/checkpoints/020000 outputs/pressbutton/act
python -m ambench_learn.eval.summarize --input-dir outputs/pressbutton/act/eval \
  --output-dir outputs/pressbutton/act/report
python -m ambench_learn.eval.validate_videos --input-dir outputs/pressbutton/act \
  --output-dir outputs/pressbutton/act/report --policy act
```

`scripts/environments/evaluate_scripted.py` provides the matching expert control.
Run five separate processes with four rollouts each, startup seed 10000 and
`--rollout-seed-base` 10000, 10004, 10008, 10012, and 10016. Use a distinct
`--output-dir` for each batch. Training is 20,000 updates × batch 32, horizon 30,
eight actions executed at 30 Hz with the native 120 Hz controller, seed 42.

Historical result: ACT 20/20 and scripted expert 20/20. This is a single-task,
fixed-instruction development baseline; it does not establish language
generalization. Raw evidence remains in the original ignored run. Reconstruction
checks aggregate that saved evidence; training and simulation are not rerun.

## 3. Isolated OpenPI setup — November 2–4, 2026

Sources: `outputs/openpi/{install.sh,run.py,run.sh,local_config.py}` and the
checkpoint downloader in `setup-20261102_2119/`. The implementation is now in
`ambench_learn.policies.pi`; it has no imports from ignored run directories.
OpenPI remains at the repository's pinned submodule commit and frozen `uv.lock`.
Its JAX environment is separate from the simulator/ACT environment.

```bash
git submodule update --init --recursive ext/openpi
bash scripts/setup/install_openpi.sh outputs/openpi
python scripts/setup/download_pi0_checkpoint.py --destination outputs/openpi/pretrained/pi0_base
# Check an already transferred checkpoint without downloading:
python scripts/setup/download_pi0_checkpoint.py --destination outputs/openpi/pretrained/pi0_base --verify-only
```

`pi0_base_manifest.json` pins the official GCS object generations, sizes, and MD5
checksums used locally. It is a dependency manifest, not model weights. Transfers
resume in a sibling `.partial` directory, and only verified files are promoted.
Use an ignored output path for weights; the base checkpoint is approximately
12 GB. Installation and downloading are explicit commands, never import effects.

Export the canonical dataset in the simulator environment using the tracked
`scripts/data/export_lerobot_to_openpi.py` with `--target_hz 30`,
`--omit_base_image`, `--task_prompt 'press the button'`, and
`--repo_id my-zzy/pressbutton_30hz`; place it under
`outputs/openpi/datasets/my-zzy/pressbutton_30hz`. This is the historical floor
resampling path; milestone 6 provides the corrected terminal-preserving path.

```bash
nvidia-smi
bash experiments/pressbutton/openpi.sh runtime doctor
# Historical normalization; milestone 6 replaces this for new corrected runs:
bash experiments/pressbutton/openpi.sh compute_norm_stats
bash experiments/pressbutton/openpi.sh infer
bash experiments/pressbutton/openpi.sh runtime smoke --exp-name setup_smoke
bash experiments/pressbutton/openpi.sh infer --checkpoint <saved-smoke-checkpoint>
bash experiments/pressbutton/openpi.sh serve --checkpoint <saved-smoke-checkpoint>
# In a second terminal, exercise the localhost WebSocket contract:
bash experiments/pressbutton/openpi.sh runtime client
```

The wrapper has a small explicit machine configuration block. Its modules also
accept `--work-dir`, `--repo-id`, and `--pretrained-checkpoint`. With an activated
OpenPI environment, the training entrypoint is
`python -m ambench_learn.policies.pi.train`. Cache redirection generates a runtime
copy of the pinned upstream trainer in ignored storage and asserts the expected
upstream cache expression; it does not edit `ext/openpi`. Existing normalization
files are never overwritten by the setup command. Serving binds to localhost
and filters observations to the wrist-only schema used for training.

Historical validation covered GPU inference, two LoRA updates, checkpoint reload,
and the WebSocket roundtrip. It did not establish task success. The upstream
LoRA filter also trains vision/projection parameters. For reconstruction, config
parity and CPU checks are performed; GPU setup is not repeated.

## 4. First π₀ training and evaluation — November 4, 2026

Sources: `outputs/openpi/runs/20261104_1315/`. Train with an optional separate
run directory to capture the upstream log, CSV metrics, and atomic progress JSON:

```bash
bash experiments/pressbutton/openpi.sh train --exp-name pressbutton_pilot_reproduction \
  --steps 10000 --batch-size 1 --run-dir outputs/openpi/pilot_reproduction
# After training exits and finishes saving:
bash experiments/pressbutton/openpi.sh infer --checkpoint <checkpoint-directory>
bash experiments/pressbutton/openpi.sh serve --checkpoint <checkpoint-directory> --port 8000
# In a simulator-environment terminal, with sufficient free GPU memory:
bash experiments/pressbutton/pi_eval.sh outputs/openpi/pilot_reproduction 8000
python -m ambench_learn.eval.summarize --input-dir outputs/openpi/pilot_reproduction/eval \
  --output-dir outputs/openpi/pilot_reproduction/report
python -m ambench_learn.eval.validate_videos --input-dir outputs/openpi/pilot_reproduction \
  --output-dir outputs/openpi/pilot_reproduction/report --policy pi
```

The PI evaluator now exposes the same separate startup/reset seeds and external
video dimensions as ACT. Optional inference timing records client roundtrip and
server inference time. The supervisor stops only its own child process group,
records failed/interrupted status, and refuses an existing log directory. A
checkpoint is considered complete only after the trainer confirms saving.
`--run-dir` is a supervisor argument; the regular trainer's `--help` lists model
and artifact arguments. No supervisor automatically starts another training run.

Historical result: **1/20**, after 10,000 updates × batch 1, approximately 24.46
minutes of training. The 20-case schedule matches ACT, but ACT received 64 times
as many sample presentations. Latency measurements in synchronous simulation do
not establish real-time flight operation. Reconstructed aggregation reproduces
the saved 1/20 result; no new learned-policy rollout was run.

## 5. Failure diagnostics — November 5, 2026, before correction

Sources: `outputs/openpi/diagnostics/20261105_1321/`. Diagnostics are separate
from optimizer updates and never replace checkpoint statistics. They cover the
public encode/normalize/pad/unnormalize/decode chain, the original exporter tails,
frozen-model errors on expert observations, and paired simulator replay controls.

```bash
# CPU, OpenPI environment:
bash experiments/pressbutton/openpi.sh audit --checkpoint <pilot-checkpoint> \
  --output-dir outputs/openpi/diagnostic_reproduction
python scripts/data/audit_openpi_tails.py --source <canonical-dataset> \
  --export outputs/openpi/datasets/my-zzy/pressbutton_30hz \
  --output outputs/openpi/diagnostic_reproduction/training_terminal_audit.json
# GPU, simulator environment; two seeds × four fresh expert episodes:
python scripts/data/diagnose_pressbutton_replay.py --mode collect \
  --output-dir outputs/openpi/diagnostic_reproduction --headless --device cuda:0
# CPU, OpenPI environment; creates heldout_export and converted replay records:
bash experiments/pressbutton/openpi.sh audit --checkpoint <pilot-checkpoint> \
  --heldout-dir outputs/openpi/diagnostic_reproduction/heldout \
  --output-dir outputs/openpi/diagnostic_reproduction
# GPU, simulator environment; default replays raw and adapter-roundtrip commands:
python scripts/data/diagnose_pressbutton_replay.py --mode replay \
  --output-dir outputs/openpi/diagnostic_reproduction --headless --device cuda:0
# Repeat with --replay-variant native_120hz, raw_hold_30hz, and terminal_preserved_30hz.
# GPU, OpenPI environment; no simulator or optimizer updates:
bash experiments/pressbutton/openpi.sh prediction_errors --checkpoint <pilot-checkpoint> \
  --dataset-root outputs/openpi/datasets/my-zzy/pressbutton_30hz \
  --reference-dir outputs/openpi/diagnostic_reproduction \
  --output-dir outputs/openpi/diagnostic_reproduction/predictions
```

The fixed prediction protocol uses eight training and eight fresh episodes,
12 anchors per episode, and three draws per anchor. Errors on overlapping expert
chunks are not independent task trials and do not evaluate recovery from the
policy's own mistakes. Replay checks initial robot, button, and wall state.
Success flags are authoritative; post-step button displacement may miss a peak
because automatic reset has already occurred.

Historical controls: native expert/replay 8/8, floor-resampled raw/roundtrip 1/8,
four-tick command hold 4/8, and appended terminal command 8/8. All 100 training
episodes lost their final command: median final-target difference 8.2 mm. This is
EE-target distance, not a task tolerance. Adapter roundtrip was accurate to
0.248 micrometres, but saved action `qw` standard deviation was zero versus a
float64 recomputation of 8.29494e-6. All sampled gripper sign errors occurred at
frame zero. These observations motivated the next milestone; they do not prove
that either data issue explains the entire learned-policy failure.

Reconstruction validation repeats the CPU training-tail and full adapter audits
against the original data. GPU prediction and paired replay are not rerun.

## 6. Terminal-preserving export and stable statistics — November 5, 2026

Sources: `outputs/openpi/runs/20261105_1505_corrected/`. The correction remains
opt-in so the original ACT and π₀ pilot can still be reproduced. It does not
change global resampling behavior or overwrite old datasets/statistics.

```bash
# OpenPI environment, CPU. The new dataset directory must not exist:
python scripts/data/repair_openpi_terminals.py --source <canonical-dataset> \
  --original outputs/openpi/datasets/my-zzy/pressbutton_30hz \
  --output outputs/openpi/datasets/my-zzy/pressbutton_30hz_terminal_v2
bash experiments/pressbutton/openpi.sh compute_norm_stats --variant corrected \
  --artifact-dir outputs/openpi/corrected_reproduction
# The following commands use the corrected dataset and its own asset namespace:
bash experiments/pressbutton/openpi.sh train --variant corrected \
  --exp-name pressbutton_corrected_reproduction --steps 10000 --batch-size 1 \
  --run-dir outputs/openpi/corrected_reproduction/training
bash experiments/pressbutton/openpi.sh infer --variant corrected --checkpoint <corrected-checkpoint>
bash experiments/pressbutton/openpi.sh serve --variant corrected --checkpoint <corrected-checkpoint> --port 8001
# Simulator environment, after training exits:
bash experiments/pressbutton/pi_eval.sh outputs/openpi/corrected_reproduction 8001
python scripts/data/diagnose_pressbutton_replay.py --mode replay --replay-variant corrected_30hz \
  --reference-dir outputs/openpi/diagnostic_reproduction \
  --output-dir outputs/openpi/corrected_replay --headless --device cuda:0
```

The repair intentionally supports the measured wrist-only, 8D absolute-EE,
120-to-30 Hz, one-source-episode-per-parquet protocol. It preserves the original
samples and appends the missing aligned tail plus the exact last paired
observation/image/action. The terminal pair may be held to the next 30 Hz grid
point for at most 25 ms. Other sample rates are rejected. Source data is unchanged.
The historical export grew from 25,149 to 25,294 frames across the same 100 episodes.

Statistics use float64 centered variance over every overlapping 30-action target,
including episode-end padding. Constant state channels keep zero variance.
The implementation checks actual-loader chunks and the public normalized/padded
roundtrip at episode starts, interiors, and endings. Existing unequal statistics
are rejected. No test-set observations contribute to normalization.

For a matched offline comparison, run `prediction_errors` on each checkpoint
with its corresponding `--repo-id`, but use the **same original observation
dataset** and `--reference-dir` for both. Then:

```bash
python -m ambench_learn.policies.pi.compare --baseline <pilot-run> \
  --candidate <corrected-run> --baseline-diagnostics <pilot-predictions> \
  --candidate-diagnostics <corrected-predictions> --output-dir outputs/openpi/comparison
```

The comparison checks observation/draw keys, exact target arrays, and rollout
case identities. Historical result: corrected expert replay **8/8**, corrected
learned policy **0/20**, versus the pilot's **1/20**. First-eight offline position
error improved from 1.72 to 1.33 cm on training observations and 1.74 to 1.37 cm
on fresh expert observations. Both repairs were applied together; this does not
isolate their individual effects. One changed success among 20 development
cases does not establish a statistically worse model. No full-parameter or
longer-budget training has been performed.

Reconstruction checks passed: boundary regression tests, exact equality of a
real repaired episode with the saved corrected export, all 400 loader/transform
checks across the complete corrected dataset, and byte-identical normalization:
`d8901da6594deeaf725f68061bfe1c4c60ac56a3fe49332d8f2ff51f55089ad0`.
Maximum roundtrip component error is 1.19e-7. Both historical configurations
match their recovered versions. The paired report reproduces 1/20 versus 0/20
from saved records. Training, rendering, and simulator rollout were not rerun.

## What Git carries and what must be transferred

Git carries the code, these recipes, dependency manifests, and the pinned
submodules. See [source mapping](SOURCE_MAP.md) for how ignored scripts were
recovered or replaced by maintained entrypoints. Clone this fork at the chosen
commit and initialize OpenPI's submodules on the training server.

Separately transfer the canonical demonstrations, the two exported datasets if
reproducing both pilots, their normalization assets, the verified π₀ base, and
any checkpoint/replay records needed for evaluation. Keep their paths under an
ignored work directory. OpenPI's tokenizer cache may also be transferred for
offline operation; rebuilding the environment alone does not download that
model asset. Recreate the Python environment from the pinned lock rather than
copying `.venv`, CUDA libraries, or workstation caches wholesale.

The original timestamped runs and presentation files remain in place. Their
generated reports, videos, checkpoints, and datasets are not added to Git. No
server installation, file transfer, training, or Git push is part of this history
reconstruction.

## Reconstruction portability checks — November 6, 2026

The final portability commit recovers the ACT dependency overlay from its
distribution metadata and applies the repository's import-order hooks. CLI
checks passed for ACT train/eval, PI eval, seeded demo collection, and corrected
replay, using the existing simulator environment plus the newly recreated overlay.
Required recorder/replay arguments were supplied when checking `--help`; no
Isaac application was launched. The corrected replay option is explicitly exposed
by the parser. OpenPI training/inference/serving help also works without loading a
model. The training supervisor's failed-child status and PID cleanup were checked.

Validation is **CPU functional and CLI/import level**, plus all applicable
pre-commit hooks on changed files. Python syntax, shell parsing, paired saved
results, normalization equality, and terminal regression tests passed. No new
GPU training, rendering, or simulator stepping was performed. These checks
preserve the historical task-success claims rather than treating code migration
as a fresh evaluation.

## Server full fine-tuning

The bulk artifact transfer remains paused. Complete the environment and hardware
checks independently; the synthetic smoke test below needs no dataset, tokenizer,
or pretrained checkpoint. It does not measure PressButton task performance.

Use a dedicated Python 3.11 Conda prefix under your home directory. The simulator
and ACT dependencies are not needed on a training-only host. Initialize only the
pinned OpenPI checkout, then install its locked dependencies into the new prefix:

```bash
git submodule update --init ext/openpi
mkdir -p outputs/openpi/server_setup/cache/conda
CONDA_PKGS_DIRS="$PWD/outputs/openpi/server_setup/cache/conda" \
  conda create --prefix "$HOME/.conda/envs/ambench_openpi" \
  --override-channels -c conda-forge python=3.11 pip rsync --copy -y
bash scripts/setup/install_openpi_conda.sh \
  "$HOME/.conda/envs/ambench_openpi" "$PWD/outputs/openpi"
```

Transfer these paths into the same ignored work directory, preserving their
relative paths. Verify a file-size/SHA-256 manifest after transfer:

- `datasets/my-zzy/pressbutton_30hz_terminal_v2/` — corrected 100-episode export.
- `assets/pi0_pressbutton_lora_30hz_corrected/` — validated normalization assets.
- `pretrained/pi0_base/` — verified official base checkpoint.
- `cache/openpi/big_vision/paligemma_tokenizer.model` — tokenizer.

The Conda installer exports the frozen OpenPI lock to `pylock.toml`, retaining
exact artifact URLs and SHA-256 hashes rather than resolving versions again from
package indexes. This also preserves upstream dependency overrides. It
does not replace the Conda prefix with a virtualenv or modify another environment.
The installation log and exported lock belong under `outputs/openpi/`.
The installer uses direct connections and does not change global proxy settings.
Its explicit `PYPI_PACKAGES_URL` setting selects the Aliyun file mirror for this
server workflow. Only download URLs change in `pylock.direct.toml`; versions and
artifact hashes remain identical to the exported lock. PyTorch's own URLs stay
unchanged. No global pip, uv, or Conda source configuration is modified.
The learning source is registered with an environment-local `.pth` file; the
`ambench_learn` distribution is not installed because its metadata assumes
Isaac/ACT dependencies, including a conflicting Gymnasium requirement. Use this
environment for OpenPI training/serving, and the simulator environment for rollouts.

Review the explicit GPU allocation in `openpi_conda.sh` against `nvidia-smi`
before starting a job. Check runtime and a two-step synthetic full-training run
before the artifacts arrive:

```bash
bash experiments/pressbutton/openpi_conda.sh runtime doctor
bash experiments/pressbutton/openpi_conda.sh runtime smoke --synthetic-data \
  --variant corrected --finetune-mode full --fsdp-devices 2 \
  --batch-size 32 --exp-name setup_synthetic \
  --artifact-dir outputs/openpi/server_setup/synthetic
# Repeat the same smoke command with --resume to check checkpoint restoration.
```

Synthetic checks use random weights and upstream fake inputs, force zero loader
workers, and write into a `synthetic_` checkpoint namespace. They cannot be
selected by the real training entrypoint. After the artifact transfer is complete
and verified, run a bounded check with the actual data and pretrained base:

```bash
bash experiments/pressbutton/openpi_conda.sh train \
  --variant corrected --finetune-mode full --fsdp-devices 2 \
  --batch-size 32 --num-workers 8 --steps 20 --exp-name full_smoke \
  --run-dir outputs/openpi/runs/full_smoke \
  --artifact-dir outputs/openpi/runs/full_smoke/validation
```

After checking memory use, finite training metrics, and checkpoint reload, launch
the longer experiment with a new experiment name and run directory:

```bash
bash experiments/pressbutton/openpi_conda.sh train \
  --variant corrected --finetune-mode full --fsdp-devices 2 \
  --batch-size 32 --num-workers 8 --steps 20000 --exp-name full_20k \
  --run-dir outputs/openpi/runs/full_20k \
  --artifact-dir outputs/openpi/runs/full_20k/validation
```

Full tuning uses `gemma_2b` and `gemma_300m`, freezes no model parameters, reuses
the corrected statistics, and starts from the official base. Its learning-rate
decay follows the requested training budget (minimum 101 steps for short smoke
tests); LoRA retains its historical 10,000-step schedule. EMA remains disabled.
Full checkpoints use the separate `pi0_pressbutton_full_30hz_corrected` namespace.
The proposed 20,000 × 32 budget presents 640,000 anchors, approximately 25.3
passes over 25,294 frames. Capacity and throughput must be measured on the host.

Pass `--finetune-mode full --variant corrected` when reloading a full checkpoint
with the `infer` or `serve` entrypoint. Training completion and finite inference
do not establish task success; use the paired Isaac Lab rollout protocol above.
To resume, keep the experiment name, total step budget, and model settings,
add `--resume`, and choose a new `--run-dir` for the supervisor logs.

### Server validation — November 6, 2026

The home Conda environment passed package compatibility checks. All 209
versioned dependencies in the platform-selected lock match, and mirrored
artifact hashes are unchanged. The two editable OpenPI packages use the pinned
submodule; the learning source is registered separately as described above.
Python is 3.11.18, JAX/JAXlib 0.5.3, Flax 0.10.2, Orbax 0.11.13, and PyTorch
2.9.0+cu130.

GPU matrix computation and a cross-device sum passed with all eight RTX PRO
5000 GPUs visible. The full-model synthetic smoke run completed two updates at
global batch 32 with two FSDP devices on GPUs 0 and 1. Losses were 2.3819 and
2.3421; gradients were finite. Five-second memory samples peaked at 33,651 and
33,573 MiB. Initialization, compilation, updates, and checkpoint saving took
69.3 seconds; this short run is not a steady-state throughput benchmark.
Checkpoint restore passed in a separate process without adding updates.

All three server setup regression checks passed. The two previously transferred
247-frame parquet episodes decoded correctly, with finite actions/poses and
30 Hz timestamps. No additional dataset, tokenizer, normalization, or pretrained
weight files were uploaded. These checks do not replace full-dataset loading,
pretrained-weight loading, real-data training, or Isaac Lab rollout validation.

Detailed logs, package inventories, GPU samples, and JSON reports are on the
server under `outputs/openpi/server_setup/`. All validation processes stopped.
The artifact transfer and real fine-tuning remain pending; no long experiment
will start automatically.
