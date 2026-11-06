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
required by these recipes. These versions describe the measured environment,
not a request to replace an existing shared environment.

```bash
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
