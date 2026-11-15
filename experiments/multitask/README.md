# Three-task π₀ experiment

Train one instruction-conditioned policy for **PressButton, PushSlider, and
PullLever**, using the existing FAHexa absolute-EE/PID environments and scripted
experts. The low-level controllers and benchmark success criteria are unchanged.
Commands run from the repository root. Generated data, checkpoints, videos, and
reports belong under ignored `outputs/openpi/multitask/` or other `outputs/` paths.

This first milestone tests whether one policy can perform three tasks. Because
these tasks have different scenes, good success rates alone cannot establish
that the policy uses language to choose an action. Prompt ablations below test
sensitivity; a later shared scene with competing valid instructions is needed
for a stronger language-grounding test.

## Dataset and interface

| Task | Canonical instruction | Native horizon | New collection seed | Held-out seed base |
|---|---|---:|---:|---:|
| PressButton | press the button | 20 s | 51000 | 61000 |
| PushSlider | push the slider to the right | 26 s | 52000 | 62000 |
| PullLever | pull the lever upwards | 20 s | 53000 | 63000 |

The first production mixture reuses the existing corrected **100 PressButton
episodes** collected with seed 100, and adds **100 new episodes per other task**.
Seed 51000 is the reproducible option for collecting new PressButton data; it is
not the provenance of the reused episodes. The separate 4-episode/task pilot
uses seeds 41000/42000/43000 and is excluded from production training.

All tasks share wrist RGB, 8D state (EE position, wxyz quaternion, gripper width),
and 8D absolute EE actions. OpenPI transforms actions to the current EE frame;
the policy uses 30 Hz data, horizon 30, and executes eight actions per request.
The native controller runs at 120 Hz. The collector records only successful
scripted episodes. It varies scenes through the existing seeded task reset logic.

With `ee_local_relative`, the adapter caches measured EE pose to decode actions,
but replaces the model's position and quaternion state with zero position and
identity rotation. Gripper width remains observed. Thus the model's contact-depth
correction depends on wrist RGB; it does not directly receive absolute EE pose.

We balance **episode counts**. The pinned OpenPI loader samples frames uniformly,
so longer episodes have higher probability. The merge and normalization reports
record exact task frame counts and sampling weights rather than claiming equal
task probabilities.

## 1. Collect demonstrations

Use the existing isolated simulator environment; see
[server simulator setup](../pressbutton/server_simulator.md). Inspect `nvidia-smi`
first and choose free GPUs. The server wrapper activates
`~/.conda/envs/ambench_sim`, uses a private temporary directory, and runs headless
RTX rendering through Xvfb. Each invocation owns and bounds its child process.

```bash
nvidia-smi
bash experiments/multitask/collect_server.sh 0 \
  --tasks PushSlider --num-demos 100 --num-envs 16 \
  --output outputs/openpi/multitask/production/collection_PushSlider
bash experiments/multitask/collect_server.sh 1 \
  --tasks PullLever --num-demos 100 --num-envs 16 \
  --output outputs/openpi/multitask/production/collection_PullLever
```

These commands can run concurrently on separate free GPUs. On a one-GPU
machine, run them sequentially. `--video` adds wrist-camera recordings; training
images are recorded regardless. `--plan-only` prints commands without launching
Isaac Sim. Finalized canonical dataset paths and actual episode counts are saved
in each output directory's `collection.json`. A failed or incomplete collector
must be resolved before export; do not infer success from directory existence.

## 2. Export and merge

Activate the simulator environment for the exporter (LeRobot 0.4.4), then export
each new finalized canonical dataset separately. Replace the two dataset-root
placeholders with paths from `collection.json`.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/ambench_sim"
export HF_HOME="$PWD/outputs/openpi/cache/huggingface"
python scripts/data/export_lerobot_to_openpi.py \
  --dataset_roots <PushSlider-canonical-dataset> --target_hz 30 \
  --omit_base_image --preserve_terminal --repo_id my-zzy/pushslider_30hz_terminal_v1 \
  --output_root outputs/openpi/datasets/my-zzy/pushslider_30hz_terminal_v1
python scripts/data/export_lerobot_to_openpi.py \
  --dataset_roots <PullLever-canonical-dataset> --target_hz 30 \
  --omit_base_image --preserve_terminal --repo_id my-zzy/pulllever_30hz_terminal_v1 \
  --output_root outputs/openpi/datasets/my-zzy/pulllever_30hz_terminal_v1
python scripts/data/merge_openpi_datasets.py --episodes-per-task 100 \
  --sources outputs/openpi/datasets/my-zzy/pressbutton_30hz_terminal_v2 \
            outputs/openpi/datasets/my-zzy/pushslider_30hz_terminal_v1 \
            outputs/openpi/datasets/my-zzy/pulllever_30hz_terminal_v1 \
  --output outputs/openpi/datasets/my-zzy/ambench_contact3_30hz_v1
```

Do not set a global `--task_prompt` when exporting multiple tasks. Instructions
come from their original episode metadata. The exporter retains the last
state/image/action pair, holding it at the next uniform 30 Hz grid point when
necessary. The merger changes only global episode, frame, and task indices;
it preserves encoded image bytes, actions, and episode boundaries. It saves
per-episode source/output SHA-256 hashes and refuses an existing destination.

## 3. Validate before training

The existing OpenPI server wrapper activates `~/.conda/envs/ambench_openpi`.
CPU commands do not allocate the GPUs selected in that wrapper.

```bash
bash experiments/pressbutton/openpi_conda.sh compute_norm_stats \
  --variant multitask --repo-id my-zzy/ambench_contact3_30hz_v1 \
  --artifact-dir outputs/openpi/multitask/production_validation
bash experiments/pressbutton/openpi_conda.sh prompt_probe \
  --repo-id my-zzy/ambench_contact3_30hz_v1 \
  --output outputs/openpi/multitask/production_validation/prompt_tokens.json
```

Statistics use float64 centered reductions over all overlapping action chunks,
including endpoint padding, in a separate multi-task asset namespace. Validation
checks actual loader chunks and prompts at episode boundaries/interiors and the
public encode/normalize/pad/decode path. Quaternion rotations are compared after
normalization and sign alignment, with separate source-norm validity checks;
PullLever's recorded quaternions contain small norm drift. Position/gripper and
rotation-component tolerances remain 2e-6. Existing statistics must match exactly
if the command is repeated.

Before a substantive run, perform a bounded two-update full-tuning smoke test on
the pilot dataset, reload its checkpoint, and run `prompt_probe --checkpoint
<checkpoint>`. This checks all three instructions, finite 30 × 8 actions, and
fixed-observation/fixed-noise prompt swaps. Two updates do not establish task
success. Inspect GPUs immediately before launching; the ordinary wrapper uses
GPUs 0 and 1 and must not overlap collection on those devices.

## 4. Full fine-tuning

After the data and smoke checks pass, the concrete server recipe is:

```bash
bash experiments/multitask/train_server.sh
```

Review its short configuration block and free GPU capacity first. This starts
**fresh pi0_base full fine-tuning**, batch 32, two-device FSDP, eight workers,
20,000 updates, seed 42, with checkpoints every 1,000 steps and retained 5,000-step
milestones. The previous comparable PressButton run took 9 h 47 min; the new
mixture's measured speed may differ. Obtain approval before this long run, as
required by this repository's `AGENTS.md`.

The experiment name is `contact3_full_20k_v1`; checkpoints live under
`outputs/openpi/checkpoints/pi0_multitask_full_30hz/`, while the supervisor log,
CSV metrics, and status live under `outputs/openpi/multitask/runs/`. Existing
single-task weights and statistics are separate and remain available as a
regression baseline.

## 5. Evaluate each task and prompt condition

Serve the final checkpoint in the OpenPI environment on a free GPU. The existing
wrapper selects GPUs 0 and 1; adjust its explicit GPU setting in a private launch
copy if simulation occupies either device.

```bash
bash experiments/pressbutton/openpi_conda.sh serve \
  --variant multitask --finetune-mode full --repo-id my-zzy/ambench_contact3_30hz_v1 \
  --checkpoint <completed-checkpoint> --port 8020
```

Use the server simulator wrapper on a separate free GPU (GPU 3 in this example):

```bash
bash experiments/multitask/eval_server.sh 3 \
  --policy-id contact3_full_20k_v1 --port 8020 --condition canonical \
  --num-rollouts 20 --num-envs 4 \
  --output outputs/openpi/multitask/evaluation/canonical
```

Repeat in distinct output directories with `--condition paraphrase` and
`--condition swapped`. `--plan-only` prints the exact task/prompt/seed schedule.
Each task gets 20 fresh cases; conditions reuse the same cases, with four cases
per batch and seed bases in the table above. Every scored trial has wrist and
external camera videos. `evaluation.json` reports per-task success, while each
batch preserves the standard evaluator results, traces, and timings.

Decode every saved clip and create a task-by-task video index in the simulator
environment (this also works on a downloaded copy of the results):

```bash
python -m ambench_learn.eval.validate_multitask_videos \
  --input-dir outputs/openpi/multitask/evaluation/canonical
```

Report per-task and macro-average success, including PressButton regression,
and compare prompt conditions without selecting a checkpoint on these held-out
cases. A drop under wrong prompts is useful evidence of language sensitivity;
a successful rollout under a wrong prompt is ambiguous in a scene with only one
available task. No multi-task learned success score exists until these rollouts
are actually run.

## 6. Diagnose incomplete slider pushes

The diagnostic module compares the registered scripted expert with the same
served PI policy. It adds a read-only contact sensor on the slider and saves
120 Hz object position/velocity, normal contact force, measured/commanded EE
pose, gripper state/targets, and executed actions. Measurements include the
terminal physics step before automatic reset. Both camera videos use 30 Hz.
Privileged measurements are saved for analysis and are not passed to PI.

Check GPU availability before starting. Use a free GPU for the existing policy
server and a different free GPU for the simulator. The proposed diagnostic
cohort uses seeds **72000 and 72004**, four environments per seed, with both
expert and policy modes: eight paired scenes and sixteen rollouts. These seeds
are separate from collection and the original full-checkpoint evaluation cohort.
Seed 72000 appeared in an earlier one-scene, two-update interface smoke test;
this diagnostic cohort is not an entirely untouched test set.

For one batch, with the checkpoint served on localhost port 8042:

```bash
bash experiments/multitask/diagnose_slider_server.sh 3 \
  --mode expert --seed 72000 --num-envs 4 \
  --output-dir outputs/openpi/multitask/diagnostics/slider_contact/expert_72000 \
  --headless --device cuda:0
bash experiments/multitask/diagnose_slider_server.sh 3 \
  --mode policy --seed 72000 --num-envs 4 --port 8042 \
  --policy-id contact3_full_20k_v1/19999 \
  --output-dir outputs/openpi/multitask/diagnostics/slider_contact/policy_72000 \
  --headless --device cuda:0
```

Repeat with seed 72004 and new output directories. The diagnostic refuses an
existing directory. Bound each process with a timeout when supervising runs.
Compare `initial_state.json` across modes before interpreting paired results;
matching seed numbers alone do not prove matching scenes. Keep policy
sampling stochastic and report this as a small diagnostic cohort, not a new
benchmark or a checkpoint-selection set.

`trace_env_<index>.npz` contains the pre-reset measurements, `summary.json`
contains native success/timeout outcomes, and `inference.jsonl` contains the
predicted action chunks in policy mode. Videos begin at the initial observation;
trace sample one follows the first physics step. The contact signal is the
slider's **net normal force**, not a gripper-specific force or friction force.
Interpret it together with relative tool position, gripper state, and slider
motion. Use a new untouched evaluation cohort after any data/model changes.

### Height-hold intervention

To test upward drift, use `--height-hold` in policy mode. At the first observed
native engagement (5% of required slider travel), the diagnostic latches the
current interpolated command's local Z and holds that coordinate through
termination. X/Y, rotation, gripper, inference frequency, and native success
criteria are unchanged. The policy still replans from its actual observations.
This action override uses privileged slider progress and is not a deployable
VLA result. Without the flag, commands remain unmodified.

For a controlled comparison, serve with common Gaussian sampling noise keyed
by scene seed, environment index, and control step:

```bash
bash experiments/pressbutton/openpi_conda.sh diagnose_slider_serve \
  --checkpoint <completed-checkpoint> --port 8043
bash experiments/multitask/diagnose_slider_server.sh 3 \
  --mode policy --seed 72000 --num-envs 4 --port 8043 --paired-noise \
  --output-dir outputs/openpi/multitask/diagnostics/height_hold/control_72000 \
  --headless --device cuda:0
bash experiments/multitask/diagnose_slider_server.sh 3 \
  --mode policy --seed 72000 --num-envs 4 --port 8043 --paired-noise --height-hold \
  --output-dir outputs/openpi/multitask/diagnostics/height_hold/hold_72000 \
  --headless --device cuda:0
```

Repeat both conditions with seed 72004. Choose free GPUs in the launcher's
explicit configuration before starting. The diagnostic server strips the noise
key before model preprocessing and supplies noise through OpenPI's existing
inference argument; it does not change checkpoint parameters or transforms.
`inference.jsonl` records a noise hash for every request. Verify matching hashes,
initial states, and pre-intervention trajectories across conditions. Traces also
include `unmodified_action`, `height_hold_active`, and `height_hold_reference_z`
to audit that only Z changes after engagement. These previously inspected
scenes are for diagnosis; reserve new scenes for subsequent model evaluation.

### Combined height and wall-normal distance hold

Add `--depth-hold` to latch the commanded tool-to-slider separation along the
wall normal after engagement. Only that projection changes; slide-axis motion,
orientation, and gripper commands remain the policy's. With `--height-hold`,
local Z is held too. Depth hold currently requires a horizontal wall normal,
matching this task's yaw-only wall placement. It uses privileged geometry for
diagnosis, not as a model input.

To control the approach before comparing no hold, height only, and height plus
depth, pass the same prior untreated batch to all three runs:

```bash
bash experiments/multitask/diagnose_slider_server.sh 3 \
  --mode policy --seed 72000 --num-envs 4 --port 8044 --paired-noise \
  --replay-prefix-dir <matching-prior-untreated-batch> \
  --height-hold --depth-hold \
  --output-dir outputs/openpi/multitask/diagnostics/depth_hold/combined_72000 \
  --headless --device cuda:0
```

Serve the paired-noise endpoint on port 8044 first. Repeat with only
`--height-hold`, then with neither hold flag, in separate output directories.
The replay loader verifies scene/reset equality, task, policy ID, seed, and
environment count. The first 32-step boundary after recorded engagement is the
intervention time. All conditions replay the recorded raw commands through one
further 32-step chunk, then resume live inference. Thus the first raw chunk at
the intervention boundary also matches; the holds alter its execution only.

Check `prefix_schedule` and source hashes in `metadata.json`, full robot and
object traces before intervention, and the first live observations saved as
`handoff_observation_env_<index>.npz`. Matching prefixes do not guarantee exact
physics or rendering: quantify remaining state differences before calling the
result causal. This hybrid replay-and-policy experiment is a diagnostic, not
an ordinary policy success benchmark. Keep the same prefix in every condition
and do not select scenes based on intervention outcomes.

### Audit targets before corrective training

The offline audit compares all 100 slider episodes against their native 120 Hz
recordings and individual 30 Hz export. It checks terminal pairs, actual loader
chunks, normalization, relative-frame round trips, and an independent SciPy
position transform. Pushing-phase depth and height are measured relative to each
expert's fixed commanded plane. The registered expert's final tool +X supplies
the wall normal; the audit validates the original collection's +X alignment.

Use the existing OpenPI environment. For checkpoint inference, inspect GPU
availability and select a free GPU in a private copy of the wrapper:

```bash
bash <private-openpi-wrapper> audit_slider_targets \
  --raw-dataset <canonical-slider-lerobot-root> \
  --slider-export outputs/openpi/datasets/my-zzy/pushslider_30hz_terminal_v1 \
  --checkpoint <completed-checkpoint> \
  --handoff-dir <prior-depth-diagnostic-server-directory> \
  --output-dir outputs/openpi/multitask/diagnostics/<new-audit-directory>
```

Omit `--checkpoint` and `--handoff-dir` for CPU checks only. The output directory
must be new. Frozen-model inference uses 12 uniformly selected training episodes,
six observations per episode, and three explicit Gaussian-noise draws each.
Optional handoffs add all eight prior untreated diagnostic observations. Their
reference is the expert's nominal depth/height plane, not a complete corrective
trajectory. Positive depth error means away from the wall.

These are teacher-forced imitation and off-trajectory diagnostics, not held-out
success scores. Preserve per-observation results and separate approach, pushing,
and handoff summaries. Do not interpret repeated noise draws or overlapping
action chunks as independent evaluation episodes. This command never trains or
changes the checkpoint, dataset, simulator, or controller.
