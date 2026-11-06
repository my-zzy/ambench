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
