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
