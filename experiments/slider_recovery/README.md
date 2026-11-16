# PushSlider contact recovery

This experiment focuses on one fixed FAHexa PushSlider scene. A privileged
scripted teacher returns to the slider's **current** position after a deliberately
induced departure, opens the gripper, aligns, approaches, closes, and resumes
pushing. It does not change the native PID controller or task success threshold.
It is a demonstration collector, not evidence that the learned VLA can recover.

## Collection

The collector is `scripts/data/collect_slider_recovery.py`; the reusable teacher
is `ambench.policies.scripted.push_slider_recovery.PushSliderRecoveryPolicy`.
Use an activated Isaac Lab environment and the usual AM-Bench runtime variables:

```bash
python scripts/data/collect_slider_recovery.py --headless \
  --cases experiments/slider_recovery/pilot_cases.json \
  --case-start 0 --case-count 4 \
  --output-dir outputs/slider_recovery/batch_00
```

On the previously configured server, `bash experiments/slider_recovery/collect_server.sh
<idle-gpu> ...` activates the existing `ambench_sim` environment and Xvfb. Inspect
GPU and disk capacity first. Each invocation requires a new output directory.

`pilot_cases.json` declares 24 trials: departures at 15%, 45%, and 75% of required
slider travel; four displacement patterns; and departure durations of 1.2 and
1.6 seconds. Normal departures are 0.10–0.35 m, with selected height and lateral
offsets. Scene geometry stays fixed. These are controlled perturbation trials,
not 24 independent scenes or a generalization benchmark.

Contact loss requires six consecutive 30 Hz observations with slider contact
force norm below 0.1 N, closed gripper width below 15 mm, and tool-to-slider normal
separation above 280 mm. After closing, the teacher makes a small push probe:
at least 15 mm of object motion, contact, gripper width, normal separation, and
alignment are required to verify a regrasp. A saved recovery must subsequently move the
slider at least 20 mm and satisfy native success (joint position at least 0.45 m)
within the native 26-second episode. Net contact force is not finger-pair-specific;
the other checks and resumed object motion are essential.

## Saved artifacts

- `summary.json`: every trial, including failures, phase timings and recovery checks.
- `metadata.json`, `initial_state.json`, `env_cfg.yaml`: protocol and exact scene setup.
- `trace_env_*.npz`: 120 Hz post-physics measurements captured before automatic reset.
- `videos/`: full attempt videos from wrist, base, and external scene cameras at 30 Hz.
- `lerobot/`: successful recovery segments only, with wrist/base RGB and 8D
  end-effector pose plus gripper width. Privileged object/contact state is excluded.
- `labels_episode_*.npz`: pre-action states, macro targets, and original control
  step indices for independently checking dataset alignment.

Each training frame contains the observation before the next **30 Hz target**.
That target is executed using four interpolated commands at 120 Hz. The forced
departure is excluded from labels. Treat the canonical dataset as already 30 Hz;
do not downsample it by four again. Each recovery segment is a separate episode.

Run the independent numeric and image audit after collection:

```bash
python experiments/slider_recovery/validate_pilot.py outputs/slider_recovery/batch_00
```

It checks finalized LeRobot metadata, image decoding, actual sustained contact
loss, native completion, state alignment before each action, and agreement
between saved targets and the commands executed at 120 Hz.

The existing wrist-only OpenPI model is unchanged. Recording the base camera
preserves an option for later training when the wrist camera loses sight of the
handle. No recovery fine-tune is launched by these tools. After validating this
pilot, collect more recoveries mixed with ordinary successful pushes and reserve
unseen departure states for learned-policy evaluation.
