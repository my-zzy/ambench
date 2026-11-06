# Source reconstruction map

All source paths below are relative to the repository root. Historical files
remain unchanged and ignored; they are provenance, never runtime imports.

| Historical source group | Tracked implementation | Milestone |
| --- | --- | --- |
| `outputs/expert_demos/20261101_144142/run_task.py`, `run_suite.py` | `scripts/data/record_expert_episode.py`, `run_expert_suite.py` | Expert recordings |
| ACT `record_training_demos.py` | Seed/camera-rate options in `scripts/data/record_demos_scripted.py` | ACT |
| ACT `train_full.sh`, `eval_full.sh`, `evaluate_act_batch.py` | `experiments/pressbutton/act_*.sh` and `ambench_learn.policies.act.eval` | ACT |
| ACT `python_deps/` distribution metadata | `experiments/pressbutton/requirements-act.txt`, `scripts/setup/install_act_runtime.sh`; recreate packages instead of tracking third-party installations | Portability verification |
| ACT `evaluate_expert.py`, `eval_expert_full.sh` | `scripts/environments/evaluate_scripted.py` and documented paired seed schedule | ACT |
| ACT `aggregate_results.py`, rollout table portions of report builders | `ambench_learn.eval.summarize` | ACT |
| ACT and PI `validate_videos.py` | `ambench_learn.eval.validate_videos`, parameterized by policy and paths | ACT / PI |
| OpenPI `install.sh`, `run.sh`, `run.py`, `local_config.py` | `scripts/setup/install_openpi.sh`, `experiments/pressbutton/openpi.sh`, `ambench_learn.policies.pi.{config,runtime,train,infer,serve,compute_norm_stats}` | OpenPI setup |
| OpenPI setup `download_checkpoint.py`, `checkpoint_objects.json` | `scripts/setup/download_pi0_checkpoint.py`, minimized `pi0_base_manifest.json` | OpenPI setup |
| Pilot `train.py`, corrected `train.py` | Optional `--run-dir` supervision in `pi.train` and `pi.training_log` | PI pilot |
| Pilot/corrected `evaluate_pi_batch.py`, `eval_full.sh`, `eval_batch.sh` | `pi.eval`, `experiments/pressbutton/pi_eval.sh` | PI pilot |
| Diagnostics `audit.py`, `audit_training_tails.py` | `pi.audit`, `scripts/data/audit_openpi_tails.py` | Diagnosis |
| Diagnostics `sim_diagnostics.py`, corrected `replay_corrected.py` | `scripts/data/diagnose_pressbutton_replay.py`, with separate replay variants | Diagnosis / correction |
| Original/corrected `infer_errors.py` | `pi.prediction_errors`, with explicit checkpoint and observation dataset | Diagnosis |
| Corrected `export_corrected.py`, `compute_stable_stats.py`, `local_config.py` | `scripts/data/repair_openpi_terminals.py`, `pi.stable_stats`, `pi.config --variant corrected` through the runtime CLI | Correction |
| Comparison portions of corrected `summarize.py` | `pi.compare`, verifying paired targets and cases before comparing results | Correction |

## Deliberately retained only in the historical runs

- `runtime/train_local_cache.py` is an upstream-derived generated file. The
  tracked runtime regenerates it from the pinned submodule with a checked cache
  substitution; committing a second trainer would conceal its provenance.
- `complete_setup.sh`, `after_training.sh`, `finish_pipeline.py`,
  `wait_and_train.py`, polling/status scripts, and process PID files coordinated
  specific sessions. The recipes now expose explicit stages, and `pi.train`
  retains owned-process supervision. They do not depend on stale timestamps,
  prior job completion markers, workstation environment names, or old PIDs.
- Config/run snapshots duplicate the recovered implementation. Machine-specific
  package inventories and environment-preservation audits remain evidence;
  install inputs are the standard project requirements and pinned OpenPI lock.
- Ad hoc data/checkpoint smoke validators are superseded for reproduction by
  the tracked canonical dataset validator, native checkpoint loading, OpenPI
  infer/client checks, and the regression/roundtrip checks described in README.
- Figure/thumbnail builders, `build_report.py`, `write_report.py`,
  `write_setup_report.py`, and English/Chinese PDF slide builders only format
  saved evidence. They are not dependencies of collection, export, training,
  inference, or evaluation. Their original sources and rendered outputs remain
  preserved locally; tracked summaries provide reproducible result tables.
- Generated normalization files, model weights, raw recordings, predictions,
  videos, dataset metadata, and experiment logs are transferred as artifacts.
  They are not source dependencies hidden inside Python imports.

No older commits were amended or backdated. These milestone commits reconstruct
the work on the current date while documenting when the experiments occurred.
