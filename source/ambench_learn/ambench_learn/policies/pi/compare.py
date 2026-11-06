# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare completed pilots only when rollout cases and offline targets match."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ambench_learn.eval.summarize import summarize


def compare_runs(
    baseline: Path,
    candidate: Path,
    output_dir: Path,
    baseline_diagnostics: Path | None = None,
    candidate_diagnostics: Path | None = None,
) -> dict:
    """Validate paired targets and retain both success and imitation-error results."""
    diagnostic_roots = (baseline_diagnostics or baseline, candidate_diagnostics or candidate)
    keys = []
    for root in diagnostic_roots:
        with (root / "prediction_errors.csv").open() as handle:
            keys.append([(r["subset"], r["episode"], r["frame"], r["draw"]) for r in csv.DictReader(handle)])
    if keys[0] != keys[1] or not keys[0]:
        raise ValueError("Offline observations/draw keys do not match")
    with np.load(diagnostic_roots[0] / "predictions.npz") as old, np.load(
        diagnostic_roots[1] / "predictions.npz"
    ) as new:
        if not np.array_equal(old["targets"], new["targets"]):
            raise ValueError("Offline action targets differ; this is not a paired comparison")
    summaries = {}
    for name, root in (("baseline", baseline), ("candidate", candidate)):
        summaries[name] = summarize(root / "eval", output_dir / name, 20, 120)
    case_keys = [[(r["trial"], r["reset_seed"], r["environment"]) for r in s["rollouts"]] for s in summaries.values()]
    if case_keys[0] != case_keys[1]:
        raise ValueError("Simulator evaluation cases do not match")
    for old_path, new_path in zip(
        summaries["baseline"]["source_batch_summaries"], summaries["candidate"]["source_batch_summaries"], strict=True
    ):
        old_meta = json.loads(Path(old_path).read_text())["metadata"]
        new_meta = json.loads(Path(new_path).read_text())["metadata"]
        for field in ("Task", "Seed", "Disturbance", "Num envs", "Eval overrides", "Prompt"):
            if old_meta.get(field) != new_meta.get(field):
                raise ValueError(f"Evaluation protocol differs: {field}")
    offline = {
        name: json.loads((root / "offline_prediction_summary.json").read_text())
        for name, root in zip(("baseline", "candidate"), diagnostic_roots, strict=True)
    }
    report = {
        "paired_offline_predictions": len(keys[0]),
        "paired_rollouts": len(case_keys[0]),
        "evaluation": summaries,
        "offline": offline,
    }
    lines = [
        "# Paired pi0 comparison",
        "",
        "| Metric | Baseline | Candidate |",
        "| --- | ---: | ---: |",
        f"| Successes | {summaries['baseline']['num_successes']}/20 | {summaries['candidate']['num_successes']}/20 |",
    ]
    for subset in ("training", "heldout"):
        a = offline["baseline"][subset]["position_mean_m_8"]["mean"] * 100
        b = offline["candidate"][subset]["position_mean_m_8"]["mean"] * 100
        lines.append(f"| {subset}: first-eight mean position error (cm) | {a:.3f} | {b:.3f} |")
    lines.extend([
        "",
        "Offline chunks overlap and use expert observations. They do not measure recovery during a rollout.",
        "The same 20 development cases and one training seed cannot establish broad generalization.",
    ])
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-diagnostics", type=Path)
    parser.add_argument("--candidate-diagnostics", type=Path)
    args = parser.parse_args()
    compare_runs(args.baseline, args.candidate, args.output_dir, args.baseline_diagnostics, args.candidate_diagnostics)


if __name__ == "__main__":
    main()
