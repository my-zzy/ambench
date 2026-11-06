# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Aggregate completed evaluation batches without counting recorder fragments."""

import argparse
import csv
import json
from pathlib import Path


def summarize(input_dir: Path, output_dir: Path, expected_rollouts: int, physics_hz: float) -> dict:
    """Read batch summaries and write a distinct CSV, JSON, and Markdown report."""
    paths = sorted(input_dir.glob("batch_*/eval_summary.json"), key=lambda p: int(p.parent.name.split("_")[-1]))
    if not paths:
        raise ValueError(f"No batch summaries under {input_dir}")
    records = []
    seed_cases = set()
    for path in paths:
        result = json.loads(path.read_text())
        if result["status"] != "completed" or result.get("error") is not None:
            raise ValueError(f"Incomplete evaluation: {path}")
        if len(result["rollouts"]) != result["summary"]["num_rollouts"]:
            raise ValueError(f"Inconsistent rollout count: {path}")
        seed = result["metadata"].get("Rollout seed base", result["metadata"].get("Seed"))
        for rollout in result["rollouts"]:
            key = (seed, rollout["batch_env_index"], rollout["rollout_idx"])
            if key in seed_cases:
                raise ValueError(f"Repeated evaluation case: {key}")
            seed_cases.add(key)
            records.append({
                "trial": len(records) + 1,
                "batch": path.parent.name,
                "reset_seed": seed,
                "environment": rollout["batch_env_index"],
                "success": rollout["success"],
                "executed_steps": rollout["executed_steps"],
                "simulation_seconds": rollout["executed_steps"] / physics_hz,
                "termination_reason": rollout["termination_reason"],
            })
    if len(records) != expected_rollouts:
        raise ValueError(f"Expected {expected_rollouts} trials, found {len(records)}")
    successes = sum(row["success"] for row in records)
    report = {
        "num_rollouts": len(records),
        "num_successes": successes,
        "success_rate": successes / len(records),
        "source_batch_summaries": [str(p.resolve()) for p in paths],
        "rollouts": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output_dir / "rollouts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# Evaluation results",
        "",
        f"Successes: **{successes}/{len(records)}**.",
        "",
        "| Trial | Reset seed | Success | Simulated seconds |",
        "| ---: | ---: | :---: | ---: |",
    ]
    for row in records:
        lines.append(f"| {row['trial']} | {row['reset_seed']} | {row['success']} | {row['simulation_seconds']:.2f} |")
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rollouts", type=int, default=20)
    parser.add_argument("--physics-hz", type=float, default=120)
    args = parser.parse_args()
    if args.expected_rollouts <= 0 or args.physics_hz <= 0:
        parser.error("Expected rollouts and physics rate must be positive")
    report = summarize(args.input_dir, args.output_dir, args.expected_rollouts, args.physics_hz)
    print(f"Successes: {report['num_successes']}/{report['num_rollouts']}")


if __name__ == "__main__":
    main()
