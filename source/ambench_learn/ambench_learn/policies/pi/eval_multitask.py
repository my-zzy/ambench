# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Run paired task/prompt conditions through the standard PI simulator evaluator."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/multitask/tasks.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=["canonical", "paraphrase", "swapped"], default="canonical")
    parser.add_argument("--num-rollouts", type=int, default=20, help="Fresh cases per task.")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if min(args.num_rollouts, args.num_envs, args.timeout_seconds) < 1:
        parser.error("Rollout/environment counts and timeout must be positive")
    manifest = json.loads(args.manifest.read_text())
    tasks = manifest["tasks"]
    root = args.output.resolve()
    batches = []
    for index, task in enumerate(tasks):
        prompt = task["prompt"]
        if args.condition == "paraphrase":
            prompt = task["paraphrase"]
        elif args.condition == "swapped":
            prompt = tasks[(index + 1) % len(tasks)]["prompt"]
        for start in range(0, args.num_rollouts, args.num_envs):
            count = min(args.num_envs, args.num_rollouts - start)
            seed = task["eval_seed"] + start
            output = root / task["name"] / f"batch_{start // args.num_envs:02d}"
            command = [
                sys.executable,
                "-u",
                "-m",
                "ambench_learn.policies.pi.eval",
                "--task",
                f"{task['name']}-Am-{manifest['robot_profile']}-Direct-v0",
                "--num-envs",
                str(args.num_envs),
                "--num-rollouts",
                str(count),
                "--seed",
                str(seed),
                "--rollout-seed-base",
                str(seed),
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--prompt",
                prompt,
                "--policy-id",
                args.policy_id,
                "--policy-target-hz",
                str(manifest["policy_hz"]),
                "--n-action-steps",
                str(manifest["n_action_steps"]),
                "--save-video",
                "--video-camera-names",
                "ee_camera",
                "scene_camera",
                "--scene-video-size",
                "640",
                "360",
                "--record-inference-timing",
                "--record-runtime-timing",
                "--output-dir",
                str(output),
                "--headless",
                "--device",
                "cuda:0",
            ]
            batches.append({
                "task": task["name"],
                "seed": seed,
                "prompt": prompt,
                "count": count,
                "output": str(output),
                "command": command,
            })
    if args.plan_only:
        print(json.dumps(batches, indent=2))
        return
    root.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    report = {
        "status": "running",
        "condition": args.condition,
        "manifest": manifest,
        "policy_id": args.policy_id,
        "plan": batches,
        "batches": [],
    }
    (root / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        for batch in batches:
            output = Path(batch["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.with_suffix(".log").open("w") as log:
                process = subprocess.Popen(
                    batch["command"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                try:
                    returncode = process.wait(timeout=args.timeout_seconds)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
            if returncode:
                raise RuntimeError(f"Evaluator failed: {output}")
            result = json.loads((output / "eval_summary.json").read_text())
            if result["status"] != "completed" or len(result["rollouts"]) != batch["count"]:
                raise RuntimeError(f"Incomplete rollout evidence: {output}")
            report["batches"].append({
                **{key: value for key, value in batch.items() if key != "command"},
                "successes": sum(item["success"] for item in result["rollouts"]),
            })
            (root / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        report["per_task"] = {
            task["name"]: {
                "successes": sum(batch["successes"] for batch in report["batches"] if batch["task"] == task["name"]),
                "rollouts": args.num_rollouts,
            }
            for task in tasks
        }
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        (root / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
