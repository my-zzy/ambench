# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Collect separate canonical expert datasets for an explicit task manifest."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/multitask/tasks.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", help="Task names to collect; defaults to all manifest tasks.")
    parser.add_argument("--num-demos", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if min(args.num_demos, args.num_envs, args.timeout_seconds) < 1:
        parser.error("Demo count, environment count, and timeout must be positive")
    manifest = json.loads(args.manifest.read_text())
    tasks = {task["name"]: task for task in manifest["tasks"]}
    selected = args.tasks or list(tasks)
    if len(set(selected)) != len(selected) or set(selected) - tasks.keys():
        parser.error("Select each known task at most once")
    root = args.output.resolve()
    repo = Path(__file__).resolve().parents[2]
    commands = {}
    for name in selected:
        task = tasks[name]
        commands[name] = [
            sys.executable,
            "-u",
            str(repo / "scripts/data/record_demos_scripted.py"),
            "--task",
            f"{name}-Am-{manifest['robot_profile']}-Direct-v0",
            "--num_envs",
            str(args.num_envs),
            "--num_demos",
            str(args.num_demos),
            "--seed",
            str(task["train_seed"]),
            "--camera_fps",
            str(manifest["policy_hz"]),
            "--step_hz",
            "100000",
            "--env_length_s",
            str(task["episode_length_s"]),
            "--dataset_root",
            str(root / name),
            "--state_keys",
            *manifest["state_keys"],
            "--task_prompt",
            task["prompt"],
            "--repo_id",
            f"ambench/multitask_{name}",
            "--headless",
            "--device",
            "cuda:0",
            *(["--video"] if args.video else []),
        ]
    if args.plan_only:
        print(json.dumps(commands, indent=2))
        return
    root.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    status = {"status": "running", "manifest": manifest, "commands": commands, "tasks": []}
    try:
        for name, command in commands.items():
            status["active_task"] = name
            (root / "collection.json").write_text(json.dumps(status, indent=2) + "\n")
            started = time.monotonic()
            with (root / f"{name}.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    cwd=repo,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
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
            if returncode or "Dataset recorder finalized." not in (root / f"{name}.log").read_text():
                raise RuntimeError(f"{name} collection failed; inspect {root / (name + '.log')}")
            candidates = list((root / name).rglob("lerobot/meta/info.json"))
            if len(candidates) != 1:
                raise RuntimeError(f"Expected one finalized canonical dataset for {name}")
            info = json.loads(candidates[0].read_text())
            if info["total_episodes"] < args.num_demos:
                raise RuntimeError(f"{name} exported fewer than {args.num_demos} successful episodes")
            status["tasks"].append({
                "task": name,
                "prompt": tasks[name]["prompt"],
                "seed": tasks[name]["train_seed"],
                "dataset": str(candidates[0].parents[1]),
                "episodes": info["total_episodes"],
                "frames": info["total_frames"],
                "seconds": time.monotonic() - started,
            })
        status["status"] = "completed"
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        (root / "collection.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
