# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Run each remaining task sequentially; preserve completed results and logs."""

import argparse
import contextlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--tasks", nargs="*")
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
root = args.output.resolve()
root.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[2]
tasks = [
    "PressButton",
    "PushSlider",
    "PullLever",
    "RotateValve",
    "OpenDoor",
    "PegInHole",
    "WipeWindow",
    "TossBall",
    "FrameAssembly",
    "CabinetPickPlace",
    "LemonHarvesting",
    "NDT",
]
requested = set(args.tasks or []) or set(tasks)
assert requested <= set(tasks), requested
packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions() if dist.metadata["Name"]}
manifest = (
    json.loads((root / "manifest.json").read_text())
    if (root / "manifest.json").exists()
    else {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "packages": packages,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], text=True
        ),
        "scope": "12 task families; FAHexa absolute end-effector PID; registered scripted experts",
        "attempt_policy": "Seeds 0,1,2; stop after first success or three native-length episodes",
        "per_task_wall_limit_seconds": 1200,
        "tasks": [],
    }
)

for task in tasks:
    if task not in requested:
        continue
    result_path = root / task / "summary.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous.get("status") in {"success", "no_success"}:
            if not any(item["task"] == task for item in manifest["tasks"]):
                manifest["tasks"].append({"task": task, "status": previous["status"], "existing_completed_run": True})
            else:
                for item in manifest["tasks"]:
                    if item["task"] == task:
                        item["status"] = previous["status"]
            print(f"SKIP completed {task}: {previous['status']}", flush=True)
            continue
        raise RuntimeError(f"Refusing to overwrite incomplete run: {result_path}")
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("record_expert_episode.py")),
        "--task",
        f"{task}-Am-FAHexa-Abs-PID-Direct-v0",
        "--output",
        str(root / task),
        "--headless",
        "--device",
        args.device,
    ]
    entry = {
        "task": task,
        "command": command,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest["tasks"] = [item for item in manifest["tasks"] if item["task"] != task] + [entry]
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    start = time.monotonic()
    print(f"START {task}", flush=True)
    with (root / f"{task}.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=repo, start_new_session=True)
        try:
            entry["returncode"] = process.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            entry["status"] = "wall_timeout"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        finally:
            # Only this run's process group, including any encoding child.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
    if entry["status"] != "wall_timeout":
        entry["status"] = json.loads(result_path.read_text())["status"] if result_path.exists() else "startup_error"
    entry["wall_seconds"] = time.monotonic() - start
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"DONE {task}: {entry['status']} ({entry['wall_seconds']:.1f}s)", flush=True)

if len(manifest["tasks"]) == len(tasks) and all(item["status"] != "running" for item in manifest["tasks"]):
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("SUITE_FINISHED" if "finished_utc" in manifest else "BATCH_FINISHED", flush=True)

lines = [
    "# Scripted expert suite",
    "",
    "Sequential runs; up to three native-horizon episodes per task.",
    "",
    "| Task | Status |",
    "| --- | --- |",
]
for item in manifest["tasks"]:
    lines.append(f"| {item['task']} | {item['status']} |")
(root / "RESULTS.md").write_text("\n".join(lines) + "\n")
