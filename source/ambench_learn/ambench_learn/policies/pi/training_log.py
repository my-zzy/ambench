# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Record upstream training output and stop only the child process group we own."""

import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def run_logged(arguments: list[str], output_dir: Path) -> int:
    """Launch the module trainer, preserving logs and terminal failure status."""
    output_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", "-m", "ambench_learn.policies.pi.runtime", "train", *arguments]
    started = time.monotonic()
    state = {"status": "starting", "command": command}
    process = None

    def save() -> None:
        state["elapsed_seconds"] = time.monotonic() - started
        temporary = output_dir / "train_status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output_dir / "train_status.json")

    def interrupted(signum: int, frame) -> None:
        raise KeyboardInterrupt(f"Received signal {signum}")

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    save()
    try:
        with (output_dir / "train.log").open("w", buffering=1) as log, (output_dir / "training_metrics.csv").open(
            "w", newline="", buffering=1
        ) as metrics:
            writer = csv.DictWriter(metrics, fieldnames=["step", "elapsed_seconds", "loss", "grad_norm", "param_norm"])
            writer.writeheader()
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True
            )
            (output_dir / "train.pid").write_text(str(process.pid) + "\n")
            for line in process.stdout:
                log.write(line)
                if line.startswith("AMBENCH_TRAIN_COMPLETED "):
                    state["training_summary"] = json.loads(line.split(" ", 1)[1])
                match = re.search(r"Step (\d+): (.+)", line)
                if match:
                    values = {
                        key: float(value)
                        for key, value in re.findall(r"(loss|grad_norm|param_norm)=([^,\s]+)", match.group(2))
                    }
                    step = int(match.group(1))
                    writer.writerow({"step": step, "elapsed_seconds": time.monotonic() - started, **values})
                    state.update(status="training", completed_steps=step + 1, latest_logged_metrics=values)
                    save()
                    print(line.strip(), flush=True)
            code = process.wait()
            state.update(status="completed" if code == 0 else "failed", exit_code=code)
            if code == 0:
                if "training_summary" not in state:
                    raise RuntimeError("Trainer exited without confirming checkpoint completion")
                state["completed_steps"] = state["training_summary"]["steps"]
            return code
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=repr(error))
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        (output_dir / "train.pid").unlink(missing_ok=True)
        save()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
