# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Decode and index every camera clip from a completed multi-task evaluation."""

import argparse
import collections
import json
from pathlib import Path

import av


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    evaluation = json.loads((root / "evaluation.json").read_text())
    if evaluation["status"] != "completed":
        raise ValueError("Video validation requires a completed evaluation")
    records = []
    case_counts: collections.Counter[str] = collections.Counter()
    primary_paths = set()
    markdown = [
        "# Multi-task rollout videos",
        "",
        f"Policy: `{evaluation['policy_id']}`. Instruction condition: `{evaluation['condition']}`.",
        "",
        "Each row is one scored rollout. Extra post-termination clips are decoded but are not additional trials.",
        "",
        "| Task | Case | Reset seed | Environment | Success | External view | Wrist view |",
        "|---|---:|---:|---:|:---:|---|---|",
    ]
    for batch in evaluation["batches"]:
        # Resolve within the copied result tree, independently of the original server prefix.
        directory = root / batch["task"] / Path(batch["output"]).name
        result = json.loads((directory / "eval_summary.json").read_text())
        if result["status"] != "completed" or len(result["rollouts"]) != batch["count"]:
            raise ValueError(f"Incomplete batch: {directory}")
        stride = result["metadata"]["Eval overrides"]["action_execution_stride"]
        expected = {}
        for rollout in result["rollouts"]:
            case_counts[batch["task"]] += 1
            env_index = rollout["batch_env_index"]
            links = {}
            for camera in ("scene_camera", "ee_camera"):
                path = directory / "videos" / f"pi-eval-{camera}-env{env_index}-eps0.mp4"
                if not path.is_file():
                    raise FileNotFoundError(path)
                primary_paths.add(path)
                expected[path] = 1 + rollout["executed_steps"] // stride
                links[camera] = str(path.relative_to(root))
            markdown.append(
                f"| {batch['task']} | {case_counts[batch['task']]} | {batch['seed']} | {env_index} |"
                f" {'Yes' if rollout['success'] else 'No'} | [Video]({links['scene_camera']}) |"
                f" [Video]({links['ee_camera']}) |"
            )
        for path in sorted((directory / "videos").glob("*.mp4")):
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                frames = sum(1 for _ in container.decode(video=0))
                dimensions = (384, 384) if "ee_camera" in path.name else (640, 360)
                if frames < 1 or float(stream.average_rate) != 30 or (stream.width, stream.height) != dimensions:
                    raise ValueError(f"Unexpected video format or empty clip: {path}")
                if path in expected and abs(frames - expected[path]) > 1:
                    raise ValueError(f"Video duration mismatch: {path}: {frames} vs {expected[path]}")
                records.append({
                    "path": str(path.relative_to(root)),
                    "primary_scored_trial": path in primary_paths,
                    "decoded_frames": frames,
                    "fps": float(stream.average_rate),
                    "width": stream.width,
                    "height": stream.height,
                    "bytes": path.stat().st_size,
                })
    if len(primary_paths) != 2 * sum(case_counts.values()):
        raise ValueError("Scored trials do not have distinct two-camera videos")
    report = {
        "passed": True,
        "per_task_rollouts": dict(case_counts),
        "primary_scored_trial_clips": len(primary_paths),
        "all_clips_decoded": len(records),
        "videos": records,
    }
    (root / "video_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "videos.md").write_text("\n".join(markdown) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "videos"}, indent=2))


if __name__ == "__main__":
    main()
