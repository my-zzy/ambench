# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Decode every saved ACT clip and index the 20 scored trials' two primary views."""
import argparse
import json
import os
from pathlib import Path

import av

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-dir", type=Path, required=True, help="Parent of eval/batch_* directories")
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--policy", choices=["act", "pi"], default="act")
args = parser.parse_args()
root = args.input_dir.resolve()
args.output_dir.mkdir(parents=True, exist_ok=True)
records = []
primary_paths = set()
markdown = [
    f"# {args.policy.upper()} rollout videos",
    "",
    (
        "Each row is one scored rollout. Extra `eps1` or later clips are post-termination fragments from the native"
        " vector-environment recorder; they are preserved but do not count as additional trials."
    ),
    "",
    "| Trial | Reset seed | Environment | Success | Simulated seconds | External view | Gripper camera |",
    "|---:|---:|---:|:---:|---:|---|---|",
]
for batch in range(5):
    batch_dir = root / "eval" / f"batch_{batch}"
    result = json.loads((batch_dir / "eval_summary.json").read_text())
    assert result["status"] == "completed" and len(result["rollouts"]) == 4
    reset_seed = result["metadata"].get("Rollout seed base", result["metadata"].get("Seed"))
    for rollout in result["rollouts"]:
        env_index = rollout["batch_env_index"]
        trial = batch * 4 + rollout["rollout_idx"] + 1
        paths = {}
        for camera in ["scene_camera", "ee_camera"]:
            path = batch_dir / "videos" / f"{args.policy}-eval-{camera}-env{env_index}-eps0.mp4"
            primary_paths.add(path)
            paths[camera] = os.path.relpath(path, args.output_dir)
        markdown.append(
            f"| {trial} | {reset_seed} | {env_index} | {'Yes' if rollout['success'] else 'No'} |"
            f" {rollout['executed_steps'] / 120:.2f} | [Video]({paths['scene_camera']}) |"
            f" [Video]({paths['ee_camera']}) |"
        )
    for path in sorted((batch_dir / "videos").glob("*.mp4")):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            frames = sum(1 for _ in container.decode(video=0))
            assert frames > 0 and float(stream.average_rate) == 30, path
            if "ee_camera" in path.name:
                assert (stream.width, stream.height) == (384, 384), path
            else:
                assert (stream.width, stream.height) == (640, 360), path
            records.append({
                "path": str(path.relative_to(root)),
                "primary_scored_trial": path in primary_paths,
                "decoded_frames": frames,
                "fps": float(stream.average_rate),
                "width": stream.width,
                "height": stream.height,
                "bytes": path.stat().st_size,
            })
    for rollout in result["rollouts"]:
        env_index = rollout["batch_env_index"]
        expected_frames = 1 + rollout["executed_steps"] // 4
        for camera in ["scene_camera", "ee_camera"]:
            relative_path = str(
                (batch_dir / "videos" / f"{args.policy}-eval-{camera}-env{env_index}-eps0.mp4").relative_to(root)
            )
            record = next(row for row in records if row["path"] == relative_path)
            assert abs(record["decoded_frames"] - expected_frames) <= 1, (record, expected_frames)
assert len(primary_paths) == 40
assert sum(row["primary_scored_trial"] for row in records) == 40
report = {
    "passed": True,
    "primary_scored_trial_clips": 40,
    "all_clips_decoded": len(records),
    "auxiliary_post_termination_clips": len(records) - 40,
    "videos": records,
}
(args.output_dir / "video_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
(args.output_dir / "videos.md").write_text("\n".join(markdown) + "\n")
print(json.dumps({key: value for key, value in report.items() if key != "videos"}, indent=2))
