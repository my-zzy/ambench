# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Audit recovery labels against independently captured pre-reset physics traces."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def validate(root: Path) -> dict:
    """Check timing, actual loss/regrasp, fixed geometry, and finalized dataset data."""
    summary = json.loads((root / "summary.json").read_text())
    metadata = json.loads((root / "metadata.json").read_text())
    initial = json.loads((root / "initial_state.json").read_text())
    assert summary["status"] == "completed"
    assert metadata["dataset_hz"] == 30 and metadata["control_hz"] == 120
    for name in ("wall", "slider", "robot"):
        pose = np.asarray(initial[name])
        np.testing.assert_allclose(pose, np.broadcast_to(pose[0], pose.shape), atol=2e-6)
    accepted = [case for case in summary["cases"] if case["verified_recovery_success"]]
    if not accepted:
        return {"status": "no_accepted_episodes", "attempts": len(summary["cases"]), "accepted": 0}
    dataset = root / "lerobot"
    info = json.loads((dataset / "meta/info.json").read_text())
    assert info["fps"] == 30 and info["total_episodes"] == len(accepted)
    assert info["ambench"]["state_keys"] == ["ee_pos", "ee_quat", "gripper_width"]
    assert info["ambench"]["action_semantics"] == "ee_absolute"
    columns = ["action", "observation.state", "episode_index", "timestamp", "frame_index"]
    tables = [pq.read_table(path, columns=columns) for path in sorted((dataset / "data").rglob("*.parquet"))]
    rows = [row for table in tables for row in table.to_pylist()]
    assert len(rows) == info["total_frames"]
    max_state_error = max_action_error = 0.0
    for index, case in enumerate(summary["cases"]):
        if not case["verified_recovery_success"]:
            continue
        episode = case["dataset_episode_index"]
        samples = [row for row in rows if row["episode_index"] == episode]
        with np.load(root / f"labels_episode_{episode}.npz") as labels, np.load(
            root / f"trace_env_{index}.npz"
        ) as trace:
            steps = labels["control_step"]
            assert steps[0] == case["recovery_start_step"] and np.all(np.diff(steps) == 4)
            assert len(samples) == len(steps) == case["candidate_frames"]
            np.testing.assert_allclose([row["action"] for row in samples], labels["action"], atol=0)
            np.testing.assert_allclose([row["observation.state"] for row in samples], labels["state"], atol=0)
            np.testing.assert_allclose([row["timestamp"] for row in samples], np.arange(len(steps)) / 30, atol=1e-6)
            assert [row["frame_index"] for row in samples] == list(range(len(steps)))
            measured = np.concatenate(
                [
                    trace["ee_pos"][steps - 1],
                    trace["ee_quat"][steps - 1],
                    trace["gripper_joint_pos"][steps - 1].sum(axis=-1, keepdims=True),
                ],
                axis=-1,
            )
            state_error = float(np.max(np.abs(measured - labels["state"])))
            assert state_error < 2e-6, f"Pre-action alignment error: {state_error}"
            complete = steps + 3 < len(trace["action"])
            action_error = float(np.max(np.abs(trace["action"][steps[complete] + 3] - labels["action"][complete])))
            assert action_error < 2e-6, f"Executed target mismatch: {action_error}"
            max_state_error = max(max_state_error, state_error)
            max_action_error = max(max_action_error, action_error)
            # Six observations preceding recovery independently confirm sustained loss.
            lost_steps = steps[0] - 1 - np.arange(6) * 4
            force = np.linalg.norm(trace["slider_normal_contact_force_w"][lost_steps], axis=-1)
            width = trace["gripper_joint_pos"][lost_steps].sum(axis=-1)
            separation = (trace["slider_body_pos"] - trace["ee_pos"])[lost_steps, 0]
            assert np.all(force < 0.1) and np.all(width < 0.015) and np.all(separation > 0.28)
            assert trace["slider_joint_pos"][-1] >= 0.45 and case["resumed_travel_m"] >= 0.02
            assert case["regrasp_step"] > case["recovery_start_step"] and case["steps"] <= 3120
            if "probe" in metadata["phase_names"]:
                regrasp = case["regrasp_step"] - 1
                probes = [event["step"] for event in case["events"] if event["phase"] == "probe"]
                probe_start = max(step for step in probes if step < regrasp)
                assert trace["slider_joint_pos"][regrasp] - trace["slider_joint_pos"][probe_start - 1] >= 0.015
                assert np.linalg.norm(trace["slider_normal_contact_force_w"][regrasp]) > 0.1
                assert trace["gripper_joint_pos"][regrasp].sum() > 0.025
            assert all(np.isfinite(labels[key]).all() for key in labels.files)
    cameras = [key for key in info["features"] if key.startswith("observation.images.")]
    assert set(cameras) == {"observation.images.ee_camera", "observation.images.base_camera"}
    checked_images = 0
    for path in sorted((dataset / "data").rglob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=cameras):
            for row in (batch.slice(0, 1).to_pylist()[0], batch.slice(len(batch) - 1, 1).to_pylist()[0]):
                for camera in cameras:
                    value = row[camera]
                    image = (
                        Image.open(io.BytesIO(value["bytes"]))
                        if value["bytes"]
                        else Image.open(dataset / value["path"])
                    )
                    rgb = np.asarray(image.convert("RGB"))
                    assert list(rgb.shape) == info["features"][camera]["shape"]
                    assert rgb.std() > 5 and np.isfinite(rgb).all(), "Blank or invalid camera image"
                    checked_images += 1
    return {
        "status": "passed",
        "attempts": len(summary["cases"]),
        "accepted": len(accepted),
        "frames": len(rows),
        "max_pre_action_state_error": max_state_error,
        "max_executed_target_error": max_action_error,
        "images_sampled": checked_images,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = validate(args.run_dir)
    (args.run_dir / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
