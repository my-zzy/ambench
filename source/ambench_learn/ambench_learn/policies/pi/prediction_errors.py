# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the frozen pilot on prespecified training and fresh expert observations."""

import argparse
import csv
import io
import json
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--work-dir", type=Path, default=Path("outputs/openpi"))
parser.add_argument("--repo-id", default="my-zzy/pressbutton_30hz")
parser.add_argument(
    "--dataset-root", type=Path, required=True, help="Expert observations to evaluate; may differ from training export"
)
parser.add_argument("--reference-dir", type=Path, required=True, help="Directory containing heldout_export")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
ROOT = args.output_dir.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
OPENPI = args.work_dir.resolve()
os.environ["OPENPI_DATA_HOME"] = str(OPENPI / "cache/openpi")
os.environ["HF_HOME"] = str(OPENPI / "cache/huggingface")
os.environ["HF_LEROBOT_HOME"] = str(OPENPI / "datasets")
DATASET = args.repo_id
CHECKPOINT = args.checkpoint.resolve()
TRAIN_EPISODES = [0, 14, 28, 42, 56, 70, 84, 99]
import numpy as np
import pyarrow.parquet as pq
from openpi import transforms
from openpi.policies import am_bench_policy as adapter
from openpi.policies import policy_config
from openpi.shared import normalize
from PIL import Image

from ambench_learn.policies.pi.config import make_config

cfg = make_config(work_dir=OPENPI, repo_id=DATASET)
keys = ["am_bench/ee_pos", "am_bench/ee_quat", "am_bench/gripper_width", "am_bench/ee_image", "prompt"]
policy = policy_config.create_trained_policy(
    cfg, CHECKPOINT, repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({key: key for key in keys})])
)
stats = normalize.load(CHECKPOINT / "assets" / DATASET)
rows, predicted, expected = [], [], []


def summarize_error(prediction: np.ndarray, target: np.ndarray, prefix: int) -> dict:
    prediction, target = prediction[:prefix], target[:prefix]
    delta = prediction[:, :3] - target[:, :3]
    qa, qt = prediction[:, 3:7].astype(np.float64), target[:, 3:7].astype(np.float64)
    qa /= np.linalg.norm(qa, axis=-1, keepdims=True)
    qt /= np.linalg.norm(qt, axis=-1, keepdims=True)
    chord = np.minimum(np.linalg.norm(qa - qt, axis=-1), np.linalg.norm(qa + qt, axis=-1))
    rotation = np.degrees(4 * np.arcsin(np.clip(chord / 2, 0, 1)))
    unambiguous = np.abs(target[:, 7]) >= 0.5
    return {
        f"position_mean_m_{prefix}": float(np.linalg.norm(delta, axis=-1).mean()),
        f"position_z_bias_m_{prefix}": float(delta[:, 2].mean()),
        f"rotation_mean_degrees_{prefix}": float(rotation.mean()),
        f"gripper_clipped_mae_{prefix}": float(np.abs(np.clip(prediction[:, 7], -1, 1) - target[:, 7]).mean()),
        f"gripper_sign_correct_{prefix}": int(np.sum((prediction[:, 7] * target[:, 7] > 0) & unambiguous)),
        f"gripper_sign_total_{prefix}": int(unambiguous.sum()),
        f"gripper_out_of_range_fraction_{prefix}": float((np.abs(prediction[:, 7]) > 1).mean()),
    }


for subset, episodes in [("training", TRAIN_EPISODES), ("heldout", list(range(8)))]:
    for episode in episodes:
        if subset == "training":
            table = pq.read_table(args.dataset_root / f"data/chunk-000/episode_{episode:06d}.parquet")
            states = np.concatenate(
                [
                    np.asarray(table[key].to_pylist(), dtype=np.float32).reshape(len(table), -1)
                    for key in ("ee_pos", "ee_quat", "gripper_width")
                ],
                axis=-1,
            )
            actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
            sample_indices = np.linspace(0, len(actions) - 30, 12).round().astype(int)
            images = [
                np.asarray(Image.open(io.BytesIO(table["ee_image"][int(i)].as_py()["bytes"])).convert("RGB"))
                for i in sample_indices
            ]
        else:
            item = np.load(args.reference_dir / f"heldout_export/episode_{episode:03d}.npz")
            states, actions = item["states_30hz"], item["actions_30hz"]
            sample_indices, images = item["sample_indices"], item["sample_images"]
        for sample_index, frame in enumerate(sample_indices):
            state = states[frame]
            target = actions[np.minimum(np.arange(frame, frame + 30), len(actions) - 1)]
            example = {
                "am_bench/ee_pos": state[:3],
                "am_bench/ee_quat": state[3:7],
                "am_bench/gripper_width": state[7:],
                "am_bench/ee_image": images[sample_index],
                "prompt": "press the button",
            }
            # A constant mean action in the learned local frame is a simple image-independent baseline.
            mean_action = adapter._to_ee_local_absolute(np.tile(stats["actions"].mean, (30, 1)), state)
            baseline = summarize_error(mean_action, target, 8)
            for draw in range(3):
                start = time.perf_counter()
                prediction = np.asarray(policy.infer(example)["actions"], dtype=np.float32)
                latency = time.perf_counter() - start
                assert prediction.shape == (30, 8) and np.isfinite(prediction).all()
                row = {
                    "subset": subset,
                    "episode": episode,
                    "frame": int(frame),
                    "episode_fraction": float(frame / (len(actions) - 1)),
                    "draw": draw,
                    "inference_seconds": latency,
                    **summarize_error(prediction, target, 8),
                    **summarize_error(prediction, target, 30),
                    "mean_action_baseline_position_m_8": baseline["position_mean_m_8"],
                    "hold_position_baseline_m_8": float(np.linalg.norm(target[:8, :3] - state[:3], axis=-1).mean()),
                }
                rows.append(row)
                predicted.append(prediction)
                expected.append(target)
        print(f"{subset} episode={episode}: {len(rows)} predictions completed", flush=True)
        with (ROOT / "prediction_errors.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

np.savez_compressed(ROOT / "predictions.npz", predictions=np.stack(predicted), targets=np.stack(expected))
report = {
    "checkpoint": str(CHECKPOINT),
    "training_episode_ids": TRAIN_EPISODES,
    "heldout_seeds": [20000, 20004],
    "observations_per_episode": 12,
    "draws_per_observation": 3,
    "model_updates_during_diagnostics": 0,
    "scope": (
        "Teacher-forced offline imitation error; samples and action chunks overlap; not a closed-loop success rate"
    ),
}
for subset in ("training", "heldout"):
    selected = [r for r in rows if r["subset"] == subset]
    group = {"predictions": len(selected), "episodes": 8, "observations": 96}
    for key in (
        "position_mean_m_8",
        "position_mean_m_30",
        "position_z_bias_m_8",
        "rotation_mean_degrees_8",
        "gripper_clipped_mae_8",
        "gripper_out_of_range_fraction_8",
        "mean_action_baseline_position_m_8",
        "hold_position_baseline_m_8",
    ):
        values = np.asarray([r[key] for r in selected])
        group[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
        }
    group["gripper_sign_accuracy_8"] = sum(r["gripper_sign_correct_8"] for r in selected) / sum(
        r["gripper_sign_total_8"] for r in selected
    )
    group["per_episode_position_mean_m_8"] = {
        str(ep): float(np.mean([r["position_mean_m_8"] for r in selected if r["episode"] == ep]))
        for ep in sorted({r["episode"] for r in selected})
    }
    report[subset] = group
(ROOT / "offline_prediction_summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
