# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Audit action transforms and normalization without changing training assets."""

import argparse
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--work-dir", type=Path, default=Path("outputs/openpi"))
parser.add_argument("--repo-id", default="my-zzy/pressbutton_30hz")
parser.add_argument("--dataset-root", type=Path)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--heldout-dir", type=Path, help="Fresh native expert records; omit to audit training data")
args = parser.parse_args()
os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = args.output_dir.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
OPENPI = args.work_dir.resolve()
os.environ["OPENPI_DATA_HOME"] = str(OPENPI / "cache/openpi")
os.environ["HF_HOME"] = str(OPENPI / "cache/huggingface")
os.environ["HF_LEROBOT_HOME"] = str(OPENPI / "datasets")
DATASET = args.repo_id
CHECKPOINT = args.checkpoint.resolve()
args.heldout = args.heldout_dir is not None
import numpy as np
import pyarrow.parquet as pq
from openpi import transforms
from openpi.policies import am_bench_policy as adapter
from openpi.shared import normalize

from ambench_learn.policies.pi.config import make_config

stats = normalize.load(CHECKPOINT / "assets" / DATASET)
norm = transforms.Normalize(stats)
unnorm = transforms.Unnormalize({"actions": stats["actions"]})
cfg = make_config(work_dir=OPENPI, repo_id=DATASET)
data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)


def rotation_error(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Quaternion angular distance insensitive to the equivalent quaternion sign."""
    lhs = lhs.astype(np.float64) / np.linalg.norm(lhs.astype(np.float64), axis=-1, keepdims=True)
    rhs = rhs.astype(np.float64) / np.linalg.norm(rhs.astype(np.float64), axis=-1, keepdims=True)
    distance = np.minimum(np.linalg.norm(lhs - rhs, axis=-1), np.linalg.norm(lhs + rhs, axis=-1))
    return np.degrees(4 * np.arcsin(np.clip(distance / 2, 0, 1)))


all_relative, max_errors, chain_errors = [], [], []
paths = (
    sorted(args.heldout_dir.glob("*.npz"))
    if args.heldout
    else sorted(((args.dataset_root or OPENPI / "datasets" / DATASET) / "data").rglob("*.parquet"))
)
assert paths, "No input episodes"
for path in paths:
    if args.heldout:
        item = dict(np.load(path))
        # Match the repository exporter: discard a trailing incomplete 120-to-30 Hz interval.
        length = len(item["actions_120hz"]) // 4
        states = item["states_30hz"][:length]
        actions = item["actions_30hz"][:length]
        item.update(states_30hz=states, actions_30hz=actions)
        (ROOT / "heldout_export").mkdir(exist_ok=True)
        np.savez_compressed(ROOT / "heldout_export" / path.name, **item)
    else:
        table = pq.read_table(path, columns=["ee_pos", "ee_quat", "gripper_width", "actions"])
        states = np.concatenate(
            [
                np.asarray(table[key].to_pylist(), dtype=np.float32).reshape(len(table), -1)
                for key in ("ee_pos", "ee_quat", "gripper_width")
            ],
            axis=-1,
        )
        actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
    indices = np.minimum(np.arange(len(actions))[:, None] + np.arange(30), len(actions) - 1)
    chunks = actions[indices]
    relative = adapter._to_ee_local_relative(chunks, states)
    normalized = norm({"actions": relative})["actions"]
    padded = transforms.pad_to_dim(normalized, 32, axis=-1)
    decoded = adapter._to_ee_local_absolute(unnorm({"actions": padded})["actions"], states)
    assert np.isfinite(decoded).all()
    max_errors.append([
        float(np.linalg.norm(decoded[..., :3] - chunks[..., :3], axis=-1).max()),
        float(rotation_error(decoded[..., 3:7], chunks[..., 3:7]).max()),
        float(np.abs(decoded[..., 7] - chunks[..., 7]).max()),
    ])
    all_relative.append(relative.reshape(-1, 8))
    # Exercise the public training/inference transforms, including their shared state cache.
    for index in np.linspace(0, len(actions) - 1, 4).round().astype(int):
        example = {
            "am_bench/ee_pos": states[index, :3],
            "am_bench/ee_quat": states[index, 3:7],
            "am_bench/gripper_width": states[index, 7:],
            "am_bench/ee_image": np.zeros((384, 384, 3), dtype=np.uint8),
            "prompt": "press the button",
            "actions": chunks[index],
        }
        mapped = transforms.compose(data_cfg.data_transforms.inputs)(example)
        mapped = transforms.PadStatesAndActions(32)(norm(mapped))
        result = transforms.compose(data_cfg.data_transforms.outputs)(unnorm({"actions": mapped["actions"]}))
        chain_errors.append(float(np.max(np.abs(result["actions"] - chunks[index]))))
    if args.heldout:
        # Decode each 8-action execution chunk with the very same recorded anchor used to encode it.
        reconstructed = actions.copy()
        for start in range(0, len(actions), 8):
            count = min(8, len(actions) - start)
            reconstructed[start : start + count] = decoded[start, :count]
        (ROOT / "converted").mkdir(exist_ok=True)
        np.savez_compressed(ROOT / "converted" / path.name, actions_30hz=reconstructed)

values = np.concatenate(all_relative).astype(np.float64)
normalized = norm({"actions": values})["actions"]
maximum = np.max(max_errors, axis=0)
report = {
    "episodes": len(paths),
    "action_targets_including_overlapping_horizons": len(values),
    "roundtrip_max_position_error_m": float(maximum[0]),
    "roundtrip_max_rotation_error_degrees": float(maximum[1]),
    "roundtrip_max_gripper_command_error": float(maximum[2]),
    "public_transform_examples": len(chain_errors),
    "public_transform_max_component_error": max(chain_errors),
    "channels": ["dx", "dy", "dz", "qw", "qx", "qy", "qz", "gripper_command"],
    "saved_mean": stats["actions"].mean.tolist(),
    "saved_std": stats["actions"].std.tolist(),
    "recomputed_float64_mean": values.mean(axis=0).tolist(),
    "recomputed_float64_std": values.std(axis=0).tolist(),
    "normalized_channel_std": normalized.std(axis=0).tolist(),
    "normalized_abs_p99": np.quantile(np.abs(normalized), 0.99, axis=0).tolist(),
    "normalized_abs_max": np.max(np.abs(normalized), axis=0).tolist(),
    "passed_roundtrip": bool(maximum[0] < 2e-6 and maximum[1] < 0.001 and maximum[2] < 1e-6),
}
for source in [CHECKPOINT / "assets" / DATASET / "norm_stats.json", cfg.assets_dirs / DATASET / "norm_stats.json"]:
    report[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
name = "heldout_adapter_audit.json" if args.heldout else "training_adapter_audit.json"
(ROOT / name).write_text(json.dumps(report, indent=2) + "\n")
assert report["passed_roundtrip"], report
print(json.dumps(report, indent=2))
