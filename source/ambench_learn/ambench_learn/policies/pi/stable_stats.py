# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Compute float64 training statistics and validate the real OpenPI data path."""

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from openpi import transforms
from openpi.policies import am_bench_policy as adapter
from openpi.shared import normalize

from ambench_learn.policies.pi.config import make_config


def action_roundtrip_errors(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    """Compare absolute EE targets, accounting for quaternion normalization and double cover."""
    if actual.shape != expected.shape or actual.shape[-1] != 8:
        raise ValueError("Expected matching absolute EE action arrays with eight components")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Nonfinite action in transform round trip")
    actual_quat, expected_quat = actual[..., 3:7], expected[..., 3:7]
    actual_norm = np.linalg.norm(actual_quat, axis=-1, keepdims=True)
    expected_norm = np.linalg.norm(expected_quat, axis=-1, keepdims=True)
    # Allow small recorder precision drift, but never hide malformed source rotations.
    if np.max(np.abs(expected_norm - 1)) > 1e-3 or np.max(np.abs(actual_norm - 1)) > 2e-6:
        raise ValueError("Invalid quaternion norm in transform round trip")
    expected_unit = expected_quat / expected_norm
    actual_unit = actual_quat / actual_norm
    sign = np.where(np.sum(actual_unit * expected_unit, axis=-1, keepdims=True) < 0, -1, 1)
    return {
        "position_m": float(np.max(np.abs(actual[..., :3] - expected[..., :3]))),
        "unit_quaternion_component": float(np.max(np.abs(actual_unit - sign * expected_unit))),
        "gripper": float(np.max(np.abs(actual[..., 7] - expected[..., 7]))),
        "source_quaternion_norm_deviation": float(np.max(np.abs(expected_norm - 1))),
        "raw_component": float(np.max(np.abs(actual - expected))),
    }


def compute_stable_stats(work_dir: Path, repo_id: str, report_path: Path, *, multitask: bool = False) -> dict:
    """Use all overlapping target horizons, including endpoint padding, without float32 reductions."""
    ROOT = work_dir.resolve()
    DATASET = repo_id
    os.environ["HF_LEROBOT_HOME"] = str(ROOT / "datasets")
    os.environ["OPENPI_DATA_HOME"] = str(ROOT / "cache/openpi")
    os.environ["HF_HOME"] = str(ROOT / "cache/huggingface")
    from openpi.training import data_loader

    cfg = make_config(work_dir=ROOT, repo_id=DATASET, corrected=True, multitask=multitask)
    tasks = {
        row["task_index"]: row["task"]
        for row in map(json.loads, (ROOT / "datasets" / DATASET / "meta/tasks.jsonl").read_text().splitlines())
    }
    task_frames: collections.Counter[str] = collections.Counter()
    episode_prompts = []
    arrays = {"state": [], "actions": []}
    original_states, original_actions = [], []
    for path in sorted((ROOT / "datasets" / DATASET / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["ee_pos", "ee_quat", "gripper_width", "actions", "task_index"])
        task_indices = set(table["task_index"].to_pylist())
        if len(task_indices) != 1:
            raise ValueError(f"Each episode must have one instruction; found {task_indices} in {path}")
        prompt = tasks[next(iter(task_indices))]
        episode_prompts.append(prompt)
        task_frames[prompt] += len(table)
        states = np.concatenate(
            [
                np.asarray(table[key].to_pylist(), dtype=np.float32).reshape(len(table), -1)
                for key in ("ee_pos", "ee_quat", "gripper_width")
            ],
            axis=-1,
        )
        actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        indices = np.minimum(np.arange(len(actions))[:, None] + np.arange(30), len(actions) - 1)
        relative = adapter._to_ee_local_relative(actions[indices], states)
        arrays["state"].append(adapter._localize_ee_state(states).astype(np.float64))
        arrays["actions"].append(relative.reshape(-1, 8).astype(np.float64))
        original_states.append(states)
        original_actions.append(actions)
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    stats = {
        key: normalize.NormStats(
            mean=value.mean(axis=0),
            std=value.std(axis=0),
            q01=np.quantile(value, 0.01, axis=0),
            q99=np.quantile(value, 0.99, axis=0),
        )
        for key, value in arrays.items()
    }
    for key, value in arrays.items():
        assert np.isfinite(value).all()
        constant = np.ptp(value, axis=0) == 0
        assert np.all(stats[key].std[~constant] > 0)
        assert np.all(stats[key].std[constant] == 0)
    destination = cfg.assets_dirs / DATASET
    if (destination / "norm_stats.json").exists():
        existing = normalize.load(destination)
        for key in stats:
            for field in ("mean", "std", "q01", "q99"):
                assert np.array_equal(getattr(existing[key], field), getattr(stats[key], field)), (key, field)
    else:
        normalize.save(destination, stats)
    norm, unnorm = transforms.Normalize(stats), transforms.Unnormalize({"actions": stats["actions"]})
    config = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = data_loader.create_torch_dataset(config, cfg.model.action_horizon, cfg.model)
    assert len(dataset) == len(arrays["state"])
    cursor, checks = 0, 0
    maximum_errors: dict[str, float] = collections.defaultdict(float)
    for episode, (states, actions) in enumerate(zip(original_states, original_actions, strict=True)):
        # All episode boundaries plus representative interiors exercise actual timestamp-based chunking.
        for index in sorted({0, len(actions) // 2, max(0, len(actions) - 30), len(actions) - 1}):
            sample = transforms.compose(config.repack_transforms.inputs)(dataset[cursor + index])
            assert str(sample["prompt"]) == episode_prompts[episode], (episode, index, sample["prompt"])
            expected = actions[np.minimum(index + np.arange(30), len(actions) - 1)]
            assert np.array_equal(sample["actions"], expected), (episode, index)
            mapped = transforms.compose(config.data_transforms.inputs)(sample)
            assert np.array_equal(mapped["state"], adapter._localize_ee_state(states[index]))
            mapped = transforms.PadStatesAndActions(32)(norm(mapped))
            output = transforms.compose(config.data_transforms.outputs)(unnorm({"actions": mapped["actions"]}))
            for key, error in action_roundtrip_errors(output["actions"], expected).items():
                maximum_errors[key] = max(maximum_errors[key], error)
            assert np.isfinite(mapped["state"]).all() and np.isfinite(mapped["actions"]).all()
            checks += 1
        cursor += len(actions)
    assert all(maximum_errors[key] < 2e-6 for key in ("position_m", "unit_quaternion_component", "gripper"))
    report = {
        "passed": True,
        "method": "Float64 two-pass centered standard deviation over all transformed training targets",
        "training_frames": len(arrays["state"]),
        "task_frame_counts": dict(task_frames),
        "task_episode_counts": dict(collections.Counter(episode_prompts)),
        "sampling": "Uniform over frames; equal episode counts do not imply equal task probabilities",
        "action_targets": len(arrays["actions"]),
        "actual_loader_and_public_transform_checks": checks,
        "roundtrip_max_component_error": maximum_errors["raw_component"],
        "roundtrip_errors": dict(maximum_errors),
        "roundtrip_comparison": "Position/gripper unchanged; unit quaternions compared up to sign",
        "action_mean": stats["actions"].mean.tolist(),
        "action_std": stats["actions"].std.tolist(),
        "normalized_action_std": norm({"actions": arrays["actions"]})["actions"].std(axis=0).tolist(),
        "stats_path": str((destination / "norm_stats.json").relative_to(ROOT)),
        "stats_sha256": hashlib.sha256((destination / "norm_stats.json").read_bytes()).hexdigest(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/openpi"))
    parser.add_argument("--repo-id", default="my-zzy/pressbutton_30hz_terminal_v2")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"
    report = compute_stable_stats(args.work_dir, args.repo_id, args.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
