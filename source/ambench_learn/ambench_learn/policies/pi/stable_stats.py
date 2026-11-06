# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Compute float64 training statistics and validate the real OpenPI data path."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from ambench_learn.policies.pi.config import make_config
from openpi import transforms
from openpi.policies import am_bench_policy as adapter
from openpi.shared import normalize


def compute_stable_stats(work_dir: Path, repo_id: str, report_path: Path) -> dict:
    """Use all overlapping target horizons, including endpoint padding, without float32 reductions."""
    ROOT = work_dir.resolve()
    DATASET = repo_id
    os.environ["HF_LEROBOT_HOME"] = str(ROOT / "datasets")
    os.environ["OPENPI_DATA_HOME"] = str(ROOT / "cache/openpi")
    os.environ["HF_HOME"] = str(ROOT / "cache/huggingface")
    from openpi.training import data_loader

    cfg = make_config(work_dir=ROOT, repo_id=DATASET, corrected=True)
    arrays = {"state": [], "actions": []}
    original_states, original_actions = [], []
    for path in sorted((ROOT / "datasets" / DATASET / "data").rglob("*.parquet")):
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
    cursor, maximum, checks = 0, 0.0, 0
    for episode, (states, actions) in enumerate(zip(original_states, original_actions, strict=True)):
        # All episode boundaries plus representative interiors exercise actual timestamp-based chunking.
        for index in sorted({0, len(actions) // 2, max(0, len(actions) - 30), len(actions) - 1}):
            sample = transforms.compose(config.repack_transforms.inputs)(dataset[cursor + index])
            expected = actions[np.minimum(index + np.arange(30), len(actions) - 1)]
            assert np.array_equal(sample["actions"], expected), (episode, index)
            mapped = transforms.compose(config.data_transforms.inputs)(sample)
            assert np.array_equal(mapped["state"], adapter._localize_ee_state(states[index]))
            mapped = transforms.PadStatesAndActions(32)(norm(mapped))
            output = transforms.compose(config.data_transforms.outputs)(unnorm({"actions": mapped["actions"]}))
            error = float(np.abs(output["actions"] - expected).max())
            maximum = max(maximum, error)
            assert np.isfinite(mapped["state"]).all() and np.isfinite(mapped["actions"]).all()
            checks += 1
        cursor += len(actions)
    assert maximum < 2e-6
    report = {
        "passed": True,
        "method": "Float64 two-pass centered standard deviation over all transformed training targets",
        "training_frames": len(arrays["state"]),
        "action_targets": len(arrays["actions"]),
        "actual_loader_and_public_transform_checks": checks,
        "roundtrip_max_component_error": maximum,
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
