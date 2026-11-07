# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise merging real parquet exports without changing supervision or images."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).resolve().parents[4]
modules = {}
for name in ["export_lerobot_to_openpi", "merge_openpi_datasets"]:
    spec = importlib.util.spec_from_file_location(name, REPO / f"scripts/data/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    modules[name] = module
EXPORT = modules["export_lerobot_to_openpi"]
MERGE = modules["merge_openpi_datasets"]


def test_merge_reindexes_tasks_without_changing_supervision(tmp_path: Path) -> None:
    roots = []
    for i, prompt in enumerate(["press the button", "pull the lever upwards"]):
        root = tmp_path / f"task_{i}"
        roots.append(root)
        frame = {
            "task": prompt,
            "ee_pos": np.array([i, 0, 0], dtype=np.float32),
            "ee_quat": np.array([1, 0, 0, 0], dtype=np.float32),
            "gripper_width": np.array([0.1], dtype=np.float32),
            "ee_image": np.full((8, 8, 3), i * 100, dtype=np.uint8),
            "actions": np.array([i, 0, 0, 1, 0, 0, 0, -1], dtype=np.float32),
        }
        writer = EXPORT.OpenPIV21Writer(
            root, fps=30, features=EXPORT.build_features(frame), export_metadata={"preserve_terminal": True}
        )
        writer.save_episode([frame] * (i + 2))
    output = tmp_path / "mixed"
    report = MERGE.merge_datasets(roots, output, 1)
    assert report["episodes"] == 2
    assert report["frames"] == 5
    assert report["task_sampling_probabilities"] == {"press the button": 0.4, "pull the lever upwards": 0.6}
    cursor = 0
    for i, root in enumerate(roots):
        old = pq.read_table(root / "data/chunk-000/episode_000000.parquet")
        new = pq.read_table(output / f"data/chunk-000/episode_{i:06d}.parquet")
        for key in ["actions", "ee_pos", "ee_quat", "gripper_width", "ee_image", "timestamp"]:
            assert old[key].equals(new[key])
        assert new["index"].to_pylist() == list(range(cursor, cursor + len(old)))
        assert set(new["episode_index"].to_pylist()) == {i}
        assert set(new["task_index"].to_pylist()) == {i}
        cursor += len(old)
    stats = [json.loads(line) for line in (output / "meta/episodes_stats.jsonl").read_text().splitlines()]
    assert stats[1]["stats"]["task_index"]["min"] == [1.0]
    with pytest.raises(FileExistsError):
        MERGE.merge_datasets(roots, output, 1)
    with pytest.raises(ValueError, match="need 2"):
        MERGE.merge_datasets(roots, tmp_path / "incomplete", 2)
    assert not (tmp_path / "incomplete").exists()
    info_path = roots[1] / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["ambench"]["openpi_export"]["preserve_terminal"] = False
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="terminal-preservation"):
        MERGE.merge_datasets(roots, tmp_path / "unsafe", 1)
