# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for the lost terminal observation/action pair."""

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location("terminal_repair", REPO_ROOT / "scripts/data/repair_openpi_terminals.py")
REPAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPAIR)


class TerminalRepairTests(unittest.TestCase):
    def fixture(self, root: Path, length: int) -> tuple[Path, Path]:
        source, original = root / "source", root / "original"
        for directory in (source, original):
            (directory / "meta").mkdir(parents=True)
            (directory / "data/chunk-000").mkdir(parents=True)
        source_info = {
            "fps": 120,
            "ambench": {"state_keys": ["ee_pos", "ee_quat", "gripper_width"], "action_semantics": "ee_absolute"},
        }
        (source / "meta/info.json").write_text(json.dumps(source_info))
        (original / "meta/info.json").write_text(
            json.dumps({"fps": 30, "total_episodes": 1, "total_frames": length // 4, "ambench": {"openpi_export": {}}})
        )
        native = []
        for index in range(length):
            pixels = np.full((384, 384, 3), index, dtype=np.uint8)
            stream = io.BytesIO()
            Image.fromarray(pixels).save(stream, format="PNG")
            native.append({
                "episode_index": 0,
                "frame_index": index,
                "observation.state": [float(index), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "action": [float(index + 1), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
                "observation.images.ee_camera": {"bytes": stream.getvalue(), "path": None},
            })
        pq.write_table(pa.Table.from_pylist(native), source / "data/chunk-000/episode_000000.parquet")
        old = []
        for index, item in enumerate(native[::4][: length // 4]):
            old.append({
                "ee_image": item["observation.images.ee_camera"],
                "actions": item["action"],
                "ee_pos": item["observation.state"][:3],
                "ee_quat": item["observation.state"][3:7],
                "gripper_width": 0.0,
                "timestamp": index / 30,
                "frame_index": index,
                "episode_index": 0,
                "index": index,
                "task_index": 0,
            })
        pq.write_table(pa.Table.from_pylist(old), original / "data/chunk-000/episode_000000.parquet")
        values = np.arange(0, length // 4 * 4, 4) / 255
        image_stats = {
            key: [[[float(value)]]] * 3
            for key, value in {
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "max": values.max(),
            }.items()
        }
        image_stats["count"] = [len(old)]
        (original / "meta/episodes_stats.jsonl").write_text(
            json.dumps({"episode_index": 0, "stats": {"ee_image": image_stats}}) + "\n"
        )
        (original / "meta/episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "length": len(old), "tasks": ["press the button"]}) + "\n"
        )
        (original / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "press the button"}\n')
        return source, original

    def test_terminal_pair_and_original_prefix_survive_grid_boundaries(self) -> None:
        for length in (4, 5, 8, 9):
            with self.subTest(length=length), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, original = self.fixture(root, length)
                before = (original / "data/chunk-000/episode_000000.parquet").read_bytes()
                report = REPAIR.repair_export(source, original, root / "corrected")
                table = pq.read_table(root / "corrected/data/chunk-000/episode_000000.parquet")
                self.assertEqual(table["actions"][-1].as_py()[0], length)
                self.assertEqual(table["ee_pos"][-1].as_py()[0], length - 1)
                pixels = np.asarray(Image.open(io.BytesIO(table["ee_image"][-1].as_py()["bytes"])))
                self.assertTrue(np.all(pixels == length - 1))
                self.assertTrue(report["all_original_samples_preserved"])
                self.assertGreaterEqual(report["max_endpoint_hold_seconds"], -1e-9)
                self.assertLessEqual(report["max_endpoint_hold_seconds"], 0.025 + 1e-9)
                self.assertEqual(before, (original / "data/chunk-000/episode_000000.parquet").read_bytes())
                self.assertTrue(np.allclose(np.diff(table["timestamp"].to_numpy()), 1 / 30))
                with self.assertRaises(FileExistsError):
                    REPAIR.repair_export(source, original, root / "corrected")

    def test_other_sample_rates_are_rejected_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, original = self.fixture(root, 8)
            info = json.loads((source / "meta/info.json").read_text())
            info["fps"] = 60
            (source / "meta/info.json").write_text(json.dumps(info))
            with self.assertRaises(ValueError):
                REPAIR.repair_export(source, original, root / "corrected")
            self.assertFalse((root / "corrected").exists())


if __name__ == "__main__":
    unittest.main()
