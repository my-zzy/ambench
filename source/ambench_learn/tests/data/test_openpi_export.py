# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
from lerobot.datasets.utils import write_info
from PIL import Image

from ambench_learn.data.action_resampling import BASE_JOINT_ABSOLUTE, EE_ABSOLUTE

REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = REPO_ROOT / "scripts" / "data" / "export_lerobot_to_openpi.py"
SPEC = importlib.util.spec_from_file_location("export_lerobot_to_openpi", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class FakeDataset:
    meta = SimpleNamespace(tasks=None)


def test_openpi_frame_exports_ee_absolute_schema() -> None:
    frame = EXPORT.openpi_frame(
        {
            "observation.state": np.array([1, 2, 3, 1, 0, 0, 0, 0.25], dtype=np.float32),
            "action": np.array([2, 3, 4, 1, 0, 0, 0, -1], dtype=np.float32),
            "observation.images.ee_camera": np.ones((3, 4, 5), dtype=np.float32),
            "task": "press the button",
        },
        dataset=FakeDataset(),
        info={"ambench": {"state_keys": ["ee_pos", "ee_quat", "gripper_width"]}},
        action_semantics=EE_ABSOLUTE,
        ee_image_key="observation.images.ee_camera",
        base_image_key="observation.images.base_camera",
        include_base_image=False,
        task_prompt="",
        task_prompt_map={},
        require_task_prompt_map=False,
    )

    assert set(frame) == {"ee_pos", "ee_quat", "gripper_width", "ee_image", "actions", "task"}
    assert frame["ee_image"].shape == (4, 5, 3)
    assert frame["ee_image"].dtype == np.uint8
    assert frame["actions"].shape == (8,)


def test_openpi_frame_exports_base_joint_absolute_schema() -> None:
    frame = EXPORT.openpi_frame(
        {
            "observation.state": np.arange(12, dtype=np.float32),
            "action": np.arange(12, dtype=np.float32),
            "observation.images.ee_camera": np.zeros((4, 5, 3), dtype=np.uint8),
            "task": "press the button",
        },
        dataset=FakeDataset(),
        info={"ambench": {"state_keys": ["base_pos", "base_quat", "arm_joint_pos", "gripper_width"]}},
        action_semantics=BASE_JOINT_ABSOLUTE,
        ee_image_key="observation.images.ee_camera",
        base_image_key="observation.images.base_camera",
        include_base_image=False,
        task_prompt="",
        task_prompt_map={},
        require_task_prompt_map=False,
    )

    assert set(frame) == {"base_pos", "base_quat", "arm_joint_pos", "gripper_width", "ee_image", "actions", "task"}
    assert frame["actions"].shape == (12,)


def test_openpi_frame_rejects_legacy_delta_action_semantics() -> None:
    with pytest.raises(ValueError, match="ee_absolute and base_joint_absolute"):
        EXPORT.openpi_frame(
            {
                "observation.state": np.zeros(8, dtype=np.float32),
                "action": np.zeros(7, dtype=np.float32),
                "observation.images.ee_camera": np.zeros((4, 5, 3), dtype=np.uint8),
            },
            dataset=FakeDataset(),
            info={"ambench": {"state_keys": ["ee_pos", "ee_quat", "gripper_width"]}},
            action_semantics="ee_delta",
            ee_image_key="observation.images.ee_camera",
            base_image_key="observation.images.base_camera",
            include_base_image=False,
            task_prompt="press the button",
            task_prompt_map={},
            require_task_prompt_map=False,
        )


@pytest.mark.parametrize("preserve_terminal", [False, True])
def test_exporter_writes_a_loadable_resampled_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preserve_terminal: bool
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": [f"state_{index}" for index in range(8)],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": [f"action_{index}" for index in range(8)],
        },
        "observation.images.ee_camera": {
            "dtype": "image",
            "shape": (8, 8, 3),
            "names": ["height", "width", "channels"],
        },
    }
    source = EXPORT.LeRobotDataset.create(
        repo_id="ambench/test_ee_absolute",
        root=source_root,
        robot_type="am_bench",
        fps=120,
        features=features,
        use_videos=False,
        image_writer_threads=1,
        image_writer_processes=0,
    )
    source.meta.info.setdefault("ambench", {})["action_semantics"] = EE_ABSOLUTE
    source.meta.info["ambench"]["state_keys"] = ["ee_pos", "ee_quat", "gripper_width"]
    write_info(source.meta.info, source_root)
    for index in range(6):
        source.add_frame({
            "observation.state": np.array([index, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
            "action": np.array([index + 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
            "observation.images.ee_camera": np.full((8, 8, 3), index * 20, dtype=np.uint8),
            "task": "press the button",
        })
    source.save_episode()
    source.finalize()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--dataset_roots",
            str(source_root),
            "--repo_id",
            "am_bench/test_ee_absolute",
            "--output_root",
            str(output_root),
            "--target_hz",
            "20",
            "--omit_base_image",
            "--task_prompt",
            "press the button",
            *(["--preserve_terminal"] if preserve_terminal else []),
        ],
    )
    EXPORT.main()

    info = json.loads((output_root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v2.1"
    assert info["total_frames"] == (2 if preserve_terminal else 1)
    assert info["total_episodes"] == 1
    assert info["data_path"] == "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    assert info["features"]["ee_image"]["shape"] == [3, 8, 8]
    assert (output_root / "meta" / "tasks.jsonl").is_file()
    assert (output_root / "meta" / "episodes.jsonl").is_file()
    assert (output_root / "meta" / "episodes_stats.jsonl").is_file()

    table = pq.read_table(output_root / "data" / "chunk-000" / "episode_000000.parquet")
    assert table.num_rows == (2 if preserve_terminal else 1)
    row = table.slice(0, 1).to_pylist()[0]
    assert len(row["actions"]) == 8
    assert row["frame_index"] == 0
    assert row["episode_index"] == 0
    assert row["task_index"] == 0
    assert row["ee_image"]["bytes"].startswith(b"\x89PNG")
    parquet_metadata = json.loads(table.schema.metadata[b"huggingface"])
    assert parquet_metadata["info"]["features"]["actions"]["_type"] == "Sequence"

    report = info["ambench"]["openpi_export"]
    assert report["target_hz"] == 20
    assert report["source_action_semantics"] == EE_ABSOLUTE
    assert report["lerobot_version"] == "v2.1"
    assert report["preserve_terminal"] is preserve_terminal
    if preserve_terminal:
        last = table.to_pylist()[-1]
        assert last["ee_pos"][0] == 5
        assert last["actions"][0] == 6
        assert np.all(np.asarray(Image.open(io.BytesIO(last["ee_image"]["bytes"]))) == 100)
        assert last["timestamp"] == pytest.approx(1 / 20)


def test_multitask_export_preserves_labels_and_short_episode_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = []
    prompts = ["press the button", "push the slider to the right", "pull the lever upwards"]
    for task_index, prompt in enumerate(prompts):
        root = tmp_path / f"source_{task_index}"
        roots.append(root)
        source = EXPORT.LeRobotDataset.create(
            repo_id=f"ambench/task_{task_index}",
            root=root,
            robot_type="am_bench",
            fps=120,
            features={
                "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
                "action": {"dtype": "float32", "shape": (8,), "names": None},
                "observation.images.ee_camera": {
                    "dtype": "image",
                    "shape": (8, 8, 3),
                    "names": ["height", "width", "channels"],
                },
            },
            use_videos=False,
            image_writer_threads=1,
            image_writer_processes=0,
        )
        source.meta.info["ambench"] = {
            "action_semantics": EE_ABSOLUTE,
            "state_keys": ["ee_pos", "ee_quat", "gripper_width"],
        }
        write_info(source.meta.info, root)
        # Lengths below the stride and a stride-aligned endpoint exercise both boundary cases.
        for index in range([1, 3, 5][task_index]):
            source.add_frame({
                "observation.state": np.array([index, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
                "action": np.array([index + task_index, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
                "observation.images.ee_camera": np.full((8, 8, 3), index, dtype=np.uint8),
                "task": prompt,
            })
        source.save_episode()
        source.finalize()
    output = tmp_path / "mixed"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--dataset_roots",
            *map(str, roots),
            "--output_root",
            str(output),
            "--target_hz",
            "30",
            "--omit_base_image",
            "--preserve_terminal",
        ],
    )
    EXPORT.main()
    tasks = [json.loads(line) for line in (output / "meta/tasks.jsonl").read_text().splitlines()]
    assert tasks == [{"task_index": i, "task": prompt} for i, prompt in enumerate(prompts)]
    for i, expected_indices in enumerate([[0], [0, 2], [0, 4]]):
        table = pq.read_table(output / f"data/chunk-000/episode_{i:06d}.parquet")
        assert set(table["task_index"].to_pylist()) == {i}
        assert [row[0] for row in table["ee_pos"].to_pylist()] == expected_indices
        assert [row[0] for row in table["actions"].to_pylist()] == [index + i for index in expected_indices]
