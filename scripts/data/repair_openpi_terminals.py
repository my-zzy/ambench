# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Preserve complete 30 Hz expert endings in a separate v2.1 export."""

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def repair_export(source: Path, original: Path, output: Path) -> dict:
    """Repair a wrist-only 120-to-30 Hz export while preserving every old sample.

    Inputs are canonical one-episode-per-parquet EE records and their original
    floor-resampled OpenPI v2.1 export. The destination must not exist.
    """
    SOURCE, ORIGINAL, OUTPUT = source.resolve(), original.resolve(), output.resolve()
    source_info = json.loads((SOURCE / "meta/info.json").read_text())
    info = json.loads((ORIGINAL / "meta/info.json").read_text())
    if source_info["fps"] != 120 or info["fps"] != 30:
        raise ValueError("This reconstruction supports the measured 120-to-30 Hz protocol only")
    if source_info["ambench"]["state_keys"] != ["ee_pos", "ee_quat", "gripper_width"]:
        raise ValueError("Expected the canonical 8D EE state prefix")
    if source_info["ambench"]["action_semantics"] != "ee_absolute":
        raise ValueError("Expected absolute EE commands")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "meta").mkdir()
    (OUTPUT / "data/chunk-000").mkdir(parents=True)
    info = json.loads((ORIGINAL / "meta/info.json").read_text())
    old_stats = [
        json.loads(line)["stats"] for line in (ORIGINAL / "meta/episodes_stats.jsonl").read_text().splitlines()
    ]
    old_episodes = [json.loads(line) for line in (ORIGINAL / "meta/episodes.jsonl").read_text().splitlines()]
    episodes, episode_stats, provenance = [], [], []
    cursor = 0
    source_paths = sorted((SOURCE / "data").rglob("*.parquet"))
    assert len(source_paths) == info["total_episodes"], "Expected one source parquet per episode"
    for episode, source_path in enumerate(source_paths):
        raw = pq.read_table(source_path)
        assert set(raw["episode_index"].to_pylist()) == {episode}
        assert raw["frame_index"].to_pylist() == list(range(len(raw)))
        original_path = ORIGINAL / f"data/chunk-000/episode_{episode:06d}.parquet"
        old = pq.read_table(original_path)
        native_indices = list(range(0, len(raw), 4))
        if native_indices[-1] != len(raw) - 1:
            native_indices.append(len(raw) - 1)
        assert len(old) == len(raw) // 4
        rows = old.to_pylist()
        for native_index in native_indices[len(old) :]:
            state = raw["observation.state"][native_index].as_py()
            rows.append({
                "ee_image": raw["observation.images.ee_camera"][native_index].as_py(),
                "actions": raw["action"][native_index].as_py(),
                "ee_pos": state[:3],
                "ee_quat": state[3:7],
                "gripper_width": state[7],
                "timestamp": 0.0,
                "frame_index": 0,
                "episode_index": episode,
                "index": 0,
                "task_index": 0,
            })
        for index, row in enumerate(rows):
            row.update(timestamp=index / 30, frame_index=index, index=cursor + index)
        result = pa.Table.from_pylist(rows, schema=old.schema)
        assert np.array_equal(
            np.array(result["actions"].to_pylist()), np.array(raw["action"].take(native_indices).to_pylist())
        )
        state = np.asarray(raw["observation.state"].take(native_indices).to_pylist(), dtype=np.float32)
        actual_state = np.concatenate(
            [
                np.asarray(result[key].to_pylist()).reshape(len(result), -1)
                for key in ("ee_pos", "ee_quat", "gripper_width")
            ],
            axis=-1,
        )
        assert np.array_equal(state, actual_state)
        for key in ("actions", "ee_image", "ee_pos", "ee_quat", "gripper_width"):
            assert result[key][: len(old)].equals(old[key]), key
        assert result["ee_image"][-1].as_py() == raw["observation.images.ee_camera"][-1].as_py()
        for row in rows[len(old) :]:
            assert np.asarray(Image.open(io.BytesIO(row["ee_image"]["bytes"]))).shape == (384, 384, 3)
        destination = OUTPUT / f"data/chunk-000/episode_{episode:06d}.parquet"
        pq.write_table(result, destination)

        # Combine the existing image statistics with only the new images. Each frame has equal pixel count.
        stats = copy.deepcopy(old_stats[episode])
        images = (
            np.stack(
                [np.asarray(Image.open(io.BytesIO(r["ee_image"]["bytes"])), dtype=np.float64) for r in rows[len(old) :]]
            )
            / 255
        )
        added = len(images)
        mean0 = np.asarray(stats["ee_image"]["mean"]).reshape(3)
        var0 = np.asarray(stats["ee_image"]["std"]).reshape(3) ** 2
        mean1, var1 = images.mean(axis=(0, 1, 2)), images.var(axis=(0, 1, 2))
        delta = mean1 - mean0
        image_mean = mean0 + delta * added / len(rows)
        image_var = (len(old) * var0 + added * var1 + delta**2 * len(old) * added / len(rows)) / len(rows)
        stats["ee_image"] = {
            "mean": image_mean.reshape(3, 1, 1).tolist(),
            "std": np.sqrt(image_var).reshape(3, 1, 1).tolist(),
            "min": (
                np.minimum(np.asarray(stats["ee_image"]["min"]).reshape(3), images.min(axis=(0, 1, 2)))
                .reshape(3, 1, 1)
                .tolist()
            ),
            "max": (
                np.maximum(np.asarray(stats["ee_image"]["max"]).reshape(3), images.max(axis=(0, 1, 2)))
                .reshape(3, 1, 1)
                .tolist()
            ),
            "count": [len(rows)],
        }
        for key in result.column_names:
            if key == "ee_image":
                continue
            values = np.asarray(result[key].to_pylist(), dtype=np.float64)
            keepdims = values.ndim == 1
            stats[key] = {
                name: getattr(np, name)(values, axis=0, keepdims=keepdims).tolist()
                for name in ("min", "max", "mean", "std")
            }
            stats[key]["count"] = [len(rows)]
        episodes.append({**old_episodes[episode], "length": len(rows)})
        episode_stats.append({"episode_index": episode, "stats": stats})
        provenance.append({
            "episode": episode,
            "native_length": len(raw),
            "old_length": len(old),
            "new_length": len(rows),
            "source_frame_indices": native_indices,
            "final_source_timestamp_s": (len(raw) - 1) / 120,
            "final_grid_timestamp_s": (len(rows) - 1) / 30,
            "endpoint_hold_seconds": (len(rows) - 1) / 30 - (len(raw) - 1) / 120,
            "old_parquet_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
        })
        cursor += len(rows)
        print(f"episode {episode}: {len(old)} -> {len(rows)}; terminal state/image/action retained", flush=True)
    info["total_frames"] = cursor
    info["ambench"]["openpi_export"].update(
        terminal_handling=(
            "Retain every fourth native sample; clamp final 30 Hz grid point to the last native observation/action"
            " pair."
        ),
        source_dataset_roots=[str(SOURCE)],
    )
    (OUTPUT / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    (OUTPUT / "meta/tasks.jsonl").write_bytes((ORIGINAL / "meta/tasks.jsonl").read_bytes())
    (OUTPUT / "meta/episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
    (OUTPUT / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episode_stats))
    report = {
        "completed": True,
        "episodes": len(episodes),
        "old_frames": sum(row["old_length"] for row in provenance),
        "new_frames": cursor,
        "all_original_samples_preserved": True,
        "all_terminal_pairs_retained": True,
        "max_endpoint_hold_seconds": max(row["endpoint_hold_seconds"] for row in provenance),
        "source": str(SOURCE),
        "output": str(OUTPUT),
        "timing": (
            "Uniform 30 Hz grid; only the final grid point may hold the paired last native state/image/action for up to"
            " 25 ms. No state/image from a reset episode is used."
        ),
        "episodes_detail": provenance,
    }
    (OUTPUT / "ambench_openpi_export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = repair_export(args.source, args.original, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "episodes_detail"}, indent=2))


if __name__ == "__main__":
    main()
