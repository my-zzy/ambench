# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Merge terminal-preserving OpenPI v2.1 exports without decoding their images."""

import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def merge_datasets(sources: list[Path], output: Path, episodes_per_task: int) -> dict:
    """Reindex complete episodes, retain image bytes/actions, and report frame mixture weights."""
    if episodes_per_task < 1:
        raise ValueError("episodes_per_task must be positive")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    loaded = []
    labels = []
    for source in sources:
        source = source.resolve()
        info = json.loads((source / "meta/info.json").read_text())
        tasks = [json.loads(line) for line in (source / "meta/tasks.jsonl").read_text().splitlines()]
        if info["codebase_version"] != "v2.1" or info["fps"] != 30:
            raise ValueError(f"Expected a 30 Hz OpenPI v2.1 export: {source}")
        if len(tasks) != 1 or not tasks[0]["task"].strip():
            raise ValueError(f"Each input must contain one nonempty instruction: {source}")
        label = tasks[0]["task"]
        if label in labels:
            raise ValueError(f"Duplicate task source: {label}")
        labels.append(label)
        terminal = info.get("ambench", {}).get("openpi_export", {}).get("preserve_terminal", False)
        # The validated historical PressButton repair predates the generalized exporter flag.
        report_path = source / "ambench_openpi_export_report.json"
        if not terminal and report_path.exists():
            report = json.loads(report_path.read_text())
            terminal = report.get("completed", False) and report.get("all_terminal_pairs_retained", False)
        if not terminal:
            raise ValueError(f"Source has no terminal-preservation evidence: {source}")
        if loaded and info["features"] != loaded[0][1]["features"]:
            raise ValueError(f"Incompatible feature schema: {source}")
        episodes = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines()]
        stats = {
            row["episode_index"]: row["stats"]
            for row in map(json.loads, (source / "meta/episodes_stats.jsonl").read_text().splitlines())
        }
        if len(episodes) < episodes_per_task:
            raise ValueError(f"{source} has {len(episodes)} episodes; need {episodes_per_task}")
        loaded.append((source, info, tasks[0], episodes[:episodes_per_task], stats))
    if len(loaded) < 2:
        raise ValueError("A multi-task mixture needs at least two task sources")
    output.mkdir(parents=True)
    (output / "meta").mkdir()
    episode_records, stats_records, provenance = [], [], []
    frame_counts: collections.Counter[str] = collections.Counter()
    cursor = 0
    for task_index, (source, info, task, episodes, source_stats) in enumerate(loaded):
        for source_episode in episodes:
            old_index = source_episode["episode_index"]
            source_path = source / info["data_path"].format(
                episode_chunk=old_index // info["chunks_size"], episode_index=old_index
            )
            table = pq.read_table(source_path)
            length = len(table)
            if length != source_episode["length"] or not length:
                raise ValueError(f"Invalid episode length: {source_path}")
            if table["frame_index"].to_pylist() != list(range(length)):
                raise ValueError(f"Noncontiguous episode frames: {source_path}")
            if set(table["episode_index"].to_pylist()) != {old_index}:
                raise ValueError(f"Episode index mismatch: {source_path}")
            if set(table["task_index"].to_pylist()) != {task["task_index"]}:
                raise ValueError(f"Task label mismatch: {source_path}")
            episode_index = len(episode_records)
            for key, values in {
                "episode_index": np.full(length, episode_index),
                "index": np.arange(cursor, cursor + length),
                "task_index": np.full(length, task_index),
            }.items():
                table = table.set_column(
                    table.schema.get_field_index(key), key, pa.array(values, type=table.schema.field(key).type)
                )
            stats = copy.deepcopy(source_stats[old_index])
            for key, feature in info["features"].items():
                if feature["dtype"] == "image":
                    continue
                values = np.asarray(table[key].to_pylist(), dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite {key} in {source_path}")
                stats[key] = {
                    name: getattr(np, name)(values, axis=0, keepdims=values.ndim == 1).tolist()
                    for name in ("min", "max", "mean", "std")
                }
                stats[key]["count"] = [length]
            relative = f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination)
            episode_records.append({"episode_index": episode_index, "tasks": [task["task"]], "length": length})
            stats_records.append({"episode_index": episode_index, "stats": stats})
            provenance.append({
                "source": str(source_path),
                "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "output": relative,
                "episode_index": episode_index,
                "task_index": task_index,
                "frames": length,
                "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            })
            cursor += length
            frame_counts[task["task"]] += length
    info = copy.deepcopy(loaded[0][1])
    info.update(
        total_episodes=len(episode_records),
        total_frames=cursor,
        total_tasks=len(labels),
        total_chunks=(len(episode_records) + 999) // 1000,
        chunks_size=1000,
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        splits={"train": f"0:{len(episode_records)}"},
    )
    info["ambench"]["openpi_export"] = {
        "schema": "openpi_am_bench",
        "lerobot_version": "v2.1",
        "target_hz": 30,
        "preserve_terminal": True,
        "source_dataset_roots": [str(item[0]) for item in loaded],
        "terminal_handling": "Preserved complete source episodes; only global task/episode/frame indices changed",
        "include_base_image": "base_image" in info["features"],
    }
    (output / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    for name, rows in {
        "tasks": [{"task_index": i, "task": label} for i, label in enumerate(labels)],
        "episodes": episode_records,
        "episodes_stats": stats_records,
    }.items():
        (output / f"meta/{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = {
        "completed": True,
        "episodes": len(episode_records),
        "frames": cursor,
        "episodes_per_task": episodes_per_task,
        "task_frame_counts": dict(frame_counts),
        "task_sampling_probabilities": {key: value / cursor for key, value in frame_counts.items()},
        "sampling": "Uniform over frames in the pinned OpenPI loader; episode counts are balanced",
        "all_terminal_pairs_retained": True,
        "provenance": provenance,
    }
    (output / "ambench_openpi_export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=100)
    args = parser.parse_args()
    report = merge_datasets(args.sources, args.output, args.episodes_per_task)
    print(json.dumps({key: value for key, value in report.items() if key != "provenance"}, indent=2))


if __name__ == "__main__":
    main()
