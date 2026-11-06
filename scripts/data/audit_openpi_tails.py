# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure terminal command loss in the exact original training export."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--export", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source, export = args.source.resolve(), args.export.resolve()
rows = []
for path in sorted((source / "data").rglob("*.parquet")):
    table = pq.read_table(path, columns=["action", "episode_index", "frame_index"])
    for episode in sorted(set(table["episode_index"].to_pylist())):
        item = table.filter(table["episode_index"].to_numpy() == episode)
        indices = np.argsort(item["frame_index"].to_numpy())
        native = np.asarray(item["action"].to_pylist(), dtype=np.float32)[indices]
        target = pq.read_table(export / f"data/chunk-000/episode_{episode:06d}.parquet", columns=["actions"])
        actions = np.asarray(target["actions"].to_pylist(), dtype=np.float32)
        assert len(actions) == len(native) // 4
        assert np.allclose(actions, native[::4][: len(actions)], atol=1e-7)
        rows.append({
            "episode": episode,
            "native_commands": len(native),
            "exported_commands": len(actions),
            "omitted_tail_ticks": len(native) - 1 - 4 * (len(actions) - 1),
            "terminal_position_difference_m": float(np.linalg.norm(native[-1, :3] - actions[-1, :3])),
        })
assert rows, "No input episodes"
distance = np.array([row["terminal_position_difference_m"] for row in rows])
report = {
    "source": str(source),
    "episodes": len(rows),
    "all_exported_actions_match_floor_stride4": True,
    "terminal_position_difference_m": {
        "min": float(distance.min()),
        "median": float(np.median(distance)),
        "max": float(distance.max()),
    },
    "episodes_with_terminal_difference_over_4mm": int((distance > 0.004).sum()),
    "details": rows,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
