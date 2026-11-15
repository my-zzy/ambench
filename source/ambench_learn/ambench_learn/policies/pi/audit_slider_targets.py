# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Audit slider exports, relative transforms, and frozen checkpoint contact predictions."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROMPT = "push the slider to the right"


def distribution(values: np.ndarray) -> dict[str, float]:
    """Summarize signed values without treating overlapping samples as independent trials."""
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite measurements")
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def read_states(table: pa.Table) -> np.ndarray:
    """Read the declared eight-dimensional measured EE observation."""
    return np.concatenate(
        [
            np.asarray(table[key].to_pylist(), dtype=np.float32).reshape(len(table), -1)
            for key in ("ee_pos", "ee_quat", "gripper_width")
        ],
        axis=-1,
    )


def summarize_predictions(rows: list[dict]) -> dict:
    """Separate approach, pushing, and off-trajectory observations."""
    groups = {}
    for group in ("training_approach", "training_push", "policy_handoff"):
        chosen = [
            row
            for row in rows
            if (
                row["subset"] == "policy_handoff"
                if group == "policy_handoff"
                else row["subset"] == "training" and (row["frame"] < 210) == (group == "training_approach")
            )
        ]
        if chosen:
            groups[group] = {
                "draws": len(chosen),
                **{
                    key: distribution(np.array([row[key] for row in chosen]))
                    for key in chosen[0]
                    if key.startswith(("outward_", "height_", "position_", "gripper_", "anchor_"))
                },
            }
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/openpi"))
    parser.add_argument("--repo-id", default="my-zzy/ambench_contact3_30hz_v1")
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--slider-export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--handoff-dir", type=Path, help="Prior replay diagnostic's server_port30 directory")
    args = parser.parse_args()
    root = args.work_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["HF_LEROBOT_HOME"] = str(root / "datasets")
    os.environ["HF_HOME"] = str(root / "cache/huggingface")
    os.environ["OPENPI_DATA_HOME"] = str(root / "cache/openpi")
    os.environ["JAX_PLATFORMS"] = "cuda" if args.checkpoint else "cpu"

    from openpi import transforms
    from openpi.policies import am_bench_policy as adapter
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.training import data_loader
    from scipy.spatial.transform import Rotation

    from ambench_learn.policies.pi.config import make_config
    from ambench_learn.policies.pi.stable_stats import action_roundtrip_errors

    cfg = make_config(work_dir=root, repo_id=args.repo_id, multitask=True, finetune_mode="full")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset_root = root / "datasets" / args.repo_id
    episodes = [json.loads(line) for line in (dataset_root / "meta/episodes.jsonl").read_text().splitlines()]
    slider_episodes = [ep["episode_index"] for ep in episodes if ep["tasks"] == [PROMPT]]
    if len(slider_episodes) != 100:
        raise ValueError("This protocol expects the original 100 slider training episodes")
    selected = [slider_episodes[i] for i in np.linspace(0, 99, 12).round().astype(int)]
    plan = {
        "training_episodes": slider_episodes,
        "inference_episodes": selected,
        "inference_frames": "frame 180, then five evenly spaced frames from 210 (7 s) to the final frame",
        "draws": 3,
        "noise_seed": "[20261112, episode, frame, draw]",
        "handoffs": "All eight prior untreated control handoffs, seeds 72000/72004",
        "scope": "Offline diagnostic; training observations are not held-out evaluation",
    }
    (output / "PLAN.json").write_text(json.dumps(plan, indent=2) + "\n")
    norm_path = root / "assets/pi0_multitask_lora_30hz" / args.repo_id / "norm_stats.json"
    stats = normalize.load(norm_path.parent)
    if args.checkpoint:
        checkpoint_norm = args.checkpoint / "assets" / args.repo_id / "norm_stats.json"
        if norm_path.read_bytes() != checkpoint_norm.read_bytes():
            raise ValueError("Training and serving normalization differ")
    norm = transforms.Normalize(stats, use_quantiles=data_cfg.use_quantile_norm)
    unnorm = transforms.Unnormalize({"actions": stats["actions"]}, use_quantiles=data_cfg.use_quantile_norm)
    raw_info = json.loads((args.raw_dataset / "meta/info.json").read_text())
    if raw_info["fps"] != 120:
        raise ValueError("Expected the native 120 Hz collection")
    raw = pa.concat_tables([
        pq.read_table(p, columns=["episode_index", "frame_index", "observation.state", "action"])
        for p in sorted((args.raw_dataset / "data").rglob("*.parquet"))
    ])
    raw_ids = np.asarray(raw["episode_index"])
    loader = data_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)
    cursors = np.cumsum([0] + [ep["length"] for ep in episodes])
    per_episode, observations, measured_depth, relative_depth = [], [], [], []
    errors = {key: 0.0 for key in ("position_m", "unit_quaternion_component", "gripper")}
    checks = 0
    independent_position_error = 0.0
    for source_episode, episode in enumerate(slider_episodes):
        path = dataset_root / f"data/chunk-000/episode_{episode:06d}.parquet"
        columns = ["ee_pos", "ee_quat", "gripper_width", "actions", "timestamp", "task_index"]
        table = pq.read_table(path, columns=columns)
        states, actions = read_states(table), np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        original = raw.filter(pa.array(raw_ids == source_episode))
        order = np.argsort(np.asarray(original["frame_index"]))
        original = original.take(pa.array(order))
        indices = list(range(0, len(original), 4))
        if indices[-1] != len(original) - 1:
            indices.append(len(original) - 1)
        expected_state = np.asarray(original["observation.state"].to_pylist(), dtype=np.float32)[indices]
        expected_action = np.asarray(original["action"].to_pylist(), dtype=np.float32)[indices]
        if not np.array_equal(states, expected_state) or not np.array_equal(actions, expected_action):
            raise ValueError(f"Native export mismatch in episode {episode}")
        single = pq.read_table(
            args.slider_export / f"data/chunk-000/episode_{source_episode:06d}.parquet", columns=columns[:-1]
        )
        if not np.array_equal(read_states(single), states) or not np.array_equal(
            single["actions"].to_pylist(), actions
        ):
            raise ValueError(f"Mixture changed targets in episode {episode}")
        if np.max(np.abs(np.asarray(table["timestamp"]) - np.arange(len(table)) / 30)) > 1e-6:
            raise ValueError("Incorrect policy timestamps")

        # The scripted grasp aligns tool +X to wall +X; validate the source pose and slide direction.
        normal = adapter._rotate_vector(actions[-1, 3:7], np.array([1, 0, 0], dtype=np.float32))
        if np.max(np.abs(normal - [1, 0, 0])) > 1e-6:
            raise ValueError("This collection's wall-axis assumptions changed")
        outward = -normal
        push = actions[210:]
        plane = push[0, :3]
        command_depth = (push[:, :3] - plane) @ outward
        depth = (states[210:, :3] - plane) @ outward
        measured_depth.extend(depth.tolist())
        horizon = actions[np.minimum(np.arange(len(actions))[:, None] + np.arange(30), len(actions) - 1)]
        relative = adapter._to_ee_local_relative(horizon, states)
        rotations = Rotation.from_quat(states[:, [4, 5, 6, 3]]).as_matrix()
        independent_local = np.einsum("nti,nij->ntj", horizon[:, :, :3] - states[:, None, :3], rotations)
        independent_position_error = max(
            independent_position_error, float(np.max(np.abs(relative[:, :, :3] - independent_local)))
        )
        decoded = adapter._to_ee_local_absolute(unnorm(norm({"actions": relative}))["actions"], states)
        for key, value in action_roundtrip_errors(decoded, horizon).items():
            if key in errors:
                errors[key] = max(errors[key], value)
        relative_depth.extend(((horizon[210:, :, :3] - states[210:, None, :3]) @ outward).reshape(-1).tolist())
        frames = [180, *np.linspace(210, len(actions) - 1, 5).round().astype(int).tolist()]
        for frame in sorted(set([0, 210, len(actions) - 30, len(actions) - 1, *frames])):
            sample = transforms.compose(data_cfg.repack_transforms.inputs)(loader[int(cursors[episode]) + frame])
            if sample["prompt"] != PROMPT or not np.array_equal(sample["actions"], horizon[frame]):
                raise ValueError(f"Loader horizon/prompt mismatch: {episode}/{frame}")
            mapped = transforms.compose(data_cfg.data_transforms.inputs)(sample)
            padded = transforms.PadStatesAndActions(32)(norm(mapped))
            result = transforms.compose(data_cfg.data_transforms.outputs)(unnorm({"actions": padded["actions"]}))
            for key, value in action_roundtrip_errors(result["actions"], horizon[frame]).items():
                if key in errors:
                    errors[key] = max(errors[key], value)
            if not np.array_equal(mapped["state"][:7], [0, 0, 0, 1, 0, 0, 0]):
                raise ValueError("Unexpected model state localization")
            checks += 1
            if episode in selected and frame in frames:
                sample.pop("actions")
                observations.append((episode, frame, sample, horizon[frame], outward, states[frame], "training"))
        per_episode.append({
            "episode": episode,
            "frames": len(actions),
            "native_frames": len(original),
            "native_file_frame_indices_contiguous": bool(
                np.array_equal(original["frame_index"], np.arange(len(original)))
            ),
            "command_depth_range_m": float(np.ptp(command_depth)),
            "command_height_range_m": float(np.ptp(push[:, 2])),
            "push_distance_m": float(plane[1] - push[-1, 1]),
            "measured_outward_from_expert_plane_m": distribution(depth),
            "measured_gripper_width_m": distribution(states[210:, 7]),
            "target_gripper_min": float(push[:, 7].min()),
            "target_gripper_max": float(push[:, 7].max()),
        })
        print(f"Audited episode {episode}: {len(actions)} frames", flush=True)
    if max(*errors.values(), independent_position_error) > 2e-6 or not all(
        ep["native_file_frame_indices_contiguous"] for ep in per_episode
    ):
        raise ValueError("Transform round trip or native sequence validation failed")
    report = {
        "passed": True,
        "episodes": 100,
        "frames": sum(ep["frames"] for ep in per_episode),
        "native_frames": len(raw),
        "actual_loader_public_transform_checks": checks,
        "roundtrip_errors": errors,
        "independent_scipy_position_encode_max_error_m": independent_position_error,
        "normalization_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        "use_quantile_norm": data_cfg.use_quantile_norm,
        "measured_push_outward_from_expert_plane_m": distribution(np.asarray(measured_depth)),
        "target_outward_displacement_from_measured_anchor_m": distribution(np.asarray(relative_depth)),
        "maximum_command_depth_range_m": max(ep["command_depth_range_m"] for ep in per_episode),
        "maximum_command_height_range_m": max(ep["command_height_range_m"] for ep in per_episode),
        "per_episode": per_episode,
    }
    (output / "target_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if not args.checkpoint:
        return

    if args.handoff_dir:
        for seed in (72000, 72004):
            batch = args.handoff_dir / f"control_{seed}"
            metadata = json.loads((batch / "metadata.json").read_text())
            for env in range(4):
                data = np.load(batch / f"handoff_observation_env_{env}.npz")
                sample = {key: data[key] for key in data.files}
                sample["prompt"] = str(sample["prompt"].item())
                state = np.concatenate([sample[f"am_bench/{key}"] for key in ("ee_pos", "ee_quat", "gripper_width")])
                normal = np.asarray(metadata["wall_normals"][env], dtype=np.float32)
                step = metadata["prefix_schedule"][env]["resume_step"]
                trace = np.load(batch / f"trace_env_{env}.npz")
                # Diagnostic traces and policy requests already use environment-relative positions.
                slider = trace["slider_body_pos"][step - 1]
                # Reference plane only: this is not a desired slide-axis trajectory or supervised recovery label.
                target = np.tile(np.r_[slider - normal * 0.23, state[3:7], -1.0], (30, 1)).astype(np.float32)
                observations.append((seed, env, sample, target, -normal, state, "policy_handoff"))

    keys = ["am_bench/ee_pos", "am_bench/ee_quat", "am_bench/gripper_width", "am_bench/ee_image", "prompt"]
    policy = policy_config.create_trained_policy(
        cfg,
        args.checkpoint.resolve(),
        repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({key: key for key in keys})]),
    )
    predictions, targets, rows = [], [], []
    for episode, frame, sample, target, outward, state, subset in observations:
        sample = {key: np.asarray(value) if key != "prompt" else value for key, value in sample.items()}
        for draw in range(3):
            noise = np.random.default_rng([20261112, episode, frame, draw]).standard_normal((30, 32)).astype(np.float32)
            started = time.monotonic()
            prediction = np.asarray(policy.infer(sample, noise=noise)["actions"], dtype=np.float32)
            if prediction.shape != (30, 8) or not np.isfinite(prediction).all():
                raise ValueError("Invalid checkpoint prediction")
            row = {
                "subset": subset,
                "episode": episode,
                "frame": frame,
                "draw": draw,
                "seconds": time.monotonic() - started,
                "noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
                "image_sha256": hashlib.sha256(np.asarray(sample["am_bench/ee_image"]).tobytes()).hexdigest(),
                "state": state.tolist(),
                "anchor_outward_from_target_m": float((state[:3] - target[0, :3]) @ outward),
            }
            for prefix in (1, 8, 30):
                delta = prediction[:prefix, :3] - target[:prefix, :3]
                row[f"outward_error_m_{prefix}"] = float(np.mean(delta @ outward))
                row[f"outward_step_m_{prefix}"] = float(np.mean((prediction[:prefix, :3] - state[:3]) @ outward))
                row[f"height_error_m_{prefix}"] = float(delta[:, 2].mean())
                if subset == "training":
                    row[f"position_error_m_{prefix}"] = float(np.linalg.norm(delta, axis=-1).mean())
                    row[f"gripper_error_{prefix}"] = float(np.abs(prediction[:prefix, 7] - target[:prefix, 7]).mean())
            rows.append(row)
            predictions.append(prediction)
            targets.append(target)
            with (output / "prediction_rows.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        print(f"Predicted {subset} {episode}/{frame}: {len(rows)} draws", flush=True)
    np.savez_compressed(output / "predictions.npz", predictions=np.stack(predictions), targets=np.stack(targets))
    summary = {
        "checkpoint": str(args.checkpoint),
        "draws": len(rows),
        "observations": len(observations),
        "groups": summarize_predictions(rows),
    }
    (output / "prediction_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"completed": True, "frames_audited": report["frames"], "predictions": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
