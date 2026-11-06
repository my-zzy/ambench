# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded, reproducible recording of a registered AM-Bench scripted expert."""

import argparse
import json
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--attempts", type=int, default=3)
parser.add_argument("--episode_seconds", type=float, help="Explicit supplemental horizon override")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.output.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
summary = {"task": args.task, "status": "starting", "episodes": [], "argv": vars(args) | {"output": str(args.output)}}


def save_summary() -> None:
    summary["wall_seconds"] = time.monotonic() - started
    destination = args.output / "summary.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    temporary.replace(destination)


save_summary()
app = AppLauncher(args).app
env = None
try:
    import importlib

    import gymnasium as gym
    import imageio.v2 as imageio
    import isaaclab.sim as sim_utils
    import isaaclab_tasks  # noqa: F401
    import numpy as np
    import torch
    from isaaclab.sensors import CameraCfg
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    import ambench.tasks  # noqa: F401
    from ambench.recording.paths import write_env_cfg
    from ambench.recording.utils import capture_camera_frames
    from ambench.utils.camera_utils import compute_camera_quat_from_lookat

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    native_horizon = cfg.episode_length_s
    if args.episode_seconds is not None:
        cfg.episode_length_s = args.episode_seconds
    cfg.seed = args.seed
    # Camera-only overrides: physics, policy, controller, and termination stay native.
    cfg.sim.render_interval = 4
    position, target = (-2.0, -3.0, 2.2), (1.0, 0.0, 1.0)
    if args.task.startswith("NDT-"):
        position, target = (-9.0, -12.0, 7.0), (1.0, 0.0, 4.0)
    cfg.scene_camera_cfg = CameraCfg(
        prim_path="/World/envs/env_.*/scene_camera",
        update_period=1 / 30,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=19.6,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=position,
            rot=compute_camera_quat_from_lookat(position, target),
            convention="ros",
        ),
    )
    if cfg.robot_profile.ee_camera is not None:
        cfg.robot_profile.ee_camera.update_period = 1 / 30
    if hasattr(cfg, "num_rerenders_on_reset"):
        cfg.num_rerenders_on_reset = 1
    write_env_cfg(args.output, cfg)
    env = gym.make(args.task, cfg=cfg).unwrapped
    policy_entry = env.spec.kwargs["scripted_policy_entry_point"]
    module_name, class_name = policy_entry.split(":")
    policy_class = getattr(importlib.import_module(module_name), class_name)
    dt = cfg.sim.dt * cfg.decimation
    frame_skip = round((1 / 30) / dt)
    summary.update(
        status="running",
        policy=policy_entry,
        physics_dt=cfg.sim.dt,
        step_dt=dt,
        native_episode_seconds=native_horizon,
        episode_seconds=cfg.episode_length_s,
        video_fps=1 / (dt * frame_skip),
        video_frame_skip=frame_skip,
        inject_noise=False,
        camera_overrides={
            "render_interval": 4,
            "update_period": 1 / 30,
            "scene_position": position,
            "scene_target": target,
        },
    )
    save_summary()

    for attempt in range(args.attempts):
        episode_dir = args.output / f"episode_{attempt:02d}"
        episode_dir.mkdir()
        episode = {
            "attempt": attempt + 1,
            "seed": args.seed + attempt,
            "status": "running",
            "steps": 0,
            "video_frames": 0,
        }
        summary["episodes"].append(episode)
        save_summary()
        episode_started = time.monotonic()
        writers, traces, criteria_ever = {}, {}, {}

        def append_trace(key: str, value) -> None:
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            array = np.asarray(value)
            if array.dtype.kind in "biuf":
                traces.setdefault(key, []).append(array.copy())

        try:
            with torch.inference_mode():
                obs, _ = env.reset(seed=args.seed + attempt)
                policy = policy_class(env=env, inject_noise=False)
            cameras = [name for name in ("scene_camera", "ee_camera") if name in env.scene.sensors]
            for name in cameras:
                writers[name] = imageio.get_writer(
                    str(episode_dir / f"{name}.mp4"),
                    fps=summary["video_fps"],
                    codec="libx264",
                    quality=8,
                    macro_block_size=1,
                    ffmpeg_params=["-pix_fmt", "yuv420p", "-threads", "2"],
                )
            with torch.inference_mode():
                for step in range(env.max_episode_length + 1):
                    if not app.is_running():
                        raise RuntimeError("Simulator stopped before episode termination")
                    if step % frame_skip == 0:
                        frames = capture_camera_frames(env, 0, cameras)
                        for name, frame in frames.items():
                            writers[name].append_data(frame)
                        episode["video_frames"] += 1
                    action = policy.advance(obs, 0)
                    if not torch.isfinite(action).all():
                        raise RuntimeError("Non-finite expert action")
                    for key, value in obs["policy"][0].items():
                        if value is not None:
                            append_trace(f"obs__{key}", value)
                    append_trace("action", action[0])
                    append_trace("time_seconds", step * dt)
                    obs, reward, terminated, truncated, info = env.step(action)
                    append_trace("reward", reward[0])
                    append_trace("terminated", terminated[0])
                    append_trace("truncated", truncated[0])
                    criteria = {key: bool(value[0].item()) for key, value in info.get("success_criteria", {}).items()}
                    for key, value in criteria.items():
                        append_trace(f"criterion__{key}", value)
                        criteria_ever[key] = criteria_ever.get(key, False) or value
                    episode["steps"] = step + 1
                    episode["criteria_final"] = criteria
                    episode["criteria_ever"] = criteria_ever
                    if step % 240 == 0:
                        save_summary()
                        print(
                            f"PROGRESS {args.task} attempt={attempt + 1} step={step} sim_s={step * dt:.2f}", flush=True
                        )
                    if bool(terminated[0]) or bool(truncated[0]):
                        episode.update(
                            status="success" if bool(terminated[0]) else "timeout",
                            success=bool(terminated[0]),
                            terminated=bool(terminated[0]),
                            truncated=bool(truncated[0]),
                        )
                        break
                else:
                    raise RuntimeError("Native episode limit exceeded without a done signal")
        except BaseException as error:
            episode.update(status="runtime_error", success=False, error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            for writer in writers.values():
                writer.close()
            arrays = {key: np.stack(values) for key, values in traces.items() if values}
            if arrays:
                np.savez_compressed(episode_dir / "trajectory.npz", **arrays)
            episode.update(wall_seconds=time.monotonic() - episode_started, simulation_seconds=episode["steps"] * dt)
            episode["return"] = float(np.sum(arrays.get("reward", [])))
            episode["nonfinite_trace_keys"] = [key for key, values in arrays.items() if not np.isfinite(values).all()]
            if "obs__ee_pos" in arrays and "action" in arrays and arrays["action"].shape[-1] >= 3:
                distances = np.linalg.norm(arrays["obs__ee_pos"] - arrays["action"][:, :3], axis=-1)
                episode["ee_target_distance_mean_m"] = float(distances.mean())
            (episode_dir / "metrics.json").write_text(json.dumps(episode, indent=2) + "\n")
            save_summary()
            print("EPISODE_RESULT " + json.dumps(episode), flush=True)
        if episode["success"]:
            break
    summary["status"] = "success" if any(episode.get("success") for episode in summary["episodes"]) else "no_success"
except BaseException as error:
    summary.update(status="runtime_error", error=repr(error), traceback=traceback.format_exc())
    traceback.print_exc()
finally:
    save_summary()
    if env is not None:
        env.close()
    app.close()
