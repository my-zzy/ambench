# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Collect fresh expert episodes, or replay exported commands using the PI execution adapter."""

import argparse
import collections
import importlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--reference-dir", type=Path, help="Directory containing heldout_export and converted records")
parser.add_argument("--mode", choices=["collect", "replay"], required=True)
parser.add_argument(
    "--replay-variant", choices=["native_120hz", "raw_hold_30hz", "terminal_preserved_30hz", "corrected_30hz"]
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
ROOT = args.output_dir.resolve()
REFERENCE = args.reference_dir.resolve() if args.reference_dir else ROOT
ROOT.mkdir(parents=True, exist_ok=True)
args.enable_cameras = True
app = AppLauncher(args).app

import av
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg

import ambench.tasks  # noqa: F401
from ambench.recording.paths import write_env_cfg
from ambench.utils.camera_utils import get_camera_rgb
from ambench.utils.image_processing import prepare_rgb_image
from ambench_learn.eval.common import configure_env_cfg, make_scene_camera_cfg
from ambench_learn.policies.pi.eval_utils import (
    default_action_for_env,
    expand_policy_actions,
)

TASK = "PressButton-Am-FAHexa-Abs-PID-Direct-v0"
SEEDS = [20000, 20004]


def state_array(obs: dict, index: int) -> np.ndarray:
    """Return the same measured state prefix used by the OpenPI interface."""
    item = obs["policy"][index]
    return np.concatenate(
        [item[key].detach().cpu().numpy().reshape(-1) for key in ("ee_pos", "ee_quat", "gripper_width")]
    ).astype(np.float32)


def camera_array(env, camera: str, index: int) -> np.ndarray:
    return np.asarray(prepare_rgb_image(get_camera_rgb(env, camera, index)), dtype=np.uint8).copy()


def replay_actions(item: dict, variant: str, episode: int) -> np.ndarray:
    """Select the native, held, terminal-preserved, or adapter-roundtrip control."""
    if variant == "native_120hz":
        return item["actions_120hz"]
    if variant == "corrected_30hz":
        native = item["actions_120hz"]
        indices = list(range(0, len(native), 4))
        if indices[-1] != len(native) - 1:
            indices.append(len(native) - 1)
        return native[indices]
    if variant == "terminal_preserved_30hz":
        return np.concatenate([item["actions_30hz"], item["actions_120hz"][-1:]], axis=0)
    if variant in ("raw_30hz", "raw_hold_30hz"):
        return item["actions_30hz"]
    return np.load(REFERENCE / f"converted/episode_{episode:03d}.npz")["actions_30hz"]


try:
    scene = make_scene_camera_cfg()
    scene.width, scene.height = 640, 360
    cfg = configure_env_cfg(
        parse_env_cfg=parse_env_cfg, task=TASK, device=args.device, num_envs=4, seed=20000, scene_camera_cfg=scene
    )
    env = gym.make(TASK, cfg=cfg).unwrapped
    assert abs(env.dt - 1 / 120) < 1e-9
    write_env_cfg(ROOT / args.mode, cfg)
    module, name = env.spec.kwargs["scripted_policy_entry_point"].split(":")
    policy_class = getattr(importlib.import_module(module), name)
    variants = ["expert"] if args.mode == "collect" else ["raw_30hz", "roundtrip_30hz"]
    if args.replay_variant:
        variants = [args.replay_variant]
    results = []
    with torch.inference_mode():
        for variant in variants:
            for batch_index, seed in enumerate(SEEDS):
                obs, _ = env.reset(seed=seed)
                initial_states = np.stack([state_array(obs, i) for i in range(4)])
                initial_buttons = env.button.data.root_pose_w.detach().cpu().numpy().copy()
                initial_walls = env.scene["wall"].data.root_pose_w.detach().cpu().numpy().copy()
                done, successes, reasons = [False] * 4, [False] * 4, ["timeout"] * 4
                steps = [0] * 4
                max_button_press = np.zeros(4)
                states, commands, images = [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)]
                plans = [collections.deque() for _ in range(4)]
                policies = (
                    [policy_class(env=env, inject_noise=False) for _ in range(4)] if args.mode == "collect" else None
                )
                data = []
                if args.mode == "replay":
                    for i in range(4):
                        episode = batch_index * 4 + i
                        item = np.load(REFERENCE / f"heldout_export/episode_{episode:03d}.npz")
                        assert np.allclose(initial_states[i], item["initial_state"], atol=2e-5)
                        assert np.allclose(initial_buttons[i], item["initial_button_pose"], atol=2e-5)
                        assert np.allclose(initial_walls[i], item["initial_wall_pose"], atol=2e-5)
                        data.append(replay_actions(item, variant, episode))
                video_writers = {}
                folder = ROOT / "videos" / variant
                folder.mkdir(parents=True, exist_ok=True)
                for i in range(4):
                    for camera in ("ee_camera", "scene_camera"):
                        path = folder / f"episode_{batch_index * 4 + i:03d}_{camera}.mp4"
                        container = av.open(str(path), "w")
                        stream = container.add_stream("libx264", rate=30)
                        stream.width, stream.height = (384, 384) if camera == "ee_camera" else (640, 360)
                        stream.pix_fmt = "yuv420p"
                        stream.options = {"crf": "18", "preset": "fast"}
                        video_writers[i, camera] = container, stream
                try:
                    for step in range(env.max_episode_length):
                        if all(done):
                            break
                        if not app.is_running():
                            raise RuntimeError("Simulator stopped before bounded diagnostic completed")
                        if step % 4 == 0:
                            for i in range(4):
                                if done[i]:
                                    continue
                                for camera in ("ee_camera", "scene_camera"):
                                    frame = camera_array(env, camera, i)
                                    if args.mode == "collect" and camera == "ee_camera":
                                        images[i].append(frame)
                                    container, stream = video_writers[i, camera]
                                    for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                                        container.mux(packet)
                        action_rows = []
                        for i in range(4):
                            if done[i]:
                                command = default_action_for_env(env, action_semantics="ee_absolute", env_index=i)
                            elif args.mode == "collect":
                                command = policies[i].advance(obs, i).reshape(8)
                                states[i].append(state_array(obs, i))
                                commands[i].append(command.detach().cpu().numpy().copy())
                            elif variant in ("native_120hz", "raw_hold_30hz"):
                                index = step if variant == "native_120hz" else step // 4
                                command = torch.from_numpy(data[i][min(index, len(data[i]) - 1)]).to(env.device)
                            else:
                                if not plans[i]:
                                    start = step // 4
                                    indices = np.minimum(np.arange(start, start + 30), len(data[i]) - 1)
                                    plans[i] = expand_policy_actions(
                                        data[i][indices],
                                        env,
                                        action_semantics="ee_absolute",
                                        n_action_steps=8,
                                        action_execution_stride=4,
                                        env_index=i,
                                    )
                                command = plans[i].popleft()
                            action_rows.append(command)
                        obs, _, terminated, truncated, _ = env.step(torch.stack(action_rows))
                        for i in range(4):
                            if done[i]:
                                continue
                            steps[i] += 1
                            max_button_press[i] = max(
                                max_button_press[i], float(env.button.data.joint_pos[i, env.button_joint_idx])
                            )
                            if bool(terminated[i]) or bool(truncated[i]):
                                successes[i] = bool(terminated[i])
                                reasons[i] = "success" if successes[i] else "timeout"
                                done[i] = True
                        if step % 600 == 0:
                            print(f"{variant} seed={seed} step={step} active={sum(not x for x in done)}", flush=True)
                finally:
                    for container, stream in video_writers.values():
                        for packet in stream.encode():
                            container.mux(packet)
                        container.close()
                for i in range(4):
                    episode = batch_index * 4 + i
                    record = {
                        "variant": variant,
                        "episode": episode,
                        "seed": seed,
                        "env_index": i,
                        "success": successes[i],
                        "steps": steps[i],
                        "reason": reasons[i],
                        "max_observed_button_displacement_m": float(max_button_press[i]),
                    }
                    results.append(record)
                    if args.mode == "collect":
                        assert successes[i], f"Fresh expert failed: {record}"
                        state = np.stack(states[i])[::4]
                        action = np.stack(commands[i])[::4]
                        assert len(images[i]) == len(state)
                        sample_indices = np.linspace(0, len(state) - 30, 12).round().astype(int)
                        directory = ROOT / "heldout"
                        directory.mkdir(exist_ok=True)
                        np.savez_compressed(
                            directory / f"episode_{episode:03d}.npz",
                            states_30hz=state,
                            actions_30hz=action,
                            actions_120hz=np.stack(commands[i]),
                            sample_indices=sample_indices,
                            sample_images=np.stack([images[i][j] for j in sample_indices]),
                            initial_state=initial_states[i],
                            initial_button_pose=initial_buttons[i],
                            initial_wall_pose=initial_walls[i],
                            seed=seed,
                            env_index=i,
                        )
                    print(record, flush=True)
                suffix = f"_{args.replay_variant}" if args.replay_variant else ""
                (ROOT / f"{args.mode}{suffix}_results.json").write_text(json.dumps(results, indent=2) + "\n")
    env.close()
finally:
    app.close()
