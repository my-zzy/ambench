# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare a scripted expert and served PI policy with pre-reset slider measurements.

Run each mode separately with the same seed and number of environments. This
diagnostic adds contact reporting, but preserves task dynamics and termination.
Privileged measurements are never added to PI inputs. Optional height/depth
holds use slider progress to latch commands after engagement. Replaying a common
approach prefix controls differences before the intervention.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("expert", "policy"), required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8042)
parser.add_argument("--policy-id", default="contact3_full_20k_v1/19999")
parser.add_argument("--height-hold", action="store_true", help="Diagnostic only: hold commanded Z after 5% travel.")
parser.add_argument("--depth-hold", action="store_true", help="Diagnostic only: hold wall-normal separation.")
parser.add_argument("--replay-prefix-dir", type=Path, help="Prior untreated diagnostic batch with matching scenes.")
parser.add_argument("--paired-noise", action="store_true", help="Requires the diagnostic paired-noise policy server.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1:
    parser.error("--num-envs must be positive")
if (args.height_hold or args.depth_hold or args.paired_noise or args.replay_prefix_dir) and args.mode != "policy":
    parser.error("Holds, paired noise, and prefix replay require policy mode")
if args.output_dir.exists():
    parser.error("Use a new output directory to preserve previous diagnostics")
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import av
import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.seed import configure_seed
from isaaclab_tasks.utils import parse_env_cfg
from openpi_client.websocket_client_policy import WebsocketClientPolicy

import ambench.tasks  # noqa: F401
from ambench.policies.utils import compute_wall_axes
from ambench.recording.paths import write_env_cfg
from ambench.tasks.push_slider.push_slider_env import (
    ENGAGEMENT_THRESHOLD_RATIO,
    PushSlider,
)
from ambench.utils.camera_utils import get_camera_rgb
from ambench.utils.image_processing import prepare_rgb_image
from ambench_learn.eval.common import configure_env_cfg, make_scene_camera_cfg
from ambench_learn.policies.pi.eval_utils import (
    build_openpi_example,
    default_action_for_env,
    expand_policy_actions,
)

TASK = "PushSlider-Am-FAHexa-Abs-PID-Direct-v0"
PROMPT = "push the slider to the right"


def load_replay_prefix(directory: Path, initial: dict, engagement_position: np.ndarray) -> list[dict]:
    """Load a common approach and one post-engagement chunk without model feedback.

    The intervention starts at the first 32-step boundary after recorded native
    engagement. One further chunk is replayed in every condition, then live
    inference resumes. All conditions therefore share the first raw command
    chunk at the intervention boundary as well as the preceding approach.
    """
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["seed"] == args.seed and metadata["num_envs"] == args.num_envs
    assert metadata["task"] == TASK and metadata["mode"] == "policy"
    assert metadata["policy_id"] == args.policy_id and metadata["control_hz"] == 120
    assert not metadata.get("height_hold") and not metadata.get("depth_hold")
    assert json.loads((directory / "initial_state.json").read_text()) == initial, "Replay reset state mismatch"
    records = []
    for index in range(args.num_envs):
        path = directory / f"trace_env_{index}.npz"
        with np.load(path) as data:
            engaged = np.flatnonzero(data["slider_joint_pos"] >= engagement_position[index])
            assert len(engaged), f"Recorded scene {index} never engaged"
            intervention_step = ((int(engaged[0]) + 1 + 31) // 32) * 32
            resume_step = intervention_step + 32
            actions = data["action"][:resume_step].copy()
            assert actions.shape == (resume_step, 8) and np.isfinite(actions).all()
        records.append({
            "actions": actions,
            "intervention_step": intervention_step,
            "resume_step": resume_step,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return records


def main() -> None:
    """Run one bounded batch and save measurements plus both camera views."""
    configure_seed(args.seed)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    scene = make_scene_camera_cfg()
    scene.width, scene.height = 640, 360
    cfg = configure_env_cfg(
        parse_env_cfg=parse_env_cfg,
        task=TASK,
        device=args.device,
        num_envs=args.num_envs,
        seed=args.seed,
        scene_camera_cfg=scene,
    )
    cfg.slider_cfg.spawn.activate_contact_sensors = True
    write_env_cfg(root, cfg)
    env = _MeasuredSlider(cfg=cfg)
    writers = {}
    started = time.perf_counter()
    try:
        assert abs(env.dt - 1 / 120) < 1e-9
        assert env.slider_contact.body_names == ["Slider"], env.slider_contact.body_names
        client = WebsocketClientPolicy(host=args.host, port=args.port) if args.mode == "policy" else None
        assert (
            not args.paired_noise
            or client.get_server_metadata().get("diagnostic_noise_protocol") == "scene_env_step_v1"
        )
        obs, _ = env.reset(seed=args.seed)
        env.slider_body_index = env.slider.find_bodies("Slider")[0][0]
        initial = {
            "robot_root_pose": env.robot.data.root_pose_w.cpu().tolist(),
            "robot_joint_pos": env.robot.data.joint_pos.cpu().tolist(),
            "wall_root_pose": env.wall.data.root_pose_w.cpu().tolist(),
            "slider_root_pose": env.slider.data.root_pose_w.cpu().tolist(),
            "slider_joint_pos": env.slider.data.joint_pos.cpu().tolist(),
            "ee_pose": env.robot.data.body_link_state_w[:, env.ee_link_idx, :7].cpu().tolist(),
            "env_origins": env.scene.env_origins.cpu().tolist(),
        }
        (root / "initial_state.json").write_text(json.dumps(initial, indent=2) + "\n")
        engagement_position = env.slider.data.default_joint_pos[
            :, env.slider_joint_idx
        ] + ENGAGEMENT_THRESHOLD_RATIO * (
            cfg.slider_pushed_threshold - env.slider.data.default_joint_pos[:, env.slider_joint_idx]
        )
        prefix = (
            load_replay_prefix(args.replay_prefix_dir, initial, engagement_position.cpu().numpy())
            if args.replay_prefix_dir
            else None
        )
        wall_normals = torch.stack([compute_wall_axes(quat)[0] for quat in env.wall.data.root_quat_w])
        assert not args.depth_hold or torch.all(
            torch.abs(wall_normals[:, 2]) < 1e-6
        ), "Depth hold requires a horizontal wall normal"
        module, name = gym.spec(TASK).kwargs["scripted_policy_entry_point"].split(":")
        expert_class = getattr(importlib.import_module(module), name)
        experts = [expert_class(env=env, inject_noise=False) for _ in range(args.num_envs)]
        metadata = {
            "task": TASK,
            "mode": args.mode,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "policy_id": args.policy_id if client else None,
            "server_metadata": client.get_server_metadata() if client else None,
            "prompt": PROMPT,
            "expert": f"{module}:{name}",
            "control_hz": 120,
            "policy_hz": 30,
            "n_action_steps": 8,
            "video_hz": 30,
            "success_threshold_m": float(cfg.slider_pushed_threshold),
            "contact_sensor_prim": env.slider_contact.cfg.prim_path,
            "contact_measurement": (
                "Slider net normal contact force in world frame; not pair-specific, no friction force"
            ),
            "trace_phase": "after physics and before automatic reset, including terminal step",
            "video_phase": "before action at every fourth control step; timestamps start at zero",
            "privileged_data_used_by_pi": False,
            "height_hold": args.height_hold,
            "depth_hold": args.depth_hold,
            "depth_hold_rule": "Latch commanded tool-to-slider projection on wall normal; preserve slide-axis motion",
            "wall_normals": wall_normals.cpu().tolist(),
            "height_hold_rule": (
                "Latch pre-intervention commanded local Z at first observed native slider engagement; "
                "override only local Z until termination"
            ),
            "height_hold_progress_fraction": ENGAGEMENT_THRESHOLD_RATIO,
            "privileged_data_used_by_action_override": args.height_hold or args.depth_hold,
            "paired_noise": args.paired_noise,
            "replay_prefix_dir": str(args.replay_prefix_dir) if prefix else None,
            "prefix_schedule": (
                [{key: value for key, value in row.items() if key != "actions"} for row in prefix] if prefix else None
            ),
            "intervention_timing": (
                "Scheduled chunk boundary after recorded engagement" if prefix else "Live engagement"
            ),
        }
        (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        plans = [collections.deque() for _ in range(args.num_envs)]
        traces = [[] for _ in range(args.num_envs)]
        done = [False] * args.num_envs
        outcomes = [None] * args.num_envs
        video_frames = [0] * args.num_envs
        hold_z = [None] * args.num_envs
        hold_depth = [None] * args.num_envs
        (root / "videos").mkdir()
        for index in range(args.num_envs):
            for camera, width, height in (("ee_camera", 384, 384), ("scene_camera", 640, 360)):
                container = av.open(str(root / "videos" / f"env_{index}_{camera}.mp4"), "w")
                stream = container.add_stream("libx264", rate=30)
                stream.width, stream.height = width, height
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "18", "preset": "fast"}
                writers[index, camera] = container, stream
        env.record_measurements = True
        with torch.inference_mode(), (root / "inference.jsonl").open("w") as inference_log:
            for step in range(env.max_episode_length):
                if all(done):
                    break
                if not simulation_app.is_running() or simulation_app.is_exiting():
                    raise RuntimeError("Simulator stopped before diagnostic finished")
                if step % 4 == 0:
                    for index in range(args.num_envs):
                        if done[index]:
                            continue
                        for camera in ("ee_camera", "scene_camera"):
                            frame = np.asarray(prepare_rgb_image(get_camera_rgb(env, camera, index)), dtype=np.uint8)
                            container, stream = writers[index, camera]
                            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                                container.mux(packet)
                        video_frames[index] += 1
                commands = []
                raw_commands = []
                for index in range(args.num_envs):
                    if done[index]:
                        command = default_action_for_env(env, action_semantics="ee_absolute", env_index=index)
                    elif args.mode == "expert":
                        command = experts[index].advance(obs, index).reshape(8)
                    elif prefix and step < prefix[index]["resume_step"]:
                        command = torch.tensor(prefix[index]["actions"][step], device=env.device, dtype=torch.float32)
                    else:
                        if not plans[index]:
                            example = build_openpi_example(
                                obs, env, env_index=index, action_semantics="ee_absolute", prompt=PROMPT
                            )
                            if prefix and step == prefix[index]["resume_step"]:
                                np.savez_compressed(root / f"handoff_observation_env_{index}.npz", **example)
                            inference_start = time.perf_counter()
                            if args.paired_noise:
                                example["diagnostic_noise_key"] = [args.seed, index, step]
                            prediction = client.infer(example)
                            actions = np.asarray(prediction["actions"], dtype=np.float32).copy()
                            assert actions.shape == (30, 8) and np.isfinite(actions).all()
                            inference_log.write(
                                json.dumps({
                                    "env_index": index,
                                    "step": step,
                                    "seconds": time.perf_counter() - inference_start,
                                    "actions": actions.tolist(),
                                    "noise_sha256": prediction.get("diagnostic_noise_sha256"),
                                    "image_sha256": hashlib.sha256(example["am_bench/ee_image"].tobytes()).hexdigest(),
                                })
                                + "\n"
                            )
                            plans[index] = expand_policy_actions(
                                actions,
                                env,
                                action_semantics="ee_absolute",
                                n_action_steps=8,
                                action_execution_stride=4,
                                env_index=index,
                            )
                        command = plans[index].popleft()
                    raw_commands.append(command.clone())
                    if not done[index]:
                        eligible = (
                            step == prefix[index]["intervention_step"]
                            if prefix
                            else env.slider.data.joint_pos[index, env.slider_joint_idx] >= engagement_position[index]
                        )
                        slider_pos = (
                            env.slider.data.body_link_pos_w[index, env.slider_body_index] - env.scene.env_origins[index]
                        )
                        if eligible:
                            assert (
                                env.slider.data.joint_pos[index, env.slider_joint_idx]
                                >= engagement_position[index] - 1e-5
                            )
                            if args.height_hold and hold_z[index] is None:
                                hold_z[index] = float(command[2])
                            if args.depth_hold and hold_depth[index] is None:
                                hold_depth[index] = float(torch.dot(command[:3] - slider_pos, wall_normals[index]))
                        if hold_z[index] is not None:
                            command = command.clone()
                            command[2] = hold_z[index]
                        if hold_depth[index] is not None:
                            command = command.clone()
                            depth = torch.dot(command[:3] - slider_pos, wall_normals[index])
                            command[:3] += (hold_depth[index] - depth) * wall_normals[index]
                    commands.append(command)
                action = torch.stack(commands)
                assert torch.isfinite(action).all(), "Nonfinite command"
                obs, _, terminated, truncated, _ = env.step(action)
                for index in range(args.num_envs):
                    if done[index]:
                        continue
                    record = {key: value[index].copy() for key, value in env.measurements.items()}
                    record["unmodified_action"] = raw_commands[index].cpu().numpy().copy()
                    record["height_hold_active"] = np.asarray(hold_z[index] is not None)
                    record["height_hold_reference_z"] = np.asarray(hold_z[index] if hold_z[index] is not None else 0.0)
                    record["depth_hold_active"] = np.asarray(hold_depth[index] is not None)
                    record["depth_hold_reference_m"] = np.asarray(
                        hold_depth[index] if hold_depth[index] is not None else 0.0
                    )
                    record["replay_prefix_active"] = np.asarray(bool(prefix and step < prefix[index]["resume_step"]))
                    traces[index].append(record)
                    if bool(terminated[index]) or bool(truncated[index]):
                        done[index] = True
                        outcomes[index] = {
                            "env_index": index,
                            "success": bool(terminated[index]),
                            "termination_reason": "success" if bool(terminated[index]) else "timeout",
                            "steps": step + 1,
                            "video_frames": video_frames[index],
                        }
                if step % 600 == 0:
                    print(
                        f"{args.mode} seed={args.seed} step={step} active={sum(not value for value in done)}",
                        flush=True,
                    )
        assert all(done), "Diagnostic did not reach native termination for every case"
        for index, records in enumerate(traces):
            data = {key: np.stack([row[key] for row in records]) for key in records[0]}
            data["time_s"] = np.arange(1, len(records) + 1) / 120
            assert all(np.isfinite(value).all() for value in data.values())
            np.savez_compressed(root / f"trace_env_{index}.npz", **data)
            outcomes[index].update(
                final_slider_position_m=float(data["slider_joint_pos"][-1]),
                max_slider_position_m=float(data["slider_joint_pos"].max()),
                mean_ee_tracking_error_m=float(np.linalg.norm(data["ee_pos"] - data["ee_command_pos"], axis=1).mean()),
            )
        (root / "summary.json").write_text(
            json.dumps(
                {"status": "completed", "elapsed_seconds": time.perf_counter() - started, "cases": outcomes}, indent=2
            )
            + "\n"
        )
    finally:
        for container, stream in writers.values():
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        env.close()


class _MeasuredSlider(PushSlider):
    """Read-only task instrumentation, confined to this diagnostic process."""

    def _setup_scene(self) -> None:
        super()._setup_scene()
        root_path = self.cfg.slider_cfg.prim_path.replace("env_.*", "env_0")
        prims = sim_utils.get_all_matching_child_prims(root_path, lambda prim: prim.GetName() == "Slider")
        if len(prims) != 1:
            raise RuntimeError(f"Expected one Slider body under {root_path}, found {len(prims)}")
        prim_path = str(prims[0].GetPath()).replace("/env_0/", "/env_.*/")
        self.slider_contact = ContactSensor(ContactSensorCfg(prim_path=prim_path, update_period=0.0))
        self.scene.sensors["slider_diagnostic_contact"] = self.slider_contact

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, truncated = super()._get_dones()
        if getattr(self, "record_measurements", False):
            origins = self.scene.env_origins
            values = {
                "slider_joint_pos": self.slider.data.joint_pos[:, self.slider_joint_idx],
                "slider_joint_vel": self.slider.data.joint_vel[:, self.slider_joint_idx],
                "slider_body_pos": self.slider.data.body_link_pos_w[:, self.slider_body_index] - origins,
                "slider_normal_contact_force_w": self.slider_contact.data.net_forces_w[:, 0],
                "ee_pos": self.robot.data.body_link_pos_w[:, self.ee_link_idx] - origins,
                "ee_quat": self.robot.data.body_link_quat_w[:, self.ee_link_idx],
                "ee_command_pos": self.ee_cmd_pos_w - origins,
                "ee_command_quat": self.ee_cmd_quat_w,
                "gripper_joint_pos": self.robot.data.joint_pos[:, self.gripper_joint_ids],
                "gripper_targets": self.gripper_targets,
                "action": self.actions,
                "robot_root_pose": self.robot.data.root_pose_w,
                "robot_root_velocity": self.robot.data.root_state_w[:, 7:],
                "robot_joint_pos": self.robot.data.joint_pos,
                "robot_joint_vel": self.robot.data.joint_vel,
            }
            self.measurements = {key: value.detach().cpu().numpy().copy() for key, value in values.items()}
        return terminated, truncated


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
