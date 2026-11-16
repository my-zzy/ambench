# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Collect verified regrasp demonstrations in one fixed PushSlider scene.

Induced departures are excluded from imitation labels. All attempts retain
videos and pre-reset physics traces; only successful recovery segments enter
the 30 Hz LeRobot dataset. Native PID, success and 26-second timeout are retained.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cases", type=Path, required=True)
parser.add_argument("--case-start", type=int, default=0)
parser.add_argument("--case-count", type=int, default=4)
parser.add_argument("--seed", type=int, default=1511)
parser.add_argument("--output-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output_dir.exists():
    parser.error("Use a new output directory to preserve previous runs")
cases = json.loads(args.cases.read_text())[args.case_start : args.case_start + args.case_count]
if len(cases) != args.case_count or args.case_count < 1 or args.case_start < 0:
    parser.error("Invalid case range")
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import av
import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.seed import configure_seed
from isaaclab_tasks.utils import parse_env_cfg

import ambench.tasks  # noqa: F401
from ambench.policies.scripted.push_slider import PushSliderPolicy
from ambench.policies.scripted.push_slider_recovery import (
    PushSliderRecoveryPolicy,
    interpolate_command,
)
from ambench.policies.utils import compute_wall_axes
from ambench.recording.lerobot_writer import LeRobotSessionWriter
from ambench.recording.paths import write_env_cfg
from ambench.tasks.push_slider.push_slider_env import PushSlider
from ambench.utils.camera_utils import get_camera_rgb
from ambench.utils.image_processing import prepare_rgb_image
from ambench_learn.data.action_resampling import (
    interpolate_absolute_action_for_execution,
)
from ambench_learn.eval.common import configure_env_cfg, make_scene_camera_cfg
from ambench_learn.policies.pi.eval_utils import default_action_for_env

TASK = "PushSlider-Am-FAHexa-Abs-PID-Direct-v0"
PHASES = ("expert", "depart", "lost_hold", "open", "align", "approach", "close", "push", "failed", "probe")


class RecoveryTrial:
    """Schedule a deliberate departure, verify loss, and invoke the recovery teacher."""

    def __init__(self, env, index: int, case: dict) -> None:
        self.env, self.index, self.case = env, index, case
        self.expert = PushSliderPolicy(env=env, inject_noise=False)
        self.recovery = PushSliderRecoveryPolicy(env, env.slider_contact)
        self.command = default_action_for_env(env, action_semantics="ee_absolute", env_index=index)
        self.normal, wall_y, _ = compute_wall_axes(env.wall.data.root_quat_w[index])
        self.direction = -wall_y
        self.phase, self.elapsed, self.loss_steps = "expert", 0.0, 0
        self.done = False
        self.events, self.trace, self.frames, self.frame_steps = [], [], [], []
        self.recovery_start_step = None
        self.regrasp_step = None
        self.regrasp_position = None
        self.outcome = None

    def transition(self, phase: str, step: int) -> None:
        self.events.append({"phase": phase, "step": step, "time_s": step / 120})
        self.phase, self.elapsed = phase, 0.0

    def advance(self, obs: dict, step: int) -> torch.Tensor:
        state = obs["policy"][self.index]
        position = float(state["slider_pos"].item())
        force = float(torch.linalg.vector_norm(self.env.slider_contact.data.net_forces_w[self.index, 0]))
        width = float(state["gripper_width"].item())
        slider = self.env.slider.data.body_link_pos_w[self.index, self.env.slider_body_index]
        slider = slider - self.env.scene.env_origins[self.index]
        separation = float(torch.dot(slider - state["ee_pos"], self.normal))
        if self.phase == "expert" and position >= -0.5 + 0.95 * self.case["progress"] and force > 0.1:
            self.depart_start = self.command.clone()
            self.depart_target = self.command.clone()
            self.depart_target[:3] -= self.normal * self.case["outward_m"]
            self.depart_target[:3] += self.direction * self.case["tangent_m"]
            self.depart_target[2] += self.case["height_m"]
            self.transition("depart", step)

        if self.phase == "expert":
            for _ in range(4):
                self.command = self.expert.advance(obs, self.index).reshape(8)
        elif self.phase == "depart":
            self.elapsed += 1 / 30
            self.command = interpolate_command(
                self.depart_start, self.depart_target, self.elapsed / self.case["departure_seconds"]
            )
            if self.elapsed >= self.case["departure_seconds"]:
                self.transition("lost_hold", step)
        elif self.phase == "lost_hold":
            self.elapsed += 1 / 30
            lost = force < 0.1 and width < 0.015 and separation > 0.28
            self.loss_steps = self.loss_steps + 1 if lost else 0
            if self.loss_steps >= 6:
                self.recovery.start(self.command, self.index)
                self.recovery_start_step = step
                self.transition("open", step)
                self.command = self.recovery.advance(obs, self.index).reshape(8)
            elif self.elapsed > 2.0:
                self.transition("failed", step)
        elif self.phase != "failed":
            self.command = self.recovery.advance(obs, self.index).reshape(8)
            if self.recovery.phase != self.phase:
                self.transition(self.recovery.phase, step)
            if self.recovery.regrasp_count and self.regrasp_step is None:
                self.regrasp_step, self.regrasp_position = step, position
        return self.command

    def finish(self, step: int, native_success: bool, reason: str) -> None:
        self.done = True
        final = float(self.trace[-1]["slider_joint_pos"])
        resumed = self.regrasp_position is not None and final - self.regrasp_position >= 0.02
        valid = native_success and self.recovery_start_step is not None and resumed
        self.outcome = {
            "case": self.case,
            "native_success": native_success,
            "verified_recovery_success": valid,
            "termination_reason": reason,
            "steps": step + 1,
            "loss_verified": self.recovery_start_step is not None,
            "recovery_start_step": self.recovery_start_step,
            "regrasp_step": self.regrasp_step,
            "regrasp_count": self.recovery.regrasp_count,
            "retries": self.recovery.retries,
            "resumed_travel_m": final - self.regrasp_position if self.regrasp_position is not None else None,
            "final_slider_position_m": final,
            "recovery_seconds": (step + 1 - self.recovery_start_step) / 120 if valid else None,
            "candidate_frames": len(self.frames),
            "events": self.events,
        }


class MeasuredSlider(PushSlider):
    """Read-only instrumentation before Isaac Lab automatically resets a done scene."""

    def _setup_scene(self) -> None:
        super()._setup_scene()
        root_path = self.cfg.slider_cfg.prim_path.replace("env_.*", "env_0")
        prims = sim_utils.get_all_matching_child_prims(root_path, lambda prim: prim.GetName() == "Slider")
        if len(prims) != 1:
            raise RuntimeError("Expected one Slider body")
        path = str(prims[0].GetPath()).replace("/env_0/", "/env_.*/")
        self.slider_contact = ContactSensor(ContactSensorCfg(prim_path=path, update_period=0.0))
        self.scene.sensors["slider_recovery_contact"] = self.slider_contact

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
                "gripper_joint_pos": self.robot.data.joint_pos[:, self.gripper_joint_ids],
            }
            self.measurements = {key: value.detach().cpu().numpy().copy() for key, value in values.items()}
        return terminated, truncated


def run_trials(env, obs: dict, trials: list[RecoveryTrial], writer: LeRobotSessionWriter, videos: dict) -> None:
    """Pair pre-action observations with 30 Hz targets; execute interpolated 120 Hz commands."""
    env.record_measurements = True
    plans = [None] * len(trials)
    with torch.inference_mode():
        for step in range(env.max_episode_length):
            if all(trial.done for trial in trials):
                break
            if not simulation_app.is_running() or simulation_app.is_exiting():
                raise RuntimeError("Simulator stopped before trials finished")
            if step % 4 == 0:
                for index, trial in enumerate(trials):
                    if trial.done:
                        continue
                    previous = trial.command.clone()
                    target = trial.advance(obs, step).clone()
                    plans[index] = interpolate_absolute_action_for_execution(previous, target, 4)
                    images = {}
                    for camera in ("ee_camera", "base_camera", "scene_camera"):
                        rgb = np.asarray(prepare_rgb_image(get_camera_rgb(env, camera, index)), dtype=np.uint8).copy()
                        container, stream = videos[index, camera]
                        for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                            container.mux(packet)
                        if camera != "scene_camera":
                            images[camera] = rgb
                    if trial.recovery_start_step is not None and trial.phase != "failed":
                        trial.frames.append(
                            writer.build_frame(obs["policy"][index], target.cpu().numpy().copy(), images)
                        )
                        trial.frame_steps.append(step)
            actions = torch.stack([
                (
                    default_action_for_env(env, action_semantics="ee_absolute", env_index=index)
                    if trial.done
                    else plans[index][step % 4]
                )
                for index, trial in enumerate(trials)
            ])
            assert torch.isfinite(actions).all()
            executed = actions.detach().cpu().numpy().copy()
            obs, _, terminated, truncated, _ = env.step(actions)
            for index, trial in enumerate(trials):
                if trial.done:
                    continue
                record = {key: value[index].copy() for key, value in env.measurements.items()}
                record.update(action=executed[index], phase=np.asarray(PHASES.index(trial.phase)))
                trial.trace.append(record)
                if bool(terminated[index]) or bool(truncated[index]) or trial.phase == "failed":
                    success = bool(terminated[index])
                    reason = "success" if success else "teacher_failed" if trial.phase == "failed" else "timeout"
                    trial.finish(step, success, reason)
                    print(json.dumps(trial.outcome), flush=True)
            if step % 240 == 0:
                print(
                    f"step={step} phases={[trial.phase for trial in trials]} done={[trial.done for trial in trials]}",
                    flush=True,
                )
    assert all(trial.done for trial in trials), "Unfinished bounded trial"


def save_trials(root: Path, trials: list[RecoveryTrial], writer: LeRobotSessionWriter) -> None:
    """Retain every attempt; admit only verified recovery segments into training data."""
    episode = 0
    for index, trial in enumerate(trials):
        arrays = {key: np.stack([row[key] for row in trial.trace]) for key in trial.trace[0]}
        arrays["time_s"] = np.arange(1, len(trial.trace) + 1) / 120
        assert all(np.isfinite(array).all() for array in arrays.values())
        np.savez_compressed(root / f"trace_env_{index}.npz", **arrays)
        if trial.outcome["verified_recovery_success"]:
            assert trial.frame_steps[0] == trial.recovery_start_step
            assert np.all(np.diff(trial.frame_steps) == 4)
            writer.save_episode(trial.frames)
            trial.outcome["dataset_episode_index"] = episode
            np.savez_compressed(
                root / f"labels_episode_{episode}.npz",
                control_step=np.asarray(trial.frame_steps),
                action=np.stack([frame["action"] for frame in trial.frames]),
                state=np.stack([frame["observation.state"] for frame in trial.frames]),
            )
            episode += 1
        trial.frames.clear()


def main() -> None:
    configure_seed(args.seed)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    scene = make_scene_camera_cfg()
    scene.width, scene.height = 640, 360
    cfg = configure_env_cfg(
        parse_env_cfg=parse_env_cfg,
        task=TASK,
        device=args.device,
        num_envs=len(cases),
        seed=args.seed,
        scene_camera_cfg=scene,
    )
    cfg.slider_cfg.spawn.activate_contact_sensors = True
    # Observations and labels are 30 Hz; physics and control remain at 120 Hz.
    cfg.sim.render_interval = 4
    cfg.events.randomize_wall_pose.params["pose_range"] = {"x": (0, 0), "y": (0, 0), "pitch": (0, 0)}
    cfg.events.randomize_slider_position.params.update(y_range=(0, 0), z_range=(0, 0))
    write_env_cfg(root, cfg)
    metadata = {
        "task": TASK,
        "seed": args.seed,
        "cases": cases,
        "scene": "fixed default wall and slider placement",
        "control_hz": 120,
        "dataset_hz": 30,
        "render_hz": 30,
        "video_hz": 30,
        "phase_names": PHASES,
        "teacher_uses_privileged_state": True,
        "learned_policy_evaluated": False,
        "state_keys": ["ee_pos", "ee_quat", "gripper_width"],
        "loss_rule": "6 consecutive 30Hz observations: contact norm <0.1N, width <15mm, separation >280mm",
        "success_rule": "verified loss, verified regrasp, >=20mm resumed travel, native joint >=0.45m",
        "dataset_scope": "recovery start through native success; deliberate departure excluded",
        "action_timing": "pre-action observation -> macro target -> four interpolated 120Hz commands",
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    env = MeasuredSlider(cfg=cfg)
    writer = LeRobotSessionWriter(
        root=root / "lerobot",
        repo_id="ambench/slider_recovery",
        robot_type="fa_hexa",
        fps=30,
        state_keys=("ee_pos", "ee_quat", "gripper_width"),
        task_prompt="push the slider to the right",
        use_videos=False,
        env_cfg=cfg,
    )
    videos = {}
    started = time.perf_counter()
    try:
        assert abs(env.dt - 1 / 120) < 1e-9 and cfg.episode_length_s == 26
        assert env.slider_contact.body_names == ["Slider"]
        obs, _ = env.reset(seed=args.seed)
        env.slider_body_index = env.slider.find_bodies("Slider")[0][0]
        initial = {}
        for name, asset in (("wall", env.wall), ("slider", env.slider), ("robot", env.robot)):
            pose = asset.data.root_pose_w.clone()
            pose[:, :3] -= env.scene.env_origins
            initial[name] = pose.cpu().tolist()
            assert torch.allclose(pose, pose[0].expand_as(pose), atol=2e-6), f"{name}: scenes differ"
        initial["slider_joint_pos"] = env.slider.data.joint_pos.cpu().tolist()
        (root / "initial_state.json").write_text(json.dumps(initial, indent=2) + "\n")
        trials = [RecoveryTrial(env, index, case) for index, case in enumerate(cases)]
        (root / "videos").mkdir()
        for index in range(len(cases)):
            for camera in ("ee_camera", "base_camera", "scene_camera"):
                rgb = prepare_rgb_image(get_camera_rgb(env, camera, index))
                container = av.open(str(root / "videos" / f"env_{index}_{camera}.mp4"), "w")
                stream = container.add_stream("libx264", rate=30)
                stream.height, stream.width = rgb.shape[:2]
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "18", "preset": "fast"}
                videos[index, camera] = container, stream
        run_trials(env, obs, trials, writer, videos)
        save_trials(root, trials, writer)
        writer.close()
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "elapsed_seconds": time.perf_counter() - started,
                    "cases": [trial.outcome for trial in trials],
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        for container, stream in videos.values():
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
