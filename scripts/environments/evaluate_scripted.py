# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded scripted-expert comparison using the learned-policy evaluator's seed schedule."""
import argparse
import importlib
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="PressButton-Am-FAHexa-Abs-PID-Direct-v0")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--num-rollouts", type=int, default=20)
parser.add_argument("--seed", type=int, default=10000)
parser.add_argument("--rollout-seed-base", type=int, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from ambench_learn.eval.common import (
    configure_env_cfg,
    make_scene_camera_cfg,
    run_evaluation_batches,
)
from ambench_learn.eval.run import EvalRun
from isaaclab.utils.seed import configure_seed
from isaaclab_tasks.utils import parse_env_cfg

import ambench.tasks  # noqa: F401
from ambench.evaluation import SuccessCriteriaTracker
from ambench.recording.paths import write_env_cfg

try:
    configure_seed(args.seed)
    scene_camera_cfg = make_scene_camera_cfg()
    scene_camera_cfg.width = 640
    scene_camera_cfg.height = 360
    cfg = configure_env_cfg(
        parse_env_cfg=parse_env_cfg,
        task=args.task,
        device=args.device,
        num_envs=args.num_envs,
        seed=args.seed,
        scene_camera_cfg=scene_camera_cfg,
    )
    env = gym.make(args.task, cfg=cfg).unwrapped
    write_env_cfg(args.output_dir, cfg)
    module, cls = env.spec.kwargs["scripted_policy_entry_point"].split(":")
    policy_class = getattr(importlib.import_module(module), cls)
    run = EvalRun(
        output_dir=args.output_dir,
        policy="scripted_expert",
        task=args.task,
        policy_source=env.spec.kwargs["scripted_policy_entry_point"],
        requested_rollouts=args.num_rollouts,
        num_envs=args.num_envs,
        action_semantics="ee_absolute",
        action_representation="ee_absolute",
        metadata={
            "Seed": args.seed,
            "Rollout seed base": args.rollout_seed_base,
            "Native horizon": cfg.episode_length_s,
        },
        tracking_enabled=False,
    )
    run.print_startup()

    def batch_rollout(batch) -> bool:
        with torch.inference_mode():
            obs, _ = env.reset(seed=batch.seed)
            policies = [policy_class(env=env, inject_noise=False) for _ in range(args.num_envs)]
            done = [i >= batch.size for i in range(args.num_envs)]
            success = [False] * args.num_envs
            steps = [0] * args.num_envs
            reasons = ["timeout"] * args.num_envs
            criteria = [SuccessCriteriaTracker(i) for i in range(args.num_envs)]
            for i in range(batch.size):
                run.start_rollout(batch.start + i, batch_env_index=i, seed=batch.seed)
            for step in range(1, env.max_episode_length + 1):
                if not app.is_running() or all(done):
                    break
                actions = torch.cat([p.advance(obs, i) for i, p in enumerate(policies)], dim=0)
                obs, _, terminated, truncated, info = env.step(actions)
                for i in range(batch.size):
                    if done[i]:
                        continue
                    steps[i] += 1
                    criteria[i].update(info)
                    if bool(terminated[i]) or bool(truncated[i]):
                        done[i] = True
                        success[i] = bool(terminated[i])
                        reasons[i] = "success" if success[i] else "timeout"
                if step % 600 == 0:
                    print(f"Batch {batch.label()} step={step} active={sum(not d for d in done)}", flush=True)
            for i in range(batch.size):
                record = run.finish_rollout(
                    batch.start + i,
                    batch_env_index=i,
                    success=success[i],
                    executed_steps=steps[i],
                    subtask_completion=criteria[i].completion_fraction,
                    termination_reason=reasons[i],
                    extra=criteria[i].to_record(),
                )
                run.print_rollout_finished(record, total=args.num_rollouts)
        return not app.is_running()

    run_evaluation_batches(
        eval_run=run,
        env=env,
        simulation_app=app,
        requested_rollouts=args.num_rollouts,
        num_envs=args.num_envs,
        run_batch=batch_rollout,
        base_seed=args.rollout_seed_base,
    )
finally:
    app.close()
