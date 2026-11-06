# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Reconstructed PressButton π₀ setup using the pinned OpenPI implementation."""

import dataclasses
from pathlib import Path

from openpi.models.pi0_config import Pi0Config
from openpi.training import config, optimizer, weight_loaders

REPO = Path(__file__).resolve().parents[5]
NAME = "pi0_pressbutton_lora_30hz"
DATASET = "my-zzy/pressbutton_30hz"


def make_config(
    *,
    work_dir: Path,
    exp_name: str = "pressbutton_lora",
    batch_size: int = 1,
    steps: int = 10_000,
    resume: bool = False,
    repo_id: str = DATASET,
    pretrained_checkpoint: Path | None = None,
) -> config.TrainConfig:
    """Build the original wrist-only 30 Hz pilot, with explicit artifact locations."""
    root = work_dir.expanduser().resolve()
    checkpoint = pretrained_checkpoint or root / "pretrained/pi0_base"
    model = Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_horizon=30,
    )
    base = config.get_config("pi0_am_bench_multitask_openpi_original_20hz_h50_ee_local_relative")
    return dataclasses.replace(
        base,
        name=NAME,
        project_name="ambench_pi0_local",
        exp_name=exp_name,
        model=model,
        data=config.LeRobotAmBenchDataConfig(
            repo_id=repo_id,
            base_config=config.DataConfig(prompt_from_task=True),
            include_base_image=False,
            action_representation="ee_local_relative",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(checkpoint / "params")),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        assets_base_dir=str(root / "assets"),
        checkpoint_base_dir=str(root / "checkpoints"),
        batch_size=batch_size,
        num_workers=2,
        num_train_steps=steps,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=100, peak_lr=2.5e-5, decay_steps=10_000, decay_lr=2.5e-6
        ),
        log_interval=1 if steps <= 10 else 20,
        save_interval=1000,
        keep_period=5000,
        resume=resume,
        overwrite=False,
        wandb_enabled=False,
        policy_metadata={
            "action_representation": "ee_local_relative",
            "policy_target_hz": 30,
            "action_horizon": 30,
            "task_prompt": "press the button",
        },
    )
