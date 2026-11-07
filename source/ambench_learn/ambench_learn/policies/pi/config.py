# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""PressButton and multi-task π₀ recipes using the pinned OpenPI implementation."""

import dataclasses
from pathlib import Path

from openpi.models.pi0_config import Pi0Config
from openpi.training import config, optimizer, weight_loaders

REPO = Path(__file__).resolve().parents[5]
NAME = "pi0_pressbutton_lora_30hz"
DATASET = "my-zzy/pressbutton_30hz"
MULTITASK_DATASET = "my-zzy/ambench_contact3_30hz_v1"


def make_config(
    *,
    work_dir: Path,
    exp_name: str = "pressbutton_lora",
    batch_size: int = 1,
    steps: int = 10_000,
    resume: bool = False,
    repo_id: str | None = None,
    corrected: bool = False,
    pretrained_checkpoint: Path | None = None,
    finetune_mode: str = "lora",
    fsdp_devices: int = 1,
    num_workers: int = 2,
    multitask: bool = False,
) -> config.TrainConfig:
    """Build wrist-only LoRA or full tuning, preserving the original pilot defaults."""
    if finetune_mode not in {"lora", "full"}:
        raise ValueError("finetune_mode must be 'lora' or 'full'")
    if min(batch_size, steps, fsdp_devices) < 1 or num_workers < 0:
        raise ValueError("Batch size, steps, and FSDP devices must be positive; workers must be nonnegative")
    root = work_dir.expanduser().resolve()
    repo_id = repo_id or (
        MULTITASK_DATASET if multitask else "my-zzy/pressbutton_30hz_terminal_v2" if corrected else DATASET
    )
    checkpoint = pretrained_checkpoint or root / "pretrained/pi0_base"
    model = Pi0Config(
        paligemma_variant="gemma_2b_lora" if finetune_mode == "lora" else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if finetune_mode == "lora" else "gemma_300m",
        action_horizon=30,
    )
    asset_name = "pi0_multitask_lora_30hz" if multitask else NAME + ("_corrected" if corrected else "")
    base = config.get_config("pi0_am_bench_multitask_openpi_original_20hz_h50_ee_local_relative")
    return dataclasses.replace(
        base,
        name=asset_name if finetune_mode == "lora" else asset_name.replace("_lora_", "_full_"),
        project_name="ambench_pi0_local",
        exp_name=exp_name,
        model=model,
        data=config.LeRobotAmBenchDataConfig(
            repo_id=repo_id,
            # Full tuning reuses the validated statistics without regenerating them.
            assets=(
                config.AssetsConfig(assets_dir=str(root / "assets" / asset_name))
                if finetune_mode == "full"
                else config.AssetsConfig()
            ),
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
        num_workers=num_workers,
        fsdp_devices=fsdp_devices,
        num_train_steps=steps,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=2.5e-5,
            decay_steps=max(steps, 101) if finetune_mode == "full" else 10_000,
            decay_lr=2.5e-6,
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
            **({"multitask": True, "requires_prompt": True} if multitask else {"task_prompt": "press the button"}),
        },
    )
