# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate multi-task prompt routing and optional checkpoint inference on recorded observations."""

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/openpi"))
    parser.add_argument("--repo-id", default="my-zzy/ambench_contact3_30hz_v1")
    parser.add_argument("--checkpoint", type=Path, help="Omit for CPU data/token checks only.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.work_dir.resolve()
    os.environ["HF_LEROBOT_HOME"] = str(root / "datasets")
    os.environ["HF_HOME"] = str(root / "cache/huggingface")
    os.environ["OPENPI_DATA_HOME"] = str(root / "cache/openpi")
    os.environ["JAX_PLATFORMS"] = "cuda" if args.checkpoint else "cpu"

    # LeRobot/OpenPI read cache paths at import time, after the explicit paths above.
    import numpy as np
    from openpi import transforms
    from openpi.policies import policy_config
    from openpi.training import data_loader

    from ambench_learn.policies.pi.config import make_config

    cfg = make_config(work_dir=root, repo_id=args.repo_id, multitask=True, finetune_mode="full")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = data_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)
    episodes = [
        json.loads(line) for line in (root / "datasets" / args.repo_id / "meta/episodes.jsonl").read_text().splitlines()
    ]
    samples = {}
    cursor = 0
    for episode in episodes:
        prompt = episode["tasks"][0]
        if prompt not in samples:
            sample = transforms.compose(data_cfg.repack_transforms.inputs)(dataset[cursor])
            if str(sample["prompt"]) != prompt:
                raise ValueError(f"Prompt routing mismatch at frame {cursor}")
            sample.pop("actions", None)
            samples[prompt] = {key: np.asarray(value) if key != "prompt" else value for key, value in sample.items()}
        cursor += episode["length"]
    if len(samples) < 2:
        raise ValueError("Expected at least two distinct instructions")
    model_input = transforms.compose([
        *data_cfg.data_transforms.inputs,
        transforms.Normalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm),
        *data_cfg.model_transforms.inputs,
    ])
    token_ids = {}
    for prompt, sample in samples.items():
        inputs = model_input(dict(sample))
        tokens = np.asarray(inputs["tokenized_prompt"])
        mask = np.asarray(inputs["tokenized_prompt_mask"], dtype=bool)
        token_ids[prompt] = tokens[mask].tolist()
    if len({tuple(tokens) for tokens in token_ids.values()}) != len(samples):
        raise ValueError("Different instructions collapsed to identical model tokens")
    report = {
        "passed": True,
        "repo_id": args.repo_id,
        "task_count": len(samples),
        "token_ids": token_ids,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "task_success_evaluated": False,
        "scope": (
            "Recorded-observation interface check; prompt sensitivity does not establish language-grounded success"
        ),
    }
    if args.checkpoint:
        keys = ["am_bench/ee_pos", "am_bench/ee_quat", "am_bench/gripper_width", "am_bench/ee_image", "prompt"]
        policy = policy_config.create_trained_policy(
            cfg,
            args.checkpoint.resolve(),
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({key: key for key in keys})]),
        )
        noise = (
            np.random.default_rng(42)
            .standard_normal((cfg.model.action_horizon, cfg.model.action_dim))
            .astype(np.float32)
        )
        actions = {}
        for prompt, sample in samples.items():
            result = np.asarray(policy.infer(sample, noise=noise)["actions"])
            if result.shape != (30, 8) or not np.isfinite(result).all():
                raise ValueError(f"Invalid checkpoint actions for {prompt}: {result.shape}")
            actions[prompt] = result.tolist()
        report["per_task_actions"] = actions
        # Hold the image, robot state, and sampling noise fixed while changing only text.
        anchor_prompt = next(iter(samples))
        anchor = samples[anchor_prompt]
        reference = np.asarray(actions[anchor_prompt])
        changes = {}
        for prompt in samples:
            candidate = np.asarray(policy.infer({**anchor, "prompt": prompt}, noise=noise)["actions"])
            if not np.isfinite(candidate).all():
                raise ValueError(f"Nonfinite actions in prompt swap: {prompt}")
            changes[prompt] = float(np.max(np.abs(candidate - reference)))
        report["same_observation_fixed_noise_prompt_swap"] = {
            "anchor_prompt": anchor_prompt,
            "max_action_component_change": changes,
            "interpretation": "Diagnostic only; a difference is not evidence that the requested task is completed",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "per_task_actions"}, indent=2))


if __name__ == "__main__":
    main()
