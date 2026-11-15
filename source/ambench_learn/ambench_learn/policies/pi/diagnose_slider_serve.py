# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Serve the multi-task checkpoint with reproducible scene/env/step sampling noise.

This diagnostic endpoint removes request ordering as a sampling confound in
paired interventions. Noise keys are stripped before model preprocessing.
"""

import argparse
import hashlib
import logging
import os
from pathlib import Path

import numpy as np


def main() -> None:
    """Load the unchanged checkpoint and serve a localhost diagnostic endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/openpi"))
    parser.add_argument("--repo-id", default="my-zzy/ambench_contact3_30hz_v1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8043)
    args = parser.parse_args()
    root = args.work_dir.expanduser().resolve()
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["HF_LEROBOT_HOME"] = str(root / "datasets")
    os.environ["HF_HOME"] = str(root / "cache/huggingface")
    os.environ["OPENPI_DATA_HOME"] = str(root / "cache/openpi")
    logging.basicConfig(level=logging.INFO)

    from openpi import transforms
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    from ambench_learn.policies.pi.config import make_config

    cfg = make_config(work_dir=root, repo_id=args.repo_id, multitask=True, finetune_mode="full")
    keys = ["am_bench/ee_pos", "am_bench/ee_quat", "am_bench/gripper_width", "am_bench/ee_image", "prompt"]
    policy = policy_config.create_trained_policy(
        cfg,
        args.checkpoint.expanduser().resolve(),
        repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({key: key for key in keys})]),
    )
    paired = _PairedNoisePolicy(policy, (cfg.model.action_horizon, cfg.model.action_dim))
    metadata = {**policy.metadata, "diagnostic_noise_protocol": "scene_env_step_v1"}
    WebsocketPolicyServer(paired, host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


class _PairedNoisePolicy:
    """Use common Gaussian noise for corresponding requests in paired rollouts."""

    def __init__(self, policy, noise_shape: tuple[int, int]) -> None:
        self.policy = policy
        self.noise_shape = noise_shape

    def infer(self, obs: dict) -> dict:
        inputs = dict(obs)
        key = inputs.pop("diagnostic_noise_key")
        if len(key) != 3 or any(not isinstance(value, int) or value < 0 for value in key):
            raise ValueError("Expected nonnegative integer [scene_seed, env_index, control_step]")
        noise = np.random.default_rng(key).standard_normal(self.noise_shape).astype(np.float32)
        output = self.policy.infer(inputs, noise=noise)
        output["diagnostic_noise_sha256"] = hashlib.sha256(noise.tobytes()).hexdigest()
        return output


if __name__ == "__main__":
    main()
