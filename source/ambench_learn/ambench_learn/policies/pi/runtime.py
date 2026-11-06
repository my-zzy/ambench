# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Local setup commands using the pinned OpenPI training and serving implementations."""

import argparse
import dataclasses
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path


def main(command: str | None = None, arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    if command is None:
        parser.add_argument("command", choices=["doctor", "stats", "train", "smoke", "infer", "serve", "client"])
    else:
        parser.set_defaults(command=command)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--exp-name", default="pressbutton_lora")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--artifact-dir", type=Path, help="Directory for inference/diagnostic outputs.")
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/openpi"))
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--variant", choices=["pilot", "corrected"], default="pilot")
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--finetune-mode", choices=["lora", "full"], default="lora")
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--synthetic-data", action="store_true", help="Smoke-test random weights and fake inputs only.")
    args = parser.parse_args(arguments)
    if args.synthetic_data and args.command != "smoke":
        parser.error("--synthetic-data is only allowed for the bounded smoke command")
    os.environ["JAX_PLATFORMS"] = "cpu" if args.command in {"stats", "client"} else "cuda"
    logging.basicConfig(level=logging.INFO)

    import numpy as np

    ROOT = args.work_dir.expanduser().resolve()
    DATASET = args.repo_id or (
        "my-zzy/pressbutton_30hz_terminal_v2" if args.variant == "corrected" else "my-zzy/pressbutton_30hz"
    )
    os.environ["HF_LEROBOT_HOME"] = str(ROOT / "datasets")
    os.environ["HF_HOME"] = str(ROOT / "cache/huggingface")
    os.environ["OPENPI_DATA_HOME"] = str(ROOT / "cache/openpi")
    from openpi.training import config as config_module

    from ambench_learn.policies.pi.config import REPO, make_config

    artifact_dir = args.artifact_dir.resolve() if args.artifact_dir is not None else ROOT / "validation"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "doctor":
        from importlib import metadata

        import jax
        import jax.numpy as jnp
        import torch

        devices = jax.devices()
        assert all(device.platform == "gpu" for device in devices), devices
        value = np.asarray(jax.jit(lambda x: x @ x)(jnp.eye(32)))
        assert np.allclose(value, np.eye(32))
        totals = jax.pmap(lambda x: jax.lax.psum(x, "devices"), axis_name="devices")(
            jnp.arange(len(devices), dtype=jnp.float32)
        )
        assert np.allclose(np.asarray(totals), sum(range(len(devices)))), totals
        report = {
            "jax_devices": [str(device) for device in devices],
            "jax_gpu_matmul": "passed",
            "jax_device_sum": "passed",
            "torch_version": torch.__version__,
            "packages": {
                name: metadata.version(name)
                for name in ["jax", "jaxlib", "flax", "orbax-checkpoint", "lerobot", "openpi", "openpi-client"]
            },
        }
        (artifact_dir / "runtime_validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return

    cfg = make_config(
        work_dir=ROOT,
        corrected=args.variant == "corrected",
        repo_id=DATASET,
        pretrained_checkpoint=args.pretrained_checkpoint,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        steps=2 if args.command == "smoke" else args.steps,
        resume=args.resume,
        finetune_mode=args.finetune_mode,
        fsdp_devices=args.fsdp_devices,
        num_workers=args.num_workers,
    )
    if args.synthetic_data:
        from openpi.training.weight_loaders import NoOpWeightLoader

        cfg = dataclasses.replace(
            cfg,
            name="synthetic_" + cfg.name,
            data=config_module.FakeDataConfig(),
            weight_loader=NoOpWeightLoader(),
            # FakeDataset generates arrays with JAX; keep it in the parent process.
            num_workers=0,
            policy_metadata={"synthetic_setup_check": True, "task_success_evaluated": False},
        )
    if args.command == "stats" and args.variant == "corrected":
        from ambench_learn.policies.pi.stable_stats import compute_stable_stats

        report = compute_stable_stats(ROOT, DATASET, artifact_dir / "normalization_validation.json")
        print(json.dumps(report, indent=2))
        return
    if args.command == "stats":
        if (cfg.assets_dirs / DATASET / "norm_stats.json").exists():
            raise FileExistsError("Statistics already exist; use a new work directory for a new experiment")
        # A batch of one includes every frame when computing full-dataset statistics.
        config_module._CONFIGS_DICT[cfg.name] = dataclasses.replace(cfg, batch_size=1)
        # Spawned data-loader workers need a normal importable module name.
        sys.path.insert(0, str(REPO / "ext/openpi/scripts"))
        stats = importlib.import_module("compute_norm_stats")
        stats.main(cfg.name)
        stats_path = cfg.assets_dirs / DATASET / "norm_stats.json"
        values = json.loads(stats_path.read_text())["norm_stats"]
        for key, fields in values.items():
            assert all(np.isfinite(value).all() for value in fields.values()), key
            assert np.all(np.asarray(fields["std"]) >= 0), key
        info = json.loads((ROOT / "datasets" / DATASET / "meta/info.json").read_text())
        report = {
            "frames": info["total_frames"],
            "episodes": info["total_episodes"],
            "finite": True,
            "state_dim": len(values["state"]["mean"]),
            "action_dim": len(values["actions"]["mean"]),
            "path": str(stats_path),
        }
        (artifact_dir / "normalization_validation.json").write_text(json.dumps(report, indent=2) + "\n")
        return
    if args.command in {"train", "smoke"}:
        if not args.synthetic_data:
            dataset_root = ROOT / "datasets" / DATASET
            norm_root = Path(cfg.data.assets.assets_dir) if cfg.data.assets.assets_dir else cfg.assets_dirs
            required = [
                dataset_root / "meta" / filename
                for filename in ("info.json", "episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl")
            ]
            required.extend([
                norm_root / DATASET / "norm_stats.json",
                ROOT / "cache/openpi/big_vision/paligemma_tokenizer.model",
            ])
            if not args.resume:
                required.append(Path(cfg.weight_loader.params_path) / "_METADATA")
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Complete and verify the artifact transfer before training. Missing: " + ", ".join(missing)
                )
            info = json.loads((dataset_root / "meta/info.json").read_text())
            episode_files = list((dataset_root / "data").rglob("*.parquet"))
            if len(episode_files) != info["total_episodes"]:
                raise ValueError(
                    f"Incomplete dataset: {len(episode_files)} episode files, expected {info['total_episodes']}"
                )
        # Only redirect upstream's hard-coded cache path. Keep ext/openpi untouched.
        source_path = REPO / "ext/openpi/scripts/train.py"
        original = source_path.read_text()
        old = 'str(epath.Path("~/.cache/jax").expanduser())'
        assert original.count(old) == 1, "Review changed upstream cache handling"
        local_source = original.replace(old, 'str(epath.Path(config.assets_base_dir).parent / "cache/jax")')
        runtime = ROOT / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        local_path = runtime / "train_local_cache.py"
        local_path.write_text(local_source)
        sys.path.insert(0, str(runtime))
        train = importlib.import_module("train_local_cache")
        (artifact_dir / "train_config.txt").write_text(repr(cfg) + "\n")
        started = time.monotonic()
        train.main(cfg)
        report = {
            "command": args.command,
            "steps": cfg.num_train_steps,
            "batch_size": cfg.batch_size,
            "finetune_mode": args.finetune_mode,
            "fsdp_devices": cfg.fsdp_devices,
            "synthetic_data": args.synthetic_data,
            "resumed": args.resume,
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint_dir": str(cfg.checkpoint_dir),
            "completed": True,
            "task_success_evaluated": False,
        }
        (cfg.checkpoint_dir / "local_run_summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print("AMBENCH_TRAIN_COMPLETED " + json.dumps(report), flush=True)
        return

    def example() -> dict:
        from openpi import transforms
        from openpi.training import data_loader

        data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
        dataset = data_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)
        sample = transforms.compose(data_cfg.repack_transforms.inputs)(dataset[0])
        sample.pop("actions", None)
        return {key: np.asarray(value) if key != "prompt" else value for key, value in sample.items()}

    if args.command == "client":
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        client = WebsocketClientPolicy("127.0.0.1", args.port)
        started = time.monotonic()
        result = client.infer(example())
        actions = np.asarray(result["actions"])
        assert actions.shape == (30, 8) and np.isfinite(actions).all(), actions.shape
        report = {
            "metadata": client.get_server_metadata(),
            "action_shape": list(actions.shape),
            "finite": True,
            "elapsed_seconds": time.monotonic() - started,
        }
        (artifact_dir / "websocket_validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return

    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.policies import policy_config
    from openpi.shared import normalize

    checkpoint = args.checkpoint
    norm_stats = None
    if checkpoint is None:
        # Wiring test of pretrained weights, not a PressButton-finetuned policy.
        checkpoint = args.pretrained_checkpoint or ROOT / "pretrained/pi0_base"
        cfg = dataclasses.replace(cfg, model=Pi0Config(action_horizon=30))
        norm_stats = normalize.load(cfg.assets_dirs / DATASET)
    checkpoint = checkpoint.resolve()
    # The generic simulator client may include a base camera. Match the cameras used in training.
    input_keys = ["am_bench/ee_pos", "am_bench/ee_quat", "am_bench/gripper_width", "am_bench/ee_image", "prompt"]
    if cfg.data.include_base_image:
        input_keys.append("am_bench/base_image")
    input_selection = transforms.Group(inputs=[transforms.RepackTransform({key: key for key in input_keys})])
    policy = policy_config.create_trained_policy(
        cfg, checkpoint, norm_stats=norm_stats, repack_transforms=input_selection
    )
    if args.command == "serve":
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer

        WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port, metadata=policy.metadata).serve_forever()
        return
    latencies = []
    sample = example()
    np.savez(artifact_dir / "recorded_example.npz", **sample)
    for _ in range(3):
        started = time.monotonic()
        result = policy.infer(sample)
        actions = np.asarray(result["actions"])
        assert actions.shape == (30, 8) and np.isfinite(actions).all(), actions.shape
        latencies.append(time.monotonic() - started)
    np.save(artifact_dir / "inference_actions.npy", actions)
    report = {
        "checkpoint": str(checkpoint),
        "action_shape": list(actions.shape),
        "finite": True,
        "inference_seconds": latencies,
        "task_success_evaluated": False,
    }
    (artifact_dir / "inference_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
