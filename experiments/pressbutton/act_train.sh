#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root after activating your Isaac Lab environment.
DATASET_DIR="${1:?Pass a canonical LeRobot dataset directory}"
RUN_DIR="${2:?Pass a new output directory}"
export HF_HOME="$RUN_DIR/cache/huggingface"
export TORCH_HOME="$RUN_DIR/cache/torch"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TQDM_DISABLE=1
export ACCELERATE_MIXED_PRECISION=no
python -u -m ambench_learn.policies.act.train \
  --dataset.repo_id=my-zzy/pressbutton_act_train \
  --dataset.root="$DATASET_DIR" --dataset.video_backend=pyav \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.use_amp=false --policy.chunk_size=30 --policy.n_action_steps=8 \
  --policy.optimizer_lr=0.0001 --policy.optimizer_lr_backbone=0.00001 \
  --policy_target_hz=30 --batch_size=32 --num_workers=4 \
  --steps=20000 --log_freq=200 --save_freq=5000 --eval_freq=0 \
  --wandb.enable=false --seed=42 --output_dir="$RUN_DIR/train"
