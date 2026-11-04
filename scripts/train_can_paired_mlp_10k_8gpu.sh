#!/bin/bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export DATASET_TYPE=paired POLICY_MODE=conditional NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export GLOBAL_BATCH_SIZE=64 GRADIENT_ACCUMULATION_STEPS=1
export BASE_PROB=0.45 GT_SHUFFLE_PROB=0 BASE_ACTION=0
export LR=2.5e-5 MIN_LR=2.5e-6 WARMUP=1000 WEIGHT_DECAY=1e-10 BETA1=0.9 BETA2=0.95
export PRECISION=float32 SEED=0 GRADIENT_CHECKPOINTING=1
export NUM_WORKERS=4 CHECKPOINT_EVERY=500 EVAL_EVERY=100 LOG_EVERY=10
export DATA="$PROJECT_ROOT/data/paper/can_paired_image.hdf5"
export METADATA="$PROJECT_ROOT/data/robomimic/can/paired/low_dim_v141.hdf5"
export CHECKPOINT="$PROJECT_ROOT/checkpoints/pi05_base_pytorch"
export OUTPUT=${OUTPUT:-$PROJECT_ROOT/outputs/pi05_full_sft_physical7_v4_mlp/policies/can_paired/conditional/seed_0}
export MASTER_PORT=${MASTER_PORT:-29611}
exec bash "$PROJECT_ROOT/scripts/train.sh" --num-steps 10000 --lr-decay-steps 13850 \
    --action-condition-projection mlp --monitor-samples 4 --monitor-seed 2026 "$@"
