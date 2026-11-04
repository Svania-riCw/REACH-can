#!/bin/bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export DATASET_TYPE=mh POLICY_MODE=baseline NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export GLOBAL_BATCH_SIZE=64 GRADIENT_ACCUMULATION_STEPS=1
export LR=2.5e-5 MIN_LR=2.5e-6 WARMUP=1000 WEIGHT_DECAY=1e-10 BETA1=0.9 BETA2=0.95
export PRECISION=float32 SEED=0 GRADIENT_CHECKPOINTING=1
export NUM_WORKERS=4 CHECKPOINT_EVERY=500 EVAL_EVERY=100 LOG_EVERY=10
export DATA="$PROJECT_ROOT/data/paper/can_mh_image.hdf5"
export CHECKPOINT="$PROJECT_ROOT/checkpoints/pi05_base_pytorch"
export OUTPUT=${OUTPUT:?Set OUTPUT to a fresh MH baseline directory}
export MASTER_PORT=${MASTER_PORT:-29615}
unset METADATA RESUME
exec bash "$PROJECT_ROOT/scripts/train.sh" --num-steps 2000 --lr-decay-steps 2000 \
    --monitor-samples 4 --monitor-seed 2026 "$@"
