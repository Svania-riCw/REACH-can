#!/bin/bash
set -euo pipefail

TRAINING_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAINING_PYTHON=${PYTHON:-$TRAINING_ROOT/../RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python}
BASELINE_OUTPUT=$TRAINING_ROOT/outputs/pi05_full_sft_physical7_v4/policies/can_paired/baseline/seed_0

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1

exec "$TRAINING_PYTHON" -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT:-29631}" \
  "$TRAINING_ROOT/scripts/train_pi05.py" \
  --data "$TRAINING_ROOT/data/paper/can_paired_image.hdf5" \
  --metadata "$TRAINING_ROOT/data/robomimic/can/paired/low_dim_v141.hdf5" \
  --dataset-type paired \
  --policy-mode baseline \
  --checkpoint "$TRAINING_ROOT/checkpoints/pi05_base_pytorch" \
  --output "$BASELINE_OUTPUT" \
  --batch-size 64 \
  --epochs 50 \
  --lr 2.5e-5 \
  --warmup-steps 1000 \
  --min-lr 2.5e-6 \
  --weight-decay 1e-10 \
  --beta1 0.9 \
  --beta2 0.95 \
  --precision float32 \
  --gradient-accumulation-steps 1 \
  --gradient-checkpointing \
  --log-every 10 \
  --seed 0 \
  --num-workers 4 \
  --device cuda \
  --base-action 0 \
  --base-prob 0.3 \
  --gt-shuffle-prob 0 \
  --resume "$BASELINE_OUTPUT/checkpoints/step_008000.pt"
