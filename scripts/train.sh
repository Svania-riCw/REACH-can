#!/bin/bash
# Independent paper experiment; dataset type never implicitly enables conditioning.
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OPENPI_ROOT=${OPENPI_ROOT:-$PROJECT_ROOT/../RoboTwin/XPolicyLab/policy/Pi_05/openpi}
PYTHON=${PYTHON:-$OPENPI_ROOT/.venv/bin/python}
DATASET_TYPE=${DATASET_TYPE:-mh}
POLICY_MODE=${POLICY_MODE:-baseline}
NUM_GPUS=${NUM_GPUS:-8}
case "$DATASET_TYPE" in ph|mh|paired) ;; *) echo "DATASET_TYPE must be ph, mh or paired" >&2; exit 1 ;; esac
case "$POLICY_MODE" in baseline|conditional) ;; *) echo "POLICY_MODE must be baseline or conditional" >&2; exit 1 ;; esac
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-${BATCH_SIZE:-64}}
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ && "$GLOBAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || (( GLOBAL_BATCH_SIZE % NUM_GPUS != 0 )); then
    echo "A positive global batch must be divisible by NUM_GPUS" >&2; exit 1
fi
DATA=${DATA:-$PROJECT_ROOT/data/paper/can_${DATASET_TYPE}_image.hdf5}
CHECKPOINT=${CHECKPOINT:-${PI05_CKPT:-$PROJECT_ROOT/checkpoints/pi05_base_pytorch}}
OUTPUT=${OUTPUT:-$PROJECT_ROOT/outputs/pi05_full_sft_physical7_v4/can_${DATASET_TYPE}_${POLICY_MODE}_seed_${SEED:-0}_$(date +%Y%m%d_%H%M%S)}
export OPENPI_ROOT JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
if [[ -n ${GPU:-} ]]; then export CUDA_VISIBLE_DEVICES=$GPU; fi
args=("$PROJECT_ROOT/scripts/train_pi05.py" --data "$DATA" --dataset-type "$DATASET_TYPE"
      --policy-mode "$POLICY_MODE" --checkpoint "$CHECKPOINT" --output "$OUTPUT"
      --batch-size "$GLOBAL_BATCH_SIZE" --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
      --lr "${LR:-2.5e-5}" --warmup-steps "${WARMUP:-1000}" --min-lr "${MIN_LR:-2.5e-6}"
      --weight-decay "${WEIGHT_DECAY:-1e-10}" --beta1 "${BETA1:-0.9}" --beta2 "${BETA2:-0.95}"
      --precision "${PRECISION:-float32}" --num-workers "${NUM_WORKERS:-4}" --device "${DEVICE:-cuda}"
      --checkpoint-every "${CHECKPOINT_EVERY:-500}" --eval-every "${EVAL_EVERY:-100}"
      --log-every "${LOG_EVERY:-20}" --seed "${SEED:-0}")
has_duration=0
for argument in "$@"; do
    case "$argument" in --num-steps|--num-steps=*|--epochs|--epochs=*) has_duration=1 ;; esac
done
if (( has_duration == 0 )); then
    if [[ -n ${NUM_STEPS:-} ]]; then args+=(--num-steps "$NUM_STEPS"); else args+=(--epochs "${EPOCHS:-50}"); fi
fi
if [[ -n ${METADATA:-} ]]; then args+=(--metadata "$METADATA"); fi
if [[ $POLICY_MODE == conditional ]]; then
    args+=(--base-action "${BASE_ACTION:-0}" --base-prob "${BASE_PROB:-0.45}" --gt-shuffle-prob "${GT_SHUFFLE_PROB:-0}")
fi
if [[ -n ${RESUME:-} ]]; then args+=(--resume "$RESUME"); fi
if [[ ${GRADIENT_CHECKPOINTING:-1} == 1 ]]; then args+=(--gradient-checkpointing); fi
cd "$PROJECT_ROOT"
if (( NUM_GPUS > 1 )); then
    exec "$PYTHON" -m torch.distributed.run --nproc_per_node="$NUM_GPUS" \
        --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-29500}" "${args[@]}" "$@"
fi
exec "$PYTHON" "${args[@]}" "$@"
