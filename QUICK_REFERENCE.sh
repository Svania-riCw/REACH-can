#!/bin/bash
cat <<'TEXT'
cd /2025233210/lmx/reach_can_training
bash scripts/prepare_can_data.sh --dataset-types ph mh paired
bash scripts/prepare_data.sh
bash scripts/auto_pipeline.sh --plan-only
NUM_GPUS=8 bash scripts/auto_pipeline.sh --stage policies
bash scripts/auto_pipeline.sh --stage detectors
# Independent initialization; no PH-to-MH policy warm start.
DATASET_TYPE=paired POLICY_MODE=conditional NUM_GPUS=8 bash scripts/train.sh
# See README.md and configs/paper_can.json for protocol choices and limitations.
TEXT
