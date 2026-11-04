#!/bin/bash
set -euo pipefail
export DATASET_TYPE=mh
export NUM_GPUS=${NUM_GPUS:-8}
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" "$@"
