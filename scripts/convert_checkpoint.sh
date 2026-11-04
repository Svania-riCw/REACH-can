#!/bin/bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OPENPI_ROOT=${OPENPI_ROOT:-$PROJECT_ROOT/../RoboTwin/XPolicyLab/policy/Pi_05/openpi}
PYTHON=${PYTHON:-$OPENPI_ROOT/.venv/bin/python}
export OPENPI_ROOT JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
exec "$PYTHON" "$PROJECT_ROOT/scripts/convert_checkpoint.py" \
    --checkpoint "${CHECKPOINT:-${PI05_CKPT:-$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_base}}" \
    --output "${OUTPUT:-$PROJECT_ROOT/checkpoints/pi05_base_pytorch}" "$@"
