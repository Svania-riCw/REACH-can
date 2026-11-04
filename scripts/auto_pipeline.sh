#!/bin/bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OPENPI_ROOT=${OPENPI_ROOT:-$PROJECT_ROOT/../RoboTwin/XPolicyLab/policy/Pi_05/openpi}
PYTHON=${PYTHON:-$OPENPI_ROOT/.venv/bin/python}
export OPENPI_ROOT JAX_PLATFORMS=cpu
exec "$PYTHON" "$PROJECT_ROOT/scripts/auto_pipeline.py" "$@"
