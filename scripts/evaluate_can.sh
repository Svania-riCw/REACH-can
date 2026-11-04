#!/bin/bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OPENPI_ROOT=${OPENPI_ROOT:-$PROJECT_ROOT/../RoboTwin/XPolicyLab/policy/Pi_05/openpi}
PYTHON=${PYTHON:-$OPENPI_ROOT/.venv/bin/python}
export JAX_PLATFORMS=cpu OPENPI_ROOT
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
exec "$PYTHON" "$PROJECT_ROOT/scripts/evaluation_matrix.py" "$@"
