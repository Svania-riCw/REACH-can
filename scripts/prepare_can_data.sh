#!/bin/bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SIM_PYTHON=${SIM_PYTHON:-$PROJECT_ROOT/../reach_simulation_env/venv/bin/python}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
exec "$SIM_PYTHON" "$PROJECT_ROOT/scripts/prepare_can_data.py" "$@"
