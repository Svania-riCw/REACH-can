#!/bin/bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-$PROJECT_ROOT/../RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python}
PAPER_OUTPUT=${PAPER_OUTPUT:-$PROJECT_ROOT/outputs/pi05_full_sft_physical7_v4}
"$PYTHON" - "$PAPER_OUTPUT/status.json" <<'PY'
import json
from pathlib import Path
import sys
path = Path(sys.argv[1])
if not path.exists():
    print(f'No paper experiment has started yet: {path}')
else:
    status = json.loads(path.read_text())
    print(f"Protocol: {status['protocol']}; status: {status['status']}")
    for name, job in status['jobs'].items():
        print(f"{name}: {job['status']} {job.get('artifact', job.get('log', ''))}")
    if status.get('error'):
        print(f"Error: {status['error']}")
PY
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv || true
fi
