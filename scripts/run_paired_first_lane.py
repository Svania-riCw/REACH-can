#!/usr/bin/env python3
"""Run one Paired-first lane with independent status, GPU assignment, and logs."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from auto_pipeline import ExperimentRunner, atomic_json
from dataset_paths import PROJECT_ROOT
from experiment_plan import build_plan


def lane_plan(lane, seed, artifact_root, num_workers=4):
    plan = build_plan(num_gpus=8, seeds=[seed], output_root=artifact_root, num_workers=num_workers)
    jobs = [j for j in plan['training_jobs'] if
            (j['stage'] == 'policies' and j['dataset_type'] == 'paired' and j['mode'] == lane)
            or (lane == 'detectors' and j['stage'] == 'detectors')]
    if lane == 'detectors':
        jobs.sort(key=lambda j: ['ph', 'paired', 'mh'].index(j['dataset_type']))
    for job in jobs:
        command = job['command']
        if job['stage'] == 'policies':
            port = 29621 if lane == 'baseline' else 29622
            command[command.index('--master_port=29500')] = f'--master_port={port}'
        else:
            command[command.index('--device') + 1] = 'cpu'
            command[command.index('--num-workers') + 1] = '2'
        job['execution_environment'] = {key: os.environ.get(key, '') for key in
            ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}
        job.pop('fingerprint')
        job['fingerprint'] = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    plan['training_jobs'] = jobs
    plan['evaluation_combinations'] = [e for e in plan['evaluation_combinations'] if e['policy_dataset'] == 'paired']
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', choices=['baseline', 'conditional', 'detectors'], required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--artifact-root', type=Path, default=PROJECT_ROOT / 'outputs/pi05_full_sft_physical7_v4')
    parser.add_argument('--run-root', type=Path, default=PROJECT_ROOT / 'outputs/paired_first_full_sft_physical7_v4')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    lane_root = args.run_root / f'seed_{args.seed}' / args.lane
    lane_root.mkdir(parents=True, exist_ok=True)
    plan = lane_plan(args.lane, args.seed, args.artifact_root)
    atomic_json(lane_root / 'experiment_manifest.json', plan)
    if args.plan_only:
        return 0
    # Each policy now uses all eight GPUs. Policy lanes serialize; CPU AEs remain independent.
    import fcntl
    with ExitStack() as stack:
        lock = stack.enter_context((lane_root / 'lane.lock').open('w'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.lane != 'detectors':
            visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            if visible is not None and len(visible.split(',')) != 8:
                raise ValueError('Each policy lane requires all eight visible GPUs; remove an old four-GPU mask')
            gpu_lock = stack.enter_context((args.run_root / 'eight_gpu_policy.lock').open('w'))
            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        runner = ExperimentRunner(plan, lane_root / 'status.json', wait_seconds=24 * 3600, poll_seconds=60)
        def interrupt(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupt)
        if not runner.run():
            return 1
        if args.lane == 'conditional':
            checkpoint = plan['training_jobs'][0]['artifact']
            report = lane_root / 'steering.json'
            command = [sys.executable, str(PROJECT_ROOT / 'scripts/diagnose_steering.py'),
                       '--checkpoint', checkpoint, '--output', str(report), '--samples', '4']
            atomic_json(lane_root / 'diagnostic_status.json', dict(status='running', command=command))
            with (lane_root / 'steering.log').open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait()
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                    raise
            atomic_json(lane_root / 'diagnostic_status.json',
                        dict(status='completed' if code == 0 else 'failed', returncode=code,
                             command=command, report=str(report)))
            return code
    return 0


if __name__ == '__main__':
    sys.exit(main())
