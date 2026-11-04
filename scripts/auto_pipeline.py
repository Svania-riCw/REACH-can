#!/usr/bin/env python3
"""Run the independent paper Can experiment matrix; never chain PH into MH."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiment_plan import build_plan
from dataset_paths import PROJECT_ROOT
from openpi_robomimic_adapter import validate_data
from data_splits import make_episode_split


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


class ExperimentRunner:
    def __init__(self, plan, status_file, wait_seconds=0, poll_seconds=60):
        self.plan, self.status_file = plan, Path(status_file)
        self.wait_seconds, self.poll_seconds = wait_seconds, poll_seconds
        self.status = dict(protocol=plan['protocol'], status='running', pid=os.getpid(),
                           started=datetime.now().isoformat(), jobs={})
        self.process = None

    def ready(self, job):
        import h5py
        started = time.monotonic()
        while True:
            try:
                with h5py.File(job['metadata'], 'r') as file:
                    expected = len(file['data'])
                validate_data(job['data'], expected)
                with h5py.File(job['data'], 'r') as file:
                    if not file.attrs.get('complete', False) or file.attrs.get('smoke_subset', False):
                        raise ValueError('Paper runs require a completed full render, not a smoke subset')
                make_episode_split(job['data'], job['dataset_type'], seed=job['seed'], metadata_path=job['metadata'])
                return
            except (OSError, KeyError, ValueError) as error:
                if time.monotonic() - started >= self.wait_seconds:
                    raise RuntimeError(f'{job["id"]}: data not ready: {error}; run scripts/prepare_can_data.sh') from error
                print(f'Waiting for {job["id"]}: {error}', flush=True)
                time.sleep(self.poll_seconds)

    def run(self, stages=None, job_ids=None):
        stages = set(stages or ['policies', 'detectors'])
        selected = [job for job in self.plan['training_jobs'] if job['stage'] in stages
                    and (not job_ids or job['id'] in job_ids)]
        if not selected or job_ids and set(job_ids) - {job['id'] for job in selected}:
            raise ValueError('No matching jobs or unknown job IDs')
        atomic_json(self.status_file, self.status)
        try:
            for job in selected:
                output, artifact = Path(job['output']), Path(job['artifact'])
                marker = output / 'job_record.json'
                if artifact.is_file() and marker.is_file() and json.loads(marker.read_text()).get('fingerprint') == job['fingerprint']:
                    self.status['jobs'][job['id']] = {'status': 'completed', 'artifact': str(artifact), 'reused': True}
                    atomic_json(self.status_file, self.status)
                    continue
                if artifact.exists():
                    raise FileExistsError(f'Output exists without matching experiment provenance: {artifact}')
                self.ready(job)
                output.mkdir(parents=True, exist_ok=True)
                self.status['jobs'][job['id']] = {'status': 'running', 'log': str(output / 'train.log')}
                atomic_json(self.status_file, self.status)
                print(f'Starting {job["id"]}; log={output / "train.log"}', flush=True)
                env = dict(os.environ, JAX_PLATFORMS='cpu', OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '1'))
                with (output / 'train.log').open('w') as log:
                    self.process = subprocess.Popen(job['command'], cwd=PROJECT_ROOT, env=env,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    code = self.process.wait()
                    self.process = None
                if code or not artifact.is_file():
                    self.status['jobs'][job['id']]['status'] = 'failed'
                    raise RuntimeError(f'{job["id"]} exited {code}; see {output / "train.log"}')
                atomic_json(marker, job)
                self.status['jobs'][job['id']] = {'status': 'completed', 'artifact': str(artifact)}
                atomic_json(self.status_file, self.status)
            self.status['status'] = 'completed'
            return True
        except (Exception, KeyboardInterrupt) as error:
            if self.process is not None and self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
            for job in self.status['jobs'].values():
                if job['status'] == 'running':
                    job['status'] = 'failed'
            self.status.update(status='failed', error=str(error) or 'Interrupted')
            print(self.status['error'], file=sys.stderr, flush=True)
            return False
        finally:
            self.status['ended'] = datetime.now().isoformat()
            atomic_json(self.status_file, self.status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'outputs/pi05_full_sft_physical7_v4')
    parser.add_argument('--checkpoint', default=os.environ.get('CHECKPOINT', str(PROJECT_ROOT / 'checkpoints/pi05_base_pytorch')))
    parser.add_argument('--num-gpus', type=int, default=int(os.environ.get('NUM_GPUS', '8')))
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--stage', choices=['all', 'policies', 'detectors'], default='all')
    parser.add_argument('--jobs', nargs='+')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--wait-seconds', type=float, default=0)
    args = parser.parse_args()
    if args.num_workers < 0 or args.wait_seconds < 0:
        parser.error('Worker count and wait time must be nonnegative')
    plan = build_plan(checkpoint=args.checkpoint, num_gpus=args.num_gpus, seeds=args.seeds,
                      output_root=args.output_root, num_workers=args.num_workers, device=args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.plan_only:
        atomic_json(args.output_root / 'experiment_manifest.json', plan)
        print(f'{len(plan["training_jobs"])} training jobs, {len(plan["evaluation_combinations"])} evaluation combinations')
        print(args.output_root / 'experiment_manifest.json')
        return 0
    with (args.output_root / 'pipeline.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('This experiment matrix is already running')
        atomic_json(args.output_root / 'experiment_manifest.json', plan)
        def interrupt(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupt)
        stages = ['policies', 'detectors'] if args.stage == 'all' else [args.stage]
        runner = ExperimentRunner(plan, args.output_root / 'status.json', wait_seconds=args.wait_seconds)
        return 0 if runner.run(stages, args.jobs) else 1


if __name__ == '__main__':
    sys.exit(main())
