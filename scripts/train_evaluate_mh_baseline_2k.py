#!/usr/bin/env python3
"""Train a fresh MH baseline on eight GPUs, then evaluate the final 2k checkpoint."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def training_progress(folder):
    path = folder / 'performance.jsonl'
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return dict(completed_steps=max((row['step'] for row in rows), default=0),
                latest_performance=rows[-1] if rows else None)


def verify_checkpoint(folder):
    import torch
    checkpoint = folder / 'checkpoints/step_002000.pt'
    if not (folder / 'final_model.pt').is_file():
        raise ValueError('Training did not produce final_model.pt')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    saved = payload['args']
    expected = dict(dataset_type='mh', policy_mode='baseline', num_steps=2000,
                    lr_decay_steps=2000, batch_size=64, seed=0, supervised_action_dims=7,
                    training_recipe='pi05_can_full_sft_physical7_v4', resume=None)
    if payload['step'] != 2000 or any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError('Checkpoint does not match the requested fresh MH baseline 2k recipe')
    if len(payload['rank_rng']) != 8:
        raise ValueError('Checkpoint was not trained with eight ranks')
    if any(key.startswith('action_condition_proj.') for key in payload['model_state_dict']):
        raise ValueError('Baseline checkpoint unexpectedly contains action conditioning')
    stats_hash = hashlib.sha256(json.dumps(payload['normalization_stats'], sort_keys=True).encode()).hexdigest()
    return dict(path=str(checkpoint), step=2000, normalization_sha256=stats_hash)


def audit_evaluation(output, catalog, checkpoint, checkpoint_sha256):
    summary = json.loads((output / 'summary.json').read_text())
    expected = json.loads(catalog.read_text())['episodes'][:50]
    rows = [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines()]
    if (not summary['complete_evaluation'] or summary['smoke'] or summary['episodes'] != 50
            or len(rows) != 50 or [row['episode_id'] for row in rows] != [item['id'] for item in expected]
            or summary['successes'] != sum(row['success'] for row in rows)):
        raise ValueError('Evaluation is incomplete or its summary disagrees with episodes')
    catalog_hash = hashlib.sha256(catalog.read_bytes()).hexdigest()
    for row, item in zip(rows, expected):
        if row['environment_seed'] != item['seed'] or row['recoveries'] != 0 or not 0 <= row['physical_steps'] <= 500:
            raise ValueError('Wrong seeds, recovery, or MH physical budget')
    traced = set()
    if len(summary['shard_runs']) != 8:
        raise ValueError('Evaluation did not use eight workers')
    for manifest in summary['shard_runs']:
        path = Path(manifest)
        run = json.loads(path.read_text())
        required = dict(policy_dataset='mh', policy_mode='baseline', checkpoint_step=2000,
                        checkpoint=str(checkpoint['path']), checkpoint_sha256=checkpoint_sha256,
                        normalization_sha256=checkpoint['normalization_sha256'], catalog_sha256=catalog_hash,
                        max_steps=500, rollouts=50, smoke=False, detector=None, scale=0.,
                        action_conditioning_enabled=False)
        if any(run.get(key) != value for key, value in required.items()):
            raise ValueError(f'Incorrect evaluation settings: {path}')
        if run['protocol']['executed_steps'] != 8 or run['protocol']['denoising_steps'] != 10:
            raise ValueError('Unexpected execution or denoising interval')
        for row in rows:
            trace_path = path.parent / f"episode_{row['episode_id']:03d}_trace.json"
            if not trace_path.exists():
                continue
            if row['episode_id'] in traced:
                raise ValueError('Duplicate episode trace across shards')
            traced.add(row['episode_id'])
            trace = json.loads(trace_path.read_text())
            if ([entry['step'] for entry in trace] != list(range(1, row['physical_steps'] + 1))
                    or any(entry['phase'] != 'forward' or entry.get('guided') for entry in trace)):
                raise ValueError(f'Incorrect physical trace: {trace_path}')
    if traced != {row['episode_id'] for row in rows}:
        raise ValueError('Missing physical episode traces')
    audit = dict(passed=True, episodes=50, policy_dataset='mh', policy_mode='baseline',
                 checkpoint_step=2000, checked_utc=now())
    atomic(output / 'audit.json', audit)
    (output / 'REPORT.md').write_text(
        '# MH baseline 2k\n\n'
        f"Success: {summary['successes']}/50 ({summary['success_rate']:.0%}).\n\n"
        f"Mean physical steps: {summary['mean_steps']:.2f}.\n\n"
        'Eight GPU workers, fixed 50-scene catalog, 500-step MH budget, '
        'eight executed actions per prediction, ten denoising steps, no detector or recovery.\n')
    return summary


class Pipeline:
    def __init__(self, output, evaluation, catalog, gpus):
        self.output, self.evaluation, self.catalog, self.gpus = output, evaluation, catalog, gpus
        self.child = None
        self.state = dict(status='starting', manager_pid=os.getpid(), hostname=socket.gethostname(),
                          started_utc=now(), target_steps=2000, total_episodes=50, gpus=gpus,
                          training_output=str(output), evaluation_output=str(evaluation))

    def save(self):
        self.state.update(training_progress(self.output), updated_utc=now())
        status = self.evaluation / 'status.json'
        if status.exists():
            self.state['evaluation'] = json.loads(status.read_text())
        atomic(self.output / 'pipeline_status.json', self.state)

    def execute(self, stage, command, environment, log):
        self.state.update(status=stage, command=command)
        with log.open('a') as stream:
            self.child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                          stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.state['child_pid'] = self.child.pid
            self.save()
            while True:
                try:
                    result = self.child.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    self.save()
            self.save()
            if result:
                raise RuntimeError(f'{stage} exited with status {result}; see {log}')

    def run(self):
        environment = dict(os.environ, OUTPUT=str(self.output), CUDA_VISIBLE_DEVICES=','.join(self.gpus),
                           PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', JAX_PLATFORMS='cpu')
        try:
            self.execute('training', ['bash', str(ROOT / 'scripts/train_can_mh_baseline_2k_8gpu.sh')],
                         environment, self.output / 'train.log')
            self.state.update(status='verifying_checkpoint', training_finished_utc=now())
            self.save()
            checkpoint = verify_checkpoint(self.output)
            self.state['checkpoint'] = checkpoint
            from evaluate_can import checkpoint_digest
            checkpoint_sha256 = checkpoint_digest(checkpoint['path'], self.evaluation / 'shards/checkpoint_hash_cache')
            self.state['checkpoint_sha256'] = checkpoint_sha256
            command = [sys.executable, str(ROOT / 'scripts/evaluate_checkpoint_shards.py'),
                       '--checkpoint', checkpoint['path'], '--catalog', str(self.catalog),
                       '--output', str(self.evaluation), '--gpus', *self.gpus, '--max-retries', '2']
            self.execute('evaluating', command, environment, self.output / 'evaluation.log')
            self.state['summary'] = audit_evaluation(self.evaluation, self.catalog, checkpoint, checkpoint_sha256)
            self.state.update(status='completed', finished_utc=now())
        except BaseException as error:
            self.state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                              error=str(error), finished_utc=now())
            raise
        finally:
            if self.child is not None and self.child.poll() is None:
                os.killpg(self.child.pid, signal.SIGTERM)
                try:
                    self.child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(self.child.pid, signal.SIGKILL)
                    self.child.wait()
            self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluation-output', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, default=ROOT / 'outputs/evaluation/can_initial_states_50.json')
    parser.add_argument('--gpus', nargs='+', default=[str(index) for index in range(8)])
    args = parser.parse_args()
    if len(args.gpus) != 8 or len(set(args.gpus)) != 8:
        parser.error('Eight distinct GPUs are required')
    if not args.catalog.is_file():
        parser.error('Evaluation catalog is missing')
    args.output.mkdir(parents=True, exist_ok=True)
    args.evaluation_output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ('config.json', 'pipeline_status.json', 'final_model.pt')):
        parser.error('Use a fresh training output directory')
    if (args.evaluation_output / 'status.json').exists():
        parser.error('Use a fresh evaluation output directory')
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    Pipeline(args.output.resolve(), args.evaluation_output.resolve(), args.catalog.resolve(), args.gpus).run()


if __name__ == '__main__':
    main()
