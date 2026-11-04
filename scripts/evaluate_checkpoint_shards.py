#!/usr/bin/env python3
"""Evaluate disjoint ranges of one fixed catalog, then verify and merge them."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def atomic(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def shard_ranges(count, workers):
    if not 1 <= workers <= count:
        raise ValueError('Worker count must be between one and the rollout count')
    width, remainder = divmod(count, workers)
    start = 0
    ranges = []
    for index in range(workers):
        end = start + width + (index < remainder)
        ranges.append((start, end))
        start = end
    return ranges


def worker_command(checkpoint, catalog, folder, start, end, detector=None, scale=0.,
                   disable_action_conditioning=False, negative_prompt_mode='full_plan'):
    if disable_action_conditioning and scale != 0:
        raise ValueError('Action-conditioning ablation requires zero guidance scale')
    command = [sys.executable, str(ROOT / 'scripts/evaluate_can.py'), '--checkpoint', str(checkpoint),
               '--catalog', str(catalog), '--output', str(folder), '--device', 'cuda:0', '--resume',
               '--episode-start', str(start), '--episode-end', str(end), '--scale', str(scale),
               '--negative-prompt-mode', negative_prompt_mode]
    if detector is not None:
        command.extend(['--detector', str(detector)])
    if disable_action_conditioning:
        command.append('--disable-action-conditioning')
    return command


def merge_results(folders, catalog_path, count, output):
    expected = json.loads(catalog_path.read_text())['episodes'][:count]
    by_id = {item['id']: item for item in expected}
    rows, reference, manifests, covered = [], None, [], set()
    for folder in folders:
        run = json.loads((folder / 'run.json').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        start, end = run['episode_range']
        indices = set(range(start, end))
        if covered & indices or not 0 <= start < end <= count:
            raise ValueError('Overlapping or invalid shard ranges')
        covered.update(indices)
        signature = {k: v for k, v in run.items() if k not in {
            'episode_range', 'fingerprint', 'execution_attempts', 'policy_cuda_visible_devices'}}
        if reference is None:
            reference = signature
        if signature != reference or run['smoke'] or run['rollouts'] != count:
            raise ValueError('Shard model, protocol, implementation or catalog differs')
        part = [json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines()]
        ids = [row['episode_id'] for row in part]
        if len(ids) != len(set(ids)) or set(ids) != {item['id'] for item in expected[start:end]}:
            raise ValueError('Missing, duplicate or wrong shard episodes')
        if summary['episodes'] != len(part) or summary['successes'] != sum(row['success'] for row in part):
            raise ValueError('Shard summary disagrees with episode records')
        for row in part:
            if row['environment_seed'] != by_id[row['episode_id']]['seed']:
                raise ValueError('Episode environment seed mismatch')
        rows.extend(part)
        manifests.append(str(folder / 'run.json'))
    if covered != set(range(count)) or len(rows) != count:
        raise ValueError('Incomplete catalog coverage')
    rows.sort(key=lambda row: row['episode_id'])
    (output / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    successes = sum(row['success'] for row in rows)
    result = dict(episodes=count, successes=successes, success_rate=successes / count,
                  mean_steps=sum(row['physical_steps'] for row in rows) / count,
                  mean_recoveries=sum(row['recoveries'] for row in rows) / count,
                  smoke=False, complete_evaluation=True, checkpoint=reference['checkpoint'],
                  checkpoint_step=reference['checkpoint_step'], protocol=reference['protocol'],
                  detector=reference.get('detector'), scale=reference.get('scale', 0.),
                  action_conditioning_enabled=reference.get('action_conditioning_enabled'),
                  shard_runs=manifests, exact_paper_reproduction=False)
    atomic(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, default=ROOT / 'outputs/evaluation/can_initial_states_50.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', required=True)
    parser.add_argument('--detector', type=Path)
    parser.add_argument('--scale', type=float, default=0.)
    parser.add_argument('--negative-prompt-mode', choices=('full_plan','repeat_last_action'),default='full_plan')
    parser.add_argument('--max-retries', type=int, default=0)
    parser.add_argument('--disable-action-conditioning', action='store_true')
    args = parser.parse_args()
    spec = json.loads((ROOT / 'configs/evaluation_can.json').read_text())
    count = spec['rollouts_per_checkpoint']
    if args.max_retries < 0:
        parser.error('max-retries must be nonnegative')
    if args.scale not in spec['guidance_scales'] or args.scale != 0 and args.detector is None:
        parser.error('Scale must be in the protocol and nonzero guidance requires a detector')
    if args.detector is not None and not args.detector.is_file():
        parser.error('Detector not found')
    if args.disable_action_conditioning and args.scale != 0:
        parser.error('Action-conditioning ablation requires zero guidance scale')
    if len(set(args.gpus)) != len(args.gpus) or not 1 <= len(args.gpus) <= count:
        parser.error('Provide distinct GPUs, no more than the rollout count')
    if not args.checkpoint.is_file():
        parser.error('Checkpoint not found')
    args.output.mkdir(parents=True, exist_ok=True)
    ranges = shard_ranges(count, len(args.gpus))
    children, folders, logs = [], [], []
    state = dict(status='starting', pid=os.getpid(), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 checkpoint=str(args.checkpoint), detector=str(args.detector) if args.detector else None,
                 scale=args.scale, negative_prompt_mode=args.negative_prompt_mode,
                 total_rollouts=count, workers=[])

    def save():
        for worker, folder in zip(state['workers'], folders):
            file = folder / 'episodes.jsonl'
            worker['completed_episodes'] = len(file.read_text().splitlines()) if file.exists() else 0
        state['completed_episodes'] = sum(w['completed_episodes'] for w in state['workers'])
        state['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic(args.output / 'status.json', state)

    def start_worker(worker, log):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=worker['gpu'], JAX_PLATFORMS='cpu', OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        child = subprocess.Popen(worker['command'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        worker.update(pid=child.pid, attempts=worker.get('attempts', 0) + 1)
        return child

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        for index, gpu in enumerate(args.gpus):
            start, end = ranges[index]
            folder = args.output / 'shards' / f'gpu_{gpu}_episodes_{start:02d}_{end:02d}'
            folder.mkdir(parents=True, exist_ok=True)
            folders.append(folder)
            command = worker_command(args.checkpoint, args.catalog, folder, start, end, args.detector, args.scale,
                                     args.disable_action_conditioning,args.negative_prompt_mode)
            log = (folder / 'driver.log').open('a')
            logs.append(log)
            worker = dict(gpu=gpu, episode_range=[start, end], output=str(folder), command=command,
                          completed_episodes=0)
            child = start_worker(worker, log)
            children.append(child)
            state['workers'].append(worker)
        state['status'] = 'running'
        save()
        while True:
            for index, child in enumerate(children):
                worker = state['workers'][index]
                if child.poll() not in (None, 0) and worker['attempts'] <= args.max_retries:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    print(f"Retrying GPU {worker['gpu']} range {worker['episode_range']}; "
                          f"previous exit {child.returncode}", flush=True)
                    children[index] = start_worker(worker, logs[index])
            save()
            if all(child.poll() is not None for child in children):
                break
            time.sleep(5)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError('An evaluation shard failed')
        summary = merge_results(folders, args.catalog, count, args.output)
        state.update(status='completed', summary=str(args.output / 'summary.json'))
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        for child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for log in logs:
            log.close()
        save()


if __name__ == '__main__':
    main()
