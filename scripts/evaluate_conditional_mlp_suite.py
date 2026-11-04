#!/usr/bin/env python3
"""Evaluate the controlled MLP ablation on a shared catalog and disjoint GPUs."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time

from evaluate_checkpoint_shards import atomic, merge_results, shard_ranges, worker_command

ROOT = Path(__file__).resolve().parents[1]


def configuration_specs(steps, detector, baseline_only=False, recovery_only=False, ablate_steps=(),
                        detector_label='ph'):
    if baseline_only and recovery_only:
        raise ValueError('Choose either baseline-only or recovery-only')
    if recovery_only and ablate_steps:
        raise ValueError('Action-conditioning ablation requires a no-detector suite')
    if len(set(ablate_steps)) != len(ablate_steps) or not set(ablate_steps) <= set(steps):
        raise ValueError('Ablation steps must be distinct selected checkpoint steps')
    settings = [('none_S0', None, 0.), (f'{detector_label}_S2p5', detector, 2.5),
                (f'{detector_label}_S0', detector, 0.), (f'{detector_label}_S0p2', detector, .2),
                (f'{detector_label}_S4p5', detector, 4.5)]
    if baseline_only:
        settings = settings[:1]
    elif recovery_only:
        settings = sorted((setting for setting in settings if setting[1] is not None), key=lambda setting: setting[2])
    result = [dict(step=step, suffix=suffix, detector=selected, scale=scale,
                   disable_action_conditioning=False)
              for suffix, selected, scale in settings for step in steps]
    result.extend(dict(step=step, suffix='none_S0_ac_disabled', detector=None, scale=0.,
                       disable_action_conditioning=True) for step in ablate_steps)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', default=[str(index) for index in range(8)])
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--max-retries', type=int, default=2)
    parser.add_argument('--checkpoint-steps', nargs='+', type=int, default=[8000, 10000])
    parser.add_argument('--policy-dir', type=Path,
                        default=ROOT / 'outputs/pi05_full_sft_physical7_v4_mlp/policies/can_paired/conditional/seed_0')
    parser.add_argument('--detector', type=Path,
                        default=ROOT / 'outputs/paper_can/detectors/can_ph/seed_0_kappa4/detector.pt')
    parser.add_argument('--detector-label', choices=('ph', 'mh', 'paired'), default='ph',
                        help='Label detector-backed configurations in output IDs')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--recovery-only', action='store_true')
    selection.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--ablate-steps', nargs='+', type=int, default=[])
    parser.add_argument('--checkpoint-cache', type=Path)
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or args.max_retries < 0:
        parser.error('Need distinct GPUs and a nonnegative retry count')
    if min(args.checkpoint_steps) < 1 or len(set(args.checkpoint_steps)) != len(args.checkpoint_steps):
        parser.error('Checkpoint steps must be positive and distinct')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        parser.error('Suite already exists; use a fresh output directory')
    policy = args.policy_dir.resolve()
    detector = args.detector.resolve()
    catalog = ROOT / 'outputs/evaluation/can_initial_states_50.json'
    protocol = json.loads((ROOT / 'configs/evaluation_can.json').read_text())
    count = protocol['rollouts_per_checkpoint']
    configurations, workers = [], []
    try:
        selected = configuration_specs(args.checkpoint_steps, detector, args.baseline_only,
                                       args.recovery_only, args.ablate_steps, args.detector_label)
    except ValueError as error:
        parser.error(str(error))
    for setting in selected:
        step = setting['step']
        suffix, selected_detector, scale = setting['suffix'], setting['detector'], setting['scale']
        checkpoint = policy / 'checkpoints' / f'step_{step:06d}.pt'
        for required in (checkpoint, catalog, selected_detector):
            if required is not None and not required.is_file():
                raise FileNotFoundError(required)
        identity = f'mlp_step{step:06d}_{suffix}'
        folder = output / identity
        configuration = dict(id=identity, checkpoint=str(checkpoint), checkpoint_step=step,
                             detector=str(selected_detector) if selected_detector else None,
                             scale=scale, disable_action_conditioning=setting['disable_action_conditioning'],
                             output=str(folder), shards=[])
        if args.checkpoint_cache:
            cache = folder / 'shards' / 'checkpoint_hash_cache'
            cache.mkdir(parents=True, exist_ok=True)
            for cached in args.checkpoint_cache.glob('*.json'):
                shutil.copy2(cached, cache / cached.name)
        for start, end in shard_ranges(count, 2):
            shard = folder / 'shards' / f'episodes_{start:02d}_{end:02d}'
            shard.mkdir(parents=True, exist_ok=True)
            command = worker_command(checkpoint, catalog, shard, start, end, selected_detector, scale,
                                     setting['disable_action_conditioning'])
            if args.video:
                command.append('--video')
            configuration['shards'].append(str(shard))
            workers.append(dict(configuration=identity, output=str(shard), command=command,
                                episode_range=[start, end], status='pending', attempts=0))
        configurations.append(configuration)
    manifest = dict(hostname=socket.gethostname(), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    gpus=args.gpus, catalog=str(catalog), protocol=protocol, configurations=configurations,
                    video=args.video, rollouts_per_configuration=count, total_rollouts=count * len(configurations),
                    max_retries=args.max_retries, training_resumed=False)
    manifest['policy_dir'] = str(policy)
    atomic(output / 'manifest.json', manifest)
    active, completed, failures = {}, {}, []

    def save(state='running'):
        for worker in workers:
            records = Path(worker['output']) / 'episodes.jsonl'
            worker['completed_episodes'] = len(records.read_text().splitlines()) if records.exists() else 0
        atomic(output / 'status.json', dict(status=state, pid=os.getpid(), workers=workers,
               completed_episodes=sum(worker['completed_episodes'] for worker in workers),
               total_episodes=manifest['total_rollouts'], completed_configurations=list(completed), failures=failures,
               updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        atomic(output / 'results.json', dict(results=completed, failures=failures,
                                            complete=len(completed) == len(configurations)))

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    state = 'running'
    try:
        while True:
            for gpu, (process, worker, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                worker['exit_code'] = code
                if code == 0:
                    worker['status'] = 'completed'
                elif worker['attempts'] <= args.max_retries:
                    worker['status'] = 'pending'
                    print(json.dumps(dict(event='retry', gpu=gpu, worker=worker)), flush=True)
                else:
                    worker['status'] = 'failed'
                    failures.append(dict(configuration=worker['configuration'], output=worker['output'], exit_code=code))
            for configuration in configurations:
                identity = configuration['id']
                selected = [worker for worker in workers if worker['configuration'] == identity]
                if identity not in completed and all(worker['status'] == 'completed' for worker in selected):
                    summary = merge_results([Path(folder) for folder in configuration['shards']],
                                            catalog, count, Path(configuration['output']))
                    completed[identity] = summary
                    print(json.dumps(dict(event='configuration_completed', id=identity,
                                          successes=summary['successes'], episodes=summary['episodes'])), flush=True)
            for gpu in args.gpus:
                if gpu in active:
                    continue
                worker = next((item for item in workers if item['status'] == 'pending'), None)
                if worker is None:
                    break
                log = (Path(worker['output']) / 'driver.log').open('a')
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, JAX_PLATFORMS='cpu',
                                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1',
                                   PYTHONDONTWRITEBYTECODE='1')
                try:
                    process = subprocess.Popen(worker['command'], cwd=ROOT, env=environment,
                                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                worker.update(status='running', gpu=gpu, pid=process.pid, attempts=worker['attempts'] + 1)
                active[gpu] = (process, worker, log)
                print(json.dumps(dict(event='shard_started', gpu=gpu, configuration=worker['configuration'],
                                      episode_range=worker['episode_range'], pid=process.pid)), flush=True)
            save()
            if not active and not any(worker['status'] == 'pending' for worker in workers):
                state = 'failed' if failures else 'completed'
                break
            time.sleep(5)
    except BaseException as error:
        state = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        failures.append(dict(error=str(error), type=type(error).__name__))
        raise
    finally:
        for process, worker, log in active.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                worker['status'] = 'interrupted'
            log.close()
        save(state)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
