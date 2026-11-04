#!/usr/bin/env python3
"""Run the PH checkpoint/ablation and recovery waves, then audit paired results."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from evaluate_checkpoint_shards import atomic

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'outputs/pi05_full_sft_physical7_v4_mlp/policies/can_ph/conditional/seed_0'
DETECTOR = ROOT / 'outputs/paper_can/detectors/can_ph/seed_0_kappa4/detector.pt'
CATALOG = ROOT / 'outputs/evaluation/can_initial_states_50.json'
CATALOG_SHA256 = '9f3a555765e617499370f26e8c7b8a85b260274bf70b6fed10ffd1548b9f246b'
BASELINES = {500: 'pi05_full_sft_ph_baseline_seed0_2k_best_validation',
             1000: 'pi05_full_sft_ph_baseline_seed0_2k_step1000',
             2000: 'pi05_full_sft_ph_baseline_seed0_2k_final'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def phase_specs(output, policy, detector, gpus, retries):
    common = [sys.executable, str(ROOT / 'scripts/evaluate_conditional_mlp_suite.py'),
              '--policy-dir', str(policy), '--detector', str(detector), '--gpus', *gpus,
              '--max-retries', str(retries), '--checkpoint-cache', str(output / 'checkpoint_hash_cache')]
    return [dict(name='wave1_policy_and_ablation', total_episodes=200,
                 command=common + ['--output', str(output / 'wave1_policy_and_ablation'),
                                   '--checkpoint-steps', '500', '1000', '2000',
                                   '--baseline-only', '--ablate-steps', '2000']),
            dict(name='wave2_recovery', total_episodes=200,
                 command=common + ['--output', str(output / 'wave2_recovery'),
                                   '--checkpoint-steps', '2000', '--recovery-only'])]


def records(folder):
    return [json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines() if line]


def comparison(rows, reference):
    current = {row['episode_id']: bool(row['success']) for row in rows}
    previous = {row['episode_id']: bool(row['success']) for row in reference}
    if current.keys() != previous.keys():
        raise ValueError('Comparison has different episode IDs')
    return dict(rescued=[key for key in current if current[key] and not previous[key]],
                lost=[key for key in current if previous[key] and not current[key]])


def audit_configuration(configuration, expected_hashes, detector_hash, normalization, expected_seeds,
                        policy_dataset='ph', max_steps=400):
    folder = Path(configuration['output'])
    rows = records(folder)
    if len(rows) != 50 or {row['episode_id']: row['environment_seed'] for row in rows} != expected_seeds:
        raise ValueError(f'Incomplete or inconsistent catalog: {folder}')
    summary = json.loads((folder / 'summary.json').read_text())
    if summary['successes'] != sum(row['success'] for row in rows):
        raise ValueError('Summary success count differs from episodes')
    counts = dict(retreat_steps=0, complete_chunk_retreats=0, rollback_failed_episodes=0,
                  budget_episodes=0, guided_decisions=0, guided_control_steps=0, detector_triggers=0)
    visited = set()
    for shard in configuration['shards']:
        shard = Path(shard)
        run = json.loads((shard / 'run.json').read_text())
        step = configuration['checkpoint_step']
        protocol = run['protocol']
        if (run['checkpoint_step'] != step or run['checkpoint_sha256'] != expected_hashes[str(step)]
                or run['policy_dataset'] != policy_dataset or run['policy_mode'] != 'conditional'
                or run['normalization_sha256'] != normalization or run['catalog_sha256'] != CATALOG_SHA256
                or run['action_conditioning_enabled'] == configuration['disable_action_conditioning']
                or run['max_steps'] != max_steps or run['smoke']
                or protocol['executed_steps'] != 8 or protocol['denoising_steps'] != 10
                or run['scale'] != configuration['scale']
                or run['detector'] != configuration['detector']
                or run['detector_sha256'] != (detector_hash if configuration['detector'] else None)):
            raise ValueError(f'Unexpected evaluation identity: {shard}')
        for row in records(shard):
            episode_id = row['episode_id']
            if episode_id in visited:
                raise ValueError('Duplicate episode across shards')
            visited.add(episode_id)
            trace = json.loads((shard / f'episode_{episode_id:03d}_trace.json').read_text())
            physical = [event for event in trace if event['phase'] in ('forward', 'retreat')]
            if ([event['step'] for event in physical] != list(range(1, row['physical_steps'] + 1))
                    or not 0 < row['physical_steps'] <= max_steps):
                raise ValueError('Physical step budget or accounting mismatch')
            counts['rollback_failed_episodes'] += row['termination'] == 'rollback_failed'
            counts['budget_episodes'] += row['termination'] == 'budget'
            counts['retreat_steps'] += sum(event['phase'] == 'retreat' for event in physical)
            guided = [event for event in physical if event.get('guided')]
            counts['guided_control_steps'] += len(guided)
            counts['guided_decisions'] += len({event['noise_seed'] for event in guided})
            counts['detector_triggers'] += sum(event['phase'] == 'detection' and event['triggered'] for event in trace)
            active_indices = None
            target_ticks = {}
            for event in trace:
                if event['phase'] == 'retreat_start':
                    prompt = event['negative_prompt']
                    if (event['negative_prompt_source'] != protocol.get('negative_prompt_source','last_full_action_pred')
                            or event['negative_prompt_steps'] != 50 or len(prompt) != 50
                            or any(len(action) != 7 for action in prompt) or event['chunk_steps'] != 8):
                        raise ValueError('Expected full 50x7 prompt and last executed 8-action chunk')
                    active_indices = list(range(7, -1, -1))
                    target_ticks = {}
                elif event['phase'] == 'retreat':
                    index = event['rollback_action_index']
                    target_ticks[index] = target_ticks.get(index, 0) + 1
                    if not active_indices or index != active_indices[0] or target_ticks[index] > protocol.get('rollback_max_control_steps',10):
                        raise ValueError('Rollback target order or per-target budget mismatch')
                elif event['phase'] == 'retreat_result':
                    if not active_indices or event['rollback_action_index'] != active_indices[0]:
                        raise ValueError('Unexpected rollback result order')
                    if event['reached']:
                        active_indices.pop(0)
                        if not active_indices:
                            counts['complete_chunk_retreats'] += 1
    if visited != set(expected_seeds):
        raise ValueError('Missing trace coverage')
    return dict(summary, mean_decisions=sum(row['decisions'] for row in rows) / 50,
                mean_retreat_steps=counts['retreat_steps'] / 50, **counts), rows


def write_report(output, manifest):
    source_hashes = json.loads((output / 'source_hashes.json').read_text())
    for source, expected_hash in source_hashes.items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f'Evaluation source changed after launch: {source}')
    catalog = json.loads(CATALOG.read_text())['episodes'][:50]
    seeds = {item['id']: item['seed'] for item in catalog}
    results, episode_rows = {}, {}
    for phase in manifest['phases']:
        suite = json.loads((output / phase['name'] / 'manifest.json').read_text())
        for configuration in suite['configurations']:
            identity = configuration['id']
            result, rows = audit_configuration(configuration, manifest['checkpoint_sha256'],
                                                manifest['detector_sha256'], manifest['normalization_sha256'], seeds)
            reference = ROOT / 'outputs/evaluation' / BASELINES[configuration['checkpoint_step']]
            result['versus_ph_baseline'] = comparison(rows, records(reference))
            results[identity], episode_rows[identity] = result, rows
    normal = episode_rows['mlp_step002000_none_S0']
    recovery = episode_rows['mlp_step002000_ph_S0']
    for identity, result in results.items():
        if identity.startswith('mlp_step002000_'):
            result['versus_conditional_2k_no_detector'] = comparison(episode_rows[identity], normal)
        if '_ph_' in identity:
            result['versus_recovery_S0'] = comparison(episode_rows[identity], recovery)
    atomic(output / 'results.json', dict(complete=True, total_episodes=400, results=results))
    atomic(output / 'audit.json', dict(passed=True, configurations=len(results), episodes=400,
                                     protocol='8 executed / 50 predicted / 10 denoising / 400 physical budget',
                                     catalog_sha256=CATALOG_SHA256, checked_utc=now()))
    lines = ['# PH conditional evaluation', '',
             '| Configuration | Success | Mean physical steps | Mean recoveries | Guided calls | Full retreats |',
             '| --- | --- | --- | --- | --- | --- |']
    for identity, result in results.items():
        lines.append(f"| {identity} | {result['successes']}/50 ({result['success_rate']:.0%}) "
                     f"| {result['mean_steps']:.1f} | {result['mean_recoveries']:.2f} "
                     f"| {result['guided_decisions']} | {result['complete_chunk_retreats']} |")
    lines += ['', 'Episode comparisons, rollback failures and guidance/control counts: `results.json`.',
              'Catalog, checkpoint identities, normalization, trace coverage and step accounting: `audit.json`.',
              'Baseline PH: step500 36/50; step1000 46/50; step2000 49/50.',
              'This is a local recovery implementation; it is not an exact author-code reproduction.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))


def prepare(output, policy, detector):
    from evaluate_can import checkpoint_digest, digest
    from checkpoint_utils import OPENPI_ROOT
    import torch

    if digest(CATALOG) != CATALOG_SHA256:
        raise ValueError('The fixed evaluation catalog changed')
    expected_seeds = {item['id']: item['seed'] for item in json.loads(CATALOG.read_text())['episodes'][:50]}
    for step, name in BASELINES.items():
        folder = ROOT / 'outputs/evaluation' / name
        baseline = json.loads((folder / 'summary.json').read_text())
        rows = records(folder)
        if (baseline['checkpoint_step'] != step or baseline['episodes'] != 50
                or len(rows) != 50 or {row['episode_id']: row['environment_seed'] for row in rows} != expected_seeds):
            raise ValueError(f'Incomplete baseline or incompatible catalog: {folder}')
    hashes, normalization = {}, None
    for step in (500, 1000, 2000):
        checkpoint = policy / 'checkpoints' / f'step_{step:06d}.pt'
        payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        args = payload['args']
        if payload['step'] != step or args['dataset_type'] != 'ph' or args['policy_mode'] != 'conditional':
            raise ValueError(f'Unexpected PH conditional checkpoint: {checkpoint}')
        current = hashlib.sha256(json.dumps(payload['normalization_stats'], sort_keys=True).encode()).hexdigest()
        if normalization is not None and current != normalization:
            raise ValueError('PH checkpoint normalization differs between steps')
        normalization = current
        del payload
        print(json.dumps(dict(event='hashing_checkpoint', step=step, time_utc=now())), flush=True)
        hashes[str(step)] = checkpoint_digest(checkpoint, output / 'checkpoint_hash_cache')
    sources = list((ROOT / 'scripts').glob('*.py')) + list((ROOT / 'tests').glob('*.py'))
    sources += list((ROOT / 'configs').glob('*.json'))
    sources += list((OPENPI_ROOT / 'src/openpi/models_pytorch').glob('*.py'))
    source_hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT) if source.is_relative_to(ROOT) else Path('openpi/models_pytorch') / source.name
        target = output / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hashes[str(source)] = digest(source)
    atomic(output / 'source_hashes.json', source_hashes)
    return dict(checkpoint_sha256=hashes, normalization_sha256=normalization,
                detector_sha256=digest(detector), catalog_sha256=CATALOG_SHA256)


def run_waves(output, manifest):
    active = None
    state = dict(status='running', pid=os.getpid(), current_phase=None, completed_episodes=0,
                 total_episodes=400, phases=manifest['phases'])

    def save():
        state['completed_episodes'] = 0
        for phase in state['phases']:
            path = output / phase['name'] / 'status.json'
            if path.exists():
                progress = json.loads(path.read_text())
                phase['completed_episodes'] = progress['completed_episodes']
                phase['suite_status'] = progress['status']
                state['completed_episodes'] += progress['completed_episodes']
        state['updated_utc'] = now()
        atomic(output / 'status.json', state)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        for phase in state['phases']:
            folder = output / phase['name']
            folder.mkdir(parents=True, exist_ok=True)
            state['current_phase'] = phase['name']
            with (folder / 'driver.log').open('a') as log:
                active = subprocess.Popen(phase['command'], cwd=ROOT, stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                phase.update(pid=active.pid, started_utc=now(), status='running')
                print(json.dumps(dict(event='wave_started', name=phase['name'], pid=active.pid)), flush=True)
                while active.poll() is None:
                    save()
                    time.sleep(5)
                phase.update(exit_code=active.returncode, finished_utc=now())
                active = None
            if phase['exit_code'] != 0:
                raise RuntimeError(f"{phase['name']} failed with exit code {phase['exit_code']}")
            result = json.loads((folder / 'results.json').read_text())
            if (not result['complete'] or result['failures'] or len(result['results']) != 4
                    or sum(item['episodes'] for item in result['results'].values()) != 200):
                raise RuntimeError(f"{phase['name']} did not finish its 200 episodes")
            phase['status'] = 'completed'
            save()
        state['current_phase'] = 'reporting'
        save()
        write_report(output, manifest)
        state.update(status='completed', current_phase=None, finished_utc=now())
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error=str(error), error_type=type(error).__name__)
        raise
    finally:
        if active is not None and active.poll() is None:
            try:
                os.killpg(active.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy-dir', type=Path, default=POLICY)
    parser.add_argument('--detector', type=Path, default=DETECTOR)
    parser.add_argument('--gpus', nargs='+', default=[str(index) for index in range(8)])
    parser.add_argument('--max-retries', type=int, default=2)
    args = parser.parse_args()
    if len(args.gpus) != 8 or len(set(args.gpus)) != 8 or args.max_retries < 0:
        parser.error('Provide eight distinct GPUs and a nonnegative retry count')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists() or (output / 'status.json').exists():
        parser.error('Choose a fresh output directory')
    atomic(output / 'status.json', dict(status='preparing', pid=os.getpid(), total_episodes=400,
                                       completed_episodes=0, updated_utc=now()))
    try:
        provenance = prepare(output, args.policy_dir.resolve(), args.detector.resolve())
    except BaseException as error:
        atomic(output / 'status.json', dict(status='failed', pid=os.getpid(), error=str(error), updated_utc=now()))
        raise
    manifest = dict(started_utc=now(), pid=os.getpid(), gpus=args.gpus,
                    policy_dir=str(args.policy_dir.resolve()), detector=str(args.detector.resolve()),
                    baseline_directories={str(step): str(ROOT / 'outputs/evaluation' / name)
                                          for step, name in BASELINES.items()},
                    phases=phase_specs(output, args.policy_dir.resolve(), args.detector.resolve(),
                                       args.gpus, args.max_retries), **provenance)
    atomic(output / 'manifest.json', manifest)
    run_waves(output, manifest)


if __name__ == '__main__':
    main()
