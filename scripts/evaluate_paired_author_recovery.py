#!/usr/bin/env python3
"""Compare PH/paired detectors with author recovery branching and normal-only caches."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from evaluate_checkpoint_shards import atomic
from evaluate_ph_conditional_2k import CATALOG_SHA256, audit_configuration, comparison, records

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path('/2025233210/lmx/reach_can_training')
POLICY = ORIGINAL / 'outputs/pi05_full_sft_physical7_v4_mlp/policies/can_paired/conditional/seed_0'
CATALOG = ORIGINAL / 'outputs/evaluation/can_initial_states_50.json'
REFERENCE = ORIGINAL / 'outputs/evaluation/paired_2k_paired_detector_mlp_ablation_sealab1_20261103_193538_utc'
KAPPA4_REFERENCE = ORIGINAL / 'outputs/evaluation/paired_conditional2k_author_recovery_timeout20_8gpu_20261104_073318_utc'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def detector_path(dataset, kappa):
    if dataset not in ('ph', 'paired') or kappa not in (2., 4.):
        raise ValueError('Expected PH/paired detector with kappa=2 or 4')
    return ORIGINAL / f'outputs/paper_can/detectors/can_{dataset}/seed_0_kappa{kappa:g}/detector.pt'


def phase_specs(output, gpus, kappa=4.):
    if len(gpus) != 8 or len(set(gpus)) != 8:
        raise ValueError('Exactly eight distinct GPUs are required')
    return [dict(name=dataset, gpus=gpus[index * 4:(index + 1) * 4], total_episodes=200,
                 command=[sys.executable, str(ROOT / 'scripts/evaluate_conditional_mlp_suite.py'),
                          '--output', str(output / dataset), '--gpus', *gpus[index * 4:(index + 1) * 4],
                          '--policy-dir', str(POLICY), '--checkpoint-steps', '2000', '--recovery-only',
                          '--detector', str(detector_path(dataset, kappa)),
                          '--detector-label', dataset, '--max-retries', '2',
                          '--checkpoint-cache', str(output / 'checkpoint_hash_cache')])
            for index, dataset in enumerate(('ph', 'paired'))]


def prepare(output, kappa=4.):
    import torch
    from checkpoint_utils import OPENPI_ROOT
    from evaluate_can import checkpoint_digest, digest

    protocol = json.loads((ROOT / 'configs/evaluation_can.json').read_text())
    required = dict(executed_steps=8, denoising_steps=10, rollback_max_control_steps=20,
                    recovery_state_machine='author_previous_detection_flag',
                    recovery_cache_update='normal_forward_only',
                    negative_prompt_source='last_normal_action_pred',
                    rollback_scope='last_normal_executed_chunk')
    if any(protocol.get(key) != value for key, value in required.items()):
        raise ValueError('Unexpected recovery protocol')
    if digest(CATALOG) != CATALOG_SHA256:
        raise ValueError('Common catalog changed')
    checkpoint = POLICY / 'checkpoints/step_002000.pt'
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    args = payload['args']
    if (payload['step'] != 2000 or args['policy_mode'] != 'conditional'
            or args['dataset_type'] != 'paired' or args['action_condition_projection'] != 'mlp'):
        raise ValueError('Expected the paired conditional MLP 2k checkpoint')
    normalization = hashlib.sha256(json.dumps(payload['normalization_stats'], sort_keys=True).encode()).hexdigest()
    del payload
    cache = output / 'checkpoint_hash_cache'
    cache.mkdir()
    for previous in (REFERENCE / 'checkpoint_hash_cache').glob('*.json'):
        shutil.copy2(previous, cache / previous.name)
    checkpoint_hash = checkpoint_digest(checkpoint, cache)
    detectors = {}
    for dataset in ('ph', 'paired'):
        detector = detector_path(dataset, kappa)
        saved = torch.load(detector, map_location='cpu', mmap=True, weights_only=False)
        calibration = saved['calibration']
        if (calibration['method'] != 'gaussian' or calibration['kappa'] != kappa
                or calibration['threshold'] != calibration['mean'] + kappa * calibration['std']):
            raise ValueError(f'Detector must use Gaussian kappa={kappa:g}')
        detectors[dataset] = dict(path=str(detector), sha256=digest(detector), calibration=calibration,
                                  recalibration=saved.get('recalibration'))
        if kappa == 2:
            original = detector_path(dataset, 4.)
            source = torch.load(original, map_location='cpu', mmap=True, weights_only=False)
            provenance = saved.get('recalibration', {})
            if provenance.get('source_sha256') != digest(original) or provenance.get('ae_weights_changed') is not False:
                raise ValueError('Missing unchanged-AE kappa4 source provenance')
            for key in ('mean', 'std', 'n_samples', 'calibration_source'):
                if calibration[key] != source['calibration'][key]:
                    raise ValueError('Kappa2 calibration must reuse the exact held-out statistics')
            if saved['model_state_dict'].keys() != source['model_state_dict'].keys():
                raise ValueError('Detector weight keys changed')
            if any(not torch.equal(value, source['model_state_dict'][key])
                   for key, value in saved['model_state_dict'].items()):
                raise ValueError('Detector weights changed during threshold recalibration')
            del source
        del saved
    sources = list((ROOT / 'scripts').glob('*.py')) + list((ROOT / 'tests').glob('*.py'))
    sources += list((ROOT / 'configs').glob('*.json')) + list((ROOT / 'configs').glob('*.py'))
    sources += list((OPENPI_ROOT / 'src/openpi/models_pytorch').glob('*.py'))
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT) if source.is_relative_to(ROOT) else Path('openpi/models_pytorch') / source.name
        target = output / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(source)] = digest(source)
    atomic(output / 'source_hashes.json', hashes)
    return dict(checkpoint_sha256={'2000': checkpoint_hash}, normalization_sha256=normalization,
                detectors=detectors, detector_kappa=kappa, catalog_sha256=CATALOG_SHA256, protocol=protocol)


def validate_kappa4_reference(manifest):
    from evaluate_can import implementation_digest

    saved = json.loads((KAPPA4_REFERENCE / 'manifest.json').read_text())
    audit = json.loads((KAPPA4_REFERENCE / 'audit.json').read_text())
    result = json.loads((KAPPA4_REFERENCE / 'results.json').read_text())
    if not audit['passed'] or not result['complete'] or audit['episodes'] != 400:
        raise ValueError('Kappa4 reference must be complete and audited')
    for key in ('checkpoint_sha256', 'normalization_sha256', 'catalog_sha256', 'protocol'):
        if manifest[key] != saved[key]:
            raise ValueError(f'Kappa4 reference differs in {key}')
    for dataset in ('ph', 'paired'):
        if manifest['detectors'][dataset]['recalibration']['source_sha256'] != saved['detectors'][dataset]['sha256']:
            raise ValueError('Kappa2 detector is not derived from the evaluated kappa4 detector')
        suite = json.loads((KAPPA4_REFERENCE / dataset / 'manifest.json').read_text())
        for configuration in suite['configurations']:
            for shard in configuration['shards']:
                run = json.loads((Path(shard) / 'run.json').read_text())
                if run['implementation_sha256'] != implementation_digest():
                    raise ValueError('Rollout implementation differs from kappa4 reference')
    return str(KAPPA4_REFERENCE)


def audit_author_trace(trace, threshold, initial_score):
    import numpy as np

    cached = None
    last_forward = None
    previous = initial_score > threshold
    expected = None
    counts = dict(normal_cache_updates=0, recovery_replans=0, post_retreat_detections=0)
    for event in trace:
        phase = event['phase']
        if phase == 'detection':
            if event['previous_triggered'] != previous:
                raise ValueError('Previous detector flag was not preserved')
            expected = 'recovery_replan' if previous else ('retreat_start' if event['triggered'] else 'forward')
            previous = event['triggered']
            counts['post_retreat_detections'] += bool(event['after_retreat'])
        elif phase in ('retreat_start', 'recovery_replan'):
            if phase != expected or cached is None:
                raise ValueError('Unexpected author recovery transition')
            prompt = np.asarray(event['negative_prompt'], dtype=np.float32)
            if (event['prediction_noise_seed'] != cached['prediction_noise_seed']
                    or hashlib.sha256(prompt.tobytes()).hexdigest() != cached['action_pred_sha256']):
                raise ValueError('Recovery prompt did not use the normal-only cache')
            counts['recovery_replans'] += phase == 'recovery_replan'
            expected = None
        elif phase == 'forward':
            if expected not in (None, 'forward'):
                raise ValueError('Recovery skipped its required branch')
            expected = None
            last_forward = event
        elif phase == 'normal_cache_update':
            if last_forward is None or last_forward['recovery_replan'] or event['prediction_noise_seed'] != last_forward['noise_seed']:
                raise ValueError('Recovery prediction refreshed the normal cache')
            cached = event
            counts['normal_cache_updates'] += 1
    return counts


def write_report(output, manifest):
    for source, expected in json.loads((output / 'source_hashes.json').read_text()).items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source changed after launch: {source}')
    seeds = {item['id']: item['seed'] for item in json.loads(CATALOG.read_text())['episodes'][:50]}
    reference = records(REFERENCE / 'no_detector_mlp_on')
    results = {}
    for dataset in ('ph', 'paired'):
        suite = json.loads((output / dataset / 'manifest.json').read_text())
        for configuration in suite['configurations']:
            result, rows = audit_configuration(configuration, manifest['checkpoint_sha256'],
                                               manifest['detectors'][dataset]['sha256'],
                                               manifest['normalization_sha256'], seeds, policy_dataset='paired')
            totals = dict(normal_cache_updates=0, recovery_replans=0, post_retreat_detections=0)
            for shard in configuration['shards']:
                folder = Path(shard)
                run = json.loads((folder / 'run.json').read_text())
                if run['protocol'] != manifest['protocol']:
                    raise ValueError('Protocol differs from launch manifest')
                for row in records(folder):
                    trace = json.loads((folder / f"episode_{row['episode_id']:03d}_trace.json").read_text())
                    counts = audit_author_trace(trace, run['threshold'], row['initial_score'])
                    for key in totals:
                        totals[key] += counts[key]
            results[configuration['id']] = dict(result, **totals, versus_no_detector=comparison(rows, reference))
    kappa = manifest.get('detector_kappa', 4.)
    kappa4_results = {}
    if manifest.get('kappa4_reference'):
        previous = Path(manifest['kappa4_reference'])
        kappa4_results = json.loads((previous / 'results.json').read_text())['results']
        for dataset in ('ph', 'paired'):
            suite = json.loads((output / dataset / 'manifest.json').read_text())
            for configuration in suite['configurations']:
                identity = configuration['id']
                results[identity]['versus_kappa4'] = comparison(records(Path(configuration['output'])),
                                                                records(previous / dataset / identity))
    atomic(output / 'results.json', dict(complete=True, episodes=400, detector_kappa=kappa,
                                        results=results, kappa4_results=kappa4_results))
    atomic(output / 'audit.json', dict(passed=True, episodes=400, detector_kappa=kappa, checked_utc=now()))
    lines = ['# Paired conditional 2k: author recovery state/cache + longer servo timeout', '',
             '| Detector | S | Success | Mean physical steps | Mean retreats | Rollback failures |',
             '| --- | --- | --- | --- | --- | --- |']
    for identity, result in results.items():
        lines.append(f"| {identity} | {result['scale']} | {result['successes']}/50 ({result['success_rate']:.0%}) "
                     f"| {result['mean_steps']:.2f} | {result['mean_recoveries']:.2f} | {result['rollback_failed_episodes']} |")
    lines += ['', f'Fixed 50 initial states, paired conditional MLP step 2000, kappa={kappa:g}, execute 8, denoise 10.',
              'Both caches update only for normal forward chunks. Post-retreat detection and previous-flag branches match the author.',
              'EEF closed-loop retreat remains local: 20 ticks/target, 5mm/0.05rad, all physical steps count toward 400.',
              'The timeout failure still terminates the episode; this is not exact author action playback.',
              'Old no-detector reference: 35/50 (70%).', '']
    if kappa4_results:
        lines += ['## Kappa2 versus matched kappa4 reference', '',
                  '| Configuration | Kappa4 | Kappa2 | Difference | Rescued IDs | Lost IDs |',
                  '| --- | --- | --- | --- | --- | --- |']
        for identity, result in results.items():
            previous = kappa4_results[identity]
            paired = result['versus_kappa4']
            difference = 100 * (result['success_rate'] - previous['success_rate'])
            lines.append(f"| {identity} | {previous['successes']}/50 | {result['successes']}/50 "
                         f"| {difference:+.0f}pp | {paired['rescued']} | {paired['lost']} |")
        lines += ['', 'Same checkpoint, AE weights, held-out calibration data, catalog, rollout implementation and recovery protocol; only kappa changes.', '']
    else:
        lines += ['This retest jointly changes state/cache handling and timeout relative to older recovery runs.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', default=[str(index) for index in range(8)])
    parser.add_argument('--kappa', type=float, choices=(2., 4.), default=4.)
    args = parser.parse_args()
    output = args.output.resolve()
    phases = phase_specs(output, args.gpus, args.kappa)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        parser.error('Use a new output directory')
    manifest = dict(prepare(output, args.kappa), phases=phases, hostname=socket.gethostname(), started_utc=now(),
                    gpus=args.gpus, source_root=str(ROOT), total_episodes=400)
    if args.kappa == 2:
        manifest['kappa4_reference'] = validate_kappa4_reference(manifest)
    atomic(output / 'manifest.json', manifest)
    active = []
    state = dict(status='running', pid=os.getpid(), completed_episodes=0, total_episodes=400, phases=phases)

    def save():
        state['completed_episodes'] = 0
        for phase in phases:
            file = output / phase['name'] / 'status.json'
            if file.exists():
                saved = json.loads(file.read_text())
                phase.update(completed_episodes=saved['completed_episodes'], suite_status=saved['status'])
                state['completed_episodes'] += saved['completed_episodes']
        state['updated_utc'] = now()
        atomic(output / 'status.json', state)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        for phase in phases:
            folder = output / phase['name']
            folder.mkdir()
            log = (folder / 'driver.log').open('a')
            child = subprocess.Popen(phase['command'], cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            phase.update(pid=child.pid, started_utc=now())
            active.append((child, log))
        while True:
            save()
            if any(child.poll() not in (None, 0) for child, log in active):
                raise RuntimeError('A detector evaluation suite failed')
            if all(child.poll() == 0 for child, log in active):
                break
            time.sleep(5)
        write_report(output, manifest)
        state.update(status='completed', finished_utc=now())
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        for child, log in active:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            log.close()
        save()


if __name__ == '__main__':
    main()
