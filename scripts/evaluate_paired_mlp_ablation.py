"""Evaluate paired 2k projection ablations with a selected kappa4 detector."""
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
ORIGINAL = ROOT.parent / 'reach_can_training'
CHECKPOINT = ORIGINAL / 'outputs/pi05_full_sft_physical7_v4_mlp/policies/can_paired/conditional/seed_0/checkpoints/step_002000.pt'
CATALOG = ORIGINAL / 'outputs/evaluation/can_initial_states_50.json'
REFERENCE = ORIGINAL / 'outputs/evaluation/paired_2k_paired_detector_mlp_ablation_sealab1_20261103_193538_utc'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def detector_path(dataset):
    if dataset not in ('paired', 'ph', 'mh'):
        raise ValueError('Unknown detector dataset')
    return ORIGINAL / f'outputs/paper_can/detectors/can_{dataset}/seed_0_kappa4/detector.pt'


def configurations(output, gpus, detector_type='paired', recovery_only=False):
    settings = [(f'{detector_type}_kappa4_S0_mlp_on', True, False),
                (f'{detector_type}_kappa4_S0_mlp_off', True, True)]
    if not recovery_only:
        settings = [('no_detector_mlp_on', False, False), ('no_detector_mlp_off', False, True)] + settings
    if not gpus or len(gpus) % len(settings):
        raise ValueError('GPU count must divide evenly across configurations')
    width = len(gpus) // len(settings)
    return [dict(id=identity, checkpoint=str(CHECKPOINT), checkpoint_step=2000,
                 detector=str(detector_path(detector_type)) if recovery else None, scale=0.,
                 detector_dataset=detector_type if recovery else None,
                 disable_action_conditioning=disabled, output=str(output / identity),
                 gpus=gpus[index * width:(index + 1) * width], shards=[])
            for index, (identity, recovery, disabled) in enumerate(settings)]


def prepare(output, detector_type='paired'):
    from evaluate_can import checkpoint_digest, digest
    from checkpoint_utils import OPENPI_ROOT
    import torch

    if digest(CATALOG) != CATALOG_SHA256:
        raise ValueError('Fixed catalog changed')
    payload = torch.load(CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    if (payload['step'] != 2000 or payload['args']['dataset_type'] != 'paired'
            or payload['args']['policy_mode'] != 'conditional'):
        raise ValueError('Expected paired conditional step2000')
    normalization = hashlib.sha256(json.dumps(payload['normalization_stats'], sort_keys=True).encode()).hexdigest()
    atomic(output / 'checkpoint_args.json', payload['args'])
    del payload
    path = detector_path(detector_type)
    detector = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    calibration = detector['calibration']
    if (detector['args']['dataset_type'] != detector_type or calibration['kappa'] != 4.
            or calibration['threshold'] != calibration['mean'] + 4. * calibration['std']):
        raise ValueError(f'Expected {detector_type} kappa4 detector')
    cache = output / 'checkpoint_hash_cache'
    cache.mkdir()
    previous = ORIGINAL / 'outputs/evaluation/conditional_mlp2k_ph_kappa4_chunk8_fullpred_8gpu_20261103/mlp_step002000_ph_S0/shards/checkpoint_hash_cache'
    for file in previous.glob('*.json'):
        shutil.copy2(file, cache / file.name)
    checkpoint_hash = checkpoint_digest(CHECKPOINT, cache)
    source_hashes = {}
    sources = list((ROOT / 'scripts').glob('*.py')) + list((ROOT / 'configs').glob('*.*'))
    sources += list((ROOT / 'tests').glob('*.py')) + list((OPENPI_ROOT / 'src/openpi/models_pytorch').glob('*.py'))
    for source in sources:
        if not source.is_file():
            continue
        relative = source.relative_to(ROOT) if source.is_relative_to(ROOT) else Path('openpi/models_pytorch') / source.name
        target = output / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hashes[str(source)] = digest(source)
    atomic(output / 'source_hashes.json', source_hashes)
    shutil.copy2(CATALOG, output / 'catalog.json')
    atomic(output / 'detector_calibration.json', calibration)
    return dict(checkpoint_sha256=checkpoint_hash, detector_sha256=digest(path), detector_dataset=detector_type,
                normalization_sha256=normalization, catalog_sha256=CATALOG_SHA256,
                detector_calibration=calibration)


def load_reference(folder, manifest, seeds):
    saved = json.loads((folder / 'manifest.json').read_text())
    audit = json.loads((folder / 'audit.json').read_text())
    state = json.loads((folder / 'status.json').read_text())
    if state['status'] != 'completed' or not audit['passed']:
        raise ValueError('Reference evaluation must be complete and audited')
    for key in ('checkpoint_sha256', 'normalization_sha256', 'catalog_sha256'):
        if saved[key] != manifest[key]:
            raise ValueError(f'Reference evaluation has different {key}')
    saved_results = json.loads((folder / 'results.json').read_text())
    if not saved_results['complete']:
        raise ValueError('Reference results are incomplete')
    summaries, episodes = {}, {}
    for configuration in saved['configurations']:
        identity = configuration['id']
        rows = records(Path(configuration['output']))
        if len(rows) != 50 or {row['episode_id']: row['environment_seed'] for row in rows} != seeds:
            raise ValueError('Reference catalog coverage differs')
        summary = saved_results['results'][identity]
        if summary['episodes'] != 50 or summary['successes'] != sum(row['success'] for row in rows):
            raise ValueError('Reference summary differs from episode records')
        summaries[identity], episodes[identity] = summary, rows
    for identity in ('no_detector_mlp_on', 'no_detector_mlp_off'):
        if identity not in summaries:
            raise ValueError(f'Reference lacks {identity}')
    return summaries, episodes


def write_report(output, manifest):
    for source, expected in json.loads((output / 'source_hashes.json').read_text()).items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source changed after launch: {source}')
    seeds = {item['id']: item['seed'] for item in json.loads(CATALOG.read_text())['episodes'][:50]}
    results, episodes = {}, {}
    for configuration in manifest['configurations']:
        summary, rows = audit_configuration(configuration, {'2000': manifest['checkpoint_sha256']},
                                            manifest['detector_sha256'], manifest['normalization_sha256'],
                                            seeds, policy_dataset='paired')
        if summary['guided_decisions'] != 0:
            raise ValueError('S=0 ablation unexpectedly used negative guidance')
        identity = configuration['id']
        results[identity], episodes[identity] = summary, rows
    references = {}
    if manifest.get('reference_run'):
        references, previous = load_reference(Path(manifest['reference_run']), manifest, seeds)
        episodes.update(previous)
    detector_type = manifest['detector_dataset']
    enabled = f'{detector_type}_kappa4_S0_mlp_on'
    disabled = f'{detector_type}_kappa4_S0_mlp_off'
    pairs = [(disabled, enabled), (enabled, 'no_detector_mlp_on'), (disabled, 'no_detector_mlp_off')]
    if 'no_detector_mlp_off' in results:
        pairs.insert(0, ('no_detector_mlp_off', 'no_detector_mlp_on'))
    if detector_type != 'paired':
        for suffix, current in [('on', enabled), ('off', disabled)]:
            previous = f'paired_kappa4_S0_mlp_{suffix}'
            if previous in episodes:
                pairs.append((current, previous))
    comparisons = {}
    for current, reference in pairs:
        comparisons[f'{current}_versus_{reference}'] = comparison(episodes[current], episodes[reference])
    count = len(results) * 50
    atomic(output / 'results.json', dict(complete=True, total_episodes=count, results=results,
                                        references=references, comparisons=comparisons))
    atomic(output / 'audit.json', dict(passed=True, episodes=count, configurations=len(results), detector_kappa=4.,
                                     guided_decisions=0, checked_utc=now()))
    lines = ['# Paired conditional 2k: inference MLP ablation', '',
             '| Configuration | Success | Mean steps | Mean recoveries | Rollback failures |',
             '| --- | --- | --- | --- | --- |']
    for identity, result in {**references, **results}.items():
        lines.append(f"| {identity} | {result['successes']}/50 ({result['success_rate']:.0%}) "
                     f"| {result['mean_steps']:.1f} | {result['mean_recoveries']:.2f} "
                     f"| {result['rollback_failed_episodes']} |")
    lines += ['', 'All four groups retain the same complete conditional checkpoint; only inference conditioning is bypassed.',
              f'New recovery groups use the {detector_type} detector at kappa=4, S=0 and whole executed-chunk physical retreat.',
              f"Existing reference results are reused from {manifest.get('reference_run') or 'none'}.",
              'Paired rescued/lost episode IDs and trace accounting: results.json and audit.json.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', default=[str(index) for index in range(8)])
    parser.add_argument('--detector-type', choices=['paired', 'ph', 'mh'], default='paired')
    parser.add_argument('--recovery-only', action='store_true')
    parser.add_argument('--reference-run', type=Path, default=REFERENCE)
    args = parser.parse_args()
    if len(args.gpus) != 8 or len(set(args.gpus)) != 8:
        parser.error('Need eight distinct GPUs')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'status.json').exists():
        parser.error('Choose a fresh output directory')
    selected = configurations(output, args.gpus, args.detector_type, args.recovery_only)
    state = dict(status='preparing', pid=os.getpid(), hostname=socket.gethostname(), completed_episodes=0,
                 total_episodes=len(selected) * 50, started_utc=now(), configurations=selected)
    atomic(output / 'status.json', state)
    active = []

    def save():
        total = 0
        for configuration in state['configurations']:
            folder = Path(configuration['output'])
            for status in folder.glob('status.json'):
                child = json.loads(status.read_text())
                configuration['suite_status'] = child['status']
                configuration['completed_episodes'] = child['completed_episodes']
                total += child['completed_episodes']
            configuration['shards'] = [str(path) for path in sorted((folder / 'shards').glob('gpu_*'))]
        state.update(completed_episodes=total, updated_utc=now())
        atomic(output / 'status.json', state)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        provenance = prepare(output, args.detector_type)
        devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                           '--format=csv,noheader,nounits'], text=True)
        for line in devices.splitlines():
            index, uuid, memory, utilization = [value.strip() for value in line.split(',')]
            if index in args.gpus and (int(memory) > 512 or int(utilization) > 5):
                raise RuntimeError(f'GPU {index} became busy; evaluation was not launched')
        manifest = dict(hostname=socket.gethostname(), server='sealab1:30183', runner=str(ROOT),
                        started_utc=now(), gpus=args.gpus, devices=devices,
                        reference_run=str(args.reference_run.resolve()) if args.recovery_only else None,
                        configurations=state['configurations'], **provenance)
        if args.recovery_only:
            seeds = {item['id']: item['seed'] for item in json.loads(CATALOG.read_text())['episodes'][:50]}
            load_reference(args.reference_run.resolve(), manifest, seeds)
        atomic(output / 'manifest.json', manifest)
        state['status'] = 'running'
        for configuration in state['configurations']:
            folder = Path(configuration['output'])
            folder.mkdir(parents=True)
            cache = folder / 'shards/checkpoint_hash_cache'
            cache.mkdir(parents=True)
            for file in (output / 'checkpoint_hash_cache').glob('*.json'):
                shutil.copy2(file, cache / file.name)
            command = [sys.executable, str(ROOT / 'scripts/evaluate_checkpoint_shards.py'),
                       '--checkpoint', str(CHECKPOINT), '--catalog', str(CATALOG), '--output', str(folder),
                       '--gpus', *configuration['gpus'], '--scale', '0', '--max-retries', '2']
            if configuration['detector']:
                command += ['--detector', configuration['detector']]
            if configuration['disable_action_conditioning']:
                command.append('--disable-action-conditioning')
            log = (folder / 'driver.log').open('a')
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            configuration.update(pid=process.pid, command=command)
            active.append((process, log, configuration))
        atomic(output / 'manifest.json', manifest)
        while any(process.poll() is None for process, log, configuration in active):
            for process, log, configuration in active:
                code = process.poll()
                if code not in (None, 0):
                    raise RuntimeError(f"{configuration['id']} failed: exit {code}")
            save()
            time.sleep(5)
        if any(process.returncode != 0 for process, log, configuration in active):
            raise RuntimeError('One or more configurations failed')
        save()
        manifest['configurations'] = state['configurations']
        atomic(output / 'manifest.json', manifest)
        write_report(output, manifest)
        state.update(status='completed', finished_utc=now())
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error=str(error), error_type=type(error).__name__)
        raise
    finally:
        for process, log, configuration in active:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            log.close()
        save()


if __name__ == '__main__':
    main()
