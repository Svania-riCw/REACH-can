"""Build independent Can policy/detector jobs and the paper Table 2 matrix."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

from dataset_paths import PROJECT_ROOT, dataset_paths


def build_plan(root=PROJECT_ROOT, checkpoint=None, num_gpus=8, seeds=None, output_root=None, python=None,
               num_workers=4, device='cuda', smoke_steps=None):
    root = Path(root).resolve()
    spec = json.loads((PROJECT_ROOT / 'configs/paper_can.json').read_text())
    implementation = hashlib.sha256()
    for relative in ('scripts/train_pi05.py', 'scripts/policy_diagnostics.py', 'scripts/diagnose_steering.py',
                     'scripts/openpi_robomimic_adapter.py', 'scripts/data_splits.py',
                     'scripts/checkpoint_utils.py', 'scripts/error_detector.py', 'scripts/train_error_detector.py',
                     'configs/paper_can.json', 'configs/pi05_robomimic_config.py'):
        implementation.update(relative.encode())
        implementation.update((PROJECT_ROOT / relative).read_bytes())
    implementation_digest = implementation.hexdigest()
    seeds = spec['seeds'] if seeds is None else seeds
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Need a nonempty list of unique seeds')
    if num_gpus < 1 or spec['training']['global_batch_size'] % (num_gpus * spec['training']['gradient_accumulation_steps']):
        raise ValueError('GPU count * gradient accumulation must divide the global batch')
    output_root = Path(output_root or root / 'outputs/pi05_full_sft_physical7_v4').resolve()
    checkpoint = str(Path(checkpoint or root / 'checkpoints/pi05_base_pytorch').expanduser().resolve())
    # Collapse '..' without resolving the venv symlink into the system Python.
    python = os.path.abspath(python or sys.executable)
    jobs, artifacts = [], {}
    for seed in seeds:
        for kind in spec['policy_datasets']:
            paths = dataset_paths(kind, root)
            for mode in spec['policy_modes']:
                name = f'policy_{kind}_{mode}_seed_{seed}'
                output = output_root / f'policies/can_{kind}/{mode}/seed_{seed}'
                command = [python]
                if num_gpus > 1:
                    command += ['-m', 'torch.distributed.run', f'--nproc_per_node={num_gpus}',
                                '--master_addr=127.0.0.1', '--master_port=29500']
                command += [str(root / 'scripts/train_pi05.py'), '--data', str(paths['images']),
                            '--metadata', str(paths['metadata']), '--dataset-type', kind, '--policy-mode', mode,
                            '--checkpoint', checkpoint, '--output', str(output),
                            '--batch-size', str(spec['training']['global_batch_size']),
                            '--lr', str(spec['training']['learning_rate']),
                            '--warmup-steps', str(spec['training']['warmup_steps']),
                            '--min-lr', str(spec['training']['min_lr']),
                            '--weight-decay', str(spec['training']['weight_decay']),
                            '--beta1', str(spec['training']['betas'][0]), '--beta2', str(spec['training']['betas'][1]),
                            '--precision', spec['training']['precision'],
                            '--gradient-accumulation-steps', str(spec['training']['gradient_accumulation_steps']),
                            '--gradient-checkpointing',
                            '--log-every', str(spec['training'].get('log_every', 10)),
                            '--seed', str(seed), '--num-workers', str(num_workers), '--device', device]
                if smoke_steps:
                    command += ['--num-steps', str(smoke_steps)]
                else:
                    command += ['--epochs', str(spec['training']['epochs'])]
                if mode == 'conditional':
                    command += ['--base-action', spec['conditioning']['base_action'],
                                '--base-prob', str(spec['conditioning']['base_probability']),
                                '--gt-shuffle-prob', str(spec['conditioning']['gt_shuffle_probability'])]
                artifact = str(output / 'final_model.pt')
                jobs.append(dict(id=name, stage='policies', dataset_type=kind, mode=mode, seed=seed,
                                 command=command, data=str(paths['images']), metadata=str(paths['metadata']),
                                 initial_checkpoint=checkpoint, output=str(output), artifact=artifact))
                artifacts[name] = artifact
        for kind in spec['detector_datasets']:
            name = f'detector_{kind}_seed_{seed}'
            paths = dataset_paths(kind, root)
            output = output_root / f'detectors/can_{kind}/seed_{seed}'
            config = spec['detector']
            command = [python, str(root / 'scripts/train_error_detector.py'), '--data', str(paths['images']),
                       '--metadata', str(paths['metadata']), '--dataset-type', kind, '--output', str(output),
                       '--seed', str(seed), '--device', device, '--num-workers', str(num_workers),
                       '--epochs', str(1 if smoke_steps else config['epochs']), '--batch-size', str(config['batch_size']),
                       '--lr', str(config['learning_rate']), '--optimizer', config['optimizer'],
                       '--hidden-dim', str(config['hidden_dim']), '--latent-dim', str(config['latent_dim']),
                       '--image-size', str(config['image_size']), '--threshold-method', config['threshold_method'],
                       '--kappa', str(config['kappa'])]
            artifact = str(output / 'detector.pt')
            jobs.append(dict(id=name, stage='detectors', dataset_type=kind, seed=seed, command=command,
                             data=str(paths['images']), metadata=str(paths['metadata']),
                             output=str(output), artifact=artifact))
            artifacts[name] = artifact
    evaluations = []
    for seed in seeds:
        for kind in spec['policy_datasets']:
            start_seed=spec['evaluation']['environment_seed_start']
            common = dict(policy_dataset=kind, seed=seed,
                          rollout_seeds=list(range(start_seed,start_seed+spec['evaluation']['rollouts_per_seed'])),
                          executed_steps=spec['evaluation']['executed_steps'],max_steps=spec['evaluation']['max_steps'][kind],
                          status='pending_training_or_evaluation',evaluation_runner=str(root/'scripts/evaluation_matrix.py'))
            for mode in spec['policy_modes']:
                evaluations.append(dict(common, id=f'{kind}_{mode}_no_detector_seed_{seed}',
                    policy_checkpoint=artifacts[f'policy_{kind}_{mode}_seed_{seed}'],
                    detector_dataset=None, detector_checkpoint=None, guidance_scale=0.0,
                    recovery_enabled=False, action_prompt='none' if mode == 'baseline' else 'base'))
            for detector in spec['detector_combinations'][kind]:
                for scale in spec['guidance_scales']:
                    evaluations.append(dict(common, id=f'{kind}_detector_{detector}_S{scale}_seed_{seed}',
                        policy_checkpoint=artifacts[f'policy_{kind}_conditional_seed_{seed}'],
                        detector_dataset=detector, detector_checkpoint=artifacts[f'detector_{detector}_seed_{seed}'],
                        guidance_scale=scale, recovery_enabled=True, action_prompt='failed_action_history'))
    for job in jobs:
        job['implementation_digest'] = implementation_digest
        job['fingerprint'] = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    return dict(protocol=spec['protocol'], specification=spec, seeds=seeds,
                implementation_digest=implementation_digest,
                training_jobs=jobs, evaluation_combinations=evaluations)
