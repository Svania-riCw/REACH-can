#!/usr/bin/env python3
"""Attach to an existing MH training process and evaluate only after verified completion."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
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
from train_evaluate_mh_baseline_2k import training_progress

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'outputs/pi05_full_sft_physical7_v4_mlp/policies/can_mh/conditional/seed_0'
DETECTOR = ROOT / 'outputs/paper_can/detectors/can_mh/seed_0_kappa4/detector.pt'
CATALOG = ROOT / 'outputs/evaluation/can_initial_states_50.json'
STEPS = (500, 1000, 1500, 2000)
BASELINE = ROOT / 'outputs/evaluation/mh_baseline_2k_8gpu_sealab1_20261103_204352_utc'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        return fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def training_alive(pid, identity, policy):
    if process_identity(pid) == identity:
        return True
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal() or process_identity(int(entry.name)) is None:
            continue
        try:
            command = (entry / 'cmdline').read_bytes().decode().split('\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if (any(Path(token).name == 'train_pi05.py' for token in command)
                and '--output' in command and command[command.index('--output') + 1] == str(policy)):
            return True
    return False


def require_training_complete(policy, log):
    rows = [json.loads(line) for line in (policy / 'performance.jsonl').read_text().splitlines() if line]
    if not any(row.get('event') == 'final_checkpoint' and row.get('step') == 2000 for row in rows):
        raise ValueError('Training exited without a complete final_checkpoint event at step 2000')
    marker = f"Training complete: {policy / 'final_model.pt'}, val_loss="
    text = log.read_text()
    if marker not in text:
        raise ValueError('Training exited without the final Training complete log marker')
    if any(token in text for token in ('Traceback (most recent call last):', 'ChildFailedError')):
        raise ValueError('Training log contains an unhandled process failure')


def verify_payload(payload, step):
    expected = dict(dataset_type='mh', policy_mode='conditional', action_condition_projection='mlp',
                    num_steps=2000, lr_decay_steps=2000, warmup_steps=1000, batch_size=64, seed=0,
                    base_prob=.45, gt_shuffle_prob=0., base_action='0', supervised_action_dims=7,
                    training_recipe='pi05_can_full_sft_physical7_v4', resume=None)
    if payload['step'] != step or any(payload['args'].get(key) != value for key, value in expected.items()):
        raise ValueError(f'Checkpoint does not match fresh MH conditional 2k recipe: step {step}')
    if len(payload['rank_rng']) != 8:
        raise ValueError('Checkpoint was not trained with eight ranks')
    model = payload['model_state_dict']
    if any(f'action_condition_proj.{layer}.weight' not in model for layer in (0, 2, 4)):
        raise ValueError('Checkpoint lacks the MLP action-condition projection')
    return hashlib.sha256(json.dumps(payload['normalization_stats'], sort_keys=True).encode()).hexdigest()


def verify_detector(path):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    calibration = payload['calibration']
    if (payload['args']['dataset_type'] != 'mh' or calibration['method'] != 'gaussian'
            or calibration.get('kappa') != 4. or not math.isclose(
                calibration['threshold'], calibration['mean'] + 4 * calibration['std'], rel_tol=1e-9)):
        raise ValueError('Expected the frozen MH Gaussian kappa=4 detector')
    return calibration


def phase_specs(output, policy, detector, gpus):
    common = [sys.executable, str(ROOT / 'scripts/evaluate_conditional_mlp_suite.py'),
              '--policy-dir', str(policy), '--detector', str(detector), '--detector-label', 'mh',
              '--gpus', *gpus, '--max-retries', '2',
              '--checkpoint-cache', str(output / 'checkpoint_hash_cache')]
    return [dict(name='wave1_policy', total_episodes=250, command=common + [
                '--output', str(output / 'wave1_policy'), '--checkpoint-steps', *map(str, STEPS),
                '--baseline-only', '--ablate-steps', '2000']),
            dict(name='wave2_recovery', total_episodes=200, command=common + [
                '--output', str(output / 'wave2_recovery'), '--checkpoint-steps', '2000', '--recovery-only'])]


def snapshot_sources(output):
    from checkpoint_utils import OPENPI_ROOT
    sources = list((ROOT / 'scripts').glob('*.py')) + list((ROOT / 'configs').glob('*.json'))
    sources += list((ROOT / 'configs').glob('*.py'))
    sources += list((OPENPI_ROOT / 'src/openpi/models_pytorch').glob('*.py'))
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT) if source.is_relative_to(ROOT) else Path('openpi/models_pytorch') / source.name
        target = output / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(source)] = digest(source)
    atomic(output / 'source_hashes.json', hashes)
    return hashes


def check_inputs(manifest):
    for source, expected in manifest['input_sha256'].items():
        if digest(source) != expected:
            raise ValueError(f'Input or evaluation source changed after watcher was armed: {source}')


def prepare_checkpoints(policy, output):
    import torch
    from evaluate_can import checkpoint_digest
    hashes, normalization, final_hash = {}, None, None
    for step, checkpoint in [(step, policy / f'checkpoints/step_{step:06d}.pt') for step in STEPS] + [
            (2000, policy / 'final_model.pt')]:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        current = verify_payload(payload, step)
        del payload
        if normalization is not None and current != normalization:
            raise ValueError('Checkpoint normalization differs between steps')
        normalization = current
        print(json.dumps(dict(event='hashing_checkpoint', path=str(checkpoint), time_utc=now())), flush=True)
        value = checkpoint_digest(checkpoint, output / 'checkpoint_hash_cache')
        if checkpoint.name != 'final_model.pt':
            hashes[str(step)] = value
        else:
            final_hash = value
    return dict(checkpoint_sha256=hashes, normalization_sha256=normalization,
                final_model_sha256=final_hash)


def baseline_comparison(folder, rows, seeds):
    if not (folder / 'audit.json').exists():
        return dict(status='pending', output=str(folder))
    try:
        audit = json.loads((folder / 'audit.json').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        previous = records(folder)
        if (not audit.get('passed') or not summary['complete_evaluation'] or summary['smoke']
                or summary['checkpoint_step'] != 2000 or summary['episodes'] != 50 or len(previous) != 50
                or summary['successes'] != sum(row['success'] for row in previous)
                or {row['episode_id']: row['environment_seed'] for row in previous} != seeds
                or len(summary['shard_runs']) != 8):
            raise ValueError('Baseline result is incomplete or uses different scenes')
        for source in summary['shard_runs']:
            run = json.loads(Path(source).read_text())
            expected = dict(policy_dataset='mh', policy_mode='baseline', checkpoint_step=2000,
                            catalog_sha256=CATALOG_SHA256, max_steps=500, smoke=False,
                            detector=None, scale=0., action_conditioning_enabled=False)
            if (any(run.get(key) != value for key, value in expected.items())
                    or run['protocol']['executed_steps'] != 8 or run['protocol']['denoising_steps'] != 10):
                raise ValueError('Baseline uses a different evaluation protocol')
        return dict(status='completed', output=str(folder), successes=summary['successes'],
                    success_rate=summary['successes'] / 50, paired=comparison(rows, previous))
    except (ValueError, KeyError, OSError) as error:
        return dict(status='incompatible', output=str(folder), error=str(error))


def write_report(output, manifest):
    check_inputs(manifest)
    catalog = json.loads(CATALOG.read_text())['episodes'][:50]
    seeds = {item['id']: item['seed'] for item in catalog}
    results, episode_rows = {}, {}
    for phase in manifest['phases']:
        folder = output / phase['name']
        suite_results = json.loads((folder / 'results.json').read_text())
        suite = json.loads((folder / 'manifest.json').read_text())
        if not suite_results['complete'] or suite_results['failures']:
            raise ValueError(f'Incomplete evaluation wave: {folder}')
        for configuration in suite['configurations']:
            result, rows = audit_configuration(configuration, manifest['checkpoint_sha256'],
                manifest['detector_sha256'], manifest['normalization_sha256'], seeds,
                policy_dataset='mh', max_steps=500)
            identity = configuration['id']
            results[identity], episode_rows[identity] = result, rows
    if len(results) != 9 or sum(len(rows) for rows in episode_rows.values()) != 450:
        raise ValueError('Expected nine configurations and 450 complete episodes')
    normal = episode_rows['mlp_step002000_none_S0']
    recovery = episode_rows['mlp_step002000_mh_S0']
    lines = ['# MH Conditional 2k 自动评测结果', '',
             '固定 50 个场景/配置；8 GPU；MH 500 物理步预算（含回退）；κ=4；完整 50×7 negative prompt。', '',
             '| 配置 | 成功/50 | 成功率 | 平均物理步 | 平均回退次数 | 回退失败 | 预算耗尽 | 相对2k无detector挽救/损失 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for identity, result in results.items():
        rows = episode_rows[identity]
        result['versus_conditional_2k_no_detector'] = comparison(rows, normal)
        if '_mh_' in identity:
            result['versus_recovery_S0'] = comparison(rows, recovery)
        paired = result['versus_conditional_2k_no_detector']
        mean_steps = sum(row['physical_steps'] for row in rows) / 50
        mean_recoveries = sum(row['recoveries'] for row in rows) / 50
        lines.append(f"| {identity} | {result['successes']}/50 | {result['successes'] / 50:.0%} | "
                     f"{mean_steps:.1f} | {mean_recoveries:.2f} | {result['rollback_failed_episodes']} | "
                     f"{result['budget_episodes']} | +{len(paired['rescued'])}/-{len(paired['lost'])} |")
    baseline = baseline_comparison(Path(manifest['baseline_output']), normal, seeds)
    lines += ['', '## 独立 MH baseline 对照', '']
    if baseline['status'] == 'completed':
        paired = baseline['paired']
        lines += [f"Baseline 2k: {baseline['successes']}/50 ({baseline['success_rate']:.0%})；"
                  f"Conditional 2k: {sum(row['success'] for row in normal)}/50；"
                  f"逐场景挽救 {len(paired['rescued'])}、损失 {len(paired['lost'])}。"]
    else:
        lines += [f"异机 baseline 状态：{baseline['status']}；本轮 450 局不等待或修改该任务。路径：`{baseline['output']}`。"]
    lines += ['', '## 分析口径', '',
              '- 主结果固定为 step 2000 的正常 conditional、无 detector；不能用测试成功率挑选主 checkpoint。',
              '- 500/1000/1500/2000 只作训练曲线诊断；MLP 旁路只作推理消融，不等同于重训 baseline。',
              '- Recovery S=0 与无 detector 比较隔离检测/回退流程；各非零 S 与 Recovery S=0 比较隔离引导收益。',
              '- 单个 seed、50 个场景的差异只作初步结论；逐场景挽救/损失 ID 见 results.json，完整步数审计见 audit.json。',
              '- 独立 MH baseline 对照见 docs/MH_CONDITIONAL_2K_EVAL_PLAN_2026-11-03.md；该异机任务不阻塞本轮评测。',
              '- 本地复现实验，不声明与论文实现完全一致。', '']
    atomic(output / 'results.json', dict(complete=True, total_episodes=450, results=results, baseline=baseline))
    atomic(output / 'audit.json', dict(passed=True, configurations=9, episodes=450, physical_budget=500,
                                     catalog_sha256=CATALOG_SHA256, checked_utc=now()))
    (output / 'REPORT.md').write_text('\n'.join(lines))
    return {identity: result['successes'] for identity, result in results.items()}


class Watcher:
    def __init__(self, policy, log, output, pid, identity, manifest, poll_seconds=10):
        self.policy, self.log, self.output = policy, log, output
        self.pid, self.identity, self.manifest, self.poll_seconds = pid, identity, manifest, poll_seconds
        self.child = None
        self.state = dict(status='waiting_training', manager_pid=os.getpid(), hostname=socket.gethostname(),
                          started_utc=now(), training_pid=pid, training_start_ticks=identity,
                          target_steps=2000, total_episodes=450, training_output=str(policy),
                          evaluation_output=str(output), gpus=manifest['gpus'])

    def save(self):
        self.state.update(training_progress(self.policy), updated_utc=now())
        for phase in self.manifest['phases']:
            status = self.output / phase['name'] / 'status.json'
            if status.exists():
                self.state.setdefault('waves', {})[phase['name']] = json.loads(status.read_text())
        self.state['completed_episodes'] = sum(wave.get('completed_episodes', 0)
                                             for wave in self.state.get('waves', {}).values())
        atomic(self.output / 'status.json', self.state)

    def execute(self, phase):
        check_inputs(self.manifest)
        self.state.update(status='evaluating', current_phase=phase['name'])
        environment = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', JAX_PLATFORMS='cpu')
        with (self.output / f"{phase['name']}.log").open('a') as stream:
            self.child = subprocess.Popen(phase['command'], cwd=ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.state['child_pid'] = self.child.pid
            self.save()
            while True:
                try:
                    code = self.child.wait(timeout=self.poll_seconds)
                    break
                except subprocess.TimeoutExpired:
                    self.save()
            if code:
                raise RuntimeError(f"{phase['name']} exited with status {code}; see its log")
        self.state.setdefault('completed_phases', []).append(phase['name'])
        self.save()

    def run(self):
        try:
            self.save()
            while training_alive(self.pid, self.identity, self.policy):
                time.sleep(self.poll_seconds)
                self.save()
            self.state.update(status='verifying_training', training_finished_utc=now())
            self.save()
            require_training_complete(self.policy, self.log)
            check_inputs(self.manifest)
            self.manifest.update(prepare_checkpoints(self.policy, self.output))
            atomic(self.output / 'manifest.json', self.manifest)
            for phase in self.manifest['phases']:
                self.execute(phase)
            self.state.update(status='auditing')
            self.save()
            self.state['success_counts'] = write_report(self.output, self.manifest)
            self.state.update(status='completed', finished_utc=now())
        except BaseException as error:
            self.state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                              error=str(error), finished_utc=now())
            raise
        finally:
            if self.child is not None and self.child.poll() is None:
                os.killpg(self.child.pid, signal.SIGTERM)
                try:
                    self.child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(self.child.pid, signal.SIGKILL)
                    self.child.wait()
            self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-dir', type=Path, default=POLICY)
    parser.add_argument('--training-log', type=Path, required=True)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--detector', type=Path, default=DETECTOR)
    parser.add_argument('--baseline-output', type=Path, default=BASELINE)
    parser.add_argument('--gpus', nargs='+', default=list(map(str, range(8))))
    args = parser.parse_args()
    if len(args.gpus) != 8 or len(set(args.gpus)) != 8:
        parser.error('Eight distinct GPUs are required')
    policy, log, output, detector = [path.resolve() for path in (
        args.policy_dir, args.training_log, args.output, args.detector)]
    identity = process_identity(args.training_pid)
    if identity is None:
        parser.error('Attach while the original training process is still running')
    command = Path(f'/proc/{args.training_pid}/cmdline').read_bytes().decode().split('\0')
    if (not any(Path(token).name == 'train_pi05.py' for token in command)
            or '--output' not in command or command[command.index('--output') + 1] != str(policy)):
        parser.error('Training PID does not belong to the requested policy output')
    if digest(CATALOG) != CATALOG_SHA256:
        parser.error('Fixed evaluation catalog has changed')
    calibration = verify_detector(detector)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((output / name).exists() for name in ('manifest.json', 'status.json', 'wave1_policy', 'wave2_recovery')):
            parser.error('Use a fresh evaluation output directory; existing evaluations are not overwritten')
        sources = snapshot_sources(output)
        sources.update({str(path): digest(path) for path in (detector, CATALOG, policy / 'config.json',
                                                            policy / 'action_stats.json', policy / 'episode_split.json')})
        manifest = dict(started_utc=now(), training_pid=args.training_pid, training_start_ticks=identity,
                        policy_dir=str(policy), training_log=str(log), gpus=args.gpus, total_episodes=450,
                        input_sha256=sources, detector_sha256=sources[str(detector)],
                        detector=str(detector), calibration=calibration, training_restarted=False,
                        baseline_output=str(args.baseline_output.resolve()),
                        phases=phase_specs(output, policy, detector, args.gpus))
        atomic(output / 'manifest.json', manifest)

        def interrupt(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupt)
        Watcher(policy, log, output, args.training_pid, identity, manifest).run()


if __name__ == '__main__':
    main()
