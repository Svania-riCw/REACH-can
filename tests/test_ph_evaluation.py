"""Check PH wave selection, weight-preserving ablation and automatic handoff."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
import torch
from eval_policy import Policy
from evaluate_checkpoint_shards import worker_command
from evaluate_conditional_mlp_suite import configuration_specs
import evaluate_ph_conditional_2k as evaluation


class PHEvaluationTests(unittest.TestCase):
    def test_wave1_order_matches_gpu_pairs_and_only_2k_is_ablated(self):
        settings = configuration_specs([500, 1000, 2000], 'ph.pt', baseline_only=True, ablate_steps=[2000])
        self.assertEqual([item['step'] for item in settings], [500, 1000, 2000, 2000])
        self.assertEqual([item['disable_action_conditioning'] for item in settings], [False, False, False, True])
        self.assertTrue(all(item['detector'] is None and item['scale'] == 0 for item in settings))

    def test_wave2_is_same_checkpoint_and_fixed_detector_with_sorted_scales(self):
        settings = configuration_specs([2000], 'ph_kappa4.pt', recovery_only=True)
        self.assertEqual([item['scale'] for item in settings], [0., .2, 2.5, 4.5])
        self.assertTrue(all(item['step'] == 2000 and item['detector'] == 'ph_kappa4.pt'
                            and not item['disable_action_conditioning'] for item in settings))

    def test_detector_label_is_reflected_in_configuration_ids(self):
        settings = configuration_specs([2000], 'mh_kappa4.pt', recovery_only=True,
                                       detector_label='mh')
        self.assertEqual([item['suffix'] for item in settings],
                         ['mh_S0', 'mh_S0p2', 'mh_S2p5', 'mh_S4p5'])

    def test_invalid_ablation_and_recovery_combinations_are_rejected(self):
        for kwargs in (dict(recovery_only=True, ablate_steps=[2000]),
                       dict(baseline_only=True, recovery_only=True), dict(ablate_steps=[500]),
                       dict(ablate_steps=[2000, 2000])):
            with self.assertRaises(ValueError):
                configuration_specs([2000], 'ph.pt', **kwargs)
        command = worker_command('policy.pt', 'catalog.json', 'out', 0, 25, disable_action_conditioning=True)
        self.assertIn('--disable-action-conditioning', command)
        command = worker_command('policy.pt', 'catalog.json', 'out', 0, 25, detector='ph.pt',
                                 disable_action_conditioning=True)
        self.assertIn('--detector', command)
        with self.assertRaises(ValueError):
            worker_command('policy.pt', 'catalog.json', 'out', 0, 25, detector='ph.pt', scale=2.5,
                           disable_action_conditioning=True)

    def test_ablation_loads_complete_conditional_weights_and_bypasses_condition_only(self):
        class SampleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor(.25))
                self.calls = []

            def sample_actions(self, device, observation, **kwargs):
                self.calls.append(kwargs)
                return kwargs['noise'] * 0 + (self.bias if 'action_condition' in kwargs else 0)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'weights.pt'
            args = dict(policy_mode='conditional', base_action='0')
            (checkpoint.parent / 'config.json').write_text(json.dumps(args))
            config = SimpleNamespace(max_token_len=4, action_dim=32, action_horizon=50)
            model = SampleModel()
            payload = dict(args=args, normalization_stats={'action': dict(min=[-1.] * 7, max=[1.] * 7)})
            with patch('eval_policy.resolve_checkpoint', return_value=checkpoint), \
                    patch('eval_policy.load_model', return_value=(model, config, payload)) as loader, \
                    patch('eval_policy.PaligemmaTokenizer'):
                normal = Policy(checkpoint, device='cpu')
                ablated = Policy(checkpoint, device='cpu', disable_action_conditioning=True)
            self.assertTrue(all(call.kwargs['enable_ac'] for call in loader.call_args_list))
            before = {key: value.clone() for key, value in model.state_dict().items()}
            with patch.object(normal, 'observation', return_value=None), \
                    patch.object(ablated, 'observation', return_value=None):
                standard = normal.predict({}, 123)
                disabled = ablated.predict({}, 123)
                with self.assertRaisesRegex(ValueError, 'enabled action conditioning'):
                    ablated.predict({}, 123, failed=np.zeros((50, 7)), scale=2.5)
                recovered = ablated.predict({}, 123, failed=np.zeros((50, 7)), scale=0.)
            torch.testing.assert_close(model.calls[0]['noise'], model.calls[1]['noise'])
            torch.testing.assert_close(model.calls[0]['action_condition'], torch.zeros(1, 50, 32))
            self.assertNotIn('action_condition', model.calls[1])
            self.assertNotIn('action_condition', model.calls[2])
            np.testing.assert_array_equal(recovered['action_pred'], disabled['action_pred'])
            self.assertGreater(float(np.max(standard['action_pred'] - disabled['action_pred'])), .2)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key])

    def test_phase_commands_point_to_ph_and_all_eight_gpus(self):
        phases = evaluation.phase_specs(Path('/out'), evaluation.POLICY, evaluation.DETECTOR,
                                         [str(index) for index in range(8)], 2)
        self.assertEqual([phase['total_episodes'] for phase in phases], [200, 200])
        for phase in phases:
            command = phase['command']
            self.assertEqual(command[command.index('--policy-dir') + 1], str(evaluation.POLICY))
            self.assertEqual(command[command.index('--gpus') + 1:command.index('--gpus') + 9],
                             [str(index) for index in range(8)])
        self.assertIn('--baseline-only', phases[0]['command'])
        self.assertIn('--ablate-steps', phases[0]['command'])
        self.assertIn('--recovery-only', phases[1]['command'])

    def waves_fixture(self, root):
        phases = [dict(name=f'wave{index}', command=['fake-suite', str(index)]) for index in (1, 2)]
        for phase in phases:
            folder = root / phase['name']
            folder.mkdir()
            (folder / 'results.json').write_text(json.dumps(dict(complete=True, failures=[],
                results={str(index): dict(episodes=50) for index in range(4)})))
            (folder / 'status.json').write_text(json.dumps(dict(status='completed', completed_episodes=200)))
        return dict(phases=phases)

    def test_both_waves_run_in_order_then_report_is_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.waves_fixture(root)
            processes = [SimpleNamespace(pid=101, poll=lambda: 0, returncode=0),
                         SimpleNamespace(pid=102, poll=lambda: 0, returncode=0)]
            with patch.object(evaluation.subprocess, 'Popen', side_effect=processes) as launch, \
                    patch.object(evaluation, 'write_report') as report:
                evaluation.run_waves(root, manifest)
            self.assertEqual([call.args[0] for call in launch.call_args_list],
                             [phase['command'] for phase in manifest['phases']])
            report.assert_called_once_with(root, manifest)
            status = json.loads((root / 'status.json').read_text())
            self.assertEqual(status['status'], 'completed')
            self.assertEqual(status['completed_episodes'], 400)

    def test_failed_first_wave_stops_before_second_wave(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.waves_fixture(root)
            with patch.object(evaluation.subprocess, 'Popen',
                              return_value=SimpleNamespace(pid=101, poll=lambda: 1, returncode=1)) as launch, \
                    patch.object(evaluation, 'write_report') as report:
                with self.assertRaisesRegex(RuntimeError, 'exit code 1'):
                    evaluation.run_waves(root, manifest)
            self.assertEqual(launch.call_count, 1)
            report.assert_not_called()
            self.assertEqual(json.loads((root / 'status.json').read_text())['status'], 'failed')

    def test_comparison_preserves_rescued_and_lost_episode_ids(self):
        current = [dict(episode_id=index, success=index != 1) for index in range(3)]
        baseline = [dict(episode_id=index, success=index != 0) for index in range(3)]
        self.assertEqual(evaluation.comparison(current, baseline), dict(rescued=[0], lost=[1]))
        with self.assertRaises(ValueError):
            evaluation.comparison(current[:-1], baseline)

    def test_report_audit_counts_full_chunk_retreat_and_checks_step_accounting(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            protocol = dict(executed_steps=8, denoising_steps=10)
            run = dict(checkpoint_step=2000, checkpoint_sha256='weights', policy_dataset='ph',
                       policy_mode='conditional', normalization_sha256='stats',
                       catalog_sha256=evaluation.CATALOG_SHA256, action_conditioning_enabled=True,
                       max_steps=400, smoke=False, protocol=protocol, scale=2.5,
                       detector='ph.pt', detector_sha256='detector')
            (folder / 'run.json').write_text(json.dumps(run))
            (folder / 'summary.json').write_text(json.dumps(dict(successes=50, episodes=50)))
            trace = [dict(phase='forward', step=index, noise_seed=0, guided=False) for index in range(1, 9)]
            trace += [dict(phase='detection', step=8, triggered=True),
                      dict(phase='retreat_start', step=8, chunk_steps=8,
                           negative_prompt_source='last_full_action_pred', negative_prompt_steps=50,
                           negative_prompt=[[0.] * 7 for index in range(50)])]
            for index in range(7, -1, -1):
                trace += [dict(phase='retreat', step=16 - index, rollback_action_index=index),
                          dict(phase='retreat_result', step=16 - index, rollback_action_index=index, reached=True)]
            trace += [dict(phase='forward', step=17, noise_seed=1009, guided=True)]
            rows = [dict(episode_id=index, environment_seed=100000 + index, physical_steps=17,
                         success=True, recoveries=1, decisions=2, termination='success') for index in range(50)]
            (folder / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            for row in rows:
                (folder / f"episode_{row['episode_id']:03d}_trace.json").write_text(json.dumps(trace))
            configuration = dict(output=str(folder), shards=[str(folder)], checkpoint_step=2000,
                                 disable_action_conditioning=False, detector='ph.pt', scale=2.5)
            seeds = {row['episode_id']: row['environment_seed'] for row in rows}
            result, audited = evaluation.audit_configuration(configuration, {'2000': 'weights'},
                                                             'detector', 'stats', seeds)
            self.assertEqual(len(audited), 50)
            self.assertEqual(result['complete_chunk_retreats'], 50)
            self.assertEqual(result['retreat_steps'], 400)
            self.assertEqual(result['guided_decisions'], 50)
            self.assertEqual(result['mean_decisions'], 2)
            trace[-1]['step'] = 19
            (folder / 'episode_000_trace.json').write_text(json.dumps(trace))
            with self.assertRaisesRegex(ValueError, 'step budget or accounting'):
                evaluation.audit_configuration(configuration, {'2000': 'weights'}, 'detector', 'stats', seeds)


if __name__ == '__main__':
    unittest.main()
