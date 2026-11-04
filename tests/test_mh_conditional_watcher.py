"""Check safe handoff from an already running MH job to the two evaluation waves."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import torch
import watch_mh_conditional_2k as watcher


class MHConditionalWatcherTests(unittest.TestCase):
    def manifest(self, output):
        return dict(gpus=list(map(str, range(8))), input_sha256={},
                    phases=watcher.phase_specs(output, output / 'policy', output / 'mh.pt',
                                               list(map(str, range(8)))))

    def payload(self):
        return dict(step=2000, rank_rng=[None] * 8, normalization_stats={'scale': 1},
                    model_state_dict={f'action_condition_proj.{layer}.weight': None for layer in (0, 2, 4)},
                    args=dict(dataset_type='mh', policy_mode='conditional', action_condition_projection='mlp',
                              num_steps=2000, lr_decay_steps=2000, warmup_steps=1000, batch_size=64, seed=0,
                              base_prob=.45, gt_shuffle_prob=0., base_action='0', supervised_action_dims=7,
                              training_recipe='pi05_can_full_sft_physical7_v4', resume=None))

    def test_wave_matrix_is_250_then_200_on_eight_gpus(self):
        phases = self.manifest(Path('/output'))['phases']
        self.assertEqual([phase['total_episodes'] for phase in phases], [250, 200])
        first, second = [phase['command'] for phase in phases]
        self.assertEqual(first[first.index('--checkpoint-steps') + 1:first.index('--baseline-only')],
                         ['500', '1000', '1500', '2000'])
        self.assertIn('--ablate-steps', first)
        self.assertIn('--recovery-only', second)
        for command in (first, second):
            self.assertEqual(command[command.index('--detector-label') + 1], 'mh')
            self.assertEqual(command[command.index('--gpus') + 1:command.index('--max-retries')],
                             list(map(str, range(8))))

    def test_success_requires_both_final_checkpoint_event_and_log_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            performance, log = folder / 'performance.jsonl', folder / 'train.log'
            performance.write_text(json.dumps(dict(event='training', step=2000)) + '\n')
            log.write_text(f"Training complete: {folder / 'final_model.pt'}, val_loss=0.1\n")
            with self.assertRaisesRegex(ValueError, 'final_checkpoint'):
                watcher.require_training_complete(folder, log)
            performance.write_text(json.dumps(dict(event='final_checkpoint', step=2000)) + '\n')
            log.write_text('Step 2000: loss=0.1\n')
            with self.assertRaisesRegex(ValueError, 'log marker'):
                watcher.require_training_complete(folder, log)
            log.write_text(f"Training complete: {folder / 'final_model.pt'}, val_loss=0.1\n")
            watcher.require_training_complete(folder, log)
            with log.open('a') as stream:
                stream.write('ChildFailedError\n')
            with self.assertRaisesRegex(ValueError, 'process failure'):
                watcher.require_training_complete(folder, log)

    def test_checkpoint_recipe_rank_count_and_mlp_are_required(self):
        payload = self.payload()
        self.assertEqual(len(watcher.verify_payload(payload, 2000)), 64)
        payload['args']['dataset_type'] = 'ph'
        with self.assertRaisesRegex(ValueError, 'recipe'):
            watcher.verify_payload(payload, 2000)
        payload = self.payload()
        payload['rank_rng'].pop()
        with self.assertRaisesRegex(ValueError, 'eight ranks'):
            watcher.verify_payload(payload, 2000)
        payload = self.payload()
        payload['model_state_dict'].pop('action_condition_proj.4.weight')
        with self.assertRaisesRegex(ValueError, 'MLP'):
            watcher.verify_payload(payload, 2000)

    def test_detector_rejects_ph_or_kappa3(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'detector.pt'
            payload = dict(args=dict(dataset_type='mh'), calibration=dict(
                method='gaussian', kappa=4., mean=.01, std=.002, threshold=.018))
            torch.save(payload, path)
            self.assertEqual(watcher.verify_detector(path)['kappa'], 4.)
            payload['calibration']['kappa'] = 3.
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'kappa=4'):
                watcher.verify_detector(path)
            payload['calibration']['kappa'] = 4.
            payload['args']['dataset_type'] = 'ph'
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'kappa=4'):
                watcher.verify_detector(path)

    def test_input_changes_are_not_silently_used(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'source.py'
            source.write_text('original')
            manifest = dict(input_sha256={str(source): watcher.digest(source)})
            watcher.check_inputs(manifest)
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed after'):
                watcher.check_inputs(manifest)

    def test_pending_baseline_does_not_block_this_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = watcher.baseline_comparison(Path(temporary), [], {})
            self.assertEqual(result['status'], 'pending')

    def test_subprocess_failure_is_not_treated_as_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manager = watcher.Watcher(output, output / 'train.log', output, 123, 'identity', self.manifest(output))
            phase = dict(name='wave1_policy', command=[sys.executable, '-c', 'raise SystemExit(3)'])
            with self.assertRaisesRegex(RuntimeError, 'status 3'):
                manager.execute(phase)
            self.assertNotIn('completed_phases', manager.state)

    def test_complete_report_contains_nine_rows_and_450_audited_episodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = dict(self.manifest(output), checkpoint_sha256={}, detector_sha256='detector',
                            normalization_sha256='stats', baseline_output=str(output / 'baseline'))
            rows = [dict(episode_id=index, environment_seed=100000 + index, success=index < 25,
                         physical_steps=100, recoveries=0) for index in range(50)]
            catalog = output / 'catalog.json'
            catalog.write_text(json.dumps(dict(episodes=[dict(id=row['episode_id'], seed=row['environment_seed'])
                                                         for row in rows])))
            identities = ['mlp_step000500_none_S0', 'mlp_step001000_none_S0', 'mlp_step001500_none_S0',
                          'mlp_step002000_none_S0', 'mlp_step002000_none_S0_ac_disabled',
                          'mlp_step002000_mh_S0', 'mlp_step002000_mh_S0p2',
                          'mlp_step002000_mh_S2p5', 'mlp_step002000_mh_S4p5']
            for phase, selected in zip(manifest['phases'], [identities[:5], identities[5:]]):
                folder = output / phase['name']
                folder.mkdir()
                (folder / 'results.json').write_text(json.dumps(dict(complete=True, failures=[])))
                (folder / 'manifest.json').write_text(json.dumps(dict(
                    configurations=[dict(id=identity) for identity in selected])))
            summary = dict(successes=25, rollback_failed_episodes=0, budget_episodes=25)
            with patch.object(watcher, 'CATALOG', catalog), \
                    patch.object(watcher, 'audit_configuration', return_value=(summary, rows)) as audit:
                result = watcher.write_report(output, manifest)
            self.assertEqual(len(result), 9)
            self.assertEqual(audit.call_count, 9)
            self.assertTrue(all(call.kwargs == dict(policy_dataset='mh', max_steps=500)
                                for call in audit.call_args_list))
            self.assertEqual(json.loads((output / 'audit.json').read_text())['episodes'], 450)
            self.assertIn('mlp_step002000_mh_S4p5', (output / 'REPORT.md').read_text())

    def test_no_handoff_while_original_process_or_ranks_are_alive(self):
        with patch.object(watcher, 'process_identity', return_value='original'):
            self.assertTrue(watcher.training_alive(123, 'original', Path('/policy')))
        entry = MagicMock()
        entry.name = '124'
        (entry / 'cmdline').read_bytes.return_value = b'python\0/scripts/train_pi05.py\0--output\0/policy\0'
        with patch.object(watcher, 'process_identity', side_effect=[None, 'rank']), \
                patch.object(Path, 'iterdir', return_value=iter([entry])):
            self.assertTrue(watcher.training_alive(123, 'original', Path('/policy')))

    def test_successful_handoff_waits_then_runs_waves_in_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manager = watcher.Watcher(output, output / 'train.log', output, 123, 'identity',
                                      self.manifest(output), poll_seconds=.01)
            with patch.object(watcher, 'training_alive', side_effect=[True, False]), \
                    patch.object(watcher.time, 'sleep') as sleep, \
                    patch.object(watcher, 'require_training_complete') as complete, \
                    patch.object(watcher, 'prepare_checkpoints', return_value={}) as prepare, \
                    patch.object(manager, 'execute') as execute, \
                    patch.object(watcher, 'write_report', return_value={'primary': 25}):
                manager.run()
            sleep.assert_called_once()
            complete.assert_called_once()
            prepare.assert_called_once()
            self.assertEqual([call.args[0]['name'] for call in execute.call_args_list],
                             ['wave1_policy', 'wave2_recovery'])
            self.assertEqual(json.loads((output / 'status.json').read_text())['status'], 'completed')

    def test_abnormal_training_exit_never_launches_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manager = watcher.Watcher(output, output / 'train.log', output, 123, 'identity', self.manifest(output))
            with patch.object(watcher, 'training_alive', return_value=False), \
                    patch.object(watcher, 'require_training_complete', side_effect=ValueError('incomplete training')), \
                    patch.object(manager, 'execute') as execute:
                with self.assertRaisesRegex(ValueError, 'incomplete training'):
                    manager.run()
            execute.assert_not_called()
            self.assertEqual(json.loads((output / 'status.json').read_text())['status'], 'failed')

    def test_failed_first_wave_does_not_start_second_wave(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manager = watcher.Watcher(output, output / 'train.log', output, 123, 'identity', self.manifest(output))
            with patch.object(watcher, 'training_alive', return_value=False), \
                    patch.object(watcher, 'require_training_complete'), \
                    patch.object(watcher, 'prepare_checkpoints', return_value={}), \
                    patch.object(manager, 'execute', side_effect=RuntimeError('wave1 failed')) as execute:
                with self.assertRaisesRegex(RuntimeError, 'wave1 failed'):
                    manager.run()
            execute.assert_called_once()
            self.assertEqual(execute.call_args.args[0]['name'], 'wave1_policy')

    def test_mh_audit_accepts_401_steps_but_ph_default_rejects_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            run = dict(checkpoint_step=2000, checkpoint_sha256='weights', policy_dataset='mh',
                       policy_mode='conditional', normalization_sha256='stats',
                       catalog_sha256=watcher.CATALOG_SHA256, action_conditioning_enabled=True,
                       max_steps=500, smoke=False, protocol=dict(executed_steps=8, denoising_steps=10),
                       scale=0., detector=None, detector_sha256=None)
            (folder / 'run.json').write_text(json.dumps(run))
            (folder / 'summary.json').write_text(json.dumps(dict(successes=50, episodes=50)))
            rows = [dict(episode_id=index, environment_seed=100000 + index, physical_steps=401,
                         success=True, recoveries=0, decisions=51, termination='success') for index in range(50)]
            (folder / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            trace = [dict(phase='forward', step=index, guided=False) for index in range(1, 402)]
            for row in rows:
                (folder / f"episode_{row['episode_id']:03d}_trace.json").write_text(json.dumps(trace))
            configuration = dict(output=str(folder), shards=[str(folder)], checkpoint_step=2000,
                                 disable_action_conditioning=False, detector=None, scale=0.)
            seeds = {row['episode_id']: row['environment_seed'] for row in rows}
            result, audited = watcher.audit_configuration(configuration, {'2000': 'weights'},
                'detector', 'stats', seeds, policy_dataset='mh', max_steps=500)
            self.assertEqual(result['successes'], 50)
            self.assertEqual(len(audited), 50)
            with self.assertRaisesRegex(ValueError, 'evaluation identity'):
                watcher.audit_configuration(configuration, {'2000': 'weights'}, 'detector', 'stats', seeds,
                                            policy_dataset='mh')


if __name__ == '__main__':
    unittest.main()
