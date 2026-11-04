"""Verify kappa4 recalibration and physical recovery with the prompt bypassed."""
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
from error_detector import calibrate
from eval_policy import Policy
from evaluate_can import rollout
import evaluate_paired_mlp_ablation as evaluation
from evaluate_paired_mlp_ablation import configurations, load_reference
from recalibrate_detector_kappa4 import recalibrate
from test_evaluation import FakeSim, OnceDetector, settings


class PairedAblationTests(unittest.TestCase):
    def test_mh_recovery_only_uses_four_gpus_per_group_and_no_ph_groups(self):
        groups = configurations(Path('/output'), list(map(str, range(8))), 'mh', recovery_only=True)
        self.assertEqual([group['id'] for group in groups], ['mh_kappa4_S0_mlp_on', 'mh_kappa4_S0_mlp_off'])
        self.assertEqual([group['gpus'] for group in groups], [['0', '1', '2', '3'], ['4', '5', '6', '7']])
        self.assertTrue(all('can_mh' in group['detector'] and 'kappa4' in group['detector'] for group in groups))
        self.assertTrue(all(group['scale'] == 0 and group['checkpoint_step'] == 2000 for group in groups))
        with self.assertRaises(ValueError):
            configurations(Path('/output'), ['0', '1', '2'], 'mh', recovery_only=True)

    def reference_fixture(self, folder):
        folder.mkdir()
        identity = dict(checkpoint_sha256='weights', normalization_sha256='stats', catalog_sha256='catalog')
        groups, results = [], {}
        for name in ('no_detector_mlp_on', 'no_detector_mlp_off',
                     'paired_kappa4_S0_mlp_on', 'paired_kappa4_S0_mlp_off'):
            output = folder / name
            output.mkdir()
            rows = [dict(episode_id=index, environment_seed=100000 + index, success=index < 35)
                    for index in range(50)]
            (output / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            results[name] = dict(episodes=50, successes=35, success_rate=.7, mean_steps=100.,
                                 mean_recoveries=0., rollback_failed_episodes=0)
            groups.append(dict(id=name, output=str(output)))
        (folder / 'manifest.json').write_text(json.dumps(dict(identity, configurations=groups)))
        (folder / 'results.json').write_text(json.dumps(dict(complete=True, results=results)))
        (folder / 'audit.json').write_text(json.dumps(dict(passed=True)))
        (folder / 'status.json').write_text(json.dumps(dict(status='completed')))
        return identity, {index: 100000 + index for index in range(50)}, rows

    def test_reused_controls_require_matching_checkpoint_and_catalog(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'reference'
            identity, seeds, rows = self.reference_fixture(folder)
            summaries, episodes = load_reference(folder, identity, seeds)
            self.assertEqual(len(summaries), 4)
            self.assertEqual(episodes['no_detector_mlp_on'], rows)
            with self.assertRaisesRegex(ValueError, 'checkpoint_sha256'):
                load_reference(folder, dict(identity, checkpoint_sha256='other'), seeds)
            with self.assertRaisesRegex(ValueError, 'catalog_sha256'):
                load_reference(folder, dict(identity, catalog_sha256='other'), seeds)

    def test_mh_report_counts_only_new_episodes_and_compares_existing_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            reference = folder / 'reference'
            identity, seeds, rows = self.reference_fixture(reference)
            catalog = folder / 'catalog.json'
            catalog.write_text(json.dumps(dict(episodes=[dict(id=key, seed=value) for key, value in seeds.items()])))
            (folder / 'source_hashes.json').write_text('{}')
            manifest = dict(identity, detector_sha256='detector', detector_dataset='mh',
                            reference_run=str(reference),
                            configurations=configurations(folder, list(map(str, range(8))), 'mh', True))
            result = dict(episodes=50, successes=35, success_rate=.7, mean_steps=100., mean_recoveries=0.,
                          rollback_failed_episodes=0, guided_decisions=0)
            with patch.object(evaluation, 'CATALOG', catalog), \
                    patch.object(evaluation, 'audit_configuration', return_value=(result, rows)):
                evaluation.write_report(folder, manifest)
            saved = json.loads((folder / 'results.json').read_text())
            self.assertEqual(saved['total_episodes'], 100)
            self.assertEqual(len(saved['results']), 2)
            self.assertEqual(len(saved['references']), 4)
            self.assertEqual(len(saved['comparisons']), 5)
            self.assertTrue(all(item == dict(rescued=[], lost=[]) for item in saved['comparisons'].values()))
            self.assertEqual(json.loads((folder / 'audit.json').read_text())['episodes'], 100)

    def test_four_groups_use_disjoint_gpu_pairs_and_only_zero_guidance(self):
        groups = configurations(Path('/output'), list(map(str, range(8))))
        self.assertEqual([group['gpus'] for group in groups], [['0', '1'], ['2', '3'], ['4', '5'], ['6', '7']])
        self.assertEqual([bool(group['detector']) for group in groups], [False, False, True, True])
        self.assertEqual([group['disable_action_conditioning'] for group in groups], [False, True, False, True])
        self.assertTrue(all(group['scale'] == 0 and group['checkpoint_step'] == 2000 for group in groups))
        self.assertTrue(all('can_paired' in group['detector'] and 'kappa4' in group['detector']
                            for group in groups if group['detector']))

    def test_recalibration_changes_only_threshold_metadata_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'kappa3/detector.pt'
            source.parent.mkdir()
            weights = {'encoder.weight': torch.arange(12).reshape(3, 4).float()}
            calibration = dict(method='gaussian', calibration_source='held_out_id_validation',
                               mean=.01, std=.002, kappa=3., threshold=.016, n_samples=20)
            torch.save(dict(model_state_dict=weights, calibration=calibration,
                            args=dict(kappa=3., dataset_type='paired'), model_config={'hidden_dim': 4}), source)
            output = Path(temporary) / 'kappa4/detector.pt'
            result = recalibrate(source, output)
            self.assertEqual(result['kappa'], 4.)
            self.assertAlmostEqual(result['threshold'], .018)
            saved = torch.load(output, weights_only=False)
            original = torch.load(source, weights_only=False)
            torch.testing.assert_close(saved['model_state_dict']['encoder.weight'], weights['encoder.weight'])
            self.assertEqual(original['calibration'], calibration)
            self.assertEqual(saved['model_config'], original['model_config'])
            self.assertFalse(saved['recalibration']['ae_weights_changed'])
            self.assertEqual(saved['args']['kappa'], 4.)
            with self.assertRaises(FileExistsError):
                recalibrate(source, output)
        default = calibrate([.01, .02, .03])
        self.assertEqual(default['kappa'], 4.)

    def test_disabled_projection_still_retreats_whole_chunk_and_replans_at_S0(self):
        policy = Policy.__new__(Policy)
        policy.conditional = True
        policy.action_conditioning_enabled = False
        policy.device = torch.device('cpu')
        policy.config = SimpleNamespace(action_horizon=50, action_dim=32)
        policy.stats = {'action': dict(min=[-1.] * 7, max=[1.] * 7)}
        policy.base = torch.zeros(1, 1, 32)
        policy.num_steps = 1
        calls = []

        def sample_actions(*args, **kwargs):
            calls.append(kwargs)
            action = torch.zeros(1, 50, 32)
            if len(calls) == 1:
                action[:, :, 0] = .2
            return action

        policy.model = SimpleNamespace(sample_actions=sample_actions)
        trace = []
        with patch.object(policy, 'observation', return_value=None):
            result = rollout(FakeSim(), policy, OnceDetector(), dict(seed=1, state={}),
                             dict(settings(), max_steps=24, executed_steps=8), 123, trace)
        self.assertEqual(result['recoveries'], 1)
        self.assertEqual(result['decisions'], 2)
        self.assertEqual(result['physical_steps'], 24)
        self.assertEqual([event['rollback_action_index'] for event in trace if event['phase'] == 'retreat'],
                         list(range(7, -1, -1)))
        self.assertTrue(all('action_condition' not in call for call in calls))
        self.assertTrue(any(event.get('recovery_replan') for event in trace))
        self.assertFalse(any(event.get('guided') for event in trace))
        prompt = next(event['negative_prompt'] for event in trace if event['phase'] == 'retreat_start')
        self.assertEqual(np.asarray(prompt).shape, (50, 7))


if __name__ == '__main__':
    unittest.main()
