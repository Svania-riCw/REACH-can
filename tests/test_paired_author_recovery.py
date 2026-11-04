"""Verify author recovery branching, cache provenance and the two-detector retest."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
import torch
from evaluate_can import rollout
from evaluate_paired_author_recovery import audit_author_trace, detector_path, phase_specs
from recalibrate_detector_kappa4 import recalibrate
from test_evaluation import FakePolicy, FakeSim, OnceDetector, settings


class AuthorRecoveryTests(unittest.TestCase):
    def test_kappa2_selects_derived_detectors_without_changing_gpu_allocation(self):
        gpus = [str(index) for index in range(8)]
        previous = phase_specs(Path('/tmp/evaluation'), gpus)
        current = phase_specs(Path('/tmp/evaluation'), gpus, 2.)
        for original, derived in zip(previous, current):
            self.assertEqual(original['gpus'], derived['gpus'])
            self.assertIn('kappa2', derived['command'][derived['command'].index('--detector') + 1])
            original_command = original['command'].copy()
            derived_command = derived['command'].copy()
            original_command[original_command.index('--detector') + 1] = 'same-detector-slot'
            derived_command[derived_command.index('--detector') + 1] = 'same-detector-slot'
            self.assertEqual(original_command, derived_command)
        with self.assertRaises(ValueError):
            detector_path('ph', 3.)

    def test_kappa2_reuses_weights_and_statistics_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'kappa4/detector.pt'
            source.parent.mkdir()
            calibration = dict(method='gaussian', calibration_source='held_out_id_validation',
                               mean=.01, std=.002, kappa=4., threshold=.018, n_samples=20)
            weights = {'encoder.weight': torch.arange(12).reshape(3, 4).float()}
            torch.save(dict(model_state_dict=weights, calibration=calibration,
                            args=dict(kappa=4., dataset_type='paired')), source)
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            output = Path(temporary) / 'kappa2/detector.pt'
            result = recalibrate(source, output, kappa=2.)
            self.assertEqual(result['kappa'], 2.)
            self.assertAlmostEqual(result['threshold'], .014)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_hash)
            saved = torch.load(output, weights_only=False)
            torch.testing.assert_close(saved['model_state_dict']['encoder.weight'], weights['encoder.weight'])
            self.assertEqual(saved['recalibration']['source_sha256'], original_hash)
            self.assertFalse(saved['recalibration']['ae_weights_changed'])
            for key in ('mean', 'std', 'n_samples', 'calibration_source'):
                self.assertEqual(saved['calibration'][key], calibration[key])
            for invalid in (-1., float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    recalibrate(source, Path(temporary) / 'invalid.pt', invalid)

    def test_gpu_allocation_is_disjoint_and_four_per_detector(self):
        phases = phase_specs(Path('/tmp/evaluation'), [str(index) for index in range(8)])
        self.assertEqual([phase['name'] for phase in phases], ['ph', 'paired'])
        self.assertEqual(phases[0]['gpus'], ['0', '1', '2', '3'])
        self.assertEqual(phases[1]['gpus'], ['4', '5', '6', '7'])
        for phase in phases:
            self.assertEqual(phase['total_episodes'], 200)
            self.assertIn('--recovery-only', phase['command'])
            self.assertIn('2000', phase['command'])
        for invalid in (['0'] * 8, ['0', '1']):
            with self.assertRaises(ValueError):
                phase_specs(Path('/tmp/evaluation'), invalid)

    def make_trace(self):
        trace = []
        rollout(FakeSim(), FakePolicy(), OnceDetector(), dict(seed=1, state={}), settings(), 123, trace)
        return trace

    def test_trace_audit_checks_post_retreat_detection_and_normal_cache(self):
        counts = audit_author_trace(self.make_trace(), .5, 0.)
        self.assertEqual(counts, dict(normal_cache_updates=2, recovery_replans=1, post_retreat_detections=1))

    def test_trace_audit_rejects_a_changed_recovery_prompt(self):
        trace = self.make_trace()
        event = next(item for item in trace if item['phase'] == 'recovery_replan')
        event['negative_prompt'][0][0] += .1
        with self.assertRaisesRegex(ValueError, 'normal-only cache'):
            audit_author_trace(trace, .5, 0.)

    def test_trace_audit_rejects_wrong_previous_flag_and_guided_cache_update(self):
        trace = self.make_trace()
        next(item for item in trace if item['phase'] == 'detection')['previous_triggered'] = True
        with self.assertRaisesRegex(ValueError, 'flag'):
            audit_author_trace(trace, .5, 0.)
        trace = self.make_trace()
        guided = next(index for index, item in enumerate(trace)
                      if item['phase'] == 'forward' and item['recovery_replan'])
        event = copy.deepcopy(next(item for item in trace if item['phase'] == 'normal_cache_update'))
        event['prediction_noise_seed'] = trace[guided]['noise_seed']
        trace.insert(guided + 1, event)
        with self.assertRaisesRegex(ValueError, 'refreshed'):
            audit_author_trace(trace, .5, 0.)

    def test_longer_deadline_allows_a_slow_servo_to_reach_history(self):
        class SlowSim(FakeSim):
            def step(self, action):
                return super().step(np.asarray(action) * .35)
        short = dict(settings(), max_steps=100)
        failed = rollout(SlowSim(), FakePolicy(), OnceDetector(), dict(seed=1, state={}), short, 123, [])
        self.assertEqual(failed['termination'], 'rollback_failed')
        trace = []
        longer = dict(short, rollback_max_control_steps=20)
        result = rollout(SlowSim(), FakePolicy(), OnceDetector(), dict(seed=1, state={}), longer, 123, trace)
        self.assertNotEqual(result['termination'], 'rollback_failed')
        targets = [item for item in trace if item['phase'] == 'retreat_result']
        self.assertEqual(len(targets), 4)
        self.assertTrue(all(item['reached'] for item in targets))


if __name__ == '__main__':
    unittest.main()
