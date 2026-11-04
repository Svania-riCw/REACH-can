"""Paired split integrity, explicit policy modes and AE calibration roundtrip."""
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import h5py
import numpy as np
import torch
from data_splits import make_episode_split
from error_detector import ErrorDetector, ImageAutoencoder, calibrate, pack_images
from train_error_detector import main as train_detector
from train_pi05 import parse_args
from test_training import fixture


class PaperTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / 'paired.hdf5'
        fixture(self.path)
        with h5py.File(self.path, 'a') as file:
            for i in range(4):
                demo = file[f'data/demo_{i}']
                demo.attrs['pair_id'] = str(i // 2)
                demo.create_dataset('rewards', data=np.full(4, i % 2, dtype=np.float32))

    def tearDown(self): self.temp.cleanup()

    def test_paired_split_preserves_good_bad_pairs_and_all_data(self):
        split = make_episode_split(self.path, 'paired')
        self.assertEqual(set(split['train']) | set(split['val']), {f'demo_{i}' for i in range(4)})
        for i in (0, 2):
            self.assertEqual(f'demo_{i}' in split['train'], f'demo_{i+1}' in split['train'])

    def test_leaking_official_masks_are_rejected(self):
        with h5py.File(self.path, 'a') as file:
            file.create_dataset('mask/train', data=np.array(['demo_0','demo_2'], dtype='S'))
            file.create_dataset('mask/valid', data=np.array(['demo_1','demo_3'], dtype='S'))
        with self.assertRaisesRegex(ValueError, 'scene leaks'):
            make_episode_split(self.path, 'paired')

    def test_baseline_conditional_are_independent_of_dataset(self):
        for kind in ('ph', 'mh', 'paired'):
            baseline = parse_args(['--data', str(self.path), '--dataset-type', kind, '--output', str(self.root)])
            self.assertEqual(baseline.policy_mode, 'baseline')
            self.assertEqual(baseline.epochs, 50)
            self.assertEqual(baseline.gt_shuffle_prob, 0)
            self.assertEqual(baseline.base_prob, .3)  # Unused, retained for existing baseline resume.
            conditional = parse_args(['--data', str(self.path), '--dataset-type', kind, '--output', str(self.root), '--policy-mode', 'conditional'])
            self.assertEqual(conditional.policy_mode, 'conditional')
            self.assertEqual(conditional.gt_shuffle_prob, 0)
            self.assertEqual(conditional.base_prob, .45)
        legacy = parse_args(['--data', str(self.path), '--dataset-type', 'paired', '--output', str(self.root),
                             '--enable-action-conditioning'])
        self.assertEqual(legacy.gt_shuffle_prob, 0)
        self.assertEqual(legacy.base_prob, .45)
        disabled = parse_args(['--data', str(self.path), '--dataset-type', 'paired', '--output', str(self.root),
                               '--policy-mode', 'conditional', '--gt-shuffle-prob', '0'])
        self.assertEqual(disabled.gt_shuffle_prob, 0)
        capped = parse_args(['--data', str(self.path), '--dataset-type', 'paired', '--output', str(self.root), '--num-steps', '2'])
        self.assertIsNone(capped.epochs)
        self.assertEqual(capped.num_steps, 2)

    def test_gaussian_and_conformal_calibration(self):
        values = np.array([.1, .2, .3, .4, .5])
        result = calibrate(values, kappa=2)
        self.assertAlmostEqual(result['threshold'], values.mean() + 2 * values.std())
        self.assertEqual(calibrate(values, 'conformal', alpha=.25)['threshold'], .5)
        with self.assertRaises(ValueError): calibrate(values, 'conformal', alpha=.01)

    def test_ae_backward_checkpoint_calibration_and_inference(self):
        output = self.root / 'detector'
        train_detector(['--data', str(self.path), '--dataset-type', 'paired', '--output', str(output),
                        '--device', 'cpu', '--epochs', '1', '--batch-size', '2', '--num-workers', '0',
                        '--image-size', '8', '--hidden-dim', '8', '--latent-dim', '2'])
        detector = ErrorDetector(output / 'detector.pt')
        import json
        diagnostics = json.loads((output / 'optimization_diagnostics.json').read_text())
        self.assertEqual(diagnostics['optimizer'], 'adam')
        self.assertEqual(diagnostics['baseline_epoch'], 0)
        self.assertEqual(diagnostics['epoch'], 1)
        self.assertTrue(np.isfinite(diagnostics['initial_val_mse']))
        images = {key: np.zeros((8,8,3), dtype=np.uint8) for key in ('agentview_image','robot0_eye_in_hand_image')}
        result = detector(images)
        self.assertEqual(tuple(result['score'].shape), (1,))
        self.assertEqual(result['is_ood'].dtype, torch.bool)
        self.assertTrue(np.isfinite(result['threshold']))
        with self.assertRaisesRegex(ValueError, 'Detector images'):
            pack_images({key: -np.ones((8,8,3),dtype=np.float32) for key in images})
        # A fresh resumed load includes both optimizer state and calibrated threshold.
        train_detector(['--data', str(self.path), '--dataset-type', 'paired', '--output', str(output),
                        '--device', 'cpu', '--epochs', '1', '--batch-size', '2', '--num-workers', '0',
                        '--image-size', '8', '--hidden-dim', '8', '--latent-dim', '2', '--resume', str(output / 'last.pt')])
        self.assertEqual(json.loads((output / 'optimization_diagnostics.json').read_text()), diagnostics)


if __name__ == '__main__': unittest.main()
