"""Full-SFT contracts: real state tokens, official quantiles, full loss, and accumulation."""
import copy
import dataclasses
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
import torch
from torch.utils.data import default_collate
from openpi_robomimic_adapter import RobomimicHDF5Dataset, compute_action_stats, normalize, denormalize
from train_pi05 import configure_full_sft, train_step, train_update, parse_args, check_resume, main, evaluate
from test_training import fixture, TinyTokenizer


class CaptureTokenizer(TinyTokenizer):
    def tokenize(self, prompt, state):
        self.state = state.copy()
        return super().tokenize(prompt, state)


class ChannelModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.ones(32))

    def forward(self, observation, actions, **kwargs):
        return (self.bias[None, None] - actions).square()


class FullSFTTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'data.hdf5'
        fixture(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_real_state_is_tokenized_before_padding_and_legacy_is_preserved(self):
        tokenizer = CaptureTokenizer()
        stats = compute_action_stats(self.path, ['demo_0', 'demo_1'])
        dataset = RobomimicHDF5Dataset(self.path, ['demo_0'], stats, tokenizer=tokenizer)
        observation, actions = dataset[0]
        self.assertEqual(len(tokenizer.state), 9)
        self.assertEqual(len(observation['state']), 32)
        np.testing.assert_array_equal(observation['state'][9:], 0)
        np.testing.assert_array_equal(actions[:, 7:], 0)
        legacy = {k: copy.deepcopy(stats[k]) for k in ('state_keys', 'action', 'state')}
        for key in ('state', 'action'):
            legacy[key] = {k: legacy[key][k] for k in ('min', 'max')}
        RobomimicHDF5Dataset(self.path, ['demo_0'], legacy, tokenizer=tokenizer)[0]
        self.assertEqual(len(tokenizer.state), 32)

    def test_train_only_quantiles_match_openpi_and_roundtrip_outside_range(self):
        from openpi.transforms import Normalize, Unnormalize
        from openpi.shared.normalize import NormStats
        stats = compute_action_stats(self.path, ['demo_0', 'demo_1'])
        self.assertEqual(stats['action']['max'][0], 103)
        expected = np.quantile(np.array([0, 1, 2, 3, 100, 101, 102, 103]), .99)
        self.assertAlmostEqual(stats['action']['q99'][0], expected)
        field = stats['action']
        reference = NormStats(mean=np.array(field['mean']), std=np.array(field['std']),
                              q01=np.array(field['q01']), q99=np.array(field['q99']))
        value = np.array([[-20., 0., 1., 10., 100., 103., 200.]], dtype=np.float32)
        ours = normalize(value, field)
        official = Normalize({'actions': reference}, use_quantiles=True)({'actions': value})['actions']
        np.testing.assert_allclose(ours, official, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(denormalize(ours, field), value, atol=2e-5)
        official_back = Unnormalize({'actions': reference}, use_quantiles=True)({'actions': ours})['actions']
        np.testing.assert_allclose(denormalize(ours, field), official_back, atol=1e-6)

    def test_only_physical_channels_update_and_accumulation_matches_global_batch(self):
        dataset = RobomimicHDF5Dataset(self.path, ['demo_0'], tokenizer=TinyTokenizer())
        batches = [default_collate([dataset[index]]) for index in (0, 1)]
        first, second = ChannelModel(), ChannelModel()
        opt1 = torch.optim.SGD(first.parameters(), lr=.01)
        opt2 = torch.optim.SGD(second.parameters(), lr=.01)
        before = first.bias.detach().clone()
        train_update(first, batches, opt1, 'cpu')
        train_step(second, default_collate([dataset[0], dataset[1]]), opt2, 'cpu')
        torch.testing.assert_close(first.bias, second.bias)
        self.assertTrue(torch.any(first.bias[:7] != before[:7]))
        self.assertTrue(torch.equal(first.bias[7:], before[7:]))
        self.assertEqual(torch.count_nonzero(first.bias.grad[7:]), 0)

    def test_full_scope_does_not_freeze_vision_or_language(self):
        model = torch.nn.Module()
        model.paligemma_with_expert = torch.nn.Module()
        model.paligemma_with_expert.paligemma = torch.nn.Linear(2, 2)
        model.paligemma_with_expert.gemma_expert = torch.nn.Module()
        model.paligemma_with_expert.gemma_expert.lm_head = torch.nn.Linear(2, 2)
        model.action_out_proj = torch.nn.Linear(2, 2)
        model.requires_grad_(False)
        configure_full_sft(model)
        self.assertTrue(all(p.requires_grad for p in model.paligemma_with_expert.paligemma.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.action_out_proj.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.paligemma_with_expert.gemma_expert.lm_head.parameters()))

    def test_validation_and_model_selection_use_physical_loss_with_full_loss_diagnostic(self):
        dataset = RobomimicHDF5Dataset(self.path, ['demo_0'], tokenizer=TinyTokenizer())
        batch = default_collate([dataset[0], dataset[1]])
        model = ChannelModel()
        metrics = evaluate(model, [batch], None, 'cpu', return_metrics=True)
        self.assertAlmostEqual(metrics['full_action_loss'],
                               (7*metrics['physical_action_loss'] + 25*metrics['padded_action_loss'])/32)
        self.assertEqual(metrics['loss'], metrics['physical_action_loss'])
        self.assertEqual(metrics['loss'], evaluate(model, [batch], None, 'cpu'))
        self.assertEqual(metrics['padded_action_loss'], 1)

    def test_legacy_resume_cannot_silently_change_training_objective(self):
        args = parse_args(['--data', str(self.path), '--dataset-type', 'paired', '--output', self.temp.name])
        with self.assertRaisesRegex(ValueError, 'different/legacy SFT'):
            check_resume({'optimizer_state_dict': {}, 'args': {}}, args, 1)

    def test_entrypoint_saves_best_final_and_resumes_at_micro_batch_boundary(self):
        from checkpoint_utils import load_payload
        @dataclasses.dataclass
        class Config:
            action_dim: int = 32
            action_horizon: int = 50
            max_token_len: int = 200
        class Model(ChannelModel):
            def __init__(self):
                super().__init__()
                self.config = Config()
                self.paligemma_with_expert = torch.nn.Module()
                self.paligemma_with_expert.gemma_expert = torch.nn.Module()
                self.paligemma_with_expert.gemma_expert.lm_head = torch.nn.Linear(1, 1)
        def load(path, *args, **kwargs):
            model = Model()
            payload = load_payload(path) if Path(path).suffix == '.pt' else None
            if payload:
                model.load_state_dict(payload['model_state_dict'])
            return model, model.config, payload
        base = Path(self.temp.name) / 'model.safetensors'
        base.touch()
        output = Path(self.temp.name) / 'run'
        args = ['--data', str(self.path), '--dataset-type', 'ph', '--output', str(output),
                '--checkpoint', str(base), '--device', 'cpu', '--num-workers', '0',
                '--num-steps', '2', '--batch-size', '2', '--gradient-accumulation-steps', '2',
                '--eval-every', '1', '--checkpoint-every', '1', '--monitor-samples', '0']
        with patch('train_pi05.load_model', side_effect=load), patch('train_pi05.PaligemmaTokenizer', return_value=TinyTokenizer()):
            main(args)
            payload = load_payload(output / 'final_model.pt')
            self.assertEqual(payload['step'], 2)
            self.assertEqual(payload['batch_in_epoch'], 4)
            self.assertEqual(payload['normalization_stats']['normalization'], 'quantile')
            self.assertIsNotNone(payload['best_val_loss'])
            self.assertTrue((output / 'best_validation.pt').exists())
            main(args + ['--resume', str(output / 'final_model.pt')])
            self.assertEqual(load_payload(output / 'final_model.pt')['step'], 2)


if __name__ == '__main__':
    unittest.main()
