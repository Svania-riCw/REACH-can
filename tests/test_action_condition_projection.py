"""Linear/MLP conditioning, gradient flow, checkpoint compatibility and run settings."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from checkpoint_utils import action_condition_projection, load_model
import torch
from torch import nn
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_experimental import ExperimentalPI0Pytorch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from train_pi05 import parse_args, cosine_warmup_lambda, check_resume


class TinyConditionedModel(ExperimentalPI0Pytorch):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.pi05 = True
        self.action_in_proj = nn.Linear(config.action_dim, 16)
        self.action_condition_proj = None
        self._active_action_condition = None


class ProjectionTests(unittest.TestCase):
    def model(self, projection):
        model = TinyConditionedModel(Pi0Config(pi05=True, action_dim=32, action_horizon=50))
        model.enable_action_conditioning(projection)
        return model

    def test_linear_keeps_legacy_shape_keys_and_zero_initial_output(self):
        model = self.model('linear')
        self.assertIsInstance(model.action_condition_proj, nn.Linear)
        self.assertEqual(tuple(model.action_condition_proj.weight.shape), (16, 32))
        self.assertIn('action_condition_proj.weight', model.state_dict())
        self.assertEqual(torch.count_nonzero(model.action_condition_proj(torch.randn(2, 50, 32))), 0)
        model.enable_action_conditioning()
        with self.assertRaisesRegex(ValueError, 'Cannot replace'):
            model.enable_action_conditioning('mlp')

    def test_mlp_zero_output_hidden_initialization_and_gradient_flow(self):
        model = self.model('mlp')
        projection = model.action_condition_proj
        self.assertEqual([tuple(layer.weight.shape) for layer in projection if isinstance(layer, nn.Linear)],
                         [(64, 32), (64, 64), (16, 64)])
        self.assertGreater(torch.count_nonzero(projection[0].weight), 0)
        self.assertGreater(torch.count_nonzero(projection[2].weight), 0)
        inputs = torch.randn(2, 50, 32)
        self.assertEqual(torch.count_nonzero(projection(inputs)), 0)
        optimizer = torch.optim.SGD(projection.parameters(), lr=.1)
        for _ in range(2):
            optimizer.zero_grad()
            loss = (projection(inputs) - 1).square().mean()
            loss.backward()
            self.assertGreater(projection[-1].weight.grad.abs().sum(), 0)
            optimizer.step()
        self.assertGreater(projection[0].weight.grad.abs().sum(), 0)
        self.assertGreater(projection[2].weight.grad.abs().sum(), 0)

    def test_embed_suffix_adds_mlp_condition_without_changing_masks(self):
        model = self.model('mlp')
        nn.init.normal_(model.action_condition_proj[-1].weight)
        embeddings = torch.randn(2, 50, 16)
        condition = torch.randn(2, 50, 32)
        masks = (torch.ones(2, 50, dtype=torch.bool), torch.zeros(50, dtype=torch.bool), None)
        with patch.object(PI0Pytorch, 'embed_suffix', return_value=(embeddings, *masks)):
            with model._condition_scope(condition):
                actual = model.embed_suffix(None, torch.zeros_like(condition), None)
            torch.testing.assert_close(actual[0], embeddings + model.action_condition_proj(condition))
            self.assertIs(actual[1], masks[0])
            self.assertIs(actual[2], masks[1])
            self.assertIsNone(model._active_action_condition)
            self.assertIs(model.embed_suffix(None, condition, None)[0], embeddings)

    def test_loader_autodetects_old_linear_and_new_mlp_checkpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            for projection in ('linear', 'mlp'):
                original = self.model(projection)
                saved_args = {'action_condition_projection': projection} if projection == 'mlp' else {}
                path = Path(folder) / f'{projection}.pt'
                torch.save({'model_state_dict': original.state_dict(), 'args': saved_args}, path)
                with patch('openpi.models_pytorch.pi0_experimental.ExperimentalPI0Pytorch', TinyConditionedModel):
                    loaded, _, _ = load_model(path, 'pi05_robomimic', 'cpu', enable_ac=True)
                    self.assertEqual(action_condition_projection(loaded.state_dict()), projection)
                    for key, value in original.state_dict().items():
                        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
                    other = 'linear' if projection == 'mlp' else 'mlp'
                    with self.assertRaisesRegex(ValueError, 'Cannot change'):
                        load_model(path, 'pi05_robomimic', 'cpu', enable_ac=True, ac_projection=other)
            with self.assertRaisesRegex(ValueError, 'metadata disagrees'):
                action_condition_projection(['action_condition_proj.0.weight'],
                                            {'action_condition_projection': 'linear'})

    def test_mlp_duration_and_schedule_are_explicit_and_baseline_rejects_mlp(self):
        args = parse_args(['--data', 'data.hdf5', '--dataset-type', 'paired', '--output', 'output',
                           '--policy-mode', 'conditional', '--action-condition-projection', 'mlp',
                           '--num-steps', '10000', '--lr-decay-steps', '13850'])
        self.assertEqual(args.action_condition_projection, 'mlp')
        self.assertEqual(args.num_steps, 10000)
        self.assertEqual(args.lr_decay_steps, 13850)
        self.assertIsNone(args.epochs)
        self.assertGreater(cosine_warmup_lambda(8000, 1000, args.lr_decay_steps),
                           cosine_warmup_lambda(8000, 1000, args.num_steps))
        with self.assertRaises(SystemExit):
            parse_args(['--data', 'data.hdf5', '--dataset-type', 'paired', '--output', 'output',
                        '--action-condition-projection', 'mlp'])
        previous = dict(vars(args), action_condition_projection='linear')
        with self.assertRaisesRegex(ValueError, 'action_condition_projection'):
            check_resume({'args': previous, 'optimizer_state_dict': {}, 'rank_rng': [{}]}, args, 1)
        previous = dict(vars(args), lr_decay_steps=10000)
        with self.assertRaisesRegex(ValueError, 'lr_decay_steps'):
            check_resume({'args': previous, 'optimizer_state_dict': {}, 'rank_rng': [{}]}, args, 1)


if __name__ == '__main__':
    unittest.main()
