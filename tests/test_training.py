"""Regression checks for data contracts, AC semantics and resume serialization."""
import dataclasses
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from checkpoint_utils import load_payload, restore_rng, rng_state, save_checkpoint
import h5py
import numpy as np
import torch
from openpi_robomimic_adapter import (RobomimicHDF5Dataset, compute_action_stats,
                                     create_robomimic_dataloaders, validate_data)
from train_pi05 import (build_action_condition, cosine_warmup_lambda, evaluate,
                        parse_base_action, prepare_batch, train_step)


class TinyTokenizer:
    def tokenize(self, prompt, state):
        return np.array([2, 3, 0]), np.array([True, True, False])


def fixture(path, episodes=4, frames=4):
    with h5py.File(path, 'w') as file:
        for i in range(episodes):
            demo = file.create_group(f'data/demo_{i}')
            # Each episode has its own distinct action range.
            actions = np.repeat((np.arange(frames) + i * 100)[:, None], 7, axis=1).astype(np.float32)
            demo.create_dataset('actions', data=actions)
            obs = demo.create_group('obs')
            for key, dim in [('robot0_eef_pos', 3), ('robot0_eef_quat', 4), ('robot0_gripper_qpos', 2)]:
                obs.create_dataset(key, data=np.full((frames, dim), i, dtype=np.float32))
            for key in ('agentview_image', 'robot0_eye_in_hand_image'):
                obs.create_dataset(key, data=np.full((frames, 84, 84, 3), i * 50, dtype=np.uint8))


@dataclasses.dataclass
class TinyConfig:
    action_dim: int = 32


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.config = TinyConfig()

    def forward(self, observation, actions, action_condition=None, **kwargs):
        assert observation.images['base_0_rgb'].device == actions.device
        prediction = self.weight * observation.state[:, None, :].expand_as(actions)
        if action_condition is not None:
            prediction = prediction + 0.1 * action_condition
        return (prediction - actions).square()


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'data.hdf5'
        fixture(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_chunk_never_crosses_episode_and_padding_is_zero(self):
        dataset = RobomimicHDF5Dataset(self.path, ['demo_0'], action_horizon=6)
        obs, actions = dataset[2]
        self.assertEqual(tuple(actions.shape), (6, 32))
        self.assertEqual(actions[:, 0].tolist(), [2, 3, 3, 3, 3, 3])
        self.assertEqual(torch.count_nonzero(actions[:, 7:]).item(), 0)
        self.assertEqual(set(obs['image']), {'base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'})
        self.assertFalse(obs['image_mask']['right_wrist_0_rgb'])
        self.assertTrue(torch.all(obs['image']['base_0_rgb'] == -1))

    def test_split_and_normalization_exclude_validation(self):
        split = {'train': ['demo_0', 'demo_1', 'demo_2'], 'val': ['demo_3']}
        train, val = create_robomimic_dataloaders(self.path, batch_size=2, num_workers=0,
                                                tokenizer=TinyTokenizer(), split=split)
        self.assertEqual(train.normalization_stats['action']['max'][0], 203)
        obs, actions = prepare_batch(next(iter(train)), 'cpu')
        self.assertEqual(tuple(actions.shape), (2, 50, 32))
        self.assertEqual(tuple(obs.state.shape), (2, 32))
        self.assertTrue(obs.tokenized_prompt_mask[:, 0].all())
        self.assertFalse(set(train.dataset.demo_names) & set(val.dataset.demo_names))

    def test_incomplete_last_episode_is_rejected(self):
        with h5py.File(self.path, 'a') as file:
            del file['data/demo_3/obs/robot0_eye_in_hand_image']
        with self.assertRaisesRegex(ValueError, 'Missing/misaligned'):
            validate_data(self.path)

    def test_singleton_batch_passes_real_openpi_preprocessing(self):
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        train, _ = create_robomimic_dataloaders(self.path, batch_size=1, num_workers=0,
                                               tokenizer=TinyTokenizer())
        obs, _ = prepare_batch(next(iter(train)), 'cpu')
        images, masks, *_ = PI0Pytorch._preprocess_observation(None, obs, train=False)
        self.assertEqual(tuple(images[0].shape), (1, 3, 224, 224))

    def test_condition_probabilities_and_shuffle_derangement(self):
        actions = torch.arange(8.).view(8, 1, 1).expand(8, 5, 32)
        base = torch.full((1, 1, 32), -5.)
        condition, _ = build_action_condition(actions, base_prob=1, base_action=base, shuffle_prob=0)
        self.assertTrue(torch.all(condition == -5))
        condition, _ = build_action_condition(actions, base_prob=0, base_action=base, shuffle_prob=0)
        self.assertTrue(torch.equal(condition, actions))
        condition, _ = build_action_condition(actions, base_prob=0, base_action=base, shuffle_prob=1)
        self.assertTrue(torch.all(condition[:, 0, 0] != actions[:, 0, 0]))
        self.assertTrue(torch.all(parse_base_action('1', 32, 'cpu')[..., 7:] == 0))

    def test_mixed_condition_branches_and_original_training_targets(self):
        actions = torch.arange(6.).view(6, 1, 1).expand(6, 5, 32).clone()
        original = actions.clone()
        draws = torch.tensor([.05, .25, .35, .55, .65, .95])
        base = torch.full((1, 1, 32), -5.)
        with patch('torch.rand', return_value=draws):
            condition, base_mask, shuffle_mask = build_action_condition(
                actions, base_prob=.3, base_action=base, shuffle_prob=.3, return_labels=True)
        self.assertEqual(base_mask.tolist(), [True, True, False, False, False, False])
        self.assertEqual(shuffle_mask.tolist(), [False, False, True, True, False, False])
        self.assertTrue(torch.all(condition[base_mask] == -5))
        self.assertTrue(torch.all(condition[shuffle_mask, 0, 0] != actions[shuffle_mask, 0, 0]))
        self.assertTrue(torch.equal(condition[~(base_mask | shuffle_mask)], actions[~(base_mask | shuffle_mask)]))
        self.assertTrue(torch.equal(actions, original))

        train, _ = create_robomimic_dataloaders(self.path, batch_size=4, num_workers=0,
                                               tokenizer=TinyTokenizer())
        batch = next(iter(train))
        class CapturingModel(TinyModel):
            def forward(self, observation, actions, action_condition=None, **kwargs):
                self.target = actions.detach().clone()
                self.condition = action_condition.detach().clone()
                return super().forward(observation, actions, action_condition=action_condition, **kwargs)
        model = CapturingModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        with patch('torch.rand', return_value=torch.full((4,), .45)):
            loss = train_step(model, batch, optimizer, 'cpu', torch.zeros(1, 1, 32),
                              base_prob=.3, shuffle_prob=.3)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.equal(model.target, batch[1]))
        self.assertTrue((model.condition - model.target).flatten(1).abs().sum(1).gt(0).all())
        # Current defaults use own GT above the45% boundary, with no shuffle.
        with patch('torch.rand', return_value=torch.full((4,), .45)):
            train_step(model, batch, optimizer, 'cpu', torch.zeros(1, 1, 32))
        self.assertTrue(torch.equal(model.condition, model.target))
        with patch('torch.rand', return_value=torch.full((4,), .44)):
            train_step(model, batch, optimizer, 'cpu', torch.zeros(1, 1, 32))
        self.assertTrue(torch.equal(model.condition, torch.zeros_like(model.condition)))
        self.assertTrue(torch.equal(model.target, batch[1]))

    def test_backward_validation_and_exact_resume(self):
        train, val = create_robomimic_dataloaders(self.path, batch_size=2, num_workers=0,
                                                tokenizer=TinyTokenizer())
        batch = next(iter(train))
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
        before = model.weight.detach().clone()
        loss = train_step(model, batch, optimizer, 'cpu')
        scheduler.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before, model.weight))
        self.assertTrue(np.isfinite(evaluate(model, val, None, 'cpu')))
        saved_rng = rng_state()
        path = Path(self.temp.name) / 'checkpoint.pt'
        save_checkpoint(path, model, optimizer, scheduler, argparse_namespace(), 1, 0, 1,
                        train.normalization_stats, train.episode_split, [saved_rng])
        expected = train_step(model, batch, optimizer, 'cpu')
        payload = load_payload(path)
        resumed = TinyModel()
        resumed.load_state_dict(payload['model_state_dict'])
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
        resumed_optimizer.load_state_dict(payload['optimizer_state_dict'])
        restore_rng(payload['rank_rng'][0])
        actual = train_step(resumed, batch, resumed_optimizer, 'cpu')
        self.assertEqual(expected.item(), actual.item())
        self.assertTrue(torch.equal(model.weight, resumed.weight))

    def test_tiny_schedule_never_divides_by_zero(self):
        self.assertGreater(cosine_warmup_lambda(0, 150, 1), 0)
        self.assertAlmostEqual(cosine_warmup_lambda(1, 150, 1), 0.1)


def argparse_namespace():
    from argparse import Namespace
    return Namespace(dataset_type='ph')


if __name__ == '__main__':
    unittest.main()
