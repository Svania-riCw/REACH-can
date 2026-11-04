"""Verify diagnostic isolation and guidance direction, independently of Pi0.5."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from checkpoint_utils import rng_state
import numpy as np
import torch
from openpi_robomimic_adapter import create_robomimic_dataloaders
from policy_diagnostics import (FixedProbe, assert_close, diagnostic_mode, euler_sample,
                                flow_metrics, monitor_branches)
from train_pi05 import prepare_batch
from test_training import fixture, TinyTokenizer


class FlowModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, observation, actions, noise, time, action_condition=None, **kwargs):
        assert not self.training and not kwargs["preprocess_train"] and kwargs["return_flow"]
        self.calls.append((noise.clone(), time.clone(),
                           action_condition.clone() if action_condition is not None else None))
        torch.rand(3)  # Even an accidentally stochastic diagnostic cannot shift training RNG.
        target = noise - actions
        velocity = target if action_condition is None else target + .1 * action_condition
        flow = dict(velocity=velocity, target_velocity=target, time=time,
                    x_t=time[:, None, None] * noise + (1-time[:, None, None]) * actions)
        return (velocity - target).square(), flow


class DiagnosticTests(unittest.TestCase):
    def test_fixed_monitor_isolates_rng_and_holds_noise_time_constant(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.hdf5"
            fixture(path)
            _, val = create_robomimic_dataloaders(path, batch_size=2, num_workers=0,
                                                 tokenizer=TinyTokenizer())
            state = rng_state()
            probe = FixedProbe(val.dataset, 3, 123, "cpu", prepare_batch)
            again = FixedProbe(val.dataset, 3, 123, "cpu", prepare_batch)
            self.assertTrue(torch.equal(probe.noise, again.noise))
            self.assertEqual(probe.sample_ids, again.sample_ids)
            model = FlowModel().train()
            first = monitor_branches(model, probe, torch.zeros(1, 1, 32))
            second = monitor_branches(model, probe, torch.zeros(1, 1, 32))
            self.assertEqual(first, second)
            self.assertTrue(model.training)
            for noise, time, _ in model.calls:
                self.assertTrue(torch.equal(noise, probe.noise))
                self.assertTrue(torch.equal(time, probe.time))
            self.assertTrue(torch.equal(model.calls[1][2].roll(1, 0), model.calls[2][2]))
            self.assertEqual(first["base"]["mse"], 0.)
            self.assertGreater(first["condition_effect"]["velocity_mse"], 0.)
            baseline = monitor_branches(FlowModel(), probe, None)
            self.assertEqual(set(baseline), {"baseline"})
            self.assertEqual(baseline["baseline"]["mse"], 0.)
            after = rng_state()
            self.assertTrue(torch.equal(state["torch"], after["torch"]))
            self.assertEqual(state["python"], after["python"])
            np.testing.assert_equal(state["numpy"], after["numpy"])

    def test_padding_does_not_change_physical_metrics(self):
        actions = torch.zeros(2, 3, 32)
        flow = dict(velocity=torch.ones_like(actions), target_velocity=torch.zeros_like(actions),
                    x_t=torch.zeros_like(actions), time=torch.tensor([.2, .8]))
        expected = flow_metrics(flow, actions)
        flow["velocity"][..., 7:] = 1e20
        self.assertEqual(expected, flow_metrics(flow, actions))
        flow["velocity"][0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            flow_metrics(flow, actions)

    def test_guidance_sign_and_zero_scale(self):
        noise = torch.ones(2, 3, 7)
        base, target = torch.zeros_like(noise), torch.ones_like(noise)
        # Exact linear bridge: dx/dt = initial_noise - condition.
        velocity = lambda x, time, condition: noise - condition
        normal = euler_sample(velocity, noise, base)
        zero = euler_sample(velocity, noise, base, target, 0.)
        self.assertTrue(torch.equal(normal, zero))
        away = euler_sample(velocity, noise, base, target, 2., "away")
        toward = euler_sample(velocity, noise, base, target, 2., "toward")
        torch.testing.assert_close(away, torch.full_like(noise, -2.))
        torch.testing.assert_close(toward, torch.full_like(noise, 2.))

    def test_consistency_failure_raises_and_mode_restores_on_error(self):
        model = torch.nn.Linear(1, 1).train()
        state = torch.get_rng_state()
        with self.assertRaises(AssertionError):
            with diagnostic_mode(model):
                torch.rand(1)
                assert_close(torch.zeros(1), torch.ones(1), 1e-4, "broken sampler")
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
