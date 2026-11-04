"""Two-rank check: probes preserve updates even with an empty val shard."""
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from checkpoint_utils import rng_state, restore_rng
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from openpi_robomimic_adapter import create_robomimic_dataloaders
from policy_diagnostics import FixedProbe, monitor_branches
from train_pi05 import setup_distributed, train_step, prepare_batch, evaluate
from test_training import fixture, TinyTokenizer


class TinyFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.1))

    def forward(self, observation, actions, noise=None, time=None, action_condition=None,
                return_flow=False, **kwargs):
        if noise is None:
            noise = torch.randn_like(actions)
        if time is None:
            time = torch.rand(len(actions))
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        velocity = self.weight * x_t + .1 * action_condition
        target = noise - actions
        losses = (velocity - target).square()
        if return_flow:
            return losses, dict(velocity=velocity, target_velocity=target, x_t=x_t, time=time)
        return losses


def main():
    device, rank, world_size = setup_distributed("cpu")
    folder = [tempfile.mkdtemp(prefix="reach_ddp_diagnostics_") if rank == 0 else None]
    dist.broadcast_object_list(folder, src=0)
    try:
        path = Path(folder[0]) / "data.hdf5"
        if rank == 0:
            fixture(path, episodes=4, frames=1)
        dist.barrier()
        train, val = create_robomimic_dataloaders(path, batch_size=1, num_workers=0,
            tokenizer=TinyTokenizer(), rank=rank, world_size=world_size)
        assert len(val.dataset) == 1
        assert len(val.sampler) == (1 if rank == 0 else 0)
        batch = next(iter(train))
        base = torch.zeros(1, 1, 32)
        probe = FixedProbe(val.dataset, 4, 2026, device, prepare_batch)
        reference, monitored = DDP(TinyFlow()), DDP(TinyFlow())
        ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=.01)
        optimizer = torch.optim.AdamW(monitored.parameters(), lr=.01)
        start_rng = rng_state()
        ref_loss = train_step(reference, batch, ref_optimizer, device, base, shuffle_prob=0)
        expected_rng = rng_state()
        restore_rng(start_rng)
        before = monitor_branches(monitored.module, probe, base)
        loss = train_step(monitored, batch, optimizer, device, base, shuffle_prob=0)
        monitor_branches(monitored.module, probe, base)
        assert torch.equal(reference.module.weight, monitored.module.weight)
        assert torch.equal(ref_loss, loss)
        assert torch.equal(expected_rng["torch"], rng_state()["torch"])
        panels = [None] * world_size
        dist.all_gather_object(panels, (probe.metadata(), before))
        assert all(panel == panels[0] for panel in panels)
        assert torch.isfinite(torch.tensor(evaluate(monitored, val, base, device)))
        if rank == 0:
            print("DDP diagnostics PASS: identical updates/RNG, shared panel, empty val shard handled")
        dist.barrier()
    finally:
        dist.destroy_process_group()
        if rank == 0:
            shutil.rmtree(folder[0])


if __name__ == "__main__":
    main()
