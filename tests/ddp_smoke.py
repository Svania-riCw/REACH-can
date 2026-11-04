"""Run with python -m torch.distributed.run --standalone --nproc-per-node=2."""
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from checkpoint_utils import PROJECT_ROOT
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from train_pi05 import setup_distributed, train_step, evaluate
from openpi_robomimic_adapter import create_robomimic_dataloaders
from test_training import TinyModel, TinyTokenizer, fixture


def main():
    device, rank, world_size = setup_distributed('cpu')
    folder = [tempfile.mkdtemp(prefix='reach_ddp_') if rank == 0 else None]
    dist.broadcast_object_list(folder, src=0)
    path = Path(folder[0]) / 'data.hdf5'
    if rank == 0:
        fixture(path, episodes=4, frames=5)
    dist.barrier()
    train, val = create_robomimic_dataloaders(path, batch_size=2, num_workers=0,
        tokenizer=TinyTokenizer(), rank=rank, world_size=world_size)
    indices = list(train.sampler)
    partitions = [None] * world_size
    dist.all_gather_object(partitions, indices)
    assert not set(partitions[0]) & set(partitions[1]), partitions
    model = DistributedDataParallel(TinyModel())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    for batch in train:
        train_step(model, batch, optimizer, device)
    loss = evaluate(model, val, None, device)
    weights = [torch.zeros_like(model.module.weight) for _ in range(world_size)]
    dist.all_gather(weights, model.module.weight)
    assert all(torch.equal(weights[0], weight) for weight in weights)
    assert torch.isfinite(torch.tensor(loss))
    if rank == 0:
        print(f'DDP PASS: {world_size} ranks, disjoint samples, synchronized weights, val_loss={loss}')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
