#!/usr/bin/env python3
"""Train an independent PH/MH/Paired image AE, then calibrate on held-out ID data."""
import argparse
import json
from pathlib import Path
import random
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_splits import make_episode_split
from error_detector import CAMERAS, ImageAutoencoder, calibrate, pack_images


class ImageDataset(Dataset):
    def __init__(self, path, episodes, image_size=84):
        self.path, self.image_size = str(path), image_size
        self.index = []
        with h5py.File(path, 'r') as file:
            for name in episodes:
                demo = file[f'data/{name}']
                length = len(demo['actions'])
                for key in CAMERAS:
                    if key not in demo['obs'] or len(demo[f'obs/{key}']) != length:
                        raise ValueError(f'Missing/misaligned {name}/{key}')
                self.index.extend((name, t) for t in range(length))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        name, timestep = self.index[index]
        with h5py.File(self.path, 'r') as file:
            images = {key: file[f'data/{name}/obs/{key}'][timestep] for key in CAMERAS}
        return pack_images(images, self.image_size).squeeze(0)


def save_atomic(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def collect_scores(model, loader, device):
    model.eval()
    values = [model.score(batch.to(device)).cpu().numpy() for batch in loader]
    if not values:
        raise ValueError('No validation/calibration images')
    return np.concatenate(values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--dataset-type', choices=['ph', 'mh', 'paired'], required=True)
    parser.add_argument('--metadata')
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--image-size', type=int, default=84)
    parser.add_argument('--val-ratio', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threshold-method', choices=['gaussian', 'conformal'], default='gaussian')
    parser.add_argument('--kappa', type=float, default=4.)
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--resume')
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.hidden_dim, args.latent_dim, args.image_size) < 1 or args.lr <= 0 or args.num_workers < 0:
        parser.error('Invalid duration, architecture, worker count or learning rate')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu explicitly')
    payload = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if payload:
        for key in ('data', 'dataset_type', 'metadata', 'epochs', 'batch_size', 'lr', 'optimizer', 'hidden_dim',
                    'latent_dim', 'image_size', 'seed', 'threshold_method', 'kappa', 'alpha'):
            if payload['args'][key] != getattr(args, key):
                raise ValueError(f'Cannot change {key} when resuming detector training')
    split = make_episode_split(args.data, args.dataset_type, args.val_ratio, args.seed,
                               payload['episode_split'] if payload else None, args.metadata)
    train = ImageDataset(args.data, split['train'], args.image_size)
    val = ImageDataset(args.data, split['val'], args.image_size)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                          pin_memory=device.type == 'cuda')
    if args.num_workers:
        loader_options['multiprocessing_context'] = 'spawn'
    val_loader = DataLoader(val, shuffle=False, **loader_options)
    model = ImageAutoencoder(args.image_size, args.hidden_dim, args.latent_dim).to(device)
    optimizer_cls = torch.optim.SGD if args.optimizer == 'sgd' else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'detector.pt').exists() and not args.resume:
        raise FileExistsError(f'Detector output already exists: {output}')
    start_epoch = 0
    if payload:
        model.load_state_dict(payload['model_state_dict'])
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        scheduler.load_state_dict(payload['scheduler_state_dict'])
        start_epoch = payload['epoch']
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
    (output / 'episode_split.json').write_text(json.dumps(split, indent=2))
    # Record the untrained (or restored) validation baseline so a completed
    # run cannot be mistaken for evidence that reconstruction actually learned.
    diagnostics = payload.get('optimization_diagnostics') if payload else None
    if diagnostics is None:
        initial_mse = float(collect_scores(model, val_loader, device).mean())
        diagnostics = dict(baseline_epoch=start_epoch, initial_val_mse=initial_mse)
    checkpoint = None
    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.perf_counter()
        # Explicit epoch seeds make data order resumable without replaying batches.
        generator = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = DataLoader(train, shuffle=True, generator=generator, **loader_options)
        model.train()
        total, count = 0., 0
        for images in train_loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.score(images).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite AE loss')
            loss.backward()
            optimizer.step()
            total += loss.item() * len(images)
            count += len(images)
        scheduler.step()
        scores = collect_scores(model, val_loader, device)
        calibration = calibrate(scores, args.threshold_method, args.kappa, args.alpha)
        val_mse = float(scores.mean())
        initial_mse = diagnostics['initial_val_mse']
        diagnostics = dict(diagnostics, epoch=epoch+1, final_val_mse=val_mse,
                           relative_val_improvement=(initial_mse-val_mse)/max(initial_mse, 1e-12),
                           optimizer=args.optimizer, initial_lr=args.lr)
        checkpoint = dict(epoch=epoch + 1, model_config=model.config, model_state_dict=model.state_dict(),
                          optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                          args=vars(args), episode_split=split, calibration=calibration, camera_keys=list(CAMERAS),
                          data_filter='all_demonstrations_no_success_filter',
                          pixel_normalization='uint8_to_float_0_1', optimization_diagnostics=diagnostics)
        save_atomic(checkpoint, output / 'last.pt')
        (output / 'calibration.json').write_text(json.dumps(calibration, indent=2, allow_nan=False))
        (output / 'optimization_diagnostics.json').write_text(json.dumps(diagnostics, indent=2, allow_nan=False))
        print(f'AE {args.dataset_type} epoch={epoch+1}/{args.epochs} train_mse={total/count:.6f} '
              f'val_mse={val_mse:.6f} threshold={calibration["threshold"]:.6f} '
              f'val_improvement={diagnostics["relative_val_improvement"]:.2%} '
              f'epoch_seconds={time.perf_counter()-epoch_started:.2f}', flush=True)
    if checkpoint is None:
        checkpoint = payload
    save_atomic(checkpoint, output / 'detector.pt')
    print(f'Detector saved: {output / "detector.pt"}', flush=True)


if __name__ == '__main__':
    main()
