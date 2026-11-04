#!/usr/bin/env python3
"""Compare AE optimizers on fixed train-only fit samples and held-out ID probes."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from torch.utils.data import default_collate
from data_splits import make_episode_split
from error_detector import ImageAutoencoder
from train_error_detector import ImageDataset


@torch.no_grad()
def mse(model, images, batch_size):
    return sum(model.score(b).sum().item() for b in images.split(batch_size)) / len(images)


def panel(dataset, count):
    indices = torch.linspace(0, len(dataset)-1, min(count, len(dataset))).long().tolist()
    return default_collate([dataset[i] for i in indices]), [dataset.index[i] for i in indices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--dataset-type', choices=['ph','paired','mh'], default='ph')
    parser.add_argument('--metadata')
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--sgd-lrs', type=float, nargs='+', help='Override the candidate set with these SGD learning rates')
    args = parser.parse_args()
    torch.set_num_threads(4)
    split = make_episode_split(args.data, args.dataset_type, seed=0, metadata_path=args.metadata)
    train, train_ids = panel(ImageDataset(args.data, split['train']), 256)
    val, val_ids = panel(ImageDataset(args.data, split['val']), 128)
    report = dict(data=args.data, split=split, train_ids=train_ids, val_ids=val_ids,
                  steps=args.steps, batch_size=128, device='cpu', seed=0,
                  constant_half_val_mse=(val-.5).square().mean().item(),
                  train_mean_image_val_mse=(val-train.mean(0,keepdim=True)).square().mean().item(),
                  candidates=[])
    output=Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates=[('sgd',1e-4,0.),('sgd',.01,0.),('sgd',.1,0.),('sgd',1.,0.),
                ('sgd',10.,0.),('sgd',1.,.9),('adam',1e-4,0.)]
    if args.sgd_lrs:
        candidates=[('sgd',lr,0.) for lr in args.sgd_lrs]
    for kind, lr, momentum in candidates:
        started=time.perf_counter()
        torch.manual_seed(0)
        model=ImageAutoencoder()
        initial={k:v.detach().clone() for k,v in model.named_parameters()}
        optimizer=(torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum)
                   if kind=='sgd' else torch.optim.Adam(model.parameters(),lr=lr))
        entry=dict(optimizer=kind,lr=lr,momentum=momentum,initial_train_mse=mse(model,train,128),
                   initial_val_mse=mse(model,val,128),curve=[])
        try:
            for step in range(1,args.steps+1):
                model.train()
                images=train[(step-1)%2*128:((step-1)%2+1)*128]
                optimizer.zero_grad(set_to_none=True)
                loss=model.score(images).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite fit loss')
                loss.backward()
                if step==1:
                    entry['initial_grad_l2']=sum(p.grad.square().sum().item() for p in model.parameters())**.5
                optimizer.step()
                if step==1:
                    delta=sum((p-initial[k]).square().sum().item() for k,p in model.named_parameters())**.5
                    norm=sum(p.square().sum().item() for p in initial.values())**.5
                    entry['first_update_relative_l2']=delta/norm
                    del initial
                if step%20==0 or step==args.steps:
                    model.eval()
                    entry['curve'].append(dict(step=step,train_mse=mse(model,train,128),val_mse=mse(model,val,128)))
            entry['status']='completed'
        except (FloatingPointError, RuntimeError) as exc:
            entry.update(status='failed',error=str(exc))
        entry['seconds']=time.perf_counter()-started
        report['candidates'].append(entry)
        output.write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps(entry),flush=True)
        del optimizer,model
        gc.collect()


if __name__=='__main__':
    main()
