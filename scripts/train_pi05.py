#!/usr/bin/env python3
"""Train independent baseline/conditional pi0.5 policies on PH, MH, or Paired."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time

from checkpoint_utils import (load_model, prepare_checkpoint, restore_rng,
                              rng_state, save_checkpoint)
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from openpi.models.model import Observation
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models_pytorch.pi0_experimental import build_action_condition
from openpi_robomimic_adapter import create_robomimic_dataloaders, validate_data
from policy_diagnostics import FixedProbe, monitor_branches

TRAINING_RECIPE = 'pi05_can_full_sft_physical7_v4'
SUPERVISED_ACTION_DIMS = 7


def configure_full_sft(model):
    """Train the VLM and action expert; the unused expert LM head is not a policy output."""
    model.requires_grad_(True)
    model.paligemma_with_expert.gemma_expert.lm_head.requires_grad_(False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset-type", choices=["ph", "mh", "paired"], required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--metadata", help="Original low-dim HDF5 for official splits and Paired scene IDs")
    parser.add_argument("--policy-mode", choices=["baseline", "conditional"], default="baseline")
    parser.add_argument("--checkpoint", default=str(Path(__file__).resolve().parents[1] / "checkpoints/pi05_base_pytorch"))
    parser.add_argument("--config", default="pi05_robomimic")
    parser.add_argument("--prompt", default="Pick up the can and place it in the bin.")
    parser.add_argument("--enable-action-conditioning", action="store_true")
    parser.add_argument("--action-condition-projection", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--base-action", default="0")
    parser.add_argument("--base-prob", type=float,
                        help="Conditional default:0.45; baseline keeps its unused legacy value0.3 for resume")
    parser.add_argument("--gt-shuffle-prob", type=float,
                        help="Probability of another local-batch sample's GT prompt; default:0 (published main conditional mixture)")
    parser.add_argument("--batch-size", type=int, default=64, help="Global batch size across all ranks")
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Global batch = GPUs * micro batch * accumulation steps')
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--num-steps", type=int)
    duration.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-10)
    parser.add_argument("--beta1", type=float, default=.9)
    parser.add_argument("--beta2", type=float, default=.95)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--min-lr", type=float, default=2.5e-6)
    parser.add_argument("--lr-decay-steps", type=int,
                        help="Cosine schedule duration; defaults to num-steps, independent of an early training stop")
    parser.add_argument("--precision", choices=['float32', 'bfloat16'], default='float32',
                        help='OpenPI supports both; float32 avoids small updates rounding away in bf16 parameters')
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--monitor-samples", type=int, default=4,
                        help="Fixed held-out branch panel at validation intervals; 0 disables it")
    parser.add_argument("--monitor-seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", help="Resume a full .pt checkpoint including optimizer and data position")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Validate data without loading a model")
    args = parser.parse_args(argv)
    args.training_recipe = TRAINING_RECIPE
    args.supervised_action_dims = SUPERVISED_ACTION_DIMS
    if args.num_steps is None and args.epochs is None:
        args.epochs = 50
    if args.enable_action_conditioning:
        args.policy_mode = "conditional"
    if args.policy_mode != "conditional" and args.action_condition_projection != "linear":
        parser.error("MLP action conditioning requires --policy-mode conditional")
    if args.lr_decay_steps is not None and args.lr_decay_steps < 1:
        parser.error("lr-decay-steps must be positive")
    if args.base_prob is None:
        # Baseline never uses prompts. Preserve its saved argument for exact resume.
        args.base_prob = 0.45 if args.policy_mode == "conditional" else 0.3
    if args.gt_shuffle_prob is None:
        args.gt_shuffle_prob = 0.0
    if ((args.num_steps is not None and args.num_steps < 1) or (args.epochs is not None and args.epochs < 1)):
        parser.error("Training duration must be positive")
    if min(args.batch_size, args.gradient_accumulation_steps, args.checkpoint_every, args.eval_every, args.log_every) < 1:
        parser.error("Batch size, steps and intervals must be positive")
    if args.monitor_samples < 0:
        parser.error("monitor-samples must be nonnegative")
    if args.num_workers < 0 or args.warmup_steps < 0 or args.lr <= 0 or not 0 <= args.min_lr <= args.lr:
        parser.error("Invalid workers or learning-rate settings")
    if not 0 < args.val_ratio < 1:
        parser.error("val-ratio must be between 0 and 1")
    if args.weight_decay < 0 or not 0 <= args.beta1 < 1 or not 0 <= args.beta2 < 1:
        parser.error('Invalid AdamW settings')
    if not 0 <= args.base_prob <= 1 or not 0 <= args.gt_shuffle_prob <= 1 or args.base_prob + args.gt_shuffle_prob > 1:
        parser.error("Condition probabilities must be nonnegative and sum to at most 1")
    return args


def cosine_warmup_lambda(step, warmup_steps, total_steps, min_lr_ratio=0.1):
    warmup_steps = min(warmup_steps, max(0, total_steps - 1))
    if step < warmup_steps:
        initial = 1 / (warmup_steps + 1)
        return initial + (1 - initial) * step / warmup_steps
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2


def to_device(value, device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value.to(device, non_blocking=True) if torch.is_tensor(value) else value


def prepare_batch(batch, device):
    observation, actions = to_device(batch[0], device), batch[1].to(device, non_blocking=True)
    # The upstream channels-last resize drops the batch axis when B=1. Resize
    # channels-first here, preserving singleton validation batches and aspect.
    from openpi.shared.image_tools import resize_with_pad_torch
    observation["image"] = {key: resize_with_pad_torch(image, 224, 224)
                            if image.shape[-2:] != (224, 224) else image
                            for key, image in observation["image"].items()}
    return Observation.from_dict(observation), actions.float()


def parse_base_action(spec, action_dim, device):
    values = [float(value) for value in spec.split(",")]
    if len(values) == 1:
        values *= 7
    if len(values) not in {7, action_dim}:
        raise ValueError(f"base action needs 1, 7, or {action_dim} values")
    values += [0.] * (action_dim - len(values))
    return torch.tensor(values, device=device, dtype=torch.float32).view(1, 1, action_dim)


def train_step(model, batch, optimizer, device, base_action=None, base_prob=0.45, shuffle_prob=0.0):
    return train_update(model, [batch], optimizer, device, base_action, base_prob, shuffle_prob)


def train_update(model, batches, optimizer, device, base_action=None, base_prob=.45, shuffle_prob=0.):
    """Accumulate a full global batch; synchronize DDP only on its last micro batch."""
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=device)
    for index, batch in enumerate(batches):
        sync = model.no_sync() if isinstance(model, DDP) and index < len(batches) - 1 else nullcontext()
        with sync:
            observation, actions = prepare_batch(batch, device)
            kwargs = {}
            if base_action is not None:
                kwargs['action_condition'], _ = build_action_condition(actions, base_prob, base_action, shuffle_prob)
            losses = model(observation, actions, preprocess_train=True, **kwargs)
            # User-selected dimension mask: only the seven physical Can controls.
            loss = losses[..., :SUPERVISED_ACTION_DIMS].mean() / len(batches)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            total += loss.detach()
    torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], 1.0,
                                  error_if_nonfinite=True)
    optimizer.step()
    return total


@torch.no_grad()
def evaluate(model, val_loader, base_action, device, return_metrics=False):
    raw = model.module if isinstance(model, DDP) else model
    was_training = raw.training
    raw.eval()
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    # Avoid stochastic evaluation changing the training noise sequence.
    state = rng_state()
    torch.manual_seed(2026)
    try:
        for batch in val_loader:
            observation, actions = prepare_batch(batch, device)
            kwargs = {} if base_action is None else {"action_condition": base_action.expand_as(actions)}
            losses = raw(observation, actions, preprocess_train=False, **kwargs)
            if not torch.isfinite(losses).all():
                raise FloatingPointError("Non-finite validation loss")
            totals[0] += losses.double().sum()
            totals[1] += losses.numel()
            totals[2] += losses[..., :7].double().sum()
            totals[3] += losses[..., :7].numel()
            totals[4] += losses[..., 7:].double().sum()
            totals[5] += losses[..., 7:].numel()
        if dist.is_initialized():
            dist.all_reduce(totals)
        if totals[1] == 0:
            raise ValueError("Empty validation set")
        metrics = dict(loss=(totals[2] / totals[3]).item(),
                       full_action_loss=(totals[0] / totals[1]).item(),
                       physical_action_loss=(totals[2] / totals[3]).item(),
                       padded_action_loss=(totals[4] / totals[5]).item() if totals[5] else None)
        return metrics if return_metrics else metrics['loss']
    finally:
        restore_rng(state)
        raw.train(was_training)


def setup_distributed(device_name):
    world_size, rank = int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
        device = torch.device("cuda", local_rank if world_size > 1 else (device.index or 0))
        torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return device, rank, world_size


def check_resume(payload, args, world_size):
    if payload is None or "optimizer_state_dict" not in payload:
        raise ValueError("--resume requires a complete training checkpoint")
    previous = payload["args"]
    if previous.get('training_recipe') != TRAINING_RECIPE:
        raise ValueError('Cannot resume a different/legacy SFT or loss protocol; start a new output from the base checkpoint')
    if previous.get('action_condition_projection', 'linear') != args.action_condition_projection:
        raise ValueError('Cannot change action_condition_projection when resuming')
    if (previous.get('lr_decay_steps') or previous['num_steps']) != (args.lr_decay_steps or args.num_steps):
        raise ValueError('Cannot change lr_decay_steps when resuming')
    for key in ("dataset_type", "batch_size", "num_steps", "warmup_steps", "lr", "min_lr", "seed", "weight_decay",
                "prompt", "base_action", "base_prob", "gt_shuffle_prob", "policy_mode", "epochs", "metadata",
                "precision", "beta1", "beta2", "training_recipe", "supervised_action_dims"):
        if previous.get(key) != getattr(args, key):
            raise ValueError(f"Cannot change {key} when resuming; use --checkpoint for a new fine-tune")
    if previous.get('gradient_accumulation_steps') != args.gradient_accumulation_steps:
        raise ValueError('Cannot change gradient_accumulation_steps when resuming')
    if len(payload["rank_rng"]) != world_size:
        raise ValueError("Resume requires the original world size")
    if Path(previous["data"]).resolve() != Path(args.data).resolve():
        raise ValueError("Resume requires the original dataset path")


def main(argv=None):
    args = parse_args(argv)
    if args.validate_only:
        print(f"Validated {len(validate_data(args.data))} episodes: {args.data}")
        return
    # Reject locked/incomplete files before allocating a multi-billion-parameter model.
    validate_data(args.data)
    device, rank, world_size = setup_distributed(args.device)
    try:
        if args.batch_size % (world_size * args.gradient_accumulation_steps):
            raise ValueError('Global batch must be divisible by world size * gradient accumulation steps')
        per_rank_batch = args.batch_size // (world_size * args.gradient_accumulation_steps)
        use_ac = args.policy_mode == "conditional"
        if use_ac and args.gt_shuffle_prob and per_rank_batch < 2:
            raise ValueError("GT shuffle requires at least two samples per GPU (or set --gt-shuffle-prob 0)")
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        output_dir = Path(args.output).resolve()
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "checkpoints").mkdir(exist_ok=True)
            if (output_dir / "final_model.pt").exists() and not args.resume:
                raise FileExistsError(f"Output already contains a trained model: {output_dir}")
        # All ranks receive either the converted path or the same conversion failure.
        conversion = [None, None]
        if rank == 0:
            try:
                conversion[0] = str(prepare_checkpoint(args.resume or args.checkpoint))
            except Exception as exc:
                conversion[1] = str(exc)
        if dist.is_initialized():
            dist.broadcast_object_list(conversion, src=0)
        if conversion[1]:
            raise RuntimeError(conversion[1])
        model, model_cfg, payload = load_model(conversion[0], args.config, device, enable_ac=use_ac,
                                              precision=args.precision,
                                              ac_projection=args.action_condition_projection)
        configure_full_sft(model)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        # Independent experiments fit their own data statistics; only resume
        # restores an existing experiment's normalization and split.
        stats = payload.get("normalization_stats") if args.resume else None
        split = payload.get("episode_split") if args.resume else None
        tokenizer = PaligemmaTokenizer(model_cfg.max_token_len)
        train_loader, val_loader = create_robomimic_dataloaders(
            args.data, batch_size=per_rank_batch, val_ratio=args.val_ratio, num_workers=args.num_workers,
            seed=args.seed, action_horizon=model_cfg.action_horizon, action_dim=model_cfg.action_dim,
            tokenizer=tokenizer, prompt=args.prompt, rank=rank, world_size=world_size, stats=stats, split=split,
            dataset_type=args.dataset_type, metadata_path=args.metadata)
        stats, split = train_loader.normalization_stats, train_loader.episode_split
        updates_per_epoch = len(train_loader) // args.gradient_accumulation_steps
        if updates_per_epoch < 1:
            raise ValueError('Training split is too small for a complete global batch')
        if args.num_steps is None:
            args.num_steps = args.epochs * updates_per_epoch
        if args.resume:
            check_resume(payload, args, world_size)
        if rank == 0:
            (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
            (output_dir / "action_stats.json").write_text(json.dumps(stats, indent=2))
            (output_dir / "episode_split.json").write_text(json.dumps(split, indent=2))
            print(f"Training {args.dataset_type}/{args.policy_mode}: world_size={world_size}, global_batch={args.batch_size}, "
                  f"micro_batch={per_rank_batch}, accumulation={args.gradient_accumulation_steps}, device={device}", flush=True)
            print(f"Full SFT: trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}, "
                  f"precision={args.precision}, normalization=quantile, loss=physical_first_7_channels", flush=True)
        if world_size > 1:
            model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                        broadcast_buffers=False, find_unused_parameters=True, gradient_as_bucket_view=True)
        optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad],
                                      lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                                      foreach=False)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: cosine_warmup_lambda(
            step, args.warmup_steps, args.lr_decay_steps or args.num_steps, args.min_lr / args.lr))
        step, epoch, batch_in_epoch = 0, 0, 0
        best_val_loss = math.inf
        if args.resume:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            scheduler.load_state_dict(payload["scheduler_state_dict"])
            step, epoch, batch_in_epoch = payload["step"], payload["epoch"], payload["batch_in_epoch"]
            best_val_loss = payload.get('best_val_loss') if payload.get('best_val_loss') is not None else math.inf
        train_loader.sampler.set_epoch(epoch)
        iterator = iter(train_loader)
        for _ in range(batch_in_epoch):
            next(iterator)
        if args.resume:
            restore_rng(payload["rank_rng"][rank])
        writer = None
        if rank == 0:
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(output_dir / "tensorboard")
            except ImportError:
                pass
        base_action = parse_base_action(args.base_action, model_cfg.action_dim, device) if use_ac else None
        model.train()
        # Every rank uses the same global held-out panel, including ranks whose
        # validation sampler is empty. Diagnostics never call the DDP wrapper.
        probe = (FixedProbe(val_loader.dataset, args.monitor_samples, args.monitor_seed, device, prepare_batch)
                 if args.monitor_samples else None)

        def monitor():
            if probe is None:
                return
            raw = model.module if isinstance(model, DDP) else model
            metrics = monitor_branches(raw, probe, base_action)
            if rank == 0:
                record = dict(step=step, probe=probe.metadata(), metrics=metrics)
                with (output_dir / "branch_monitor.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                if writer:
                    for branch, values in metrics.items():
                        for name, value in values.items():
                            writer.add_scalar(f"monitor/{branch}/{name}", value, step)
                losses = {name: round(values["mse"], 6) for name, values in metrics.items() if "mse" in values}
                print(f"Step {step}: fixed_branch_mse={losses}", flush=True)

        monitor()

        def performance(event, **values):
            if rank == 0:
                with (output_dir / "performance.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(event=event, step=step, timestamp=time.time(), **values)) + "\n")

        log_started, last_log_step = time.perf_counter(), step

        def checkpoint(path):
            states = [None] * world_size
            if dist.is_initialized():
                dist.all_gather_object(states, rng_state())
            else:
                states[0] = rng_state()
            if rank == 0:
                save_checkpoint(path, model, optimizer, scheduler, args, step, epoch, batch_in_epoch,
                                stats, split, states, best_val_loss=best_val_loss if math.isfinite(best_val_loss) else None)
            if dist.is_initialized():
                dist.barrier()

        while step < args.num_steps:
            # Drop only the incomplete final global batch, as with a global-batch DataLoader.
            if batch_in_epoch + args.gradient_accumulation_steps > len(train_loader):
                epoch += 1
                batch_in_epoch = 0
                train_loader.sampler.set_epoch(epoch)
                iterator = iter(train_loader)
            batches = [next(iterator) for _ in range(args.gradient_accumulation_steps)]
            loss = train_update(model, batches, optimizer, device, base_action, args.base_prob, args.gt_shuffle_prob)
            scheduler.step()
            step += 1
            batch_in_epoch += args.gradient_accumulation_steps
            if step % args.log_every == 0:
                if dist.is_initialized():
                    dist.all_reduce(loss)
                    loss /= world_size
                if rank == 0:
                    loss_value = loss.item()  # Synchronizes the measured training window.
                    seconds_per_step = (time.perf_counter() - log_started) / (step - last_log_step)
                    peak_gib = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.
                    performance("training", loss=loss_value, seconds_per_step=seconds_per_step,
                                peak_allocated_gib=peak_gib)
                    print(f"Step {step}: loss={loss_value:.6f}, lr={scheduler.get_last_lr()[0]:.3g}, "
                          f"wall_s_per_step={seconds_per_step:.3f}, peak_allocated_gib={peak_gib:.2f}", flush=True)
                    if writer:
                        writer.add_scalar("train/loss", loss_value, step)
                        writer.add_scalar("performance/wall_seconds_per_step", seconds_per_step, step)
                log_started, last_log_step = time.perf_counter(), step
            if step % args.eval_every == 0:
                eval_started = time.perf_counter()
                val_metrics = evaluate(model, val_loader, base_action, device, return_metrics=True)
                val_loss = val_metrics['loss']
                monitor()
                eval_seconds = time.perf_counter() - eval_started
                performance("validation", seconds=eval_seconds, **val_metrics)
                if rank == 0:
                    print(f"Step {step}: val_loss={val_loss:.6f}, "
                          f"physical_7={val_metrics['physical_action_loss']:.6f}, "
                          f"padding_25={val_metrics['padded_action_loss']:.6f}, "
                          f"eval_seconds={eval_seconds:.2f}", flush=True)
                    if writer:
                        writer.add_scalar("val/loss", val_loss, step)
                        writer.add_scalar('val/physical_action_loss', val_metrics['physical_action_loss'], step)
                        writer.add_scalar('val/padded_action_loss', val_metrics['padded_action_loss'], step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_started = time.perf_counter()
                    checkpoint(output_dir / 'best_validation.pt')
                    performance('best_checkpoint', seconds=time.perf_counter() - save_started, validation_loss=val_loss)
            if step % args.checkpoint_every == 0:
                save_started = time.perf_counter()
                checkpoint(output_dir / "checkpoints" / f"step_{step:06d}.pt")
                performance("checkpoint", seconds=time.perf_counter() - save_started)
        eval_started = time.perf_counter()
        final_val_metrics = evaluate(model, val_loader, base_action, device, return_metrics=True)
        final_val_loss = final_val_metrics['loss']
        if step % args.eval_every:
            monitor()
        performance("final_validation", seconds=time.perf_counter() - eval_started, **final_val_metrics)
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            save_started = time.perf_counter()
            checkpoint(output_dir / 'best_validation.pt')
            performance('best_checkpoint', seconds=time.perf_counter() - save_started, validation_loss=final_val_loss)
        save_started = time.perf_counter()
        checkpoint(output_dir / "final_model.pt")
        performance("final_checkpoint", seconds=time.perf_counter() - save_started)
        if rank == 0:
            if writer:
                writer.close()
            print(f"Training complete: {output_dir / 'final_model.pt'}, val_loss={final_val_loss:.6f}", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
