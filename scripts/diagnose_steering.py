#!/usr/bin/env python3
"""Probe a trained conditional policy on held-out data, without robot rollouts."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse
import hashlib
import json
import math
from pathlib import Path

from checkpoint_utils import load_model, resolve_checkpoint
import torch
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi_robomimic_adapter import create_robomimic_dataloaders
from policy_diagnostics import FixedProbe, monitor_branches, steering_probe
from train_pi05 import parse_base_action, prepare_batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Conditional training .pt with split and normalization")
    parser.add_argument("--data", help="Override saved dataset location (must be the same dataset)")
    parser.add_argument("--output", required=True, help="JSON report path")
    parser.add_argument("--config", default="pi05_robomimic")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--scales", type=float, nargs="+", default=[0., .2, 2.5, 4.5])
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if (args.samples < 1 or args.num_steps < 1 or not math.isfinite(args.tolerance) or args.tolerance <= 0
            or any(not math.isfinite(s) or s < 0 for s in args.scales)):
        parser.error("Need positive samples, steps, tolerance and nonnegative scales")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    model, cfg, payload = load_model(args.checkpoint, args.config, device, enable_ac=True)
    if not payload or payload.get("args", {}).get("policy_mode") != "conditional":
        raise ValueError("Need a trained conditional checkpoint, not a base model with a new zero projection")
    saved = payload["args"]
    _, val = create_robomimic_dataloaders(
        args.data or saved["data"], batch_size=args.samples, num_workers=0,
        action_horizon=cfg.action_horizon, action_dim=cfg.action_dim,
        tokenizer=PaligemmaTokenizer(cfg.max_token_len), prompt=saved["prompt"],
        seed=saved["seed"], stats=payload["normalization_stats"], split=payload["episode_split"],
        dataset_type=saved["dataset_type"], metadata_path=saved.get("metadata"))
    probe = FixedProbe(val.dataset, args.samples, args.seed, device, prepare_batch)
    base = parse_base_action(saved["base_action"], cfg.action_dim, device)
    report = dict(checkpoint=str(resolve_checkpoint(args.checkpoint)), training_step=payload["step"],
                  data=str(Path(args.data or saved["data"]).resolve()), dataset_type=saved["dataset_type"],
                  split=payload["episode_split"], base_action=saved["base_action"],
                  normalization_sha256=hashlib.sha256(json.dumps(payload["normalization_stats"],
                                                               sort_keys=True).encode()).hexdigest(),
                  probe=probe.metadata(), num_steps=args.num_steps, tolerance=args.tolerance,
                  scope="held-out normalized-action diagnostic; not rollout success or recovery evaluation")
    report["branches"] = monitor_branches(model, probe, base)
    print("Fixed GT/base/foreign flow probes complete; checking samplers and steering.", flush=True)
    report["steering"] = steering_probe(model, probe, base, args.scales, args.num_steps, args.tolerance)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(output)
    print(f"Diagnostic complete; consistency checks passed: {output}", flush=True)


if __name__ == "__main__":
    main()
