"""Local model loading and atomic, resumable checkpoints for Robomimic."""
from pathlib import Path
import dataclasses
import hashlib
import os
import random
import subprocess
import sys

import numpy as np
import safetensors.torch
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = Path(os.environ.get("OPENPI_ROOT", str(PROJECT_ROOT.parent / "RoboTwin/XPolicyLab/policy/Pi_05/openpi")))
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "configs"))
from runtime_bootstrap import install_transformers_overrides
install_transformers_overrides(OPENPI_ROOT)


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.suffix in {".pt", ".safetensors"}:
        return path
    if (path / "model.safetensors").is_file():
        return path / "model.safetensors"
    if (path / "params").is_dir():
        raise ValueError(f"JAX checkpoint {path}: run scripts/convert_checkpoint.sh first")
    raise FileNotFoundError(f"No .pt or model.safetensors checkpoint at {path}")


def prepare_checkpoint(path):
    """Convert official JAX weights once, using a lock and CPU-only JAX."""
    import fcntl
    path = Path(path).expanduser().resolve()
    if not (path / "params").is_dir():
        return resolve_checkpoint(path)
    official = (Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_base").resolve()
    tag = path.name if path == official else f"{path.name}_{hashlib.sha256(str(path).encode()).hexdigest()[:10]}"
    output = PROJECT_ROOT / "checkpoints" / f"{tag}_pytorch"
    output.parent.mkdir(exist_ok=True)
    with (output.parent / f"{tag}.conversion.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (output / "model.safetensors").is_file():
            env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
            subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/convert_checkpoint.py"),
                            "--checkpoint", str(path), "--output", str(output)], env=env, check=True)
    return resolve_checkpoint(output)


def load_payload(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def action_condition_projection(state_keys, saved_args=None):
    keys = tuple(state_keys)
    inferred = "mlp" if any(key.startswith("action_condition_proj.0.") for key in keys) else "linear"
    saved = (saved_args or {}).get("action_condition_projection", inferred)
    if saved != inferred:
        raise ValueError("Checkpoint action condition projection metadata disagrees with its weights")
    return inferred


def load_model(path, config_name, device, enable_ac=False, precision=None, ac_projection=None):
    from pi05_robomimic_config import get_config
    from openpi.models_pytorch.pi0_experimental import ExperimentalPI0Pytorch
    model_file = resolve_checkpoint(path)
    payload = load_payload(model_file) if model_file.suffix == ".pt" else None
    config = get_config(config_name)
    if payload and "model_config" in payload:
        config = dataclasses.replace(config, **payload["model_config"])
    if precision is not None:
        config = dataclasses.replace(config, dtype=precision)
    model = ExperimentalPI0Pytorch(config)
    if payload:
        state = payload["model_state_dict"]
        state_keys = tuple(state)
    else:
        from safetensors import safe_open
        with safe_open(model_file, framework="pt", device="cpu") as handle:
            state_keys = tuple(handle.keys())
    has_ac = any(key.startswith("action_condition_proj.") for key in state_keys)
    projection = ac_projection or "linear"
    if has_ac:
        projection = action_condition_projection(state_keys, payload.get("args") if payload else None)
        if ac_projection is not None and projection != ac_projection:
            raise ValueError("Cannot change the projection architecture of an existing AC checkpoint")
        model.enable_action_conditioning(projection)
    if payload:
        model.load_state_dict(state, strict=True)
    else:
        safetensors.torch.load_model(model, model_file, strict=True)
    if enable_ac:
        model.enable_action_conditioning(projection)
    if has_ac and not enable_ac:
        raise ValueError("An AC checkpoint must be trained/evaluated with action conditioning enabled")
    return model.to(device), config, payload


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def save_checkpoint(path, model, optimizer, scheduler, args, step, epoch, batch_in_epoch,
                    stats, split, rank_rng, best_val_loss=None):
    from torch.nn.parallel import DistributedDataParallel
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"step": step, "epoch": epoch, "batch_in_epoch": batch_in_epoch,
                "model_state_dict": raw.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "args": vars(args),
                "model_config": dataclasses.asdict(raw.config), "normalization_stats": stats,
                "episode_split": split, "rank_rng": rank_rng, "best_val_loss": best_val_loss}, temporary)
    temporary.replace(path)
