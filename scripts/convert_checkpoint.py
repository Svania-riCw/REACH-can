"""Convert an existing JAX pi0.5 checkpoint using the vendored OpenPI mapper."""
import argparse
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from checkpoint_utils import PROJECT_ROOT
from pi05_robomimic_config import get_config
from convert_jax_weights import convert_pi0_checkpoint
import torch
torch.set_num_threads(4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_pi0_checkpoint(args.checkpoint, "bfloat16", args.output, get_config())
