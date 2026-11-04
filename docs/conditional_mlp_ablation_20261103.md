# Conditional MLP-only ablation

This experiment replaces the per-token linear action-condition projection with
`Linear(32,64) -> ReLU -> Linear(64,64) -> ReLU -> Linear(64,1024)`.
Only the final layer is initialized to zero; hidden layers retain PyTorch's
default initialization so they can learn once the output layer begins updating.
All projection parameters train jointly with the full Pi0.5 policy.

The output is still added to the noisy action-token embeddings. There is no
new temporal projector, visual concatenation, detector training, or recovery
change. This isolates nonlinear prompt encoding, not full author-architecture
equivalence.

## Controlled settings

- Start from the same original `checkpoints/pi05_base_pytorch`, not the previous
  linear conditional checkpoint or baseline checkpoint.
- Paired data, official scene-isolated 180/20 split, seed 0, quantile
  normalization, two RGB cameras, horizon 50, and physical-first-7 loss remain.
- Eight GPUs, micro batch 8, accumulation 1, global batch 64, FP32 full SFT.
- Empty prompt probability 0.45, own GT probability 0.55, shuffle probability 0.
- AdamW: peak 2.5e-5, minimum 2.5e-6, warmup 1000, betas 0.9/0.95, decay 1e-10.
- Stop at exactly 10,000 optimizer updates. Preserve the previous linear run's
  13,850-update cosine schedule using `--lr-decay-steps 13850`; shortening the
  cosine schedule to 10,000 would introduce a learning-rate confound.
- Validation and fixed branch probes every 100 updates; checkpoints every 500.
- Projection architecture is saved in run/checkpoint arguments and detected from
  state-dict keys when loading. Old linear checkpoints remain supported.

## Launch

```bash
bash scripts/train_can_paired_mlp_10k_8gpu.sh
```

Default output:
`outputs/pi05_full_sft_physical7_v4_mlp/policies/can_paired/conditional/seed_0`.
The launcher accepts `OUTPUT` and `MASTER_PORT` overrides. Do not reuse an
existing output directory for an independent fresh experiment.

Compare linear and MLP checkpoints first with no detector on the same 50-state
catalog. Recovery evaluations should use identical detector, threshold, retreat
settings, guidance scale, and checkpoint update number.

## Evaluation launched November 3

Training finished at step 10,000, with a final training loss of 0.010374 and
validation loss of 0.384393. Validation flow loss is not rollout success rate.

The evaluation suite covers steps 8,000 and 10,000. Each checkpoint has five
configurations: no detector, plus the existing frozen PH detector with scales
0, 0.2, 2.5, and 4.5. Each configuration executes the same 50 catalog episodes,
for 500 episodes total. Scale zero with a detector still includes physical
retreat and replanning; the separate no-detector control is necessary.

Eight GPU workers process 25-episode shards from a shared queue. The first
four configurations use GPU pairs 0/1 (8k without detector), 2/3 (10k without
detector), 4/5 (8k PH, scale 2.5), and 6/7 (10k PH, scale 2.5). Finished workers
take queued shards immediately. Failed shards retry at most twice, retaining
completed episodes. Existing merge validation checks common model, protocol,
catalog, unique episode IDs, and full 50-episode coverage before publishing
each configuration summary.

All episodes save two-camera MP4s and action/detector/recovery traces. The
existing paired evaluation protocol stays at 400 physical steps, 10 denoising
steps, and 8 executed actions per replan. Detector thresholds remain frozen.
No training is restarted by this suite.

```bash
../RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/evaluate_conditional_mlp_suite.py \
  --output outputs/evaluation/conditional_mlp_8k_10k_ph_suite_20261103 \
  --gpus 0 1 2 3 4 5 6 7 --video --max-retries 2
```

The suite directory contains `manifest.json`, `launch.json`, `status.json`,
`results.json`, `suite.log`, and `checkpoint_verification.json`. Each cell has
its merged `summary.json` and `episodes.jsonl`; shard subdirectories retain
videos, traces, run fingerprints, and simulator/driver logs. The launcher
rejects a duplicate suite directory.

Compare MLP step 8,000 against the completed linear step 8,000 PH suite at
`outputs/evaluation/physical7_v4_conditional8k_ph_suite_20261102`. That is the
primary projection comparison. MLP step 10,000 additionally measures the
effect of continued training; differences from linear step 8,000 alone do
not isolate architecture. All results currently use only training seed 0.
