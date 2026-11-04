# MH Conditional 2k Training and Evaluation

## Training

Training starts from the original `checkpoints/pi05_base_pytorch` weights and the full `data/paper/can_mh_image.hdf5` dataset. It is a separate seed-0 run, not a resume from the MH baseline or another policy.

| Setting | Value |
| --- | --- |
| GPUs | 8, CUDA devices 0-7 |
| Policy | Pi0.5 conditional, MLP action-condition projection |
| Prompt mixture | 45% zero/base prompt, 55% own ground-truth action, 0% shuffled action |
| Training | Full fine-tuning, physical7 action loss, FP32, global batch 64, gradient checkpointing |
| Schedule | 2,000 optimizer steps, peak LR 2.5e-5, 1,000-step warmup, cosine decay to 2.5e-6 |
| Checkpoints | Every 500 steps; validation every 100 steps; best validation checkpoint retained |
| Output | `outputs/pi05_full_sft_physical7_v4_mlp/policies/can_mh/conditional/seed_0/` |

The launcher is `scripts/train_can_mh_conditional_2k_8gpu.sh`. It refuses to reuse an existing output path and does not set `RESUME` or metadata overrides. The concurrent MH baseline run is left untouched.

This run started on 2026-11-03 with 8 DDP ranks. At 2026-11-04 05:42 Beijing time (2026-11-03 21:42 UTC), training had reached step 400/2000. Ordinary training windows take approximately 3.29 seconds per optimizer step; each validation takes approximately 91 seconds, and a checkpoint save takes approximately 36 seconds. Live output is in `logs/mh_conditional_2k_8gpu_20261103.log`; `performance.jsonl` and `checkpoints/` are in the policy output directory.

## Evaluation Protocol

- Use the fixed `outputs/evaluation/can_initial_states_50.json` catalog, environment seeds 100000-100049, and 50 rollouts per configuration.
- Follow `configs/evaluation_can.json`: MH horizon 500 physical control steps, 50-step predicted action horizon, execute 8 steps per policy call, and 10 diffusion denoising steps. Retreat actions count against the 500-step budget.
- For recovery, use the frozen MH κ=4 detector at `outputs/paper_can/detectors/can_mh/seed_0_kappa4/detector.pt`; do not recalibrate its threshold during this comparison.
- Keep the policy checkpoint, catalog, simulator protocol, recovery implementation, and detector fixed across each comparison. Do not provide future ground-truth actions at evaluation time.

## Evaluation Matrix

Run the two waves sequentially after the entire training process has completed, including final validation and final checkpoint saving. Each configuration has two 25-episode shards. Each suite schedules at most eight workers concurrently on GPUs 0-7. Wave 1 has ten shards, so its final two shards queue behind the first eight; wave 2 has eight shards.

| Wave | Checkpoint | Detector / guidance | Additional condition | Episodes | Purpose |
| --- | --- | --- | --- | ---: | --- |
| 1 | 500, 1000, 1500, 2000 | None, S=0 | Normal conditional inference | 200 | Training-curve diagnosis; step 2000 is the primary result |
| 1 | 2000 | None, S=0 | Bypass action-condition projection at inference | 50 | Measure the trained MLP's inference contribution |
| 2 | 2000 | MH κ=4, S=0 | Detector-triggered chunk retreat and replanning | 50 | Recovery-process control without negative guidance |
| 2 | 2000 | MH κ=4, S=0.2, 2.5, 4.5 | Full cached failed action plan as negative prompt | 150 | Compare guidance strengths on identical scenes |

Total new evaluation is 450 episodes. The main policy comparison is step-2000 conditional without a detector against the independently trained step-2000 MH baseline on the same catalog. Reuse the baseline result at `outputs/evaluation/mh_baseline_2k_8gpu_sealab1_20261103_204352_utc/` once that run completes. Report absolute success counts and rates, paired per-episode wins/losses, mean physical steps, detector triggers, recoveries, retreat failures, and budget terminations. Do not treat the small 50-scene success-rate difference alone as decisive.

## Commands

Wave 1 (four checkpoints plus the step-2000 projection ablation):

```bash
export OPENPI_PYTHON=../RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python
$OPENPI_PYTHON scripts/evaluate_conditional_mlp_suite.py \
  --policy-dir outputs/pi05_full_sft_physical7_v4_mlp/policies/can_mh/conditional/seed_0 \
  --detector outputs/paper_can/detectors/can_mh/seed_0_kappa4/detector.pt \
  --detector-label mh --gpus 0 1 2 3 4 5 6 7 \
  --checkpoint-steps 500 1000 1500 2000 --baseline-only --ablate-steps 2000 \
  --output outputs/evaluation/mh_conditional_2k_8gpu_20261103/wave1_policy
```

Wave 2 (fixed step-2000 checkpoint and MH detector):

```bash
export OPENPI_PYTHON=../RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python
$OPENPI_PYTHON scripts/evaluate_conditional_mlp_suite.py \
  --policy-dir outputs/pi05_full_sft_physical7_v4_mlp/policies/can_mh/conditional/seed_0 \
  --detector outputs/paper_can/detectors/can_mh/seed_0_kappa4/detector.pt \
  --detector-label mh --gpus 0 1 2 3 4 5 6 7 \
  --checkpoint-steps 2000 --recovery-only \
  --output outputs/evaluation/mh_conditional_2k_8gpu_20261103/wave2_recovery
```

Each wave writes a manifest, status, per-configuration summaries, episode records, and shard logs. The output directories must be fresh. Review paired episode IDs and trigger/retreat traces before drawing conclusions about recovery.

## Automatic Handoff — Armed

`scripts/watch_mh_conditional_2k.py` is running as a detached supervisor (PID `349620`, training PID `338282`). It attached to the existing run at 2026-11-04 05:41:33 Beijing time. Training was not restarted or modified; the separate MH baseline job is not controlled by this supervisor.

The supervisor:

1. Polls every ten seconds without loading a GPU model. It checks the original process start identity and waits for both the launcher and matching training ranks to exit; a step-2000 file alone does not release the GPUs.
2. Requires a step-2000 `final_checkpoint` performance event and the final `Training complete` log marker, without an unhandled process failure. Since the original trainer is not its child, it does not claim to obtain that process's exit code.
3. CPU-validates the 500/1000/1500/2000 and final checkpoints: MH conditional recipe, MLP projection, eight ranks, fixed normalization, and no resume. Records checkpoint hashes and validates the frozen MH Gaussian κ=4 detector.
4. Runs wave 1 and then wave 2, each using all eight GPUs. Failed workers can retry twice with the same episode IDs. A failed wave prevents the next wave from starting.
5. Audits all 450 episodes, MH physical-step accounting, detector/checkpoint identities, complete negative prompts, and rollback order. Writes the aggregate table and paired episode comparisons. If the separately trained MH baseline is already complete and compatible, includes its comparison without modifying its files or waiting for it.

Source snapshots, input hashes and an exclusive supervisor lock are retained. If relevant recorded inputs or source files change after arming, the supervisor stops rather than silently mixing evaluation implementations. Training failure also stops evaluation and records the reason; it does not restart training.

### Outputs

- Root: `outputs/evaluation/mh_conditional_2k_8gpu_20261103/`
- Live supervisor status: `status.json` (`waiting_training` → `verifying_training` → `evaluating` → `auditing` → `completed`, or `failed`/`interrupted`).
- Supervisor log: `logs/mh_conditional_2k_auto_eval_20261103.log`.
- Wave logs: `wave1_policy.log`, `wave2_recovery.log`; detailed worker logs and traces remain under each wave.
- Final artifacts: `REPORT.md`, `results.json`, `audit.json`.
- Supervisor/handoff tests and related regression tests: 34 passed before arming.

### Estimated Completion — Beijing Time (UTC+8)

Estimate recorded at 2026-11-04 05:42 Beijing time, using the live MH training timings and the previous PH two-wave evaluation (400 episodes, approximately 35 minutes excluding preparation). The MH test is longer: 450 episodes, a 500-step rather than 400-step budget, and an extra queued pair of policy shards.

| Milestone | Estimated Beijing time |
| --- | --- |
| Training fully finished, including final validation and saves | 2026-11-04 07:40–08:00 |
| All 450 evaluation episodes audited and aggregate report written | 2026-11-04 08:40–09:30 |

Plan around **09:00 on November 4, 2026** for the complete results, not just the trained checkpoint. Allow roughly 55–90 additional minutes after training for checkpoint hashing, both evaluation waves and audit. These are estimates, not scheduled guarantees: filesystem speed, checkpoint saves, rollout lengths and worker retries can move the finish time. The supervisor continues independently of this chat session.
