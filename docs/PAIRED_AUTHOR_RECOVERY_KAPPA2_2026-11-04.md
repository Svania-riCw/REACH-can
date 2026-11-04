# Paired conditional 2k: kappa=2 versus matched kappa=4

## Scope

User-requested threshold-only retest of both PH and Paired error detectors. The existing AE weights, policy checkpoint, action normalization, held-out calibration statistics, 50-scene catalog, rollout implementation and recovery protocol remain unchanged. Gaussian threshold changes from mean + 4 × std to mean + 2 × std. This experiment does not retrain either detector or policy.

The kappa4 detector files remain untouched. Derived files are saved separately under `outputs/paper_can/detectors/can_ph/seed_0_kappa2/` and `outputs/paper_can/detectors/can_paired/seed_0_kappa2/`, including original-file SHA256, original calibration and unchanged-weight provenance. The evaluator also checks every saved model tensor against its kappa4 source before launch.

| Detector | Calibration samples | Calibration mean | Calibration std | Kappa2 threshold |
| --- | --- | --- | --- | --- |
| PH | 2324 | 0.006589236245258181 | 0.0022671275533066504 | 0.011123491351871483 |
| Paired | 2015 | 0.0063050449618991125 | 0.002577740218687983 | 0.011460525399275077 |

## Fixed evaluation matrix

- Policy: Paired conditional MLP, seed 0, step 2000.
- Detector datasets: PH and Paired, both kappa=2.
- Guidance scales: 0, 0.2, 2.5, 4.5, each with the same 50 catalog episodes; total 400 episodes.
- Eight GPUs: 0–3 for PH and 4–7 for Paired. Two 25-episode shards per configuration, four active workers per detector suite.
- Eight executed actions per inference, ten denoising steps, 400 total physical steps per episode.
- Author previous/current detector-flag branching and normal-forward-only action/prompt cache updates.
- EEF servo retreat: at most 20 control steps per historical target, 5 mm / 0.05 rad tolerances. Timeout failure still ends the episode.

## Reproduction and outputs

Entry point: `scripts/evaluate_paired_author_recovery.py --kappa 2 --output <fresh output directory> --gpus 0 1 2 3 4 5 6 7`.

Frozen runner: `/2025233210/lmx/reach_can_training_author_recovery_kappa2_20261104_090449_utc`.

Run output: `outputs/evaluation/paired_conditional2k_author_recovery_kappa2_timeout20_8gpu_20261104_090449_utc`.

Matched kappa4 reference: `outputs/evaluation/paired_conditional2k_author_recovery_timeout20_8gpu_20261104_073318_utc` (400/400, audited). Kappa4 success rates: PH 72% at every scale; Paired 68%, 68%, 70%, 70%.

The evaluator validates the rollout implementation hash, policy hash, normalization hash, catalog, recovery protocol and detector-source provenance against this reference. It writes live `status.json`, final `REPORT.md`, `results.json` with per-episode rescued/lost comparisons, and `audit.json`.

Validation: all 105 unit tests pass, including threshold-only kappa2 derivation, preservation of kappa4 files, unchanged tensors/statistics and disjoint GPU allocation.
