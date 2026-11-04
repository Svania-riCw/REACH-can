# Paired conditional 2k: author recovery state and cache retest

## Requested changes

- Preserve the local physical EEF servo retreat, without simulator-state restoration.
- Match the author's previous/current detector-flag branches. After retreat, detect again before negative-prompt inference. A previous true flag selects negative-prompt inference regardless of the current flag; otherwise a current true flag selects retreat; both false select ordinary forward inference.
- Refresh the executed EEF chunk and complete action prediction only on ordinary forward execution. Neither cache is refreshed by a recovery prediction. Thus a later retreat can revisit the last normal chunk rather than the latest guided chunk, like the author's normal-forward-only action buffer.
- Increase the per-target servo deadline from 10 to 20 control steps, while retaining 5 mm / 0.05 rad tolerances. Eight targets allow at most 160 retreat steps, within the unchanged 400-step episode budget. A deadline miss still ends the episode as `rollback_failed`.
- Record cache-update provenance, post-retreat detections and recovery branches in episode traces; audit them against the cached full-plan hash and prediction seed.

## Retest matrix

| Policy | Error detector | Kappa | Guidance scales | Episodes per scale | GPUs |
| --- | --- | --- | --- | --- | --- |
| Paired conditional MLP, seed 0, step 2000 | PH | 4 | 0, 0.2, 2.5, 4.5 | 50 | 0–3 |
| Same checkpoint | Paired | 4 | 0, 0.2, 2.5, 4.5 | 50 | 4–7 |

Total: 400 episodes, fixed catalog seeds 100000–100049, eight executed actions per inference, ten denoising steps, full 50×7 negative prompt normalized with checkpoint statistics. Each detector suite uses two 25-episode shards per configuration and schedules four GPU workers at a time.

Checkpoint: `outputs/pi05_full_sft_physical7_v4_mlp/policies/can_paired/conditional/seed_0/checkpoints/step_002000.pt`.

Catalog: `outputs/evaluation/can_initial_states_50.json`, SHA256 `9f3a555765e617499370f26e8c7b8a85b260274bf70b6fed10ffd1548b9f246b`.

Run output: `outputs/evaluation/paired_conditional2k_author_recovery_timeout20_8gpu_20261104_073318_utc`.

Frozen runner: `/2025233210/lmx/reach_can_training_author_recovery_20261104_073318_utc`.

Entry point: `scripts/evaluate_paired_author_recovery.py`. It validates the policy, detector calibration, catalog and protocol; schedules both detector suites concurrently; then automatically writes `results.json`, `audit.json` and `REPORT.md`. `status.json` tracks live progress. Source hashes and snapshots are retained with the output.

## Interpretation

The previous no-detector result for this checkpoint is 35/50 (70%). Previous PH-detector recovery results were 36/50 at each guidance scale. This retest jointly changes the recovery branching/cache handling and the servo timeout, so it cannot isolate the effect of either modification alone. The EEF servo implementation and timeout-failure termination remain different from the author's reverse-action playback.

Validation before launch: 56 related unit tests pass, including persistent anomaly handling, normal-only cache provenance, post-retreat detection and slow-servo convergence with the longer deadline.

## Completed results

All 400 episodes completed and passed the policy/protocol/catalog, physical-step accounting and author-state/cache trace audits at Beijing time 2026-11-04 16:22:24. The full 103-test suite also passes.

| Guidance S | PH detector success | PH mean physical steps | PH rollback failures | Paired detector success | Paired mean physical steps | Paired rollback failures |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 36/50 (72%) | 176.66 | 3 | 34/50 (68%) | 193.08 | 1 |
| 0.2 | 36/50 (72%) | 179.22 | 2 | 34/50 (68%) | 193.08 | 1 |
| 2.5 | 36/50 (72%) | 179.22 | 2 | 35/50 (70%) | 188.26 | 1 |
| 4.5 | 36/50 (72%) | 183.44 | 1 | 35/50 (70%) | 188.24 | 1 |

PH recovers episode 27 relative to the same checkpoint's no-detector result and loses none at every scale. Paired loses episode 14 at S=0/0.2; S=2.5/4.5 has the exact same successful episode set as the no-detector reference. Higher guidance has not demonstrated an additional success-rate benefit with the PH detector in this 50-scene test.

At S=0, rollback-failure terminations fall from 5 to 3 for PH and from 4 to 1 for Paired, but success counts remain 36 and 34 respectively. PH mean physical steps increase from 167.50 to 176.66. Fewer timeout failures therefore do not by themselves demonstrate more successful task recoveries; some episodes instead exhaust the common budget. Because state/cache handling and timeout changed together, neither change has an isolated causal estimate here.

Full report: `../outputs/evaluation/paired_conditional2k_author_recovery_timeout20_8gpu_20261104_073318_utc/REPORT.md`.

Per-episode comparisons and audit: `results.json` and `audit.json` in the same run directory. All eight GPUs are idle after completion.
