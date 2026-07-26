# suggest_next2 v7 PR10 continuous-bootstrap amendment (2026-07-26)

Status: **r20-v3 superseded before calibration metric inspection; fixed
r20-v4 preregistered**.

## Observed failure

The first r20-v3 dataset was the global-null continuous design, replicate 1,
seed `2009894363`, with 24 subjects.  Subject bootstrap index 45 sampled only
two A-context and 22 B-context subject blocks; source subject D18 was drawn six
times.  The complete refit correctly returned
`no_estimable_subject_fold_plan`, which would leave the bootstrap distribution
incomplete.  Memory was not causal: system memory was about 22% and the
process-tree peak was about 15.3 GiB.

The campaign was stopped after this failure was observed.  No calibration
metric, Type-I estimate, FDR estimate, coverage estimate, method ranking, or
real/locked result was inspected.  The interrupted root is retained as
diagnostic evidence only:

```text
/media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/
  pr10_global_null_calibration_pilot_r20_v3_20260726
```

Its campaign manifest remains `running` because the process was deliberately
terminated during the first serial memory pilot; it contains zero completed
datasets and must not be resumed or summarized as a result campaign.

## Root cause and correction

The v1 policy was named design-stratified, but for a continuous differential
design it did not include the categorical A/B context used by the cross-fit
score model.  Thus the differential slope still had 24 sampled dose/response
pairs, while the learned score functional could lose the contextual support
required to construct both held-out folds.

Commit `721202898a611587e3594d2d52a8c354c8a6f739` introduces bootstrap policy
`design_stratified_complete_subject_block_with_subject_invariant_context_v2`.
For a continuous design, only cross-fit context columns that are constant
within each subject are added to subject-bootstrap strata.  Dose remains a
resampled continuous exposure.  Paired and repeated contexts vary within a
subject, so their complete-trajectory bootstrap is unchanged.

## Verification

- Independent, continuous, and paired planner regressions passed.  All 99
  continuous test plans preserved 12 A and 12 B draws; paired plans still
  resampled five complete trajectories with replacement.
- The exact formal dataset seed and fixed failing resample index 45 produced
  12 A and 12 B draws and completed both v7-primary cross-fit folds.
- The complete v7 full-refit resampling unit suite passed 9/9 tests, including
  serial, thread and safe-spawn process determinism.
- The campaign runner suite passed 11/11 before the new config amendment; the
  expanded config-focused suite passed 11/11.  Ruff, compilation, JSON and
  diff checks passed.

## Frozen replacement

Config:
`benchmarks/configs/suggest_next2_v7_pr10_calibration_pilot_r20_v4.json`
(SHA256 `20230a5351f925269d70b511b31071c6142adae48e8e01a494e079ae8de4e6c3`).

r20-v4 retains 20 replicates for each of six global-null designs, 12 subjects
per level, 99 bootstraps, 99 legal condition permutations, complete LOSO, the
same M0--M5 estimator, 64 process workers per dataset, automatic three-dataset
concurrency, and the 80% memory cap.  It changes only the continuous
subject-bootstrap stratification needed to preserve the pre-existing
cross-fit context axis.  It is still a descriptive calibration/runtime pilot,
not the formal 1,000-replicate or 1,000-resample release evaluation.
