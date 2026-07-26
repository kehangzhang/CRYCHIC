# suggest_next2 v7 PR10 performance record (2026-07-26)

## Scope and status

This record freezes the performance diagnosis and execution-only changes used
before the PR10 global-null calibration pilot.  No DGP, estimand, candidate,
cross-fitting split, bootstrap draw, permutation draw, M0--M5 score, or
multiple-testing rule was changed.  The frozen pilot remains descriptive:
99 bootstraps/permutations and 20 replicates do **not** satisfy the formal
release floor of 1,000 resamples and 1,000 null replicates per scenario.

Frozen execution config:
`benchmarks/configs/suggest_next2_v7_pr10_calibration_pilot_r20_v3.json`
(SHA256 `9423d07ed3b0b6c6f583e249a325dae8592946bb8ae1e4e6e21b2cf4e6e90262`).

## Root cause

The worst diagnosed independent-multi-group fold spent about 259 seconds in
the public full cross-fit path.  Only about 4.4 seconds were training and 41.7
seconds were held-out application; approximately 180 seconds were consumed by
legacy receiver-family/program diagnostics that PR10 never reads.  The old
family-common path alone performed 15 repeated score-table builds per fold.
Additional avoidable costs were a 50 ms process-tree RSS poll on a large host,
repeated full-ledger scans in the inference finalizer, repeated stable-event-ID
hashing per row, and redundant validation/digest work when persisting already
validated campaign artifacts.

## Execution-only changes

- `974ada8`: added the internal `v7_primary_m0_m5_v1` refit profile.  It keeps
  training, frozen availability, M0, signed M1, M2, sender attribution, M4 and
  M5, while omitting only legacy outputs unused by the v7 estimator.  The
  public full `run_subject_crossfit()` behavior is unchanged.
- `e4161e2`: indexed resampling ledger rows once by event, contrast and
  operation, replacing per-hypothesis full-ledger scans.
- `6cc21e7`: reused worker-validated fold/campaign artifacts and prevalidated
  persistence records; public validation remains the default.
- Stable event IDs are cached by unique identity, and process-tree RSS polling
  is performed every 2 seconds.  BLAS/OpenMP thread oversubscription remains
  disabled when process workers are used.

## Measured performance

Worst diagnosed dataset:
`global_null/independent_multi_group/r0003`, seed `1178079459`, 12 subjects per
level and five candidate senders.

| Measurement | Before | After | Improvement |
|---|---:|---:|---:|
| Cross-fit fold | 259.0 s | 42.8 s | 6.05x |
| 64-process finalizer | 332.69 s | 13.81 s | 24.09x |
| 64-process end to end | 866.07 s | 496.73 s | 1.74x |
| 64-process full resampling core | 428.76 s | 430.60 s | unchanged (+0.43%) |

The serial one-bootstrap optimized run completed in 344.73 seconds
(291.70 seconds resampling, 35.23 seconds finalization, 7.65 seconds
persistence).  The unoptimized point-plus-one-bootstrap run did not complete
within ten minutes.  Its four old folds alone require about 1,036 seconds, so
the directly supported end-to-end improvement is conservatively **at least
3.0x**; a larger direct claim is not made without a completed old run.

The final 64-process run completed all 64/64 resamples in 496.73 seconds:
430.60 seconds in full refits, 13.81 seconds in finalization, and 41.67 seconds
in persistence.  Peak process-tree RSS was 16,691,806,208 bytes (15.55 GiB),
and peak whole-system memory fraction was 22.61%, well below the 80% limit.

## Scientific equivalence and verification

- The small full-profile versus v7-primary path matched exactly for output
  digests, score provenance, effects and omnibus results.
- The one-bootstrap pre/post optimization comparison matched all 11 scientific
  tables exactly; maximum absolute numeric difference was 0.
- The final 64-worker v2/v3 comparison used the same dataset ID and random
  draws.  All columns in all 11 Parquet tables matched exactly, including
  33,600 continuous-ledger, 11,200 hypergraph-ledger and 6,720
  occurrence-ledger rows; maximum absolute numeric difference was 0.
- Relevant unit/integration coverage passed: v7 resampling (7 tests), v7 DGP
  and occurrence (22), design-aware inference (18), public full cross-fit
  M0/M2 integration (2), and campaign runner tests (11).  Ruff, compilation
  and diff checks passed.

Reference completed manifest:
`/media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/pr10_v7_primary_multigroup_r3_b64_process_scaling_v3_20260726/datasets/v7_null_calibration_global_null_independent_multi_group_r0003_n12_pr10_calpilot_r20v3_scalecheck_v2/manifest.json`
(SHA256 `81489335776ce66921ddec8ea7f5a9802d693e58bb10b4c0c46a467e7fa5e6c0`).

## Interpretation boundary

These measurements justify using the optimized execution path for the fresh
r20-v3 pilot.  They do not establish calibration, ranking, biological
validity, or a release decision.  Locked simulation, formal null calibration,
independent real-cohort, and spatial validation results must be reported
separately after completion.
