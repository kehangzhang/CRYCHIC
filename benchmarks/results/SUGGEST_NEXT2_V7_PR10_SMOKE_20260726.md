# Suggest-next2 v7 PR10 full-refit smoke

Status: **six-design integration complete; formal release not evaluated**.

## Frozen evidence

- Source commit: `ff9b117c7a6be50101952132eea4ca957e341c68`
  (`dirty=false` in every dataset manifest).
- Runner: `benchmarks/simulation/run_v7_full_refit_campaign.py`.
- Frozen config:
  `benchmarks/configs/suggest_next2_v7_pr10_smoke_v1.json`.
- Config SHA-256:
  `bc0e313f079e7fc990d39e1b95728d9f4ba1634cb04358a2d538379ce8df8768`.
- M4 protocol SHA-256:
  `89ca902e09e4828ba910f8641e613b764fed72bfdcc2efcedc7419c670e1aae5`.
- Output:
  `/media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/pr10_six_design_smoke_v1_20260726`.
- Campaign manifest SHA-256:
  `34e06edd33428b6160a2275a8036302ad5ef13fdcc6a80e2b7778a5dd0e9d962`.
- All dataset tables and compressed plan/record artifacts were independently
  rehashed after completion; every recorded SHA-256 matched.

This is a diagnostic smoke with two bootstraps, two legal condition
permutations, and complete LOSO. It cannot satisfy the frozen minimum of 1,000
bootstraps and 1,000 permutations and was run without a calibration gate.
Consequently all formal p/q/CI fields remain closed.

## Core defects fixed before execution

1. Continuous M5 now accepts the registered `slope:dose` contrast.
2. Continuous null plans jointly permute `condition` and `dose`.
3. Multi-cohort permutations are stratified by cohort.
4. Three-level repeated trajectories require an explicit reviewed
   within-subject complete-context permutation.
5. Differential-only sample metadata is retained and content-bound without
   adding it to the M0/M1/M2 nuisance design.
6. The unified v7 estimator now uses the fold-fitted M4 projector rather than
   dropping the ECDF functional columns.
7. M4 resampling uses prevalence difference for categorical designs and the
   exposure log-odds slope for continuous designs.

## Execution result

| Design | Full refits | Successful | Wall seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|
| continuous | 20 | 20 | 309.59 | 274.0 |
| independent two-group | 20 | 20 | 302.40 | 288.4 |
| independent multi-group | 28 | 28 | 967.10 | 308.8 |
| paired | 12 | 12 | 194.60 | 278.5 |
| repeated | 12 | 12 | 422.78 | 307.2 |
| multi-cohort | 20 | 20 | 307.42 | 280.2 |
| **Total** | **112** | **112** | **2,504 summed** | **308.8 max** |

Campaign wall time was about 21.3 minutes versus 41.7 summed dataset-minutes.
The maximum observed system-memory fraction was 18.63%, well below the frozen
80% limit. The pilot selected dataset-level process parallelism. A fixed-input
scaling diagnostic showed that in-process resample threads were counterproductive:
1, 2, and 4 threads required 32.16, 51.18, and 64.16 seconds respectively for
the same eight plans and produced the same scientific result ID.

Every successful record observed all ten required stages: fold planning,
availability, M0, M1, sender attribution, M2 coupling, held-out application,
design-aware effects, M4, and M5. All 112 child cross-fit IDs and all 112 child
M4 result IDs were unique and excluded the point result. Every child M5 fit ID
excluded the point fit; 111 distinct content IDs were observed because two
multi-group refits produced the same fitted M5 content.

## Hypothesis completeness

| Channel | Rows | Bootstrap complete | Permutation complete | LOSO complete |
|---|---:|---:|---:|---:|
| continuous raw effect | 1,750 | 1,750 | 1,750 | 1,750 |
| M5 posterior effect | 1,050 | 1,050 | 1,050 | 1,050 |
| M4 occurrence effect | 350 | 349 | 349 | 209 |

The M4 deficit is localized and explained:

- continuous: one of 35 rows was NE in one bootstrap, permutation, and LOSO;
- paired and repeated: every M4 LOSO effect was NE;
- all independent categorical M4 distributions were complete.

The paired/repeated issue is a frozen-scale incompatibility. These DGPs contain
eight complete subject trajectories. After LOSO, seven remain; a two-fold
cross-fit then has one training side with only three subjects, below the M4
fold-calibration minimum of four. The child pipeline correctly returns
`occurrence_design_not_estimable`. At least nine trajectories are required;
12 is the next validation scale to provide margin.

## Null diagnostics

Truth is supplied only for the registered `Receiver` events; unknown receiver
events were excluded rather than treated as negatives. All 350 continuous,
210 M5, and 70 M4 known-null point effects were estimable. Descriptive pooled
point RMSE was 0.1842 for continuous raw effects, 0.0924 for M5, and 0.5651 for
M4 (the continuous M4 subset is on the log-odds scale).

No diagnostic p-value was at most 0.05 and no diagnostic q-value was at most
0.10. This is not Type-I evidence: with only two permutations, the smallest
possible corrected p-value is 1/3. Likewise, two-bootstrap percentile coverage
was only 93/350 continuous, 63/210 M5, and 14/69 M4 intervals. These values
demonstrate that the smoke distributions are intentionally too small for
calibration, not that a release gate passed.

## Verdict and next gate

- **PR10 integration gate: pass.** All six design families execute complete raw
  refits, rerun M4/M5, retain compact authenticated children, and persist
  resumable checksum-bound outputs.
- **PR10 distribution-completeness gate: pass for continuous and M5; fail for
  paired/repeated M4 at the frozen n=8 scale.**
- **Formal inference gate: not evaluated and not passed.** Zero formal rows were
  released.

The next evidence-producing run is a paired/repeated n=12 LOSO scale check.
After that passes, the runtime blocker is process-based parallelism within one
large resampling distribution; the current thread backend is slower than
serial and is unsuitable for the required 1,000 + 1,000 formal distributions.
