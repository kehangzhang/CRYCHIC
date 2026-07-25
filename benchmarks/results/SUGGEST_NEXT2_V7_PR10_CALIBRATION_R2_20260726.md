# Suggest-next2 v7 PR10 global-null calibration integration r2-v2

Status: **pipeline integration passed; formal calibration and inference release
were not evaluated**. Six continuous-design M4 hypotheses had typed partial
bootstrap distributions and remain ineligible for formal release.

## Frozen evidence

- Source commit: `e6fb46b120431d7ecfc41be451e9523cdaa8159f`
  (`dirty=false` in all 12 dataset manifests).
- Frozen config:
  `benchmarks/configs/suggest_next2_v7_pr10_calibration_integration_r2_v2.json`.
- Config SHA-256:
  `4b1f94ff2bc36b2083d1f4e8a32b07045076a5e7b73f7447feef8885e6ff0f6f`.
- M4 protocol SHA-256:
  `89ca902e09e4828ba910f8641e613b764fed72bfdcc2efcedc7419c670e1aae5`.
- Output:
  `/media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/pr10_global_null_calibration_integration_r2_v2_20260726`.
- Campaign manifest SHA-256:
  `f270cd202d7edafe9abdd5d51e23192e17faa31fc86ba3d3ac1b9514b0a1a7c4`.
- Calibration summary manifest SHA-256:
  `2c2ff382626ff2ba27acf99bd1a5d01561ae6c2c01a706226be7d37a928b8dbb`.

The original r2-v1 run was stopped after its first continuous-design pilot
revealed that LOSO counts depend on design structure. No calibration metric was
inspected. The v2 amendment froze the exact per-design counts before restarting
in a new output directory; the v1 config and its checksum remain authenticated
as superseded provenance.

## Execution result

| Design | Datasets | Planned refits | Successful | Sum dataset seconds | Mean dataset seconds | Max process-tree RSS GiB |
|---|---:|---:|---:|---:|---:|---:|
| continuous | 2 | 124 | 124 | 123.61 | 61.81 | 16.42 |
| independent multi-group | 2 | 148 | 148 | 358.75 | 179.38 | 18.37 |
| independent two-group | 2 | 124 | 124 | 136.50 | 68.25 | 16.39 |
| multi-cohort | 2 | 124 | 124 | 124.83 | 62.42 | 16.38 |
| paired | 2 | 100 | 100 | 121.58 | 60.79 | 13.34 |
| repeated | 2 | 100 | 100 | 239.23 | 119.61 | 14.60 |
| **Total** | **12** | **720** | **720** | **1,104.50** |  | **18.37 max** |

Campaign wall time was 447.55 seconds (7.46 minutes), including the serial
memory pilot. The pilot selected three concurrent datasets with 64 persistent
spawn workers per dataset, using up to 192 logical cores. Maximum observed
system-memory fraction was 33.05%, below the frozen 80% limit. The output uses
22 MiB on disk.

All datasets completed, all 720 plan records succeeded, and every dataset
observed the ten required full-refit stages. The post-run verifier rehashed the
frozen config, campaign files, every dataset manifest, every result table, and
every compressed resampling artifact before producing 36 dataset-channel rows
and 18 scenario rows.

## Typed partial bootstrap distributions

Across all three channels there were 381,360 hypothesis-by-resample ledger
records: 381,353 were observed and seven were typed `not_estimable`. All seven
were continuous-design occurrence bootstrap records with reason
`occurrence_design_not_estimable`; there were no failed records. Every
continuous and hypergraph record, every permutation record, and every LOSO
record was observed.

The seven typed NE records affected six of 70 continuous-design occurrence
hypotheses. Five retained 18/19 bootstrap estimates and one retained 17/19.
They represent 0.526% of continuous-occurrence bootstrap records and 0.00184%
of all ledger records. A subject bootstrap can contain only one occurrence
class for a rare event, so its logistic occurrence slope is not identified.
This is a data-resampling estimability outcome, not a crashed full-pipeline
task.

The summary therefore correctly reports
`all_distributions_complete=false`, while the campaign-level pipeline endpoint
passes. The formal finalizer requires
`n_bootstrap_observed == n_bootstrap_total`; consequently all six affected rows
remain unreleased. The campaign has zero formal rows, and no p/q release field
was populated.

## Descriptive null diagnostics

These values are integration diagnostics only. Two independent null DGP
replicates and 19 bootstraps/permutations are far below the frozen formal
requirements of 1,000 null replicates and 1,000 distributions.

| Metric across 18 design-channel scenarios | Minimum | Median | Maximum | Exploratory threshold count |
|---|---:|---:|---:|---:|
| Empirical Type-I at 0.05 | 0.0000 | 0.0467 | 0.0914 | 9/18 in 0.035-0.065 |
| All-null replicate FDR at q=0.10 | 0.0000 | 0.0000 | 0.0000 | 18/18 at or below 0.12 |
| Diagnostic 95% CI coverage | 0.8257 | 0.8710 | 0.9571 | 5/18 in 0.92-0.97 |

The low-resolution percentile bootstrap is not expected to establish coverage,
and these values must not be used to tune the estimator. The gate input was
written for schema validation but the release decision was deliberately not
evaluated.

## Next frozen step

Run the already-preregistered r20-v2 precision/runtime pilot unchanged:

- 20 global-null DGP replicates per design, 120 datasets total;
- 99 bootstraps, 99 permutations, and complete design-specific LOSO;
- 26,400 full-pipeline refits;
- descriptive Type-I, FDR, CI coverage, and typed NE frequency only;
- no formal release decision and no algorithm optimization from r2 metrics.

Linear scaling from this campaign gives a central wall-time estimate of about
4.6 hours, with a practical 3.5-6 hour range because independent multi-group
and repeated designs are slower than the other four designs.

## Integrity hashes

- `run_plan.tsv`:
  `1b38540d7bcedda25f1cec1855c145f9eccab853fe7d06944a53f0c19e78af07`.
- `runs.tsv`:
  `169535e17ed69e9dfee74cdac9c207013fbf8dcacf2b643fd922c5f2a931dfb8`.
- `dataset_channel_metrics.tsv`:
  `c35c43ba5aced3ffd799591d0b1c4d3888929e4632249eac6c2b076658125f96`.
- `scenario_metrics.tsv`:
  `9a9552376043b81e5fa70aa026b0fefca51ef5bc572482d21e2f707ab865c49e`.
- `calibration_gate_input.tsv`:
  `8b2d99f429154776dc9f9efe73e0c87ceced9534de8e32ef88eca4114285e6ac`.
