# suggest_next2 v7 Kuppe/MS real E1 protocol

Date: 2026-07-26

Status: implementation, checksum preflight and focused verification complete;
formal Kuppe/MS refits are queued behind the active PR10 calibration campaign.
No result is claimed here.

## Estimator and evidence boundary

The runner freshly fits G0, G2, G3, G4 and G5 with the v7-primary cross-fit path
and paired I1 design-aware effect engine. G1 is not refit: it is a checksum-bound
projection of the existing held-out legacy family-common component table. The
output records this distinction per score row and in the run manifest.

Kuppe is development/non-independent. MS is a reused external cohort and cannot
be used for parameter, K, weighting or rank-aggregation selection. Analytic I1
p/q fields are withheld; real-data edge AUROC and causal sender claims are
forbidden.

Frozen protocol:
`benchmarks/configs/suggest_next2_v7_real_e1_v1.json`, SHA256
`2a733fb6a076996c70075ad36f6085329aa7578b72b59db16fc3ef6a4211d0c8`.

## Bound cohort, truth and resource inputs

| Dataset | Cells x genes | H5AD SHA256 | Preparation manifest SHA256 | Truth manifest SHA256 | Expected-set SHA256 |
|---|---:|---|---|---|---|
| Kuppe | 76,141 x 29,126 | `c47112ce01a192bb157af1ba5feb09c1616601e280102fd11a658f570698c926` | `2254d4e9ec46723cf0619ad8ad26fa8ee2f7ce83e7e223b88cd8f5174571a10e` | `4fe12433c953edaab760c64f369bb44762335346b2ba7a57fe10549ae71fd9b9` | `b883f58e1df3c773d09f80a32f966a5f8db5dee91740f10e2db70d6cd4e7119b` |
| MS paper-matched | 69,168 x 32,115 | `433717d9fd98e57e15a444a338a6e1002f7ca3224022321d28a386c0a8498e1c` | `35644aec92e6e383ef982e18bb3c6aff24c2e5faf8ae52ea921c01c440f0c15d` | `f93042a947ce6dd0260a40f58ebc3155890c03b44d8674b38ffe275fbb2c20bd` | `9c91513503b7715949d247a558e23e9801d33cf56a327a0df52a87b1a2e5342d` |

Both truth manifests use `subject_id`, floor-sized top sets and exclude self
pairs; repeated sections are averaged within subject. The MS binding excludes
the earlier approximately 75k-cell object. Shared inputs
are frozen to the 2,293-row ConnectomeDB2020 payload SHA256 `e7813632...`, its
manifest `3dd10324...`, the NicheNet manifest `4e2a1077...`, and runtime Parquet
`42a6fa37...`. The runner validates all payloads before creating the output
directory, then validates H5AD dimensions and truth-output lineage after load.

## Bound legacy G1 inputs

| Dataset | Source manifest SHA256 | Cross-fit manifest SHA256 | Component SHA256 | Rows |
|---|---|---|---|---:|
| Kuppe | `7e32915ee0b7ee3b18a738aa8571baa7ae54a09abb5d9e9e5f22ce3e34f82735` | `0c14c65fd1c1eb1beb191eb7acb8daf10f08418bf878fdbee6d8534bb2db9da0` | `89363b7a66b811e02deb1ad911aad517bbbb50545bf605b31940b7fd0ecb0a95` | 3,361,380 |
| MS | `cd1c4e7ebb9581e234494dde98ffc930e20a69409e1c1d64327be0cf295ec1ad` | `216f6fa71055f34251ebcc098d7ca8d0b305e1b9f2a16a884ff9c7e3b1aa79a5` | `766f646cff1f8bcdbcf020b01ba9360c12bda519d5e5b5d2d32a1f8ef892dcb5` | 1,750,815 |

The source runs are clean commit `2d10f36` and retain the original fold/sample
lineage. The v7 runner rejects a component row count, provenance ID or checksum
change.

## Execution allocation

Both cohorts have two outer folds. Run them concurrently with two fold workers
and up to 24 BLAS threads per active fold. Their maximum requested compute concurrency is
approximately 96 logical threads. `CRYCHIC_PROGRESS=1` emits fold-stage progress
and ETA to stderr; `run.jsonl` records every high-level stage and wall time.

```bash
CRYCHIC_PROGRESS=1 python -m benchmarks.literature.run_v7_real_multigroup \
  --dataset kuppe \
  --input-h5ad /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/prepared/kuppe_ctrl_iz/Kuppe_MI_CTRL_vs_IZ.h5ad \
  --input-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/prepared/kuppe_ctrl_iz/Kuppe_MI_CTRL_vs_IZ.manifest.json \
  --output-dir /media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/real_kuppe_v7_e1_v1_20260726 \
  --resource /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/resources/connectomedb2020/connectomedb2020.tsv \
  --resource-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/resources/connectomedb2020/manifest.json \
  --database-root /media/subunit/bioinfo/crychic_dev/databases \
  --nichenet-manifest /media/subunit/bioinfo/crychic_dev/databases/nichenet/v2_2021/manifest.json \
  --truth-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/spatial_truth/kuppe_subject_floor_exclude_self_v2_20260726/manifest.json \
  --legacy-source-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/latest_core_2d10f36_20260724/runs/kuppe/manifest.json \
  --legacy-crossfit-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/latest_core_2d10f36_20260724/runs/kuppe/crossfit_result/crossfit_manifest.json \
  --legacy-components /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/latest_core_2d10f36_20260724/runs/kuppe/crossfit_result/family_common_components.parquet \
  --fold-jobs 2 \
  --threads 24 \
  --min-cells 10
```

```bash
CRYCHIC_PROGRESS=1 python -m benchmarks.literature.run_v7_real_multigroup \
  --dataset ms \
  --input-h5ad /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/prepared/ms_figure3_5ctrl_6ca/UCSC_Lerma_Martin_MS_CA_vs_Ctrl_5ctrl_6ca.h5ad \
  --input-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/prepared/ms_figure3_5ctrl_6ca/UCSC_Lerma_Martin_MS_CA_vs_Ctrl_5ctrl_6ca.manifest.json \
  --output-dir /media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/real_ms_v7_e1_v1_20260726 \
  --resource /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/resources/connectomedb2020/connectomedb2020.tsv \
  --resource-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/resources/connectomedb2020/manifest.json \
  --database-root /media/subunit/bioinfo/crychic_dev/databases \
  --nichenet-manifest /media/subunit/bioinfo/crychic_dev/databases/nichenet/v2_2021/manifest.json \
  --truth-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/spatial_truth/ms_subject_floor_exclude_self_v2_20260726/manifest.json \
  --legacy-source-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/latest_core_2d10f36_20260724/runs/ms/manifest.json \
  --legacy-crossfit-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/latest_core_2d10f36_20260724/runs/ms/crossfit_result/crossfit_manifest.json \
  --legacy-components /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/latest_core_2d10f36_20260724/runs/ms/crossfit_result/family_common_components.parquet \
  --fold-jobs 2 \
  --threads 24 \
  --min-cells 10
```

## Required outputs

Each run persists score views, I1 effects, sender-resolved event ledger, three DES
variants, coverage, score geometry and effect summaries. The manifest must be
`complete`, `dirty=false`, bind every output checksum, and retain
`formal_inference_allowed=false` before spatial-geometry evaluation is allowed.

Focused verification: 10 real-runner tests passed. These tests cover frozen roles,
input and payload drift rejection, subject-level contrasts, G1 component
projection, the G4 sender-resolution guard, withheld formal fields,
fixed-K/continuous DES and complete cell-pair axes.
