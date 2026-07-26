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
`benchmarks/configs/suggest_next2_v7_real_e1_v1.json`.

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
  --truth-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/spatial_truth/kuppe_figure3_sample_floor_exclude_self_v2/manifest.json \
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
  --truth-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/spatial_truth/ms_figure3_sample_floor_exclude_self_v2/manifest.json \
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

Focused verification: 8 real-runner tests passed. These tests cover frozen roles,
subject-level contrasts, G1 component projection, the G4 sender-resolution guard,
withheld formal fields, fixed-K/continuous DES and complete cell-pair axes.
