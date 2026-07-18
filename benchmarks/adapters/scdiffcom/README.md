# scDiffCom 1.1.1 condition-aware benchmark

This adapter reproduces the condition-aware scDiffCom arm from Supplementary
Section S4 of Cesaro et al. (`10.1093/nargab/lqaf084`). It is pinned to
scDiffCom 1.1.1 at Git commit
`7877de254380cfb3a0457d5548969129aeb47137` and the harmonized 2,293 simple,
directed ConnectomeDB2020 pairs.

The paper-side normalized Seurat matrices are not present in the frozen
counts-ready H5AD files. The adapter therefore exports only ConnectomeDB genes
from the raw `counts` layer, creates a Seurat object, and applies the default
`LogNormalize` transform. This normalization recomputation is a declared
protocol deviation. All `run_interaction_analysis` settings remain defaults
except the S4 settings (`iterations=1000`, BH differential threshold 0.05,
`threshold_logfc=log(1.5)`), the fixed seed 20260717, and the custom LRI table.

scDiffCom 1.1.1 calls the legacy `GetAssayData(slot=...)` API, which is defunct
in the installed SeuratObject 5 environment. The isolated R runner installs a
process-local namespace bridge from `slot` to the identical `layer` semantics.
It does not modify installed package files, and the compatibility action is
recorded in the run manifest.

`cond1_name` is the reference condition and `cond2_name` is the target, so
scDiffCom `UP` calls belong to the target and `DOWN` calls belong to the
reference. Rankings count native significant directed LR calls after collapsing
cell-pair direction. The complete source cell-pair universe is materialized;
pairs involving a cell type removed by scDiffCom's default five-cell rule are
`not_estimable`, while an analyzed pair with no UP/DOWN call receives a valid
zero.

The R permutation stage uses at most eight `future` multicore workers on Unix,
one BLAS/data.table thread per worker, and reproducible future seeds. A run may
start only below 60% system memory use and is terminated if use reaches 68%.

## Kuppe CTRL versus IZ

```bash
cd "${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
: "${R_ENV_PREFIX:?Set R_ENV_PREFIX to the isolated R environment prefix}"
.venv/bin/python -m benchmarks.adapters.scdiffcom.run \
  ../benchmark_work/multi-group/prepared/kuppe_ctrl_iz/Kuppe_MI_CTRL_vs_IZ.h5ad \
  ../benchmark_work/multi-group/runs/kuppe/scdiffcom_condition_aware_connectomedb \
  --input-manifest ../benchmark_work/multi-group/prepared/kuppe_ctrl_iz/Kuppe_MI_CTRL_vs_IZ.manifest.json \
  --resource ../benchmark_work/multi-group/resources/connectomedb2020/connectomedb2020.tsv \
  --resource-manifest ../benchmark_work/multi-group/resources/connectomedb2020/manifest.json \
  --environment-manifest ../benchmark_work/multi-group/envs/cellchat_2_1_2.manifest.json \
  --rscript "$R_ENV_PREFIX/bin/Rscript" \
  --dataset-id Kuppe_MI_CTRL_vs_IZ --condition-key condition \
  --target IZ --reference CTRL --cores 8
```

## Lerma-Martin MS Ctrl versus CA

Run only after the Kuppe process has exited:

```bash
cd "${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
: "${R_ENV_PREFIX:?Set R_ENV_PREFIX to the isolated R environment prefix}"
.venv/bin/python -m benchmarks.adapters.scdiffcom.run \
  ../benchmark_work/multicondition_v01/prepared/UCSC_Lerma_Martin_MS_CA_vs_Ctrl.h5ad \
  ../benchmark_work/multi-group/runs/ms/scdiffcom_condition_aware_connectomedb \
  --input-manifest ../benchmark_work/multicondition_v01/prepared/UCSC_Lerma_Martin_MS_CA_vs_Ctrl.subset.json \
  --resource ../benchmark_work/multi-group/resources/connectomedb2020/connectomedb2020.tsv \
  --resource-manifest ../benchmark_work/multi-group/resources/connectomedb2020/manifest.json \
  --environment-manifest ../benchmark_work/multi-group/envs/cellchat_2_1_2.manifest.json \
  --rscript "$R_ENV_PREFIX/bin/Rscript" \
  --dataset-id LermaMartin_MS_CA_vs_Ctrl --condition-key lesion_type \
  --target CA --reference Ctrl --cores 8
```

Each completed run contains the full scDiffCom RDS, all detected calls,
UP/DOWN condition-specific calls, full-universe rankings, session information,
logs, exported checksum-bound inputs, and `run_manifest.json`.
