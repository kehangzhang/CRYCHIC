# scSeqCommDiff 2.0.0 paper benchmark adapter

This adapter reproduces the differential communication protocol in Cesaro et
al. (2025). It runs the method's two-condition or multi-sample scenario and
retains the paper's intracellular evidence gate. The `native` arm uses the
frozen 2,293 directed pairs in ConnectomeDB2020. The `H-common` arm accepts a
checksum-bound, fully covered subset of those pairs so that methods can be
compared on the same LR axis; it does not change the scSeqCommDiff algorithm.

For the multi-sample scenario, intercellular scores use the native Wilcoxon
test, intracellular evidence uses pseudo-Wilcoxon, and significant calls use
the unadjusted intercellular p-value below 0.05. For the two-condition scenario,
the runner uses 1,000 label permutations, Wilcoxon intracellular testing, and
BH-adjusted intercellular p-values below 0.05. Both scenarios require either a
maximum `S_intra` above 0.5 or no available `S_intra`. The sign of native
`logFC_S_inter` assigns a call to the target or reference condition.

## Version provenance

Zenodo record 12790607 contains a source archive whose `DESCRIPTION` reports
version 1.0.0. Its complete `R/` implementation is byte-identical to Git commit
`5a29240a237c5520c83ab82527ba55e0e1ceaff5`, the first commit reporting version
2.0.0. The adapter therefore pins that commit. Later repository HEADs are not a
paper reproduction arm.

The benchmark resource has checksum
`e781363288a26c15e03246500111bfecb818eef997f5ebe1b936aaa465151c3a`.
It exactly matches the 2,293 unique ligand-receptor pairs in the fixed package's
`LR_pairs_ConnectomeDB_2020` object.

For a harmonized comparison, pass `--resource-mode H-common` together with a
manifest accepted by `load_harmonized_resource`; every row must have
`scseqcommdiff_covered=true`. Native runs may omit `--resource-mode` because
`native` is the default. Native and H-common outputs are separate benchmark
arms and must not share a leaderboard.

## Environment

Create the isolated environment, then install the package's bundled
`chorddiag`, the CRAN-only dependency, and the pinned package source:

```bash
REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-$REPO_ROOT/../benchmark_work/multi-group}
SCSEQ_ENV=${SCSEQ_ENV:-$BENCHMARK_ROOT/envs/scseqcomm_2_0_0_conda}
WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT
mamba env create -p "$SCSEQ_ENV" \
  -f "$REPO_ROOT/benchmarks/adapters/scseqcommdiff/environment.yml"
git clone https://gitlab.com/sysbiobig/scseqcomm.git "$WORK_DIR/scseqcomm-paper"
git -C "$WORK_DIR/scseqcomm-paper" checkout 5a29240a237c5520c83ab82527ba55e0e1ceaff5
unzip "$WORK_DIR/scseqcomm-paper/other_deps/chorddiag_v0.1.3.zip" \
  -d "$WORK_DIR/scseqcomm-chorddiag"
"$SCSEQ_ENV/bin/R" CMD INSTALL \
  "$WORK_DIR/scseqcomm-chorddiag/chorddiag-0.1.3"
"$SCSEQ_ENV/bin/Rscript" -e \
  'install.packages("add2ggplot", repos="https://cloud.r-project.org")'
"$SCSEQ_ENV/bin/R" CMD INSTALL "$WORK_DIR/scseqcomm-paper"
```

`doRNG` must be attached before the fixed implementation enters its parallel
loop; `run.R` does this explicitly. The Linux runner uses `doMC`, which shares
the sparse expression matrix through copy-on-write instead of serializing it to
PSOCK workers. BLAS thread counts are fixed at one by the Python adapter.

In the multi-sample scenario, the fixed package assumes that every analyzed
cell type has pseudobulk columns in both conditions. Its internal aggregation
drops sample-cell-type groups containing one cell and fails on a cell type that
has no remaining column in either condition. Before inference, the adapter
therefore applies the symmetric native estimability rule already implied by the
package: retain a cell type only when both conditions contain at least two
sample units with more than one cell. It records every sample-cell-type count
in `scseqcommdiff_pseudobulk_support.tsv`, retains the full cell-pair universe
in the ranking output, and marks pairs involving excluded types as
`not_estimable`; missing types are never zero-filled. This rule is independent
of condition order and does not modify the pinned package implementation.

The fixed multi-condition permutation code likewise requires every analyzed
cell type to contain at least two cells in both conditions. It removes a
condition-specific cluster with fewer than two cells and then subtracts the two
score frames without aligning them. When this native precondition is violated,
the adapter applies it symmetrically before inference, records pooled condition
counts in the support table, and marks pairs involving an excluded type as
`not_estimable`. The support table retains its shared multi-sample column names
for schema compatibility; the manifest policy gives their scenario-specific
interpretation.

## Formal multi-sample runs

The primary paper-matched arm treats tissue/sample sections as `Sample_ID`.
Run Kuppe CTRL versus IZ as follows:

```bash
cd "${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
.venv/bin/python -m benchmarks.adapters.scseqcommdiff.run \
  ../benchmark_work/multi-group/prepared/kuppe_ctrl_iz/Kuppe_MI_CTRL_vs_IZ.h5ad \
  ../benchmark_work/multi-group/runs/kuppe/scseqcommdiff_sample_common_estimable_connectomedb \
  --input-manifest ../benchmark_work/multi-group/prepared/kuppe_ctrl_iz/Kuppe_MI_CTRL_vs_IZ.manifest.json \
  --resource ../benchmark_work/multi-group/resources/connectomedb2020/connectomedb2020.tsv \
  --resource-manifest ../benchmark_work/multi-group/resources/connectomedb2020/manifest.json \
  --rscript ../benchmark_work/multi-group/envs/scseqcomm_2_0_0_conda/bin/Rscript \
  --python-executable "$PWD/.venv/bin/python" \
  --dataset-id Kuppe_MI_CTRL_vs_IZ --scenario multi-sample \
  --condition-key condition --sample-unit-key sample_id \
  --target IZ --reference CTRL --cores 8 --min-cells 30
```

Run the MS Ctrl versus chronic active lesion contrast as follows:

```bash
cd "${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
.venv/bin/python -m benchmarks.adapters.scseqcommdiff.run \
  ../benchmark_work/multicondition_v01/prepared/UCSC_Lerma_Martin_MS_CA_vs_Ctrl.h5ad \
  ../benchmark_work/multi-group/runs/ms/scseqcommdiff_sample_common_estimable_connectomedb \
  --input-manifest ../benchmark_work/multicondition_v01/prepared/UCSC_Lerma_Martin_MS_CA_vs_Ctrl.subset.json \
  --resource ../benchmark_work/multi-group/resources/connectomedb2020/connectomedb2020.tsv \
  --resource-manifest ../benchmark_work/multi-group/resources/connectomedb2020/manifest.json \
  --rscript ../benchmark_work/multi-group/envs/scseqcomm_2_0_0_conda/bin/Rscript \
  --python-executable "$PWD/.venv/bin/python" \
  --dataset-id UCSC_Lerma_Martin_MS_CA_vs_Ctrl --scenario multi-sample \
  --condition-key lesion_type --sample-unit-key sample_id \
  --target CA --reference Ctrl --cores 8 --min-cells 30
```

Use `--sample-unit-key subject_id` in separately named sensitivity arms to avoid
treating repeated sections from one donor as independent. This changes Kuppe
from 4 CTRL/11 IZ samples to 4 CTRL/7 IZ subjects and MS from 6 Ctrl/6 CA
sections to 6 Ctrl/5 CA subjects, so it must not overwrite the primary arm.

To run the condition-aware paper arm, use a new output directory and change
`--scenario` to `multi-condition`; the default `--nrep 1000` is the paper
setting.

Each completed directory contains the complete native R result, selected
differential interactions, unordered cell-pair cardinality rankings, logs,
session information, and a checksum-bound run manifest. Use `--preflight-only`
to validate an input, resource, and R environment without loading the full
matrix or starting inference.
