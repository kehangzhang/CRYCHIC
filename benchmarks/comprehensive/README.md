# Comprehensive multi-context communication benchmark

This directory freezes the design for comparing multi-group and multi-context
cell-cell communication methods. It separates estimands that are often mixed in
one leaderboard and makes unavailable or non-estimable results explicit.

The registry was frozen on 2026-07-22 from the local literature review,
published benchmark protocols, downloaded data, and prior CRYCHIC runs. Raw
data, third-party source trees, environments, and large outputs live outside
Git under `benchmark_work/comprehensive_multicontext_20260722`.

## Terminology correction

There is no identified paper or package named **CytoSigDiff**. Two distinct
benchmarks must not be conflated:

- Cesaro et al. 2025 (`10.1093/nargab/lqaf084`) evaluates differential CCC in
  Kuppe myocardial infarction and Lerma-Martin multiple sclerosis data. Figure
  3 compares CellChat, scDiffCom, scSeqCommDiff, LIANA+, and MultiNicheNet.
- Dimitrov et al. 2022 (`10.1038/s41467-022-30755-0`) evaluates static CCC
  scores against CytoSig and CITE-seq. Its seven method components are CellChat,
  CellPhoneDB, Connectome, Crosstalk, logFC mean, NATMI, and
  SingleCellSignalR. CytoSig itself is downstream cytokine-activity inference,
  not a sender-receiver differential CCC method.

The historical CytoSig panel contains the TNBC and HER2 arms of GSE176078.
Their local truth reconstructions differ, so they are reported separately and
never averaged as if they shared an identical gold standard.

## Non-negotiable analysis rules

1. `subject_id` is the inferential unit. Cells and spatial spots are
   observations used for aggregation, not independent replicates.
2. Repeated samples, tissues, or sections from one subject remain in the same
   fold and bootstrap block. MIS-C family members are additionally clustered by
   `family_id` where the source data document a real family.
3. Structural absence of a cell type is missingness. It is not zero expression
   or a zero communication score.
4. `not_estimable`, `not_returned`, unsupported resources, missing assets, and
   method failure are distinct states. None is scored as zero.
5. Pooled-condition methods are descriptive. Their cell-level permutation
   values are not reinterpreted as population-level condition p-values.
6. H-common, H-covered, and native-resource runs are separate arms. A method
   that cannot represent one arm is marked unsupported, not penalized as a
   false negative.
7. Target genes used to build a NicheNet/MultiNicheNet/CRYCHIC score cannot be
   reused as independent validation truth. Olink, ADT, perturbation, or a
   disjoint held-out target set is required.

## Tracks and estimands

| Track | Unit and output | Methods represented | Primary interpretation |
|---|---|---|---|
| `differential_sample` | Subject-level event effects or priorities | CRYCHIC, by-sample CellChat/CellPhoneDB/LIANA, LIANA+, MultiNicheNet, scSeqCommDiff, STACCato, DCST | Multi-sample condition association |
| `differential_pooled` | One pooled network per condition | CellChat, scDiffCom, pooled scSeqCommDiff, CRYCHIC descriptive arm | Descriptive condition contrast only |
| `downstream_response` | Ligand-to-receiver target support | CRYCHIC, NicheNet, MultiNicheNet, LIANA+, DCST | Mechanistic prioritization |
| `context_program` | Context loadings and communication factors | Tensor-cell2cell, LIANA tensor/MOFA route, CRYCHIC programs | Exploratory multi-context programs |
| `orthogonal_validation` | Independent protein, perturbation, or spatial support | All representable differential methods | Supporting evidence, not complete event truth |
| `orthogonal_static` | Single-condition CytoSig/CITE-seq/IPF checks | Dimitrov components, CRYCHIC, NicheNet proxy | Historical/static appendix only |
| `performance` | Complete declared pipeline | All runnable methods | Runtime, memory, stability, and failure |

The exact method estimand, score direction, resource support, adapter, and
historical-exact flag are recorded in `methods.tsv`. The runnable historical
NicheNet arm is a checksum-pinned source-agnostic prior proxy, not native
`predict_ligand_activities`; a separate native NicheNet arm is registered as
planned so these estimands cannot be conflated.

## Evidence hierarchy

| Tier | Evidence | Valid claims |
|---|---|---|
| G1 | Hidden expression-level simulation with planted effects and legal nulls | Event discrimination, direction, RMSE, type-I error, FDR, and power |
| G2 | Ligand/receptor/drug perturbation | Functional direction and held-out receiver-target recovery |
| G3 | Olink, ADT, phosphoprotein, TF, or ATAC | Orthogonal molecular support, not complete LR truth |
| G4 | Condition-specific spatial support | DES, colocalization, and direction agreement |
| G5 | Phenotype, cross-cohort replication, or incomplete literature gold | Program recovery and robustness, not event specificity |

Kuppe, MS, cSCC, Kang, the seven local simulations, CytoSig, CITE-seq, and IPF
were used during earlier development or diagnosis. They remain regression,
legacy-comparability, or appendix panels and cannot support a new independent
superiority claim.

## Frozen panels

The minimal credible experiment is defined by `P01`-`P04`, `P06`-`P07`,
`P10`, and `P12` in `panels.tsv`:

| Panel | Data | Purpose |
|---|---|---|
| P01/P02 | New hidden two-group, three-group, continuous, null, confounded, composition, structural-absence, and topology simulations | Sample-aware primary truth plus a separate pooled descriptive leaderboard |
| P03 | Tensor-cell2cell 12-context simulation | Program/factor recovery with Hungarian factor matching |
| P04 | MIS-C, healthy siblings, adult severe COVID-19, and held-out Olink | New real three-group orthogonal primary panel |
| P06/P07 | Kuppe MI and Lerma-Martin MS | Paper-compatible Figure 3 legacy DES, split into sample-aware and pooled panels |
| P10 | CytoSig TNBC and HER2 | Dimitrov historical-method static appendix |
| P12 | Hidden simulation sizes and MIS-C | One-core and best-practical scaling profiles |

The full phase adds anti-PD-1 breast cancer, PDAC plus spatial validation,
COVID PBMC/BALF, ASD cortex, scACCorDiON phenotype cohorts, COMMUNITY cohorts,
and static CITE-seq/IPF checks. Controlled SLE/EGA data and inaccessible
scAgeCom data remain blocked; they are not silently replaced with convenience
datasets.

MIS-C is currently the cleanest new real cohort: 20,332 genes, 13,343 cells,
16 subjects, 16 samples, three conditions, and three broad cell types. The
prepared file retains sparse counts and logcounts. The four unrelated adult
subjects share source code `UNR`; preparation deliberately assigns them
separate family blocks rather than treating them as one family.

## Metrics and ranking

All metrics are defined in `metrics.tsv`, including their direction, required
truth, calibration requirement, and aggregation grain.

- Sparse event truth uses prevalence-adjusted average precision as the primary
  discrimination metric. Native AP, AUROC, MCC, precision/recall at frozen k,
  sign accuracy, effect RMSE, and Spearman correlation are secondary.
- Calibrated methods must pass subject-block label-permutation type-I error and
  empirical FDR gates at q=0.01, 0.05, and 0.10. Rank-only or exploratory
  methods receive NE for calibration metrics.
- Figure 3 DES fixes positive GSEA, `gseaParam=1`, raw event cardinality,
  non-self pairs, floor top-set sizes, and native tie handling. Matched-top-k
  DES is reported separately to expose output-density effects.
- Program methods use Hungarian-matched context correlation, factor-event
  AUPRC, normalized reconstruction error, held-out-context error, and factor
  stability. Factor loadings never compete directly with event p-values.
- Robustness reports donor/cell downsampling, annotation and resource
  perturbation, RBO/top-k Jaccard, sign/effect correlation, and failure rate.
- Performance reports wall time, core-hours, process-tree peak RSS, GPU memory
  when applicable, and log-log scaling slopes for cells, subjects, LR pairs,
  and cell types. One-core and best-practical profiles are separate.

Ranking occurs within `(track, gold tier, resource mode, dataset)`. Each study
contributes equal weight regardless of its cell count. Ties receive average
ranks; cross-study normalized ranks use a subject/study-cluster bootstrap with
2,000 replicates. A method must cover at least 80% of preregistered eligible
method-dataset pairs to enter a cross-study summary. Representability and output
coverage are always shown beside accuracy.

There is no grand score that adds AUPRC, DES, factor reconstruction, and
runtime. The report provides separate track leaderboards and a performance-
accuracy Pareto front.

## Freeze and leakage control

1. Freeze the method commit, resource checksums, adapter, metric registry,
   simulation generator, seeds, and environment before generating hidden G1
   truth.
2. Split complete subject and family blocks. Inner folds select parameters;
   outer subjects are evaluation only. Leave-site-out is used where sites or
   batches permit it.
3. Run and checksum every MIS-C method output before reading the Olink endpoint.
   The Olink sheet only covers the MIS-C-versus-healthy contrast; adult severe
   COVID-19 remains a third group for omnibus and prespecified pairwise checks.
   In particular, run the RNA-only MultiNicheNet core workflow and disable its
   tutorial's Olink prioritization extension; otherwise the held-out endpoint
   would be used as a method input.

The MultiNicheNet tutorial and Zenodo description call this Olink material
serum, whereas the Diorio paper Methods describes plasma. The benchmark records
both source terms and refers generically to an external Olink protein endpoint;
the supplied workbook is a 1,463-analyte group summary, not subject-level data.
4. Any method change after viewing an outer-test result creates a new candidate
   version and requires a new hidden seed or cohort.

CRYCHIC RC9 is the frozen current head for the checksum-bound Kuppe/MS replay.
It has no general adapter for a new three-group H5AD, and a calibrated omnibus
test is not yet validated; formal p/q fields remain disabled. This benchmark
must not imply otherwise.
The general-purpose `run_hcommon.py` entrypoint exercises the generic CRYCHIC
multigroup baseline, not the dataset-specific RC9 held-out replay adapter. A
result from that entrypoint must therefore be labelled `CRYCHIC generic
multigroup baseline`; it cannot be presented as an RC9 result.

## Execution order

1. Run the registry/asset audit and resolve checksum or partial-download errors.
2. Smoke-test every adapter on a tiny fixed-universe fixture.
3. Run external simulations and generate the hidden local G1 suite.
4. Run MIS-C methods without opening Olink, then freeze output checksums.
5. Evaluate Olink and produce per-track metrics.
6. Run legacy and static appendix panels without feeding them back into tuning.
7. Run scaling profiles with a 70% no-new-job threshold and a 78% hard memory
   guard, and log wall time, threads, environment, exit status, and peak
   process-tree RSS.

Planned adapters or environments are skipped with a reason until independently
validated. At the current freeze this applies to native NicheNet,
MultiNicheNet, STACCato, DCST, Tensor-cell2cell/LIANA tensor, and historical
Crosstalk. scDiffCom has frozen Figure 3 output but not yet a general isolated
environment.

P13 reproduces the separate scACCorDiON estimand with DW-OT, CORR-OT, GOT, and
Tabular PCA on identical patient communication graphs. Its primary endpoint is
ARI at the known number of labels; the paper's maximum ARI across k=2..7 is
reported only as an optimistic secondary analysis. Of the seven paper cohorts,
the pinned PDAC graph input is ready, COVID and MI are partial, and the exact
breast, kidney-AKI, lung, and RCC objects remain planned.

The currently frozen Kuppe/MS summaries use a Python positive running-sum
analogue of `fgsea`, not the paper's original R `fgsea` call. They are suitable
for regression and protocol-compatible comparison, but an exact R rerun is
still required before claiming byte-level Figure 3 reproduction.

## Reproduction commands

The latest corrected five-seed pilot and MIS-C/Olink summary is
[`results/multigroup_headtohead_20260723.md`](results/multigroup_headtohead_20260723.md).
It reports the generic CRYCHIC baseline separately from RC9 and binds every
large external output by checksum.

Generate the independent-subject A/B/C pilot fixture used by the four-method
head-to-head benchmark. Group sizes are unequal as registered, `X` contains
deterministic CP10K-log1p expression, and `layers["counts"]` retains raw integer
counts for CRYCHIC. The generator freezes the exact five-pair simulation subset
from the supplied four-method resource and writes pairwise inputs because
scSeqCommDiff is natively two-condition:

```bash
.venv/bin/python -m benchmarks.comprehensive.generate_three_group_fixture \
  --output-dir <output>/fixture-5seed \
  --resource <output>/misc-prepared/resource/harmonized_lr.tsv \
  --seeds 20260723,20260724,20260725,20260726,20260727 \
  --group-subjects 8,10,12 --mean-cells-per-sample 180
```

CellChat, LIANA, and scSeqCommDiff must read normalized `X`; do not pass the
raw `counts` layer. CRYCHIC reads `layers["counts"]`. The scSeqCommDiff calls
must pass the generated five-pair table and manifest with
`--resource-mode H-common`; its 2,293-pair native arm is a separate diagnostic.

After all four method adapters finish, score the active and matched global-null
runs on the exact 45-event axis:

```bash
.venv/bin/python -m benchmarks.comprehensive.evaluate_three_group \
  --fixture-dir <output>/fixture-5seed \
  --runs-dir <output>/method-runs \
  --output-dir <output>/evaluation
```

Prepare MIS-C pairwise inputs and the exact 455-LR four-method intersection:

```bash
.venv/bin/python -m benchmarks.comprehensive.prepare_misc_benchmark \
  --input-h5ad ../benchmark_work/comprehensive_multicontext_20260722/prepared/misc_olink/misc_olink.h5ad \
  --input-manifest ../benchmark_work/comprehensive_multicontext_20260722/prepared/misc_olink/preparation_manifest.json \
  --harmonized-resource ../benchmark_work/multicondition_v01/resources/harmonized_simple_lr/harmonized_lr.tsv \
  --harmonized-manifest ../benchmark_work/multicondition_v01/resources/harmonized_simple_lr/manifest.json \
  --connectome-resource ../benchmark_work/multi-group/resources/connectomedb2020/connectomedb2020.tsv \
  --connectome-manifest ../benchmark_work/multi-group/resources/connectomedb2020/manifest.json \
  --output-dir <output>/misc-prepared
```

Freeze RNA-only M-vs-S ligand predictions before invoking the Olink evaluator:

```bash
.venv/bin/python -m benchmarks.comprehensive.export_misc_olink_predictions \
  --prepared-dir <output>/misc-prepared \
  --runs-dir <output>/misc-method-runs \
  --input-h5ad ../benchmark_work/comprehensive_multicontext_20260722/prepared/misc_olink/misc_olink.h5ad \
  --output-dir <output>/misc-predictions-frozen
```

The optional `derive_misc_complete_case` output is a post-hoc method-support
sensitivity universe. It is never substituted for the preregistered 251-ligand
primary universe.

Audit all registries, local assets, environments, and the expanded execution
matrix:

```bash
.venv/bin/python -m benchmarks.comprehensive.audit \
  --output-dir ../benchmark_work/comprehensive_multicontext_20260722/audit \
  --overwrite
```

Prepare the MIS-C SCE without retaining unnecessary clinical fields:

```bash
Rscript benchmarks/comprehensive/prepare_misc.R \
  --input-rds ../benchmark_comprehensive/Zenodo8010790_MultiNicheNet_Tutorials/sce_subset_misc.rds \
  --olink ../benchmark_comprehensive/Zenodo10908003_MISC_Olink/summary_difference_diorioMIS-C-vs-HC.xlsx \
  --output-dir ../benchmark_work/comprehensive_multicontext_20260722/prepared/misc_olink \
  --python-env .venv --overwrite
```

Export the six current LIANA component arms from one frozen per-sample run:

```bash
.venv/bin/python -m benchmarks.comprehensive.export_liana_components \
  --raw-dir <liana-run>/raw \
  --template <liana-run>/interactions_long.parquet \
  --output-dir <output>/liana-components --overwrite
```

The exporter preserves the fixed universe and all structural status/reason
codes. It never fills an absent score with zero and leaves all between-condition
differential fields empty.

Evaluate a standardized hidden G1 event table:

```bash
.venv/bin/python -m benchmarks.comprehensive.evaluate_g1 \
  --input <frozen-events>.parquet --output-dir <output>/g1-metrics \
  --target-prevalence 0.10 --threshold <preregistered-threshold>
```

`predicted_score` must already be oriented so larger values mean a more
positive condition effect. The MCC threshold is frozen per comparable score
space; methods with different raw scales are evaluated in separate calls or
after a preregistered scale transformation. AP and AUROC remain threshold-free.

Prepare and validate the held-out Olink endpoint:

```bash
Rscript benchmarks/comprehensive/prepare_olink_truth.R \
  --input-xlsx ../benchmark_comprehensive/Zenodo10908003_MISC_Olink/summary_difference_diorioMIS-C-vs-HC.xlsx \
  --source-rmd ../benchmark_work/comprehensive_multicontext_20260722/vendor/multinichenetr/vignettes/add_proteomics_MISC.Rmd \
  --output-dir ../benchmark_work/comprehensive_multicontext_20260722/prepared/olink_truth \
  --overwrite
```

Only after the RNA-only ligand prediction file is frozen, evaluate it with its
literal SHA256:

```bash
.venv/bin/python -m benchmarks.comprehensive.evaluate_olink \
  --predictions <frozen-ligand-predictions>.tsv \
  --predictions-sha256 <sha256> \
  --truth ../benchmark_work/comprehensive_multicontext_20260722/prepared/olink_truth/olink_truth.tsv \
  --truth-manifest ../benchmark_work/comprehensive_multicontext_20260722/prepared/olink_truth/manifest.json \
  --output-dir <output>/olink-evaluation
```

The prediction table must contain exactly `dataset_id`, `method_id`,
`resource_mode`, `universe_id`, `contrast`, `ligand`, `score`, and `status`.
It materializes the complete Olink-blind ligand universe, including explicit
non-observed statuses. All H-common methods must use the same `universe_id` and
ligand rows; the evaluator verifies that the ID is the canonical digest of the
complete ligand set. Metrics are calculated on the intersection of that frozen
universe and measured Olink analytes; the 80% ranking gate is scored coverage
within this representable intersection, not coverage of all 1,463 assay
proteins. Overall assay representability is reported separately. Only canonical
`misc_m_vs_s` H-common values are rankable; native, H-covered, and reverse
values are diagnostic.

The canonical primary contrast is `misc_m_vs_s`. The evaluator-derived reverse
`misc_s_vs_m` is diagnostic only. Unmeasured analytes and non-observed method
statuses remain missing rather than being assigned a neutral or zero score.

## Registry files

- `datasets.tsv`: access, role, design, subject/sample/context/cell counts, and
  local asset state expected for each dataset.
- `methods.tsv`: estimand and statistical scope for all methods.
- `metrics.tsv`: endpoint definitions and aggregation rules.
- `panels.tsv`: preregistered dataset-method-resource matrix.
- `literature.tsv`: DOI, license, local full text, and download status.
- `vendor_sources.tsv`: canonical upstream URL, exact commit, license, and
  retrieval date. Third-party source trees are not committed.
- `execution_profiles.tsv`: one-core, best-practical, and validated-GPU
  resource contracts with a hard memory ceiling below 80%.
- `simulation_scenarios.tsv`: frozen positive, null, confounding, topology,
  missingness, annotation, and resource-perturbation designs.
- `contrasts.tsv`: explicit omnibus, pairwise, one-vs-rest, interaction, and
  legacy estimands with subject/family blocking and method applicability.

The generated audit contains `asset_audit.tsv`, `literature_audit.tsv`,
`vendor_audit.tsv`, `method_audit.tsv`, `execution_matrix.tsv`,
`panel_readiness.tsv`, and `audit_manifest.json` with checksums and hardware
metadata.
