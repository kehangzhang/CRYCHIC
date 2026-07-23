# `suggest_v5.md` 第二、第三部分 benchmark 落实审计

审计日期：2026-07-24

状态：**文档所列 benchmark 尚未全部落实。**

## 中文摘要

`suggest_v5.md` 由多轮调研拼接而成，存在两套重复编号的“第二、第三
部分”。本报告同时核验两套范围，结论如下：

- 文献 benchmark 目录中的四项任务没有任何一项完整完成：
  Tensor-cell2cell 和 DCST 有部分运行结果；STACCato 与 scACCorDiON
  只有源码、数据或协议准备，没有完成指定 benchmark。
- Tensor-cell2cell 已在 Kang 真实队列跑通，但并非文档要求的
  12-context 植入真值模拟，不能替代原 benchmark。
- DCST 已完成 `n=15`、每种细胞 500 个的单一切片及 25 个 seed，覆盖
  110 个要求参数组合中的 1 个；没有形成样本数×细胞数功效曲面。
- 后半部分的 signed-estimand 合约已在行为层面落实：15/15 手算案例、
  11/11 invariance 检查通过，三组 detection/localization/direction/effect
  指标已拆分实现；历史结果 manifest 仍有 dirty provenance，需要干净重跑。
- score-generator×differential-engine 只完成 CRYCHIC 内部 7×2 子矩阵，
  尚未把 LIANA、CellChat、CellPhoneDB 分数接入 GLM、CR2、STACCato、
  common-functional OOF 和 scSeqCommDiff 等统一引擎。

因此不能写成“第二、第三部分均已落实”。准确表述应为：**一项行为层面
完成、三项部分落实、两项仅完成准备工作。**

## Scope and interpretation

`../suggest_v5.md` is a 5,448-line transcript containing two separately
numbered pairs of sections II and III. This audit covers both, so that the
answer does not depend on an ambiguous section reference.

1. Primary catalog scope, lines 1665-2330:
   - Section II, known-truth multi-context simulations: Tensor-cell2cell,
     STACCato, and DCST/dominoSignal.
   - Section III, patient graph/hypergraph phenotype recovery:
     scACCorDiON across seven cohorts.
2. Later optimization scope, lines 4099-4382:
   - Section II, signed-estimand contracts and decomposed three-group metrics.
   - Section III, score-generator by differential-engine crossover.

The source-code baseline for this audit is branch
`optimize/suggest-v6-bounded-evidence-20260724` at commit `6c3bc72` before
this report was added. Workspace result artifacts through 2026-07-24 were
also inspected.

The status labels mean:

- **Complete**: required input, executable versioned adapter, required method
  matrix, required metric family, successful result, and provenance manifest
  are all present.
- **Partial**: a real subset or related surrogate was executed, but one or
  more required axes, methods, metrics, or provenance requirements are absent.
- **Preparation only**: literature/source/data registration exists, but the
  specified benchmark did not produce a result.
- **NE**: the endpoint is not statistically estimable from the available
  design; NE is never treated as zero.

## Executive conclusion

### Primary catalog sections

| Section | Benchmark | Status | Required benchmark completed? | Main evidence |
|---|---|---|---|---|
| II.1 | Tensor-cell2cell 12-context planted programs | Partial | **No** | Cell2cell environment and a Kang real-data tensor run exist; the planted 12-context truth asset and truth metrics are absent |
| II.2 | STACCato condition/batch simulation | Preparation only | **No** | Source is pinned, but exact simulation input, environment, adapter, and results are absent |
| II.3 | DCST subjects-by-cells power simulation | Partial | **No** | One `n=15`, 500-cells/type slice was run for 25 seeds; the full 11 by 10 grid was not run |
| III.4 | scACCorDiON seven-cohort phenotype recovery | Preparation only | **No** | PDAC graph input is present, but no adapter, clustering sweep, ARI/RI, robustness, or seven-cohort result exists |

Therefore, among the four named benchmarks in the primary sections:

- complete: **0/4**;
- partially executed: **2/4**;
- preparation only: **2/4**.

The result is not “all implemented.” Source download, method smoke tests, and
registry entries must not be counted as completion of the requested benchmark.

### Later duplicated sections

| Section | Benchmark | Status | Main conclusion |
|---|---|---|---|
| II | Signed-estimand contract and decomposed three-group metrics | Complete behaviorally; provenance caveat | 15/15 hand cases and 11/11 invariants pass; decomposed three-group metrics are implemented |
| III | Score-generator by differential-engine crossover | Partial | A restricted CRYCHIC-only 7 by 2 result exists; the proposed external generators and five differential engines are not crossed |

## Evidence hierarchy and registry discrepancy

The 2026-07-22 comprehensive freeze correctly records the original state:

- `P03_external_dynamic_program`: readiness `0.000`, gate `not_met`;
- `P13_patient_graph_phenotype`: readiness `0.000`, gate `not_met`;
- Tensor-cell2cell, STACCato, DCST, and all four scACCorDiON method rows are
  marked `planned` in `benchmarks/comprehensive/methods.tsv`.

Post-freeze workspace runs on 2026-07-23 added two partial results that are not
reflected in that old readiness table:

1. Tensor-cell2cell on the real Kang paired IFN-beta cohort.
2. A fixed DCST simulation slice with 25 seeded realizations.

Those runs improve implementation coverage, but do not satisfy the original
P03 or full DCST specifications. Their reusable runners are stored under
`benchmark_work/` and checksum-bound by manifests, but are not tracked in the
latest Git tree. The expanded campaign root also records
`dirty_at_execution=true` at commit `67ac7ff`. These are usable diagnostic
artifacts, not clean release-grade reproductions.

## Primary Section II.1: Tensor-cell2cell planted dynamic programs

### Required by `suggest_v5.md`

The requested benchmark is the published planted simulation with:

- 3 cell types;
- 300 LR pairs;
- 12 contexts;
- four planted dynamics: oscillation, pulse, exponential decay, and linear
  decrease;
- planted LR, sender, receiver, and full-event identities;
- Tensor-cell2cell, LIANA-Tensor-cell2cell, and CRYCHIC outputs on a common
  contract.

The requested original and extended endpoints are:

| Endpoint family | Required endpoints |
|---|---|
| Original factor recovery | LR-set Jaccard, loading Pearson correlation, NRE, CorrIndex, noise robustness |
| Context recovery | Hungarian matching, Pearson, Spearman, RMSE, DTW, peak-context error |
| Event recovery | AUPRC, AUROC, MCC, precision/recall at K, nDCG, signed support |
| Axis recovery | sender/receiver top-1 accuracy and loading Jaccard |
| Full event/hypergraph | event AUPRC/MCC/F1, exact and partial hyperedge recovery, direction recovery |

### What is implemented

- The upstream `CCC-Benchmark` source is pinned at
  `598ba07e10f73bfb4cbd0b92d27b8c42a66cdee9`.
- An isolated Cell2cell 0.5.1 environment exists and passed import and toy
  factorization checks.
- The metric registry defines factor context correlation, factor-event AUPRC,
  normalized reconstruction error, held-out-context error, and generic top-K
  Jaccard contracts.
- A full real-data Kang track completed on:
  - 24,673 cells;
  - 8 paired donors and 16 donor-condition contexts;
  - a `16 x 455 x 8 x 8` tensor;
  - fixed rank 7;
  - observed-mask NRE `0.2357597`;
  - 26.34 seconds wall time and 0.511 GiB sampled process-tree peak RSS.
- The Kang run exported all four factor axes and paired stimulation
  associations. Factor 6 had paired-t BH `q=6.47e-9`.

### What is not implemented

- The exact published 3-cell-type, 300-LR, 12-context planted tensor is absent.
  The local vendor checkout contains PBMC/BALF timing code but not the original
  simulation payload; the Code Ocean capsule returned HTTP 403 during the
  recorded audit.
- No planted-program truth table was normalized to the common contract.
- No CRYCHIC versus Tensor-cell2cell versus LIANA-tensor head-to-head run exists
  on that truth.
- No planted LR Jaccard, loading-truth Pearson, CorrIndex, noise sweep,
  Hungarian truth matching, event AUPRC/AUROC/MCC, sender/receiver accuracy, or
  hyperedge recovery result exists.
- The Kang result is a real-data context association with no event truth,
  held-out validation, or known dynamic program; it cannot substitute for the
  requested known-truth benchmark.
- The Kang runner is checksum-bound in the raw artifact but is not part of the
  current versioned source tree.

### Verdict

**Partial.** Method execution and NRE are demonstrated on Kang, but the core
known-truth Tensor-cell2cell benchmark is not implemented.

## Primary Section II.2: STACCato condition/batch simulation

### Required by `suggest_v5.md`

The requested design includes approximately 60 samples, 300 LR pairs, two
senders, two receivers, multiple noise levels, and three condition-batch
designs:

- balanced: `15/15` samples in each condition-by-batch cell;
- partial confounding: `25/5` versus `5/25`;
- extreme confounding: `29/1` versus `1/29`.

Required endpoints include condition-effect MSE, batch-effect MSE, error SD,
sign accuracy, top-K effect recovery, type-I error at 0.01/0.05/0.10,
empirical FDR, CI coverage and width, power by effect size/sample size,
complete-confounding failure rate, runtime, and memory.

### What is implemented

- The STACCato source is pinned at
  `cb76f210198aa3a72da4df8122ff950bd3277e9e`.
- The dataset, method, panel, and general G1 metric contracts are registered.
- General metric code exists for effect RMSE, effect Spearman, sign accuracy,
  type-I error at 0.05, empirical FDR at 0.05, and 95% CI coverage.

### What is not implemented

- The pinned repository contains ASD/SLE tutorials but not the frozen
  simulation payload required by the paper benchmark.
- The external Dropbox input and the required `tensorregress`, `R.matlab`, and
  `rTensor` environment are absent.
- `benchmarks/comprehensive/methods.tsv` still identifies both the adapter and
  environment as `planned`.
- No balanced, partial-confounding, or extreme-confounding run exists.
- No STACCato output is present in the result artifacts.
- Batch-effect MSE, error SD, top-K effect recovery, the three-alpha type-I
  panel, CI width, power curves, and confounding failure-rate endpoints are not
  implemented as a completed STACCato report.
- STACCato was not used as a swappable differential engine in the component
  crossover benchmark.

### Verdict

**Preparation only.** Literature and source pinning are complete; the actual
STACCato benchmark has not been run.

## Primary Section II.3: DCST/dominoSignal power grid

### Required by `suggest_v5.md`

The paper-style grid requires:

- subjects per group: `5, 10, ..., 55` (11 values);
- B-cell count: `50, 100, ..., 500` (10 values);
- 10 independent initializations per parameter setting;
- two differential links and one null link;
- paper binary-link thresholding, Fisher exact test, and receiver-family BH;
- sensitivity and specificity surfaces across both axes.

This is 110 parameter settings and 1,100 requested seeded settings before any
additional method repeats. The proposed CRYCHIC extension also requests a
pre-frozen binary threshold plus continuous-score AUPRC, AUROC, effect RMSE,
sign accuracy, empirical FDR, and power.

### What is implemented

- The upstream source is pinned at
  `e37265adf2de414d9c1c8129b839f67dbb3e214d`.
- The published binary-expression parameter table and linkage/Fisher functions
  are locally available.
- A protocol-compatible local reimplementation completed one parameter slice:
  - `n=15` pseudo-samples per group;
  - 500 cells per cell type per pseudo-sample;
  - 30,000 cells per realization;
  - 25 independent seeds (`123-147`);
  - DCST reimplementation and a generic CRYCHIC synthetic-prior baseline.
- All 25/25 realizations completed. Manifest verification passed for 1,086
  bound artifacts.
- Fixed-slice results:

| Method | Adj. AP | AUPRC | AUROC | Direction | Raw-p type-I 0.05 | Raw-p false-discovery fraction |
|---|---:|---:|---:|---:|---:|---:|
| DCST protocol reimplementation | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| CRYCHIC generic synthetic prior | 0.100 | 0.250 | 0.310 | 0.000 | NE | NE |

The CRYCHIC AUROC 95% t interval across seeds was `[0.281, 0.339]`. No run
failed and the 80% memory limit was not approached.

The two last columns use unadjusted native `p < 0.05` in the local evaluator,
despite the protocol also exporting BH q-values. They are diagnostics and are
not a completed BH-calibration benchmark.

### What is not implemented

- Only `1/110` parameter settings, approximately 0.9% of the requested grid,
  was evaluated. The sample-size and cell-count power surfaces do not exist.
- The run used 25 seeds on one slice rather than 10 seeds at every grid point.
- It is a protocol reimplementation, not execution of the released
  `dominoSignal` package.
- The CRYCHIC arm is explicitly a generic synthetic-prior baseline, not RC9,
  RC12, or the latest production candidate.
- Paper sensitivity/specificity curves and receiver-family BH power curves are
  not in the final tables. The report instead emphasizes event AUPRC/AUROC,
  direction, and raw-p diagnostics on eight events.
- Effect RMSE, a method-common pre-frozen binary threshold study, mixed-model
  extension, and full power curves are absent.
- Each algorithm-facing subject is a pooled-cell bootstrap pseudo-sample, not
  an independent biological donor. The run cannot establish donor-level
  calibration.
- The two runner scripts are checksum-bound under `benchmark_work/`, but are
  not version-controlled in the current Git branch.

### Verdict

**Partial.** A useful and verified fixed-slice failure-mode benchmark exists,
but the full DCST benchmark requested by `suggest_v5.md` is not implemented.

## Primary Section III.4: scACCorDiON patient graph/hypergraph benchmark

### Required by `suggest_v5.md`

The requested benchmark contains seven cohorts:

1. pancreatic adenocarcinoma, 35 samples;
2. COVID-19, 130 samples;
3. myocardial infarction, 23 samples;
4. breast cancer, 126 samples;
5. kidney AKI, 36 samples;
6. lung atlas, 165 samples;
7. renal clear-cell carcinoma, 17 samples.

For common frozen patient graphs it requires:

- canonical event-vector and native hypergraph distances;
- directed weighted OT, correlation OT, GOT, and tabular baselines;
- k-medoids, k-barycenter, and Leiden clustering;
- `k=2..7` and Leiden resolution `0..1` by 0.01;
- ARI and Rand index at the known label count, maximum ARI, and
  Friedman-Nemenyi comparison;
- PDAC cell-retention fractions `1.0, 0.75, 0.50, 0.375, 0.25`, five repeats;
- MI/RCC coarse-versus-fine annotation analysis;
- optional PDAC survival validation with stage adjustment.

### What is implemented

- The upstream source is pinned at
  `6f5c1b612b52867743ed5165c347b5714e710d75`.
- The exact upstream PDAC graph input is local:
  `Peng_PDAC_metaacc.h5ad`, SHA256
  `0fb2f946d4eba66c0f950f280d5fb184129969ced19554c339a8b2f13aaf69e7`.
- The registry marks PDAC `ready`, COVID and MI `partial`, and the remaining
  four cohorts `planned`.
- Metric contracts are registered for phenotype ARI, Rand index, and
  maximum-over-K ARI.
- The source checkout contains the authors' distance and clustering code.

### What is not implemented

- All four scACCorDiON method adapters remain `planned`.
- The frozen P13 matrix has 28 method-dataset pairs, zero eligible/reusable
  pairs, readiness `0.000`, and gate `not_met`.
- No non-vendor scACCorDiON result directory or patient-distance matrix exists.
- No seven-cohort ARI/RI result, Friedman-Nemenyi test, clustering sweep, or
  native hypergraph distance comparison exists.
- COVID and MI are not frozen to paper-exact graph inputs; breast cancer,
  kidney AKI, lung, and RCC are not downloaded in paper-exact form.
- PDAC downsampling, MI/RCC annotation granularity, distance stability,
  differential-hyperedge overlap, and survival validation have not been run.

### Verdict

**Preparation only.** One exact graph input and the source package are present,
but the Section III benchmark itself has not produced results.

## Later Section II: signed-estimand and decomposed three-group benchmark

### Deterministic contrast contracts

The following requested cases are implemented through public design and OOF
inference APIs:

- two-group forward and reverse contrasts;
- all three pairwise contrasts for three groups;
- balanced one-versus-rest;
- positive and negative 2 by 2 DID;
- three-group omnibus equality;
- context-chain local edges, local neighborhoods, global one-versus-rest, and
  change-point localization.

The stored result reports 15/15 hand-computable cases and 11/11 invariants
passing. Invariants cover reverse contrast, row order, context relabeling,
factor argument order, positive scaling, sender/receiver relabeling, missing
scores, and structural cell-type absence.

The current targeted regression suite for signed contracts, decomposed
three-group evaluation, component crossover, external panel execution, and
same-seed comparison passed **26 tests** during this audit.

### Three-group metric decomposition

`benchmarks/comprehensive/evaluate_three_group.py` implements:

- omnibus prevalence-adjusted AP, native AUPRC, AUROC, MCC, precision@K, and
  recall@K;
- localization macro/micro AUPRC, Hamming loss, exact contrast-set accuracy,
  and per-contrast confusion rows;
- positive/negative direction AP and both all-active and detected-active
  direction accuracy;
- RMSE, MAE, Pearson, Spearman, sign concordance, and top-effect recovery;
- explicit event and contrast-cell coverage;
- CI/formal-calibration NE states when comparable full-pipeline uncertainty is
  unavailable.

The latest checksum-bound same-seed panel contains 20 active and 20 matched
global-null seeds with CRYCHIC, scSeqCommDiff, CellChat, and LIANA. It is an
actual execution result, not only a unit-test contract.

### Provenance limitation

The historical signed-contract manifest is complete and checksum-bound, but it
records `dirty=true` at commit `67ac7ff`. The behavior is covered by clean
versioned code and current tests, but a clean rerun should replace that
historical manifest before a formal release claim.

### Verdict

**Complete behaviorally, with a release-provenance caveat.** Formal p/q values
remain intentionally NE until full-pipeline resampling is available; that is a
declared statistical boundary, not a silently missing value.

## Later Section III: score-generator by differential-engine crossover

### Required by `suggest_v5.md`

The proposed score generators are:

1. CRYCHIC native;
2. CRYCHIC availability-only;
3. availability plus prior;
4. CRYCHIC plus sender;
5. CRYCHIC plus downstream;
6. LIANA magnitude;
7. CellChat probability;
8. CellPhoneDB mean;
9. a simple mean/product baseline.

The proposed differential engines are:

1. common sample-level GLM or paired model;
2. common clustered/CR2 model;
3. STACCato;
4. CRYCHIC native common-functional OOF inference;
5. scSeqCommDiff multi-sample for supported two-group settings.

### What is implemented

A clean 20-seed result evaluates a restricted internal matrix of seven
CRYCHIC-derived score layers with two simple effect engines:

- score layers: strict geometric, sender/downstream 90:10 blend,
  availability-only, mechanistic geometric, availability/downstream geometric,
  downstream-only, and sender-only;
- engines: native raw subject mean and within-sample-rank subject mean;
- result size: 14 arms;
- frozen candidate: sender/downstream 90:10 with native raw mean;
- candidate AUPRC `0.1350`, AUROC `0.5417`, localization AP `0.1195`;
- candidate gate: accepted against the strict-geometric rank reference.

Subsequent RC12 work validated a related sender-response head on fresh seeds
and ran external methods on the same fixtures. That establishes a fair
same-seed method comparison, but it is not a generator-by-engine crossover.

### What is not implemented

- No LIANA magnitude, CellChat probability, or CellPhoneDB mean tensor is fed
  through each common differential engine.
- There is no exact availability-plus-prior-only arm corresponding to the
  proposed list.
- The crossover does not include a common GLM, common CR2, STACCato,
  common-functional OOF refit, or scSeqCommDiff engine.
- The two implemented engines are fixed subject-mean transforms; the report
  explicitly states that no model is refit.
- External methods in the 20-seed panel use their own adapters and native
  semantics. Their scores are not swapped through the CRYCHIC engines.
- The current source defines an eighth mechanism-guarded layer, but the cited
  clean result artifact predates that addition and contains seven layers.

### Verdict

**Partial.** The internal CRYCHIC ablation is useful and complete for its
restricted 7 by 2 matrix, but the diagnostic crossover proposed in Section III
has not been implemented.

## Completion gaps by priority

To truthfully change the four primary catalog rows to complete, the minimum
remaining work is:

1. Acquire or reconstruct under a preregistered equivalence contract the exact
   Tensor-cell2cell 12-context planted truth; version the adapter and run all
   factor/event/axis metrics for CRYCHIC, Tensor-cell2cell, and LIANA-tensor.
2. Version the DCST runners, add the latest frozen CRYCHIC head, execute all
   110 parameter settings with prespecified seeds, and publish sensitivity,
   specificity, power, FDR, runtime, and failure surfaces.
3. Build the STACCato R environment and adapter, recover the exact simulation
   input, then run all three confounding designs and calibration endpoints.
4. Implement the scACCorDiON patient-graph adapter and clustering sweep, first
   on the available PDAC graph and then on the remaining six paper-exact
   cohorts; add robustness and survival tracks separately.
5. Complete the external score-generator by common-engine matrix rather than
   treating the existing same-seed native-method panel as a component swap.
6. Rerun the signed-estimand publication once from a clean worktree to remove
   the remaining provenance caveat.

## Checksum-bound evidence index

Paths below are workspace-relative; large raw outputs remain outside Git.

| Evidence | Path | SHA256 or commit |
|---|---|---|
| Comprehensive source freeze | Git commit | `566710a` |
| P03/P13 readiness | `benchmarks/results/comprehensive_multicontext_20260722/panel_readiness.tsv` | versioned at audit baseline |
| Tensor-cell2cell source | `benchmark_work/comprehensive_multicontext_20260722/vendor/CCC-Benchmark` | commit `598ba07e10f73bfb4cbd0b92d27b8c42a66cdee9` |
| Kang Tensor-cell2cell track | `benchmark_work/comprehensive_multicontext_20260722/runs/expanded_full_20260723/context_tracks/tensor_cell2cell_kang_full/manifest.json` | `f6880c48be8ec277afd084caafcb9067c64cc5216b79f82e4384cf867f3203b3` |
| Kang Tensor analysis | `benchmark_work/comprehensive_multicontext_20260722/runs/expanded_full_20260723/context_tracks/tensor_cell2cell_kang_full/outputs/analysis_manifest.json` | `bace4cec60bdc67e2a03d29fec3ef3468b934b853a599021bd39efce0e1ff54d` |
| STACCato source | `benchmark_work/comprehensive_multicontext_20260722/vendor/STACCato` | commit `cb76f210198aa3a72da4df8122ff950bd3277e9e` |
| DCST source | `benchmark_work/comprehensive_multicontext_20260722/vendor/Differential_Cell_Signaling_Test` | commit `e37265adf2de414d9c1c8129b839f67dbb3e214d` |
| DCST 25-seed fixed slice | `benchmark_work/comprehensive_multicontext_20260722/runs/expanded_full_20260723/simulation_tracks/dcst_n15_full_25rep/manifest.json` | `03d2501512fa09a928874b36c73f60e005001d14f55007fa3b77a781a636f49b` |
| scACCorDiON source | `benchmark_work/comprehensive_multicontext_20260722/vendor/scACCorDiON` | commit `6f5c1b612b52867743ed5165c347b5714e710d75` |
| Signed-estimand result | `benchmark_work/suggest_v5_iterations/iteration_01_signed_contract_20260723/manifest.json` | `68e2b2889f98c7bd4fce3d4707a34945eec583bbbe5dcce855209bada4798e9b` |
| Clean 20-seed internal crossover | `benchmark_work/suggest_v5_iterations/iteration_03_three_group_validation_20seed/frozen_candidate_validation_gated/manifest.json` | `079a2bd8cb8f46202b952a7aef63aaf95ebc15097888ccb75db5cafb08f617df` |
| Same-seed external comparison | `benchmark_work/suggest_v6_iterations/rc12_external_comparison_9155f5a_20260724/manifest.json` | `c62bc35ba5c43442629e3424500e5b3c635ac84a73e281e73e4cd8b9112e5f65` |

## Final answer

The primary Section II and III literature benchmarks are **not all
implemented**. Tensor-cell2cell and DCST have meaningful partial executions;
STACCato and scACCorDiON remain preparation-only. The later signed-estimand
Section II is implemented behaviorally, while the later score-generator by
differential-engine Section III remains a restricted internal ablation rather
than the requested full crossover.
