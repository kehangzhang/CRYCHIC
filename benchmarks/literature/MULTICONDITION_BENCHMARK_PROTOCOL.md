# Preregistered multi-condition CCC benchmark protocol

Status: protocol draft frozen before observing the new multi-dataset method
comparison. Retrieval date: 2026-07-12. This protocol refines `../../../benchmark.md`
using the primary literature in `literature_manifest.tsv` and the statistical
contracts in `../../DEVELOPMENT_PLAN.md`.

## 1. Scientific questions and method tracks

There is no scientifically valid single leaderboard because the methods do not
all estimate the same object. Results must be reported by track.

### Track A: ligand-receptor STLR prioritization

Methods: CellChat, CellPhoneDB, LIANA rank aggregate, and CRYCHIC. NicheNet may
enter only where its LR score is a documented native output; it is not a primary
competitor in this track.

Canonical row key:

```text
dataset, subject, context, sender, receiver,
ligand_complex, receptor_complex, resource_arm
```

Primary real-data endpoint: within-subject differential-rank reproducibility,
summarized as a macro average over eligible sender-receiver families. Secondary
endpoints are top-k Jaccard, rank Spearman, literature support, and, when matched
spatial data become available, condition-specific DES.

### Track B: ligand-target/downstream-response recovery

Methods: NicheNet, MultiNicheNet where applicable, CRYCHIC attributed signatures,
and LIANA+ target/context extensions where they expose a comparable target score.

Primary synthetic/perturbation endpoint: signed target-program macro-AUPRC.
Secondary endpoints: target-gene AUPRC/AUROC, top-k target recall, cosine
similarity, direction accuracy, explained variance, and residual calibration.
CellChat and CellPhoneDB are not scored as failed target predictors because they
do not natively solve this task.

### Track C: multi-sample differential communication

Every method first produces a sample-level score or presence indicator within
each `subject x context`. The statistical unit is the biological subject.

- Paired two-condition data: paired sign-flip/permutation or a subject-aware
  model with a condition effect.
- Unpaired groups: subject-block permutation within prespecified strata or a
  subject-level model with declared covariates.
- Three or more groups: one omnibus test followed by preregistered contrasts;
  do not replace the omnibus test with all pairwise comparisons.
- Repeated regions/time points: preserve the subject block and audit design rank.
- Pooled animals or cells bootstrapped from a pool: descriptive only.

CellPhoneDB's cell-label permutation p-value is within-dataset cell-type
specificity, not a condition-difference p-value. A CellChat difference between
two pooled condition objects is descriptive, not subject-level inference.

### Track D: robustness and computational performance

Run biological replicate, subject downsampling, fixed-subject/increasing-cell,
dropout, count noise, 5/10/15% cell-label perturbation, resource replacement,
LR-prior perturbation, composition imbalance, structural absence, and legal
context permutation scenarios. Measure wall time, peak RSS, disk footprint,
success/failure, output determinism, top-k Jaccard and rank Spearman.

## 2. Dataset tiers

The local eligibility decision is frozen in `DATASET_ELIGIBILITY.tsv`.

1. **Primary real paired cohort:** Ji cSCC, 10 patients with normal and tumor
   samples. All 10 patients contain both contexts. Use `level1_celltype` for the
   primary analysis, exclude `Multiplet`, and use level 2 only as sensitivity.
2. **Primary multi-region cohort after estimability audit:** Kuppe MI, 20
   patients, 29 samples. `patient_group` has 11 myogenic, 7 ischemic and 5
   fibrotic patients; repeated cross-group observations occur only for a subset.
   Do not call this a fully paired three-group design.
3. **Planned primary MS cohort:** Lerma-Martin MS. GEO confirms that the 16 local
   10x matrices are snRNA-seq, but they lack cell annotations. The official UCSC
   processed atlas has 103,794 nuclei from 13 subjects/16 samples; its barcode
   overlap with the local raw matrices varies by sample, so annotations must not
   be joined naively. Download the matching UCSC matrix or validate the exact
   QC/alignment transform. Use CA (5 subjects) vs Ctrl (6) as the primary
   comparison. CI has only 2 independent subjects; CI contrasts and the
   three-group result are secondary/descriptive even though there are 4 CI samples.
4. **Demonstration only:** Panc02 treatment because relevant animals were pooled;
   CellChat embryonic skin and CellPhoneDB trophoblast because provided objects
   do not contain independent subject replicates.
5. **Secondary positive controls:** paired Kang IFN-beta and patient-level AD
   skin after reconstructing original sample metadata.

## 3. Common preprocessing contract

1. Freeze input SHA256, species, gene namespace, sample/subject/context keys,
   cell-type hierarchy and exclusion list before method execution.
2. Use the same cells and primary cell-type labels for all methods. Document any
   method-specific cell-count minimum and resulting skipped strata.
3. Preserve raw counts for CRYCHIC count models. Give competitors the normalized
   representation required by their official protocol, derived once from the
   same counts; record the transform rather than forcing an incompatible input.
4. No label-dependent filtering outside the method's training fold. Freeze the
   candidate interaction universe without looking at context labels.
5. Missing cell types remain missing. Never impute zero state expression.
6. Define one sender/receiver ontology map and one complex canonicalization map.
   A complex is eligible only when the method/resource representation can be
   mapped without silently dropping required subunits.
7. Log random seed, threads, hardware, software image, runtime, peak RSS,
   warnings, failures, and every output transformation.

## 4. Fairness arms

Run both arms and never average them into one score.

### H: harmonized resource arm

Use a frozen, license-compatible human or mouse LR universe. The intersection
needed to run all methods is not the only analysis because it preferentially
selects well-known edges. Report both:

- `H-common`: exact mappable common universe, used for like-for-like ranking;
- `H-covered`: frozen union with per-method coverage and missing-resource status,
  used to expose coverage loss without scoring unavailable edges as false.

Resource conversion must preserve ligand/receptor direction, complex subunits,
cofactor semantics, species and original evidence IDs. A method incapable of a
custom resource is `not_supported`, not zero.

### N: native resource arm

Use each official default resource and recommended parameters. This measures the
released method-plus-resource system. It cannot establish that the algorithm,
rather than its database, is superior.

NicheNet's ligand-target prior stays native and version-locked in Track B. It
must not be reduced to the LR-only harmonized universe for the primary target
endpoint.

## 5. Score harmonization

Adapters export the native score, native significance field and direction
unchanged. A separate comparison rank is computed within each
`method x dataset x subject x context` over the frozen eligible universe.

The executable metric-table identity is
`dataset x method x method_version x resource x resource_version x
resource_mode x score_semantics x contrast`. Native `score` and
`score_direction` (`higher` or `lower`) are retained without transformation.
Comparison fields are derived separately: `comparison_rank` is one-best and
uses average rank for ties, while `comparison_strength` is a dimensionless
rank strength in `[0, 1]`. Top-k sets include every edge tied at the kth cutoff,
so their realized size may exceed k.

- Lower-is-better native statistics are reversed only for comparison rank, with
  the transform recorded.
- Ties receive deterministic average rank with canonical ID as final sort key.
- Absence because an interaction was not predicted is distinct from resource
  unavailable, cell type missing, filtered, failed, and not estimable.
- Every sample materializes the same frozen edge universe. `not_predicted` is
  comparison-eligible at zero comparison strength; `missing`,
  `resource_unavailable`, `cell_type_missing`, and `not_estimable` are retained
  as explicit rows but excluded from ranks and effect estimates. Sparse adapter
  output must therefore be outer-joined to the frozen universe and classified;
  an absent row must never be guessed to mean zero.
- An uncalibrated rank is never converted to a p-value or probability.
- CRYCHIC v0.1 `p/q/posterior/comm_probability` fields remain `NA`; it competes
  only on its legitimate exploratory outputs until the G3 gates pass.

## 6. Endpoints and aggregation

### Primary endpoints

1. Track A real cohorts: macro mean of within-subject differential-rank
   reproducibility, first over eligible edge families per subject, then equally
   over subjects and datasets. Report a subject-block bootstrap 95% interval.
2. Track B synthetic/perturbation: signed target-program macro-AUPRC, equally
   weighted across scenario cells, as required by CRYCHIC G4.
3. Track C simulation: type-I error, empirical FDR, 95% coverage and direction
   power under the `DEVELOPMENT_PLAN.md` G3-F thresholds.
4. Matched spatial extension: macro DES over prespecified top 10/20/30/40%
   spatially differential cell pairs, following scSeqCommDiff. Keep each
   threshold visible; no post-hoc best-threshold selection.

### Secondary endpoints

- STLR truth: AUPRC, AUROC, MCC, precision, recall, F1 and top-k precision.
- Differential truth: direction accuracy, effect RMSE, power, replicate
  prevalence and false-discovery proportion.
- Stability: top 30/100/500 Jaccard, weighted rank correlation, leave-one-subject-
  out stability and degradation curves under perturbations.
- Paired differential leave-one-subject-out stability first averages multiple
  samples/regions within each `subject x context x edge`, compares the held-out
  subject's target-minus-reference effect with the mean effect of the remaining
  subjects, and never counts multiple regions as independent subjects. Direction
  agreement excludes edges for which both compared effects are exactly zero.
- Biology: preregistered pathway/edge recovery with evidence tier and sign. It is
  supportive, not a complete gold standard.
- Performance: wall time, peak RSS and disk output at fixed thread caps, plus
  cells/genes/cell-types/LR universe size and failure status.

### Multiplicity and uncertainty

Do not pool hypotheses across incompatible tracks. Synthetic G3 calibration uses
at least 1,000 null replicates and the exact one-sided Monte Carlo bounds frozen
in the development plan. Real-data method differences use paired subject-block
bootstrap intervals; label any exploratory nominal p-values explicitly. No
winner is declared when simultaneous uncertainty includes the preregistered
noninferiority margin or a method has insufficient dataset coverage.

## 7. Negative controls

Required before interpreting real biology:

- legal full-pipeline context permutation;
- fixed subjects with increasing cells (must not create apparent sample size);
- abundance-only changes: ecosystem may change, state should remain stable;
- receiver-autonomous, ligand-only and target-only scenarios;
- receptor or required subunit knockout;
- degree/evidence-matched target-prior shuffle;
- context-correlated sampling missingness;
- partial/complete batch-context confounding;
- topology jump, wrong topology and disconnected graph;
- 5/10/15% annotation perturbation and resource replacement.

Cell-level context permutation is prohibited. A permutation that only relabels
already-computed scores is not a full-pipeline CRYCHIC calibration test.

## 8. Iteration rule

The benchmark may identify defects, but the test set must not become a tuning
set. Each cycle follows:

1. freeze a run manifest and execute all methods;
2. classify failures as input-contract, adapter, statistical, algorithmic,
   resource-coverage or performance failures;
3. propose one versioned change with an expected endpoint and regression tests;
4. tune only on simulations or a designated development cohort;
5. rerun negative controls and the full benchmark with a new run ID;
6. evaluate the untouched holdout cohort once per release candidate;
7. retain every prior compact metric table and report regressions, not only the
   best iteration.

Suggested split for the present local data: Kang/AD and simulations for adapter
development; Ji cSCC as the first real development cohort; Kuppe as an external
multi-region validation after contrast audit; Lerma MS as a locked holdout once
the exactly aligned annotation/expression object and spatial evidence are acquired.
Do not tune CRYCHIC directly to
maximize agreement with CellChat, CellPhoneDB, LIANA or NicheNet; competitor
agreement is not ground truth.

## 9. Stop/go decisions

- Do not enable formal CRYCHIC p/q values until G3-F passes.
- Do not enable communication probability until the separate G3-P gate passes.
- Keep graph fusion opt-in unless the G2 macro-AUPRC improvement and
  wrong-topology noninferiority margins pass.
- Keep multi-view opt-in unless G4 target-program recovery improves without
  degrading G3-F calibration.
- A real-cohort biological result is reportable only after the corresponding
  negative controls, sample-support audit and sensitivity to resource arm are
  shown beside it.

## 10. Required output tables

Each iteration writes compact versioned tables for dataset manifest, environment
manifest, method run status, resource coverage, native and canonicalized scores,
sample-level differential effects, spatial DES, truth metrics, robustness,
runtime/RSS, known-biology evidence, and change-versus-previous-iteration. Raw
method output and downloaded data stay outside Git; code, configuration,
checksums and compact metrics are tracked.

## 11. Evidence basis and cautions

- Dimitrov et al. (2022), DOI `10.1038/s41467-022-30755-0`: separate method and
  resource effects; use complementary cytokine, CITE-seq and spatial evidence.
- Liu et al. (2022), DOI `10.1186/s13059-022-02783-y`: spatial DES is useful but
  proximity is an indirect proxy, not molecular truth.
- ESICCC (2023), DOI `10.1101/gr.278001.123`: separate LR and L/R-target tasks;
  include perturbation recovery, stability and computational burden.
- RobustCCC (2023), DOI `10.3389/fgene.2023.1236956`: report robustness curves,
  not one perturbation-averaged rank.
- scSeqCommDiff (2025), DOI `10.1093/nargab/lqaf084`: condition-specific DES on
  Kuppe/MS, harmonized ConnectomeDB2020 and top 10/20/30/40% sensitivity.
- dominoSignal (2026), DOI `10.1093/bioinformatics/btag089`: run CCC by sample;
  its Fisher test applies to binary linkage prevalence. Its proposed 15 samples
  and 150 cells are simulation-specific guidance, not universal eligibility.
- Kim et al. (2026), DOI `10.1186/s13059-026-04063-5`: latest peer-reviewed
  spatial benchmark, including simulations, robustness and empirical
  scalability. It supersedes a preprint-only view of the 2026 literature.

Known biological edges and literature-curated disease networks are incomplete
and publication-biased. Spatial co-localization cannot validate endocrine
signaling and does not prove signaling. Tool agreement measures consensus, not
truth. These limitations must remain visible in every report.
