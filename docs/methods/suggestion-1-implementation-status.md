# Suggestion 1 implementation status

- Audit date: 2026-07-13
- Starting audit baseline: `5e8b4b8`; the current implementation batch builds
  on `d1b1e1f` and the multi-condition v02 report
- Scope: implementation evidence, not a biological or superiority claim

This matrix records which recommendations are present in the public workflow,
which exist only as candidate primitives, and which remain absent. A primitive
is not counted as complete when the end-to-end workflow does not invoke it.

| Requirement | Status | Evidence and remaining work |
| --- | --- | --- |
| Four semantic outputs | Partial | Workflow contracts separate components, but public integrated scoring remains the legacy baseline and incremental downstream is `not_estimable`. |
| Explained-share v3 | Partial | Implemented and tested as a candidate support method; the baseline default remains relative-coefficient v1. |
| Held-out incremental downstream gain | Partial | Fit/apply primitives and mechanism tests exist, but the public stage-level cross-fit orchestrator does not invoke them. The primitive still lacks canonical sample-subject-context row provenance, subject-blocked inner tuning, subject-grain held-out output and an autonomous nuisance basis; an overlapping generic-program/LR-target counterexample can therefore produce a near-one false gain. |
| Soft-min integration | Partial | Candidate integration and ablation tests exist; the public baseline still uses legacy four-component scoring. |
| Hard receptor eligibility | Partial | Public partial cross-fit freezes a condition-blind, scale-invariant hard gate in each training fold and performs no held-out gate fitting. It remains absent from the legacy default integrated score and lacks a receiver-stage exact-coverage audit. |
| Winsorized normalized precision | Partial | The transform and attribution integration exist, but independent audit found that `precision_transform_id` does not hash the raw/transformed feature-aligned values and the result is not producer-owned. This identity contract must be repaired before public cross-fit integration. |
| Strict complete-link families | Implemented | Chaining counterexamples are covered by family tests. The deterministic incremental complete-link implementation matches a brute-force reference across random sparse profiles, zero-similarity and tolerance cases, and reuses one static partition per fold feature universe while retaining receiver-specific gates and artifact IDs. |
| True family-first fitting | Partial | Public partial cross-fit freezes hard eligibility, strict complete-link families and one medoid basis column per family for every planned training cell type. Separate attribution primitives preserve signed residuals and evidence-only member allocation, but family attribution/tuning and integrated scoring are not connected and the legacy baseline still fits members before aggregation. |
| Complete scoring manifest | Implemented | Every tracked candidate artifact participates in the model/function identity. |
| Receiver-child scoring collection provenance | Partial | Core `ScoringCollectionManifest` contracts and a versioned result extension bind each contrast/repeat/fold to emitted receiver children, scoring functional IDs, source row counts and source-key digests. Persistence validates exact emitted `sample_scores` coverage and receiver/contrast edge partitions, while fixing `composition_status=partial_emitted_only` and `common_functional_across_receivers=false`. No planned receiver universe or authoritative persisted child model/version/universe registry exists yet. |
| State/ecosystem eligibility separation | Implemented | State rows no longer depend on abundance eligibility. |
| All-missing sender evidence | Implemented | Returns missing values rather than fabricated uniform weights. |
| Common sender functional | Partial | The public subject-cross-fit path now freezes the sender universe, cross-context training prevalence prior, minimum support and temperature inside each training fold, applies only sample-local held-out ligand evidence, and verifies exact stage-level OOF coverage. Low-level rows retain `partial_not_oof`, the aggregate result is `verified_train_only_oof_partial_pipeline`, and the functional is deliberately not connected to the legacy context-specific default score. |
| Subject fold planning and OOF audit | Implemented | Subject blocks, K fallback, design re-audit, and exact coverage are tested. |
| Train/apply/cross-fit public workflow | Partial | Public `run_subject_crossfit` derives folds from raw metadata, creates physical sanitized scopes, and freezes interaction/sender, a complete-formula EMM design reparameterization, condition-blind receptor, strict family and training-reference receiver-program artifacts. Explicit categorical registries prevent numeric batch codes from being treated as continuous. Held-out poison cannot change training IDs; overlap, feature drift, unseen nuisance levels and incomplete planned receiver coverage fail closed. Exact OOF coverage is still audited only for availability/common-sender rows; response precision, family attribution/tuning, incremental downstream and common scoring remain. |
| Exact-zero complex soft-min | Implemented | Zero and missingness behavior are covered by tests. |
| Train-only frozen interaction universe | Implemented | Fit/apply identity, reversed-test perturbation, resource provenance, and cap conflicts are tested. |
| H-common data-driven cap ban | Implemented | v0.2 specs reject caps and the cSCC/MS H-common real-data runs completed without a cap. The harmonized resource has 638 interactions, below the former cap of 800, so this run validates the uncapped path but does not demonstrate coverage expansion from removing the cap. |
| Coverage-risk hierarchy | Partial | Nine-level candidate ledger and monotone threshold contract exist; no real threshold campaign has run. |
| G1.5 multi-edge, multi-seed truth | Implemented | 50-seed development and 200-seed holdout both pass all frozen gates. |
| G1.5 through public workflow | Missing | The simulation invokes candidate primitives directly, not the public cross-fit path. |
| RBO, weighted Kendall, top-k curve, rank interval, stable tier | Partial | Frozen-universe APIs and the v02 finalizer/report run are complete with fixed `p=0.9`, Kendall power `1`, 200 split repeats, 2,000 subject bootstraps, 95% intervals, 0.80 top-k threshold, and seed `20260712`. All frozen members and receiver strata are required in every repeat. The real-data agreement rows are consequently all `NE`, rather than being inflated by shared-item intersection. A true LR equivalence/driver-family identifier remains absent, so `lr_family` is also explicit `NE`. |
| cSCC influence diagnostics | Partial | Paired LOSO, bootstrap, rank intervals, availability frequency, and stable-tier tables now exist. Strict frozen-member propagation makes the new rank endpoints `NE`; explicit per-edge missingness trajectories and family identifiers remain absent. |
| Persisted edge evidence | Partial | Optional versioned Parquet persistence, manifest linkage, semantic cross-table validation, lazy reads, and backward compatibility are implemented. Real cSCC opposite-edge root-cause analysis remains. |
| Signed/reverse response | Partial | Directional semantics are implemented but not connected to the workflow. |
| Track-B signed macro-AUPRC | Partial | A strict frozen-universe evaluator now expands every program into forward/reverse activation-compatible channels, penalizes wrong-direction scores, uses tie-aware AP and equal registered scenario-cell weighting, and propagates incomplete seeds/channels as `not_estimable`. No frozen multi-seed campaign or native NicheNet signed result has run, and the existing proxy cannot be relabeled as native. |
| D-common batch/effect model | Partial | A shared, frozen subject-level OLS primitive now applies one formula, categorical level registry and model ID across methods; it adjusts unpaired and paired-difference effects for declared batch covariates, detects target/batch confounding, and fails closed for mixed paired/unpaired designs. It has not yet run on the MS benchmark. |
| Multi-node repeated-measures backend | Partial | A frozen categorical OLS backend retains mixed paired/unpaired subjects, averages technical replicates within subject/context/adjustment cells, and reports exploratory effects with subject-cluster CR1 diagnostic SE only. A backed-`obs` audit of the real 29-sample/20-subject Kuppe design found 10/10 region contrasts estimable only under complete method-score coverage; no effect was fitted. Three condition contrasts adjusted for region were rank deficient, no auditable batch field exists, and finalizer integration remains. |
| Sparse/block/streaming core | Partial | Pseudobulk counts/detection use one group-indicator sparse multiplication. The finalizer scans repeated identity metadata in Arrow batches and reads included `run_id` views using Parquet predicates/projection. On the 92.83 MB cSCC input, metadata scanning fell from 29.94 s to 7.95 s and one-view memory from about 15.89 GB to 5.78 GB; one selected view still materializes and Track B retains its full-column contract. Availability/scoring long tables remain. |
| G2 graph fusion | Missing | No graph-fused attribution solver is present. |
| G3-F/G3-P inference | Missing | Correctly disabled; p/q and communication probabilities remain unavailable. |
| Default switch | Not eligible | Requires certified public cross-fit, real no-cap candidate noninferiority, and legacy reproducibility gates. |

## Frozen evidence

The G1.5 development run contains 1,050 observations and the independent
holdout contains 4,200 observations. Both have zero failed gates. The holdout
equal-edge active-minus-ligand-only margin is `0.341761` with 95% interval
`[0.335538, 0.347985]`. This supports candidate mechanism specificity only;
it does not establish real-data superiority.

The v02 no-cap real-data campaign completed for cSCC and MS. CRYCHIC comparison
coverage is `0.3118` and `0.5150`, respectively, versus `0.3034` and `0.5082`
in v01. This small change reflects current scoring/status handling and receiver
row-union output, not an expanded H-common interaction universe.

The v01 cSCC and MS stability estimates (`0.2986` and `0.7188`) are now retained
only as historical context. Independent review established that their adapter
views row-unioned receiver-specific scoring functionals, so a global rank
contrast was not scientifically comparable. The v02 primary, LOSO, concordance,
rank-stability, and report-level biology endpoints therefore fail closed with
`receiver_child_functionals_not_globally_comparable`. Receiver-scoped supportive
diagnostics remain separate: cSCC has two directional and no opposite locked
components; MS has four strong, four directional, and no opposite components,
including support for the locked control oligodendrocyte network. These are
silver-standard diagnostics, not a global primary or edge-truth claim.

The real Kuppe metadata audit used `backed='r'` on the 7.873 GB h5ad and did
not access the expression matrix. Its 29 samples cover 20 subjects and five
regions; the backed `obs` and prepared sample-design TSV have the same region
manifest digest. All ten pairwise region designs can be frozen under the
explicit assumption that a method/edge has complete sample-score coverage.
This is design evidence only, not an observed effect, uncertainty estimate, or
known-biology result. The compact tracked audit is
`benchmarks/results/kuppe_repeated_measures_design_summary.json`.

The public partial-cross-fit v3 smoke used the same two-fold functional for
synthetic active and ligand-only inputs and a four-donor/three-cell-type Kang
subset. Active and ligand-only completed in `11.86` and `11.11` seconds; their
two design applications were observed, while four of six receiver models were
explicitly unavailable because the non-receiver cell types had no eligible
training receptor family. The 10,057-cell Kang subset completed in `26.28`
seconds with all six receiver-family applications observed, 97/87 interactions
and 25/23 eligible families in its two folds. The combined process peak RSS was
`828,464 KiB`. This verifies implemented-stage execution and performance only:
no integrated LR score, active-versus-ligand-only mechanism endpoint, biology,
or method-superiority claim was evaluated. Compact evidence is in
`benchmarks/results/algorithm_crossfit_smoke_v3_summary.json`.

## Execution order

1. Connect response precision, family attribution and incremental downstream
   gain to public artifacts, add exact receiver-stage coverage, and rerun G1.5
   through the public workflow. Before connection, make the precision transform
   producer-owned and bind its ID to raw/transformed feature-aligned values;
   likewise bind incremental models to canonical sample-subject-context rows,
   raw response/design matrices, inner-fold tuning and held-out overlap checks.
2. Reuse prepared fold aggregates across stages and complete exact-equivalence
   sparse/block performance work.
3. Replace receiver row-union views with a true contrast-common OOF functional,
   then rerun cSCC development and one locked MS holdout for a valid CRYCHIC
   primary comparison.
4. Persist edge evidence and finish opposite/rank diagnostics.
5. Run a frozen forward/reverse signed Track-B campaign, apply D-common to MS,
   and feed real Kuppe method scores through the repeated-measures backend.
   Native NicheNet is a separate adapter claim and cannot be inferred from the
   existing prior-activity proxy.
6. Start G2/G3 only after the preceding scientific and computational gates.
