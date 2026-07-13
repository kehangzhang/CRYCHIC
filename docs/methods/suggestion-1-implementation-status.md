# Suggestion 1 implementation status

- Audit date: 2026-07-13
- Starting audit baseline: `5e8b4b8`; current evidence is frozen through
  `a205385` and the multi-condition v02 report
- Scope: implementation evidence, not a biological or superiority claim

This matrix records which recommendations are present in the public workflow,
which exist only as candidate primitives, and which remain absent. A primitive
is not counted as complete when the end-to-end workflow does not invoke it.

| Requirement | Status | Evidence and remaining work |
| --- | --- | --- |
| Four semantic outputs | Partial | Workflow contracts separate components, but public integrated scoring remains the legacy baseline and incremental downstream is `not_estimable`. |
| Explained-share v3 | Partial | Implemented and tested as a candidate support method; the baseline default remains relative-coefficient v1. |
| Held-out incremental downstream gain | Partial | Fit/apply primitives and mechanism tests exist, but the public stage-level cross-fit orchestrator does not yet fit or apply downstream models. |
| Soft-min integration | Partial | Candidate integration and ablation tests exist; the public baseline still uses legacy four-component scoring. |
| Hard receptor eligibility | Partial | Scale-invariant candidate policy is tested; it is not the baseline default. |
| Winsorized normalized precision | Implemented | Attribution workflow and integration tests retain precision-transform provenance. |
| Strict complete-link families | Implemented | Chaining counterexamples are covered by family tests. |
| True family-first fitting | Partial | A high-level candidate path now enforces hard receptor eligibility, strict complete-link families, one medoid solver column per family, signed-residual preservation, and separate evidence-only member allocation. The legacy baseline still fits members before aggregation, and the stage is not yet in public cross-fit. |
| Complete scoring manifest | Implemented | Every tracked candidate artifact participates in the model/function identity. |
| Receiver-child scoring collection provenance | Partial | Core `ScoringCollectionManifest` contracts and a versioned result extension bind each contrast/repeat/fold to emitted receiver children, scoring functional IDs, source row counts and source-key digests. Persistence validates exact emitted `sample_scores` coverage and receiver/contrast edge partitions, while fixing `composition_status=partial_emitted_only` and `common_functional_across_receivers=false`. No planned receiver universe or authoritative persisted child model/version/universe registry exists yet. |
| State/ecosystem eligibility separation | Implemented | State rows no longer depend on abundance eligibility. |
| All-missing sender evidence | Implemented | Returns missing values rather than fabricated uniform weights. |
| Common sender functional | Partial | The public subject-cross-fit path now freezes the sender universe, cross-context training prevalence prior, minimum support and temperature inside each training fold, applies only sample-local held-out ligand evidence, and verifies exact stage-level OOF coverage. Low-level rows retain `partial_not_oof`, the aggregate result is `verified_train_only_oof_partial_pipeline`, and the functional is deliberately not connected to the legacy context-specific default score. |
| Subject fold planning and OOF audit | Implemented | Subject blocks, K fallback, design re-audit, and exact coverage are tested. |
| Train/apply/cross-fit public workflow | Partial | Public `run_subject_crossfit` derives estimability-aware folds from raw metadata, creates physical sanitized subject scopes, invokes the producer-owned train/apply primitives, emits actual sender applications, and passes exact OOF, scope, and held-out-poison tests. It verifies only availability/common-sender stages; receptor, response, family, downstream, encoder, and common-scoring stages remain fail-closed. |
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
| Multi-node repeated-measures backend | Missing | Kuppe mixed paired/unpaired design is still unsupported. |
| Sparse/block/streaming core | Partial | Pseudobulk counts and detection now use one group-indicator sparse multiplication with dense/sparse parity tests. Availability, scoring and finalization still materialize large long-form tables. |
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

## Execution order

1. Complete the remaining ADR-004 stages and rerun G1.5 through the public workflow.
2. Complete exact-equivalence sparse/block performance work.
3. Replace receiver row-union views with a true contrast-common OOF functional,
   then rerun cSCC development and one locked MS holdout for a valid CRYCHIC
   primary comparison.
4. Persist edge evidence and finish opposite/rank diagnostics.
5. Run a frozen forward/reverse signed Track-B campaign, then add D-common and
   repeated-measures backends. Native NicheNet is a separate adapter claim and
   cannot be inferred from the existing prior-activity proxy.
6. Start G2/G3 only after the preceding scientific and computational gates.
