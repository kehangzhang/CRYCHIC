# Suggestion 1 implementation status

- Audit date: 2026-07-13
- Starting audit baseline: `5e8b4b8`; this document also records the changes
  delivered in its own implementation commit
- Scope: implementation evidence, not a biological or superiority claim

This matrix records which recommendations are present in the public workflow,
which exist only as candidate primitives, and which remain absent. A primitive
is not counted as complete when the end-to-end workflow does not invoke it.

| Requirement | Status | Evidence and remaining work |
| --- | --- | --- |
| Four semantic outputs | Partial | Workflow contracts separate components, but public integrated scoring remains the legacy baseline and incremental downstream is `not_estimable`. |
| Explained-share v3 | Partial | Implemented and tested as a candidate support method; the baseline default remains relative-coefficient v1. |
| Held-out incremental downstream gain | Partial | Fit/apply primitives and mechanism tests exist; no certified public cross-fit orchestrator calls them. |
| Soft-min integration | Partial | Candidate integration and ablation tests exist; the public baseline still uses legacy four-component scoring. |
| Hard receptor eligibility | Partial | Scale-invariant candidate policy is tested; it is not the baseline default. |
| Winsorized normalized precision | Implemented | Attribution workflow and integration tests retain precision-transform provenance. |
| Strict complete-link families | Implemented | Chaining counterexamples are covered by family tests. |
| True family-first fitting | Partial | Candidate solver, allocation, conservation, and entropy contracts exist; the baseline still fits members before aggregation. |
| Complete scoring manifest | Implemented | Every tracked candidate artifact participates in the model/function identity. |
| State/ecosystem eligibility separation | Implemented | State rows no longer depend on abundance eligibility. |
| All-missing sender evidence | Implemented | Returns missing values rather than fabricated uniform weights. |
| Common sender functional | Missing | Current sender assignment is context-specific and in-sample. |
| Subject fold planning and OOF audit | Implemented | Subject blocks, K fallback, design re-audit, and exact coverage are tested. |
| Train/apply/cross-fit public workflow | Partial | Producer-owned sanitized train/apply APIs now fit and apply the frozen interaction universe and pass scope/poison tests, while explicitly remaining `partial_not_oof`. All later learned stages and the cross-fit orchestrator remain. |
| Exact-zero complex soft-min | Implemented | Zero and missingness behavior are covered by tests. |
| Train-only frozen interaction universe | Implemented | Fit/apply identity, reversed-test perturbation, resource provenance, and cap conflicts are tested. |
| H-common data-driven cap ban | Partial | v0.2 specs reject caps; no v0.2 real-data run exists yet. |
| Coverage-risk hierarchy | Partial | Nine-level candidate ledger and monotone threshold contract exist; no real threshold campaign has run. |
| G1.5 multi-edge, multi-seed truth | Implemented | 50-seed development and 200-seed holdout both pass all frozen gates. |
| G1.5 through public workflow | Missing | The simulation invokes candidate primitives directly, not the public cross-fit path. |
| RBO, weighted Kendall, top-k curve, rank interval, stable tier | Partial | Frozen-universe APIs, explicit missing/NE semantics, contracts, and tests are implemented; real cSCC/MS campaigns and report integration remain. |
| cSCC influence diagnostics | Partial | Paired LOSO, bootstrap, and influence tables exist; per-edge/family rank intervals and missingness trajectories do not. |
| Persisted edge evidence | Partial | Optional versioned Parquet persistence, manifest linkage, semantic cross-table validation, lazy reads, and backward compatibility are implemented. Real cSCC opposite-edge root-cause analysis remains. |
| Signed/reverse response | Partial | Directional semantics are implemented but not connected to the workflow. |
| Track-B signed macro-AUPRC | Missing | Native NicheNet and signed target-program evaluation remain `not_estimable`. |
| D-common batch/effect model | Missing | Current MS comparison is equal-subject, unadjusted, and descriptive. |
| Multi-node repeated-measures backend | Missing | Kuppe mixed paired/unpaired design is still unsupported. |
| Sparse/block/streaming core | Missing | Availability and scoring still materialize or iterate over large long-form data. |
| G2 graph fusion | Missing | No graph-fused attribution solver is present. |
| G3-F/G3-P inference | Missing | Correctly disabled; p/q and communication probabilities remain unavailable. |
| Default switch | Not eligible | Requires certified public cross-fit, real no-cap candidate noninferiority, and legacy reproducibility gates. |

## Frozen evidence

The G1.5 development run contains 1,050 observations and the independent
holdout contains 4,200 observations. Both have zero failed gates. The holdout
equal-edge active-minus-ligand-only margin is `0.341761` with 95% interval
`[0.335538, 0.347985]`. This supports candidate mechanism specificity only;
it does not establish real-data superiority.

The completed v0.1 real-data benchmark is an exploratory, in-sample baseline.
Its cSCC CRYCHIC stability estimate is `0.2986` with interval
`[-0.1475, 0.7237]`; its MS CA-versus-control estimate is `0.7188` with
interval `[0.6174, 0.7690]`. MS is a two-group analysis, and neither result is
a candidate cross-fit noninferiority result.

## Execution order

1. Complete the remaining ADR-004 stages and rerun G1.5 through the public workflow.
2. Complete exact-equivalence sparse/block performance work.
3. Freeze and run cSCC development followed by one locked MS holdout with the
   no-cap candidate and legacy methods.
4. Persist edge evidence and finish opposite/rank diagnostics.
5. Add native signed Track-B evaluation, then D-common and repeated-measures
   backends.
6. Start G2/G3 only after the preceding scientific and computational gates.
