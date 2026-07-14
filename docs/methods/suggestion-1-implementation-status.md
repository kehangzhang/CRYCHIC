# Suggestion 1 implementation status

- Audit date: 2026-07-14
- Starting audit baseline: `5e8b4b8`; this implementation batch starts from
  `86a5213` after the multi-condition v02 report
- Scope: implementation evidence, not a biological or superiority claim

This matrix records which recommendations are present in the public workflow,
which exist only as candidate primitives, and which remain absent. A primitive
is not counted as complete when the end-to-end workflow does not invoke it.

| Requirement | Status | Evidence and remaining work |
| --- | --- | --- |
| Four semantic outputs | Partial | Workflow contracts separate the components. The public opt-in path now fits and applies a source-, receptor- and sender-agnostic `receiver_program_score` parent, emits trusted paired incremental children plus family/member/sender diagnostics, and binds each component to its producer-owned lineage. The receiver program remains diagnostic only and is explicitly excluded from integrated edge evidence; public integrated scoring still uses the legacy baseline. |
| Explained-share v3 | Partial | Implemented and tested as a candidate support method; the baseline default remains relative-coefficient v1. |
| Held-out incremental downstream gain | Partial | Public cross-fit builds a producer-owned fold response, standardized-unit precision transform and typed incremental train/apply chain from the exact frozen design and eligible receiver-family basis. Fully paired and fully independent low-level estimands support technical-row aggregation and explicit structural-zero gains. The v7 contract maps raw family and autonomous bases into one frozen standardized response coordinate system, then evaluates the exact `null + residualized-context x family-effect` low-rank nuisance factorization instead of materializing a family x nuisance x feature tensor. The public opt-in paired path connects a code-registered `manifest_verified_static_trusted_v1` autonomous resource and producer-owned subject-blocked tuning. Response centering/scaling, nuisance projection, solver fitting and penalty scaling are refit in each inner training split while precision, encoder and family basis remain outer-frozen. Selection v2 uses subject-paired candidate-minus-best loss differences; eligible candidates use an explicit L1-first sparsity priority and L2 second. This is a declared policy order and descriptive heuristic, not a total cross-axis regularization magnitude, CI or noninferiority test. Estimable trusted chains can produce official `observed` training and held-out rows; caller-declared resources and unsupported independent/mixed inner tuning fail closed. Full-pipeline certification still requires repeated inference. |
| Soft-min integration | Partial | Family-common v2 applies the released soft-min only after same-row receptor eligibility and common-sender v3 ligand-contrast support. It preserves structural zero versus `not_estimable` precedence and binds `score_version=family_first_mechanistic_ligand_contrast_gated_softmin_v2`. The public baseline still uses legacy four-component scoring. |
| Hard receptor eligibility | Partial | Public partial cross-fit freezes a condition-blind, scale-invariant hard receptor gate in each training fold and performs no held-out gate fitting. Repeated samples are pooled with equal context weight within subject and equal subject weight within the fold. Family-common v2 combines that receptor gate with the separate train-only interaction ligand-contrast gate on the same row. Every planned receiver/sample application is retained in an exact-coverage audit. The combined gate remains absent from a certified common integrated score. |
| Winsorized normalized precision | Implemented | Precision v3 binds its exact fold-response parent, residual degrees of freedom, downstream feature scale, training row manifest, sample/subject provenance, encoder, ordered feature IDs and immutable raw/transformed values. Residual df <= 4 and missing-scale cases use equal supported-feature weights; higher-df inverse variance is converted to standardized-response units before winsorization and median normalization. This deterministic guardrail drives autonomous projection and the signed-residual solver boundary and is not an empirical-Bayes moderation claim. |
| Strict complete-link families | Implemented | Chaining counterexamples are covered by family tests. The deterministic incremental complete-link implementation matches a brute-force reference across random sparse profiles, zero-similarity and tolerance cases, and reuses one static partition per fold feature universe while retaining receiver-specific gates and artifact IDs. |
| True family-first fitting | Partial | Public opt-in cross-fit freezes hard eligibility, strict complete-link families and one medoid basis column per family, tunes and fits only eligible family columns, computes held-out family attribution and subject-family differential effects, then applies family-common v2. Only interactions that pass both receptor eligibility and the frozen common-sender v3 ligand-contrast gate enter family availability or member evidence. No supported gate plus any eligible `not_estimable` gate propagates NE; all observed unsupported gates give structural zero. For an otherwise positive core, a mixed supported plus eligible-NE family may retain the supported family core, but entropy, member weights/scores, and sender descendants fail closed; an earlier structural zero remains zero. The receiver program remains diagnostic only and the legacy baseline is unchanged. Source-bound v2 smoke and full-seven campaign summaries now pass their frozen checks, but the path remains noncertifying and synthetic. |
| Complete scoring manifest | Implemented | Every tracked candidate artifact participates in the model/function identity. Common-sender v3 additionally binds the exact `FrozenInteractionUniverse`, complete receiver x interaction candidate-sender manifest, canonical training availability, sanitized raw training input, complete-subject effects, and jointly reproducible Holm family. Family-common functional/application and cross-fit bindings use producer version `v2`. |
| Receiver-child scoring collection provenance | Partial | Core `ScoringCollectionManifest` contracts and a versioned result extension bind each contrast/repeat/fold to emitted receiver children, scoring functional IDs, source row counts and source-key digests. Persistence validates exact emitted `sample_scores` coverage and receiver/contrast edge partitions, while fixing `composition_status=partial_emitted_only` and `common_functional_across_receivers=false`. No planned receiver universe or authoritative persisted child model/version/universe registry exists yet. |
| State/ecosystem eligibility separation | Implemented | State rows no longer depend on abundance eligibility. |
| All-missing sender evidence | Implemented | Returns missing values rather than fabricated uniform weights. |
| Common sender functional | Partial | Public subject-cross-fit common-sender v3 consumes the intact producer-owned `FrozenInteractionUniverse` and a complete frozen receiver x interaction candidate-sender manifest. It averages repeated rows per sender/subject/context, maximizes at interaction level across frozen candidates, retains only subjects complete for every contrast context without weight renormalization, and uses a one-sided Student-t contrast. Every frozen interaction remains in one outer-fold x receiver x contrast Holm family; NE uses effective p=1 and remains in `m`. The same functional separately freezes prevalence priors and applies only sample-local held-out ligand evidence for non-causal sender allocation. Family-common v2 consumes the immutable interaction gate; the path remains separate from the legacy default and noncertifying. |
| Subject fold planning and OOF audit | Implemented | Subject blocks, K fallback, design re-audit, and exact coverage are tested. |
| Repeated full train/apply split stability | Partial | `CrossFitSpec.repeat_index` defaults to `0`, preserving the base cross-fit policy identity while deriving a distinct repeat identity. `RepeatedCrossFitSpec` reruns the complete public train/apply workflow for every repeat and returns producer-owned `RepeatedCrossFitDiagnostics` with four immutable-digest tables: `repeat_registry`, `family_fold_events`, `subject_family_repeat_values`, and `family_repeat_stability`. Exact subject x repeat opportunities are enforced. A family absent from an observed training-fold universe is explicitly `not_estimable`, never a zero or nonselection event; the released universe is declared as the union of observed fold-family IDs. A row-order-invariant subject-content manifest derives every fold scope digest, and a defensive snapshot binds expression, metadata, configuration, resource-bundle content, and target-prior content. Selection and effect distributions require complete repeats and distinct family-specific partitions; fit-level and subject-exposure selection denominators are separate, and structural zeros do not count as estimable observations. It is descriptive split-stability only: inference eligibility is always false, full children are still retained in memory, and streaming bootstrap, permutation, confidence intervals, p/q values, communication probabilities, and G3 calibration remain absent. |
| Train/apply/cross-fit public workflow | Partial | Public `run_subject_crossfit` derives folds from raw metadata, creates physical sanitized scopes, and freezes the interaction universe, candidate-sender manifest, common-sender v3 support, complete-formula EMM design, receptor, family, receiver-program, response and precision artifacts. An opt-in `PenaltyTuningSpec` connects paired subject-blocked inner evaluation, selected incremental parents and family-common v2 held-out applications. Training receiver reference expression must cover every expected sample exactly with authoritative subject/context lineage; held-out program applications bind the exact sample/subject/context manifest and expression digest. Producer-owned target-profile and input identities, tuning, selected penalties, trusted autonomous resources, gate-bearing edge evidence and receiver-specific sender inputs reject forged or stale parents. Independent/mixed inner one-SE tuning, repeated full-pipeline inference and default-method integration remain. |
| Exact-zero complex soft-min | Implemented | Zero and missingness behavior are covered by tests. |
| Train-only frozen interaction universe | Implemented | Fit/apply identity, reversed-test perturbation, resource provenance, cap conflicts, complete per-receiver candidate manifests, row-order invariance, out-of-manifest rejection, and Holm-family integrity are tested. Missing/NE interactions remain in the frozen family rather than shrinking the multiplicity denominator. |
| H-common data-driven cap ban | Implemented | v0.2 specs reject caps and the cSCC/MS H-common real-data runs completed without a cap. The harmonized resource has 638 interactions, below the former cap of 800, so this run validates the uncapped path but does not demonstrate coverage expansion from removing the cap. |
| Coverage-risk hierarchy | Partial | Nine-level candidate ledger and monotone threshold contract exist; no real threshold campaign has run. |
| G1.5 multi-edge, multi-seed truth | Implemented | 50-seed development and 200-seed holdout both pass all frozen gates. |
| G1.5 through public workflow | Partial | The frozen public runner sends three known edges and all seven registered scenarios through `run_subject_crossfit`. A source-bound pre-gate two-seed all-seven run exposed target-only integrated false positives; generator, truth, seeds, and tolerances were unchanged. A focused post-gate check on untouched `public-g15-003` recovered all 12 active state/ecosystem member/sender pairs and made ligand-only, target-only, and receptor-knockout integrated scores zero. The final source-bound seeds 001/002 full-seven regression recovered 24/24 active pairs, retained positive active-minus-ligand-only margins with zero coverage loss, and gave zero paired/raw integrated family false positives in all six controls. Target-only used `ligand_contrast_not_supported`; knockout used receptor ineligibility. This remains synthetic, noncertifying, and without repeated inference. |
| RBO, weighted Kendall, top-k curve, rank interval, stable tier | Partial | Frozen-universe APIs and the v02 finalizer/report run are complete with fixed `p=0.9`, Kendall power `1`, 200 split repeats, 2,000 subject bootstraps, 95% intervals, 0.80 top-k threshold, and seed `20260712`. All frozen members and receiver strata are required in every repeat. The real-data agreement rows are consequently all `NE`, rather than being inflated by shared-item intersection. A true LR equivalence/driver-family identifier remains absent, so `lr_family` is also explicit `NE`. |
| cSCC influence diagnostics | Partial | Paired LOSO, bootstrap, rank intervals, availability frequency, and stable-tier tables now exist. Strict frozen-member propagation makes the new rank endpoints `NE`; explicit per-edge missingness trajectories and family identifiers remain absent. |
| Persisted edge evidence | Partial | Optional versioned Parquet persistence, manifest linkage, semantic cross-table validation, lazy reads, and backward compatibility are implemented. Real cSCC opposite-edge root-cause analysis remains. |
| Signed/reverse response | Partial | Directional semantics are implemented but not connected to the workflow. |
| Track-B signed macro-AUPRC | Partial | A strict frozen-universe evaluator now expands every program into forward/reverse activation-compatible channels, penalizes wrong-direction scores, uses tie-aware AP and equal registered scenario-cell weighting, and propagates incomplete seeds/channels as `not_estimable`. No frozen multi-seed campaign or native NicheNet signed result has run, and the existing proxy cannot be relabeled as native. |
| D-common batch/effect model | Partial | A shared, frozen subject-level OLS primitive now applies one formula, categorical level registry and model ID across methods; it adjusts unpaired and paired-difference effects for declared batch covariates, detects target/batch confounding, and fails closed for mixed paired/unpaired designs. It has not yet run on the MS benchmark. |
| Multi-node repeated-measures backend | Partial | A frozen categorical OLS backend retains mixed paired/unpaired subjects, averages technical replicates within subject/context/adjustment cells, and reports exploratory effects with subject-cluster CR1 diagnostic SE only. A backed-`obs` audit of the real 29-sample/20-subject Kuppe design found 10/10 region contrasts estimable only under complete method-score coverage; no effect was fitted. Three condition contrasts adjusted for region were rank deficient, no auditable batch field exists, and finalizer integration remains. |
| Sparse/block/streaming core | Partial | Pseudobulk counts/detection use one group-indicator sparse multiplication. Incremental v7 removes the dense family x nuisance x feature coefficient tensor through an exact low-rank factorization while retaining a sparse frozen family basis. The finalizer scans repeated identity metadata in Arrow batches and reads included `run_id` views using Parquet predicates/projection. On the 92.83 MB cSCC input, metadata scanning fell from 29.94 s to 7.95 s and one-view memory from about 15.89 GB to 5.78 GB; one selected view still materializes and Track B retains its full-column contract. Availability/scoring long tables remain. |
| G2 graph fusion | Missing | No graph-fused attribution solver is present. |
| G3-F/G3-P inference | Missing | Correctly disabled; p/q and communication probabilities remain unavailable. |
| Default switch | Not eligible | Requires certified public cross-fit, real no-cap candidate noninferiority, and legacy reproducibility gates. |

## Pre-alpha API migration

Reference and held-out response paths now require aligned context IDs in
addition to sample and subject IDs. Application artifacts are producer-owned;
callers must use `run_subject_crossfit()` or the corresponding exported
`fit_*`/`apply_*` producer rather than constructing application dataclasses.
In-memory functional, training, application or cross-fit artifacts created
before the v7 contracts must be discarded and refitted so strict context-row
manifests, held-out input digests and receiver-program target-profile bindings
are regenerated.

Common-sender parameters and functionals now require schema `3.0.0`, and their
interaction support/gate IDs use identity schema `3`. Family-common functional,
application, cross-fit binding, and edge-evidence producers use `v2`; score rows
carry `family_first_mechanistic_ligand_contrast_gated_softmin_v2`. Older
common-sender objects, family-common v1 objects, cached fold artifacts, and rows
with `family_first_mechanistic_softmin_v1` must also be discarded and refitted
from raw fold scopes. Old and new IDs or score rows cannot be merged.

## Frozen evidence

The structural-zero v3 development rerun contains 1,050 observations across 50
fixed seeds and three known edges. It passed every supplied development gate,
while `default_switch_allowed` remains false. Its manifest records 600 positive
receiver-null denominators and 450 explicit structural-zero denominators:
global-null, abundance-only and ligand-only contribute 150 structural zeros
each. No v3 publication holdout has run; the v3 holdout namespace remains
reserved and unavailable for claims.

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

The typed public-cross-fit v4 smoke reran the active and ligand-only controls
through fold response, response-parented precision and formula-nuisance
incremental application. Each dataset produced six aligned typed chains and 48
exact receiver/sample coverage rows; all six precision parents were estimable
and all 48 official incremental rows correctly remained `not_estimable`. The
two observed diagnostic applications per dataset had mean bounded gain
`0.003032` for active and `0.000459` for ligand-only, while both mean raw gains
were negative. This verifies execution and fail-closed semantics, not mechanism
specificity or biological recovery. Compact evidence is in
`benchmarks/results/algorithm_crossfit_smoke_v4_summary.json`.

The fixed-penalty v5 rerun uses the paired branch of the loss contract and
explicit zero-denominator semantics; every synthetic subject has both control
and target samples. It deliberately supplies neither a trusted autonomous
resource nor `PenaltyTuningSpec`. The two observed receiver diagnostics have
mean raw gain `0.051055` for active and `-0.017472` for ligand-only; mean bounded
gain is `0.051055` and `0`, respectively. All 48 official rows per dataset remain
`not_estimable` under that fixed-penalty spec. The independent-group branch is
covered separately by low-level and public-workflow integration tests.
Separately, the autonomous-overlap counterexample was rerun under the v7 frozen
response-coordinate contract. It projects a generic program overlapping the LR
target: generic-only gain is zero, active unique gain is `0.937149`, and changing
subject-constant baselines from 0 to 25 changes gain by at most `1.11e-16`.
These are algorithm diagnostics, not certification or biology. Compact evidence
is in
`benchmarks/results/algorithm_crossfit_smoke_v5_summary.json` and
`benchmarks/results/autonomous_overlap_smoke_v1_summary.json`.

The source-bound trusted/tuned family-common smoke was regenerated with common-
sender v3 and family-common v2. Active selects `lambda1_fraction=0.1` in both
folds and retains `[1, 2]` nonzero receiver families. Synthetic
`CXCL10-CXCR3` is uniquely rank 1 in both state/ecosystem member and sender
views; mean member scores are `0.021175/0.020994`, and mean sender scores are
`0.010755/0.010671`. Ligand-only, receiver-autonomous, and global-null each
emit `480/480` structural-zero family rows and retain no nonzero family. The
four scenarios took `343.59` seconds and peaked at `569,096 KiB`. These are
single-seed synthetic algorithm diagnostics, not certification or biology.

The separate public G1.5 runner freezes three known edges, three registered seed
sets, and all seven scenarios before execution. A source-bound pre-gate run of
the first two registered seeds across all seven scenarios exposed the motivating
failure: target-only had `any_positive_integrated_family_rate=1.0`, and only
`2/12` target-only integrated truth rows conformed. This was an algorithm
failure, not a raw-score-only diagnostic. The failing artifacts are retained as
`public_family_common_g15_campaign_v1_f48e4d0_pre_ligand_gate_failure.json` and
its compact summary under `benchmark_work/algorithm_smoke/`. They must not be
overwritten or presented as validation evidence.

The generator, truth matrix, registered seeds, and tolerances were not changed
in response. Common-sender v3 instead freezes complete interaction/candidate
families, uses subject-equal complete-case interaction contrasts and
receiver-wise Holm support, and family-common v2 requires that upstream gate
before integration. A focused check on untouched seed `public-g15-003` ran
active, ligand-only, target-only, and receptor-knockout before the inspected-
seed regression. It recovered all 12 active state/ecosystem member/sender pairs;
all three controls had zero paired/raw integrated family positives. Target-only
known-edge rows were structural zero with `ligand_contrast_not_supported`, and
knockout rows used receptor ineligibility.

The final source-bound seeds 001/002 regression executed all seven scenarios.
All 24 active pairs were recovered, all 24 active-minus-ligand-only margins were
positive, and maximum coverage loss was zero. Mean margins were
`0.002446/0.010474` for state member/sender and `0.017997/0.016316` for
ecosystem member/sender. All six controls had zero paired and raw integrated
family false positives. The 14 runs took `23:14.45` and peaked at `584,420`
KiB. Scope is `development_full_seven_scenario_diagnostic`; this remains
synthetic, noncertifying evidence and supports neither biology, method
superiority, full OOF certification, nor a default switch.

The tiny active repeat-aware development smoke reran the complete public
train/apply chain twice for 8 subjects and 1,260 cells. It produced two distinct
subject partitions and exact coverage of all 16 subject-repeat opportunities.
Receiver known-family fit-level conditional selection frequencies were `0.75`
for `CXCL10-CXCR3`, `0.50` for `CCL5-CCR5`, and `0.75` for `EGF-EGFR`; equal
fold sizes made the subject-exposure frequencies identical. The aggregate was
`partially_observed`: 5/1,512 family rows had observed selection and effect
stability, while 1,507 were explicitly not estimable. Reusing each prepared
fold across training/application and receiver stages reduced observed elapsed
time from `225.97` to `172.13` seconds and peak RSS was `529,372` KiB. These are
implementation observations rather than a performance baseline or guarantee.
The four released diagnostics remain
descriptive split-stability tables, not stability probabilities, confidence
intervals, p/q values, biology evidence, or G3. Subject bootstrap, context
permutation, and formal repeated-pipeline inference have not been implemented.

## Execution order

1. Preserve the pre-gate failure, focused unseen-seed artifact, and post-gate
   full-seven artifact as immutable versioned evidence. Do not retune the
   generator, truth, tolerances, or gate after these results. Run the separately
   frozen competitive-sparsity sensitivity only if future registered repeats
   reproduce unstable recovery or active decoy selection.
2. Extend the implemented repeated full train/apply split-stability diagnostic
   with legal full-pipeline subject bootstrap, context permutation, effect-model
   refitting, and G3 calibration. Add independent and mixed-subject inner tuning
   only after their statistical estimands are frozen.
3. Reuse prepared fold aggregates across stages and continue exact-equivalence
   sparse/block performance work beyond the completed v7 nuisance
   factorization.
4. Develop a globally comparable contrast-common OOF functional, then rerun
   cSCC development and one locked MS holdout for a valid CRYCHIC primary
   comparison. The current receiver-specific family-common diagnostic does not
   validate receiver-row-union global rankings.
5. Persist edge evidence and finish opposite/rank diagnostics.
6. Run a frozen forward/reverse signed Track-B campaign, apply D-common to MS,
   and feed real Kuppe method scores through the repeated-measures backend.
   Native NicheNet is a separate adapter claim and cannot be inferred from the
   existing prior-activity proxy.
7. Start G2/G3 only after the preceding scientific and computational gates.
