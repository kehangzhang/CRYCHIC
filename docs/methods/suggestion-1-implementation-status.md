# Suggestion 1 implementation status

- Audit date: 2026-07-17
- Starting audit baseline: `5e8b4b8`; this implementation batch starts from
  `86a5213` after the multi-condition v02 report
- Scope: implementation evidence, not a biological or superiority claim
- Latest contract additions: ADR-010 G2 gate and ADR-011 context-common
  certification requirement are accepted. Producer-owned run-level receiver,
  strict-family and directional LR hypothesis universes are now frozen before
  folds in the public directional workflow. The dedicated directional
  integrated-LR collection remains opt-in rather than a default-workflow
  release contract.

This matrix records which recommendations are present in the public workflow,
which exist only as candidate primitives, and which remain absent. A primitive
is not counted as complete when the end-to-end workflow does not invoke it.

| Requirement | Status | Evidence and remaining work |
| --- | --- | --- |
| Four semantic outputs | Implemented | `build_crossfit_semantic_scores()` exposes four producer-owned, independent views: held-out interaction availability, source-agnostic receiver program, family-first integrated LR member score, and subject-family differential effect. Result v9 persists all four exact tables plus their collection manifest, per-view grain/status/reason, row count and semantic digest. Integrated LR and differential rows replay against their canonical component ledgers; receiver-program source rows replay their source digest; availability remains bound to exact fold training/application/filter lineage. Rehashed integrated-LR tampering is rejected after ordinary file hashes are updated. An untuned run emits availability/program only and persists canonical empty integrated/differential tables as `not_produced`, without inventing zero or NE rows. `query_integrated_lr_scores()` has one stable public schema across v8-v9 exact-table reads and v1-v7 legacy projections, fixes `sender_unresolved_strength`, and excludes every registered directional contrast. The receiver program remains diagnostic and is never relabeled as a receptor- or sender-specific edge; all four views keep `formal_inference_allowed=false`. The graph-fused conditional family effect remains a separate opt-in parent and is not silently substituted into integrated LR scoring. Legacy `fit()` remains unchanged. |
| Explained-share v3 | Partial | Implemented and tested as a candidate support method; the baseline default remains relative-coefficient v1. |
| Held-out incremental downstream gain | Partial | Public cross-fit builds a producer-owned fold response, standardized-unit precision transform and typed incremental train/apply chain from the exact frozen design and eligible receiver-family basis. Paired, independent and mixed allocations have frozen validation-loss estimands and enter subject-blocked inner tuning with exact train/validation parent IDs. Mixed and repeated-multi-context outer responses now use the strict subject-equal CR2 fold producer. Per-feature failures remain typed NE with zero downstream precision weight; at least one eligible feature lets the response reach the precision layer, while `min_positive_features` still fails the complete incremental chain closed. Fewer than six effective subject clusters, rank, leverage, covariance or degenerate-variance failures never fall back to CR1. The old CR1 producer/API and identities remain available as diagnostic-only and can never produce official rows. The v7 incremental contract maps raw family and autonomous bases into one frozen standardized response coordinate system, then evaluates the exact `null + residualized-context x family-effect` low-rank nuisance factorization instead of materializing a family x nuisance x feature tensor. `FrozenLatentNuisanceSpec` learns a small precision-weighted receiver-autonomous basis separately in every outer and inner training split. Controls exclude the complete frozen target-prior universe, including receptor-ineligible families; observed, hybrid and typed-NE artifacts bind exact training rows and numerical diagnostics. Response centering/scaling, latent learning, nuisance projection, solver fitting and penalty scaling are refit in each inner training split while precision, encoder and family basis remain outer-frozen. The selected candidate retains its exact inner-OOF parents and produces a typed subject-family gain calibration artifact. An eligible CR2 response with approved static or nested fold-learned nuisance and certified tuning can produce descriptive OOF `observed` training and held-out rows; caller-declared static resources remain noncertifying. Result v8 persists and replays a backend manifest, ordered feature diagnostics, precision method/lineage/support and ordered-value digest. Full-pipeline inference still requires repeated resampling calibration. |
| Soft-min integration | Partial | Family-common v2 applies the released soft-min only after same-row receptor eligibility and common-sender v3 ligand-contrast support. The receiver-balanced v4 collection maps receiver-relative gain through the selected-penalty inner-OOF positive-gain ECDF, re-evaluates soft-min from absolute availability and the frozen percentile, then applies prior quality. The former Q0.75 coefficient multiplier is absent from the primary score. Both paths preserve structural zero versus `not_estimable` precedence. This does not establish a common biological functional across receivers, and the public baseline still uses legacy four-component scoring. |
| Hard receptor eligibility | Partial | Public partial cross-fit freezes a condition-blind, scale-invariant hard receptor gate in each training fold and performs no held-out gate fitting. Repeated samples are pooled with equal context weight within subject and equal subject weight within the fold. Family-common v2 combines that receptor gate with the separate train-only interaction ligand-contrast gate on the same row. The receiver-balanced collection preserves those gates and structural-zero priority over later missing components. Every planned receiver/sample application is retained in an exact-coverage audit; cross-receiver comparability and formal inference remain absent. |
| Winsorized normalized precision | Implemented | Precision v3 binds its exact fold-response parent, residual degrees of freedom, downstream feature scale, training row manifest, sample/subject provenance, encoder, ordered feature IDs and immutable raw/transformed values. Residual df <= 4 and missing-scale cases use equal supported-feature weights; higher-df inverse variance is converted to standardized-response units before winsorization and median normalization. This deterministic guardrail drives autonomous projection and the signed-residual solver boundary and is not an empirical-Bayes moderation claim. |
| Strict complete-link families | Implemented | Chaining counterexamples are covered by family tests. The deterministic incremental complete-link implementation matches a brute-force reference across random sparse profiles, zero-similarity and tolerance cases, and reuses one static partition per fold feature universe while retaining receiver-specific gates and artifact IDs. |
| True family-first fitting | Partial | Public opt-in cross-fit freezes hard eligibility, strict complete-link families and one medoid basis column per family, tunes and fits only eligible family columns, computes held-out family attribution and subject-family differential effects, then applies family-common v2. Only interactions that pass both receptor eligibility and the frozen common-sender v3 ligand-contrast gate enter family availability or member evidence. No supported gate plus any eligible `not_estimable` gate propagates NE; all observed unsupported gates give structural zero. For an otherwise positive core, a mixed supported plus eligible-NE family may retain the supported family core, but entropy, member weights/scores, and sender descendants fail closed; an earlier structural zero remains zero. The receiver program remains diagnostic only and the legacy baseline is unchanged. Source-bound v2 smoke and full-seven campaign summaries now pass their frozen checks, but the path remains noncertifying and synthetic. |
| Complete scoring manifest | Implemented | Every tracked candidate artifact participates in the model/function identity. Common-sender v3 additionally binds the exact `FrozenInteractionUniverse`, complete receiver x interaction candidate-sender manifest, canonical training availability, sanitized raw training input, complete-subject effects, and jointly reproducible Holm family. Family-common functional/application and cross-fit bindings use producer version `v2`. |
| Receiver-child scoring collection provenance | Implemented | Core `ScoringCollectionManifest` v4 contracts bind each contrast/repeat/fold to the complete run receiver universe, exact training-support IDs, typed emitted/not-produced/not-estimable children, scoring functional IDs, filter-universe IDs and source row digests. Cross-fit result v9 persists the canonical fold x receiver support grid, complete run-root receiver-family opportunity universe and authoritative v4 registry; v8 remains read-only with the semantic-table contract, v7 remains read-only with support-grid authority, v2-v6 remain read-only under the older v3 authority, and v1 retains its explicit legacy boundary. |
| Cross-receiver contrast-common scoring | Partial | The opt-in public path creates one producer-owned receiver-balanced v4 collection per contrast/fold over the exact planned receiver universe and requires an identical held-out grid across receiver children. Each tuned receiver retains selected-penalty inner-OOF subject-family losses and a zero-preserving positive-gain ECDF. LR rows use `softmin(absolute availability, calibrated gain percentile) x prior quality`; sender-LR rows allocate that prior-adjusted parent by the frozen common-sender `assignment_weight`, so every complete sender group sums exactly to `global_lr_score`. Raw `ligand_availability x training_prevalence_prior` remains diagnostic and is no longer multiplied into the primary sender score. Positive gain under missing or low-support calibration is NE and never falls back to raw gain or factor one. Functional identity binds exact receiver calibration artifacts, knots, support, tuning, selected penalty and outer functional lineage. Result schema v9 and contrast-common sender table schema `3.0.0` validate assignment completeness, entropy, row formula, zero/NE precedence, group conservation and receiver-support lineage; v1-v8 remain read-only. `cross_receiver_percentile_rank_eligible` requires every receiver mapping observed, while `common_functional_across_receivers=false` remains invariant. Receiver-stratified and predeclared equal-weight macro endpoints are supportive; pooled raw-score AUROC/AUPRC/global top-k remain forbidden. The bounded cSCC smoke has not established complete calibration, biology, or superiority. |
| State/ecosystem eligibility separation | Implemented | State rows no longer depend on abundance eligibility. |
| All-missing sender evidence | Implemented | Returns missing values rather than fabricated uniform weights. |
| Common sender functional | Partial | Public subject-cross-fit common-sender v3 consumes the intact producer-owned `FrozenInteractionUniverse` and a complete frozen receiver x interaction candidate-sender manifest. It averages repeated rows per sender/subject/context, maximizes at interaction level across frozen candidates, retains only subjects complete for every contrast context without weight renormalization, and uses a one-sided Student-t contrast. Every frozen interaction remains in one outer-fold x receiver x contrast Holm family; NE uses effective p=1 and remains in `m`. The same functional separately freezes prevalence priors and applies only sample-local held-out ligand evidence for non-causal sender allocation. Family-common v2 consumes the immutable interaction gate; the path remains separate from the legacy default and noncertifying. |
| Subject fold planning and OOF audit | Implemented | Subject blocks, K fallback, design re-audit, and exact coverage are tested. |
| Complete descriptive OOF certification | Implemented | `CrossFitOOFCertificationAudit` produces a stable requirement ledger over the authoritative registry, trusted resource, tuning policy, every planned train/apply child, subject disjointness and binding lineage. Certified persisted rows are explicitly descriptive OOF only; formal inference remains disabled. Audit and requirement payloads are strict exact-field, ID-recomputed and order-canonical on load. |
| Repeated full train/apply split stability | Partial | `CrossFitSpec.repeat_index` defaults to `0`, preserving the base cross-fit policy identity while deriving a distinct repeat identity. `RepeatedCrossFitSpec` reruns the complete public train/apply workflow for every repeat and returns producer-owned diagnostics v3 with six immutable-digest tables: the repeat registry, fold events, subject-repeat values, family stability, subject point estimates and family point estimates. The run-root receiver x strict-family universe is frozen before folds, mandatory on ordinary `CrossFitArtifacts`, and replayed by result v9. Every observed fold parent must reproduce the exact prior, feature, driver, threshold and family axes; missing receivers/families stay typed `not_estimable` on the complete grid. Point estimation averages OOF values within subject across complete repeats and then uses an equal-subject mean; any incomplete subject-repeat chain withholds the family estimate. Structural zeros stay explicit numeric algorithmic zeros in point estimation but never count as estimable stability observations. Full-pipeline bootstrap/permutation, active-null reruns and repeated cross-fit support bounded shared-snapshot threads with serial-equivalent scientific IDs and default `n_jobs=1`. Repeated diagnostics retain a separate authenticated execution metadata ID, deterministic repeat-index collection order and explicit memory-budget warning. Streaming execution and real G3 campaigns remain absent, all repeat estimates remain descriptive, and public `comm_probability` still requires the exact producer-owned G3-P gate. |
| Train/apply/cross-fit public workflow | Partial | Public `run_subject_crossfit` derives folds from raw metadata, creates physical sanitized scopes, and freezes the receiver, interaction, candidate-sender and receiver-family opportunity universes before fitting, followed by common-sender support, complete-formula EMM design, receptor, family, receiver-program, response and precision artifacts. An opt-in `PenaltyTuningSpec` connects subject-blocked inner evaluation, selected incremental parents, gain calibration, family-common held-out applications and receiver-balanced collections. Mixed-workflow tests verify CR2 responses, exact inner parents, disjoint subjects and complete coverage. Training-absent receivers remain typed `not_estimable` in every registry and resampling child without entering model parents. Result v9 round-trips the complete root family universe, support, response/precision, conserved sender, directional, latent and four semantic-view lineages; v1-v8 remain read-only. Default-method integration and real calibration remain. |
| Exact-zero complex soft-min | Implemented | Zero and missingness behavior are covered by tests. |
| Train-only frozen interaction universe | Implemented | Fit/apply identity, reversed-test perturbation, resource provenance, cap conflicts, complete per-receiver candidate manifests, row-order invariance, out-of-manifest rejection, and Holm-family integrity are tested. Missing/NE interactions remain in the frozen family rather than shrinking the multiplicity denominator. |
| H-common data-driven cap ban | Implemented | v0.2 specs reject caps and the cSCC/MS H-common real-data runs completed without a cap. The harmonized resource has 638 interactions, below the former cap of 800, so this run validates the uncapped path but does not demonstrate coverage expansion from removing the cap. |
| Coverage-risk hierarchy | Partial | Nine-level candidate ledger and monotone threshold contract exist; no real threshold campaign has run. |
| G1.5 multi-edge, multi-seed truth | Implemented | 50-seed development and 200-seed holdout both pass all frozen gates. |
| G1.5 through public workflow | Partial | The frozen public runner sends three known edges and all seven registered scenarios through `run_subject_crossfit`. A source-bound pre-gate two-seed all-seven run exposed target-only integrated false positives; generator, truth, seeds, and tolerances were unchanged. A focused post-gate check on untouched `public-g15-003` recovered all 12 active state/ecosystem member/sender pairs and made ligand-only, target-only, and receptor-knockout integrated scores zero. The final source-bound seeds 001/002 full-seven regression recovered 24/24 active pairs, retained positive active-minus-ligand-only margins with zero coverage loss, and gave zero paired/raw integrated family false positives in all six controls. The current-source fixed-partition tiny seeds 001/002 four-scenario regression again recovered all 24 active endpoint pairs: all four macro recovery and positive-margin fractions were `1.0`, coverage loss was zero, the minimum margin was `3.47e-7`, and ligand-only, receiver-autonomous and global-null had zero training, held-out, paired-integrated and raw-integrated family false positives. A prior current-source single-seed seven-scenario debug used a different implicit split and selected one ligand-only EGF family; that warning remains historical rather than validation evidence. The runner now binds `outer_fold_partition_seed` to the preregistered cross-fit seed so future code comparisons keep the same subject split. This remains synthetic, noncertifying, and without repeated inference. |
| RBO, weighted Kendall, top-k curve, rank interval, stable tier | Partial | Frozen-universe APIs and the v02 finalizer/report run are complete with fixed `p=0.9`, Kendall power `1`, 200 split repeats, 2,000 subject bootstraps, 95% intervals, 0.80 top-k threshold, and seed `20260712`. All frozen members and receiver strata are required in every repeat. The real-data agreement rows are consequently all `NE`, rather than being inflated by shared-item intersection. A resource-independent `molecular_lr_equivalence_id`, separate mechanistic-variant ID, complete source crosswalk, and frozen molecular axis are implemented across directional LR result-v8/collection/sidecar contracts and the optional multi-method score contract. The finalizer accepts a checksum-bound per-run crosswalk, requires a complete many-to-one join before score validation, and fails closed on partial, unsupported, duplicate, tampered or unapplied mappings. Molecular-equivalence ranking is now estimable for newly finalized bound score tables; historical real-data tables remain `NE` until rerun with valid crosswalks. |
| cSCC influence diagnostics | Partial | Paired LOSO, bootstrap, rank intervals, availability frequency, and stable-tier tables now exist. Strict frozen-member propagation makes the new rank endpoints `NE`; explicit per-edge missingness trajectories and family identifiers remain absent. |
| Persisted edge evidence | Partial | Optional versioned Parquet persistence, manifest linkage, semantic cross-table validation, lazy reads, and backward compatibility are implemented. Real cSCC opposite-edge root-cause analysis remains. |
| Signed/reverse response | Partial | `DirectionalContrastPairSpec` registers exact forward/reverse contrasts. Before fold planning, the public workflow freezes a producer-owned external-resource `receiver x family x LR x mode` universe with explicit mapped, unmapped and ambiguous accounting, then makes every supported fold reproduce that family/LR axis. Universe v2 additionally binds each source interaction to a resource-independent molecular LR class while keeping the TargetPrior strict family orthogonal. Result schema v9 persists two independent channel-lineage rows per fitted binding, excludes them from the generic semantic LR/effect tables, and semantically replays the complete nested molecular/family universe while retaining v1-v8 read compatibility. `CrossFitResult.read_directional_opportunities()` and `query_directional_opportunities()` reconstruct the authenticated pair x fold x root-receiver x channel-role opportunity grid; a receiver absent from outer training retains stable static opportunity IDs with null fitted parents and typed `not_estimable`, while `query_directional_channels()` remains the fitted-binding-only compatibility view. `DirectionalIntegratedLRCollection` v4 exposes the two family-common LR applications as exact-grid, independently conserved activation-compatible channels and writes the molecular ID on every supported or typed-NE score row. Supported rows preserve their response, incremental, family-common, source-value and design parents. A training-absent receiver instead receives the complete held-out sample x frozen LR x mode x two-channel typed-NE grid with stable static IDs, null source values, null fitted/model parents and nonzero symmetric row counts. Neither channel is subtracted, divided, compared or relabeled as inhibition. Its atomic sidecar v2 binds the exact result-v9 parent and replays the molecular universe, support, design, directional registry, family-common components, observed rows, pair-level NE rows, and all-absent receiver grids on write and defensive read; tampered lineage and tables fail closed. `DirectionalTargetProgramScoreCollection` separately derives one common-scale signed OOF gene effect from the exact shared held-out `log1p_cpm` matrices, then emits complete source-agnostic increased/reduced activation-compatible target-program channels. The reference-transformed LR channels remain cross-channel non-comparable; active-inhibition claims and formal inference stay disabled. |
| Track-B signed macro-AUPRC | Partial | The prior-only `FrozenDirectionalTargetProgramUniverse` retains every TargetPrior driver, including typed-NE zero-match programs. The score collection uses the frozen run receiver axis and binds every fold training-support parent plus every real forward/reverse raw application. A receiver absent from any training fold is entirely typed `not_estimable`, preventing partial-OOF fitting; an all-absent receiver still has the complete program x two-channel NE grid without fabricated application IDs. A vectorized subject-equal OOF producer is numerically matched to the scalar OOF WLS backend and feeds the strict evaluator through a no-rescaling adapter. The adapter and evaluator explicitly reject LR, sender, native-NicheNet, active-inhibition and formal-inference claims. The evaluator penalizes wrong-direction scores, uses tie-aware AP and equal registered scenario-cell weighting, and propagates incomplete seeds/channels as `not_estimable`. No frozen multi-seed release campaign or native NicheNet signed result has run; this descriptive source-agnostic endpoint cannot be relabeled as LR recovery or native NicheNet. |
| D-common batch/effect model | Partial | A shared, frozen subject-level OLS primitive now applies one formula, categorical level registry and model ID across methods; it adjusts unpaired and paired-difference effects for declared batch covariates, detects target/batch confounding, and fails closed for mixed paired/unpaired designs. The multi-condition finalizer now accepts a checksum-bound sample design and emits an independent exploratory D-common table without replacing legacy effects; paired/no-batch intercept-only fitting is covered. It has not yet run on a checksum-bound real MS batch design, and receiver-child functionals remain excluded from global endpoints. |
| Multi-node repeated-measures backend | Partial | The original frozen categorical OLS/subject-cluster CR1 path remains exploratory and backward compatible, with fixed response/application/precision golden identities. ADR-008's subject-equal WLS/CR2 receiver backend is now the public outer-fold producer for mixed and repeated-multi-context cross-fit designs: technical replicates are averaged within frozen cells, mixed paired/unpaired subjects receive equal total weight, and at least six effective subject clusters are required. Producer-owned results retain condition/leverage/adjustment diagnostics, recompute the declared contrast and covariance before eligibility, support typed per-feature partial eligibility, and use response-scale-equivariant numerical checks. Small cluster, rank, contrast, leverage, covariance and degenerate-variance failures return `not_estimable` without CR1 fallback. Precision has a separate CR2 method and lineage; CR1 alone is forcibly diagnostic. Directional forward/reverse parents must share exact repeated design policy, cell mapping, support and ordered feature diagnostics; a symmetric partial feature grid becomes pair-level typed NE rather than shrinking the universe. Result v8 audits backend kind, response/design/effect IDs, ordered feature status/support, precision method/lineage/support and ordered-value digest, and rejects rehashed backend tampering. The exact-parent-bound `fit_crossfit_family_effect()` feeds the OOF CR2 effect layer from family-common sample scores. Eligible point fits still require full-pipeline resampling and G3-F before p/q release. The multi-condition finalizer separately exposes `repeated_measures_cr2`; its legacy CR1 option remains unchanged. A backed-`obs` audit of the real 29-sample/20-subject Kuppe design found 10/10 region contrasts estimable only under complete method-score coverage; no real method scores have been fitted through the new backend. Three condition contrasts adjusted for region were rank deficient and no auditable batch field exists. |
| Sparse/block/streaming core | Partial | Pseudobulk counts/detection use one group-indicator sparse multiplication. Incremental v7 removes the dense family x nuisance x feature coefficient tensor through an exact low-rank factorization while retaining a sparse frozen family basis. The active-null planner now uses integer edge encoding, O(1) occupancy updates and delayed link materialization: the 3,000-edge guard improved from 6.57 s to about 0.15 s, and the complete 306,250-edge NicheNet prior completes in 52.68 s at 378,928 KiB while preserving the frozen plan ID and all switch/overlap counts. Independent null reruns and complete repeated cross-fit children can use bounded threads over a shared read-only snapshot; execution metadata has its own integrity ID and does not alter scientific IDs. The finalizer scans repeated identity metadata in Arrow batches and reads included `run_id` views using Parquet predicates/projection. On the 92.83 MB cSCC input, metadata scanning fell from 29.94 s to 7.95 s and one-view memory from about 15.89 GB to 5.78 GB; one selected view still materializes and Track B retains its full-column contract. Availability/scoring long tables remain. |
| G2 graph fusion | Partial | The opt-in public cross-fit registry reconstructs every outer/inner physical subject scope from the authenticated raw input, refits node-specific receptor evidence on inner-train data, performs subject-blocked tuning, and applies each frozen graph fit only to its matching held-out response/design parents. Before any graph fit, `FrozenReceiverFamilyOpportunityUniverse` freezes the complete TargetPrior/root-feature strict-family axis across the run receiver universe without context labels or receiver expression; every supported fold parent must reproduce its prior, features, drivers, threshold and family definitions exactly. Registry schema v2 covers every fold x run receiver. A training-absent receiver has a support-bound record with null graph parents, while `derive_graph_fused_family_effects()` expands its full held-out subject x graph context x frozen-family grid as `not_estimable / receiver_absent_in_outer_training` with stable opportunity IDs and no fabricated training artifact, basis, workflow, application, problem or fit. The graph QP backend now applies connected-component active-set polishing to the trust-constr solution and certifies the original nonsmooth fused objective with KKT conditions before releasing coefficients. Exact-zero and fully fused plateaus are restored to exact boundaries; an uncertified solution becomes a typed numerical failure rather than a small positive edge. `lambda_F=0` retains exact independent elastic-net delegation and isolated components cannot exchange information. Observed records reject applications from another fit and require their canonical table to match producer records. Family effects remove one frozen family contribution at a time and retain source digests, negative raw gains, bounded gains and structural zeros. The opt-in `build_graph_fused_integrated_lr_scores()` adapter binds the exact focal global-one-vs-rest gate and family-common availability/allocation parents, uses the actual held-out sample domain per context, and fails closed when receiver training support is incomplete because no legal sample-level LR parent exists. Family/LR rows explicitly set `cross_receiver_comparable=false`, `cross_context_comparable=false`, `cross_mode_comparable=false`, and `formal_inference_allowed=false`; the adapter is not connected to the default or persistence path. Registry/effect coverage, root-input mismatch, graph/family/axis/subject lineage, lambda_F=0 parity, held-out poison, context/contrast mismatch and integrated-LR conservation tests pass. A 16-cell/1-replicate G2 smoke had mean fused-vs-unfused macro-AUPRC `0.8213` vs `0.7385` (10 wins, 3 ties, 3 losses), but the preregistered gate correctly returned `NE` because it requires 200 replicates per cell. The full frozen campaign and independent-group real regression remain; graph fusion is opt-in and cannot support a default or superiority claim. |
| G3-F/G3-P inference | Partial | Subject-equal OOF WLS/CR2 effects, a response-scale-equivariant multi-context Wald omnibus, legal full-pipeline bootstrap/permutation reruns and empirical `+1` p-values are implemented. ADR-012 freezes the state-mode primary/secondary endpoints and the only two-level TreeBH/Benjamini-Bogomolov procedure. The G3-F producer fixes three dependence structures crossed with global-null and 0.5/1/5/10% mixed cells, then derives type-I, mixed/primary/selective-child FDR, coverage, permutation diagnostics, paired power noninferiority and paired relative interval-width noninferiority from complete hypothesis-level replicate ledgers. The baseline ID and larger-is-better lower-bound margins (`-0.05` power, `-0.10` width) are protocol-bound; missing paired widths fail closed, and scenario metrics/evidence/gates are no longer caller-constructible. ADR-013 is a separate active-edge chain with the frozen v4 sender-edge universe, matched rewiring null, exact candidate x plan rectangle, empirical p, receiver/mode-stratified BUM local FDR, explicit eligible/structural/NE/failed calibration candidates, and equal-replicate metrics computed separately per runtime stratum. Package-owned generator manifests, immutable replay registries and exact serial/parallel attestations now support both G3 families. Independent atomic G3-F and G3-P results persist authoritative raw ledgers, reconstruct every derived table/evidence/gate, and require exact replay before write and on load for attested campaigns. Released active-probability artifacts additionally bind the replayed G3-P result artifact and recompute the BUM collection from raw point/null rows. Public raw-ledger summarizers remain diagnostic, and no production approved generator or real attested 1,000-replicate campaign has run; production q and `comm_probability` release therefore remain unauthorized. |
| Default switch | Not eligible | Deliberately unchanged: `fit()` remains the reproducible legacy baseline. The separate high-level `analyze()` usability entry defaults to the versioned `crossfit_descriptive_v1` profile, requires an explicit `CrossFitSpec` or `recommended_crossfit_spec(contrasts=...)`, and never falls back to legacy. The builder freezes an uncapped universe, train-fold latent nuisance learning, outer/inner seed lineage and penalty tuning. This is descriptive OOF UX, not a scientific default switch. A release switch still requires certified real biological nuisance resources, real no-cap candidate noninferiority, and stronger legacy reproducibility gates; current synthetic certification and real fail-closed execution are insufficient. |

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

The code-named cross-receiver common scorer now uses spec schema `4.0.0`,
producer identities `v4`, and score version
`receiver_gain_percentile_mechanistic_conserved_sender_v4`. Its identity binds
the selected-penalty inner-OOF gain calibration for every receiver and the
frozen common-sender allocation policy. Result schema `6.0.0` and
contrast-common sender table schema `3.0.0` persist assignment weights and
normalized entropy, and validate exact conservation of every complete
sender-resolved LR parent. Result schemas v1-v5 remain read-only; v4/v5 retain
their historical raw-evidence multiplication only inside private compatibility
branches. Old and new rows must not be mixed, and no schema authorizes pooled
cross-receiver endpoints.

ADR-013 active-edge artifacts now require the v4 conserved sender score and the
v2 active-edge source adapter. The active-probability result schema is `2.0.0`
and is independent of baseline/cross-fit result directories. Candidate empirical
p-values and local-FDR values remain diagnostic. A public `comm_probability`
table is written only when a producer-owned G3-P gate bound to the exact
universe, null, estimator, stratum and calibration protocol has passed.

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

The historical source-bound trusted/tuned family-common smoke was regenerated with common-
sender v3 and family-common v2. Active selects `lambda1_fraction=0.1` in both
folds and retains `[1, 2]` nonzero receiver families. Synthetic
`CXCL10-CXCR3` is uniquely rank 1 in both state/ecosystem member and sender
views; mean member scores are `0.021175/0.020994`, and mean sender scores are
`0.010755/0.010671`. Ligand-only, receiver-autonomous, and global-null each
emit `480/480` structural-zero family rows and retain no nonzero family. The
four scenarios took `343.59` seconds and peaked at `569,096 KiB`. These are
single-seed synthetic algorithm diagnostics, not certification or biology.

A post-change two-fold Kang 2018 real-data regression was rerun on the frozen
24,673-cell/full-transcriptome input. Cross-fit took `56.1` seconds and the
complete diagnostic took `136.4` seconds; the same descriptive statuses and
remaining-stage contract were retained (`receiver_autonomous_nuisance` and
`selected_penalty_inner_oof_gain_calibration`). This is a public-workflow
regression check only, not a graph-integrated real-data benchmark or superiority
claim. The graph-fused integrated adapter's full real-data smoke was not
completed because the current trust-constr graph solver remained computationally
heavy; that is tracked under production performance work.

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

After the run-level receiver-axis changes, a new current-source debug reran all
seven scenarios for `public-g15-003`. All 12 active member/sender endpoints were
recovered; every active-minus-ligand-only margin was positive and coverage loss
was zero. Target-only, receiver-autonomous, receptor-knockout, global-null and
abundance-only retained zero paired/raw integrated family activation. The run
also exposed that the historical seed label did not freeze the outer partition:
the changed split selected the EGF family in ligand-only, with raw integrated
mean `8.07e-4` and an ecosystem paired mean `1.66e-10`. This result is retained
as `public_family_common_g15_seed003_receiver_axis_20260716*.json`, not promoted
to campaign evidence. `build_campaign_spec()` now explicitly sets
`outer_fold_partition_seed=crossfit_seed`; no truth, tolerance, score, or gate
was changed after inspecting the result.

After molecular-equivalence/finalizer integration, the fixed-partition
`public-g15-001` tiny regression reran `active` and `ligand_only` through the
public workflow. All four state/ecosystem member/sender macro endpoints had
active recovery and positive-margin fractions of `1.0`, maximum coverage loss
was zero, and the minimum active-minus-ligand-only margin was `3.1094e-5`.
Ligand-only training selection, held-out selection, paired integrated and raw
integrated family false-positive rates were all zero. The stricter zero-truth
check also found no negative deviation incorrectly counted as conforming. The
artifacts are
`public_family_common_g15_post_molecular_core_tiny_seed001_20260716*.json`
under `benchmark_work/algorithm_smoke/`; their single-seed debug scope remains
excluded from campaign or superiority claims.

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
The six released repeat tables remain
descriptive split-stability and point-estimate tables, not stability probabilities, confidence
intervals, p/q values, biology evidence, or G3. Subject bootstrap, context
permutation and exact-universe inference are implemented. Active-null reruns
additionally support bounded shared-snapshot threads with exact serial
equivalence; repeated cross-fit remains serial, while full-pipeline bootstrap
and permutation now use the same bounded shared-snapshot execution contract.
No real G3 campaign has authorized formal release.

## Execution order

1. Preserve the pre-gate failure, focused unseen-seed artifact, and post-gate
   full-seven artifact as immutable versioned evidence. Do not retune the
   generator, truth, tolerances, or gate after these results. Run the separately
   frozen competitive-sparsity sensitivity only if future registered repeats
   reproduce unstable recovery or active decoy selection.
2. **Completed:** integrate `FrozenReceiverUniverse` into root cross-fit planning,
   every outer-fold registry, result v9 persistence and resampling reruns. A
   receiver absent from outer training retains its run-level opportunity as typed
   NE and is never passed to receptor gating or silently removed.
3. **Completed:** independent and mixed-subject inner tuning use frozen,
   subject-blocked validation estimands, and mixed/repeated outer responses use
   strict CR2 with result-v9 backend/precision audit. The legacy CR1 producer
   remains diagnostic-only. Repeated split-stability now freezes and audits the
   complete run-root family opportunity denominator. The same root parent is now
   mandatory in ordinary cross-fit and result-v9 persistence, and repeat point
   estimates use complete within-subject repeat means followed by equal-subject
   aggregation. Next run real full-pipeline effect refits and G3 calibration.
4. Reuse prepared fold aggregates across stages and continue exact-equivalence
   sparse/block/streaming work beyond the completed v7 nuisance factorization,
   optimized active-null planner and bounded null-plan concurrency.
5. Use the implemented receiver-balanced collection to run receiver-stratified
   cSCC development metrics and predeclared equal-weight macro supportive
   summaries. Freeze that descriptive contract before one locked MS holdout.
   Do not compute pooled cross-receiver AUROC/AUPRC or global top-k, and do not
   reuse historical receiver-child row-union rankings. No current result
   establishes biology or method superiority.
6. Persist edge evidence and finish opposite/rank diagnostics.
7. Run a frozen forward/reverse signed Track-B campaign, apply D-common to MS,
   and feed real Kuppe method scores through the repeated-measures backend.
   Native NicheNet is a separate adapter claim and cannot be inferred from the
   existing prior-activity proxy.
8. Start G2/G3 only after the preceding scientific and computational gates.
