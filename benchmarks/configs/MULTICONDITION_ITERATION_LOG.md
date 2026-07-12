# Multi-condition benchmark iteration log

This log records benchmark-driven changes without discarding earlier outputs.
Real-cohort agreement and supportive biology are never used as tuning truth.

## Frozen inputs

- Retrieval date: 2026-07-12.
- cSCC input SHA256: `b15759def47df2c2aa5e1936398c9e57fab92dac6814b77d31839ae50b675b81`.
- MS CA-versus-Ctrl input SHA256: `612fe9c4cdaf88694a47e13ba4458c828f206cd945eba9e9c72f46e9bd0196c7`.
- H-common LR payload SHA256: `e24148ad3d6ee0be0d1a71503c98934085072e22fcb46bf3f9a8f315d3424988`.
- Supportive biology was locked in
  `benchmarks/truth/multicondition_supportive_biology.yaml` before method
  outputs were inspected.

## I0: initial real-cohort baseline

- Scope: native CellChatDB resource, at most 800 interactions.
- cSCC: paired subject design, `~ condition`, 10 complete subject pairs.
- MS: independent subjects, `~ batch + lesion_type`, 5 CA and 6 Ctrl
  subjects.
- Inferential fields remained disabled by the v0.1 contract.
- Retained output: `benchmark_work/multicondition_v01/crychic_initial`.

Observed implementation defect: a subset invocation was reported as failed
unless every dataset in the mutable configuration file had a result with the
same configuration hash. The fitted MS result itself was complete. The runner
now reports invocation completion separately from configuration-wide
completion, and final runs use one immutable config snapshot.

## I1: formula-aware subject response

- Failure class: statistical design.
- Change: build a Patsy design matrix and reference grid from the declared
  formula; calculate estimable marginal contrasts after covariate adjustment.
- Pairing rule: paired effects use only complete subject pairs; independent
  groups first collapse repeated libraries within subject and context.
- Refusal rule: mixed paired/unpaired multi-region contrasts remain
  `not_estimable` until a validated repeated-measures model is available.
- Regression evidence: hand-calculated paired effects, batch-adjusted effects,
  rank-deficiency and mixed-design refusal tests.

## I2: equivalent contrast alias removal

- Failure class: performance.
- Observation: with two context nodes, global and topology-local contrast
  vectors are exactly equal, but the workflow scheduled both names.
- Change: retain the canonical global contrast and omit exact weight-vector
  aliases; direct response APIs still expose both families for diagnostics.
- Expected endpoint: lower wall time and attribution work with unchanged
  primary global score values.
- Final native config: `benchmarks/configs/multicondition_v01_final.json`.

## I3: NicheNet Track-B correction

- Failure class: task mismatch and performance.
- Observation: expanding one ligand-target score over every sender type created
  redundant rows and incorrectly implied sender attribution. Validation of the
  resulting table dominated runtime and an initial cSCC run was interrupted.
- Change: emit one `__source_agnostic__` ligand-to-receiver target-program score.
  Sender prioritization is explicitly outside this Track-B proxy.
- Additional preprocessing gate: integer count inputs are library-size
  normalized to 10,000 and log1p transformed; continuous prepared expression is
  preserved. The resolved transform is recorded in every manifest.
- Expected endpoint: a much smaller fixed universe, lower runtime, and no false
  sender claim. Track-B output is never entered into LR-STLR concordance.

## I4: equal-weight sender entropy boundary

- Failure class: numerical contract.
- Observation: normalized entropy for 13 exactly equal candidates evaluated as
  `1.0000000000000002`, causing a valid H-common cSCC run to fail the strict
  unit-interval contract.
- Change: clip the producer's normalized Shannon entropy to its mathematical
  range `[0, 1]`; all other interval validation remains strict.
- Regression evidence: equal-weight groups of 5, 13 and 19 candidates.

## I5: synthetic cross-method parity and CellChat seed boundary

- Failure class: adapter input contract and execution provenance.
- Scope: seven identical synthetic h5ad inputs across CRYCHIC, CellChat,
  CellPhoneDB, LIANA and NicheNet. CRYCHIC, CellChat, CellPhoneDB and LIANA use
  the five-edge H-common fixture (payload SHA256
  `f831fff1c3c03b8a7e2581e449cc50ca7e920e7cd0639354a7f61a5bffe72495`);
  NicheNet remains the separate source-agnostic Track B.
- Observation: three deterministic scenario seeds were valid uint32 values but
  exceeded R's signed-integer maximum. Direct `as.integer` conversion produced
  `NA`, so CellChat failed in `set.seed` before scoring. These failures were
  never interpreted as zero communication.
- Change: for each sorted sample ordinal, preserve
  `requested_seed + ordinal` when it is in `1..2147483647`; otherwise map it to
  `1 + ((effective - 1) modulo 2147483647)`. Thus all previously legal R seeds
  remain byte-compatible while uint32 overflow is handled deterministically.
- Provenance: every CellChat manifest records the requested uint32 seed, valid
  R range, mapping rule and the effective seed for every `sample_id`.
- Recovery rule: selected overwrite runs are marked `pending` together, then
  `running`, before old outputs are removed. A suite cannot report complete
  while a selected output is absent or being replaced.
- Truth boundary: Track A uses the exact 45 sender-receiver-LR edges per
  scenario; only active Sender-to-Receiver CXCL10-CXCR3 is positive. All-zero
  mechanisms are explicitly single-class/not-estimable. Track B stores only
  scenario-level receiver-response expectations and is not labeled as LR edge
  truth.
- Frozen CRYCHIC config:
  `benchmarks/configs/synthetic_multimethod_v01.json` (`state`,
  `min_cells=10`, `min_subjects_per_context=4`, `max_interactions=5`).

## I6: paired differential truth estimand

- Failure class: evaluation estimand mismatch, without an algorithm change.
- Observation: per-sample edge ranking can recover a constitutively top-ranked
  edge while remaining insensitive to its paired condition change. In the
  active control, CRYCHIC's Sender-to-Receiver CXCL10-CXCR3 native score rose
  in every subject (mean paired effect `0.0914694`) but its rank-strength
  difference was exactly zero because the edge was already top-ranked in both
  contexts.
- Primary estimand: orient each method's native score using its declared
  `score_direction`, average technical rows within subject and context, then
  calculate paired `stim - ctrl` differences. Native magnitudes are never
  compared across methods. `not_returned` is not imputed as native zero.
- Sensitivity estimand: repeat the paired contrast on the direction-normalized
  within-sample rank strength. Here only `not_returned` receives the
  contractually defined bottom rank; failures and missing measurements remain
  missing.
- Active primary results over the exact 45-edge truth universe: CRYCHIC
  AUROC `0.977273`, average precision `0.5`, known-edge rank `2/45`;
  CellChat AUROC/AP `1/1` among only `5/45` raw-score-estimable edges;
  CellPhoneDB AUROC `0.5`, average precision `0.0227273`; LIANA AUROC
  `0.954545`, average precision `0.333333`, known-edge rank `3/45` after
  lower-score orientation.
- Estimand diagnostic: the active known-edge rank-strength difference was
  `0` for CRYCHIC, CellChat and CellPhoneDB, versus `0.00277778` for LIANA.
  This explains why native differential recovery and rank-difference recovery
  answer different questions; no method parameters were tuned in response.
- Negative controls: AUROC, average precision and top-k truth metrics are
  explicitly `not_estimable` because the truth is all-zero/single-class.
  Pre-registered CXCL10-CXCR3 effects are retained only as directional
  diagnostics and never presented as p-values, FDR or type-I error.
- Frozen evaluator config:
  `benchmarks/configs/track_a_differential_truth_v01.json`. The finalizer input
  `track_a_differential_truth/simulation_records.tsv` combines 728 Track-A
  records with the unchanged 42 Track-B receiver-program records.

## I7: gate-aware downstream-support candidate holdout

- Scope: post-benchmark method development only. The primary benchmark,
  finalizer input, real-data results and default v1 algorithm remain frozen.
- Failure class: weak attribution amplification. The historical downstream
  support `beta_j / max(beta)` can assign a large relative weight when every
  fitted coefficient explains only a small fraction of the positive receiver
  response.
- Candidate: opt-in
  `gated_response_norm_attribution_support_v2` uses
  `clip(||B_j beta_j||_2 / ||y_positive||_2, 0, 1)` and multiplies the same
  sample target-activity projection as v1. `B_j` is already receptor-gated.
- Contract: default and explicit v1 produce identical sample scores and
  effective run parameters. V2 must be selected by the workflow field
  `downstream_attribution_support_method`; its formula and application are
  persisted under `attribution.downstream_support`. Unknown values fail before
  fitting.
- Holdout: base seed `20260819`, never used by the frozen benchmark; eight full
  CRYCHIC runs pair v1/v2 on identical active, ligand-only, target-only and
  receptor-knockout inputs.
- Active: known-edge effect changed from `0.0759201` to `0.0578804`, while
  retaining rank `1/45`, positive direction in `8/8` subjects, and AUROC/AP
  `1/1`.
- Negative diagnostics: ligand-only effect decreased from `0.0770808` to
  `0.0528437`; target-only from `0.0117114` to `0.00830725`. The preregistered
  future-change gate therefore passed. Receptor-knockout remained rank `45/45`
  and negative, but its magnitude weakened from `-0.616414` to `-0.460676`.
- Limitation: ligand-only remains a positive rank-1 edge under both versions;
  v2 attenuates weak attribution but does not by itself distinguish ligand-only
  availability from integrated communication. Passing this single holdout
  supports further default-change evaluation, not an immediate switch.
- Candidate output:
  `benchmark_work/multicondition_v01/downstream_support_candidate_v2_holdout`.
  It is explicitly excluded from the finalizer and emits no formal inference.

## I8: exact-preserving metric validation vectorization

- Failure class: report-finalization performance; no scientific estimand or
  adapter result changed.
- Observation: fixed-universe validation repeatedly built Python edge sets for
  every sample and iterated edge-by-edge to verify `resource_unavailable` state.
  Comparison ranks and denominators were also assigned one sample at a time.
- Change: validate the fixed universe with per-identity declared-size and
  sample-size invariants plus vectorized grouped Boolean checks; calculate
  comparison sizes, ranks and strengths with grouped transforms. The original
  five derived columns retain their historical order.
- Controlled A/B: on the same 2,500,960-row cSCC CellChat H-common mapped score
  table, the reconstructed pre-I8 path required `68.6274 s` and the I8 path
  required `15.0791 s` (`4.551x` faster). `pandas.testing.assert_frame_equal`
  passed with exact values, dtypes, index and column order.
- Readback validation now performs one strict validation per CRYCHIC score view
  after materialization instead of revalidating each intermediate table. The MS
  native readback completed in `594.56 s` at `10,877.32 MiB` peak RSS; this is
  an execution record, not a controlled cross-dataset speed comparison.
- Regression evidence: sparse shifted-universe rejection,
  sample-varying-resource-state rejection, legacy derived-column ordering, and
  strict readback output-contract tests.

## I9: zero-recovery F1 boundary

- Failure class: metric reporting contract; no method score or ranking changed.
- Observation: for the active CRYCHIC rank-strength sensitivity estimand, the
  top-k set contained no true positive, so precision and recall were both zero.
  The evaluator emitted `NaN` for `0/0` while retaining `status=observed`, which
  the report layer correctly rejected.
- Change: define top-k F1 as `0` when finite precision and recall are both zero;
  retain `NaN + not_estimable` only when the truth class or prediction set is
  genuinely unavailable.
- Impact: one of 770 simulation records changed from invalid observed-NaN to
  observed `0.0`. AUROC, average precision, coverage, known-edge diagnostics,
  negative-control states, adapter outputs and all real-data results are
  unchanged.
- Regression evidence: a top-1 fixture that selects only a negative edge now
  requires precision `0`, recall `0`, F1 `0`; the report source contract also
  rejects any future observed non-finite estimate.

## I10: exact-preserving subject-effect and family-stability vectorization

- Failure class: finalization performance and public metric-API robustness; no
  method score, benchmark estimand, frozen input or biological interpretation
  changed.
- Observation: paired and independent-group edge effects entered Python once
  per frozen edge. On the real H-common views this required `449.967 s` for
  125,048 cSCC effects and `210.699 s` for 51,678 MS effects. Repeated
  split-half stability also rebuilt family filters and top-k edge indexes.
- Change: materialize subject-by-edge matrices within each complete method
  identity, preserve the historical first-eligible-subject reduction order,
  aggregate status flags in batches, and compute the small per-edge numerical
  reductions in the exact legacy order. Family metrics build one family-row
  lookup and use merged row identity for tie-inclusive top-k sets. Paired LOSO
  and effect APIs now normalize numeric subject/context labels consistently.
- Controlled full-table A/B: cSCC paired effects required `15.475 s` in the
  final implementation (`29.08x` faster) and MS independent-group effects
  required `7.553 s` (`27.90x` faster). Both complete DataFrames passed
  `assert_frame_equal(check_exact=True)` against the scalar reference; hashes
  were `7c49c29118ac977ee459cae6af62e5fe8fe532364430a3fd4d2d7e67bc371e92`
  and `4fa7d497a0b74f3866e4b069528fa5a2bd79b54252c70646faa34639fea5a7c4`.
- Family benchmark: on 50,000 edges across 100 families, the generic family
  metric helper improved from `1.078 s/call` to `0.335 s/call` (`3.22x`) with
  an exact result dictionary; 200 repeated calls completed in `64.985 s`.
- Regression evidence: all canonical missing/status reasons, repeated
  libraries, disjoint multi-identity subjects, numeric contexts, mixed key
  types, ties, constant ranks and 50 randomized paired/unpaired tables. Numeric
  family keys no longer collide with same-text string keys, and macro coverage
  cannot exceed one through key coercion.

## I11: publication-scope and provenance hardening

- Failure class: report integration and claim boundary; adapter scores, frozen
  biological evidence and method rankings are unchanged.
- Observation: the first final render attempt exposed four presentation-layer
  risks: synthetic LOSO rows could enter a figure titled as real data;
  concordance/performance labels could overwrite different datasets;
  `not_estimable` wall-time/RSS comparisons were assigned signed changes; and
  the frozen-prior receiver-program proxy could be mistaken for the
  preregistered Track B target-program primary endpoint.
- Scope contract: finalizer outputs now carry explicit `truth_scope` on
  coverage, primary, stability, concordance, performance, edge-effect and LOSO
  tables. Cross-dataset primary aggregation rejects synthetic scopes. Report
  inputs with missing scope fail closed rather than defaulting to real data.
- Figure contract: Figures 2-4 use real-data rows only and retain dataset,
  resource arm and analysis track. Figure 4 no longer overwrites repeated method
  pairs or performance labels. Figure 7 computes changes only for observed
  controlled records; concurrent timing/RSS rows remain `NE`.
- Track B contract: the current source-agnostic frozen-prior output is labelled
  `proxy diagnostic`. Native `predict_ligand_activities` and the preregistered
  signed target-program macro-AUPRC primary endpoint were not run and remain
  `NE`.
- Statistical boundary: the MS cross-method endpoint is equal-subject,
  unadjusted CA-vs-Ctrl descriptive stability. The recorded
  `~ batch + lesion_type` formula belongs to the CRYCHIC response layer and is
  not claimed as a shared batch-adjusted endpoint across methods.
- Evidence boundary: the active Track A truth has one positive and 44 negative
  edges; AUROC/AP are labelled single-positive diagnostics. The final 81-row
  biology table reports all 101 estimable components (43 strong, 51
  directional, 7 opposite), not only the zero-opposite native increment.
- Provenance: the report package records generator/finalizer/metric code,
  protocol, frozen specification, dependency lock, dirty-state hashes,
  WeasyPrint version and per-artifact byte counts/SHA256.
- Regression evidence: synthetic scope exclusion for Figures 2-4, explicit
  biology aliases, single-class AUROC rejection, >=1,000-null calibration gates,
  NE iteration semantics and artifact checksum completeness.

## Frozen rerun observations

The optimized native-resource rerun preserved the primary score artifact
exactly while removing duplicate response rows:

| Dataset | Initial responses | Final responses | Initial wall (s) | Final wall (s) | Sample-score SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| cSCC | 1,833,328 | 916,664 | 1,892.606 | 1,880.645 | `5d7be5f0f66409e8eb87c856a352362ba9a8ca8afef84a3b6a77b7c777f3ce26` |
| MS CA vs Ctrl | 1,156,140 | 578,070 | 1,022.320 | 893.522 | `c0adad4c8e36d54645f5c62b05ac422e598a6ededeaf65c0eb37b397c4120854` |

For both datasets, the initial and final `sample_scores.parquet` files are
byte-identical. The cSCC and MS runs occurred under different concurrent I/O
loads, so wall-time and peak-RSS changes are descriptive rather than a
controlled speed ranking. The 50% response-row reduction and exact score
preservation are the release criterion for I2.

## Release-candidate decision rules

- Primary real-data comparison uses H-common only. Native-resource results are
  separate sensitivity arms.
- cSCC primary CRYCHIC view is `global:'Tumor'`; MS primary view is
  `global:'CA'`. Reverse views are sensitivity analyses.
- Real data receive no AUROC or AUPRC. Locked biology observations are
  supportive evidence, not edge truth.
- A release candidate must pass all synthetic negative controls, fixed-universe
  validation, the configured test suite and static checks before report
  generation.
