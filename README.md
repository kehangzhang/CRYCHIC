# CRYCHIC

CRYCHIC: Cellular Relational dYnamics for Contextual Hypergraph Inference of Communication

CRYCHIC currently provides a runnable v0.1 exploratory baseline for
sample-aware cell-cell communication analysis. It validates count or explicitly
declared normalized AnnData, aggregates sample x cell-type pseudobulks, builds
balanced context contrasts, scores LR availability, optionally attributes
receiver responses with a versioned NicheNet prior, and writes atomic,
queryable results.

This is not the completed roadmap or a calibrated inference release. The
default v0.1 strengths are descriptive and in-sample. The opt-in subject
cross-fit path can produce held-out strength, effect and candidate inference
diagnostics, but public q-values and `comm_probability` remain unavailable
until their exact producer-owned calibration gates pass. Causal sender claims
are not supported.

## Current Development Status

The current branch adds more explicit evidence and provenance contracts while
preserving the exploratory v0.1 behavior:

- downstream output separates the source-, receptor- and sender-agnostic
  `receiver_program_score` from `incremental_downstream`. Public cross-fit now
  fits and applies a producer-owned receiver-program parent from fold-training
  medoid target profiles. Its identity binds the target-profile matrix, exact
  training sample/subject/context rows, subject-equal reference transform, and
  exact held-out expression and row manifest. The program is diagnostic only:
  it is explicitly excluded from `integrated_lr_score` and is not incremental
  or integrated edge evidence. Public cross-fit also fits a typed fold response,
  response-parented precision and incremental diagnostic. The opt-in path can
  consume a typed static autonomous resource and/or a
  `FrozenLatentNuisanceSpec`. The latter learns a small precision-weighted
  receiver-autonomous basis separately inside every outer and inner training
  split. Its control genes exclude the complete frozen target-prior universe,
  including targets of receptor-ineligible families; held-out rows are never
  accepted by the fitting API. Static resources can consume a
  manifest-backed registration that pins the manifest digest, payload path,
  SHA-256, size, canonical matrix digest, release, species, namespace, license,
  review scope and static TSV schema. With
  `manifest_verified_static_trusted_v1` provenance, or with an observed nested
  fold-learned latent artifact, an estimable subject-blocked tuning plan lets
  incremental training and held-out applications receive descriptive OOF
  `observed` status. A failed latent fit is retained as a typed diagnostic
  artifact. Caller-built static resources remain unverified and fail closed at
  the official gate;
- attribution now consumes a producer-owned v3 response-precision artifact. It
  binds the exact fold-response parent, residual degrees of freedom, downstream
  feature scale, training rows, subjects, encoder, ordered features and values.
  At residual df <= 4 it assigns equal weight to supported features; at higher
  df it converts inverse variance to standardized-response units before
  winsorization and median normalization. Missing scale also falls back to equal
  supported-feature weights. This is a deterministic guardrail, not empirical-
  Bayes variance moderation. The legacy full-data API remains unparented and
  exploratory;
- low-level incremental downstream uses explicit sample-subject-context row
  manifests, keyed response/design alignment, full training-input digests,
  sample diagnostics, paired-subject contrast losses and frozen independent-
  group pseudocontrast losses. A zero receiver contrast is recorded as an
  explicit structural-zero downstream gain rather than an undefined 0/0 ratio.
  The v7 contract maps raw family and autonomous bases into the same frozen
  standardized response coordinates and uses the exact
  `null + residualized-context x family-effect` low-rank nuisance factorization.
  This exact refactor preserves the prediction algebra while removing the
  dense family x nuisance x feature coefficient tensor.
  Exact/near autonomous overlap, precision-supported rank loss and technical-row
  replication fail closed or remain invariant as appropriate. Deterministic
  relative-penalty scaling and one-SE selection primitives exist for both the
  ordinary and signed residual spaces. Paired inner-fold production is connected
  inside the physical outer-training scope: response centering/scaling, nuisance
  projection, solver fitting and penalty scaling are refit per inner training
  split while the precision, family basis and encoder stay outer-frozen.
  Candidate selection uses the versioned subject-paired delta one-SE heuristic:
  common subject difficulty cancels in candidate-minus-best losses before the
  eligible set is ordered by explicit L1-first sparsity priority, then L2. This
  is a policy order, not a claim that cross-axis elastic-net candidates have a
  single physical regularization magnitude. The heuristic is not a confidence
  interval or noninferiority test. Independent and mixed-subject paths now use
  their own frozen subject-blocked validation-loss estimands. Mixed and
  repeated-multi-context outer folds use strict subject-equal CR2 responses;
  partial feature failures receive zero precision weight, insufficient feature
  or cluster support fails closed, and the legacy CR1 producer remains
  diagnostic-only. Result v9 persists a response-backend and precision audit,
  while analytic p/q remain unavailable;
- opt-in attribution candidates provide hard receptor eligibility, directional
  response channels, strict family-first bases, and evidence-weighted member
  allocation without changing the legacy default;
- tracked scores carry a `score_version` and `model_manifest_id` covering the
  fitted basis, coefficient digest, receptor gates, target weights, sender and
  downstream functionals, availability and precision transforms, filtering,
  and tuning artifacts;
- integrated baseline result directories retain their versioned emitted-score
  registry. Public cross-fit artifacts additionally derive an authoritative v3
  registry for every planned `contrast x repeat x fold x receiver` child,
  including typed not-produced/not-estimable states and the frozen filter
  universe. The registry and its stable identity are embedded in the source
  cross-fit manifest;
- availability state and ecosystem eligibility have separate status and reason
  fields, and missing sender evidence no longer receives an implicit uniform
  assignment;
- fold manifests enforce subject-blocked train/test separation, frozen
  interaction universes prevent test-fold filtering leakage, and the H-common
  benchmark arm rejects data-driven top-k interaction caps;
- common-sender v3 consumes the intact outer-training
  `FrozenInteractionUniverse` and a complete receiver x interaction candidate-
  sender manifest. It averages repeated rows per sender/subject/context,
  maximizes availability across frozen senders at interaction level, retains
  only subjects complete for every contrast context, and applies a receiver-
  wise Holm correction over every frozen interaction. Not-estimable
  interactions remain in the multiplicity denominator with effective p=1;
  held-out data cannot change family membership, rank, status, or gate ID;
- an edge-evidence ledger records component status, missingness reasons and
  provenance at sample x context x sender x receiver x interaction grain. It
  can be persisted as an optional, versioned Parquet result extension with
  `persist_edge_evidence=True`;
- the public `run_subject_crossfit` API now derives estimability-aware subject
  folds, creates physical sanitized train/test scopes, calls the producer-owned
  train/apply primitives, and audits exact OOF coverage for the frozen
  interaction universe and contrast-common sender functional. Fold artifacts
  also freeze nuisance encoders, receptor gates, strict family bases,
  receiver-program transforms, fold responses and response precision. A second
  audit covers every planned held-out `fold x contrast x receiver x sample`
  row, including receiver missingness. When a `PenaltyTuningSpec` is supplied,
  the same path produces family-first attribution, held-out subject-family
  differential effects, state/ecosystem family and member scores, and conserved
  sender allocation. Producer-owned bindings cover the exact incremental,
  tuning, effective autonomous-source, latent spec/artifact, static resource,
  receiver-program target-profile and held-out input, edge-evidence and
  receiver-specific sender input lineage. Training
  receiver reference expression must cover every expected training sample
  exactly and preserve its authoritative subject and context identity; partial
  or relabeled coverage fails closed. Trusted paired incremental children can
  be officially observed, and the source-/receptor-/sender-agnostic receiver
  program is connected as a diagnostic-only parent. A separate fail-closed
  audit certifies the aggregate only when the authoritative registry, approved
  static or nested fold-learned nuisance policy, tuning policy, every planned
  train/apply child and all subject leakage barriers are complete. This means
  complete OOF descriptive scoring
  only; untrusted, diagnostic-only or inestimable chains remain uncertified.
  Full-pipeline resampling and multiplicity workflows are separate opt-in
  artifacts and do not retroactively turn this descriptive audit into an
  inferential release;
- `CrossFitSpec.repeat_index` now defaults to `0`, preserving the existing
  cross-fit policy identity while deriving a distinct repeat identity for each
  nonzero repeat. `RepeatedCrossFitSpec` and
  `run_repeated_subject_crossfit()` rerun the complete public train/apply chain
  for every declared repeat and return producer-owned
  `RepeatedCrossFitDiagnostics`. Its six immutable-digest tables are
  `repeat_registry`, `family_fold_events`,
  `subject_family_repeat_values`, `family_repeat_stability`,
  `subject_family_point_estimates`, and `family_point_estimates`. Point
  estimation first averages OOF values within each subject across complete
  repeats, then takes an equal-subject family mean. Incomplete subject-repeat
  coverage fails the family estimate closed instead of conditioning on easier
  subjects. The contract enforces exact subject x repeat coverage. Diagnostics
  v3 freezes the complete
  receiver x strict-family opportunity universe before repeats from the root
  feature/receiver axes and TargetPrior, requires every observed fold parent to
  reproduce that family axis, and materializes every missing opportunity as
  typed `not_estimable`, never zero, unselected, or silently absent. The same
  root-family parent is mandatory in ordinary cross-fit and result-v9
  persistence. A row-order-invariant
  subject-content manifest derives every fold scope digest, while a defensive
  snapshot binds expression, metadata, configuration, in-memory resource
  content, and target-prior content. Complete-repeat fit-level and
  subject-exposure selection denominators remain separate, and structural zeros
  do not count as estimable observations in stability denominators. Numeric
  structural zeros remain explicit in the complete-grid descriptive point
  estimator. This object does not provide formal inference. Bootstrap,
  permutation, effect/q and
  active-probability workflows are separate, exact-parent-bound entry points;
- `fit_active_probability()` now composes a real point cross-fit, a frozen
  sender-LR-receiver candidate universe, complete degree/evidence-matched
  target-prior rewiring reruns, receiver/mode-stratified beta-uniform-mixture
  local FDR, and independent atomic persistence. It uses the v4 conserved
  sender statistic: frozen assignment weights divide each prior-adjusted LR
  parent and complete sender groups sum exactly to that parent. Candidate
  empirical p/local-FDR values remain diagnostic. A public
  `comm_probability` table appears only when a producer-owned G3-P gate bound
  to the exact runtime contracts has passed; callers cannot unlock it. The
  calibration producer derives metrics from explicit candidate-status ledgers,
  evaluates each runtime stratum separately, and has an independent atomic
  diagnostic result whose loader recomputes all summaries. Replay-backed
  generator manifests, immutable registries and attested summarizers are now
  implemented; attested G3-P persistence requires exact replay both before
  write and on load. Public raw ledgers remain diagnostic, and no production
  release campaign has yet supplied the required approved registry;
- G3-F calibration now fixes a dependence/prevalence grid and derives type-I,
  mixed/primary/selective-child FDR, coverage, and permutation diagnostics from
  complete hypothesis-level replicate ledgers. Hand-filled scenario metrics no
  longer form a gate. Independent G3-F persistence reconstructs the universe,
  procedure, raw ledgers, summaries and gate, with exact attested replay before
  write and on load. A real attested 1,000-replicate campaign has not run, so
  formal effect/q release remains unauthorized;
- active-null planning now uses integer edge occupancy and delayed link
  materialization. The 3,000-edge regression improved from 6.57 seconds to
  about 0.15 seconds; the complete 306,250-edge NicheNet prior completes in
  52.68 seconds at 378,928 KiB while preserving the pre-optimization plan ID and
  all switch/overlap counts. Independent full-pipeline null plans accept bounded
  `n_jobs`; threads share the immutable snapshot but each worker retains
  fold-local model memory, so the default remains `n_jobs=1`. Serial and
  parallel scientific IDs and values are exact-equivalent;
- complete subject-bootstrap and context-permutation resampling also accept
  bounded `n_jobs` with the same shared immutable snapshot policy. Plan order
  and scientific `result_id` are invariant to scheduling; requested/effective
  worker counts are stored in a separate execution metadata identity;
- a current-code Kang IFN-beta smoke processed 7,427 cells from 8 paired donors,
  two cell types and ctrl/stim contexts. Its two-fold v4 cross-fit took 56.00
  seconds and the full process peaked at 3,064,156 KiB. Family applications were
  observed, while official incremental rows remained `not_estimable` because no
  reviewed biological receiver-autonomous nuisance resource exists. This is a
  real-H5AD execution check, not probability calibration, biological validation
  or method superiority;
- a tiny active two-repeat development smoke reran the full chain for 8
  subjects and 1,260 cells, producing two distinct subject partitions. Receiver
  known-family fit-level conditional selection frequencies were `0.75` for
  `CXCL10-CXCR3`, `0.50` for `CCL5-CCR5`, and `0.75` for `EGF-EGFR`; equal fold
  sizes made the subject-exposure frequencies identical. Its status was
  `partially_observed`: 5/1,512 family rows had observed selection and effect
  stability, while 1,507 were explicitly not estimable. Reusing each prepared
  fold across training/application and receiver stages reduced observed elapsed
  time from `225.97` to `172.13` seconds. This is an implementation observation,
  not a performance guarantee, G3 result, biological claim, or formal inference;
- frozen design encoding now uses the complete declared Patsy formula and an
  exact EMM-contrast reparameterization. Numeric-coded categorical covariates
  can be declared with `categorical_covariates`; held-out levels are checked
  against the frozen registry and the restricted formula must reproduce the
  training column contract. Mutable Patsy state is not trusted;
- strict complete-link family clustering uses an output-equivalent incremental
  minimum-update heap and shares the static partition once per fold feature
  universe. The v3 partial smoke completed active and ligand-only controls in
  11.86 and 11.11 seconds, and a 10,057-cell Kang subset in 26.28 seconds;
  these are pipeline and performance checks, not integrated-edge accuracy or
  biological claims;
- current small algorithm checks record zero generic-only gain, `0.937149`
  active unique gain under autonomous overlap, and exact invariance to large
  subject-constant baselines. The fixed-penalty public cross-fit v5 smoke gives
  active bounded/raw diagnostic gain `0.051055/0.051055`, versus
  `0/-0.017472` for ligand-only; it deliberately supplies neither the trusted
  autonomous resource nor inner tuning, so all 48 official rows per dataset
  remain `not_estimable`.
  Independent-group execution is covered separately by unit and public-workflow
  integration tests but independent inner one-SE tuning is not yet supported;
- the historical source-bound one-seed four-scenario trusted/tuned family-common v2 smoke
  selects `lambda1_fraction=0.1` in both active folds, with nonzero receiver-
  family counts `[1, 2]`. The known synthetic `CXCL10-CXCR3` truth is uniquely
  rank 1 in state/ecosystem member and sender summaries; mean member scores are
  `0.021175/0.020994` and mean sender scores are `0.010755/0.010671`.
  Ligand-only, receiver-autonomous and global-null each emit `480/480`
  structural-zero family-score rows and retain zero nonzero families. Each
  scenario has 2/6 tuned and officially observed incremental children;
  family-common tables remain noncertifying diagnostics. The four scenarios
  took `343.59` seconds in total and peaked at `569,096 KiB`. This is a
  single-seed synthetic algorithm check, not biological validation;
- the frozen public family-common G1.5 runner now covers three known edges and
  all seven registered scenarios through `run_subject_crossfit`. The first
  source-bound two-seed full-seven run exposed a target-only failure under the
  pre-gate score: `any_positive_integrated_family_rate=1.0` and only `2/12`
  integrated truth rows conformed. That immutable failure artifact is retained;
  the generator, truth, seeds, and tolerances were not changed. A focused
  post-gate check on untouched seed `public-g15-003` then ran active,
  ligand-only, target-only, and receptor-knockout: all 12 active edge-view pairs
  were recovered, while every target-only and knockout integrated score was
  zero. The final source-bound regression ran seeds 001/002 across all seven
  scenarios. All 24 active state/ecosystem member/sender pairs were recovered,
  all 24 active-minus-ligand-only margins were positive, and coverage loss was
  zero. Mean margins were `0.002446/0.010474` for state member/sender and
  `0.017997/0.016316` for ecosystem member/sender. All six controls had zero
  paired and raw integrated family false positives. Target-only known-edge rows
  were structural zero with `ligand_contrast_not_supported`; receptor-knockout
  rows were structural zero with `receptor_interaction_ineligible`. The 14 runs
  took `23:14.45` and peaked at `584,420 KiB`. Scope remains
  `development_full_seven_scenario_diagnostic`: this is synthetic algorithm
  evidence, not biological validation, superiority, full OOF certification, or
  a default switch;
- the benchmark layer retains its exploratory categorical CR1 repeated-measures
  backend and now adds a separate `repeated_measures_cr2` finalizer option. The
  latter batches method/receiver edge matrices through the formal-ready core
  fitter and persists effect, SE, cluster, condition and leverage diagnostics
  without analytic p/q. A metadata-only audit of the 29-sample Kuppe atlas found
  all 10 region contrasts design-estimable under complete method-score coverage,
  but fitted no effects and made no biological claim. Condition contrasts with
  region adjustment were rank deficient and remain `not_estimable`;
- benchmark utilities now provide frozen-universe RBO, weighted Kendall,
  top-k curves, rank intervals, and stable-tier assignments. The v02 real-data
  finalizer/report integration requires all frozen items and fixed receiver
  strata in every subject resample. No real-data rank agreement or stable-tier
  row met that strict contract, so these endpoints remain explicit `NE` rather
  than using a favorable shared-item intersection;
- the uncapped H-common cSCC/MS campaign is complete. CRYCHIC coverage increased
  modestly to `0.3118` and `0.5150`, but its receiver-row-union views contain
  different receiver-specific functionals. The earlier global stability values
  are therefore withdrawn from the valid primary comparison and v02 fails
  closed with `receiver_child_functionals_not_globally_comparable`.

### Pre-alpha API migration

Aligned context IDs are now required wherever reference or held-out response
rows are fitted or applied. Application artifacts are producer-owned and cannot
be constructed directly; use `run_subject_crossfit()` or the corresponding
exported `fit_*`/`apply_*` producer. In-memory functional, training, application,
or cross-fit artifacts created before the v7 contracts must be discarded and
refitted so their context manifests, input digests and target-profile bindings
are regenerated.

Common-sender parameter and functional schemas are now `3.0.0`; interaction
support/gate identities use schema `3`; family-common functional/application,
cross-fit binding, and edge-evidence producers use `v2`; and score rows carry
`family_first_mechanistic_ligand_contrast_gated_softmin_v2`. Older common-sender
or family-common objects, cached folds, and persisted v1 score rows are
incompatible and must be refitted from the raw fold scope.

Cross-fit result bundles now use schema `5.0.0`. The four descriptive score
tables retain their v4 calibrated semantics, while a separate
`directional_channel_registry` records opt-in forward/reverse lineage without
changing LR scores. The loader retains explicit read-only compatibility for
schemas v1-v4; legacy bundles cannot acquire a directional registry or a
certified claim through loading.

`EXPLAINED_SHARE_V3` is available only as an opt-in attribution-support
candidate. It allocates bounded model-level explained gain across driver
contributions and suppresses tiny-response support. The public workflow default
remains `gated_prior_attribution_v1`; the baseline score remains the tracked
geometric v1 path. Candidate soft-min/mechanistic scoring helpers do not replace
that default.

The G1.5 mechanism-specificity contract is frozen in
`benchmarks/configs/mechanism_specificity_v2.json`, with component truth in
`benchmarks/truth/component_truth_matrix.yaml` and a paired-seed evaluator in
`benchmarks/metrics/mechanism_specificity.py`. The deterministic campaign has
now been run with 50 development seeds and 200 independent holdout seeds across
three known edges and seven scenarios. Both phases passed all supplied G1.5
gates; the holdout equal-edge active-minus-ligand-only margin was `0.341761`
with a 95% CI of `[0.335538, 0.347985]`. This is a synthetic mechanism-specificity
result, not evidence of real-data accuracy, and G1.5 alone cannot switch the
default method. See [the result summary](docs/results/g1-5-mechanism-specificity.md).
The current sample-keyed structural-zero generator uses the separate
`mechanism_specificity_v3.json` development contract and seed namespace. Its
holdout namespace is nonpublication test-only and cannot be promoted to an
independent holdout, so the live runner rejects `--phase independent_holdout`.

## Install

Python 3.11-3.13 is supported. From this repository:

```bash
uv sync --extra dev --extra resources --extra plotting --extra benchmark
```

## Repository Scope

This branch tracks the core package, benchmark code and configurations, tests,
schemas, method documentation, lightweight benchmark summaries, and
checksum-pinned resource manifests. Large or locally generated assets are
intentionally not versioned:

- input `.h5ad`/HDF5 matrices;
- database payloads under the workspace-level `databases/` directory;
- `benchmark_work/` intermediate and finalized run directories;
- generated `reports/` documents and figures.

The resource manifests in `resources/` identify the expected database versions
and checksums. The benchmark and report commands below require those external
assets at the paths supplied by the caller.

## Minimal Run

```python
from pathlib import Path

import anndata as ad
import crychic

database_root = Path("../databases")
bundle = crychic.load_cellchat_resource(
    database_root,
    "human",
    manifest_path="resources/cellchatdb_human_v2.json",
)
prior = crychic.load_nichenet_target_prior(
    database_root,
    manifest_path="resources/nichenet_human_v2_2021.json",
)

adata = ad.read_h5ad("cohort.h5ad")
config = crychic.CrychicConfig(
    context_keys=["condition"],
    counts_layer="counts",
    sample_key="sample_id",
    subject_key="subject_id",
    cell_type_key="cell_type",
    random_seed=20260712,
)
model = crychic.Crychic(
    config,
    resource_bundle=bundle,
    target_prior=prior,
)
plan = model.dry_run(adata)
result = model.fit(
    adata,
    output_dir="crychic_result",
    input_digest="<sha256-of-input-h5ad>",
)
```

`model.fit()` returns in-memory `BaselineArtifacts` when `output_dir` is
omitted. With an output directory it returns `CrychicResult`, which supports
validated table reads and ranked interaction queries.

For new descriptive analyses, the high-level entry point defaults to the
subject-blocked cross-fit profile. Contrasts remain explicit and are never
inferred from the observed data:

```python
from crychic.design import balanced_contrast

contrast = balanced_contrast(
    ("treated",),
    ("control",),
    name="treated_vs_control",
)
crossfit_spec = crychic.recommended_crossfit_spec(contrasts=(contrast,))
result = model.analyze(
    adata,
    spec=crossfit_spec,
    output_dir="crychic_crossfit_result",
)
```

`crossfit_descriptive_v1` is a user-entry profile, not a scientific default
switch or an inference release. It uses an uncapped interaction universe,
train-fold latent nuisance learning and subject-blocked penalty tuning. Missing
support remains typed `not_estimable`; the API never falls back to the legacy
baseline. Use `model.analyze(..., profile="legacy_v01")` or `model.fit()` only
when the historical exploratory workflow is explicitly intended.

## Reproduce Benchmarks

```bash
uv run --extra benchmark python benchmarks/run_canonical_v01.py
uv run --extra benchmark python -m benchmarks.simulation.run_negative_controls \
  --output-dir ../benchmark_work/synthetic_v01 --seed 20260712
uv run --extra benchmark python -m benchmarks.simulation.run_mechanism_specificity \
  --phase development
uv run --extra resources --extra benchmark --extra plotting \
  python benchmarks/report/generate_canonical_report.py --workspace-root ..
uv run python -m benchmarks.metrics.finalize_multicondition \
  --spec benchmarks/configs/multicondition_v02_finalize.json \
  --output-dir ../benchmark_work/multicondition_v02/final --overwrite
uv run python -m benchmarks.report.generate_multicondition_report \
  --final-dir ../benchmark_work/multicondition_v02/final \
  --output-dir reports/multicondition_v02 \
  --report-id multicondition_v02
uv run python -m benchmarks.simulation.run_crossfit_smoke \
  --workspace-root .. --scenarios active ligand_only \
  --output benchmark_work/algorithm_smoke/crossfit_summary_v5.json
uv run python -m benchmarks.simulation.run_family_common_crossfit_smoke \
  --workspace-root .. \
  --scenarios active ligand_only receiver_autonomous global_null \
  --output benchmark_work/algorithm_smoke/family_common_crossfit_smoke_v1.json
uv run python -m benchmarks.simulation.run_family_common_g15_campaign \
  --workspace-root .. --profile quick --seed-count 2 \
  --scenarios active global_null abundance_only ligand_only target_only \
  receiver_autonomous receptor_knockout \
  --output benchmark_work/algorithm_smoke/public_family_common_g15_campaign_v1.json \
  --summary-output CRYCHIC/benchmarks/results/public_family_common_g15_campaign_v1_summary.json
uv run python -m benchmarks.simulation.run_family_common_g15_campaign \
  --workspace-root .. --profile quick --seed-ids public-g15-003 \
  --scenarios active ligand_only target_only receptor_knockout \
  --output benchmark_work/algorithm_smoke/public_family_common_g15_seed003_holm_v3_final.json \
  --summary-output benchmark_work/algorithm_smoke/public_family_common_g15_seed003_holm_v3_final_summary.json
uv run python -m benchmarks.datasets.audit_kuppe_repeated_measures \
  --h5ad ../dataset/Kuppe_MI_Zenodo6578047/snRNA-seq-submission.h5ad \
  --output ../benchmark_work/kuppe_repeated_measures/design_audit.json
uv run python -m benchmarks.summarize_cscc_crossmethod_smoke \
  --workspace-root .. --repo-root . --overwrite
```

The [developer cross-fit tutorial](tutorials/developer_subject_crossfit.ipynb)
is a runnable multi-condition H5AD workflow. It validates the input contract,
loads CellChatDB/CellPhoneDB plus a NicheNet prior, constructs contrasts and
penalty tuning, persists `CrossFitResult`, queries family/LR/sender ledgers,
demonstrates optional repeated/full-pipeline resampling diagnostics, and writes
publication-ready descriptive plots. A tiny synthetic fixture runs without
external data; real paths remain caller-configured.

Persisted condition-specific outputs can be selected with
`query_family_scores()`, `query_integrated_lr_scores()`, and
`query_sender_lr_pairs()`. The integrated-LR query fixes the canonical
family-allocated LR-member component and excludes registered directional
contrasts, whose two channels use the dedicated directional result. The lower
level `query_lr_pairs()` remains available for explicit component inspection.
Queries retain structural-zero and not-estimable rows unless a status filter is
requested, and comparisons remain receiver-scoped.

Result v9 persists the exact fold-by-receiver training-support grid and the
complete run-root receiver-family opportunity universe.
`read_receiver_training_support()` returns the complete table, while
`query_receiver_training_support()` selects receiver, fold, support status, or
reason without reconstructing fitted model objects.

The same v8 bundle persists all four producer-owned semantic views as separate
Parquet tables. They can be read exactly after reload:

```python
from crychic import write_crossfit_result

persisted = write_crossfit_result(artifacts, "crossfit-result")
availability = persisted.read_semantic_availability()
receiver_program = persisted.read_semantic_receiver_programs()
integrated_lr = persisted.read_semantic_integrated_lr_scores()
differential = persisted.read_semantic_differential_effects()
view_status = persisted.semantic_score_manifest
```

The views are also available directly from intact in-memory artifacts without
refitting:

```python
from crychic import build_crossfit_semantic_scores

artifacts = model.analyze(adata, spec=crossfit_spec)
semantic = build_crossfit_semantic_scores(artifacts)
availability = semantic.availability_score
receiver_program = semantic.receiver_program_score
integrated_lr = semantic.integrated_lr_score
differential = semantic.differential_effect
```

Each view has its own biological grain, source IDs and digest, typed status and
reason, and `formal_inference_allowed=False`. Without penalty tuning, the
availability and receiver-program views remain available while integrated LR
and differential views are explicitly `not_produced` in `view_manifest`; no
zero or not-estimable score rows are invented. Result versions v1-v8 remain
read-only compatible; their canonical integrated-LR query uses the historical
ledger projection because those bundles do not contain the four exact tables.

Graph-fused conditional family effects can be joined to the family-common LR
evidence through a separate experimental adapter. The mapping keys are the
stable context IDs used by the persisted workflow, not display labels:

```python
from crychic import (
    GraphFusedIntegratedLRSpec,
    build_graph_fused_integrated_lr_scores,
    derive_graph_fused_family_effects,
)
from crychic.design import node_context_fields

effects = derive_graph_fused_family_effects(artifacts, graph_registry)
context_contrasts = {
    node_context_fields(node, config.context_keys)[0]: contrast.name
    for node, contrast in zip(graph.nodes, focal_global_contrasts, strict=True)
}
graph_scores = build_graph_fused_integrated_lr_scores(
    artifacts,
    effects,
    spec=GraphFusedIntegratedLRSpec(context_contrasts=context_contrasts),
)
family_scores = graph_scores.family_scores
lr_scores = graph_scores.integrated_lr_scores
```

This adapter is opt-in and is not connected to `analyze()` or result
persistence. Its graph-joint conditional-gain estimand is distinct from the
ordinary family differential effect; rows are descriptive only and declare
`cross_receiver_comparable=False`, `cross_context_comparable=False`,
`cross_mode_comparable=False`, and `formal_inference_allowed=False`. The graph
registry freezes a run-level receiver-by-family opportunity universe from the
complete TargetPrior and root feature axis. Training-absent receivers retain a
complete typed-NE family-effect grid with no model parents; the integrated LR
adapter fails closed because no legitimate sample-level LR rows exist for that
receiver.

The explicit active-edge workflow accepts the same H5AD and `CrossFitSpec` and
targets one declared contrast at a time:

```python
active_result = model.fit_active_probability(
    adata,
    spec=crossfit_spec,
    contrast_id_or_name="treated_vs_control",
    n_plans=200,
    n_jobs=2,  # choose from available RAM; the default is 1
    output_dir="crychic_active_probability",
)

# Without a matching passed G3-P gate this is intentionally diagnostic-only.
assert active_result.has_released_probabilities is False
```

Pass only a gate built from the preregistered calibration producer for these
exact score, universe, null, estimator and stratum contracts. Public raw-ledger
summaries are diagnostic; verified release additionally requires replayable
generator attestation. Persisting a released result also requires the exact
`G3PCalibrationResult` parent and its `CalibrationReplayRegistry`; the loader
replays that parent and recomputes the probability collection from raw
point/null rows. There is no caller-controlled release switch.

Live producer-owned `CrossFitArtifacts` also support two descriptive post-fit
views without refitting:

```python
artifacts = model.fit_crossfit(adata, spec=crossfit_spec)
signatures = model.export_crossfit_signatures(
    artifacts, mode="state", entropy_threshold=0.8
)
hypergraph = model.export_crossfit_hypergraph(artifacts)

assert signatures.inference_eligible is False
assert hypergraph.inference_eligible is False
```

The signature export preserves receiver-context, LR-attributed, and
sender-LR-receiver layers with an explicit availability audit. The hypergraph
uses subject-equal sender-resolved strengths and explicit ligand/receptor
complex nodes. Neither view contains p-values, q-values, or calibrated
communication probabilities. These methods require in-memory
`CrossFitArtifacts`; a persisted `CrossFitResult` intentionally does not
reconstruct fitted model objects.

An opt-in cross-fit configured with an explicit `directional_pairs` registry
can also emit common-scale signed Track-B target-program rows:

```python
from crychic import (
    freeze_crossfit_directional_target_program_universe,
    score_crossfit_directional_target_programs,
)

program_universe = freeze_crossfit_directional_target_program_universe(artifacts)
pair = artifacts.spec.directional_pairs[0]
signed_programs = score_crossfit_directional_target_programs(
    artifacts,
    program_universe,
    pair_spec_id=pair.pair_spec_id,
)
signed_program_table = signed_programs.scores
```

This table is source-agnostic and descriptive. It compares positive and
negative held-out OOF gene-effect channels with a complete prior-only program
universe; it is not sender attribution, LR-edge recovery, active inhibition,
or a native NicheNet result. Programs without enough matched targets remain as
typed `not_estimable` rows rather than being dropped. The receiver axis is the
frozen run axis: if a receiver is absent from any outer-training fold, its
complete program-by-channel grid is `not_estimable` and no partial-OOF effect
is fitted.

The same directional run can expose the two family-common LR applications in a
dedicated view without refitting:

```python
from crychic import build_directional_integrated_lr_scores

directional_lr = build_directional_integrated_lr_scores(
    artifacts,
    pair_spec_id=pair.pair_spec_id,
)
forward_and_reverse_lr = directional_lr.scores
receiver_opportunities = directional_lr.opportunity_registry
```

These are two independently conserved, activation-compatible LR channels on an
exact shared key grid. They are not cross-channel comparable and must not be
subtracted, divided, converted to a signed LR score, or interpreted as active
inhibition. Registered directional contrasts are omitted from the generic
semantic LR/effect row union so their direction is never implicit. The
opportunity registry covers every outer fold and run receiver. A
training-absent receiver is retained there as typed `not_estimable` with null
model/application parents; no family/LR score rows are invented without a
frozen receiver-family/LR hypothesis grid.

## Quality Checks

```bash
uv run --extra dev --extra resources --extra plotting --extra benchmark pytest -q
uv run --extra dev ruff check src tests benchmarks scripts
uv run --extra dev mypy
uv run --extra dev python -m build
```

- [Detailed development plan](DEVELOPMENT_PLAN.md)
- [v0.1 method specification](docs/methods/v0.1-exploratory-baseline.md)
- [Suggestion 1 implementation status](docs/methods/suggestion-1-implementation-status.md)
- [Train-only cross-fit boundary decision](docs/adr/ADR-004-train-only-crossfit-boundary.md)
- [Common-scale signed Track-B producer](docs/adr/ADR-014-common-scale-signed-track-b.md)
- [G1.5 mechanism-specificity results](docs/results/g1-5-mechanism-specificity.md)
- [Multi-condition v02 benchmark summary](benchmarks/results/multicondition_v02_summary.json)
- [Fixed-penalty public cross-fit smoke summary](benchmarks/results/algorithm_crossfit_smoke_v5_summary.json)
- [Trusted/tuned family-common smoke summary](benchmarks/results/family_common_crossfit_smoke_v1_summary.json)
- [Graph-fusion G2 core smoke summary](benchmarks/results/graph_fusion_g2_smoke_core_v1_summary.json)
- [Public family-common G1.5 quick summary](benchmarks/results/public_family_common_g15_campaign_v1_summary.json)
- [Developer subject-crossfit tutorial](tutorials/developer_subject_crossfit.ipynb)
- [Repository development instructions](AGENTS.md)
