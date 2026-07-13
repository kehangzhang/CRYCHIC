# CRYCHIC

CRYCHIC: Cellular Relational dYnamics for Contextual Hypergraph Inference of Communication

CRYCHIC currently provides a runnable v0.1 exploratory baseline for
sample-aware cell-cell communication analysis. It validates count or explicitly
declared normalized AnnData, aggregates sample x cell-type pseudobulks, builds
balanced context contrasts, scores LR availability, optionally attributes
receiver responses with a versioned NicheNet prior, and writes atomic,
queryable results.

This is not the completed roadmap or a calibrated inference release. v0.1
strengths are descriptive and in-sample; p-values, q-values,
`comm_probability`, posterior quantities, and causal sender claims are disabled.

## Current Development Status

The current branch adds more explicit evidence and provenance contracts while
preserving the exploratory v0.1 behavior:

- downstream output separates the source-agnostic `receiver_program_score`
  from `incremental_downstream`. Training/application primitives for held-out
  receiver-null loss gain and subject-fold contracts are available, while the
  public workflow still reports this field as `not_estimable` until every
  upstream data-driven transform is fitted inside the training fold;
- attribution now consumes a producer-owned v2 winsorized, median-normalized
  response-precision artifact whose identity binds the raw/transformed ordered
  feature vector and receiver/contrast/fold scope. The exploratory solver path
  validates artifact compatibility and exact consumed weights; training-subject
  provenance remains a public cross-fit integration requirement;
- low-level incremental downstream v2 uses explicit sample-subject-context row
  manifests, keyed response/design alignment, full training-input digests,
  sample diagnostics and technical-row-mean/equal-context/equal-subject
  held-out losses, signed diagnostic gains and bounded score gains. It rejects
  training sample/subject reuse, freezes the contrast context universe, and
  requires caller-declared design IDs to match. Authentic design lineage still
  requires direct `FrozenDesignApplication` integration; the primitive remains
  partial until autonomous nuisance, inner tuning and public cross-fit coverage
  are connected;
- opt-in attribution candidates provide hard receptor eligibility, directional
  response channels, strict family-first bases, and evidence-weighted member
  allocation without changing the legacy default;
- tracked scores carry a `score_version` and `model_manifest_id` covering the
  fitted basis, coefficient digest, receptor gates, target weights, sender and
  downstream functionals, availability and precision transforms, filtering,
  and tuning artifacts;
- integrated result directories now persist a versioned
  `scoring_collections.json` registry. Each contrast/repeat/fold collection
  enumerates only the emitted receiver-specific scoring functional IDs and
  source-score key digests. It is marked `partial_emitted_only`, records
  `common_functional_across_receivers=false`, and does not claim a complete
  planned receiver universe or verified persisted model metadata;
- availability state and ecosystem eligibility have separate status and reason
  fields, and missing sender evidence no longer receives an implicit uniform
  assignment;
- fold manifests enforce subject-blocked train/test separation, frozen
  interaction universes prevent test-fold filtering leakage, and the H-common
  benchmark arm rejects data-driven top-k interaction caps;
- an edge-evidence ledger records component status, missingness reasons and
  provenance at sample x context x sender x receiver x interaction grain. It
  can be persisted as an optional, versioned Parquet result extension with
  `persist_edge_evidence=True`;
- the public `run_subject_crossfit` API now derives estimability-aware subject
  folds, creates physical sanitized train/test scopes, calls the producer-owned
  train/apply primitives, and audits exact OOF coverage for the frozen
  interaction universe and contrast-common sender functional. Fold artifacts
  also freeze nuisance encoders, condition-blind hard receptor gates, strict
  medoid family bases, and training-reference receiver-program transforms,
  then apply them without refitting. Planned receiver coverage gaps are explicit
  `not_estimable` results. These receiver artifacts do not yet have their own
  exact-coverage audit; response precision, family attribution/tuning,
  incremental downstream gain, and common scoring remain unconnected. The
  aggregate status is `verified_train_only_oof_partial_pipeline`, while
  `is_oof_certified` remains false and low-level rows retain `partial_not_oof`;
- frozen design encoding now uses the complete declared Patsy formula and an
  exact EMM-contrast reparameterization. Numeric-coded categorical covariates
  can be declared with `categorical_covariates`; held-out levels are applied
  through training `DesignInfo` without refitting, and sample/design matrices
  participate in artifact identity;
- strict complete-link family clustering uses an output-equivalent incremental
  minimum-update heap and shares the static partition once per fold feature
  universe. The v3 partial smoke completed active and ligand-only controls in
  11.86 and 11.11 seconds, and a 10,057-cell Kang subset in 26.28 seconds;
  these are pipeline and performance checks, not integrated-edge accuracy or
  biological claims;
- the benchmark layer includes a frozen categorical repeated-measures backend
  for mixed paired/unpaired subjects and within-subject technical replicates.
  A metadata-only audit of the 29-sample Kuppe atlas found all 10 region
  contrasts design-estimable under complete method-score coverage, but fitted
  no effects and made no biological claim. Condition contrasts with region
  adjustment were rank deficient and remain `not_estimable`;
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

## Reproduce Benchmarks

```bash
uv run --extra benchmark python benchmarks/run_canonical_v01.py
uv run --extra benchmark python -m benchmarks.simulation.run_negative_controls \
  --output-dir ../benchmark_work/synthetic_v01 --seed 20260712
uv run --extra benchmark python -m benchmarks.simulation.run_mechanism_specificity \
  --phase development \
  --output-dir benchmark_work/g1_5_v2/development
uv run --extra benchmark python -m benchmarks.simulation.run_mechanism_specificity \
  --phase independent_holdout \
  --output-dir benchmark_work/g1_5_v2/independent_holdout
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
  --workspace-root .. --include-kang-subset \
  --output ../benchmark_work/algorithm_smoke/crossfit_summary_v3.json
uv run python -m benchmarks.datasets.audit_kuppe_repeated_measures \
  --h5ad ../dataset/Kuppe_MI_Zenodo6578047/snRNA-seq-submission.h5ad \
  --output ../benchmark_work/kuppe_repeated_measures/design_audit.json
```

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
- [G1.5 mechanism-specificity results](docs/results/g1-5-mechanism-specificity.md)
- [Multi-condition v02 benchmark summary](benchmarks/results/multicondition_v02_summary.json)
- [Repository development instructions](AGENTS.md)
