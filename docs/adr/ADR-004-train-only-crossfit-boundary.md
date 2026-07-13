# ADR-004: Train-only cross-fit boundary

- Status: Accepted
- Date: 2026-07-13
- Implementation status: Partial; the public orchestrator verifies the frozen
  interaction/common-sender stages and exact planned receiver-row coverage for
  a typed design/response/precision/incremental/family-common chain. A trusted
  registered autonomous resource plus estimable paired subject-blocked tuning
  can now produce officially observed incremental children. Held-out
  family-common applications are connected behind that gate but remain
  noncertifying diagnostics. The source-agnostic receiver-program parent,
  independent/mixed inner tuning and full-pipeline inference remain absent.

## Context

CRYCHIC has subject-blocked fold planning, frozen interaction-universe
contracts, incremental downstream fit/apply primitives, and out-of-fold table
validators. Those pieces do not by themselves prove that every data-derived
artifact was learned from training subjects only. A caller can otherwise pass
precomputed matrices or provenance labels derived from the complete data and
still produce an output table that looks out of fold.

The certified workflow therefore needs a physical data boundary, not only
matching identifiers. This boundary must cover filtering, receptor gates,
family construction, precision transforms, nuisance encoders, tuning,
downstream models, sender functionals, and scoring manifests.

## Decision

The only public cross-fit entry point will accept raw observations and
pre-registered, data-independent inputs:

```python
def run_subject_crossfit(
    adata: AnnData,
    config: CrychicConfig,
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
    *,
    spec: CrossFitSpec,
) -> CrossFitArtifacts:
    ...
```

It will not accept a `FoldPlan`, subject identifiers, response or availability
matrices, fitted artifacts, family bases, scoring functionals, or caller-made
provenance identifiers. The orchestrator derives folds from declared metadata
and creates physical, sanitized training and test `AnnData` copies. These
copies retain only the declared count matrix, required observation columns,
and variable names; they do not retain `raw`, `uns`, embeddings, or parent
views.

An internal training scope derives its subject and sample identifiers from its
own observations. Every data-driven fit runs inside that scope. Application
receives only immutable training artifacts plus a held-out scope and may call
fixed transforms and apply functions, but no fit, pooling, clustering,
winsorization, or tuning function.

Certified out-of-fold status requires raw counts. The current public cross-fit
entry point rejects normalized-only input because CRYCHIC cannot audit how its
upstream normalization was learned. Low-level exploratory paths may consume a
declared normalized matrix, but cannot claim train-only provenance.

## Fold stages

The base training path performs, in order:

1. pseudobulk construction and training-only expression support filtering;
2. interaction selection, optional cap, and receptor/path eligibility;
3. continuous response and precision-transform fitting;
4. strict family construction and frozen target basis fitting;
5. nuisance encoder and fixed-penalty incremental diagnostic fitting;
6. a contrast-common sender functional;
7. an aggregate fold/contrast scoring manifest.

Supplying `PenaltyTuningSpec` additionally enables conditional subject-blocked
inner tuning, selected-penalty incremental fitting and family-common
construction. The current producer-owned one-SE path supports fully paired
subjects. Independent and mixed allocations emit typed `not_estimable` tuning
artifacts instead of falling back to caller-authored losses.

Application performs only sample-local fixed count transforms, frozen-universe
availability, frozen gate and design encoding, downstream and sender
application, and, when enabled, family-common diagnostic scoring.

A fold/contrast may contain multiple receiver models. Its scoring collection
hashes the sorted receiver child manifest identifiers and explicitly declares
`common_functional_across_receivers=false`. This is a composition registry, not
a claim that distinct receiver models are one scoring function. Persisted
source-key digests validate the emitted rows, while
`composition_status=partial_emitted_only` makes clear that no planned receiver
universe or authoritative child model registry is persisted yet.

## Globally frozen inputs

The following may be fixed before folds are observed: resource and target
prior checksums, namespace maps, static complex definitions, the explicit
context graph and contrasts, predefined nuisance gene programs, formulas,
threshold candidates, component weights, fold policy, and seed lineage.

The following must never be learned from the complete dataset: expression or
interaction support, a top-k cap, receptor gates, family definitions,
precision winsorization, selected penalties, attribution coefficients, latent
nuisance factors, downstream centering/scaling/models, sender prevalence or
temperature, and data-driven component calibration.

## Acceptance tests

The workflow is not certified until all of these tests pass:

- Altering only test-subject expression leaves every training artifact ID
  unchanged while allowing held-out scores to change.
- Test-only poison values that would alter a global filter, gate, precision
  transform, family, downstream model, or sender model do not alter training
  manifests.
- Monkeypatching every fit function to raise during application does not stop
  application.
- Each training stage observes exactly its fold's training subjects.
- Public signatures reject precomputed matrices, artifacts, IDs, and claimed
  provenance.
- Every subject appears in exactly one test fold and every score has one
  aggregate and one receiver-child manifest.
- An unseen held-out categorical level becomes `not_estimable`; application
  does not refit the encoder.
- Normalized-only input cannot produce certified out-of-fold status.

## Alternatives considered

Allowing callers to supply `TrainingArtifacts` or a claimed list of training
subjects was rejected. Digests can prove object identity after construction,
but cannot prove that the object was not fitted on held-out observations.

Validating only the final score table was also rejected. Exact fold coverage
is necessary, but it audits output bookkeeping rather than stage-level data
access.

## Consequences

The public API is intentionally narrower than the low-level research APIs.
Advanced users may still call primitives directly, but their results remain
exploratory and cannot be relabeled as certified OOF.

The partial implementation now includes a public `run_subject_crossfit` entry
point. It derives an estimability-aware fold plan from declared sample metadata,
creates physical sanitized train/test copies, calls the producer-owned
`fit_training_artifacts` and `apply_training_artifacts` primitives per fold,
and audits exact subject/fold/contrast/context coverage for availability and the
common-sender stage. The same physical boundary now fits a frozen nuisance
encoder from the complete declared formula, condition-blind hard receptor
gates, strict complete-link/medoid family bases, a reference-arm receiver
program, fold response and response-parented precision. A typed adapter derives
design rows, reference masks and eligible family bases without caller matrices
or IDs. Held-out application uses the exact parent chain; subject overlap,
feature drift, unseen levels and incomplete receiver coverage fail closed. A
separate table verifies each planned fold/contrast/receiver/sample row.
Held-out sender rows
retain their low-level `partial_not_oof` assignment mode; the aggregate result
separately reports `verified_train_only_oof_partial_pipeline` and
`is_oof_certified=false`. This distinction verifies the implemented stages
without certifying the complete method.

The incremental gain remains a candidate development estimand. Public cross-fit
may project either a typed caller-declared static receiver-autonomous resource
or a code-reviewed registration. The caller-declared constructor cannot prove
the supplied matrix came from a trusted preregistered manifest and therefore
remains `not_estimable` with `receiver_autonomous_nuisance_not_frozen` at the
official gate. The registered loader binds manifest and payload identity,
canonical matrix digest, license, review scope, release, species, namespace and
the static TSV schema, and emits `manifest_verified_static_trusted_v1`.
Cross-fit now connects that artifact to paired subject-blocked inner tuning;
when every parent is estimable, incremental training and held-out application
receive official `observed` status. Independent and mixed-subject one-SE tuning
still fail closed. No default-method switch is eligible.

The low-level precision and incremental primitives were hardened before their
public integration. Precision v3 binds the raw and transformed ordered feature
vector, residual degrees of freedom, downstream feature scale, and the exact
receiver/contrast/fold parent scope. Low residual df and missing-scale cases use
equal supported-feature weights; higher-df weights are converted to standardized
response units before winsorization and median normalization. Incremental
application requires explicit sample-to-subject-to-context manifests and
independently keyed design rows,
hashes every raw training input, rejects held-out sample or subject reuse,
reports signed and bounded gains, averages technical rows within
subject/context, supports paired contrasts and frozen independent-group
pseudocontrasts, and records a zero receiver contrast as an explicit structural
zero rather than an undefined ratio. Direct calls
still rely on caller-declared design IDs and remain exploratory. The public
adapter closes that lineage gap by accepting only producer-owned encoder,
fold-response, response-precision and receiver-family parents and by deriving
all matrices and IDs internally. Deterministic relative-penalty and one-SE
selection artifacts now exist for ordinary and signed residual spaces;
caller-authored fold losses remain noncertifying. The public paired branch now
records producer-owned inner evaluations and refits response centering/scaling,
nuisance projection, the solver and penalty scale within each inner training
split. Precision, the encoder and the family basis remain outer-frozen, so the
claim is explicitly conditional on the outer-frozen representation rather than
fully nested feature discovery.

Candidate selection uses subject-paired candidate-minus-empirical-best losses.
A candidate is eligible when its mean paired excess loss is no greater than the
sample-SE of those subject differences. Eligible candidates use an explicit
L1-fraction-first sparsity priority and L2-fraction second; this is a declared
policy order, not a total physical ordering of cross-axis elastic-net strength.
This v2 rule removes common subject-difficulty variance that can otherwise admit
the L1 zero-solution boundary. It is a descriptive selection heuristic, not a
confidence interval, noninferiority test or coverage claim.
The empirical best is selected on the same grid and subjects within an inner
fold share a fitted training model, so later simulations must assess grid-size
and selection-frequency sensitivity.

The opt-in path also composes the selected incremental parent with frozen
families and the contrast-common sender functional. It emits fold-family
attribution, held-out subject-family differential effects, state/ecosystem
family and member scores, and sender-resolved allocation. Exact application
coverage and producer-owned bindings include the training application, tuning,
selected penalty, autonomous resource, canonical edge evidence and
receiver-specific sender inputs. Sender weights only allocate an established
member score. These family-common outputs remain diagnostic rather than OOF-
certified. The source-agnostic receiver-program parent remains explicitly
`not_estimable`, and the complete pipeline therefore remains uncertified.

The nuisance encoder reparameterizes the full training Patsy matrix as
`[X N, X q]`, where the nuisance basis satisfies `l^T N = 0` and the contrast
direction satisfies `l^T q = 1`. Consequently, the final regression coefficient
is exactly the declared equal-context EMM contrast even for unbalanced sample
counts and local contrasts over a larger context universe. Training sample
mappings, full design matrices, formula/type registries and input key schema
are hashed into the producer-owned identity. Held-out application checks the
frozen level registry, rebuilds the restricted formula, requires the exact
training column contract and applies the frozen reparameterization. Mutable
Patsy `DesignInfo` is not trusted.
