# ADR-004: Train-only cross-fit boundary

- Status: Accepted
- Date: 2026-07-13
- Implementation status: Partial; the public orchestrator now verifies
  subject-blocked OOF coverage for the frozen interaction-universe and
  contrast-common sender stages. It also preserves train-only frozen design,
  receptor-family and reference-program artifacts, but these have no separate
  exact-coverage audit and incremental downstream/common scoring remain absent

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

Training performs, in order:

1. pseudobulk construction and training-only expression support filtering;
2. interaction selection, optional cap, and receptor/path eligibility;
3. continuous response and precision-transform fitting;
4. strict family construction and frozen target basis fitting;
5. nuisance encoder and inner-training parameter selection;
6. incremental downstream model fitting;
7. a contrast-common sender functional;
8. an aggregate fold/contrast scoring manifest.

Application performs only sample-local fixed count transforms, frozen-universe
availability, frozen gate and design encoding, downstream and sender
application, and common-functional scoring.

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
gates, strict complete-link/medoid family
bases, and a reference-arm receiver-program transform. Held-out application
uses the frozen feature order and transform; subject overlap, feature drift,
unseen nuisance levels and incomplete receiver coverage fail closed. These
receiver artifacts are retained in the fold manifest but do not yet have a
separate exact-coverage table. Held-out sender rows
retain their low-level `partial_not_oof` assignment mode; the aggregate result
separately reports `verified_train_only_oof_partial_pipeline` and
`is_oof_certified=false`. This distinction verifies the implemented stages
without certifying the complete method.

Response precision, fitted family attribution/tuning, incremental downstream
gain, a common scoring manifest, and exact receiver-stage coverage still remain.
Until that complete path exists and passes every poison test above, incremental
downstream evidence remains `not_estimable` in the public baseline workflow and
no default-method switch is eligible. The legacy sender assignment remains
unchanged and context-specific.

The low-level precision and incremental primitives were hardened before their
public integration. Precision v2 is producer-owned and binds the raw and
transformed ordered feature vector plus receiver/contrast/fold scope. The
incremental v2 primitive requires explicit sample-to-subject-to-context row
manifests and independently keyed design rows, hashes every raw training input,
rejects held-out reuse of training samples or subjects, reports signed and
bounded gains, requires matching nuisance/regressor lineage, averages technical
rows within subject/context before squared loss, and then weights contexts and
subjects equally. The low-level design check is only a caller-declared ID
assertion; authentic lineage requires the future public consumer to accept a
producer-owned `FrozenDesignApplication` from the same encoder. These contracts
make direct research calls auditable, but they do not prove OOF provenance:
neither primitive is accepted as caller input by the public workflow, and
incremental v2 remains
`partial_not_oof_certified` until training response precision, autonomous
nuisance, inner tuning and exact receiver coverage are connected inside the
physical fold boundary.

The nuisance encoder reparameterizes the full training Patsy matrix as
`[X N, X q]`, where the nuisance basis satisfies `l^T N = 0` and the contrast
direction satisfies `l^T q = 1`. Consequently, the final regression coefficient
is exactly the declared equal-context EMM contrast even for unbalanced sample
counts and local contrasts over a larger context universe. Training sample
mappings, full design matrices, formula/type registries and input key schema
are hashed into the producer-owned identity. Held-out application uses the
training `DesignInfo` and fixed reparameterization only.
