# ADR-007: Receiver-balanced cross-receiver scoring collection

- Status: Accepted
- Date: 2026-07-16
- Implementation status: The v4 receiver-balanced collection producer and its
  selected-penalty inner-OOF gain calibration are implemented in the opt-in
  public subject-cross-fit workflow. Result schema v6 persists conserved sender
  allocation plus the separate directional registry and retains v1-v5 read-only
  compatibility. Its rows remain descriptive held-out/OOF diagnostics, not one
  common biological functional.

## Context

Family-common scoring produces one immutable child functional per receiver. A
child is common across the contexts in its registered contrast, but its response
model, receptor gates, family coefficients, and within-family and sender
allocation denominators are receiver-specific. As established by ADR-005,
row-unioning those child tables does not create a common cross-receiver
estimand. In particular, values normalized within a receiver's supported family
members or candidate senders change when that local candidate set changes.

The collection therefore adds a train-only descriptive rank calibration without
claiming a common response functional. For each receiver it derives an ECDF
from selected-penalty inner-OOF subject-family prediction gains and applies that
frozen mapping to held-out gains. One producer-owned parent freezes the complete
receiver universe and applies one documented formula without reusing local
allocation weights or fitting on held-out rows. This supports auditable
receiver-stratified and equal-weight macro diagnostics while preserving the
outer subject split, explicit missingness, and the ban on held-out rescaling.

## Decision

### Parent grain and shared collection contract

CRYCHIC builds exactly one code-named
`CrossReceiverCommonScoringFunctional` collection parent for each
registered contrast and physical outer fold. Repeated runs retain their normal
repeat identity outside that fold-local parent. The collection parent consumes
all receiver-specific `FamilyCommonScoringFunctional` children and accepts them
only when their receiver IDs exactly equal the non-empty, unique
`planned_receiver_ids` universe.

Every child must share the same contrast and manifest, fold, context IDs,
training subjects, filter universe, common-sender functional, feature IDs,
family IDs, and interaction-to-family/driver mapping. Observed incremental
parents must additionally share the downstream loss design, family-gain
estimand, response-basis coordinate transform, context regressor, nuisance
design, minimum scale, and null-loss floor. Soft-min parameters and score
versions must also agree.

The parent identity binds this shared contract, the complete ordered-canonical
receiver and child-functional manifests, and the complete receiver-to-
calibration bindings, including artifact/spec IDs, status/reason, source and
percentile knots, support counts, tuning ID, selected resolved penalty, and
outer incremental functional ID.
The application identity binds the exact child application IDs, common-sender
application digests, held-out subjects, output row counts, and order-independent
table digests. Producer integrity checks recompute these identities.

Shared provenance, formula, and row coverage do not make the receiver-specific
response models one functional or prove a common numeric scale. The released
scientific contract is therefore:

```text
common_functional_across_receivers = false
comparability_scope = within_receiver_across_contexts_only
cross_receiver_percentile_rank_eligible = all_receiver_calibrations_observed
```

The `CrossReceiverCommon*` class names and `global_*` column names are schema
identifiers retained for compatibility; they do not override these semantics.

Application requires every receiver child to cover the same exact held-out
grid:

```text
sample x subject x context x family x driver x interaction x mode
```

All planned receivers must remain present. Missing evidence is represented by
explicit `not_estimable` rows; it is not repaired by dropping a receiver or a
row. Sender applications must likewise cover the complete planned receiver
universe and exactly the held-out LR groups for which frozen sender candidates
exist. Training and held-out subject IDs must be disjoint.

### Receiver-balanced LR and sender-LR rows

The collection contains receiver-relative mechanistic evidence over a frozen
receiver universe. Each row remains conditional on its receiver-specific
response model and null loss; it is neither an absolute communication rate nor
a probability. For an eligible interaction, the receiver's frozen train-only
mapping is applied before the interaction components are integrated:

```text
calibrated_family_gain_percentile = receiver_inner_oof_ecdf(
    receiver_relative_family_gain
)
global_lr_core = softmin(
    absolute_interaction_availability,
    calibrated_family_gain_percentile,
)
global_lr_score = global_lr_core * fixed_prior_quality
```

Here "absolute" means that interaction availability is used directly rather
than normalized across supported members. The receiver-relative gain is the
bounded held-out loss reduction produced by the already-frozen receiver child.
Prior quality is a multiplier with one as its neutral value. The current child
`within_family_lr_weight` remains a decomposition diagnostic and never enters
the receiver-balanced LR row. The normalized sender `assignment_weight` does
not change that parent LR row; it allocates the parent across frozen senders.

Sender-resolved collection scoring uses the common-sender functional's frozen
sample-local softmax assignment over the complete candidate set:

```text
raw_sender_evidence = (
    heldout_ligand_availability * training_prevalence_prior
)
assignment_weight = frozen_softmax(raw_sender_evidence)
global_sender_lr_score = global_lr_score * assignment_weight
sum_sender(global_sender_lr_score) = global_lr_score
```

Candidate membership is frozen from the outer-training contract. Changing that
candidate set would change both the functional identity and its allocation, so
applications cannot add or remove a sender. Raw evidence remains a diagnostic;
the primary sender score is a conserved decomposition of its receiver-specific
LR parent. Conservation does not imply comparability with another receiver.

### Selected-penalty inner-OOF gain calibration

Calibration uses only outer-training subjects. After subject-blocked penalty
tuning selects one candidate, CRYCHIC retains that candidate's exact inner-fold
fit/apply parents. Every outer-training subject is an inner validation subject
exactly once. The artifact recomputes bounded subject-family loss reduction
from the held-out null and family losses and builds a right-ECDF over finite
positive observations from supported families. Zero maps exactly to zero;
missing remains missing; positive values are mapped by zero-preserving
piecewise-linear interpolation. No held-out outer-fold value, label, rank, or
candidate count enters the mapping.

The production support policy requires at least two inner folds, eight subjects,
three supported families, four subjects per family, twelve positive
subject-family observations, and five distinct positive gains. Smaller
explicit policies may be used only by versioned synthetic or small-scale smoke
tests and are identity-bound. Insufficient support produces a typed
`not_estimable` artifact. A positive held-out gain under missing or
not-estimable calibration remains `not_estimable`; it never falls back to the
raw gain, a coefficient multiplier, or factor one.

This ECDF removes direct coefficient-magnitude scaling from the primary score
and provides a common unit-interval percentile convention. It still does not
make receiver-specific response models, null distributions, or biological
functionals identical. `cross_receiver_percentile_rank_eligible=true` means
only that every planned receiver has an observed train-only mapping. It does
not authorize pooled raw-score AUROC/AUPRC, a global top-k, formal inference, or
a method-superiority claim.

### Status precedence and fail-closed rules

Structural absence has priority over later missing downstream evidence. The LR
row precedence is:

1. Receptor ineligibility, an observed unsupported ligand-contrast gate, a
   non-positive estimable training family coefficient, zero interaction
   availability, or zero family gain produces an exact structural zero.
2. If no earlier zero applies, a not-estimable ligand gate, family,
   availability, gain, or gain calibration propagates `not_estimable`.
3. Once the calibrated LR core is estimable, a missing prior propagates `not_estimable`
   and a zero prior produces structural zero; otherwise the row is observed.

At sender grain, a structural-zero LR row remains zero even when sender
evidence is missing and cannot be revived. A not-estimable LR row remains not
estimable. For an observed LR row, a missing or incomplete frozen assignment
makes the whole sender group `not_estimable`; a zero assignment weight is an
exact structural zero; otherwise the score is `global_lr_score x
assignment_weight`. Complete weights and scores must each sum to their frozen
group totals.

Collection fitting fails closed for incomplete receiver coverage, incompatible
child or training-scale contracts, incomplete common-sender receiver coverage,
and mismatched score parameters. It also rejects any child with
`family_selection_threshold != 0`. A held-out family-selection cut would make
the reported universe data-dependent; any reporting threshold must instead be
applied after collecting the complete OOF table.

Collection application fails closed for stale children, different held-out subject
scopes, train/test subject overlap, unequal held-out grids, incomplete sender
coverage, duplicate keys, or identity/table tampering.

### Interpretation and compatibility

One fold application is a held-out diagnostic and does not certify itself as a
complete OOF analysis. When all subject-disjoint folds are present and the
authoritative cross-fit coverage and lineage audit passes, the collection may
be used for descriptive OOF receiver-stratified analyses. Rows remain neither
probabilities, p-values, q-values, confidence intervals, causal effects, nor
formal repeated-pipeline inference. Persistence forbids those inferential
fields for this score path.

Supportive benchmarking may compute AUROC, AUPRC, top-k recovery, or rank
diagnostics separately within each receiver using a predeclared receiver-local
hypothesis universe. A macro summary may then average those receiver-level
metrics with predeclared equal receiver weights and explicit propagation of
not-estimable strata. Such a macro value is supportive aggregation of
within-receiver endpoints, not evaluation of one pooled score scale. Adapters
must not concatenate raw rows across receivers before computing AUROC/AUPRC,
selecting a global top-k, or comparing score magnitudes.

ADR-005 remains in force for receiver-child collections. Rows emitted by the
same collection parent share provenance and coverage, but do not carry a common
functional claim: `common_functional_across_receivers=false`. Row-unioning
legacy or current receiver-child family-common tables remains forbidden for
cross-receiver ranking. Existing receiver-local tables remain useful
diagnostics and retain their original identities and semantics. Result schema
v6 and contrast-common sender table schema v3 store the calibration and
assignment lineage needed to recompute every released value. Result schemas
v1-v5 are read-only; v3 coefficient-scale fields and v4/v5 raw-evidence sender
multiplication are interpreted only by private legacy readers and must never be
mixed with current rows. The directional registry is a separate lineage table
and does not enter this score formula.

## Consequences

The public opt-in cross-fit path can preserve every planned receiver in one
identity-bound LR and sender-LR collection per contrast/fold. The LR parent is
independent of local family-member and sender normalization, while frozen sender
weights now conserve that parent exactly. Train-only selected-penalty inner-OOF
calibration replaces coefficient-scale shrinkage, and explicit structural-zero/
not-estimable precedence prevents missing evidence from becoming positive
support.

The implementation supports auditable receiver-stratified and macro supportive
benchmarks. It does not supply a strictly common cross-receiver scale, justify
pooled global AUROC/top-k analysis, validate biological recovery, establish
method superiority, authorize a default-method switch, or replace the separate
G3 inference work.
