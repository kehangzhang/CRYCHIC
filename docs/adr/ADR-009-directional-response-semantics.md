# ADR-009: Directional response semantics

Status: Accepted for the opt-in diagnostic workflow

## Context

CRYCHIC currently consumes non-negative ligand-target priors. Such a prior can
support an activation-compatible explanation of a positive receiver response,
but it cannot by itself distinguish reduced activation from active molecular
repression. Fitting an unconstrained negative coefficient would therefore make
a biological claim that the resource does not identify.

The receiver-program reference transform is fitted from the negative-weight
side of each contrast. Explicit forward and reverse contrasts consequently use
different reference transforms even when their raw gene effects are exact
negatives. Their downstream scores are not automatically on a common signed
scale.

## Decision

Directional analysis is opt-in through an explicit
`DirectionalContrastPairSpec`. The registered reverse contrast must:

- have a distinct name;
- share the forward contrast family, mode, and context support; and
- exactly negate every forward contrast weight.

The workflow emits two independent channels:

- `increased_activation_compatible` for the forward contrast; and
- `reduced_activation_compatible` for the explicit reverse contrast.

Every planned `fold x run receiver x pair` remains in a producer-owned
opportunity registry. An outer-training-supported receiver has one
`DirectionalCrossFitBinding` over the exact response, incremental training,
and held-out application parents. Both channels must share receiver, fold,
features, frozen family parent and axis, training and held-out rows, subjects,
and input lineage. Observed raw effects must negate each other, while their
standard errors and raw precision must be symmetric. Any unavailable component
makes the complete pair typed `not_estimable`.

Before fold planning or response fitting, a run-root
`FrozenReceiverFamilyLRHypothesisUniverse` freezes the complete
`receiver x uniquely mapped LR x mode` opportunity axis. Its family partition
uses only the target prior and root feature axis; its LR membership uses only
the complete external `ResourceBundle` and `TargetPrior`. Mapping is explicit:
every resource interaction is uniquely mapped, unmapped, or ambiguous, and all
membership, hypothesis, receiver-opportunity, and universe identities are
stable and independently replayable.

Version 2 keeps molecular identity orthogonal to the target-profile family.
`molecular_lr_equivalence_id` is resource-record independent and binds only
species, canonical gene namespace, ligand-to-receptor direction, and the exact
unordered ligand/receptor component sets. The existing `family_id` remains a
TargetPrior profile family. Optional agonist, antagonist, and coreceptor sets
form a separate mechanistic-variant identity. The complete molecular universe,
source crosswalk, upstream mapping report, and axis IDs are embedded in the
run-root LR universe and replayed by result-v8 persistence; historical v1-v7
directional universes remain read-only compatible.

An outer-training-absent receiver instead has no fabricated response, model,
application, directional-binding, or family-common parent. Its opportunity row
and every held-out sample x frozen LR x mode row are typed `not_estimable` with
`receiver_absent_in_outer_training`. Both channels retain the exact same frozen
key grid, all fitted/model parents and source values are null, and no row is
relabeled as a structural zero. Supported receivers must reproduce the same
static grid exactly and retain their original family-common source values and
parent lineage.

The combination rule is
`independent_contrast_views_not_additive_v1`. The following are always false:

- `supports_active_inhibition_claim`;
- `active_inhibition_allowed`;
- `paired_score_comparison_allowed`; and
- `formal_inference_allowed`.

The two channel scores must not be added, subtracted, or relabeled as a signed
communication score. Reverse-channel evidence means attenuation or reduced
activation compatible with the explicit reverse contrast, not active
inhibition.

`DirectionalIntegratedLRCollection` is the dedicated descriptive LR view. It
reuses the exact forward and reverse family-common held-out applications,
requires the run-root frozen LR key grid, conserves real source allocations in
each channel independently, and binds the complete receiver-level opportunity
registry and frozen opportunity axis into collection identity. It does not
refit, rescale, subtract, divide, rank one channel against the other, or produce
an inhibition field. Generic semantic LR/effect views exclude registered
directional contrasts so their direction cannot be lost in an unlabelled row
union.

Directional pair specifications and bindings participate in cross-fit identity
only when the feature is enabled. The default non-directional specification and
cross-fit identity remain unchanged. Persisted directional lineage belongs in
a separate registry rather than in the existing LR score tables. The dedicated
collection is persisted as an atomic sidecar bound to its exact result-v8
parent. Loading replays the frozen LR universe, support, design, directional
registry, and family-common component semantics; missing, mutated, or
coherently rehashed lineage fails closed.

## Consequences

This closes the semantics and provenance of an opt-in forward/reverse
diagnostic without changing the primary LR scoring formula. ADR-014 separately
defines a source-agnostic common-scale target-program estimand from the shared
held-out raw response applications. Track-B release still requires complete
multi-seed evaluation and its independent release decision.
