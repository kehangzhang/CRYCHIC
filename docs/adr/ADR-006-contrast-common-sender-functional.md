# ADR-006: Contrast-common sender functional

- Status: Accepted
- Date: 2026-07-13
- Implementation status: Common-sender v3 is implemented in the public
  train/apply workflow and consumed by family-common scoring v2. The resulting
  family/member/sender tables remain noncertifying diagnostics.

## Context

The v0.1 sender assignment aggregates availability and prevalence separately
inside each context. That is useful as an exploratory descriptive table, but it
does not provide one sender function shared by all contexts in a differential
contrast. Sender evidence also contains ligand availability, so it must not
define a primary sender-unresolved LR score or be interpreted as a causal
source.

The target-only G1.5 control also exposed a separate failure mode. Repeating a
family-level downstream gain across held-out interaction rows and combining it
with sample-local absolute ligand availability can produce an integrated score
without evidence that sender-side ligand availability changed in the declared
contrast. A train-only, contrast-common interaction gate is therefore required
before family-common integration.

## Decision

### Frozen interaction and sender universes

The producer-owned fold training path accepts an intact
`FrozenInteractionUniverse`, not a caller-authored interaction ID list. The
universe binds the complete outer-training interaction IDs, training subjects,
resource identity and checksum, pooled-support threshold, optional exploratory
cap, selection policy, and `filter_universe_id`.

Before common-sender fitting, the workflow also freezes one canonical candidate
manifest row for every receiver x frozen interaction. Each row contains a
non-empty, sorted tuple of candidate sender IDs. A receiver binds the complete
frozen interaction universe even when one interaction has no local training
row; in that case the receiver-wide frozen sender set supplies the candidate
tuple. Training rows outside this manifest are rejected. The functional binds
the manifest, exact canonical training-availability digest, and sanitized raw
training-input digest.

### Interaction-level ligand-contrast support

Common-sender v3 fits one producer-owned
`InteractionLigandContrastSupport` for every frozen receiver x interaction:

1. Within receiver x interaction x sender x subject x context, repeated
   sample/technical rows are averaged.
2. Within receiver x interaction x subject x context, availability is then
   maximized across the frozen candidate senders. Senders are not tested
   separately; the complete candidate tuple is frozen and identity-bound so the
   max operation cannot silently change between fitting and application.
3. A subject contributes only when every context in the registered contrast is
   finite. The original contrast weights are never renormalized around missing
   contexts, and every complete subject receives equal weight.
4. The subject contrast is the registered weighted sum of those context
   availabilities. Estimation requires at least
   `max(2, parameters.min_subjects)` complete subjects.
5. An estimable interaction receives a one-sided Student-t test of mean subject
   contrast above `ligand_contrast_minimum_effect`. Zero sample variance gives
   raw p=0 only when the mean is strictly above that minimum effect; otherwise
   it gives raw p=1.
6. Within each outer-training fold x receiver x contrast, every frozen
   interaction belongs to one Holm step-down family. Not-estimable interactions
   remain in the family with effective p=1, so they remain in the multiplicity
   denominator `m`. Raw-p ties are broken deterministically by interaction ID.
7. Support is `supported` only when the Holm-adjusted p-value is strictly below
   `1 - ligand_contrast_confidence_level`. An estimable interaction that misses
   that threshold is `unsupported`; insufficient complete-subject support is
   `not_estimable`.

The support and gate identities bind the complete interaction family, candidate
sender manifest, training subjects, contrast weights, minimum support, minimum
effect, confidence level, filter universe, Holm rank and adjusted p-value. A
gate is therefore immutable and cannot be recomputed from held-out values.

The raw and Holm-adjusted p-values are internal training-gate diagnostics. They
are not result-level inferential p-values or q-values, do not authorize a
biological claim, and do not control multiplicity across receivers, contrasts,
folds, or repeated pipeline fits.

### Held-out sender allocation

The same functional retains one subject-pooled prevalence prior per frozen
candidate, the prevalence threshold, minimum training support, and softmax
temperature. For a held-out sample, sender application uses only that sample's
ligand availability:

```text
raw_sender_evidence = local_ligand_availability * training_prevalence_prior
assignment_weight = frozen_temperature_softmax(raw_sender_evidence)
```

Held-out senders outside the frozen candidate universe are ignored. A missing
local ligand value or non-estimable training prior cannot be refitted from
held-out data. When every candidate lacks evidence, every weight and entropy
remains NA; there is no uniform fallback.

Sender resolution is a secondary conservation operation. It multiplies an
already-established sender-unresolved member strength by normalized assignment
weights and verifies that resolved strengths sum back to the unchanged member
value. The interaction ligand-contrast gate and the sender assignment have
different responsibilities: the former establishes train-only upstream
contrast support, while the latter allocates an established held-out score.

### Family-common v2 consumption

`family_first_mechanistic_ligand_contrast_gated_softmin_v2` requires the same
interaction row to pass both receptor eligibility and the frozen ligand-
contrast gate. Only rows that pass both conditions contribute family
availability or member evidence. The family-core precedence is:

1. no receptor-eligible interaction: structural zero with
   `receptor_family_ineligible`;
2. no supported interaction and at least one receptor-eligible not-estimable
   gate: `not_estimable` with `ligand_contrast_not_estimable`;
3. all receptor-eligible gates observed but unsupported: structural zero with
   `ligand_contrast_not_supported`;
4. otherwise, continue through family selection, availability, incremental
   downstream gain, and the released soft-min integration.

When a family contains at least one supported interaction and another receptor-
eligible not-estimable interaction, an otherwise positive family core may
remain observed using only supported rows. Member allocation nevertheless fails
closed: entropy, within-family weights, supported-member scores, the not-
estimable member, and sender-resolved descendants remain not estimable. This
prevents normalization over a selectively observed family denominator. An
earlier family structural zero short-circuits allocation and remains exact zero;
receptor-ineligible and observed-unsupported members likewise receive exact
zero and cannot borrow family score.

## Versioning and migration

This decision introduces incompatible producer and identity versions:

- `ContrastCommonSenderParameters.schema_version = "3.0.0"`;
- `ContrastCommonSenderFunctional.schema_version = "3.0.0"`;
- interaction support and gate IDs use identity schema `3`;
- family-common functional/application producers use `v2`;
- family rows carry
  `score_version=family_first_mechanistic_ligand_contrast_gated_softmin_v2`;
- the public cross-fit family binding and edge-evidence policy use `v2`.

Common-sender v1/v2 objects, family-common v1 functionals/applications, and
cached or persisted rows with `family_first_mechanistic_softmin_v1` are not
compatible with the new lineage or status semantics. They must be discarded and
refitted from the raw fold scope. IDs, tables, or score magnitudes from the two
score versions must not be merged or compared without an explicit versioned
analysis.

## Consequences

The candidate functional is common across the registered contrast contexts and
insensitive to held-out context relabeling or test-only sender poisons. The
receiver-wise Holm gate blocks target-only downstream signal from becoming
integrated interaction evidence solely through absolute ligand availability.
Explicit frozen universes keep missing or not-estimable interactions in the
multiplicity family instead of silently shrinking it after results are seen.

The functional remains non-causal. Family-common v2 is still a diagnostic path,
not a certified global OOF functional, and distinct receiver children cannot be
row-unioned into one globally comparable cross-receiver score. The legacy
`assign_senders` API and default v0.1 baseline score remain unchanged.
