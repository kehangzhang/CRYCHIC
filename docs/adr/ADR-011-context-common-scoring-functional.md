# ADR-011: Context-common scoring functional

- Status: Accepted
- Date: 2026-07-16
- Implementation status: Implemented for descriptive subject cross-fit;
  formal inference remains gated by G3

## Context

A context effect is meaningful only when every held-out context is evaluated
with the same function. Fitting a separate feature universe, target basis,
gate, transform, penalty, or sender rule for each held-out context would write
the context label into the outcome and invalidate the contrast.

This requirement is receiver-scoped. Different receivers have different
response models, receptor eligibility, nuisance fits, null losses, and selected
penalties. Combining those receiver children does not create a common
cross-receiver biological scale; ADR-005 and ADR-007 continue to prohibit that
row union.

## Decision

The authoritative context-common functional has grain:

```text
repeat x physical_outer_fold x contrast x receiver
```

One producer-owned `FamilyCommonScoringFunctional` is fitted from the physical
outer-training subjects at that grain and is applied unchanged to every
held-out context registered by the contrast. Its stable identity binds:

- the exact contrast manifest and canonical context IDs;
- training subjects, fold and frozen interaction/filter universe;
- receiver-family basis, feature/family/interaction mapping and receptor gates;
- receiver response, precision, nuisance, coordinate-transform and incremental
  functional lineage;
- tuning manifest, selected resolved penalty and family-gain estimand;
- contrast-common ligand support/gate and sender functional;
- receiver-program parent, autonomous/latent nuisance source, soft-min policy,
  score version and all parent artifact IDs/digests.

Held-out context may select only the corresponding input rows and frozen
contrast-regressor values. It cannot change any learned parent, candidate
universe, gate, target weight, scale, threshold, penalty, or assignment rule.

The application must cover the exact held-out sample/subject/context manifest.
All contexts in the functional remain represented. Structural zero, missing
and `not_estimable` states retain their typed semantics and cannot be repaired
by dropping rows or substituting another functional.

The released contract is:

```text
common_across_contexts = true
common_functional_across_receivers = false
comparability_scope = within_receiver_across_registered_contrast_contexts
formal_inference_allowed = false
```

The authoritative cross-fit registry records one child functional per
`fold x contrast x receiver`. `CrossFitOOFCertificationAudit` verifies the
registered functional, family-common train/apply chain, subject disjointness,
complete context coverage and source lineage. An effect producer must reject a
fold containing multiple scoring functional IDs or contexts outside the frozen
functional.

## Fail-closed conditions

The context-common claim is unavailable when any of the following occurs:

- context-specific feature, family, interaction, target-weight or gate parents;
- context-specific response transform, precision, nuisance or selected penalty;
- held-out refitting, rescaling, selection or candidate-set changes;
- incomplete registered context coverage or unseen held-out context;
- stale parent IDs, mixed score versions, lineage/table digest mismatch;
- training/held-out subject overlap or an unverified static nuisance source.

Failure emits a typed `not_estimable` child and reason code. It never falls back
to the legacy baseline or a receiver-specific context functional.

## Consequences

Descriptive OOF context effects may compare registered contexts within the same
receiver after the complete certification audit passes. This does not authorize
cross-receiver raw-score pooling, p-values, q-values, communication
probabilities, causal sender claims, or a scientific default switch. Formal
effect and uncertainty release still requires full-pipeline resampling and the
appropriate G3 gate.
