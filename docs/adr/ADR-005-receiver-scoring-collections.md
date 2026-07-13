# ADR-005: Receiver scoring collections

- Status: Accepted
- Date: 2026-07-13
- Implementation status: Partial emitted-score provenance

## Context

The exploratory workflow fits one child scoring functional per receiver and
contrast. Each child is common across the contexts compared by that contrast,
but different receivers may consume different response bases, coefficients,
target weights, gates, and model manifests. Treating all child IDs as competing
whole-contrast candidates duplicates score rows in downstream adapters. Calling
one child a cross-receiver common functional is statistically false.

The result contract needs a verifiable way to enumerate receiver children that
were emitted for one contrast/repeat/fold output, while preserving the boundary
between emitted coverage and a planned receiver universe.

## Decision

CRYCHIC persists `scoring_collections.json` as a versioned optional extension
of the v0.1 result schema. A `ScoringCollectionManifest` has exactly one
`contrast x repeat_id x fold_id` grain and records:

- `partition_key=receiver` and `receiver_scope=emitted_sample_scores`;
- `composition_status=partial_emitted_only`;
- the exact emitted receiver set, without a planned-universe claim;
- one child manifest per emitted receiver, including the scoring functional ID;
- `provenance_status=functional_metadata_not_persisted_unverified`, because no
  authoritative persisted model/version/universe registry exists yet;
- each child's source-score row count and order-independent source-key digest;
- a collection ID derived from the complete normalized payload;
- `common_functional_across_receivers=false` as an invariant.

The source-key digest covers the persisted sample-score primary-key projection:
subject, sample, context, design row, edge, scoring functional, repeat, fold,
and mode. Persistence proves that every sample-score functional belongs to one
and only one child, every child has the declared repeat/fold and key digest,
and every child edge belongs to its declared receiver and contrast in the
interaction table.

## Compatibility

The extension does not change any required v0.1 table. Results created before
this ADR, results without integrated scoring, and results containing only the
edge-evidence extension remain readable. New integrated baseline results emit
the compact collection registry automatically.

## Consequences

Downstream readers can group emitted receiver children without row duplication.
The collection proves exact coverage of emitted score rows only; it does not
prove that a planned receiver universe is complete, that unscored biological
receivers were estimable, that child model metadata is authoritatively
registered, that different receivers share one model, or that the current
in-sample baseline is certified out of fold.
