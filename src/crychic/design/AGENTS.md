# Design Module

## Owns

- Context topology `G_C`, canonical context tuples, formula/design matrices,
  estimated marginal means, estimability diagnostics, and contrast builders.
- Chain, complete, custom-edge, disconnected, and Cartesian-product graphs.

## Dependencies

- May import `core` and metadata contracts from `data`/`pseudobulk`.
- Must not import molecular resource or output hypergraph types.

## Rules

- Never reuse a generic graph type for molecular topology or result
  hypergraphs.
- Validate rank, aliasing, confounding, replication, and contrast estimability
  before fitting.
- Global one-vs-rest uses balanced, estimated-marginal-mean, or explicit user
  weights, not cell-count weights.
- Local contrasts use only declared graph neighbors and normalized edge
  weights. Disconnected components cannot exchange information.
- Contrast definitions are serializable, named, and have a clear hypothesis
  family.

## Required Tests

- Contrast weights sum to zero when appropriate and match hand calculations.
- Context relabeling/permutation invariance, graph-product edge fixtures,
  disconnected components, rank deficiency, and treatment-by-region designs.
