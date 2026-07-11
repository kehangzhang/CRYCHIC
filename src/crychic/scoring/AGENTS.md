# Scoring Module

## Owns

- Sample-level out-of-fold communication components and integrated
  `comm_strength`, including state/ecosystem decomposition and QC flags.

## Dependencies

- May import `core`, `availability`, `attribution`, and `sender` contracts.
- Probability calibration and hypothesis tests belong to `inference`.

## Rules

- Do not call a raw expression product or integrated strength a probability.
- Inferential scores are generated out of fold; retain fold and training-model
  identity on every sample score.
- Preserve availability, downstream, sender, prior-quality, and abundance
  components alongside the composite geometric score.
- Missing core evidence yields missing/flagged scores according to policy, not
  an implicit zero.
- State and ecosystem scores must be independently queryable.

## Required Tests

- Formula hand checks, component monotonicity, missingness propagation,
  out-of-fold provenance, abundance-only separation, and fold-scale handling.
