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
- For a registered contrast, derive one frozen scoring functional per training
  fold and apply the same functional, feature universe, transformations, and
  scale to every context being compared. Context-specific score functions are
  mechanistic outputs, not directly comparable inferential outcomes.
- Preserve availability, downstream, sender, prior-quality, and abundance
  components alongside the composite geometric score.
- Missing core evidence yields missing/flagged scores according to policy, not
  an implicit zero.
- State and ecosystem scores must be independently queryable.

## Required Tests

- Formula hand checks, component monotonicity, missingness propagation,
  out-of-fold provenance, common-functional context comparability,
  fold-specific-functional null, abundance-only separation, and scale handling.
