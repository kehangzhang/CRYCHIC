# Availability Module

## Owns

- Ligand, receptor, and multi-subunit complex availability; detection-rate
  shrinkage; receiver gating; and state/ecosystem availability components.

## Dependencies

- May import `core`, `pseudobulk`, and static definitions from package
  `resources`.
- Must not compute downstream attribution or statistical significance.

## Rules

- Distinguish missing/unavailable estimates from biological zero.
- Required complexes use a documented soft-min or generalized harmonic
  aggregation; a limiting required subunit must limit availability.
- Preserve both state and ecosystem modes. Ecosystem incorporates abundance;
  state mode does not.
- Gating uses baseline or training-fold information and must not use the
  response contrast being explained.
- Return individual components and QC flags, not only a composite product.

## Required Tests

- Complex limiting-subunit behavior, monotonicity, missingness, detection
  shrinkage boundaries, abundance-only null, and no-outcome-leakage tests.
