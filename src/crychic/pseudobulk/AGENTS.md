# Pseudobulk Module

## Owns

- Aggregation to `sample x context x cell_type x gene` counts.
- Detection fractions, cell counts, cell proportions, library sizes, median
  UMI, aggregation QC, eligibility masks, and missingness classification.

## Dependencies

- May import `core` and validated contracts from `data`.
- Must not contain differential expression or communication scoring logic.

## Rules

- Aggregate raw counts by sample unit while retaining biological-subject and
  context metadata.
- Structural cell-type absence is represented by an eligibility/missingness
  state and never filled with expression zero.
- Low-cell-count units may contribute abundance evidence while being excluded
  from unstable state-expression estimation according to an explicit policy.
- Group-specific presence/absence and within-cell-type state changes are
  separate outputs.
- Sparse and dense paths must be numerically equivalent.

## Required Tests

- Hand-calculated aggregation, detection, abundance, and library-size cases.
- Sparse/dense equality, row-order invariance, structural-zero handling, and
  low-cell threshold boundaries.
