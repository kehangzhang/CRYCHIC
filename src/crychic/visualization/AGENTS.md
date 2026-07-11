# Visualization Module

## Owns

- Static and interactive views of contexts, communication edges, uncertainty,
  signatures, residuals, topology, and diagnostic summaries.

## Dependencies

- May consume the stable `api`, `results`, and design graph representations.
- Computational modules must not import visualization code.

## Rules

- Read result objects only; never recompute tests, probabilities, or scores in
  plotting functions.
- Display sample support, uncertainty, missingness, mode, contrast, and
  state/ecosystem meaning where relevant.
- Do not visually convert missing or non-estimable values to zero.
- Return figure objects and data selections; avoid implicit display or global
  theme mutation.
- Keep plotting dependencies optional and lazily imported.

## Required Tests

- Headless smoke tests, missing/non-estimable rendering, deterministic selected
  data, no result mutation, and representative visual regression tests.
