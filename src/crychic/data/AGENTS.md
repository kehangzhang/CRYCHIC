# Data Module

## Owns

- `h5ad`/AnnData loading, backed and sparse access, input schema validation,
  gene identifier mapping, metadata normalization, and input-level QC reports.
- Validation of `counts`, `sample_id`, `subject_id`, `cell_type`, context keys,
  and covariates before any model is fitted.

## Dependencies

- May import `core` only. Aggregation belongs to `pseudobulk`.
- Must not import statistical, attribution, scoring, or workflow modules.

## Rules

- Do not mutate caller-owned AnnData objects.
- Counts used for inference must be finite, non-negative integers. Report the
  layer and offending coordinates when validation fails.
- Normalized-only exploratory input declares its source and transform; never
  infer a transform from value ranges. It forces formal p/q/posterior fields to
  unavailable with a reason code.
- Validate that each `sample_id` maps to exactly one subject and one value for
  every declared context key.
- Gene mapping is explicit, species-aware, many-to-one policy controlled, and
  fully recorded in provenance.
- Preserve sparse/backed representations and estimate memory before loading.

## Required Tests

- Dense, CSR/CSC, backed, duplicated-gene, missing-field, non-integer-count,
  normalized-only/unknown-transform, and inconsistent sample-mapping fixtures.
- Validation must not alter input values, order, layers, or metadata.
