# Response Module

## Owns

- Sample-level receiver response models and gene, TF, and pathway activity
  views.
- Continuous context contrasts, effect sizes, standard errors, precision
  weights, diagnostics, and response-backend adapters.

## Dependencies

- May import `core`, `pseudobulk`, `design`, and static knowledge from package
  `resources`.
- Must not depend on attribution, sender assignment, or integrated scores.

## Rules

- Fit sample-level models; never treat cells as independent replicates.
- Model input is a complete continuous response vector, not a hard-thresholded
  DEG set.
- Report effect, uncertainty, direction, convergence, sample counts, and
  estimability for every response.
- Repeated-measure support must use a validated clustered/GEE/weighted backend;
  otherwise reject or explicitly limit the design.
- In cross-fitting, fit filters, transforms, and activities on training folds
  only.

## Required Tests

- Hand-calculated contrasts, null/type-I fixtures, continuous response
  retention, repeated-subject designs, rank-deficient refusal, and backend
  parity on supported simple designs. Include the case where subject fixed
  effects absorb a between-subject treatment effect and require refusal.
