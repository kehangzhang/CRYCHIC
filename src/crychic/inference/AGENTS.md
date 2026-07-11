# Inference Module

## Owns

- Unpenalized models of out-of-fold sample scores; omnibus and post-hoc tests;
  clustered/CR2, GEE, or weighted backends; multiple-testing control;
  empirical p-values; local FDR; and specificity/selection probabilities.

## Dependencies

- May import `core`, `design`, `resampling`, and `scoring` contracts.
- Must not derive uncertainty directly from penalized attribution coefficients.

## Rules

- Validate subject-level replication and estimability for each hypothesis.
- Define hypothesis families and use a statistically valid hierarchical
  procedure; omnibus filtering followed by ordinary BH is not assumed valid.
- Keep effect, SE, p, q, active probability, specificity probability, and
  selection probability as separate fields.
- Return empirical p-value and `NA` probability when local-FDR diagnostics are
  unstable or the candidate set is too small.
- Refuse formal inference for complete confounding, rank deficiency, too few
  clusters, or cross-fitting leakage.

## Required Tests

- At least 1,000-replicate release calibration for null type-I/FDR scenarios,
  confidence-interval coverage, blocked permutation uniformity, hierarchical
  testing, probability calibration, and all refusal paths.
