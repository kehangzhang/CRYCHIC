# Inference Module

## Owns

- Unpenalized models of common-functional out-of-fold sample scores; omnibus and post-hoc tests;
  clustered/CR2, GEE, or weighted backends; multiple-testing control;
  empirical p-values; local FDR; bootstrap specificity support; and selection
  frequency summaries.

## Dependencies

- May import public `core`, `design`, `resampling`, and `scoring` artifact
  contracts. It consumes null/resampled score distributions and never invokes
  attribution, sender, or workflow computation itself.
- Must not derive uncertainty directly from penalized attribution coefficients.

## Rules

- Validate subject-level replication and estimability for each hypothesis.
- Define hypothesis families and use a statistically valid hierarchical
  procedure frozen before implementation; omnibus filtering followed by
  ordinary BH is not assumed valid.
- Keep effect, SE, p, q, active posterior probability, bootstrap specificity
  support, and selection frequency as separate fields and grains.
- Analytic OOF regression does not automatically account for learned nuisance
  models. Until valid theory exists, intervals/tests consume repeated-cross-fit
  full-pipeline subject bootstrap/permutation outputs.
- Frequency-based q-value calibration and local-FDR posterior calibration have
  independent release gates and failure statuses.
- Return empirical p-value and `NA` probability when local-FDR diagnostics are
  unstable or the candidate set is too small.
- Refuse formal inference for complete confounding, rank deficiency, too few
  clusters, or cross-fitting leakage.

## Required Tests

- At least 1,000-replicate release calibration for null type-I/FDR scenarios,
  confidence-interval coverage, blocked permutation uniformity, hierarchical
  testing, full-pipeline nuisance uncertainty, probability calibration, and all
  refusal paths.
