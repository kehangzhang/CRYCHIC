# ADR-008: Repeated-measures CR2 backend

- Status: Accepted
- Scope: v0.3 receiver and out-of-fold effect estimation
- Decision date: 2026-07-16

## Problem

Repeated regions, times, or treatments from one subject are not independent
replicates.  The existing receiver backend fits a clustered OLS contrast with
CR1 standard errors, but intentionally labels every result exploratory.  A
separate, auditable backend is needed before repeated-measures effects can
enter full-pipeline bootstrap/permutation inference.

The backend must support mixed paired/unpaired designs, refuse models whose
subject fixed effects absorb the requested contrast, and make small-cluster,
rank, leverage, and covariance failures explicit.  It must not turn an
analytic sandwich standard error into a released p-value because learned OOF
nuisance models are not covered by that covariance alone.

## Options considered

1. Keep CR1 as the only backend.  Rejected for formal work because its finite
   cluster bias correction is not adequate for the preregistered v0.3 path.
2. Add a GEE dependency.  Deferred because the current package has no
   `statsmodels` dependency and a defensible GEE contract would additionally
   require a frozen working correlation, convergence policy, and small-sample
   covariance correction.
3. Add a strict subject-equal WLS/CR2 backend using the existing NumPy/SciPy
   stack.  Accepted.

## Decision

`fit_repeated_measures_cr2_receiver_effect()` is an opt-in backend separate
from the existing CR1 function.  The default exploratory and cross-fit paths
are unchanged.

For mixed paired/unpaired data without subject fixed effects, technical
replicates are first averaged within frozen subject/context/covariate cells.
The model then assigns total weight one to each subject and uses a
Bell-McCaffrey CR2 cluster adjustment:

```text
A_g = (I - H_gg)^(-1/2)
V_CR2 = B [sum_g X_g' A_g e_g e_g' A_g X_g] B
```

where all matrices include the square-root subject weights.  This prevents a
subject with more sampled contexts from automatically receiving more total
weight.

For a subject-fixed-effect design, the backend first honors the frozen design
refusal. If the contrast is estimable, a subject-level contrast backend is
allowed only when there are no additional covariates and every retained
subject has exactly one frozen cell in every contrast context. The shortcut is
also restricted to the package's default saturated context-factorial formula;
a custom or multi-factor formula returns typed `not_estimable` rather than
silently changing the estimand. It estimates the mean of the independent
subject contrasts; this is the intercept-only CR2 form of a paired analysis.
Incomplete subjects do not contribute, and the complete-subject minimum is
rechecked. More general absorbed-FE covariance handling is not silently
approximated.

## Eligibility and degrees of freedom

Every feature has a typed status: `eligible` or `not_estimable`.  Eligibility
requires all of the following:

- the frozen design and contrast are estimable;
- every contrast context meets its subject minimum;
- the contrast-support cluster count meets the preregistered minimum;
- the feature-specific weighted design has full column rank and the contrast
  lies in its row space;
- residual degrees of freedom are positive and the condition number is below
  the frozen maximum;
- every `I - H_gg` is finite and positive definite above `1e-10`;
- the CR2 covariance is finite and positive semidefinite within numerical
  tolerance; and
- the contrast variance is finite and strictly positive.

The formal backend additionally freezes an absolute floor of six independent
subject clusters, even when a diagnostic design requests a lower minimum.
Condition, leverage, covariance and contrast-variance checks are response-scale
equivariant: multiplying every response by a positive constant cannot change
eligibility merely because of an absolute variance threshold.

The recorded degrees-of-freedom policy is
`contrast_support_clusters_minus_one_guard_only_v1`.  It is used as a support
diagnostic, not to manufacture an analytic t-test. The existing OOF effect
backend is assessed analogously with
`subject_clusters_minus_one_guard_only_v1`. Its fit result and eligibility
result are producer-owned; formal assessment rechecks hypothesis/spec lineage,
the context contrast against stored coefficients, the standard error against
the CR2 covariance, the condition number, and leverage diagnostics.

Common refusal codes include `insufficient_subject_clusters`,
`insufficient_complete_subject_clusters`, `rank_deficient_feature_design`,
`feature_contrast_not_estimable`, `cr2_cluster_leverage_not_estimable`, and
`degenerate_cr2_contrast_variance`.  Refused results carry `None` for effect,
SE, and precision; they never fall back to CR1.

## Release boundary

`formal_backend_eligible` means that a point fit can enter the frozen
full-pipeline subject bootstrap/permutation chain.  It does not mean that a
formal result can be released.  Consequently these artifacts always report
`formal_inference_allowed = false`, contain no analytic p/q fields, and record
the requirement `full_pipeline_resampling_and_g3f_gate`.

The existing CR1 result remains `exploratory_only`.  Downstream code must not
reinterpret its diagnostic SE as CR2 or use it as a fallback when CR2 refuses
a feature.

## Verification

Unit fixtures cover a hand-calculated mixed paired/unpaired contrast,
subject-fixed-effect paired contrasts, feature-specific missingness, small
cluster refusal, singular cluster leverage, response-scale equivariance,
custom-formula refusal, immutable result lineage, adversarial OOF effect/SE
lineage, OOF formal eligibility, and public API identity. Existing CR1 fixtures
remain in the test suite to guard backward compatibility.

## Migration and rollback

Callers opt in by importing the new CR2 function.  No persisted CR1 schema or
default workflow changes.  The new public symbols can be withdrawn before a
stable v1 API if calibration fails; existing exploratory artifacts remain
readable and retain their original identities.
