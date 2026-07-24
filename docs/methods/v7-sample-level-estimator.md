# CRYCHIC v7 sample-level estimator contract

Status: implementation target. This document freezes the estimands before the
v7 implementation and benchmark cycle. It does not claim that the complete v7
pipeline is already released or calibrated.

## Legacy freeze

The v2 family-common score, RC12 sender/program ranker, and RC14
pair-prioritized event ranker are retained as diagnostic comparators. Their
machine-readable freeze is
`benchmarks/configs/v7_legacy_baselines_v1.json`. Each is
`benchmark_diagnostic_only=true` and `formal_inference_eligible=false`.
Further RC rank-head tuning is outside the v7 development path.

## Primary measurement order

For every outer fold, expression and reliability transforms are fitted using
training subjects only. The frozen transforms are then applied to held-out
samples to produce one row per sample, sender, LR interaction, and receiver.
Condition labels do not select, gate, rescale, or zero the primary activity
measurement. Differential models consume the concatenated subject-level OOF
rows only after this measurement stage.

## Separate output heads

The v7 table separates quantities that were previously conflated:

| Head | Estimand | Formal differential target |
|---|---|---|
| `sender_detection_raw` | Non-conserving sender-specific LR support | Yes |
| `parent_activity_raw` | LR-receiver activity, with peak, total, and mean summaries | Yes |
| `program_signed` | Signed receiver mechanism response | Separate test or frozen ranking aid |
| `active_probability` | Subject-level event occurrence probability | Occurrence model only |
| `sender_attribution` | Conditional relative sender contribution, including a null sender | Compositional model or display |

Cell count and coverage produce reliability or precision fields; they never
multiply raw communication intensity. Resource confidence, contrast support,
family selection, downstream support, coupling, and hypergraph topology remain
annotations or separately tested components unless a locked benchmark proves a
pre-registered integration rule.

## Missingness semantics

`structural_impossible`, `not_estimable`, and `low_evidence` are distinct.
Structural impossibility may create a structural zero. Missing measurement or
coverage creates an NA row with an explicit reason. Low evidence remains a
finite continuous score and cannot be converted to zero by a condition-derived
gate.

## Inference boundary

Analytic HC3/CR2 models provide point effects and diagnostics. Formal p-values,
q-values, and confidence intervals require the declared design to be estimable
and require full-pipeline subject-level resampling while nuisance learning is
in scope. Penalized coefficients and post-hoc ranking heads never feed ordinary
Wald inference.

The v7 design contract supports independent two-group and multi-group models,
paired subject differences, repeated measurements with subject-cluster CR2,
multi-cohort fixed effects, and continuous exposures. Categorical analyses use
pre-registered contrasts; independent multi-group analyses also emit an
omnibus diagnostic. Samples are collapsed to one equally represented row per
subject and context before fitting. Cell-count and coverage reliability enter
only as precision weights. A repeated design with too few subject clusters does
not silently fall back to an independence covariance model.

Fields named `diagnostic_p_value`, `diagnostic_ci_lower`, and
`diagnostic_ci_upper` describe only the analytic regression diagnostic. The
formal `p_value`, `q_value`, `ci_lower`, and `ci_upper` fields remain NA and
`formal_inference_allowed=false` until the PR10 full-pipeline resampling release
gate is satisfied.
