# Test Suite Instructions

## Scope

- Organize tests as unit, integration, contract, statistical, regression, and
  performance suites. Mark slow calibration and external-resource tests.
- Prefer fixed-seed synthetic generators and hand-computable fixtures. Real
  data require de-identification and explicit redistribution permission.

## Required Invariants

- No subject appears in both training and test folds.
- Structural absence is never silently converted to zero state expression.
- Penalized coefficients never feed ordinary p-value formulas.
- Non-estimable designs do not emit inferential fields.
- Every compared context uses the same fold-frozen scoring functional.
- Full-pipeline resampling captures filtering/tuning/attribution uncertainty.
- Sparse/dense and serial/parallel paths agree within declared tolerance.

## Quality Rules

- Test behavior and contracts, not private implementation details.
- Float assertions state justified absolute/relative tolerances.
- Statistical tests report seeds, repetitions, Monte Carlo intervals, and
  expected operating ranges; avoid brittle exact stochastic assertions.
- Golden data remain tiny, readable where possible, versioned, and accompanied
  by the command/config that generated them.
