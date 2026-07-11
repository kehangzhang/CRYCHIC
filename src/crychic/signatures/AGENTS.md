# Signatures Module

## Owns

- Receiver context signatures, LR-attributed signatures, full sender-LR-
  receiver signatures, direction agreement, stability ranking, and residuals.

## Dependencies

- May import `core`, `response`, `attribution`, and `sender` contracts.
- It must not perform new model fitting or significance tests.

## Rules

- Preserve observed response, predicted contribution, and unexplained residual
  as distinct quantities.
- Enforce direction-agreement rules explicitly and record the rule/version.
- Contributions must reconcile with fitted response within numeric tolerance.
- Do not force unmodeled receiver-autonomous effects into communication edges.
- Export stable gene namespace, direction, weight, fold, and selection
  stability for external projection.

## Required Tests

- Contribution reconstruction, residual identity, direction filtering,
  deterministic ranking, empty/unknown-prior behavior, and schema keys.
