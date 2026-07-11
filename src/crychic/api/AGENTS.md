# API Module

## Owns

- Stable Python facade, user configuration entry points, validation helpers,
  and intentional public re-exports.

## Dependencies

- May import `core`, `design`, `workflow`, and `results` public contracts.
- It is a thin boundary and must not implement numerical methods.

## Rules

- Public naming uses `CRYCHIC`/`crychic`; never expose the provisional
  `TopoCCC` name.
- The main flow remains concise: configure, validate/dry-run, fit, query/save.
- Default subject field is `subject_id`; `subject_key` may override it.
- Public signatures, defaults, deprecations, exceptions, and result semantics
  are documented and contract-tested.
- API compatibility follows SemVer; experimental surfaces are explicitly
  labeled and excluded from stability guarantees.

## Required Tests

- Import smoke tests from an installed wheel, documented minimal usage,
  signature snapshots, deprecation behavior, optional-dependency errors, and
  exploratory/inferential mode reporting.
