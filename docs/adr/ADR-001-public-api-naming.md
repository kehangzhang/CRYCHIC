# ADR-001: Public API naming

- Status: Accepted
- Date: 2026-07-12
- Implementation status: Implemented for the v0.1 exploratory surface

## Context

The concept document used a provisional project name, while the public project
name is CRYCHIC and the import package is `crychic`. Parallel implementation
requires a single spelling for the model facade and a deliberately small root
import surface.

## Decision

The primary facade is named `Crychic`. The distribution is `CRYCHIC`, the
Python package is `crychic`, and provisional names are never public. The root
package initially exports only reviewed symbols routed through `crychic.api`:
`Crychic`, `CrychicConfig`, `InputSchema`, and the configuration enums.

`Crychic` accepts an immutable `CrychicConfig` and optional versioned LR and
target-prior resources. In v0.1, `validate`, `dry_run`, and `fit` route to the
implemented workflow. `fit` returns in-memory exploratory artifacts unless an
`output_dir` is supplied, in which case it atomically publishes and returns a
queryable `CrychicResult`. Missing required LR resources raise a structured
`FeatureUnavailableError`.

The reviewed root surface now also exposes context design, versioned result,
and resource-loader contracts. Each addition is covered by the public API
contract test.

## Alternatives considered

- A capitalized acronym class such as `CRYCHIC` is visually ambiguous and does
  not follow Python class naming conventions.
- Exporting internal producer contracts from the root would make accidental
  APIs difficult to retract.
- Omitting the facade until fitting exists would prevent installed-wheel API
  contract tests during Phase 0.

## Consequences

The API remains intentionally small. Numerical modules cannot depend on the
facade, and adding a reviewed root export is an API change requiring a
contract test.

## Statistical and compatibility impact

The v0.1 facade authorizes only deterministic exploratory output. Persisted
p/q/posterior/probability fields remain null, and an in-sample strength cannot
be interpreted as calibrated or causal. Public names follow SemVer once
stabilized.

## Migration and rollback

No legacy public API exists. Before v1.0, a rename would use a documented
deprecation alias in `crychic.api`; rollback removes only the new reviewed
export and leaves producer contracts unchanged.
