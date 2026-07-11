# Python Package Instructions

## Scope

Applies to all runtime code under `src/crychic` in addition to the repository
rules.

## Package Boundary

- The root package exposes only reviewed public symbols from `api`: the model
  facade, configuration, context graph, contrast specification, input schema,
  validation entry point, and result object.
- Internal modules are private unless explicitly re-exported. Do not make an
  implementation detail public merely for test convenience.
- Keep imports acyclic and follow the dependency graph documented in
  `../../DEVELOPMENT_PLAN.md`.
- Put a business artifact contract beside its upstream producer and expose it
  through that subpackage's public surface. Use `core` only for shared
  primitives and generic protocol/configuration machinery.
- Use protocols for response, attribution, inference, and resource backends.
- Import optional dependencies inside the relevant adapter and raise an
  actionable extra-installation error when unavailable.

## Cross-Cutting Contracts

- Use canonical, stable IDs for contexts, samples, LR interactions, folds, and
  output rows. IDs must not depend on input row order.
- Preserve sparse matrices and backed access until an algorithm explicitly
  requires materialization and has checked memory cost.
- Random functions receive an explicit generator or seed lineage; no hidden
  global RNG state.
- Public functions and data models require type annotations and concise API
  documentation.
- Each subpackage owns tests for its invariants as described by its local
  `AGENTS.md`.
