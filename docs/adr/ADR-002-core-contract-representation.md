# ADR-002: Core contract representation

- Status: Accepted
- Date: 2026-07-12
- Implementation status: Implemented for Phase 0 contracts

## Context

Core configuration, identifiers, provenance, and seed lineage must remain
lightweight, serializable, deterministic, and importable without numerical or
dataframe backends. Persisted JSON Schema must remain distinct from in-memory
business artifacts owned by producer modules.

## Decision

Core value contracts use standard-library, frozen, slotted dataclasses and
typed protocols. Construction validates contradictions and unknown persisted
fields. Canonical JSON uses sorted keys, explicit enum values, compact
separators, ASCII encoding, and rejects non-finite numbers. SHA-256 over that
representation supplies configuration digests and stable IDs.

Core owns only shared primitives. Arrays, sparse matrices, and tables remain in
their producer modules; their exact NumPy/SciPy/Arrow representations are
deferred until the relevant contract is implemented. JSON files are governed
by the schemas under `schemas/`, not by a second validation model embedded in
core.

## Alternatives considered

- Pydantic provides convenient coercion but adds a base dependency and can
  silently change accepted inputs across major versions.
- Untyped dictionaries make ownership, invariants, and compatibility unclear.
- A central core DTO for every artifact would invert producer ownership and
  couple otherwise independent modules.

## Consequences

Configuration conversion code is explicit and conservative. Frozen dataclasses
prevent ordinary mutation, while nested collections are normalized to tuples
or immutable mappings. Producer modules may choose specialized array and table
containers without expanding the base import cost.

## Statistical and compatibility impact

No estimator is selected by this decision. Deterministic serialization and
seed derivation support reproducible folds and resamples, but do not establish
statistical validity by themselves. Persisted schema versions evolve
independently from package versions.

## Migration and rollback

Compatible dataclass changes require matching schema and round-trip tests.
Breaking persistence changes require a new schema major version and targeted
migration. A future validation library may replace constructors behind the
same public contract after benchmarked compatibility evidence.
