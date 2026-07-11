# Persisted Schema Instructions

## Scope

- Own machine-readable schemas for configuration, provenance, drivers,
  interactions, active probabilities, differential tests, signatures, sample
  scores, graphs, and diagnostics.
- This directory is the single normative source for persisted schemas;
  `crychic.results` implements validation and migrations, while `core` owns only
  in-memory shared primitives and schema-version identifiers.

## Rules

- Result-schema versions are independent of package versions and follow an
  explicit compatibility policy.
- Every table defines primary keys, required fields, types, units, null
  semantics, enums, and referential constraints.
- Add a migration and round-trip fixtures for every compatible schema change;
  breaking changes require a new major schema version.
- Schemas must distinguish zero, missing, not estimable, filtered, and failed.
- Do not weaken validation merely to accept a buggy historical artifact; write
  a targeted migration with provenance instead.
