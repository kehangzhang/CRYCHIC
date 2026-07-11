# Results Module

## Owns

- Versioned result schemas, immutable result facade, validation, persistence,
  lazy queries, migrations, and Parquet/Zarr/GraphML directory layout.

## Dependencies

- May import `core` and typed output contracts. It must not trigger numerical
  computation or import workflow implementations.

## Rules

- Do not place large multidimensional outputs in `adata.uns`.
- Writes are atomic or transactional; an interrupted result is marked
  incomplete and cannot be loaded as successful.
- Enforce unique keys, data types, null semantics, units, and schema version for
  interactions, differential results, signatures, scores, and diagnostics.
- Package version and result-schema version evolve independently.
- Queries do not silently recalculate or reinterpret statistics.

## Required Tests

- Schema validation, key uniqueness, atomic-failure recovery, round-trip,
  forward/backward migration, lazy filtering, and corrupted-artifact errors.
