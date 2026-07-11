# Core Module

## Owns

- Serializable configuration, enums, stable identifiers, typed protocols,
  shared immutable data contracts, errors, provenance primitives, and seed
  lineage.
- Cross-module contracts such as pseudobulk, response, attribution, score, and
  result metadata interfaces. Concrete numerical implementations stay outside
  this module.

## Dependencies

- May depend only on the Python standard library and explicitly approved light
  typing/validation libraries.
- Must not import any other `crychic` business module.

## Rules

- Configuration round-trips without loss and rejects unknown or contradictory
  fields.
- IDs are deterministic, order-independent, and schema-versioned.
- Do not perform file I/O, network access, logging configuration, or expensive
  computation at import time.
- Exceptions carry structured context without embedding patient-level data.

## Required Tests

- Serialization and schema migration round-trips.
- Stable-ID permutation invariance and collision fixtures.
- Seed-tree determinism and protocol conformance.
