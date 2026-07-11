# Core Module

## Owns

- Serializable configuration, common enums, stable identifiers, generic typed
  protocols, errors, provenance primitives, seed lineage, and schema-version
  identifiers.
- Business artifacts such as pseudobulk, response, attribution, and score DTOs
  are owned by their producer modules, not by `core`.

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

- Configuration and in-memory metadata serialization round-trips.
- Stable-ID permutation invariance and collision fixtures.
- Seed-tree determinism and protocol conformance.
