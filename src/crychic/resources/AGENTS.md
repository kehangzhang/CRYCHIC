# Package Resources Module

## Owns

- LR resources, ligand/receptor complexes, static ligand-target priors,
  molecular signaling topology `G_S`, resource adapters, manifests, caching,
  checksums, licenses, and citations.

## Dependencies

- May import `core` and gene namespace contracts from `data`.
- Dynamic sample availability belongs to `availability`; activity estimation
  belongs to `response`.

## Rules

- Loaded resources are immutable and versioned. Preserve complex subunits,
  direction, sign, species, gene namespace, and evidence level.
- Network downloads are explicit, checksum-verified, cached outside Git, and
  support offline mode.
- Do not silently discard unmapped genes or unsupported complex components;
  return an auditable mapping report.
- Keep precomputed prior loading available when network-derived priors are not
  installed.

## Required Tests

- Manifest/schema validation, checksum failure, offline cache behavior,
  complex parsing, direction/sign preservation, and deterministic mapping.
