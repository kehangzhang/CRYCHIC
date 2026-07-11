# Resampling Module

## Owns

- Subject-stratified folds, paired subject bootstrap, exchangeability-aware
  blocked permutation, seed manifests, and reusable resampling indices.

## Dependencies

- May import `core` and design metadata. It must be independent of model
  implementations and call them only through protocols/callbacks.

## Rules

- A subject and all of that subject's samples/contexts stay in one fold or one
  bootstrap block.
- Never split or permute at cell level for inferential procedures.
- Respect pairing, strata, batches, and exchangeability restrictions declared
  by the design.
- Every resample has a stable ID, explicit index set, seed lineage, and
  provenance record.
- Fail when a requested split cannot preserve minimum estimability rather than
  silently changing fold count.

## Required Tests

- Zero subject overlap between train/test, paired-block preservation,
  determinism, stratum balance, impossible-split errors, and null permutation
  behavior.
