# Resampling Module

## Owns

- Estimability-aware subject folds, paired subject bootstrap, explicit
  exchangeability maps, restricted permutation/sign-flip plans, seed manifests,
  and reusable resampling indices.

## Dependencies

- May import `core` and design metadata. It must be independent of model
  implementations and call them only through protocols/callbacks.

## Rules

- A subject and all of that subject's samples/contexts stay in one fold or one
  bootstrap block. Every training fold must independently satisfy context,
  covariate, cell-type support, rank, and contrast estimability checks.
- Never split or permute at cell level for inferential procedures.
- Encode immutable covariates and permitted operations per design factor.
  Between-subject labels permute subject blocks within strata; paired
  within-subject effects use only valid restricted swaps or sign flips.
- Every resample has a stable ID, explicit index set, seed lineage, and
  provenance record.
- A planner may reduce `K` only through a documented pre-fit rule; if no
  `K >= 2` is estimable, reject cross-fitting. Never silently change fold count.

## Required Tests

- Zero subject overlap between train/test, paired-block preservation,
  fold-wise rank/estimability, determinism, stratum balance, impossible-split
  errors, and design-specific null permutation behavior.
