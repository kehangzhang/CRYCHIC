# Workflow Module

## Owns

- End-to-end stage orchestration, dependency resolution, cross-fitting
  boundaries, caching, checkpoint/resume, run manifests, and failure states.

## Dependencies

- May compose all computational modules through their public contracts.
- No lower-level module may import `workflow`.

## Rules

- Orchestrate only; statistical formulas and resource parsing stay in owning
  modules.
- Make train/test boundaries explicit in stage inputs and block access to test
  data during learned preprocessing.
- Full-pipeline bootstrap and permutation rerun filtering, gating, clustering,
  tuning, attribution, sender assignment, scoring, and effect fitting as
  required by the estimand. Do not resample only a previously generated score
  table when nuisance-learning uncertainty is in scope.
- Workflow produces typed null-score and resampled-attribution artifacts for
  inference; inference never calls computational stages in reverse.
- Cache keys include input/config/resource/code/schema digests and fold ID.
- Resume is idempotent; incomplete or incompatible checkpoints are rejected.
- Emit a dry-run plan showing estimability, resource needs, stages, and output
  paths before expensive work.

## Required Tests

- Tiny end-to-end synthetic run, no subject leakage, cache invalidation,
  full-pipeline resampling, interruption/resume equivalence, partial-failure
  status, and deterministic serial/parallel results.
