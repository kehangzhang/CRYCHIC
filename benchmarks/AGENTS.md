# Benchmark Instructions

## Scope

- Own simulation scenarios, external-method adapters, isolated environments,
  workflow definitions, metric computation, resource harmonization, and
  benchmark reports. Runtime `crychic` code must never import this directory.

## Fairness and Reproducibility

- Compare methods once with a harmonized LR resource and once with each method's
  default resource; do not confuse resource effects with algorithm effects.
- Record method/resource versions, licenses, checksums, seeds, hardware,
  threads, environment images, and failure status.
- Compare only quantities each method legitimately produces. Do not treat an
  uncalibrated rank as a calibrated p-value.
- Store configurations, code, manifests, and compact metric summaries in Git;
  keep raw outputs and downloaded resources outside Git.

## Required Scenarios

- Include global null, abundance-only, receiver-autonomous, ligand-only,
  target-only, composition imbalance, graph-smooth, topology jump, wrong
  topology, disconnected graph, collinear LR, structural absence, batch/context
  confounding, annotation perturbation, and prior replacement.
