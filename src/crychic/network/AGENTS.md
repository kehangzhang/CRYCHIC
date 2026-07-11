# Network Module

## Owns

- Directed heterogeneous communication hypergraph `H`, stable hyperedge IDs,
  graph exports, modules, hubs, context gradients, and output topology metrics.

## Dependencies

- May consume public scoring, inference, and signature artifacts plus shared
  `core` primitives. It must not import the persistence facade in `results`.
- No fitting module may depend on this output representation.

## Rules

- Build result hypergraphs after model fitting; do not fit a GNN as part of the
  core statistical method.
- Keep sender, ligand/complex, receptor/complex, receiver, program, context,
  and direction explicit in each hyperedge.
- Do not mix `G_C`, `G_S`, and `H` in a single graph class or serialization.
- Topology metrics state their weighting, missing-edge, and direction policy.

## Required Tests

- Stable hyperedge IDs, direction preservation, serialization round-trips,
  disconnected structures, known small-graph metrics, and context filtering.
