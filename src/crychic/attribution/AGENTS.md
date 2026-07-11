# Attribution Module

## Owns

- Context-gated LR target basis construction, normalization, equivalence
  classes, positive elastic net, graph-fused optimization, hyperparameter
  selection, stability selection, and solver diagnostics.

## Dependencies

- May import `core`, `design`, package `resources`, `response`, and
  `availability` contracts.
- Must not import sender, scoring, inference, workflow, or visualization.

## Rules

- Jointly explain receiver responses using non-negative, consistently
  normalized bases and explicit precision weights.
- A non-negative prior may explain only a direction-compatible response.
  Reverse contrasts or separately defined channels handle attenuation; do not
  fit a signed response with a non-negative basis and call the residual closed.
  Signed molecular priors require an explicit, reviewed sign convention.
- Cluster highly correlated LR profiles before interpreting individual LR
  identities; report family-level results and assignment uncertainty.
- Graph fusion operates only over `G_C` edges and preserves disconnected
  components and topology-local jumps.
- Every solver result reports status, objective terms, iterations, KKT or
  primal/dual residuals, tolerance, and initialization.
- Penalized coefficients never directly produce ordinary p-values.

## Required Tests

- CVXPY golden problems, KKT/convergence checks, `lambda_F=0` equivalence,
  large-fusion limits, disconnected graphs, topology jumps, relabeling
  invariance, signed/directional response behavior, collinear-family recovery,
  and deterministic tuning.
