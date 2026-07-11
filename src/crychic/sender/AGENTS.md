# Sender Module

## Owns

- Sender evidence, cell-type ligand specificity, prevalence, adjusted
  cross-subject coupling, soft sender assignment, entropy, and uncertainty.

## Dependencies

- May import `core`, `pseudobulk`, `response`, `availability`, and attribution
  result contracts.
- Must not refit receiver attribution or perform formal differential testing.

## Rules

- Keep receiver driver inference separate from sender-source assignment.
- Adjust coupling for context, batch, and declared covariates, or residualize
  them before association.
- Use training-fold data for learned coupling and assignment parameters.
- When sample support is weak, mark coupling as weak/ unavailable instead of
  using it as a hard filter.
- Sender assignment is evidence-based, not a causal claim; always expose
  assignment uncertainty.

## Required Tests

- Context-confounded coupling null, missing sender, group-specific sender,
  softmax normalization, uncertainty under indistinguishable senders, and fold
  leakage checks.
