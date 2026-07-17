# ADR-010: Graph-fusion G2 decision gate

- Status: Accepted method contract; default graph fusion remains opt-in until the gate passes
- Date: 2026-07-16
- Scope: v0.2 graph-fused attribution and its preregistered benchmark

## Context

Graph fusion is a structural regularizer, not evidence that a communication
edge is biologically active.  A topology-aware solver can improve recovery on
one graph while harming a disconnected, locally discontinuous, or misspecified
graph.  The numerical endpoint and topology controls therefore have to be
frozen before a campaign is inspected.

The graph-fused integrated adapter is also a distinct estimand from the
ordinary null-versus-single-family differential effect.  Its output is
receiver- and context-scoped, descriptive only, and cannot authorize pooled
cross-receiver or cross-mode comparisons.

## Decision

The G2 primary manifest contains the full 16-cell factorial formed by:

- chain and product context graphs;
- smooth-gradient and single-local-jump coefficient fields;
- two signal-to-noise levels; and
- two LR-collinearity levels.

Scenario cells receive equal weight.  Each seed is paired between fused and
unfused fits on the same generated data and frozen truth.

The sole primary endpoint is driver-family by context recovery macro-AUPRC.
The primary paired improvement is the fused-minus-unfused difference.  Its
one-sided 95% bootstrap lower bound must be at least `0.02` for a PASS.

Two topology controls are evaluated separately: no-topology and wrong-topology
fits must each have a one-sided 95% lower bound of at least `-0.02` relative to
the unfused reference.  AUROC, coefficient error, context assignment, and jump
localization are secondary diagnostics and cannot replace the primary gate.

The formal campaign requires at least 200 complete paired replicates per
scenario cell.  Any cell below that support is `NE`, not PASS or FAIL.  A smoke
run may use fewer replicates, but its report must display `NE` and must not
change the frozen thresholds.

Graph fusion remains opt-in experimental unless the primary and both topology
controls pass and the independent biological/regression requirements are also
met.  A numerical G2 pass alone does not switch the public default, enable
formal inference, or establish biological superiority.

## Required lineage

Every result must bind the generator manifest, truth digest, scenario cell,
seed, graph digest, fitted family axis, solver status, held-out response/design
parent IDs, and fused/unfused variant. Failed or incomplete solver results are
retained as typed `not_estimable` and make the affected paired support
incomplete. They cannot be silently treated as zero recovery or dropped until a
nominally sufficient cell is made to pass.

## Consequences

The existing `G2_REPORT_CONTRACT.md` and graph-fusion evaluator implement this
decision.  The gate is a method-development decision, not a p-value or a
claim that graph-fused scores are comparable across receivers.  ADR-007,
ADR-009, ADR-011, and ADR-014 continue to govern receiver scale, directional
channels, context-common scoring, and source-agnostic signed Track-B output.
