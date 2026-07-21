# RC2 LIANA baseline plus hypergraph residual simulation v1

Status: **accepted for checksum-bound real-cohort evaluation**.

The solver, edge-fold CV policy, conservative sign correction, scenarios, and
development/holdout seeds were committed at `d6fb756` before this run.  Truth
labels are used only for final scoring; lambda and CRYCHIC-anchor weights are
selected from held-out baseline-statistic prediction within each simulated
dataset.

Across the four structured holdout scenarios, the residual head improved mean
pair-rank Spearman by 0.00424 over the strong baseline.  The gain was positive
with a helpful anchor (+0.00556), noisy/antagonistic anchor (+0.00376), and
topology jumps (+0.00389).  Wrong topology triggered exact fallback (mean
difference 0), and the global-null fallback-gate 95th percentile was 0.

The effect is intentionally conservative because only edges with
`|baseline statistic| <= 1` may have their direction corrected.  The candidate
is accepted for a one-time Kuppe/MS evaluation, but the simulation operates at
the persisted edge-statistic layer rather than rerunning the full pseudobulk
pipeline.  It is a benchmark head, not released scientific inference.
