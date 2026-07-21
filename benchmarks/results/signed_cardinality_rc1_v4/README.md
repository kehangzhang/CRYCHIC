# RC1 directional signed-cardinality simulation v4

Status: **rejected for lack of ranking gain; no Kuppe/MS run performed**.

The direction-specific slab floor of 3.0 and fourth independent seed split were
committed at `05e6e0b` before this run.  All directional candidates passed the
development null gate.  The strongest directional candidate was
`directional_eb_delta_0.25`, with development/holdout pair Spearman 0.858/0.859
versus 0.861/0.862 for the symmetric RC1-v2 head.

The stronger identifiability constraint removed the v3 null failure but also
reduced sensitivity to small directional effects.  Directional delta 0.25
improved rare-effect scenarios but regressed in both small-effect scenarios;
its macro average did not beat the frozen symmetric candidate on development
or holdout.  The selected candidate was therefore unchanged, so another real
cohort evaluation would only reuse the same test data and was skipped.

Across v2-v4, soft expected cardinality robustly improves independent
effect/SE simulations and fixes much of the MS aggregate failure, but it cannot
match scSeqCommDiff on Kuppe/MS using the current CRYCHIC sample-level
representation.  Subsequent work should move to the preregistered strong
pseudobulk baseline plus residual-hypergraph fallback rather than continue
tuning this head.  All working probabilities remain benchmark-only.
