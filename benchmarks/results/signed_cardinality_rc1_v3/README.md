# RC1 directional signed-cardinality simulation v3

Status: **directional candidate rejected by the preregistered null gate; no
Kuppe/MS run performed**.

The three-component implementation, asymmetric scenarios, candidate grid, and
new development/holdout seeds were committed at `56658c1` before this run.

`directional_eb_delta_0.5` had the best unconstrained ranking performance:
development/holdout pair Spearman 0.870/0.868 versus 0.860/0.860 for the frozen
symmetric RC1-v2 head.  It improved all four directional scale/prevalence
imbalance scenario families on holdout.

However, its development pure-null false expected-count rate had a 95th
percentile of 0.00185, above the preregistered 0.001 gate; holdout was also
above the gate at 0.00169.  The eligible winner therefore remained
`symmetric_eb_delta_0.5`.  Re-running that unchanged head on Kuppe/MS would add
no information and was skipped.

Additional pure-null numerical diagnostics, separate from the v3 seeds, show
that raising only the directional slab identifiability floor from 1.5 to 3.0
reduces the 95th-percentile false rate to 7.49e-5.  A new v4 must preregister
that constraint and use new development/holdout seeds before any real-cohort
evaluation.  All probabilities remain benchmark-only and unreleased.
