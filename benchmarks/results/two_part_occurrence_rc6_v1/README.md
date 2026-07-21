# RC6 subject-level occurrence and positive-magnitude hurdle

Status: **accepted for one-shot real-cohort benchmarking**. The candidate grid,
development and holdout seeds, simulation scenarios, and acceptance gates were
frozen at commit `3b427c5`. Commit `1da5f83` added post-hoc null opportunity
diagnostics without changing candidate selection or acceptance.

RC6 keeps the exact RC3 residual sign, LIANA call set, and receiver-program
weight. It adds two separately reported subject-level channels:

- occurrence: a Jeffreys-Beta posterior comparison of linkage presence;
- positive magnitude: a log-scale contrast among active subjects only.

Each channel is reliability-shrunk by sign agreement with the independent
baseline statistic. Missing evidence is neutral, antagonistic evidence closes
the channel, and the combined hurdle log weight is capped at 1.5. These are
benchmark ranking weights, not released communication probabilities.

## Frozen candidate

Development selected `two_part_20_05`: occurrence maximum alpha 2.0 and
positive-magnitude maximum alpha 0.5, with each effective channel capped at
1.0. Its development primary pair-rank Spearman delta versus RC3 was `+0.02484`.

## Independent holdout acceptance

| Check | Result | Gate | Pass |
|---|---:|---:|---|
| Primary-scenario delta versus RC3 | +0.03218 | at least +0.002 | yes |
| Occurrence-scenario delta versus RC3 | +0.03417 | at least +0.001 | yes |
| Magnitude-scenario delta versus RC3 | +0.01224 | at least 0.000 | yes |
| Safety-scenario delta versus RC3 | +0.00136 | at least -0.003 | yes |
| Structural-missingness delta versus RC3 | +0.03259 | at least -0.003 | yes |
| Global-null total effective-alpha q95 | 0.06889 | at most 0.25 | yes |

The safety average covers abundance-only, noisy, and antagonistic two-part
evidence. Structural absence remains missing rather than being converted to
zero.

## Post-hoc null opportunity audit

The audit was added because RC1 had previously exposed a null opportunity
failure. Across the 20 frozen holdout null seeds and both directions, RC6/RC3
mean false pair-score ratio was `1.00052`; the mean change in pair-score versus
LR-opportunity Spearman was `+0.000116`. This audit did not select the candidate
or alter acceptance, but it found no material amplification of the prior RC1
failure mode.

## Scope

The simulation operates on effect summaries and synthetic subject hurdle
observations rather than rerunning the complete raw-count pseudobulk pipeline.
Real Kuppe/MS evaluation is therefore allowed once, but scientific inference
and formal posterior release remain disabled.
