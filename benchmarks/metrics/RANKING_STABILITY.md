# Ranking stability diagnostics

`benchmarks.metrics.ranking_stability` adds top-sensitive diagnostics for
resampled edge or family rankings. These are descriptive benchmark estimands;
they are not calibrated probabilities or inferential p-values.

Every `Ranking` declares the same ordered frozen-universe contract and one of
two states. Both membership and order must match across replicates, which keeps
output ordering deterministic and makes preregistration drift a hard error:

- `observed`: `items` is an ordered, possibly partial list. Universe members
  absent from the list are explicit missing selections.
- `not_estimable`: no numeric ranking was produced and a `reason_code` is
  mandatory. Metrics propagate this state and return `estimate=None`.

The public diagnostics are:

- `rank_biased_overlap`: normalized finite-list RBO, with geometric emphasis on
  early prefixes. Missing suffixes are not imputed.
- `weighted_kendall_tau`: symmetric top-weighted tau-b. Missing universe
  members are tied at the bottom, and tie mass remains in the denominator.
- `top_k_stability_curve`: top-k Jaccard at every cutoff. A cutoff beyond either
  observed list is `not_estimable`, rather than filled with arbitrary items.
- `bootstrap_rank_intervals`: percentile intervals conditional on selection,
  accompanied by unconditional selection frequency, missing counts, and the
  number of entire non-estimable replicates.
- `assign_stable_tiers`: labels items as `stable_top_k`, `possible_top_k`,
  `stable_below_top_k`, `unstable`, or `not_estimable` using a preregistered
  top-k cutoff and minimum selection frequency.

Example:

```python
from benchmarks.metrics import (
    Ranking,
    assign_stable_tiers,
    bootstrap_rank_intervals,
    rank_biased_overlap,
    top_k_stability_curve,
    weighted_kendall_tau,
)

universe = ("family_A", "family_B", "family_C")
left = Ranking.observed(("family_A", "family_B"), universe=universe)
right = Ranking.observed(("family_A", "family_C"), universe=universe)

rbo = rank_biased_overlap(left, right, persistence=0.9)
tau = weighted_kendall_tau(left, right)
curve = top_k_stability_curve(left, right)
intervals = bootstrap_rank_intervals((left, right), confidence=0.95)
tiers = assign_stable_tiers(intervals, top_k=1)
```

The frozen universe, RBO persistence, Kendall weight power, interval confidence,
minimum observed ranks, top-k cutoff, and selection-frequency threshold must be
recorded in the benchmark configuration before interpreting the output.
