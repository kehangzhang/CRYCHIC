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

## Multi-condition integration

`benchmarks.metrics.multicondition_rank_stability` applies these primitives to
subject-resampled differential rankings. The preregistered policy is RBO
`p=0.9`, weighted-Kendall power `1`, 200 split repeats, 2,000 subject
bootstraps, 95% intervals, a minimum selection frequency of `0.80`, and seed
`20260712`. It emits four report tables: agreement summaries, complete top-k
curves, bootstrap rank intervals, and stable tiers.

The score contract supports three always-available observed ranking grains:

- `lr`: interaction identity aggregated over comparison-eligible cell pairs;
- `sender`: sender identity aggregated over eligible receivers and LR items;
- `sender_receiver_pair`: a network pair aggregated over eligible LR items.

`sender_receiver_pair` is not called a biological LR family. The optional
`molecular_lr_equivalence_id` score column enables the compatibility-named
`lr_family` level (`k=1..25`). Its items are resource-independent molecular LR
equivalence classes, explicitly not strict target/driver families. Multiple
frozen source rows mapped to one molecular class form one equally weighted
ranking item; every frozen source member remains required in every
subject-context aggregate. With no mapping column, the historical result stays
explicit `not_estimable` with reason
`lr_family_mapping_not_available_in_score_contract`. LR curves cover `k=1..100`.
NicheNet Track B cannot emit LR or sender rankings, and unsupported
Kuppe/PancVAX designs retain explicit NE rows.

## Performance provenance

Finalizer adapter runs default to `performance_role=method_total`, preserving
the v1 behavior. Readback-only adapters must use
`performance_role=adapter_readback`; that elapsed time is retained as a
component but excluded from method runtime. A bound source runtime can be
declared with `performance_override.source_manifest`, an optional
`expected_sha256`, and `source_role` (`method_total`, `core_fit`, or
`source_pipeline_total`). This prevents a compact readback duration from being
reported as the complete method runtime.
