# Figure 3 CRYCHIC extension (2026-07-22)

## Conclusion

The earlier CRYCHIC result was not evaluated with the corrected Figure 3
protocol. It used the former MS cohort, subject-level rankings, old spatial
truth, and an unweighted DES walk (`gseaParam=0`). It must not be compared to
the corrected paper-method leaderboard.

This extension now evaluates CRYCHIC under the same Figure-compatible DES
contract as the multi-sample paper methods:

- biological sample (`sample_id`) analysis unit;
- corrected Kuppe and 5 Ctrl + 6 CA MS cohorts;
- sample-level spatial truth with floor top-set sizes and self-pairs excluded;
- `scoreType="pos"`, `gseaParam=1`, raw cardinality, and fgsea-native ties;
- two conditions by 10%, 20%, 30%, and 40% expected sets (eight DES strata).

CRYCHIC was not a method in Cesaro et al. This is therefore a method extension
to the reconstructed Figure 3 panels, not a claim that the paper ran CRYCHIC.

## Sample-level CRYCHIC ranking

The adapter does not relabel or reuse a subject-level differential result. It
reads checksum-bound held-out score layers and semantic availability from the
two-fold CRYCHIC backbone, verifies that every sample occurs in exactly one
held-out fold, and recomputes the differential head with `sample_id` as the
unit. The ranking is the cardinality of one-standard-error-stable,
sender-specific HC2 LR effects after collapsing both communication directions
to an unordered cell-type pair.

Subject-blocked outer folds are retained, so libraries from the same subject
cannot leak across training and held-out application. Only the downstream
Figure 3 comparison unit is sample-level. Formal p-values or q-values are not
emitted.

## MS cohort and fold audit

The corrected MS input contains 69,168 nuclei, 5 Ctrl tissues, 6 CA tissues,
and 10 independent subjects. The previous 75,004-cell input included the
unmatched control CO45 and is not reused.

With the corrected cohort, outer partition seed `20260717` produced one
training fold in which `~ batch + lesion_type` was rank deficient. This was a
partition-specific failure, not a globally confounded cohort: a design-only
seed scan performed before spatial truth evaluation showed many estimable
partitions. Model and tuning seed `20260717` were retained, while the separately
recorded outer partition seed was set to `20260718`. Both training folds then
passed the full-rank and contrast-estimability gates.

The full fit used two concurrent outer folds and 16 BLAS threads per fold. It
completed in 38 minutes 32 seconds with 37.3 GiB peak RSS. A clean-commit
materialization from the checksum-keyed fold cache took 21 minutes 51 seconds.
All persisted numerical cross-fit tables were byte-identical between the two
materializations. The final source and replay are bound to commit `9003484` and
both report `dirty=false`.

## Results

Only checksum-compatible multi-sample panels are ranked below. The unchanged
condition-aware paper panels remain in the extension bundle for context.

| Dataset | Method | DES strata | Median DES | Mean DES | Median rank |
|---|---|---:|---:|---:|---:|
| Kuppe | scSeqCommDiff | 8/8 | 0.701 | 0.691 | 1 |
| Kuppe | LIANA+ | 8/8 | 0.490 | 0.470 | 2 |
| Kuppe | CRYCHIC | 8/8 | 0.428 | 0.437 | 3 |
| MS | scSeqCommDiff | 8/8 | 0.825 | 0.673 | 1 |
| MS | CRYCHIC | 8/8 | 0.050 | 0.069 | 2 |
| MS | LIANA+ | 6/8 | 0.500 | 0.401 | not ranked |

The MS rank of 2 means second among methods with all eight observed strata; it
does not mean that CRYCHIC's DES exceeds the incomplete LIANA+ point estimate.

The low MS score is not an all-zero sender-output failure. Both conditions
have non-degenerate rankings: 28 of 45 unordered pairs are observed, with 28
nonzero rows in each condition. The condition/fraction DES values are:

| Condition | 10% | 20% | 30% | 40% |
|---|---:|---:|---:|---:|
| chronic active | 0.100 | 0.100 | 0.100 | 0.251 |
| control | 0.000 | 0.000 | 0.000 | 0.000 |

CRYCHIC and scSeqCommDiff have the same minimum eligible-pair coverage in the
MS panel (0.583) and the same minimum expected-set coverage (0.333). Thus the
large DES difference is primarily an ordering/alignment result, not merely a
coverage difference. CRYCHIC's CA ranking also has a high correlation with the
number of estimable LR opportunities, which is an important diagnostic for
future ranking-head work.

## Artifacts

The portable, checksum-bound report is in
[`benchmarks/results/spatial_des_figure3_crychic_extension_20260722`](../../../benchmarks/results/spatial_des_figure3_crychic_extension_20260722/README.md).
It contains the unified summary, leaderboard, figures, and publication
manifest. The sample replay implementation is
[`benchmarks/adapters/crychic/replay_v3_sample.py`](../../../benchmarks/adapters/crychic/replay_v3_sample.py).

MultiNicheNet remains a declared skip because no validated frozen adapter is
available. It is not assigned a zero score.
