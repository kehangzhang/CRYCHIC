# Suggestions-next M5 bounded-rewiring locked validation

Status: **ACCEPT** on the frozen synthetic additive H-prior effect-summary
fixture. This closes the preregistered 10%--25% prior-rewiring robustness gate;
it does not establish real-data robustness or formal inference.

## Frozen perturbation contract

M5's fitted formula and penalties are unchanged. For each requested fraction,
one deterministic subset of frozen edges is selected before outcomes are read.
Within that subset, membership stubs are independently rearranged for sender,
ligand, receptor, receiver, and pathway. The operation preserves every node's
degree exactly in every view and never changes edge IDs or the hypothesis
universe.

| Item | Frozen value |
|---|---|
| Partial-rewiring implementation | `25a93fe` |
| Evaluator and development contract | `c345f88` |
| Validation config freeze | `75ab19d` |
| Requested fractions | 10%, 25%, 50% |
| Gated fractions | 10%, 25% |
| Rewiring seed base | `20260740` |
| Full-permutation seed | `8675309` |
| Development seeds | `20290101`--`20290120` |
| Validation seeds | `20290901`--`20290920` |
| Validation fixture manifest | `e8e186c2c5b4ef012f23be2dfc4ccacda5ba041c7f58bf46c22558f6820f5403` |
| Frozen topology SHA256 | `a95e0bb52320581995a6423255dffbee711410cef1ee794d69b3e90c9a2dddc9` |

The 10% and 25% candidates had to satisfy all of the following without gate
changes after development:

- paired MSE improvement over raw and fully permuted priors with CI upper
  bound below zero;
- paired AP improvement over raw and fully permuted priors with CI lower bound
  above zero;
- at least 70% and 50%, respectively, of the correct-topology MSE and AP gain
  over the fully permuted prior;
- exact requested edge fraction within 0.001 and exact per-view degree parity.

The 50% point was frozen as a dose-response diagnostic only.

## Development

All 15 development checks passed. MSE topology-gain retention at 10%, 25%,
and 50% rewiring was 0.790, 0.548, and 0.249; AP retention was 0.815, 0.571,
and 0.264. The candidate therefore proceeded without changing the algorithm,
fractions, seeds, thresholds, or metrics.

## Fresh locked validation

Validation used 20 new seeds and 30,000 new edge truths generated from clean
commit `c345f88`. Evaluation ran from clean commit `75ab19d`.

| Prior | MSE | Spearman | AP | AUROC | Direction | MSE/AP rank |
|---|---:|---:|---:|---:|---:|---:|
| Correct H-prior | **0.3202** | **0.8495** | **0.7444** | **0.8489** | **0.8309** | **1/6** |
| 10% degree-matched rewire | 0.3746 | 0.8081 | 0.7007 | 0.8146 | 0.8112 | 2/6 |
| 25% degree-matched rewire | 0.4407 | 0.7535 | 0.6437 | 0.7685 | 0.7841 | 3/6 |
| 50% degree-matched rewire | 0.5221 | 0.6843 | 0.5663 | 0.7151 | 0.7492 | 4/6 |
| Fully permuted H-prior | 0.5883 | 0.6298 | 0.5067 | 0.6726 | 0.7265 | 5/6 MSE, 6/6 AP |
| No-prior raw effect | 1.3372 | 0.6445 | 0.5172 | 0.6810 | 0.7317 | 6/6 MSE, 5/6 AP |

The exact changed-edge fractions were 0.10, 0.25, and 0.50. Membership-level
fractions were 0.0952, 0.2311, and 0.4575 because repeated labels can remain in
place. Every per-view degree profile was identical to the correct prior.

| Rewire | MSE gain retained | AP gain retained | MSE delta vs permuted (95% CI) | AP delta vs permuted (95% CI) |
|---:|---:|---:|---:|---:|
| 10% | 0.797 | 0.816 | -0.2137 [-0.2229, -0.2051] | +0.1941 [+0.1865, +0.2020] |
| 25% | 0.551 | 0.577 | -0.1476 [-0.1558, -0.1389] | +0.1371 [+0.1302, +0.1436] |
| 50% diagnostic | 0.247 | 0.251 | -0.0662 [-0.0730, -0.0602] | +0.0596 [+0.0541, +0.0653] |

The 10% and 25% priors also beat raw effects on both metrics in all 20 seeds.
For 25% rewiring, MSE delta versus raw was -0.8965
(95% CI [-0.9131, -0.8794]) and AP delta was +0.1266
(95% CI [+0.1201, +0.1328]). All 15 validation checks passed. Runtime was
4.71 seconds.

Validation output:

```text
/media/subunit/bioinfo/crychic_dev/benchmark_work/
  suggestions_next_m0_iterations/m5_rewiring_validation_75ab19d_20260724
```

Output manifest SHA256:
`45cf4704c4ba9c542daddd00453e22639f5b9a474c1791840fb62ee6dd2cbb89`.

## Claim boundary

The result shows a reproducible topology-dose response and retained benefit at
10%--25% corruption under the additive synthetic mechanism used for M5. It
does not show robustness to real resource errors, missing nodes, biased prior
confidence, non-additive biology, or cohort shift. The same frozen topology
and rewiring maps are shared across seeds, while outcomes are newly generated.
No p value, q value, calibrated probability, or published-method comparison is
produced. Real subject-level effect integration remains required before M5 can
be a public default.
