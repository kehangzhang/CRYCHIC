# Suggestions-next M4 locked validation

Status: **ACCEPT** on the frozen independent-subject occurrence fixture. M4 is
an experimental prevalence head and does not replace continuous communication
strength or release formal inference.

## Estimand

M4 consumes one explicit binary linkage call per `event x subject x condition`.
Missing calls remain missing and never become absence. For each independent
group it reports a Jeffreys-Beta posterior prevalence, then keeps these outputs
separate:

- target-minus-reference prevalence difference;
- posterior log-odds ratio;
- descriptive posterior-standardized prevalence difference for ranking;
- coverage and typed not-estimable status.

The standardized value is not converted to a p value. M4 emits no q value and
no event activity probability. ADR-013 `comm_probability` remains a separate,
active-null/G3-P-gated estimand and is not reused here.

## Fixture and baselines

Each seed contains 300 events and 16 independent subjects per group. The 90
true occurrence events comprise prevalence increases, decreases, and a
structural-missingness family. The 210 negative events comprise magnitude-only,
abundance-only, and global-null families. Truth is stored separately from the
method input.

The locked comparison includes:

- continuous M0 activity contrast;
- raw subject prevalence difference;
- a two-sided Fisher exact occurrence baseline with DCST-style semantics;
- M4 posterior-standardized prevalence.

The Fisher baseline is not a run of the full DCST package and is labeled as
such in every manifest.

## Development

The fixed candidate uses Jeffreys prior 0.5 and requires eight observed
subjects per group. The formula was not tuned on seed metrics. A five-seed
smoke confirmed that occurrence was identifiable separately from continuous
magnitude and that missingness degraded coverage rather than becoming zero.

On 20 frozen development seeds (6,000 events), M4 achieved AP 0.947 and AUROC
0.968. Continuous M0 achieved AP 0.411 and AUROC 0.701. M4 prevalence Brier was
0.01139 versus 0.01275 for raw prevalence, and log-odds RMSE was 0.854 versus
0.916.

## Locked validation

Validation used 20 fresh root seeds (`20280501` through `20280520`) and 6,000
new events. The validation fixture was generated from clean commit `97bae8c`;
its manifest SHA256 is
`b90ffccc0dc097d6eeea1c935f910c193f9b4a2763ecd7fe856bd33e8a54def1`.
Evaluation used clean commit `9138ea0`.

| Method | AP | AUROC | Direction | Coverage | AP rank | AUROC rank |
|---|---:|---:|---:|---:|---:|---:|
| Raw prevalence difference | **0.952** | **0.971** | 0.998 | 1.000 | **1/4** | **1/4** |
| M4 posterior prevalence | 0.943 | 0.965 | **0.998** | 0.9985 | 2/4 | 2/4 |
| Fisher exact, DCST-style | 0.939 | 0.964 | 0.998 | 1.000 | 3/4 | 3/4 |
| Continuous M0 contrast | 0.396 | 0.690 | 0.996 | 1.000 | 4/4 | 4/4 |

M4 beat continuous M0 in all 20 seeds. Its paired AP delta was +0.547
(95% bootstrap CI [+0.537, +0.557]); AUROC delta was +0.274
(95% CI [+0.266, +0.283]). M4 also exceeded the Fisher baseline by AP +0.0040
(95% CI [+0.0026, +0.0054]).

Raw prevalence difference remained the strongest pure ranker. M4's added value
was calibrated shrinkage: prevalence Brier improved by -0.00131
(95% CI [-0.00137, -0.00126]) and log-odds RMSE by -0.0601
(95% CI [-0.0651, -0.0551]) relative to raw estimates. Minimum per-seed M4
coverage was 0.9967 and minimum direction accuracy was 0.9889. All eight locked
gates passed. Runtime was 16.98 seconds.

Validation output:

```text
/media/subunit/bioinfo/crychic_dev/benchmark_work/
  suggestions_next_m0_iterations/m4_occurrence_validation_9138ea0_20260724
```

Output manifest SHA256:
`b9544e0eb176f437979eac8dbc54eb127b027df6b3cd829a3959fc22f8e1a721`.

## Claim boundary

This result validates descriptive prevalence estimation for supplied binary
linkage calls in synthetic independent groups. It does not validate how binary
calls are derived from raw expression, paired/repeated designs, an actual DCST
software run, M4 p/q values, or a general SOTA claim. Raw prevalence should
remain a mandatory ranking baseline. The next isolated cycle is M5 frozen
hypergraph shrinkage with a degree-matched topology-permutation control.
