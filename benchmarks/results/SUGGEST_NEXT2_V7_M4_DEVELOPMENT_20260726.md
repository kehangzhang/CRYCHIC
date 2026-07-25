# Suggest-next2 v7 M4 development benchmark

Status: **DEVELOPMENT COMPLETE; RELEASE REJECTED**.

This run evaluates the fold-fitted M4 occurrence candidate after the raw
activity threshold saturated in the preceding smoke campaign. It is parameter
development evidence, not locked or formal release evidence.

## Frozen run

- Estimator commit: `d78b22b934b6a118fa4a25f08d28353f99c5a4f2`
- Dirty worktree in all dataset manifests: `false`
- Protocol: `suggest_next2_v7_benchmark_v3.json`
- Protocol SHA256: `89ca902e09e4828ba910f8641e613b764fed72bfdcc2efcedc7419c670e1aae5`
- Protocol digest: `9b1e3e8a448ca1c66cfc02e639c542403acab2faecf0a6df40ed7710376d7dca`
- DGP: `occurrence_heterogeneity`
- Designs: 100 independent two-group seeds and 100 paired seeds
- Scale: eight subjects per condition, four cells per type, five candidate cell types
- Completion: 200/200 datasets, zero failed datasets
- Integrity: 5,800/5,800 dataset output files passed manifest SHA256 checks

The M4 state is `fold_fitted_parent_ecdf > 0.8`. Its probability-like output
uses `upper_tail_excess_v1` and remains explicitly labeled a working score.
Raw prevalence, Fisher, the protocol-compatible DCST proxy, and logistic
comparators are recomputed from an isolated `raw_activity_threshold` state;
changing the M4 state can no longer change comparator inputs.

## Ranking results

| Design | Method | AUPRC | AUROC | Direction | Effect coverage |
|---|---|---:|---:|---:|---:|
| Independent | CRYCHIC M4 | **0.4999** | **0.7475** | **0.9400** | 1.000 |
| Independent | Raw prevalence | 0.1429 | 0.5000 | 0.0000 | 1.000 |
| Independent | Fisher / DCST proxy | 0.1429 | 0.5000 | 0.0000 | 1.000 |
| Independent | Logistic / beta-binomial | NE | NE | NE | 0.000 |
| Paired | CRYCHIC M4 | **0.6526** | **0.8258** | **0.9300** | 1.000 |
| Paired | Raw prevalence | 0.1429 | 0.5000 | 0.0000 | 1.000 |
| Paired | Fisher / DCST proxy | NE | NE | NE | 0.000 |
| Paired | Logistic / beta-binomial | NE | NE | NE | 0.000 |

Against raw prevalence, M4 AUPRC improved by +0.3570 in the independent
design (95% paired-bootstrap CI +0.2915 to +0.4239; 89 wins, 11 ties) and by
+0.5097 in the paired design (CI +0.4425 to +0.5780; 94 wins, 6 ties).
AUROC improved by +0.2475 and +0.3258, respectively.

## Effect and probability diagnostics

| Design | M4 prevalence RMSE | Raw prevalence RMSE | M4 subject Brier | Raw subject Brier |
|---|---:|---:|---:|---:|
| Independent | 0.2025 | **0.1890** | **0.07230** | 0.96536 |
| Paired | 0.1958 | **0.1890** | **0.07221** | 0.96446 |

M4 won the Brier comparison on all 200 seeds. The raw comparator's Brier is
pathological because its fixed raw threshold calls nearly every subject-event
active. M4 prevalence-effect RMSE is nevertheless worse on average: the
zero-effect raw comparator has low squared error in this sparse one-positive,
six-negative truth axis, whereas M4 recovers the positive direction but also
introduces noisy null effects. The v7 DGP does not define a finite population
log-odds truth, so the requested log-odds RMSE gate is not evaluable here.

Pooled subject-level calibration remains inadequate:

| Design | Truth rate | Mean M4 score | Calibration intercept | Calibration slope |
|---|---:|---:|---:|---:|
| Independent | 0.03464 | 0.09873 | -2.0889 | 0.1570 |
| Paired | 0.03554 | 0.13049 | -2.0961 | 0.1852 |

Per-dataset calibration-slope means are misleading because sparse quantized
predictions produce extreme fits; the pooled slopes above are the primary
diagnostic. They are far from one, so `active_probability` is not released as
a calibrated posterior probability.

## Formal inference diagnostics

| Design | Formal p/q coverage, known events | Positive-event coverage | Type-I on known negatives | Power at q <= 0.10 |
|---|---:|---:|---:|---:|
| Independent | 0.7914 | 0.43 | 0.0000 | 0.0000 |
| Paired | 0.7743 | 0.39 | 0.0117 | 0.0000 |

All events retain descriptive prevalence effects, but separated or
single-class adjusted logistic fits correctly withhold formal p/q values. No
positive event survives global BH at q <= 0.10. With eight subjects per
condition and 35 tested parent events, this suite is underpowered for formal
discovery. These alternative-family type-I diagnostics do not replace the
required 1,000-replicate null calibration, and repeated designs remain pending.

## Development scan and decision

An offline scan used only persisted development-fold ECDF values. Raising the
threshold from 0.80 to 0.85 improved independent prevalence RMSE from 0.2025
to 0.1837, but reduced AUPRC from 0.4999 to 0.3760. In paired data, thresholds
0.80 and 0.85 produce the same quantized state and the same RMSE 0.1958. A
threshold of 0.90 collapses all paired states to absence. No single threshold
fixes both designs, so no threshold amendment is accepted from this scan.

The next isolated candidate is uncertainty-aware shrinkage of the prevalence
effect, using a preregistered Beta posterior grid on development data. It must
leave state ranking and formal tests unchanged, then pass a fresh development
rerun before any locked evaluation. Supervised calibration to synthetic truth
is explicitly excluded because it would encode this simulator's event
prevalence and would not justify real-data probability semantics.

Raw outputs:

```text
/media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/
  m4_occurrence_development_r100_v3_20260726
```
