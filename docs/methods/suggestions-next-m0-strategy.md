# CRYCHIC suggestions-next M0 optimization strategy

Status: accepted on the locked expression-driven validation fixture, but still
experimental and benchmark-only. This document does not change the public
default score or enable formal inference.

## Objective

The first optimization cycle addresses the failure mode identified in
`suggestions_next.md`: the current score retains sparse Top-K signal but loses
absolute cross-sample LR intensity and global negative-event ordering. RC12 and
RC14 remain frozen diagnostic baselines.

The cycle changes one statistical mechanism only: it introduces a bounded,
log-additive absolute ligand/receptor activity head and separates sender
detection from conditional sender attribution. Downstream programs, inference,
and hypergraph regularization remain unchanged.

## Root-cause evidence

The development fixture uses CPM-scale pseudobulk expression, while the legacy
availability transform uses a Hill half-saturation value of 1. In one active
replicate, planted-event genes had approximately 40,000 to 350,000 CPM. Their
legacy ligand/receptor availability values therefore saturated near 0.97 to
0.99, compressing planted 3-fold to 8-fold changes into changes of roughly one
percentage point.

On the frozen 20-seed development campaign:

| Development head | Omnibus AUPRC | Omnibus AUROC |
|---|---:|---:|
| RC12 sender/response 90/10 | 0.1350 | 0.5417 |
| RC12 plus 0.1 legacy availability | 0.0746 | 0.3647 |
| M0 fixed log-additive L/R 50/50 prototype | 0.6156 | 0.9484 |

The failed legacy-availability addition is retained as negative development
evidence. It rules out simple reweighting of the saturated component.

## Frozen M0 equations

For non-negative pseudobulk expression `x` on the CPM scale, define descriptive
evidence

```text
E(x) = clip(log1p(x) / log1p(1,000,000), 0, 1).
```

The reference is fixed before seeing outcomes. It is the natural upper bound of
one gene on a CPM scale and is not fitted to group labels. For required ligand
or receptor complexes, subunit evidence is combined locally with the existing
generalized harmonic soft minimum.

For sample `s`, receiver `r`, interaction `e`, and candidate sender `u`:

```text
sender_detection[s,e,u] = 0.5 E(L[s,e,u]) + 0.5 E(R[s,e,r])
parent_activity_raw[s,e] = max_u sender_detection[s,e,u]
```

`sender_detection` is absolute and does not conserve parent mass. The existing
softmax weight is retained only as conditional `sender_attribution`, which sums
to one within a parent event. A null-aware display allocation is defined by

```text
active_sender_mass[s,e] = max_u E(L[s,e,u])
null_sender_attribution[s,e] = 1 - active_sender_mass[s,e]
sender_attribution_with_null[s,e,u]
    = active_sender_mass[s,e] * sender_attribution[s,e,u]
sender_mass[s,e,u] = parent_activity_raw[s,e] * sender_attribution[s,e,u].
```

Thus low evidence can remain mostly in the null sender class, while detection is
not diluted when unrelated candidate senders are added. Legacy mechanistic
availability is retained separately as `mechanism_support`; it does not multiply
or hard-zero M0 detection.

## Data separation and iteration

1. Development uses only seeds 20260801-20260820 and source-evaluation manifest
   SHA256 `ce5140f55759b9175fdec8773c2f00be78ba4faded91d948b85efa806d1d6bcf`.
2. M0 weights are fixed at 0.5/0.5 from the additive model definition. No fine
   weight grid is selected against validation data.
3. Locked validation uses new seeds 20261201-20261220 with the same unequal
   8/10/12-subject design and H-common resource.
4. A failed validation is retained and triggers a root-cause revision on the
   development data. Validation weights are never retuned in place.
5. Passing this expression-driven fixture is necessary but not sufficient.
   Later cycles must use separate ligand-only null, receptor-only null,
   downstream-only, inhibitory, occurrence, decoy-sender, complex-AND, and
   prior-misspecification mechanism families.

## M0 release gates

All intervals use paired bootstrap over simulation seed, never over event rows.

- AUPRC: candidate minus RC12 lower 95% bound must be at least -0.01.
- AUROC: candidate minus RC12 lower 95% bound must be greater than 0.
- Direction: candidate accuracy must not be more than 0.02 below the retained
  canonical signed head.
- Coverage: evaluated event coverage must be at least 0.80.
- Geometry: candidate structural-zero and tie rates must be lower than the
  canonical mechanistic head.
- Sender invariants: adding an irrelevant candidate cannot change existing
  detection; conditional attribution must sum to one; null-aware attribution
  plus null must sum to one; low absolute sender evidence must favor null.
- Claim boundary: M0 emits no calibrated probability, p-value, or q-value and
  does not replace the public communication-strength default.

After this gate, the next single-mechanism cycles are signed downstream M1,
residualized coupling M2, occurrence M4, and frozen hypergraph shrinkage M5.

## Locked validation result

The preregistered validation fixture used fresh seeds 20261201--20261220,
8/10/12 independent subjects per condition, and a mean of 180 cells per
sample. The fixture manifest SHA256 is
`afbf43f29ed2e9214ef0e920efb4751dc05f97bc3eaff506be0efea48e28ac6c`.
All 40 legacy CRYCHIC runs were generated from clean implementation commit
`209e6a3`; the source-evaluation manifest SHA256 is
`a725ef7a854dfda7a201d24b7c2139bad87670a0fd6e7cc4e4024f2b0669efd6`.

M0 passed every preregistered gate (`ACCEPT`): AUPRC 0.6112, AUROC 0.9472,
localization AP 0.6578, direction accuracy 1.0000, zero fraction 0, and tie
fraction 0.0074. Relative to RC12, the paired seed AUPRC delta was +0.4452
(95% CI [0.3927, 0.4912]) and AUROC delta was +0.3980 (95% CI
[0.3188, 0.4752]); both won all 20 seeds. The validation manifest SHA256 is
`b0378ed07dab013991b5c613473fa5344f66bcfb7f1d6239992d69a63bd90c31`.

On the same fixture and H-common resource, an external panel of CellChat,
LIANA, and scSeqCommDiff also completed. M0 ranked 1/5 on prevalence-adjusted
AP (0.6859), AUPRC (0.6112), AUROC (0.9472), localization AP (0.6578),
direction accuracy (1.0000), and effect Spearman (0.3614). The external-panel
evaluation manifest SHA256 is
`85e0de732dd1c147b9137685187d5c29481c150381cd9f3c66e0ab78b39f3db8`.

This is evidence for the score geometry on an expression-driven mechanism
family only. It is not a general SOTA claim and does not establish behavior on
ligand-only, receptor-only, generic-downstream, inhibitory, occurrence,
decoy-sender, complex-AND, confounding, or hypergraph-prior mechanisms.

## Locked M1 mechanism-family result

M1 was evaluated as a separate signed target-program concordance head, without
changing M0 activity or the public default score. It uses no fitted parameter:
the fixed contrast diagnostic is the signed geometric mean of the M0 LR effect
and the frozen-direction receiver-program effect when their directions agree,
and zero otherwise.

Development used 20 root seeds from 20270101--20270120. Locked validation used
20 fresh root seeds from 20270501--20270520, seven paired mechanism families,
10 subjects, and a mean of 180 cells per sample. The validation fixture
manifest SHA256 is
`7660c9a63fb4f8cabd2be54acc869cd46783be9a717b1ddbd369a85f646fd640`;
the final validation manifest SHA256 is
`a2a1b9943feadd721cb5b0efb3c91f80c1cce6d80be782ed0a62a03f60a96ad2`.

M1 passed all locked gates: mechanism-family AP 1.000, AUROC 1.000, median
active rank 1, full active retention, full coverage, and a maximum partial to
active mean-score ratio of 0.197. M0 scored AP 0.350 and AUROC 0.683 on the
same estimand. M1 beat M0 on all 20 seeds for both metrics. In particular,
target-only and receptor-knockout had zero M1 concordance while their component
effects remained visible separately.

The benchmark also exposed incomplete candidate-sender coverage in one
development sample. Detection now remains available while conditional
attribution becomes typed not-estimable; remaining sender weights are not
renormalized. This preserves the intended M0 detection/attribution separation.

M1 remains an experimental contrast diagnostic and does not claim active
inhibition, probability calibration, or formal inference. The next isolated
cycle is M2 residualized coupling on decoy-sender, composition, and confounding
families. The full locked evidence and claim boundary are recorded in
`benchmarks/results/SUGGESTIONS_NEXT_M1_VALIDATION_20260724.md`.

## Locked M2 sender-specificity result

M2 adds one descriptive component: fold-training residual correlation between
the M0 candidate-sender ligand contrast and the M1 receiver-program contrast,
after adjustment for recorded batch, composition design, and candidate-sender
proportion changes. It retains signed correlation separately and uses only its
positive part as coupling support. It emits no probability, p value, or q
value.

Development used 20 root seeds from 20270801--20270820. Locked validation used
20 fresh seeds from 20271201--20271220, 20 paired subjects, four candidate
senders, and 191,683 cells. The fixture manifest SHA256 is
`4cf8d84779608f1c1cfffc7dc57980f804a353676db769cca8ae6fb9cfe25e6b`;
the validation output manifest SHA256 is
`89abe24e86fd0039bb273f85c90934eacfd6c0e540a29cf1394917bfaa13b32c`.

M2 passed all locked gates: AP 1.000, AUROC 1.000, true-sender rank 1 in all 20
seeds, complete candidate/fold coverage, and rank 1/4. Raw unadjusted coupling
scored AP 0.421 and AUROC 0.417. The paired M2-minus-raw AP delta was +0.579
(95% CI [+0.479, +0.658]); AUROC delta was +0.583 (95% CI
[+0.467, +0.700]). The full development history, including rejected raw
proportion, full-log-ratio, and PC1 nuisance alternatives, is retained in
`benchmarks/results/SUGGESTIONS_NEXT_M2_VALIDATION_20260724.md`.

M2 evidence is limited to a synthetic paired estimand with recorded nuisance
covariates. It does not establish behavior under unmeasured confounding or a
general SOTA claim. The next isolated cycle is M4 occurrence, followed by M5
frozen hypergraph shrinkage.
