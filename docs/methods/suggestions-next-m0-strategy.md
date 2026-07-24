# CRYCHIC suggestions-next M0 optimization strategy

Status: experimental, benchmark-only candidate. This document does not change
the public default score or enable formal inference.

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
