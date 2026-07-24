# Suggestions-next M2 locked validation

Status: **ACCEPT** on the frozen sender-specificity fixture. M2 remains an
experimental descriptive head and does not replace the public CRYCHIC score or
emit formal inference.

## Estimand and formula

M2 asks whether between-subject variation in a candidate sender's paired
ligand contrast tracks variation in the receiver's frozen-direction target
program after recorded nuisance adjustment. For each four-fold training split,
both effects are residualized on the same fixed design:

```text
intercept + batch_contrast + composition_design_contrast
          + candidate_sender_proportion_contrast
```

The head reports the signed residual correlation and
`max(correlation, 0)` as positive coupling support. Fold support is aggregated
by the median. Held-out subjects, truth labels, and generation latents never
enter a fit. The output is not a probability and has no p or q value.

## Stress-test design

Each paired cell-level dataset contains 20 subjects, two conditions, one
receiver, and four candidate senders for CXCL10-CXCR3:

| Candidate | Generating role |
|---|---|
| `TrueSender` | subject variation coupled to the receiver program |
| `BatchDecoy` | strong ligand contrast driven by recorded batch |
| `CompositionDecoy` | strong ligand contrast driven by recorded composition |
| `IndependentDecoy` | unrelated subject variation |

Mean group contrasts are deliberately similar across candidates, so M0/M1
group means cannot identify the sender reliably. Cell-level observations are
never treated as replicates; all coupling inputs are subject-level paired
contrasts.

## Development iterations

The retained development history includes unsuccessful alternatives:

1. A 12-subject smoke was not diagnostic because the true sender could already
   rank first under raw coupling.
2. A five-seed fixture using one raw ecosystem-proportion summary improved M2
   over raw coupling but reached only AP 0.617 and AUROC 0.600, with median true
   rank 2.
3. Adjusting four composition log-ratios directly reached AP 0.750 and AUROC
   0.767 but overused degrees of freedom and left unstable decoy correlations.
4. A training-fold PC1 of those log-ratios was worse (AP 0.467, AUROC 0.600)
   and was rejected.
5. The final design uses the recorded composition-design index plus observed
   sender proportion. No formula or gate changed after development freezing.

On the frozen 20-seed development panel (192,042 cells), M2 achieved AP 0.975,
AUROC 0.983, median true rank 1, and rank 1/4 on both metrics. Raw coupling
achieved AP 0.346 and AUROC 0.292. M2 ranked the true sender first in 19/20
seeds and passed all preregistered gates.

## Locked validation

Validation used 20 fresh root seeds (`20271201` through `20271220`), 191,683
cells, and clean evaluation commit `8d82ada`. The validation fixture manifest
SHA256 is
`4cf8d84779608f1c1cfffc7dc57980f804a353676db769cca8ae6fb9cfe25e6b`.

| Method | AP | AUROC | Median true rank | AP rank | AUROC rank |
|---|---:|---:|---:|---:|---:|
| M2 residualized coupling | **1.000** | **1.000** | **1** | **1/4** | **1/4** |
| M0 mean ligand contrast | 0.800 | 0.867 | 1 | 2/4 | 2/4 |
| M1 group program concordance | 0.800 | 0.867 | 1 | 2/4 | 2/4 |
| Raw sender-program coupling | 0.421 | 0.417 | 3 | 4/4 | 4/4 |

M2 ranked the true sender first in all 20 validation seeds. Relative to raw
coupling, the paired seed AP delta was +0.579 (95% bootstrap CI
[+0.479, +0.658]) and the AUROC delta was +0.583 (95% CI [+0.467, +0.700]).
It won 18 seeds and tied two on each metric. Candidate and fold coverage were
both 1.0. Mean true support was 0.799; the 95th percentile of the maximum
decoy-to-true ratio was 0.697, below the frozen 0.90 gate.

The 20 datasets ran with 20 workers in 11.48 seconds wall time. Median
single-dataset time was 4.93 seconds for about 9,584 cells; summed dataset CPU
time was 99.05 seconds.

Validation output:

```text
/media/subunit/bioinfo/crychic_dev/benchmark_work/
  suggestions_next_m0_iterations/m2_sender_validation_8d82ada_20260724
```

Output manifest SHA256:
`89abe24e86fd0039bb273f85c90934eacfd6c0e540a29cf1394917bfaa13b32c`.

## Claim boundary and next step

This validates sender specificity for one synthetic paired mechanism with
recorded batch and composition covariates. It does not establish robustness to
unmeasured confounding, a general SOTA advantage, calibrated activity
probability, active inhibition, or valid significance testing. M2 should next
be tested on real paired cohorts with recorded nuisance variables and on a
separate noisy/misspecified-covariate fixture. The next isolated architecture
cycle remains M4 occurrence; frozen hypergraph shrinkage follows only after
the occurrence head is stable.
