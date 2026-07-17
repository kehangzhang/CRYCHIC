# ADR-012: Frozen hypothesis universe and two-level selective FDR

- Status: Accepted method contract; formal q release remains G3-F gated
- Date: 2026-07-15
- Accepted scope: universe identity, primary/secondary roles, pre-fit filtering,
  exact coverage, the primary omnibus, the only hierarchical procedure, nominal
  level, collection-level q-value semantics, and paired power/interval-width
  noninferiority operating characteristics.
- Release status: not authorized until procedure- and universe-bound G3-F
  calibration passes. This ADR does not itself release q-values.

## Context

Formal G3-F inference requires one hypothesis universe fixed before context
effects, resamples, p-values, or q-values are inspected. Fold-specific family
selection, filtering, and failed resamples must not remove hypotheses or shrink
a multiplicity denominator. Ordinary BH after selecting families by an omnibus
does not by itself control the intended error rate.

The first accepted procedure is intentionally narrow. Its primary layer is the
gene-view, state-mode `driver_family x receiver` context omnibus. Its secondary
layer contains only preregistered, same-family, same-receiver, same-mode context
contrasts that are logical post-hoc children of that omnibus. Sender-resolved,
ecosystem, TF/pathway, signed Track-B, and cross-receiver pooled hypotheses are
not members of this first hierarchy and cannot borrow its q-value release.

## Frozen Universe

Each declaration binds endpoint, contrast, receiver, driver-family ID, mode,
role, multiplicity family, and an optional primary parent key. Primary
hypotheses have no parent. Every secondary has exactly one primary parent for
the same receiver, family, and mode. Duplicate keys, dangling parents,
secondary-to-secondary parentage, and scope mismatches fail closed.

Only pre-fit, outcome-independent filtering is allowed. Reviewed policies are:

- no prefilter;
- an external-resource prefilter;
- pooled context-label-blind support.

Prefiltered rows remain in the frozen universe and exact coverage. A child
cannot remain included when its parent is prefiltered. Every result collection
contains exactly one typed `observed`, `not_estimable`, or `failed` row per
hypothesis. Missing, extra, or duplicate rows are contract violations.

## Primary Omnibus

For a primary with `J >= 2` frozen contexts, fit the subject-equal unpenalized
OOF WLS model with fold nuisance columns and CR2 diagnostic covariance already
used by the effect backend. Let `mu_hat` be the context estimates and let `H`
be the canonical Helmert basis with rank `J - 1`:

```text
theta = H mu_hat
W = theta' (H V_CR2 H')^-1 theta
```

`H V_CR2 H'` must be finite, positive definite, and full rank. The producer
must not use a pseudoinverse to turn a singular omnibus into an observed test.
Rank failure, incomplete OOF coverage, insufficient clusters, or unavailable
CR2 covariance yields typed `not_estimable`.

The formal primary p-value is empirical. Every legal context permutation reruns
the complete pipeline and the same omnibus statistic:

```text
p_primary = (1 + count(W_permuted >= W_observed)) / (B + 1)
```

At least 1,000 complete, synchronized permutation plans are required. Analytic
chi-square or CR2 p-values are not a release substitute.

Secondary p-values use the existing full-pipeline empirical permutation test
for their preregistered contrast and direction. Every primary and child in one
hierarchy must bind the exact same context-permutation plan ID set. A secondary
uses its own frozen contrast weights, but its sample-level family scores come
from the exact primary parent's scoring functional and application. Refitting
or substituting a child-specific score scale is not a valid post-hoc test.

## Unique Hierarchical Procedure

The only accepted v1 procedure is two-level TreeBH / Benjamini-Bogomolov
selective FDR with nominal `alpha = 0.05`:

1. Let `M` be the total number of frozen primary hypotheses, including
   prefiltered primaries. Apply BH at `alpha` to all primary p-values and let
   `R` be the number selected.
2. Only within a selected primary, apply BH to all of its frozen children at
   `alpha_child = alpha * R / M`.
3. Prefiltered hypotheses use internal effective p-value 1 and remain in every
   denominator. Their public p and q fields remain `NA`.
4. Any included primary or child that is `not_estimable` or `failed` blocks q
   release for the complete collection. Candidate calculations may be retained
   for debugging, but no subset may be released.

Primary q-values are ordinary BH adjusted p-values. For child `h`, the
hierarchical q-value is the smallest level `a` at which both conditions hold:

```text
parent(h) is selected by primary BH(a)
within_parent_BH_q(h) <= a * R(a) / M
```

The implementation scans the finite primary/child breakpoints; it does not use
floating-point binary search or run-time procedure switching. A child can never
be rejected when its parent is not selected.

The controlled child-layer quantity is the expected average FDP across selected
parent families (selective FDR). It is not pooled leaf FDR or FWER. Every output
must carry `q_value_scope=selective_fdr_not_pooled_leaf_fdr`.

## Release Boundary

### Paired noninferiority contract

The calibration protocol must name one non-empty, preregistered baseline ID
before any campaign replicate is generated. The baseline is evaluated on the
same simulated dataset, frozen hypothesis universe, truth ledger, and nominal
alpha as the candidate method. Its hypothesis-level rejection decisions and CI
widths are raw replicate inputs; callers cannot submit power, width summaries,
Monte Carlo bounds, evidence, or a gate.

The accepted v1 power estimand is computed only in mixed scenarios. Within each
replicate having at least one included true non-null hypothesis, it is

```text
candidate true-positive rejection rate - baseline true-positive rejection rate
```

Hypotheses are equally weighted within an informative replicate and informative
replicates are equally weighted. The point estimate and one-sided 95% lower
bound use 10,000 whole-replicate bootstrap draws. The frozen direction is
larger-is-better and the absolute noninferiority margin is `-0.05`.

The accepted v1 interval-width estimand is computed in global-null and mixed
scenarios. For every included hypothesis it is

```text
(baseline CI width - candidate CI width) / baseline CI width
```

The producer averages hypotheses within a replicate and then equally weights
replicates. It uses the same 10,000-draw whole-replicate bootstrap and a
one-sided 95% lower bound. The frozen direction is larger-is-better and the
relative noninferiority margin is `-0.10`. Baseline widths must be finite and
strictly positive; candidate widths must be finite and nonnegative. Widths for
prefiltered hypotheses are absent, while every included hypothesis must have a
paired candidate/baseline width. Incomplete or unpaired width coverage makes
the complete scenario not estimable.

The baseline ID, both estimand IDs, directions, margins, bootstrap size,
confidence level, universe, procedure, generator and seed lineage are part of
the protocol identity. Each producer-owned scenario repeats the applicable
baseline, estimand, direction and margin. Evidence and the final gate bind the
same contract and fail closed if any scenario is missing, duplicated or
inconsistent. A global-null cell has no power estimand; every mixed cell must
have both power and interval-width results.

The complete collection may expose formal q-values only when all of the
following are true:

- exact universe coverage is producer-derived from intact point and resampled
  results;
- every included primary and child has a complete formal empirical p-value;
- primary and child permutation plan IDs match exactly;
- a producer-owned G3-F gate is passed and binds the exact universe ID,
  hierarchical procedure ID, and preregistered protocol ID;
- at least 1,000 replicates cover the frozen 0.5%, 1%, 5%, and 10% non-null
  prevalence grid and the preregistered dependence structures;
- the exact dependence structures are `independent_hypotheses`,
  `positive_within_family_block`, and `negative_within_family_block`; each is
  crossed with a global-null cell and every mixed-prevalence cell;
- the one-sided mixed, primary, and selective-child FDR upper bounds are each no
  greater than 0.07 at nominal 0.05; the global-null primary-family type-I upper
  bound is no greater than 0.06, the 95% interval-coverage lower bound is at
  least 0.92, every mixed-cell power noninferiority lower bound is at least
  `-0.05`, every global-null and mixed-cell interval-width noninferiority lower
  bound is at least `-0.10`, and each dependence-specific permutation
  distribution satisfies its frozen 0.05 DKW band; and
- public raw-ledger summarization is diagnostic only. Release additionally
  requires an internally verified, replayable generator attestation; a caller
  string, seed label, or source artifact ID cannot self-attest the generator.

Until that boundary is met, collection status is candidate-only or blocked and
public `q_value` remains `None`. Single-hypothesis effect and omnibus objects
always keep `q_value=None`; adjustment is only valid as a complete collection.
Failure does not trigger Holm, ordinary BH, stageR, or any other fallback.

TreeBH relies on valid super-uniform permutation p-values, preregistered
exchangeability, logical parent-child nesting, and the dependence conditions
covered by the G3 mixed-correlation simulation. If the frozen campaign fails,
the procedure remains disabled and this ADR must be revised before a different
method is implemented.

## Consequences

The implementation can compute auditable candidate omnibus and hierarchical
values before the release campaign, while making it impossible for a caller
boolean, a single passed effect gate, a filtered subset, or a different
universe to unlock q-values. `comm_probability` remains governed independently
by G3-P and ADR-013.
