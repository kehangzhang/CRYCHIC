# ADR-013: Active-edge null and local-FDR probability

- Status: Accepted method contract; public probability release remains G3-P gated
- Date: 2026-07-15
- Accepted scope: active-edge estimand, score grain, candidate universe, the
  only primary null, matching and rewiring rules, local-FDR estimator,
  fallback behavior, and the G3-P release gate.
- Release status: not authorized until a null/spec/universe-bound G3-P
  calibration campaign passes. This ADR does not itself release
  `comm_probability`.

## Context

`comm_strength`, a context differential effect, specificity support, and an
active-edge posterior answer different questions. A communication strength is
not a probability. A context-label permutation tests a context effect, while a
sender-label shuffle tests sender allocation. Mixing either null with a
mechanism-matching null would make a local-FDR posterior uninterpretable.

CRYCHIC therefore needs one structure-preserving null that asks whether the
held-out sender-LR-receiver evidence could arise when an LR driver is connected
to a receiver target profile with no specific mechanistic match. The null must
rerun all learned nuisance stages. Relabeling an already fitted score table is
not admissible.

The current receiver-balanced score remains receiver-relative. ADR-007 does
not authorize pooling raw scores across receivers, even when all rows share a
collection parent. Probability calibration must consequently remain within a
receiver-specific stratum.

## Estimand And Score

For one frozen analysis collection, `comm_probability` means:

```text
P(edge belongs to the active component |
  observed expression, abundance, context,
  frozen resources, configuration, and G3-P stratum)
```

"Active" means that the edge's held-out integrated evidence is inconsistent
with the accepted no-mechanism-match LR-target component. It is not a physical
binding probability, a causal sender probability, or the probability of a
context difference.

The internal edge key is:

```text
contrast_id, context_id, mode, receiver, sender, interaction_id
```

`contrast_id` binds the unique contrast-common scoring collection used to
construct the score. The public biological projection is a context,
sender-LR-receiver edge; the same context edge from different contrast
functionals cannot be merged or treated as duplicate evidence.

The statistic is the v4 conserved `global_sender_lr_score`. For each held-out
`sample x subject x context x receiver x interaction x mode` parent, the frozen
common-sender functional supplies assignment weights over the complete sender
candidate set and the sender rows exactly conserve the prior-adjusted
sender-unresolved `global_lr_score`. The statistic is reduced as follows:

1. average technical samples within `subject x context x edge`;
2. average repeated-cross-fit values within that same subject-edge unit; and
3. average subjects with equal weight.

Cell counts and numbers of technical samples never change a subject's weight.
The score source, score version, collection, fold applications, OOF audit,
input, configuration, LR resource, and original target prior are all bound to
the active-edge universe.

The v3 statistic based on unnormalised `ligand_availability x
training_prevalence_prior` evidence is retired. Its point, null, and probability
artifacts are not compatible with v4 and must be regenerated from raw fold
scopes. Raw sender evidence remains diagnostic; it is not the allocation weight.

## Frozen Candidate Universe

The candidate universe is frozen before point or null scores are inspected.
Every declared edge remains in every point and null opportunity. Fold
selection, structural zeros, missing evidence, or a failed null rerun cannot
remove an edge from the denominator.

Each opportunity has one typed state:

- `observed`: finite score in `[0, 1]`;
- `structural_zero`: exact zero with a reason;
- `not_estimable`: no score and a reason; or
- `failed`: no score and a failure reason.

The null artifact must contain the exact Cartesian product:

```text
frozen candidate edge IDs x requested null plan IDs
```

Missing, extra, or duplicate cells are contract violations. Failed plans emit
one failed cell for every candidate edge. A local-FDR denominator is never
formed from only the successful or observed subset.

## Unique Active Null

The only accepted primary null is a degree- and evidence-matched LR-target
reassignment. Context permutations, sender-label shuffles, receiver-signature
shuffles, and unrelated network rewiring are prohibited from this artifact.
They may remain separate diagnostics for their own estimands.

The reassignment operates on the `TargetPrior` bipartite driver-target graph.
For ligand-grain priors, all LR interactions mapped to the same ligand consume
the same reassigned profile. The training-fold, context-label-blind graph is
partitioned into evidence tiers using rank bands:

```text
1-10, 11-25, 26-50, 51-100, 101-250, >250
```

When ranks are absent, a deterministic within-driver rank is derived by
`weight descending, target_id ascending`. Within each evidence tier, the
planner performs bipartite double-edge switches:

```text
(driver_1, target_1), (driver_2, target_2)
    ->
(driver_1, target_2), (driver_2, target_1)
```

Drivers and targets must differ, both proposed cross-edges must be absent, and
no no-op or duplicate link is allowed. Weight and rank remain attached to the
driver stub. The operation therefore preserves exactly:

- driver degree;
- target degree within each evidence tier;
- every driver's weight and rank multiset; and
- evidence-tier counts.

Each tier must accept at least `20 * number_of_tier_edges` switches. The final
link overlap with the original graph must be no greater than 5%. A singleton,
structurally fixed, or insufficiently rewired tier makes that null plan typed
`not_estimable`; the planner cannot borrow a donor from another tier.

Point and null workflows use the same sanitized raw input, subject fold policy,
configuration, LR resource, score specification, and seed tree. Only the
target-prior content may change. Every accepted null prior reruns filtering,
gating, family clustering, tuning, attribution, sender assignment, and scoring
from raw fold scopes. Null children may be discarded after their complete
score column and provenance are materialized.

## Probability Strata And Runtime Support

The only v1 local-FDR stratum is:

```text
score_version, view, mode, contrast_id, context_id, receiver
```

The view is initially `gene`. No raw score is pooled across receivers,
contrasts, contexts, modes, score versions, resources, priors, or runs. The
resource, prior, configuration, null spec, and candidate universe are exact
artifact bindings rather than optional grouping labels.

Each stratum requires:

```text
at least 200 eligible non-structural candidate edges
at least 200 complete null plans
one complete candidate-edge x null-plan matrix
```

Structural-zero point edges retain probability zero when the matrix and gate
are otherwise valid, but they do not count toward the 200 mixture candidates.
Any `not_estimable` or `failed` matrix cell blocks both the mixture and the
edge-specific empirical p-values for that stratum.

## Empirical P And Local FDR

For edge `e`, the matched right-tail empirical active-null p-value is:

```text
p_e = (1 + sum_b I[T_null(e, b) >= T_observed(e)]) / (B + 1)
```

Ties count as null-extreme. This p-value is an active-null diagnostic and must
not be relabeled as a context differential p-value or automatically passed to
the G3-F hierarchical procedure.

The only v1 local-FDR estimator is a beta-uniform mixture over these empirical
p-values:

```text
p ~ pi0 * Uniform(0, 1) + (1 - pi0) * Beta(a, 1)
pi0 in [0.5, 1]
a   in [0.05, 0.95]
```

The constrained maximum-likelihood fit uses a deterministic preregistered
multi-start optimizer. All successful starts must agree in log likelihood to
`1e-8`, and the selected solution must have projected gradient norm at most
`1e-6`. The null-only solution `pi0=1` is valid and yields zero active
probability. Hitting `pi0=0.5` or an `a` boundary, non-finite curvature,
non-convergence, or disagreement among starts is not identifiable and does not
trigger a different estimator.

Internal candidate values are:

```text
local_fdr(p) = pi0 / (pi0 + (1 - pi0) * a * p ** (a - 1))
candidate_comm_probability = 1 - local_fdr(p)
```

Values are clipped only for floating-point tolerance after the model and all
diagnostics pass. Caller-provided probabilities or pass booleans are forbidden.

## Fallback And Status

The unique fallback policy is:

- complete null matrix but an insufficient stratum, unidentifiable mixture, or
  absent/failed G3-P gate: retain the empirical active-null p-value when it is
  estimable and publish `comm_probability=NA` with a reason;
- incomplete null matrix: both empirical p and probability are `NA`;
- structural-zero point edge under an otherwise valid, released stratum:
  `comm_probability=0`;
- never borrow a G3-F q-value, fit a different local-FDR model, shrink the
  candidate set, or pool another stratum as a fallback.

Stable reason codes include:

```text
g3p_gate_not_passed
active_null_stratum_too_small
active_null_insufficient_plans
active_null_incomplete_score_matrix
active_null_reassignment_infeasible
active_null_insufficient_rewiring
active_null_train_test_leakage
active_null_universe_mismatch
active_null_source_mismatch
local_fdr_nonconvergent
local_fdr_nonidentifiable
local_fdr_parameter_boundary
g3p_calibration_failed
```

Candidate local-FDR values may be retained in an internal diagnostic artifact.
They cannot populate a public probability column before calibration release.

## G3-P Release Gate

The producer-owned gate binds the exact active-null spec, BUM estimator spec,
stratum policy, score version, candidate-universe policy, and calibration
protocol. It is independent of G3-F and cannot be unlocked by a caller boolean.

The v1 release grid contains the exact Cartesian product of these dependence
structures:

```text
independent_candidate_edges
sender_lr_block_correlated_edges
target_degree_correlated_edges
```

and 0.5%, 1%, 5%, and 10% non-null prevalence. The first cell is the
well-specified independent baseline. The second preserves the shared
sender-unresolved LR parent and conserved sender-allocation dependence. The
third couples truth and score error to the frozen target-prior degree strata
that the accepted active null is required to preserve. A release protocol
cannot remove, rename, or add a grid cell after predictions are inspected.

Every cell contains at least 1,000 replicated datasets. The eligible candidate
universe and its stratum membership are frozen before truth or candidate
probability is inspected. Every replicate retains the complete eligible
candidate ledger; missing, duplicated, failed, or not-estimable candidates make
the cell not estimable. Structural-zero edges are recorded separately and are
not allowed to dilute calibration of the at-least-200 non-structural candidates
required in each runtime stratum.

Metrics are computed independently inside every frozen runtime stratum in every
dependence/prevalence scenario. Candidates from different receivers or other
stratum keys are never pooled. A scenario is estimable only when every one of
its strata is estimable, and the gate applies each threshold to every
scenario-by-stratum result; opposite calibration errors cannot cancel.

Metrics use the following frozen v1 definitions:

1. Within each replicate, candidates receive equal weight. Replicate-level
   metric summaries then receive equal weight within a scenario cell, so a
   replicate with more eligible edges cannot dominate the campaign.
2. The Brier baseline is the constant preregistered non-null prevalence for
   that cell, not the realized truth prevalence or a value refitted from the
   evaluated predictions.
3. ECE uses ten fixed equal-width bins on `[0, 1]`; the final bin includes one.
   Empty bins contribute zero. Its point estimate is the equal-replicate mean.
4. The one-sided 95% ECE upper bound uses 10,000 deterministic whole-replicate
   bootstrap draws. Candidate rows are never resampled as independent units.
   The seed is derived from the campaign seed lineage, scenario-cell ID, and
   runtime-stratum ID.
5. Calibration-in-the-large is the intercept from a weighted logistic
   recalibration with the prediction logit as an offset. Calibration slope is
   the slope from a weighted intercept-plus-logit recalibration. Candidate
   logits alone are clipped to `[1e-6, 1-1e-6]`; Brier and ECE use the original
   probabilities. Replicates retain equal total weight. A single-class truth
   vector, non-finite fit, curvature at or below `1e-10`, or failure to reach a
   score infinity norm of `1e-8` makes the cell not estimable; no penalized or
   fallback calibration fit is substituted.

The protocol, complete replicate ledger, scenario metrics, evidence and gate
are one producer-owned, checksum-bound campaign artifact. Scenario metrics and
replicate counts cannot be supplied directly by a caller. A public raw-ledger
summarizer produces diagnostic evidence only. Release additionally requires an
internally verified, replayable generator attestation binding generator kind,
configuration digest, code version, seed lineage, and typed source probability
collections. A caller-provided `generator_id` or source ID cannot self-attest
this requirement. Every required scenario-by-runtime-stratum cell must satisfy:

- Brier score improves at least 5% relative to the prevalence-only predictor;
- ECE point estimate is at most 0.05;
- the one-sided 95% ECE upper bound is at most 0.07;
- absolute calibration-in-the-large is at most 0.02; and
- calibration slope lies in `[0.8, 1.2]`.

Missing scenarios or insufficient replications produce `not_estimable`.
Threshold failure produces `failed`. Only a passed gate bound to the exact
runtime contracts may copy candidate values into public `comm_probability`.

## Persistence And Compatibility

The first implementation persists active probabilities as an independent
versioned result containing candidate/stratum diagnostics, a complete
null/source registry, and a public probability table only for a released
collection. Existing v0.1 interactions and cross-fit schemas through v6 continue
to reject non-null probability fields. A future result schema may project a
released active-probability result into `interactions.comm_probability` only
with explicit migration and linkage checks.

## Consequences

CRYCHIC gains an auditable path from a unique mechanistic null to an empirical
active-null p-value and, eventually, a calibrated posterior. The implementation
cost is high: a releasable stratum requires at least 200 full-pipeline null
reruns and a large frozen edge universe. This cost is intentional and cannot be
replaced by score relabeling.

Before the G3-P campaign passes, V03-10 artifacts and V03-14 candidate values
remain diagnostic. `comm_probability` stays `NA`, while independently valid
G3-F effect, p, and q fields retain their own release status.
