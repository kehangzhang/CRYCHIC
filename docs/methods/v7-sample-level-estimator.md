# CRYCHIC v7 sample-level estimator contract

Status: implementation and calibration target. PR0--PR10 estimator and
full-refit execution code is present, but the locked simulation and real-data
campaigns remain release evidence rather than assumptions. This document does
not claim that v7 is already calibrated or released.

## Legacy freeze

The v2 family-common score, RC12 sender/program ranker, and RC14
pair-prioritized event ranker are retained as diagnostic comparators. Their
machine-readable freeze is
`benchmarks/configs/v7_legacy_baselines_v1.json`. Each is
`benchmark_diagnostic_only=true` and `formal_inference_eligible=false`.
Further RC rank-head tuning is outside the v7 development path.

## Primary measurement order

For every outer fold, expression and reliability transforms are fitted using
training subjects only. The frozen transforms are then applied to held-out
samples to produce one row per sample, sender, LR interaction, and receiver.
Condition labels do not select, gate, rescale, or zero the primary activity
measurement. Differential models consume the concatenated subject-level OOF
rows only after this measurement stage.

## Separate output heads

The v7 table separates quantities that were previously conflated:

| Head | Estimand | Formal differential target |
|---|---|---|
| `sender_detection_raw` | Non-conserving sender-specific LR support | Yes |
| `parent_activity_raw` | LR-receiver activity, with peak, total, and mean summaries | Yes |
| `program_signed` | Signed receiver mechanism response | Separate test or frozen ranking aid |
| `active_probability` | Subject-level event occurrence probability | Occurrence model only |
| `sender_attribution` | Conditional relative sender contribution, including a null sender | Compositional model or display |

Cell count and coverage produce reliability or precision fields; they never
multiply raw communication intensity. Resource confidence, contrast support,
family selection, downstream support, coupling, and hypergraph topology remain
annotations or separately tested components unless a locked benchmark proves a
pre-registered integration rule.

## Missingness semantics

`structural_impossible`, `not_estimable`, and `low_evidence` are distinct.
Structural impossibility may create a structural zero. Missing measurement or
coverage creates an NA row with an explicit reason. Low evidence remains a
finite continuous score and cannot be converted to zero by a condition-derived
gate.

## Sender detection and attribution

`sender_detection_raw` is never normalized across candidate senders. Adding a
candidate therefore cannot change another sender's raw detection measurement.
Attribution is a separate fold-fitted head. Training subjects define a robust
parent-specific detection scale and a subject-equal empirical parent-activity
reference distribution; condition labels and receiver outcomes are not inputs.

On held-out samples, the conditional sender distribution uses 1.5-entmax. Its
mass is multiplied by the fold-reference parent activity calibration, while the
remaining mass is assigned to an explicit null sender. Missing candidate rows
stay NA and do not invalidate estimable candidates. Sender weights, null mass,
and normalized attribution entropy are saved separately and sum to one over
the estimable sender set plus the null sender.

The empirical parent activity calibration is marked as a partial occurrence
head. It is used only to define null-sender mass and is not a posterior
probability or formal M4 occurrence result. M4 and its resampled p/q values have
their own later release gate.

M2 consumes a different training-only estimand: the subject-level M0 ligand
contrast for each candidate sender and the frozen-direction M1 receiver-program
contrast. Both are residualized on the same declared batch, composition, and
other nuisance design. For residual correlation `r`, M2 clips only for numeric
stability, applies Fisher `z = atanh(r)`, and uses sampling variance
`v = 1 / (n - p - 3)`. A zero-mean cross-edge second-moment estimate of `tau^2`
gives shrinkage factor `tau^2 / (tau^2 + v)` and the saved prior
`tanh(shrinkage_factor * z)`.

The coupling prior enters only the attribution logit with a globally frozen
weight in `{0, 0.25, 0.5}`. It never multiplies communication intensity. A
missing or weak coupling record contributes zero to the logit, remains visibly
not estimable in the coupling head, and cannot filter an otherwise measurable
sender.

The cross-fit implementation binds M2 to exactly one pre-registered contrast.
Runs with multiple contrasts must name that contrast explicitly. Each candidate
edge retains one row for every training subject; a subject missing any required
within-subject context has NA contrasts. Consequently, paired and repeated
designs are estimable when replication permits, while a purely independent
group comparison is explicitly marked not estimable rather than being turned
into a synthetic paired effect. The contrast ID and context-ID weights are part
of the coupling functional lineage.

## Signed receiver mechanism head

M1 is fitted inside each outer training fold and applied unchanged to held-out
samples. The interaction-to-target mapping comes only from the frozen target
prior. Activation/attenuation direction comes from that prior or a
pre-registered, content-bound per-interaction override in `SignedProgramV2Spec`.
The override permits heterogeneous activation and inhibition while the legacy
v0.1 attribution path continues to consume its required unsigned positive
prior. Condition labels are not inputs to the program fit and cannot select
targets, define a threshold, or orient the sign.

For each receiver, target expression is transformed with the same fold-fitted
size-factor, median, and MAD rules as M0. A subject-equal nuisance regression
removes log cell count, an all-gene global receiver state, optional predeclared
generic-state features, and declared batch/composition covariates. Invariant
nuisance columns contain no removable information and are omitted; unseen
held-out categorical levels remain explicitly not estimable.

The output keeps both `program_unaligned_raw` and `program_signed`. Activation
uses the raw positive-minus-negative target program; attenuation reverses its
sign. Neither negative values nor large positive values are clipped or
saturated. An unknown mechanism direction may retain the unaligned diagnostic
but cannot emit a signed value. The program is parent-level and therefore
constant across candidate senders for one sample, receiver, and interaction.
M1 remains a separate measurement head: it does not multiply M0 intensity and
is not itself a formal p/q result.

## Two-part occurrence inference

M4 is a post-cross-fit analysis of the concatenated held-out raw activity rows.
It uses one globally pre-registered raw-activity threshold and transition scale;
neither is estimated from condition labels, truth labels, or the evaluated
event. Replicate samples are first collapsed to one equally weighted
subject-context activity. Each resulting row records the fixed-threshold binary
state, a bounded monotone working active probability, and raw conditional
intensity only when the state is active. Missing measurements remain absent and
are never recoded as zero.

Independent pairwise contrasts use two-sided Fisher exact tests. Paired and
unadjusted repeated contrasts use exact McNemar/binomial tests, including the
zero-discordance null case. Designs requiring batch, cohort, or continuous
covariate adjustment use logistic estimating equations with HC1 or
subject-cluster sandwich covariance. Occurrence p-values are adjusted together
with Benjamini-Hochberg and are formally scoped to the fixed-threshold OOF
occurrence estimand. Active-only conditional-intensity effects remain a
separate design-aware diagnostic channel until full-pipeline resampling in
PR10; they do not borrow the occurrence p/q values.

The PR10 occurrence resampling statistic follows the registered design:
categorical contrasts use the subject-level prevalence difference, including
when the analytic exact/logistic test is not estimable, while continuous
designs use the fitted exposure log-odds slope because no two-group prevalence
difference exists. Compact resampling records bind this effect scale explicitly.

## Uncertainty-aware hypergraph shrinkage

M5 v2 consumes one locked design-aware contrast with a raw edge effect and
standard error. Its frozen H-prior is represented as a sparse intercept-plus-
view incidence matrix; resource-confidence values are not multiplied into the
topology. A weighted ridge fit estimates the additive topology mean, while a
zero-mean residual second moment estimates the common prior variance `tau^2`.
For edge standard error `SE`, the saved weight is
`kappa = tau^2 / (tau^2 + SE^2)` and the posterior working mean is
`kappa * raw_effect + (1 - kappa) * topology_mean`. High-precision edges retain
more of their data, while weak edges borrow more topology information.

The reported posterior standard error is
`sqrt(kappa * SE^2)` and is explicitly conditional on the fitted topology mean.
It omits topology-fit uncertainty, emits no p/q values, and cannot be used for
formal inference unless PR10 refits the complete shrinkage model inside every
subject-level resample. A legitimate zero analytic SE is retained with
`kappa=1`; negative or non-finite SE values are rejected. Missing raw effects
remain not estimable rather than being replaced by their topology prediction.
No-prior, partial-view,
exact-degree-matched permutation, and partial-rewiring priors remain required
controls for every M5 benchmark claim.

## Full-refit resampling

PR10 starts from the sanitized raw-count AnnData for every operation. It never
relabels a saved score table. The point run and every subject bootstrap,
condition permutation, and leave-one-subject-out run independently rebuild the
subject fold plan, availability, M0 transform, configured sender/M1/M2 heads,
held-out score table, design-aware effect, configured M4 result, and configured
M5 fit. Large cross-fit children are discarded after their small estimator and
stage-lineage records have been retained. Per-resample retention is limited to
the minimal continuous, occurrence, and M5 effect columns; M4 subject-event
tables, omnibus diagnostics, fold models, and matrices are released at the end
of each worker.

Independent-group bootstraps draw complete subjects with replacement within
condition and declared immutable strata, preserving group sizes. Paired and
repeated bootstraps draw complete subject trajectories. Condition permutations
use the frozen exchangeability map. Multi-cohort permutations are restricted
within cohort. Continuous designs jointly permute the scoring context and the
registered exposure column, rather than relabeling the context while leaving
the exposure fixed. Sample-level fields needed only by the differential design
are retained as auxiliary snapshot metadata: they are required to be constant
within sample and are bound by the snapshot metadata digest, but they are not
added to the M0/M1/M2 nuisance design. Complete three-or-more-context
trajectories remain fail-closed unless the caller explicitly registers the
reviewed complete within-subject permutation; this prevents ordered
longitudinal visits from being treated as exchangeable by default. LOSO removes
all cells and samples of one subject before rerunning fold planning. A resample
that can no longer form an estimable fold is a typed failed record; it is not
omitted from completeness counts. The result identity and manifest bind the
exchangeability map ID and the exact context, stratum, and immutable-covariate
columns.

The finalizer keeps three multiplicity scopes separate: continuous raw effect,
fixed-threshold occurrence prevalence difference, and M5 posterior effect.
Within each scope it reports the bootstrap standard deviation and percentile
interval, the two-sided empirical permutation value with the `+1` correction,
global event-by-contrast BH, and LOSO maximum/median effect displacement and
sign agreement. These remain explicitly diagnostic until all requested
resamples are observed and a producer-owned calibration gate passes every
supported design family with at least 1,000 null replicates per scenario,
empirical type-I in `[0.035, 0.065]`, FDR no greater than `0.12`, and 95%
coverage in `[0.92, 0.97]`.

## Benchmark diagnostic layer

`build_v7_diagnostics()` consumes the producer-owned OOF cross-fit artifact;
it does not reconstruct scores from a final ranking. Its gate waterfall has the
fixed order: common candidate universe, measurable ligand/receptor, receptor
eligibility, ligand-contrast support, family selection, downstream support,
sender assignment, native nonzero score, and significance. The first two
stages come from the v7 measurement table. Intermediate legacy stages come
from the exact fold application or an explicit benchmark-owned annotation
ledger. Significance must be supplied at the exact sender-child grain. An
unavailable stage is `not_estimable` or `not_computed`, never a failed gate.
The cumulative retained count requires every preceding stage to have passed,
while separate status counts expose why it stopped.

Waterfalls are emitted for every fold and the combined OOF set. Marginal
strata cover explicit truth positive/negative/unknown, contact/secreted/ECM
mechanism, candidate-sender bins `2`, `3-5`, `6-10`, and `>10`, minimum
sender/receiver cell-count bins, and receptor-complex cardinality. One-sender
and unknown bins are retained rather than discarded. Truth and cell counts are
optional benchmark inputs with unique keys; missing truth stays unknown and is
never counted as a negative.

Score geometry is computed separately for each configured head and fold.
Parent heads are deduplicated at sample-by-LR-receiver grain before counting,
whereas sender detection, attribution, coupling, and mechanism support retain
sender-child grain. The report includes zero and NA fractions, unique values,
tie fraction `(n_finite - n_unique) / (n_finite - 1)`, IQR and fixed
quantiles, truth-stratified quantiles, pooled within-condition variance,
event-centered between-condition variance, and the SD of null-event condition
effects. A head with no finite values remains explicitly not estimable.

Candidate-number diagnostics use the same fixed bins and report sender
detection means, parent mean activity, maximum attribution, attribution
entropy, a pre-registered detection-threshold false-positive rate, top-one
true-sender accuracy, AUPRC, and tie-aware AUROC. Only explicitly labelled
truth rows enter discrimination metrics. The raw sender detection score is
never renormalized for this report.

Multi-resolution evaluation accepts an explicit benchmark-owned ledger for LR,
LR-receiver parent, sender-LR-receiver child, pathway, and exact-hyperedge
units. It reports AUPRC, AUROC, effect Spearman correlation, RMSE, and direction
accuracy. CRYCHIC intentionally does not derive coarse truth from child truth:
sum, mean, maximum, and any-positive aggregation describe different estimands,
so the benchmark must declare the correct unit and truth effect. Diagnostic
inputs and outputs are content-bound, independent of input row order, and
tamper checked.

## Inference boundary

For continuous communication intensity, analytic HC3/CR2 models provide point
effects and diagnostics. Formal p-values, q-values, and confidence intervals
require the declared design to be estimable and require full-pipeline
subject-level resampling while nuisance learning is in scope. The M4 exception
is narrowly scoped to its pre-registered fixed-threshold binary occurrence
estimand; its p/q values do not make the continuous intensity channel formal.
Penalized coefficients and post-hoc ranking heads never feed ordinary Wald
inference.

The v7 design contract supports independent two-group and multi-group models,
paired subject differences, repeated measurements with subject-cluster CR2,
multi-cohort fixed effects, and continuous exposures. Categorical analyses use
pre-registered contrasts; independent multi-group analyses also emit an
omnibus diagnostic. Samples are collapsed to one equally represented row per
subject and context before fitting. Cell-count and coverage reliability enter
only as precision weights. A repeated design with too few subject clusters does
not silently fall back to an independence covariance model.

For independent multi-group cross-fitting, a pairwise contrast legitimately
contains only subjects observed in either declared context. The OOF audit
persists and content-binds the exact expected subject set for every fold by
contrast. It still rejects leakage, a missing eligible subject, an extra
subject from an unrelated context, incomplete context coverage, or multiple
fold functionals. Paired and repeated contrasts retain whole subject
trajectories; a repeated multi-context tuning fit and its final functional use
the same mixed subject-equal prediction-loss estimand.

Fields named `diagnostic_p_value`, `diagnostic_ci_lower`, and
`diagnostic_ci_upper` describe only the analytic regression diagnostic. The
formal `p_value`, `q_value`, `ci_lower`, and `ci_upper` fields remain NA and
`formal_inference_allowed=false` until the PR10 full-pipeline resampling and
multi-design calibration gates are both satisfied.
