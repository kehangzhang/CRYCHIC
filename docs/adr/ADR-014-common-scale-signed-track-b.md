# ADR-014: Common-scale signed Track-B producer

Status: Accepted for the opt-in descriptive benchmark workflow

## Context

The forward and reverse receiver-program pipelines use separately fitted
reference transforms. Their downstream values are therefore not comparable
signed scores even when the raw contrast effects are exact negatives. Signed
Track B needs two non-negative direction-compatible channels over one frozen
program universe, while the current non-negative TargetPrior cannot identify
active molecular inhibition.

Every directional cross-fit fold already retains the held-out receiver gene
matrix on the common `log1p_cpm` scale for both contrast directions. Forward
and reverse applications must contain the same rows, subjects, features and
numeric matrix. This common source can define a target-program recovery
endpoint without combining the non-comparable downstream transforms.

## Decision

`FrozenDirectionalTargetProgramUniverse` is built only from the complete
TargetPrior and an ordered feature universe. It retains every prior driver,
including programs with no matched target, and L2-normalizes each non-zero
non-negative target profile. No expression, condition label, truth direction,
sender or held-out outcome is accepted by the universe producer.

For one registered `DirectionalContrastPairSpec`, the producer:

1. authenticates the complete in-memory cross-fit and directional registry;
2. requires forward and reverse held-out applications to share the exact raw
   expression matrix and `log1p_cpm` scale;
3. selects only the pre-registered forward contrast context support;
4. averages technical rows within subject, outer fold and context;
5. fits all genes together with subject-equal weighted least squares using
   context means plus outer-fold nuisance terms;
6. applies the registered contrast to obtain one signed gene-effect vector;
7. forms `max(effect, 0)` and `max(-effect, 0)` channels; and
8. reports the cosine of each channel with every frozen non-negative target
   profile.

The multivariate solve is algebraically identical to applying the existing
scalar OOF effect backend to every gene, but factorizes the shared design once.
Programs below the pre-registered matched-target threshold remain in the table
as typed `not_estimable` rows. A channel with zero norm receives an observed
zero for every otherwise estimable program.

The result receiver axis is the frozen run receiver axis, not the union of
fold-training receivers. If any outer fold lacks a receiver in training, that
receiver's complete program x channel grid is typed `not_estimable` with
`receiver_absent_in_outer_training`; absent folds contribute support records but
no fabricated response applications or numeric gene effects.

The two channels are named:

- `increased_activation_compatible`; and
- `reduced_activation_compatible`.

The output is source-agnostic and never contains p-values, q-values, posterior
probabilities or communication probabilities. `supports_active_inhibition_claim`
and `formal_inference_allowed` are always false.

## Consequences

The strict signed Track-B evaluator can now consume native CRYCHIC program rows
without comparing fold-specific reference transforms or dropping unavailable
programs. This endpoint is a descriptive receiver target-program recovery
analysis. It is not sender attribution, LR-edge recovery, a native NicheNet
result, or evidence of active inhibition.

Release claims still require a frozen multi-seed simulation or perturbation
campaign with complete receiver x program x channel truth, followed by the
pre-registered equal-scenario macro-AUPRC evaluation. Real-data biology labels
cannot be used to release Track-B AUPRC.
