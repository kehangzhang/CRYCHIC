# suggest_next2 v7 execution matrix

Date: 2026-07-26

This is an execution and evidence-status ledger. It is not a benchmark result
summary and does not promote smoke, development, or reused real cohorts to locked
evidence.

## Implementation status

| Scope | Implementation | Focused verification | Evidence status |
|---|---|---|---|
| PR0 legacy freeze and estimand split | complete | protocol/config tests | legacy comparator only |
| PR1 sample-level score contracts and persistence | complete | contract and round-trip tests | code evidence |
| PR2 absolute activity v2 | complete | unit and synthetic tests | code evidence |
| PR3 v7-primary cross-fit integration | complete | full-profile equivalence tests | code evidence |
| PR4 design-aware I0/I1/I2 inference | complete | independent, paired, repeated, multigroup tests | analytic p/q withheld |
| PR5 sender detection, null attribution and diagnostics | complete | decoy/cardinality tests | code evidence |
| PR6 EB-shrunken M2 coupling | complete | sender-coupling tests | code evidence |
| PR7 signed M1 head | complete | program/inhibitory/null tests | separate head |
| PR8 M4 two-part occurrence | complete | smoke and 200-dataset development slice | development ranking gain; release rejected |
| PR9 uncertainty-aware M5 topology swaps | complete | topology and smoke tests | locked evidence pending |
| PR10 full-pipeline resampling | complete | serial/process identity and six-design regression | r20 calibration active |
| intervention validation | complete | Kang/BRCA preflight and 10 tests | formal runs pending |
| sparse spatial geometry | complete | Kuppe/MS real-section smoke and 13 tests | formal runs pending |
| subject multi-sample v2 replay | complete | evaluator/lineage tests plus actual-data MS preflights | Kuppe external partial; CRYCHIC/MS formal runs pending |

The current real/spatial benchmark branch is
`benchmark/suggest-next2-v7-real-locked-20260726`. The spatial implementation is
commit `9d26b0d`; the intervention implementation is commit `1173e40`; strict
subject-v2 replay lineage is commit `82081c4`.

## Experiment coverage

The integrated runner persists all preregistered diagnostic tables for every
dataset: gate attrition, score geometry, candidate-sender bias,
parent-versus-child resolution, E2 gate swaps, E3 sender swaps, M4 occurrence,
and E5 topology swaps. The score-primary matrix contains G0--G5 with I1; the
inference crossover fixes G3 and compares I0/I1/I2.

| Experiment | Required comparison | Code | Formal execution |
|---|---|---|---|
| E1 M0 integration | G0/G1/G2/G3 under I1 | complete | development/locked and Kuppe/MS pending |
| E2 hard-gate attrition | annotation-only versus each hard gate | complete | development/locked pending |
| E3 sender detection/attribution | detection, legacy assignment, null sender, +M2 at 2/5/10/20 candidates | complete | development/locked pending |
| E4 signed program | G3 versus G4 on program, inhibitory and generic-state null | complete | locked families pending |
| E5 hypergraph | no/partial/full/permuted/rewired/clique/tensor priors | complete | locked families pending |
| E6 occurrence | raw/Fisher/DCST-compatible/logistic/beta-binomial/M4 | complete | occurrence development complete; current candidate rejected for release |

## Dataset-family coverage

The frozen protocol contains all requested development families:
expression-joint, occurrence heterogeneity, sender decoy, candidate cardinality,
complex AND, and graph smooth.

The locked family axis contains ligand-only, receptor-only, program-only,
inhibitory program, multi-sender, alternative OR, spatial range, hypergraph weak
effects, prior corruption, generic-state null, batch/context confounding null,
composition-only null, receiver-autonomous null, abundance-only null, topology
jump, wrong topology, disconnected graph, collinear LR, structural absence,
annotation perturbation, prior replacement, and context-correlated sampling
missingness. Independent two-group, independent multigroup, paired, repeated,
multicohort and continuous designs are represented where preregistered.

## Completed and pending runs

| Run | Frozen size | Current state | Permitted use |
|---|---:|---|---|
| integrated v1 regression smoke | 900/900 datasets | complete, 0 failed | regression only; predates M4 amendments |
| M4 raw-threshold smoke | 40/40 datasets | complete, 0 failed | diagnosed saturation only |
| M4 amended occurrence development | 200/200 datasets | complete, 0 failed | development parameter evidence |
| PR10 six-design process regression | 6/6 datasets, 112/112 refits | complete | execution equivalence only |
| PR10 amended r20 global-null calibration | 120 datasets planned | active | descriptive calibration pilot only |
| full amended development | 800 datasets | 200 reusable, 600 pending | parameter selection only |
| locked family evaluation | 5,750 datasets | pending | final synthetic family holdout |
| formal null calibration | 1,000 replicates per frozen null scenario | pending | required before formal p/q release |
| Kuppe latest v7 real refit | one complete cohort | pending | development/non-independent |
| MS latest v7 real refit | one complete cohort | pending | reused locked external, no tuning |
| Kang stimulation validation | 8 complete donor pairs | pending | supportive intervention evidence |
| BRCA paired treatment validation | E: 9 pairs; NE: 20 pairs | pending | descriptive external response evidence |
| Kuppe sparse geometry | 13 sections, 43,051 spots | formal run pending | indirect spatial silver evidence |
| MS sparse geometry | 11 sections, 40,587 spots | formal run pending | indirect spatial silver evidence |
| subject-level floor/exclude-self truth | Kuppe 4/7; MS 5/5 subjects | complete | same-unit v2 replay |
| Kuppe subject-v2 external replay | scSeqCommDiff, LIANA+, LIANA, CellChat | scSeq native/continuous/fixed-K and LIANA continuous tracks complete; LIANA+ provenance rerun pending; legacy CellChat NE | no final CRYCHIC rank until v7 refit |
| MS subject-v2 external replay | paper-matched 69,168 cells | scSeqCommDiff/LIANA+ preflight complete; formal runs pending | old 75,004-cell outputs excluded |

At 2026-07-26 17:26 CST, the active r20 campaign had completed 67/120 datasets
with zero failures. It had finished the 20 continuous, 20
independent-multigroup and 20 independent-two-group replicates plus 7/20
multi-cohort replicates. Its live campaign manifest, not this timestamped row,
is the authoritative status source.

The current M4 candidate is not a release candidate despite its development
ranking gain. Mean AUPRC was 0.4999 (independent) and 0.6526 (paired), but pooled
calibration slopes were 0.1570 and 0.1852, prevalence-effect RMSE did not improve,
and no positive event survived q <= 0.10. Its output remains a working score and
must not be described as a calibrated probability.

## Release boundaries

- The r20 campaign uses 20 dataset replicates and 99 bootstrap/permutation draws;
  it cannot satisfy the preregistered 1,000-replicate formal calibration floor.
- Kuppe remains development/non-independent. MS is not fresh after prior analysis
  and must not be used for parameter selection.
- Kang supports receiver-response and pathway consistency; it is not edge-level
  CCC ground truth.
- BRCA E and NE are fit separately. Their difference-in-change is descriptive and
  must not be called a native mixed 2x2 interaction test.
- Kuppe/MS spatial geometry is indirect spot-level encounter evidence, not direct
  binding, sender direction, physical contact, or causality.
- Cross-platform spatial replication remains not estimable because both available
  cohorts are Visium and their cell ontologies do not align.

## Frozen continuation order

1. Finish and summarize the active r20 global-null calibration without changing
   its config or resampling draws.
2. Apply its prespecified diagnostic gate; do not tune on MS or locked families.
3. Run Kuppe and MS v7 real refits in parallel, then Kang/BRCA and both formal
   geometry jobs within the 80% system-memory boundary.
4. Run the paper-matched MS subject-v2 external methods and current CellChat
   sensitivity arms; rank only within identical native/continuous contracts.
5. Resume the amended development root so its existing 200 occurrence datasets
   are reused and the remaining 600 datasets are added.
6. Run the untouched 5,750-dataset locked campaign.
7. Run the 1,000-replicate formal null calibration required for any formal p/q
   release, summarize all leaderboards, then generate notebooks and archive.
