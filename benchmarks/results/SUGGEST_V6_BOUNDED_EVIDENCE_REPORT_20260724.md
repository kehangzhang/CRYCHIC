# Suggestion v6 bounded-evidence benchmark report

Status: **partial, not release complete**

## Main conclusions

1. The new RC12 unsigned detection head passed a fresh 20-seed validation gate.
   Relative to the canonical mechanistic head, mean omnibus AUPRC increased
   from 0.0786 to 0.1392 and AUROC from 0.4123 to 0.5480. The canonical signed
   head remains responsible for direction.
2. On the independent MS cohort, RC11 legacy pair-rank DES has median 0.8330
   and mean 0.7931. It ranks first among complete-coverage methods and exceeds
   scSeqCommDiff in 5 of 8 strata.
3. On Kuppe, which was used for development, RC11 remains below scSeqCommDiff
   in all 8 strata. Median DES is 0.6287 versus 0.7008.
4. The latest core event ledger now supports original-count, count-matched,
   continuous-weighted, and three mechanism-stratified DES variants. The
   frozen RC14 Top-K head is competitive with scSeqCommDiff, but native
   original-count and continuous-weighted DES remain lower. Direction-
   preserving and diffusible-long-range DES are still not estimable.
5. `suggestion_v6.md` remains only partially implemented; the expanded
   external-method simulation panel and several calibration tracks are absent.

## Version boundary

| Item | Frozen value |
|---|---|
| Branch | `optimize/suggest-v6-bounded-evidence-20260724` |
| Pair-rank robustness evaluator | `8ac34d389ac41160d32fa962ce168b73d4cb6e2b` |
| RC14 head freeze | `55074fae594f2327d1a0648c47288e664fdcdc64` |
| Final event-level DES evaluator | `0bc2703340794a88ad2ac088c1ab9ae14b6c2162` |
| RC12 preregistration | `e9fbc6796fba35325382f25558df63e7085a4a38` |
| RC12 validation evaluator | `240dc30053e487f5f2dda7dbb58cd04f2ed99aff` |
| Candidate status | benchmark-only, unreleased |
| Formal p/q values | disabled |
| Canonical communication strength | not replaced |
| Public default | not replaced |

## Three-group validation

The candidate is `0.90 * sender + 0.10 * receiver response`. It is an unsigned
event-detection ranking, not communication strength or a calibrated
probability.

| Head | AUPRC | AUROC | Localization AP | Null SD | Direction diagnostic |
|---|---:|---:|---:|---:|---:|
| RC12 sender-response | 0.1392 | 0.5480 | 0.1214 | 0.00339 | 0.5500 |
| RC11 bounded detection | 0.1112 | 0.5099 | 0.1029 | 0.00383 | 0.5857 |
| Canonical mechanistic | 0.0786 | 0.4123 | 0.0769 | 0.00728 | 0.6857 |

RC12 versus RC11 bounded detection:

- AUPRC: +0.0280, or +25.2% relative.
- AUROC: +0.0381.
- Global-null SD: 11.6% lower.

RC12 versus canonical mechanistic:

- AUPRC: +0.0606, or +77.1% relative; paired 95% CI [0.0314, 0.0956].
- AUROC: +0.1357; paired 95% CI [0.0577, 0.2157].
- Global-null SD: 53.5% lower; improvement CI [0.00357, 0.00424].

The lower direction diagnostic is not used for direction. The retained signed
canonical head has direction accuracy 0.6857. External methods were not rerun
on these new seeds, so this validation compares CRYCHIC score heads only.

## Kuppe and MS

These values are the existing pair-rank spatial sensitivity endpoint. They are
not a literal reconstruction of the paper's selected-event-count DES.

| Dataset | Role | RC11 median / mean | Complete-method rank | RC9 median / mean | scSeq median / mean | RC11 strata wins |
|---|---|---:|---:|---:|---:|---:|
| Kuppe | development | 0.6287 / 0.6000 | 2 / 2 of 15 | 0.6130 / 0.5833 | 0.7008 / 0.6913 | 0/8 |
| MS | independent validation | 0.8330 / 0.7931 | 1 / 1 of 9 | 0.7961 / 0.7538 | 0.8250 / 0.6726 | 5/8 |

The endpoint-only robustness runs used 1,000 stratified slice bootstraps, all
available leave-one-slice-out omissions, and 1,000 slice-label permutations.
The CRYCHIC ranking was fixed throughout.

| Dataset | Pair coverage | LOSO rank rho median | Bootstrap rank rho median | Permutation strata with p<0.05 |
|---|---:|---:|---:|---:|
| Kuppe | 1.000 | 0.936 | 0.578 | 2/8 |
| MS | 1.000 | 0.947 | 0.705 | 3/8 |

The permutation p values are diagnostics for the spatial endpoint only. They
are not formal CRYCHIC inference because the algorithm was not refit.

## Strict event-level DES

The event-level evaluator consumes the latest core's checksum-bound directed
LR effects. It exactly reconstructs the persisted scSeqCommDiff native count
ranking. RC14 is a benchmark-only head selected on Kuppe: within each
condition, events are ranked by `abs(HC2 z) * RC11 pair percentile`, and the
same K is retained for both methods. MS was evaluated once after freezing this
rule, but it had already been inspected during RC11 work and is not a fresh
independent cohort for RC14.

### Native original-count DES

| Dataset | CRYCHIC median / mean | Historical CRYCHIC Arm A | scSeq median / mean | Result |
|---|---:|---:|---:|---|
| Kuppe | 0.3869 / 0.4378 | 0.4284 / 0.4369 | 0.7008 / 0.6913 | lower |
| MS | 0.3000 / 0.2601 | 0.0500 / 0.0689 | 0.8250 / 0.6726 | improved versus old CRYCHIC, still lower |

Thus the latest native event count does **not** exceed scSeqCommDiff. The
Kuppe current-versus-historical pair-count Spearman is 0.971; on MS it is
0.653, reflecting a material version change rather than a parity replay.

### Per-condition Top-K count-matched DES

| Dataset | K | CRYCHIC median / mean | scSeq median / mean | Mean delta |
|---|---:|---:|---:|---:|
| Kuppe | 100 | 0.7446 / 0.7456 | 0.4956 / 0.4885 | +0.2571 |
| Kuppe | 250 | 0.7745 / 0.7543 | 0.8399 / 0.8378 | -0.0835 |
| Kuppe | 500 | 0.7725 / 0.7713 | 0.8024 / 0.8094 | -0.0381 |
| Kuppe | 1000 | 0.7626 / 0.7389 | 0.7427 / 0.7476 | -0.0087 |
| MS | 100 | 0.8005 / 0.6729 | 0.8000 / 0.6473 | +0.0256 |
| MS | 250 | 0.8325 / 0.6841 | 0.8812 / 0.7141 | -0.0299 |
| MS | 500 | 0.8015 / 0.6531 | 0.8250 / 0.6726 | -0.0194 |
| MS | 1000 | 0.8461 / 0.6762 | 0.8250 / 0.6726 | +0.0036 |

Across the four K values, mean DES averages 0.7525 versus 0.7208 on Kuppe and
0.6716 versus 0.6766 on MS. RC14 therefore leads at the sparse K=100 endpoint
in both datasets and at K=1000 on MS, but it does not dominate every K.

### Weighted and mechanism endpoints

| Dataset | Endpoint | CRYCHIC median / mean | scSeq median / mean |
|---|---|---:|---:|
| Kuppe | continuous weighted | 0.5069 / 0.5132 | 0.7247 / 0.7290 |
| MS | continuous weighted | 0.5963 / 0.4747 | 0.8000 / 0.6638 |
| Kuppe | contact | 0.4713 / 0.4599 | 0.7120 / 0.7253 |
| MS | contact | 0.5778 / 0.4651 | 0.8063 / 0.6690 |
| Kuppe | secreted | 0.5152 / 0.5243 | 0.7347 / 0.7417 |
| MS | secreted | 0.5963 / 0.4747 | 0.8000 / 0.6638 |

ECM/receptor contains only one resource LR and is not informative. The spatial
truth collapses sender/receiver direction, so direction-preserving DES remains
not estimable. ConnectomeDB2020 lacks diffusion-range annotation, so a strict
diffusible/long-range stratum also remains not estimable.

## Suggestion v6 audit

| Track | Status | Evidence or remaining gap |
|---|---|---|
| Signed hand fixtures | complete behaviorally | 15/15 pass; historical run was dirty |
| Invariance fixtures | complete behaviorally | 11/11 pass; historical run was dirty |
| Detection versus direction split | complete | separate RC12 unsigned and canonical signed heads |
| Fresh internal-head validation | complete | 20 active plus 20 matched-null seeds |
| Four-method expanded three-group panel | partial | not rerun on the new seeds |
| Score generator x engine | partial | CRYCHIC layers and two simple engines only |
| Null calibration | partial | 50 historical repeats, not 1,000 full-pipeline repeats |
| BRCA semisynthetic | partial | several scenarios and the full grid remain missing |
| Native hypergraph truth | partial | narrow synthetic track only |
| Kuppe/MS robustness | partial | pair-rank resampling plus original/Top-K/weighted/mechanism DES complete; directed and long-range endpoints NE |
| MIS-C coverage curves | missing | native/common/all-axis and risk-coverage curves absent |
| Differential spatial simulation | missing | no implanted differential spatial truth track |
| DCST full grid | missing | full subject-by-cell grid absent |
| BRCA hurdle model | missing | prevalence/intensity joint benchmark absent |
| Runtime scaling | partial | stage profiling exists; full scaling exponents absent |
| Additional v6 cohorts | missing | scACCorDiON, reduced SLE, and context tensor tracks |
| Formal release | missing | no canonical tag or container digest |

## Next optimization direction

1. Keep RC12 as a separate detection head and retain canonical signed effects.
   Do not optimize direction through an unsigned score.
2. Do not tune further on Kuppe/MS. Kuppe selected RC14 and MS has already
   informed earlier RC11 work. Validate
   any real-cohort ranking change on a newly preregistered cohort or implanted
   spatial truth.
3. Native count and continuous DES remain the main weaknesses. Address them
   through calibrated event selection and opportunity-aware effect estimation
   on simulations with event truth, not by optimizing against spatial DES.
4. Complete the common score-generator by differential-engine matrix before
   changing the algorithm backbone.

## Checksum-bound artifacts

- RC12 validation manifest:
  `benchmark_work/suggest_v6_iterations/sender_response_detection_rc12_240dc30_20260724/validation/manifest.json`
  (`6863b20f53e3b02fc7228139547eab4afbf5dd7b742bd81c2a9d3a2d99381fb7`)
- Kuppe robustness manifest:
  `benchmark_work/multi-group/bounded_des_extensions_8ac34d3_20260724/kuppe/manifest.json`
  (`e2c84b7c7fa4e749856b2550aafb411167dd57dc03ddbb2ab19b6fcf0beb1517`)
- MS robustness manifest:
  `benchmark_work/multi-group/bounded_des_extensions_8ac34d3_20260724/ms/manifest.json`
  (`f9c0a06f2193be4f37e716cb45f720cf201cfe1d7ed6f5ed7cfbf7b8c2e60d0b`)
- RC14 Kuppe event-level DES manifest:
  `benchmark_work/multi-group/event_level_des_rc14_0bc2703_20260724/kuppe/manifest.json`
  (`e3b51b8dea5d1a3b8dfe128dbd6a0b7faf1843df8aaf00da6bda760ba6b9c3cb`)
- RC14 MS event-level DES manifest:
  `benchmark_work/multi-group/event_level_des_rc14_0bc2703_20260724/ms/manifest.json`
  (`61a9d1fa051aac12595eb29a659c822c543bb82f8c6ab2e20061ceaad7039510`)
