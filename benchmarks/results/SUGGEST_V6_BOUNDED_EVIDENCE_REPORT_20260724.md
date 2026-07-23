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
4. `suggestion_v6.md` is not fully implemented. In particular, the current
   RC11 export cannot support the five requested event-level DES variants.

## Version boundary

| Item | Frozen value |
|---|---|
| Branch | `optimize/suggest-v6-bounded-evidence-20260724` |
| Benchmark evaluator | `8ac34d389ac41160d32fa962ce168b73d4cb6e2b` |
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

## Strict DES availability

| Requested v6 endpoint | Status | Blocking input |
|---|---|---|
| Original count DES | not estimable | selected directed LR event ledger |
| Top-K count-matched DES | not estimable | ranked directed LR events |
| Continuous weighted DES | not estimable | event effects or signed p evidence |
| Direction-preserving DES | not estimable | ordered sender-receiver event axis |
| Mechanism-stratified DES | not estimable | event-level mechanism annotation |

RC11 currently exports `condition x unordered cell-pair` rankings. Converting
those pair scores into any of the above event-level endpoints would change the
estimand and was therefore rejected.

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
| Kuppe/MS robustness | partial | pair-rank resampling complete; five strict endpoints NE |
| MIS-C coverage curves | missing | native/common/all-axis and risk-coverage curves absent |
| Differential spatial simulation | missing | no implanted differential spatial truth track |
| DCST full grid | missing | full subject-by-cell grid absent |
| BRCA hurdle model | missing | prevalence/intensity joint benchmark absent |
| Runtime scaling | partial | stage profiling exists; full scaling exponents absent |
| Additional v6 cohorts | missing | scACCorDiON, reduced SLE, and context tensor tracks |
| Formal release | missing | no canonical tag or container digest |

## Next optimization direction

1. Export a checksum-bound directed event ledger containing event rank,
   selected status, signed effect, direction, and mechanism. This is required
   before the five strict DES endpoints can be computed.
2. Keep RC12 as a separate detection head and retain canonical signed effects.
   Do not optimize direction through an unsigned score.
3. Do not tune further on Kuppe. It is already a development cohort. Validate
   any real-cohort ranking change on a newly preregistered cohort or implanted
   spatial truth.
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
