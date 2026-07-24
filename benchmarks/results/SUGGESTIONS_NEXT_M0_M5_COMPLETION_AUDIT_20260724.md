# Suggestions-next M0--M5 completion audit

Status: **the preregistered experimental optimization cycle is complete**.
This is not a CRYCHIC release gate, a public-default change, or a claim of
general superiority over published methods.

## Scope audited

The selected strategy in `docs/methods/suggestions-next-m0-strategy.md` froze
RC12/RC14 as diagnostic baselines, then introduced one isolated mechanism per
cycle: M0 absolute activity and sender separation, M1 signed program support,
M2 residualized sender coupling, M4 occurrence prevalence, and M5 frozen
H-prior shrinkage. The audit treats each locked configuration, fixture
checksum, paired-seed gate, claim boundary, and regression invariant as a
required deliverable.

The much broader benchmark wish list in `suggestions_next.md` remains a roadmap,
not part of this bounded cycle. In particular, real-data SOTA, formal p/q
calibration, a third unseen spatial cohort, and public-default integration are
not silently redefined as complete.

## Requirement-by-requirement evidence

| Requirement | Implementation evidence | Locked evidence | Status |
|---|---|---|---|
| Work on a dedicated branch | `optimize/suggestions-next-m0-20260724` | Conventional commits with independent fixture/config freezes | Complete |
| Freeze RC12/RC14 instead of further test-set tuning | Existing frozen configs and `SUGGEST_V6_BOUNDED_EVIDENCE_REPORT_20260724.md` | RC12/RC14 remain benchmark-only and public default is unchanged | Complete |
| M0 additive absolute LR score | `209e6a3`, `scoring/absolute.py` | 20 fresh seeds; M0 AUPRC 0.6112, AUROC 0.9472; rank 1/5 on the expression-driven external panel | Complete |
| Separate sender detection and attribution, with a null sender | Nonconserved `sender_detection`, conditional `sender_attribution`, null-aware allocation | Candidate-count invariance, mass conservation, low-evidence null, and missing-candidate tests | Complete |
| M1 signed downstream support | Core API `28197e7`, version `signed_geometric_program_concordance_m1_v1` | AP/AUROC 1.000/1.000 on seven mechanism families; five scientific replay tables byte-identical after core migration | Complete |
| M2 residualized sender coupling | `f6248c5`, fold-training nuisance residualization | AP/AUROC 1.000/1.000; true sender rank 1 in 20/20 fresh seeds | Complete |
| M4 occurrence/prevalence head | `9a1fa30`, Jeffreys-Beta independent-group contract | AP 0.943, AUROC 0.965, rank 2/4; improved calibration over raw prevalence | Complete with raw ranker retained |
| M5 pre-fit frozen H-prior | `fa55587`, five-view additive incidence ridge | MSE/AP/AUROC rank 1/6; significantly better than raw, exact-degree permutation, and three single-view priors | Complete for the synthetic effect-summary estimand |
| M5 10%--25% prior rewiring robustness | `25a93fe`, bounded exact-degree rewiring; evaluator `c345f88` | Fresh validation retained 79.7%/55.1% MSE gain and 81.6%/57.7% AP gain at 10%/25%; all 15 gates passed | Complete |
| Preserve independent/paired/repeated design contracts | Existing first-class design and CR2/repeated-measures modules | Full regression covers subject clustering, paired/repeated collapse, rank deficiency, OOF common scoring, and typed NE | Complete for existing supported designs |
| Do not expose descriptive heads as formal inference | All M0--M5 outputs explicitly disable formal inference | Tests reject p/q/probability fields and preserve missing/NE semantics | Complete |

## Feedback-driven decisions retained

The branch records negative results instead of selecting only successful arms:

- adding legacy saturated availability to RC12 reduced development AUPRC and
  motivated M0's fixed log-reference geometry;
- M1 exposed incomplete sender-candidate coverage, leading to detection being
  preserved while attribution becomes typed not-estimable;
- M2 rejected raw proportion, full-log-ratio, and PC1 nuisance alternatives
  before freezing the recorded-covariate residualization;
- M4 retained raw prevalence as the best ranking baseline even though M4
  improved Brier score and log-odds RMSE;
- M5 rejected a consensus group-mean smoother that did not fit
  `beta = B theta + delta`, then froze the additive-incidence model;
- M5b reports the full 10%/25%/50% topology-dose response, not only the two
  fractions used for acceptance.

## Final regression evidence

The final clean source snapshot before this audit was `9bcc946`.

```text
PYTHONPATH=src:. pytest -q
2755 passed, 10 skipped, 110 warnings in 1438.26s (0:23:58)
```

All M0--M5 targeted suites and scoped Ruff checks for modified Python files
passed. The ten skips were explicit: untracked/historical full smoke artifacts,
one unavailable canonical database, and optional `conorm`/`ot` dependencies.
The current v5 crossfit and autonomous smoke payloads were separately rerun and
their tracked provenance summaries refreshed.

Repository-wide Ruff is not clean and was not misreported as such. Its existing
debt is 257 findings: 148 `UP038`, 96 `I001`, 7 `E501`, 3 `F404`, 2 `UP012`,
and 1 `B018`. Scoped checks on all files introduced or materially changed by
M1/M5/M5b passed. The pytest run emitted 110 pandas future warnings; none were
test failures.

## Claim boundary and remaining roadmap

This cycle establishes that the proposed score decomposition can recover the
intended synthetic mechanism families and that correct frozen hypergraph
topology contributes measurable, corruption-tolerant shrinkage gain. It does
not establish a universal multi-group SOTA result.

Known remaining work is intentionally outside this completed bounded cycle:

- integrate M5 with real subject-level pseudobulk effects and evaluate genuine
  resource misspecification and cohort shift;
- validate the new heads on additional real perturbation and unseen spatial
  cohorts;
- complete full-pipeline Type-I/FDR/CI calibration before enabling formal p/q;
- decide through independent evidence whether any experimental head should
  replace the public default;
- clear repository-wide lint and pandas-deprecation debt separately from the
  statistical optimization history.

Accordingly, the defensible endpoint is: **M0--M5 experimental optimization and
locked validation complete; release integration and general real-data claims
remain future work.**
