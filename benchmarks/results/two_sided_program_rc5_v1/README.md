# RC5 two-sided receiver-program calibration

Status: **rejected before real-cohort evaluation**. The implementation,
candidate grid, development seeds, holdout seeds, and acceptance gates were
frozen at commit `691fdda` before this run. Kuppe and MS were not used for
selection or acceptance.

RC5 tested whether contradictory receiver-program evidence should be
continuously attenuated when the observed program effects have weak support for
both signs. Consistent evidence was unchanged, missing evidence remained
neutral, and `signed_rc3_reference` reproduced the RC3 weighting rule exactly.

## Development selection

`two_sided_005` ranked first among candidates that passed the preregistered
safety and global-null gates. Its mean primary pair-rank Spearman delta versus
the signed RC3 reference was only `+0.00014`; its safety delta was `-0.00006`.
The development result was therefore already weak, but selection remained
mechanical under the frozen policy.

## Independent holdout

| Gate | Result | Required | Pass |
|---|---:|---:|---|
| Biased helpful-program delta versus RC3 | -0.00151 | at least +0.002 | no |
| Balanced-program delta versus RC3 | 0.00000 | at least -0.003 | yes |
| Safety-scenario delta versus RC3 | +0.00004 | at least -0.003 | yes |
| Novel candidate selected | yes | yes | yes |

The failure was not caused by null or antagonistic behavior. Attenuating the
contradiction penalty slightly reduced ranking quality in both positive- and
negative-biased helpful holdouts, so the primary gain did not replicate.

## Decision

RC5 is not eligible for Kuppe/MS evaluation and must not be presented as an
improvement. A global sign-coverage statistic is too coarse to identify when a
downstream program should override or preserve contradictory LR evidence. The
next iteration should return to the component-swap result: improve the
sample-level occurrence/selection head or use a strong pseudobulk baseline with
hypergraph residual fallback, with all choices selected outside Kuppe/MS.

This remains an effect-summary simulation rather than a full pseudobulk
pipeline. The candidate is benchmark-only and is not released scientific
inference.
