# RC1 signed expected-cardinality simulation v1

Status: **rejected before real-cohort evaluation**.

The candidate grid and development/holdout split were committed at `36c4aa1`
before this run.  Kuppe and MS were not used for selection or evaluation.

## Development selection

`eb_delta_0.5` ranked first on the preregistered development metric:

| candidate | pair Spearman | top-quartile AUROC | tie fraction |
| --- | ---: | ---: | ---: |
| eb_delta_0.5 | 0.885 | 0.960 | 0.000 |
| wald_p05 | 0.852 | 0.945 | 0.819 |
| hard_z1 | 0.802 | 0.925 | 0.584 |
| continuous_sign | 0.795 | 0.921 | 0.000 |

## Independent holdout

Across 240 scenario-seed-direction evaluations, `eb_delta_0.5` reached mean
pair Spearman 0.879 versus 0.796 for `hard_z1` (paired mean difference 0.083;
96.3% wins; Wilcoxon p=7.29e-40).  It exceeded `hard_z1` in every non-null
scenario family, including structural missingness (0.860 versus 0.738).

## Rejection reason

The pure-null diagnostic exposed a marginal-likelihood identifiability failure.
Most seeds fitted the null boundary correctly, but several fitted a very narrow
slab with almost all mass assigned to it.  On holdout null simulations this
produced a mean 7.80 false expected interactions per pair and an opportunity
Spearman of 0.943.  A pair score with this failure can rank LR opportunity
rather than differential biology.

The v1 candidate is therefore not eligible for Kuppe/MS.  The next version must
preregister an identifiable minimum slab scale and an explicit null false-count
gate, then use new development and holdout seeds.  These working-model
probabilities remain benchmark-only and are not released scientific posterior
probabilities.
