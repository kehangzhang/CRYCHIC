# Sample-level design and estimated marginal means

Status: implemented exploratory G1 gate; no inferential p/q values are enabled.

CRYCHIC constructs the design from one row per biological sample. Cell counts
never replicate a sample in the formula matrix and never weight estimated
marginal means (EMMs). The reviewed formula grammar accepts only declared
context and covariate field names with Patsy main-effect and interaction
operators. Outcomes, arbitrary Python expressions, and undeclared fields are
rejected before matrix construction.

For categorical context factors, the reference grid is the Cartesian product
of all declared levels. Categorical nuisance covariates are averaged equally
over their levels; continuous covariates are fixed at the sample-level mean.
This defines balanced EMMs independently of observed cell abundance or the
number of cells captured from any sample.

The design audit records the formula, coefficient names, matrix rank, aliased
columns, condition number, balanced reference grid, context-level EMM matrix,
and registered-contrast estimability. A fit is blocked when a context factor is
omitted, the design is rank deficient, or a registered contrast is outside the
observed design row space. Ill-conditioning is reported as a warning at the
reviewed threshold but does not silently change the formula.

Subject allocation is audited from the complete sample table before filtering
any receiver cell type. Two-context studies must be either fully paired or
fully independent. A mixture of paired and unpaired subjects is returned as
not estimable; it is never reduced post hoc to the paired subset. In a fully
paired study, receiver-specific missingness may use explicit complete cases
when at least the configured number of pairs remain. Samples present only on
one receiver side are excluded from both the effect and its standard error.
Multiple libraries from the same subject and context are averaged first, so
each biological subject has equal weight. Repeated designs spanning more than
two referenced contexts remain unsupported in this exploratory backend.

Factorial main effects average the requested factor levels equally over all
declared context cells. Two-factor interactions use the standard
difference-in-differences scale:

```text
(A1,B1) - (A1,B0) - (A0,B1) + (A0,B0)
```

When additional context factors exist, each of the four cells is itself an
equal marginal mean. Missing factorial cells are rejected instead of being
imputed. Passing the design and subject-allocation audits does not by itself
authorize formal inference.

The workflow computes each distinct contrast weight vector once. On a complete
context graph, a focal node's topology-local neighbor contrast is identical to
its global one-versus-rest contrast; the local name is therefore an omitted
alias and the global name is canonical. This changes neither the estimand nor
its result, but avoids repeating response, attribution, and scoring work. Local
contrasts remain separate on non-complete graphs whenever their weights differ.
