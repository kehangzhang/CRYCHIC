# Suggestions-next M5 locked validation

Status: **ACCEPT** on the frozen synthetic additive H-prior effect-summary
fixture. M5 is experimental descriptive shrinkage. It does not replace the
public score, emit formal p/q values, or establish a general real-data benefit.

## Estimand and implementation

M5 consumes one noisy effect estimate and one positive observation precision
per frozen edge. Before any outcome is generated or fitted, an external
`H_prior` fixes five memberships for every edge: sender, ligand, receptor,
receiver, and pathway. The post-fit result hypergraph is never used as a prior.

For incidence design `B`, M5 fits the additive decomposition
`beta = intercept + B theta + delta`. The fixed objective uses weighted squared
error, an L2 penalty of 1.0 on node effects, and an L2 penalty of 1.0 on the
edge-specific residual. The intercept is unpenalized. The closed-form solver
returns a shrunk edge effect plus authenticated fit and topology identifiers.
It emits no probability, p value, or q value.

## Frozen design and controls

| Item | Frozen value |
|---|---|
| Implementation commit | `fa55587a` |
| Development fixture/config commit | `70cecc2b` |
| Validation config commit | `5fb2a73d` |
| Candidate version | `additive_incidence_ridge_m5_v1` |
| Edges per dataset | 1,500 |
| Development seeds | `20290101`--`20290120` |
| Validation seeds | `20290501`--`20290520` |
| Observation noise SD | 1.15 |
| Frozen H-prior SHA256 | `a95e0bb52320581995a6423255dffbee711410cef1ee794d69b3e90c9a2dddc9` |
| Development fixture manifest | `b6bbd440b8ef933481cfdc92f13b708e485a05e35bf2616cb68db3daa1b1d4d0` |
| Validation fixture manifest | `e63d689a59b5ed7e67c33db89583d180b9d11428e04ef40889254b55ae97eb1d` |

The same outcome-blind 1,500-edge topology is reused across seeds. Each seed
draws new node effects, an edge residual, and observation noise. Truth is held
in a separate evaluator-only table. Active-edge AP uses the top 30% of absolute
true effects.

The locked panel contains all controls requested for this cycle:

- no H-prior raw effect;
- full correct H-prior;
- independently stub-permuted H-prior with the exact degree profile preserved
  in every view;
- ligand-only, receptor-only, and pathway-only H-priors.

## Development history

An early exploratory consensus group-mean smoother was rejected. With either
per-view or fixed-total penalties, full-view MSE was approximately 0.606 while
pathway-only MSE was approximately 0.605, and a penalty grid of 0.5, 1, 2, and
4 did not make the full view win both MSE and AP. That prototype projected
view-wise means rather than fitting `beta = B theta + delta`, so it did not
test the intended model. These exploratory values were not locked and are
reported only as design history.

The replacement additive-incidence candidate was frozen before the 20-seed
development fixture. It passed all nine development gates: full-view MSE was
0.3223, AP 0.7399, AUROC 0.8450, and direction accuracy 0.8269. It ranked first
of six for MSE, AP, and AUROC. Its output manifest SHA256 is
`b77ce2c757e4fab53d6b97a710ef976176f65fba680979ffac909f2e1c287d26`.

## Locked validation

Validation used 20 fresh seeds and 30,000 new edge truths. Evaluation ran from
clean commit `5fb2a73d` without changing the candidate, penalties, permutation,
or gates.

| Method | MSE | Spearman | AP | AUROC | Direction | MSE rank | AP rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full correct H-prior | **0.3156** | **0.8477** | **0.7425** | **0.8437** | **0.8289** | **1/6** | **1/6** |
| Ligand-only | 0.5189 | 0.6807 | 0.5580 | 0.7082 | 0.7479 | 2/6 | 2/6 |
| Pathway-only | 0.5215 | 0.6795 | 0.5530 | 0.7077 | 0.7461 | 3/6 | 3/6 |
| Receptor-only | 0.5227 | 0.6808 | 0.5505 | 0.7048 | 0.7450 | 4/6 | 4/6 |
| Degree-matched permutation | 0.5914 | 0.6264 | 0.5077 | 0.6718 | 0.7231 | 5/6 | 6/6 |
| No-prior raw effect | 1.3258 | 0.6415 | 0.5200 | 0.6808 | 0.7314 | 6/6 | 5/6 |

All paired comparisons favored the full H-prior in all 20 seeds:

| Comparison | Metric delta | Mean delta | Paired bootstrap 95% CI |
|---|---|---:|---:|
| Full - raw | MSE | -1.0102 | [-1.0247, -0.9950] |
| Full - permuted | MSE | -0.2758 | [-0.2813, -0.2701] |
| Full - raw | AP | +0.2225 | [+0.2149, +0.2304] |
| Full - permuted | AP | +0.2347 | [+0.2269, +0.2432] |
| Full - ligand-only | MSE | -0.2033 | [-0.2125, -0.1939] |
| Full - receptor-only | MSE | -0.2071 | [-0.2185, -0.1954] |
| Full - pathway-only | MSE | -0.2059 | [-0.2214, -0.1891] |
| Full - raw | Direction | +0.0975 | [+0.0933, +0.1023] |

All nine preregistered gates passed, including minimum paired seeds,
topology-specific MSE/AP gains, superiority to every partial prior, direction
noninferiority, and rank one for MSE and AP. Runtime was 5.03 seconds.

Validation output:

```text
/media/subunit/bioinfo/crychic_dev/benchmark_work/
  suggestions_next_m0_iterations/m5_hprior_validation_5fb2a73_20260724
```

Output manifest SHA256:
`e6fe1dad470be950aea45d2e2eb4f9ba31a9b34a77c7be446ae907356f2196d6`.

## Claim boundary and remaining work

This result supports one narrow claim: under an additive synthetic node-effect
estimand, correct frozen hypergraph topology improves edge-effect shrinkage
over no prior, an exact-degree-matched wrong topology, and single-view priors.

It does not establish benefit on real pseudobulk effects, misspecified or
partially wrong biological resources, raw-expression end-to-end fitting,
formal uncertainty, or published-method SOTA comparisons. The generator is
aligned with the fitted additive incidence model. Only one fixed fully
permuted topology was used; the suggested 10%--25% prior-rewiring robustness
gate remains outstanding. M5 must remain experimental until that robustness
test and subject-level pipeline integration are completed.
