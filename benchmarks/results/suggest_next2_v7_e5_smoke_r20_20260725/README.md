# Suggest-next2 v7 E5 smoke report

This report is descriptive evidence from the checksum-bound 20-seed v7 smoke
campaign. All 900 datasets completed, including typed non-estimable outputs for
40 fully confounded datasets. It is not formal release inference.

## Locked hypergraph family

- Full versus raw MSE gain: 0.0119
- Full versus permuted MSE gain: 0.0034
- 25% rewired versus permuted MSE gain: 0.0009
- Full versus raw exact-hyperedge AP gain: -0.0430
- Full versus permuted exact-hyperedge AP gain: 0.0583
- Full conditional CI coverage: 0.0310

Full topology improves MSE over raw effects, but its paired MSE advantage over
the degree-matched permutation is not established. Exact-hyperedge AP does not
improve over raw, and conditional posterior intervals are severely
under-covered. Tensor factorization has the best locked exact-hyperedge AP;
full hypergraph has the best locked MSE.

## Gate status

- m5_full_mse_gain_vs_no_prior_ci_positive: PASS
- m5_full_mse_gain_vs_permuted_ci_positive: FAIL
- m5_rewired25_mse_gain_vs_permuted_ci_positive: FAIL
- m5_full_exact_ap_gain_vs_permuted_ci_positive: FAIL
- m5_full_exact_ap_gain_vs_no_prior_ci_positive: FAIL
- m5_locked_conditional_ci_coverage_between_0_92_and_0_97: FAIL
- all_estimable_structural_fits_converged: PASS
- all_degree_matched_controls_preserve_degrees: PASS

The accepted claim is limited to hypergraph representation and descriptive
shrinkage. Topology estimation and calibrated interval claims remain withheld.
