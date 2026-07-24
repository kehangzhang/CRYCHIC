from __future__ import annotations

import numpy as np

from benchmarks.comprehensive.run_tensor_cell2cell_under100k import (
    LR_PER_PATHWAY,
    evaluate_factorization,
    generate_tensor,
    match_factors,
    truth_factor_matrices,
)


def test_noiseless_planted_tensor_has_expected_shape_and_program_count() -> None:
    tensor, programs = generate_tensor(seed=1, noise=0.0)

    assert tensor.shape == (12, 300, 3, 3)
    assert len(programs) == 4
    assert np.all(tensor >= 0.0)
    assert np.all(tensor <= 1.0)
    assert (
        sum(program["lr_stop"] - program["lr_start"] for program in programs)
        == 4 * LR_PER_PATHWAY
    )


def test_matching_is_invariant_to_factor_order() -> None:
    _, programs = generate_tensor(seed=1, noise=0.0)
    context, lr, sender, receiver = truth_factor_matrices(programs)
    permutation = np.asarray((2, 0, 3, 1))

    matches = match_factors(
        context,
        lr,
        sender,
        receiver,
        context[:, permutation],
        lr[:, permutation],
        sender[:, permutation],
        receiver[:, permutation],
    )

    assert matches == [(0, 1), (1, 3), (2, 0), (3, 2)]


def test_exact_factorization_has_perfect_program_recovery() -> None:
    tensor, programs = generate_tensor(seed=1, noise=0.0)
    context, lr, sender, receiver = truth_factor_matrices(programs)
    reconstruction = np.zeros_like(tensor)
    for index, program in enumerate(programs):
        reconstruction[:, :, program["sender"], program["receiver"]] += np.outer(
            context[:, index], lr[:, index]
        )

    summary, details = evaluate_factorization(
        tensor,
        programs,
        context,
        lr,
        sender,
        receiver,
        reconstruction,
    )

    assert summary["lr_jaccard_top100"] == 1.0
    assert summary["sender_top1_accuracy"] == 1.0
    assert summary["receiver_top1_accuracy"] == 1.0
    assert summary["event_auprc"] == 1.0
    assert summary["normalized_reconstruction_error"] == 0.0
    assert len(details) == 4
