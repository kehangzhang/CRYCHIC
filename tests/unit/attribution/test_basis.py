from __future__ import annotations

import numpy as np
import pytest

from crychic.attribution import (
    DriverFamilyDefinition,
    GatedTargetBasis,
    build_gated_target_basis,
    cluster_driver_families,
)
from crychic.core import ContractError, stable_id

_COSINE_TOLERANCE = 1e-12


def _slow_cluster_driver_families(
    basis: GatedTargetBasis,
    *,
    cosine_threshold: float,
) -> tuple[DriverFamilyDefinition, ...]:
    """Brute-force reference for the deterministic complete-link contract."""

    cosine = (basis.normalized_profiles.T @ basis.normalized_profiles).tocsr()
    clusters = [
        (index,)
        for index in sorted(
            range(len(basis.driver_ids)), key=lambda index: basis.driver_ids[index]
        )
    ]
    while True:
        best: (
            tuple[
                tuple[float, tuple[str, ...], tuple[str, ...], tuple[str, ...]],
                int,
                int,
            ]
            | None
        ) = None
        for left_offset, left in enumerate(clusters):
            left_names = tuple(sorted(basis.driver_ids[index] for index in left))
            for right_offset in range(left_offset + 1, len(clusters)):
                right = clusters[right_offset]
                similarity = min(
                    min(
                        1.0,
                        max(0.0, float(cosine[left_index, right_index])),
                    )
                    for left_index in left
                    for right_index in right
                )
                if similarity + _COSINE_TOLERANCE < cosine_threshold:
                    continue
                right_names = tuple(sorted(basis.driver_ids[index] for index in right))
                merged_names = tuple(sorted((*left_names, *right_names)))
                candidate = (
                    (-similarity, merged_names, left_names, right_names),
                    left_offset,
                    right_offset,
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None:
            break
        _, left_offset, right_offset = best
        merged = tuple(sorted((*clusters[left_offset], *clusters[right_offset])))
        clusters = [
            cluster
            for offset, cluster in enumerate(clusters)
            if offset not in {left_offset, right_offset}
        ]
        clusters.append(merged)
        clusters.sort(
            key=lambda cluster: tuple(
                sorted(basis.driver_ids[index] for index in cluster)
            )
        )

    families: list[DriverFamilyDefinition] = []
    for indices in clusters:
        members = tuple(sorted(basis.driver_ids[index] for index in indices))
        if len(indices) == 1:
            mean_cosine = 0.0
        else:
            similarities = [
                float(cosine[left, right])
                for offset, left in enumerate(indices)
                for right in indices[offset + 1 :]
            ]
            mean_cosine = float(np.clip(np.mean(similarities), 0.0, 1.0))
        family_id = stable_id(
            "driver_family",
            {
                "driver_ids": members,
                "prior_resource_id": basis.prior_resource_id,
                "prior_version": basis.prior_version,
            },
        )
        families.append(
            DriverFamilyDefinition(
                family_id=family_id,
                driver_ids=members,
                mean_pairwise_cosine=mean_cosine,
                assignment_uncertainty=mean_cosine,
            )
        )
    return tuple(sorted(families, key=lambda family: family.family_id))


def test_prior_columns_are_l2_normalized_before_receptor_gate(prior_factory) -> None:
    prior = prior_factory(
        {
            "L1": {"G1": 3.0, "G2": 4.0},
            "L2": {"G1": 6.0, "G2": 8.0},
        }
    )
    basis = build_gated_target_basis(
        prior,
        ("G1", "G2"),
        {"L1": 1.0, "L2": 0.5},
    )

    np.testing.assert_allclose(basis.pre_normalization_norms, [5.0, 10.0])
    np.testing.assert_allclose(
        basis.normalized_profiles.toarray(),
        [[0.6, 0.6], [0.8, 0.8]],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        basis.matrix.toarray(),
        [[0.6, 0.3], [0.8, 0.4]],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.sqrt(np.asarray(basis.matrix.power(2).sum(axis=0)).ravel()),
        [1.0, 0.5],
    )


def test_missing_target_overlap_is_reported_and_zero_column_preserved(
    prior_factory,
) -> None:
    prior = prior_factory({"L1": {"G1": 1.0}, "L2": {"OTHER": 2.0}})
    basis = build_gated_target_basis(prior, ("G1",), {"L1": 1.0, "L2": 1.0})

    assert basis.report.unmatched_prior_targets == ("OTHER",)
    assert basis.report.zero_norm_drivers == ("L2",)
    np.testing.assert_allclose(basis.matrix.toarray(), [[1.0, 0.0]])


def test_cosine_family_is_stable_for_scaled_collinear_profiles(
    prior_factory,
) -> None:
    prior = prior_factory(
        {
            "A": {"G1": 1.0, "G2": 2.0},
            "B": {"G1": 3.0, "G2": 6.0},
            "C": {"G3": 1.0},
        }
    )
    basis = build_gated_target_basis(
        prior,
        ("G1", "G2", "G3"),
        {"A": 1.0, "B": 0.0, "C": 1.0},
    )
    first = cluster_driver_families(basis, cosine_threshold=0.999)
    second = cluster_driver_families(basis, cosine_threshold=0.999)

    assert first == second
    collinear = next(family for family in first if len(family.driver_ids) == 2)
    assert collinear.driver_ids == ("A", "B")
    assert collinear.mean_pairwise_cosine == pytest.approx(1.0, abs=1e-12)
    assert collinear.assignment_uncertainty == pytest.approx(1.0, abs=1e-12)


def test_cosine_family_cache_reuses_pre_gate_profiles(
    prior_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crychic.attribution import families as families_module

    prior = prior_factory(
        {
            "A": {"G1": 1.0, "G2": 2.0},
            "B": {"G1": 3.0, "G2": 6.0},
            "C": {"G3": 1.0},
        },
        resource_id="family-cache-pre-gate-test",
    )
    first_basis = build_gated_target_basis(
        prior,
        ("G1", "G2", "G3"),
        {"A": 1.0, "B": 0.0, "C": 1.0},
    )
    second_basis = build_gated_target_basis(
        prior,
        ("G1", "G2", "G3"),
        {"A": 0.1, "B": 1.0, "C": 0.0},
    )
    original = families_module._complete_link_split
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(families_module, "_complete_link_split", counted)
    first = cluster_driver_families(first_basis, cosine_threshold=0.999)
    calls_after_first = calls
    second = cluster_driver_families(second_basis, cosine_threshold=0.999)

    assert calls_after_first > 0
    assert calls == calls_after_first
    assert second is first


def test_complete_link_family_prevents_single_linkage_chain(prior_factory) -> None:
    root_three = float(np.sqrt(3.0))
    threshold = 0.85
    prior = prior_factory(
        {
            "A": {"G1": 1.0},
            "B": {"G1": root_three, "G2": 1.0},
            "C": {"G1": 1.0, "G2": root_three},
        }
    )
    basis = build_gated_target_basis(
        prior,
        ("G1", "G2"),
        {"A": 1.0, "B": 1.0, "C": 1.0},
    )
    profiles = basis.normalized_profiles.toarray()
    index = {driver: position for position, driver in enumerate(basis.driver_ids)}

    cosine_ab = float(profiles[:, index["A"]] @ profiles[:, index["B"]])
    cosine_bc = float(profiles[:, index["B"]] @ profiles[:, index["C"]])
    cosine_ac = float(profiles[:, index["A"]] @ profiles[:, index["C"]])
    assert cosine_ab >= threshold
    assert cosine_bc >= threshold
    assert cosine_ac < threshold

    families = cluster_driver_families(basis, cosine_threshold=threshold)

    assert sorted(len(family.driver_ids) for family in families) == [1, 2]
    assert all(family.driver_ids != ("A", "B", "C") for family in families)
    for family in families:
        for offset, left in enumerate(family.driver_ids):
            for right in family.driver_ids[offset + 1 :]:
                observed = float(profiles[:, index[left]] @ profiles[:, index[right]])
                assert observed + 1e-12 >= threshold


def test_complete_link_family_is_deterministic_under_input_and_feature_order(
    prior_factory,
) -> None:
    root_three = float(np.sqrt(3.0))
    first_prior = prior_factory(
        {
            "A": {"G1": 1.0},
            "B": {"G1": root_three, "G2": 1.0},
            "C": {"G1": 1.0, "G2": root_three},
        }
    )
    reordered_prior = prior_factory(
        {
            "C": {"G2": root_three, "G1": 1.0},
            "A": {"G1": 1.0},
            "B": {"G2": 1.0, "G1": root_three},
        }
    )
    first_basis = build_gated_target_basis(
        first_prior,
        ("G1", "G2"),
        {"A": 1.0, "B": 1.0, "C": 1.0},
    )
    reordered_basis = build_gated_target_basis(
        reordered_prior,
        ("G2", "G1"),
        {"C": 1.0, "A": 1.0, "B": 1.0},
    )

    first = cluster_driver_families(first_basis, cosine_threshold=0.85)
    repeated = cluster_driver_families(first_basis, cosine_threshold=0.85)
    reordered = cluster_driver_families(reordered_basis, cosine_threshold=0.85)

    assert first == repeated == reordered


def test_heap_complete_link_matches_brute_force_on_random_sparse_profiles(
    prior_factory,
) -> None:
    thresholds = (0.0, 0.5, 0.85, 0.95, 1.0)
    for seed, n_drivers in enumerate((2, 5, 8, 10, 12), start=8100):
        rng = np.random.default_rng(seed)
        features = tuple(f"G{index:02d}" for index in range(9))
        columns: dict[str, dict[str, float]] = {}
        for driver_index in range(n_drivers):
            mask = rng.random(len(features)) < 0.35
            # Deliberately retain some zero-norm columns in the differential set.
            if driver_index % 7 and not np.any(mask):
                mask[int(rng.integers(0, len(features)))] = True
            columns[f"D{driver_index:02d}"] = {
                feature: float(weight)
                for feature, weight, selected in zip(
                    features,
                    rng.uniform(0.05, 2.0, size=len(features)),
                    mask,
                    strict=True,
                )
                if selected
            }
        prior = prior_factory(columns, resource_id=f"random-prior-{seed}")
        feature_order = tuple(np.asarray(features)[rng.permutation(len(features))])
        basis = build_gated_target_basis(
            prior,
            feature_order,
            dict.fromkeys(prior.driver_ids, 1.0),
        )

        for threshold in thresholds:
            observed = cluster_driver_families(
                basis,
                cosine_threshold=threshold,
            )
            expected = _slow_cluster_driver_families(
                basis,
                cosine_threshold=threshold,
            )
            assert observed == expected, (seed, threshold)


def test_equal_similarity_tie_uses_lexicographic_family_names(
    prior_factory,
) -> None:
    prior = prior_factory(
        {
            "A": {"G1": 1.0},
            "B": {"G1": 1.0, "G2": 1.0},
            "C": {"G2": 1.0},
        }
    )
    basis = build_gated_target_basis(
        prior,
        ("G1", "G2"),
        dict.fromkeys(prior.driver_ids, 1.0),
    )

    observed = cluster_driver_families(basis, cosine_threshold=0.7)

    assert observed == _slow_cluster_driver_families(
        basis,
        cosine_threshold=0.7,
    )
    assert {family.driver_ids for family in observed} == {("A", "B"), ("C",)}


def test_zero_similarity_and_zero_norm_columns_merge_at_zero_threshold(
    prior_factory,
) -> None:
    prior = prior_factory(
        {
            "A": {"G1": 1.0},
            "B": {"G2": 1.0},
            "ZERO": {"OUTSIDE": 1.0},
        }
    )
    basis = build_gated_target_basis(
        prior,
        ("G1", "G2"),
        dict.fromkeys(prior.driver_ids, 1.0),
    )

    observed = cluster_driver_families(basis, cosine_threshold=0.0)

    assert observed == _slow_cluster_driver_families(
        basis,
        cosine_threshold=0.0,
    )
    assert basis.report.zero_norm_drivers == ("ZERO",)
    assert len(observed) == 1
    assert observed[0].driver_ids == ("A", "B", "ZERO")
    assert observed[0].mean_pairwise_cosine == 0.0


def test_complete_link_threshold_uses_declared_cosine_tolerance(
    prior_factory,
) -> None:
    prior = prior_factory(
        {
            "A": {"G1": 1.0},
            "B": {"G1": 4.0, "G2": 3.0},
        }
    )
    basis = build_gated_target_basis(
        prior,
        ("G1", "G2"),
        dict.fromkeys(prior.driver_ids, 1.0),
    )
    profiles = basis.normalized_profiles.toarray()
    cosine = float(profiles[:, 0] @ profiles[:, 1])

    within_tolerance = cluster_driver_families(
        basis,
        cosine_threshold=cosine + 0.5 * _COSINE_TOLERANCE,
    )
    outside_tolerance = cluster_driver_families(
        basis,
        cosine_threshold=cosine + 2.0 * _COSINE_TOLERANCE,
    )

    assert within_tolerance == _slow_cluster_driver_families(
        basis,
        cosine_threshold=cosine + 0.5 * _COSINE_TOLERANCE,
    )
    assert outside_tolerance == _slow_cluster_driver_families(
        basis,
        cosine_threshold=cosine + 2.0 * _COSINE_TOLERANCE,
    )
    assert [len(family.driver_ids) for family in within_tolerance] == [2]
    assert sorted(len(family.driver_ids) for family in outside_tolerance) == [1, 1]


def test_receptor_gate_must_cover_exact_driver_universe(prior_factory) -> None:
    prior = prior_factory({"L1": {"G1": 1.0}, "L2": {"G1": 1.0}})
    with pytest.raises(ContractError, match="exactly match"):
        build_gated_target_basis(prior, ("G1",), {"L1": 1.0})
