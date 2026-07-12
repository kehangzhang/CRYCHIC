"""Deterministic cosine equivalence families for target-profile drivers."""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse

from crychic.core import ContractError, stable_id

from .contracts import DriverFamilyDefinition, GatedTargetBasis

_COSINE_TOLERANCE = 1e-12


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        self.parent[larger] = smaller


def _cluster_names(
    indices: tuple[int, ...], driver_ids: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(sorted(driver_ids[index] for index in indices))


def _complete_link_similarity(
    left: tuple[int, ...],
    right: tuple[int, ...],
    cosine: sparse.csr_matrix,
) -> float:
    return min(
        min(1.0, max(0.0, float(cosine[left_index, right_index])))
        for left_index in left
        for right_index in right
    )


def _complete_link_split(
    indices: list[int],
    *,
    driver_ids: tuple[str, ...],
    cosine: sparse.csr_matrix,
    threshold: float,
) -> tuple[tuple[int, ...], ...]:
    clusters: list[tuple[int, ...]] = [
        (index,)
        for index in sorted(indices, key=lambda index: driver_ids[index])
    ]
    while True:
        best: tuple[
            tuple[float, tuple[str, ...], tuple[str, ...], tuple[str, ...]],
            int,
            int,
        ] | None = None
        for left_index, left in enumerate(clusters):
            left_names = _cluster_names(left, driver_ids)
            for right_index in range(left_index + 1, len(clusters)):
                right = clusters[right_index]
                similarity = _complete_link_similarity(left, right, cosine)
                if similarity + _COSINE_TOLERANCE < threshold:
                    continue
                right_names = _cluster_names(right, driver_ids)
                merged_names = tuple(sorted((*left_names, *right_names)))
                key = (-similarity, merged_names, left_names, right_names)
                candidate = (key, left_index, right_index)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None:
            break
        _, left_index, right_index = best
        merged = tuple(sorted((*clusters[left_index], *clusters[right_index])))
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left_index, right_index}
        ]
        clusters.append(merged)
        clusters.sort(key=lambda cluster: _cluster_names(cluster, driver_ids))
    return tuple(clusters)


def cluster_driver_families(
    basis: GatedTargetBasis,
    *,
    cosine_threshold: float = 0.95,
) -> tuple[DriverFamilyDefinition, ...]:
    """Cluster drivers by deterministic complete-link cosine families.

    High-cosine connected components define candidate blocks. Each block is
    then split by complete linkage so every pair in a final family meets the
    threshold. Clustering uses pre-gate unit-L2 profiles, so a context-specific
    receptor gate cannot change the molecular equivalence-class definition.
    """

    if not math.isfinite(cosine_threshold) or not 0 <= cosine_threshold <= 1:
        raise ContractError(
            "cosine_threshold must be finite and lie in [0, 1]",
            code="invalid_family_threshold",
            field="cosine_threshold",
            remediation="Choose a pre-registered target-profile cosine threshold",
        )
    n_drivers = len(basis.driver_ids)
    disjoint = _DisjointSet(n_drivers)
    cosine = sparse.coo_matrix(
        basis.normalized_profiles.T @ basis.normalized_profiles
    )
    for left, right, raw_value in zip(
        cosine.row, cosine.col, cosine.data, strict=True
    ):
        if left >= right:
            continue
        value = min(1.0, max(0.0, float(raw_value)))
        if value + _COSINE_TOLERANCE >= cosine_threshold:
            disjoint.union(int(left), int(right))

    components: dict[int, list[int]] = {}
    for index in range(n_drivers):
        components.setdefault(disjoint.find(index), []).append(index)
    cosine_csr = cosine.tocsr()
    families: list[DriverFamilyDefinition] = []
    split_components = (
        cluster
        for indices in components.values()
        for cluster in _complete_link_split(
            indices,
            driver_ids=basis.driver_ids,
            cosine=cosine_csr,
            threshold=cosine_threshold,
        )
    )
    for indices in split_components:
        members = tuple(sorted(basis.driver_ids[index] for index in indices))
        if len(indices) == 1:
            mean_cosine = 0.0
        else:
            similarities: list[float] = []
            for offset, left in enumerate(indices):
                for right in indices[offset + 1 :]:
                    similarities.append(float(cosine_csr[left, right]))
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
