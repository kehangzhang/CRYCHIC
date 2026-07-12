"""Deterministic cosine equivalence families for target-profile drivers."""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse

from crychic.core import ContractError, stable_id

from .contracts import DriverFamilyDefinition, GatedTargetBasis


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


def cluster_driver_families(
    basis: GatedTargetBasis,
    *,
    cosine_threshold: float = 0.95,
) -> tuple[DriverFamilyDefinition, ...]:
    """Cluster drivers by connected components of high profile cosine.

    Clustering uses pre-gate unit-L2 profiles, so a context-specific receptor
    gate cannot change the molecular equivalence-class definition.
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
        if value + 1e-12 >= cosine_threshold:
            disjoint.union(int(left), int(right))

    components: dict[int, list[int]] = {}
    for index in range(n_drivers):
        components.setdefault(disjoint.find(index), []).append(index)
    cosine_csr = cosine.tocsr()
    families: list[DriverFamilyDefinition] = []
    for indices in components.values():
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
