"""Deterministic cosine equivalence families for target-profile drivers."""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping

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


def _pair_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _candidate_key(
    left: int,
    right: int,
    *,
    similarity: float,
    names: Mapping[int, tuple[str, ...]],
) -> tuple[float, tuple[str, ...], tuple[str, ...], tuple[str, ...], int, int]:
    left_names = names[left]
    right_names = names[right]
    if right_names < left_names:
        left, right = right, left
        left_names, right_names = right_names, left_names
    return (
        -similarity,
        tuple(sorted((*left_names, *right_names))),
        left_names,
        right_names,
        left,
        right,
    )


def _complete_link_split(
    indices: list[int],
    *,
    driver_ids: tuple[str, ...],
    pairwise_similarity: dict[tuple[int, int], float],
    threshold: float,
) -> tuple[tuple[int, ...], ...]:
    ordered = sorted(indices, key=lambda index: driver_ids[index])
    clusters: dict[int, tuple[int, ...]] = {index: (index,) for index in ordered}
    names: dict[int, tuple[str, ...]] = {
        index: (driver_ids[index],) for index in ordered
    }
    active = set(ordered)
    similarities: dict[tuple[int, int], float] = {}
    candidates: list[
        tuple[float, tuple[str, ...], tuple[str, ...], tuple[str, ...], int, int]
    ] = []
    for offset, left in enumerate(ordered):
        for right in ordered[offset + 1 :]:
            similarity = pairwise_similarity.get(_pair_key(left, right), 0.0)
            if similarity + _COSINE_TOLERANCE < threshold:
                continue
            similarities[_pair_key(left, right)] = similarity
            heapq.heappush(
                candidates,
                _candidate_key(
                    left,
                    right,
                    similarity=similarity,
                    names=names,
                ),
            )

    next_cluster = max(ordered, default=-1) + 1
    while candidates:
        *_, left, right = heapq.heappop(candidates)
        if left not in active or right not in active:
            continue
        merged = next_cluster
        next_cluster += 1
        clusters[merged] = tuple(sorted((*clusters[left], *clusters[right])))
        names[merged] = tuple(sorted((*names[left], *names[right])))
        remaining = active.difference({left, right})
        active.remove(left)
        active.remove(right)
        for other in remaining:
            left_similarity = similarities.get(_pair_key(left, other))
            right_similarity = similarities.get(_pair_key(right, other))
            if left_similarity is None or right_similarity is None:
                continue
            similarity = min(left_similarity, right_similarity)
            if similarity + _COSINE_TOLERANCE < threshold:
                continue
            similarities[_pair_key(merged, other)] = similarity
            heapq.heappush(
                candidates,
                _candidate_key(
                    merged,
                    other,
                    similarity=similarity,
                    names=names,
                ),
            )
        active.add(merged)
    return tuple(
        clusters[index] for index in sorted(active, key=lambda value: names[value])
    )


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
    cosine = sparse.coo_matrix(basis.normalized_profiles.T @ basis.normalized_profiles)
    pairwise_similarity: dict[tuple[int, int], float] = {}
    for left, right, raw_value in zip(cosine.row, cosine.col, cosine.data, strict=True):
        if left >= right:
            continue
        value = min(1.0, max(0.0, float(raw_value)))
        if value + _COSINE_TOLERANCE >= cosine_threshold:
            # Complete-link only distinguishes passing pairs from failures;
            # omitted sub-threshold similarities already resolve to zero.
            pairwise_similarity[(int(left), int(right))] = value
            disjoint.union(int(left), int(right))

    if cosine_threshold <= _COSINE_TOLERANCE and n_drivers:
        for index in range(1, n_drivers):
            disjoint.union(0, index)

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
            pairwise_similarity=pairwise_similarity,
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
