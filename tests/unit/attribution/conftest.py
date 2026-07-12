from __future__ import annotations

from collections.abc import Mapping

import pytest

from crychic.resources import (
    GeneNamespace,
    MappingReport,
    Species,
    TargetPrior,
)


def make_prior(
    columns: Mapping[str, Mapping[str, float]],
    *,
    resource_id: str = "tiny_prior",
) -> TargetPrior:
    driver_ids = tuple(sorted(columns))
    target_ids = tuple(
        sorted({target for links in columns.values() for target in links})
    )
    target_index = {target: index for index, target in enumerate(target_ids)}
    indptr = [0]
    target_indices: list[int] = []
    weights: list[float] = []
    ranks: list[int] = []
    for driver in driver_ids:
        ordered = sorted(
            columns[driver].items(), key=lambda item: (-item[1], item[0])
        )
        for rank, (target, weight) in enumerate(ordered, start=1):
            target_indices.append(target_index[target])
            weights.append(weight)
            ranks.append(rank)
        indptr.append(len(weights))
    return TargetPrior(
        resource_id=resource_id,
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="ligand",
        target_ids=target_ids,
        driver_ids=driver_ids,
        indptr=tuple(indptr),
        target_indices=tuple(target_indices),
        weights=tuple(weights),
        ranks=tuple(ranks),
        direction=1,
        evidence="synthetic",
        mapping_report=MappingReport(
            source_rows=len(weights),
            loaded_rows=len(weights),
            mapped_entities=len(driver_ids) + len(target_ids),
        ),
        manifest_digest="tiny-manifest",
    )


@pytest.fixture
def prior_factory():
    """Return the tiny immutable TargetPrior builder."""

    return make_prior
