from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import pytest

from crychic.data import InputMode
from crychic.resources import (
    GeneNamespace,
    MappingReport,
    Species,
    TargetPrior,
)
from crychic.response import ResponseEstimate, ResponseStatus


def make_prior(columns: Mapping[str, Mapping[str, float]]) -> TargetPrior:
    driver_ids = tuple(sorted(columns))
    target_ids = tuple(
        sorted({target for links in columns.values() for target in links})
    )
    target_index = {target: index for index, target in enumerate(target_ids)}
    indptr = [0]
    indices: list[int] = []
    weights: list[float] = []
    for driver in driver_ids:
        for target, weight in sorted(columns[driver].items()):
            indices.append(target_index[target])
            weights.append(weight)
        indptr.append(len(weights))
    return TargetPrior(
        resource_id="signature_prior",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="ligand",
        target_ids=target_ids,
        driver_ids=driver_ids,
        indptr=tuple(indptr),
        target_indices=tuple(indices),
        weights=tuple(weights),
        ranks=None,
        direction=1,
        evidence="synthetic",
        mapping_report=MappingReport(
            source_rows=len(weights),
            loaded_rows=len(weights),
            mapped_entities=len(driver_ids) + len(target_ids),
        ),
        manifest_digest="signature-manifest",
    )


def make_response(
    values: Sequence[float],
    *,
    receiver: str = "R",
    contrast: str = "treated-v-control",
    statuses: Sequence[str] | None = None,
    reasons: Sequence[str | None] | None = None,
) -> ResponseEstimate:
    feature_ids = tuple(f"G{index + 1}" for index in range(len(values)))
    resolved_statuses = list(statuses or [ResponseStatus.OK.value] * len(values))
    resolved_reasons = list(reasons or [None] * len(values))
    contrast_rows = []
    for gene, value, status, reason in zip(
        feature_ids,
        values,
        resolved_statuses,
        resolved_reasons,
        strict=True,
    ):
        contrast_rows.append(
            {
                "receiver": receiver,
                "gene": gene,
                "contrast": contrast,
                "family": "local",
                "mode": "local_neighbor",
                "effect": value,
                "standard_error": 1.0,
                "z_score": value,
                "precision": 1.0,
                "direction": (
                    "positive"
                    if value > 0
                    else "negative"
                    if value < 0
                    else "zero"
                ),
                "n_samples": 4,
                "n_subjects": 4,
                "paired": False,
                "method": "independent_context_means",
                "status": status,
                "reason_code": reason,
            }
        )
    sample_columns = [
        "unit_id",
        "sample_id",
        "subject_id",
        "cell_type",
        "context_node",
        "response_eligible",
        "response_reason",
    ]
    context_columns = [
        "receiver",
        "context",
        "gene",
        "mean_response",
        "n_samples",
        "n_subjects",
        "status",
        "reason_code",
    ]
    return ResponseEstimate(
        sample_values=np.empty((0, len(feature_ids)), dtype=float),
        sample_metadata=pd.DataFrame(columns=sample_columns),
        context_means=pd.DataFrame(columns=context_columns),
        contrasts=pd.DataFrame(contrast_rows),
        feature_ids=feature_ids,
        contrast_specs=(),
        input_mode=InputMode.NORMALIZED_ONLY,
        value_scale="diagnostic_z",
        expression_source="synthetic_response",
        reason_codes=("normalized_only",),
    )


@pytest.fixture
def prior_factory():
    return make_prior


@pytest.fixture
def response_factory():
    return make_response
