"""Producer-owned contracts for sample-level receiver response estimates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from crychic.data import InputMode
from crychic.design import ContrastSpec


class ResponseStatus(StrEnum):
    """Estimability of one descriptive response quantity."""

    OK = "ok"
    NOT_ESTIMABLE = "not_estimable"


class ResponseMethod(StrEnum):
    """Diagnostic estimator used for a response contrast."""

    INDEPENDENT = "independent_context_means"
    PAIRED = "paired_subject_difference"
    PAIRED_COMPLETE_CASE = "paired_subject_difference_complete_case"
    EMM_INDEPENDENT = "formula_emm_independent_subjects"
    EMM_PAIRED = "formula_emm_paired_subject_difference"
    EMM_PAIRED_COMPLETE_CASE = "formula_emm_paired_complete_case"
    NOT_ESTIMABLE = "not_estimable"


_SAMPLE_COLUMNS = {
    "unit_id",
    "sample_id",
    "subject_id",
    "cell_type",
    "context_node",
    "response_eligible",
    "response_reason",
}
_CONTEXT_COLUMNS = {
    "receiver",
    "context",
    "gene",
    "mean_response",
    "n_samples",
    "n_subjects",
    "status",
    "reason_code",
}
_CONTRAST_COLUMNS = {
    "receiver",
    "gene",
    "contrast",
    "family",
    "mode",
    "effect",
    "standard_error",
    "z_score",
    "precision",
    "direction",
    "n_samples",
    "n_subjects",
    "paired",
    "method",
    "status",
    "reason_code",
}
_FORBIDDEN_INFERENCE_COLUMNS = {
    "p",
    "p_value",
    "q",
    "q_value",
    "posterior_probability",
}


@dataclass(frozen=True, slots=True)
class ResponseEstimate:
    """Complete continuous gene responses and diagnostic context contrasts.

    Rows of ``sample_values`` align exactly with ``sample_metadata``. Ineligible
    state-expression units are represented by all-``NaN`` rows, never zeros.
    The v0.1 contrast table is diagnostic and deliberately has no p/q fields.
    """

    sample_values: np.ndarray
    sample_metadata: pd.DataFrame
    context_means: pd.DataFrame
    contrasts: pd.DataFrame
    feature_ids: tuple[str, ...]
    contrast_specs: tuple[ContrastSpec, ...]
    input_mode: InputMode
    value_scale: str
    expression_source: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        expected_shape = (len(self.sample_metadata), len(self.feature_ids))
        if self.sample_values.shape != expected_shape:
            raise ValueError(
                f"sample_values has shape {self.sample_values.shape}, "
                f"expected {expected_shape}"
            )
        if not np.issubdtype(self.sample_values.dtype, np.floating):
            raise ValueError("sample_values must use a floating dtype to preserve NaN")
        missing_sample = _SAMPLE_COLUMNS.difference(self.sample_metadata.columns)
        missing_context = _CONTEXT_COLUMNS.difference(self.context_means.columns)
        missing_contrast = _CONTRAST_COLUMNS.difference(self.contrasts.columns)
        if missing_sample:
            raise ValueError(
                f"sample_metadata is missing columns: {sorted(missing_sample)}"
            )
        if missing_context:
            raise ValueError(
                f"context_means is missing columns: {sorted(missing_context)}"
            )
        if missing_contrast:
            raise ValueError(
                f"contrasts is missing columns: {sorted(missing_contrast)}"
            )
        forbidden = _FORBIDDEN_INFERENCE_COLUMNS.intersection(self.contrasts.columns)
        if forbidden:
            raise ValueError(
                "v0.1 diagnostic responses must not contain inferential columns: "
                f"{sorted(forbidden)}"
            )
        if self.sample_metadata["unit_id"].duplicated().any():
            raise ValueError("sample_metadata unit_id values must be unique")
        if not self.value_scale:
            raise ValueError("value_scale must be non-empty")

    @property
    def inference_eligible(self) -> bool:
        """v0.1 response estimates are diagnostic, never formal inference."""

        return False

    def sample_vector(self, receiver: object, gene: str) -> pd.Series:
        """Return the full sample vector, including NaN ineligible units."""

        try:
            gene_index = self.feature_ids.index(gene)
        except ValueError as error:
            raise KeyError(gene) from error
        selected = self.sample_metadata["cell_type"] == receiver
        return pd.Series(
            self.sample_values[selected.to_numpy(), gene_index],
            index=self.sample_metadata.loc[selected, "unit_id"],
            name=gene,
            dtype=float,
        )
