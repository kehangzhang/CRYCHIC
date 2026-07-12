"""Typed contracts produced by AnnData input validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData


class InputMode(StrEnum):
    """Expression semantics selected during validation."""

    COUNTS = "counts"
    NORMALIZED_ONLY = "normalized_only"


ExpressionTransform = Literal["linear_normalized", "log1p_normalized"]


@dataclass(frozen=True, slots=True)
class InputSchema:
    """Declare where CRYCHIC input fields and expression values are stored."""

    context_keys: tuple[str, ...]
    counts_layer: str | None = "counts"
    sample_key: str = "sample_id"
    subject_key: str = "subject_id"
    cell_type_key: str = "cell_type"
    covariates: tuple[str, ...] = ()
    expression_layer: str | None = None
    expression_source: str | None = None
    expression_transform: ExpressionTransform | None = None
    normalized_zero_is_nondetection: bool = False
    species: str | None = None
    gene_namespace: str | None = None
    allow_duplicate_genes: bool = False

    def __post_init__(self) -> None:
        if not self.context_keys:
            raise ValueError("context_keys must contain at least one context field")
        declared = (
            self.sample_key,
            self.subject_key,
            self.cell_type_key,
            *self.context_keys,
            *self.covariates,
        )
        if any(not key for key in declared):
            raise ValueError("declared AnnData field names must be non-empty")
        if len(set(self.context_keys)) != len(self.context_keys):
            raise ValueError("context_keys must not contain duplicates")
        if len(set(self.covariates)) != len(self.covariates):
            raise ValueError("covariates must not contain duplicates")
        identifiers = (self.sample_key, self.subject_key, self.cell_type_key)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("sample, subject, and cell-type keys must be distinct")
        roles = (*identifiers, *self.context_keys, *self.covariates)
        if len(set(roles)) != len(roles):
            raise ValueError(
                "identifier, context, and covariate fields must be distinct"
            )
        for name in (
            self.counts_layer,
            self.expression_layer,
            self.expression_source,
            self.species,
            self.gene_namespace,
        ):
            if name is not None and (not name or name != name.strip()):
                raise ValueError(
                    "declared names and provenance labels must be non-empty"
                )


@dataclass(frozen=True, slots=True)
class InputReport:
    """Auditable summary of a validated AnnData object."""

    mode: InputMode
    reason_codes: tuple[str, ...]
    expression_location: str
    expression_source: str
    expression_transform: ExpressionTransform | None
    detection_eligible: bool
    input_inference_eligible: bool
    n_obs: int
    n_vars: int
    n_samples: int
    n_subjects: int
    matrix_kind: str
    matrix_dtype: str
    duplicate_genes: tuple[str, ...]
    sample_metadata: pd.DataFrame
    cell_type_support: pd.DataFrame
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidatedInput:
    """A non-owning, validated view over caller-owned AnnData."""

    adata: AnnData
    schema: InputSchema
    report: InputReport
    matrix: Any

    @property
    def mode(self) -> InputMode:
        return self.report.mode

    @property
    def is_counts(self) -> bool:
        return self.mode is InputMode.COUNTS

    @property
    def feature_ids(self) -> tuple[str, ...]:
        return tuple(map(str, self.adata.var_names))

    @property
    def context_keys(self) -> tuple[str, ...]:
        return self.schema.context_keys


class InputValidationError(ValueError):
    """Raised when an AnnData field violates the declared input contract."""
