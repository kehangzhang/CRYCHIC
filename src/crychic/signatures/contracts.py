"""Producer-owned contracts for gene and driver contribution signatures."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import pandas as pd

from crychic.core import ContractError


class SignatureStatus(StrEnum):
    """Availability of modeled signature quantities."""

    OK = "ok"
    EMPTY_ATTRIBUTION = "empty_attribution"
    ATTRIBUTION_FAILED = "attribution_failed"
    RESPONSE_NOT_ESTIMABLE = "response_not_estimable"


class DirectionAgreement(StrEnum):
    """Agreement of positive-prior prediction with observed response."""

    AGREES = "agrees"
    DISCORDANT = "discordant"
    NO_PREDICTION = "no_prediction"
    NOT_AVAILABLE = "not_available"


GENE_SIGNATURE_COLUMNS = (
    "signature_id",
    "context_id",
    "context",
    "receiver",
    "contrast",
    "gene",
    "gene_namespace",
    "observed",
    "predicted",
    "residual",
    "direction_consistent_predicted",
    "observed_direction",
    "predicted_direction",
    "direction_agreement",
    "status",
    "reason_code",
    "solver_status",
    "basis_id",
    "fold_id",
    "response_scale",
    "response_source",
    "direction_rule_version",
    "observed_rank",
    "predicted_rank",
    "residual_rank",
)

CONTRIBUTION_COLUMNS = (
    "contribution_id",
    "signature_id",
    "context_id",
    "context",
    "receiver",
    "contrast",
    "gene",
    "gene_namespace",
    "driver_id",
    "family_id",
    "coefficient",
    "basis_weight",
    "contribution",
    "direction_consistent_contribution",
    "direction_agreement",
    "status",
    "reason_code",
    "solver_status",
    "basis_id",
    "fold_id",
    "direction_rule_version",
    "contribution_rank",
)

_FORBIDDEN_INFERENCE_COLUMNS = {
    "p",
    "p_value",
    "q",
    "q_value",
    "posterior",
    "posterior_probability",
}


def _matches(series: pd.Series, value: object) -> pd.Series:
    return series.map(lambda observed: observed == value).astype(bool)


@dataclass(frozen=True, slots=True)
class SignatureTable:
    """Long-form gene signatures and sparse driver contributions."""

    genes: pd.DataFrame
    contributions: pd.DataFrame
    gene_namespace: str
    direction_rule_version: str
    schema_version: str = "0.1.0"

    def __post_init__(self) -> None:
        genes = self.genes.copy(deep=True)
        contributions = self.contributions.copy(deep=True)
        missing_gene = set(GENE_SIGNATURE_COLUMNS).difference(genes.columns)
        missing_contribution = set(CONTRIBUTION_COLUMNS).difference(
            contributions.columns
        )
        if missing_gene or missing_contribution:
            raise ContractError(
                "Signature tables are missing required columns",
                code="invalid_signature_schema",
                field="columns",
                remediation=(
                    "Build signatures through the producer-owned table constructor"
                ),
            )
        forbidden = _FORBIDDEN_INFERENCE_COLUMNS.intersection(
            set(genes.columns) | set(contributions.columns)
        )
        if forbidden:
            raise ContractError(
                "Signature tables cannot contain inferential fields: "
                + repr(sorted(forbidden)),
                code="forbidden_signature_inference",
                field="columns",
                remediation="Keep p/q/posterior quantities in inference results",
            )
        if genes["signature_id"].duplicated().any():
            raise ContractError(
                "Gene signature IDs must be unique",
                code="duplicate_signature_id",
                field="signature_id",
                remediation="Use the stable context/receiver/contrast/gene key",
            )
        if contributions["contribution_id"].duplicated().any():
            raise ContractError(
                "Contribution IDs must be unique",
                code="duplicate_contribution_id",
                field="contribution_id",
                remediation="Include the driver in contribution stable keys",
            )
        reconstructable = genes["status"].isin(
            [SignatureStatus.OK.value, SignatureStatus.EMPTY_ATTRIBUTION.value]
        )
        for row in genes.loc[reconstructable].itertuples(index=False):
            values = (
                float(cast(float, row.observed)),
                float(cast(float, row.predicted)),
                float(cast(float, row.residual)),
            )
            if any(not math.isfinite(value) for value in values) or not math.isclose(
                values[0], values[1] + values[2], rel_tol=1e-9, abs_tol=1e-10
            ):
                raise ContractError(
                    "Signature predicted plus residual must reconstruct observed",
                    code="invalid_signature_reconstruction",
                    field="residual",
                    remediation="Preserve the complete signed observed response",
                )
        successful = genes.loc[genes["status"] == SignatureStatus.OK.value]
        if not successful.empty:
            contribution_sum = contributions.groupby("signature_id", sort=False)[
                "contribution"
            ].sum()
            consistent_sum = contributions.groupby("signature_id", sort=False)[
                "direction_consistent_contribution"
            ].sum()
            for row in successful.itertuples(index=False):
                modeled = float(
                    cast(float, contribution_sum.get(row.signature_id, 0.0))
                )
                consistent = float(
                    cast(float, consistent_sum.get(row.signature_id, 0.0))
                )
                if not math.isclose(
                    modeled,
                    float(cast(float, row.predicted)),
                    rel_tol=1e-9,
                    abs_tol=1e-10,
                ):
                    raise ContractError(
                        "Driver contributions must sum to predicted response",
                        code="invalid_contribution_reconstruction",
                        field="contribution",
                        remediation=(
                            "Multiply each fitted coefficient by its basis column"
                        ),
                    )
                if not math.isclose(
                    consistent,
                    float(cast(float, row.direction_consistent_predicted)),
                    rel_tol=1e-9,
                    abs_tol=1e-10,
                ):
                    raise ContractError(
                        "Direction-consistent contributions do not match gene summary",
                        code="invalid_direction_contribution",
                        field="direction_consistent_contribution",
                        remediation="Apply one versioned direction rule at both grains",
                    )
        object.__setattr__(self, "genes", genes)
        object.__setattr__(self, "contributions", contributions)

    @property
    def inference_eligible(self) -> bool:
        """Signatures are descriptive producer artifacts, never formal inference."""

        return False

    def query_genes(
        self,
        *,
        context: object | None = None,
        receiver: object | None = None,
        contrast: str | None = None,
        status: SignatureStatus | str | None = None,
    ) -> pd.DataFrame:
        """Return a defensive copy filtered at gene-signature grain."""

        selected = pd.Series(True, index=self.genes.index, dtype=bool)
        if context is not None:
            selected &= _matches(self.genes["context"], context)
        if receiver is not None:
            selected &= _matches(self.genes["receiver"], receiver)
        if contrast is not None:
            selected &= self.genes["contrast"] == contrast
        if status is not None:
            selected &= self.genes["status"] == SignatureStatus(status).value
        return self.genes.loc[selected].copy(deep=True).reset_index(drop=True)

    def query_contributions(
        self,
        *,
        context: object | None = None,
        receiver: object | None = None,
        contrast: str | None = None,
        driver: str | None = None,
    ) -> pd.DataFrame:
        """Return a defensive copy filtered at driver-gene contribution grain."""

        selected = pd.Series(True, index=self.contributions.index, dtype=bool)
        if context is not None:
            selected &= _matches(self.contributions["context"], context)
        if receiver is not None:
            selected &= _matches(self.contributions["receiver"], receiver)
        if contrast is not None:
            selected &= self.contributions["contrast"] == contrast
        if driver is not None:
            selected &= self.contributions["driver_id"] == driver
        return (
            self.contributions.loc[selected]
            .copy(deep=True)
            .reset_index(drop=True)
        )
