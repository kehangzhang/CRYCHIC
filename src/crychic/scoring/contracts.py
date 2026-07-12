"""Frozen scoring functionals and sample communication score contracts."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import pandas as pd

from crychic.core import canonical_json, stable_id

CORE_COMPONENTS = ("availability", "downstream", "sender", "prior_quality")


class ScoringFunctionalStatus(StrEnum):
    """Whether a functional is exploratory in-sample or genuinely OOF."""

    EXPLORATORY_IN_SAMPLE = "exploratory_in_sample"
    OUT_OF_FOLD = "out_of_fold"


class CommunicationScoreStatus(StrEnum):
    """Validity of an integrated communication strength."""

    OK = "ok"
    MISSING_CORE_EVIDENCE = "missing_core_evidence"


def _stable_tuple(
    values: tuple[Hashable, ...], *, field_name: str
) -> tuple[Hashable, ...]:
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique values")
    try:
        return tuple(sorted(values, key=canonical_json))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} values must support canonical JSON serialization"
        ) from error


def _stable_names(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{field_name} must contain non-empty strings")
    normalized = tuple(value.strip() for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique values")
    return tuple(sorted(normalized))


def _component_mapping(
    values: Mapping[str, float],
    *,
    field_name: str,
    allow_zero: bool,
) -> Mapping[str, float]:
    if set(values) != set(CORE_COMPONENTS):
        raise ValueError(f"{field_name} must define exactly {list(CORE_COMPONENTS)}")
    normalized: dict[str, float] = {}
    for component in CORE_COMPONENTS:
        value = float(values[component])
        valid = math.isfinite(value) and (value >= 0 if allow_zero else value > 0)
        if not valid:
            relation = "non-negative" if allow_zero else "positive"
            raise ValueError(
                f"{field_name}[{component!r}] must be finite and {relation}"
            )
        normalized[component] = value
    if allow_zero and not any(normalized.values()):
        raise ValueError("at least one component weight must be positive")
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoringFunctional:
    """One immutable, contrast-level communication scoring functional."""

    contrast_name: str
    contrast_contexts: tuple[Hashable, ...]
    training_subject_ids: tuple[str, ...]
    interaction_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    status: ScoringFunctionalStatus = ScoringFunctionalStatus.EXPLORATORY_IN_SAMPLE
    fold_id: str | None = None
    component_weights: Mapping[str, float] = field(
        default_factory=lambda: dict.fromkeys(CORE_COMPONENTS, 1.0)
    )
    component_scales: Mapping[str, float] = field(
        default_factory=lambda: dict.fromkeys(CORE_COMPONENTS, 1.0)
    )
    frozen: bool = True
    output_scale: str = "unit_interval"
    scoring_function_id: str = field(init=False)
    interaction_universe_id: str = field(init=False)
    target_universe_id: str = field(init=False)
    reason_code: str | None = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.contrast_name, str) or not self.contrast_name.strip():
            raise ValueError("contrast_name must be a non-empty string")
        if not self.frozen:
            raise ValueError("a ScoringFunctional must be frozen before application")
        if self.output_scale != "unit_interval":
            raise ValueError("v0.1 scoring supports only output_scale='unit_interval'")
        status = ScoringFunctionalStatus(self.status)
        contexts = _stable_tuple(
            tuple(self.contrast_contexts), field_name="contrast_contexts"
        )
        if len(contexts) < 2:
            raise ValueError("contrast_contexts must contain at least two contexts")
        training_subjects = _stable_names(
            tuple(self.training_subject_ids), field_name="training_subject_ids"
        )
        interactions = _stable_names(
            tuple(self.interaction_ids), field_name="interaction_ids"
        )
        targets = _stable_names(tuple(self.target_ids), field_name="target_ids")
        weights = _component_mapping(
            self.component_weights,
            field_name="component_weights",
            allow_zero=True,
        )
        scales = _component_mapping(
            self.component_scales,
            field_name="component_scales",
            allow_zero=False,
        )
        if status is ScoringFunctionalStatus.OUT_OF_FOLD:
            if not isinstance(self.fold_id, str) or not self.fold_id.strip():
                raise ValueError("out-of-fold functional requires a non-empty fold_id")
            reason_code = None
        else:
            if self.fold_id is not None:
                raise ValueError("in-sample functional must not declare an OOF fold_id")
            reason_code = "exploratory_not_cross_fitted"

        interaction_universe_id = stable_id(
            "interaction_universe", {"interaction_ids": list(interactions)}
        )
        target_universe_id = stable_id("target_universe", {"target_ids": list(targets)})
        payload = {
            "component_scales": [[key, scales[key]] for key in CORE_COMPONENTS],
            "component_weights": [[key, weights[key]] for key in CORE_COMPONENTS],
            "contrast_contexts": list(contexts),
            "contrast_name": self.contrast_name.strip(),
            "fold_id": self.fold_id,
            "frozen": True,
            "interaction_universe_id": interaction_universe_id,
            "output_scale": self.output_scale,
            "status": status.value,
            "target_universe_id": target_universe_id,
            "training_subject_ids": list(training_subjects),
        }
        function_id = stable_id("scoring_function", payload)

        object.__setattr__(self, "contrast_name", self.contrast_name.strip())
        object.__setattr__(self, "contrast_contexts", contexts)
        object.__setattr__(self, "training_subject_ids", training_subjects)
        object.__setattr__(self, "interaction_ids", interactions)
        object.__setattr__(self, "target_ids", targets)
        object.__setattr__(self, "component_weights", weights)
        object.__setattr__(self, "component_scales", scales)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "interaction_universe_id", interaction_universe_id)
        object.__setattr__(self, "target_universe_id", target_universe_id)
        object.__setattr__(self, "scoring_function_id", function_id)

    @property
    def is_out_of_fold(self) -> bool:
        return self.status is ScoringFunctionalStatus.OUT_OF_FOLD

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-ready frozen functional definition."""

        return {
            "scoring_function_id": self.scoring_function_id,
            "contrast_name": self.contrast_name,
            "contrast_contexts": list(self.contrast_contexts),
            "training_subject_ids": list(self.training_subject_ids),
            "fold_id": self.fold_id,
            "interaction_ids": list(self.interaction_ids),
            "interaction_universe_id": self.interaction_universe_id,
            "target_ids": list(self.target_ids),
            "target_universe_id": self.target_universe_id,
            "component_weights": dict(self.component_weights),
            "component_scales": dict(self.component_scales),
            "status": self.status.value,
            "reason_code": self.reason_code,
            "frozen": self.frozen,
            "output_scale": self.output_scale,
        }


SCORE_COLUMNS = {
    "sample_id",
    "subject_id",
    "context",
    "sender",
    "receiver",
    "interaction_id",
    "mode",
    "availability",
    "downstream_activity",
    "sender_component",
    "prior_quality",
    "abundance_component",
    "comm_strength",
    "status",
    "reason_code",
    "functional_status",
    "functional_reason_code",
    "contrast",
    "fold_id",
    "scoring_function_id",
    "interaction_universe_id",
    "target_universe_id",
}


def validate_common_functional(table: pd.DataFrame) -> None:
    """Reject context-specific functionals within one contrast/fold score group."""

    required = {
        "contrast",
        "fold_id",
        "context",
        "functional_status",
        "scoring_function_id",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"score table is missing functional fields: {sorted(missing)}")
    formal = table["functional_status"] == ScoringFunctionalStatus.OUT_OF_FOLD.value
    for (contrast, fold_id), group in table.loc[formal].groupby(
        ["contrast", "fold_id"], dropna=False, sort=False
    ):
        if group["scoring_function_id"].nunique(dropna=False) != 1:
            raise ValueError(
                "out-of-fold contexts use different scoring_function_id values "
                f"for contrast={contrast!r}, fold_id={fold_id!r}"
            )


@dataclass(frozen=True, slots=True)
class CommunicationScores:
    """Component-preserving sample communication strengths."""

    table: pd.DataFrame
    functional: ScoringFunctional

    def __post_init__(self) -> None:
        table = self.table.copy(deep=True)
        missing = SCORE_COLUMNS.difference(table.columns)
        if missing:
            raise ValueError(f"communication score table is missing: {sorted(missing)}")
        forbidden_names = {
            "p",
            "p_value",
            "q",
            "q_value",
            "posterior",
            "posterior_probability",
        }
        forbidden = [
            str(column)
            for column in table
            if "probability" in str(column).lower()
            or str(column).lower() in forbidden_names
        ]
        if forbidden:
            raise ValueError(
                "scoring contains no probability or hypothesis-test fields; "
                "forbidden columns: "
                f"{sorted(forbidden)}"
            )
        if not table.empty:
            function_ids = set(table["scoring_function_id"])
            if function_ids != {self.functional.scoring_function_id}:
                raise ValueError(
                    "CommunicationScores rows must use their functional's stable ID"
                )
            if set(table["contrast"]) != {self.functional.contrast_name}:
                raise ValueError("score rows must use their functional's contrast")
            expected_provenance = {
                "functional_status": self.functional.status.value,
                "fold_id": self.functional.fold_id or "in_sample",
                "interaction_universe_id": self.functional.interaction_universe_id,
                "target_universe_id": self.functional.target_universe_id,
            }
            for column, expected in expected_provenance.items():
                if set(table[column]) != {expected}:
                    raise ValueError(
                        f"score rows have incompatible {column} provenance"
                    )
            observed_contexts = set(table["context"])
            unknown_contexts = observed_contexts.difference(
                self.functional.contrast_contexts
            )
            if unknown_contexts:
                raise ValueError(
                    "score rows contain contexts outside the functional: "
                    f"{unknown_contexts}"
                )
            if self.functional.is_out_of_fold and observed_contexts != set(
                self.functional.contrast_contexts
            ):
                raise ValueError(
                    "out-of-fold score table must contain every contrast context"
                )
        key = [
            "sample_id",
            "context",
            "sender",
            "receiver",
            "interaction_id",
            "mode",
            "scoring_function_id",
        ]
        if table.duplicated(key).any():
            raise ValueError("communication score primary key must be unique")
        if not set(table["mode"]).issubset({"state", "ecosystem"}):
            raise ValueError("communication score mode must be state or ecosystem")
        allowed_status = {status.value for status in CommunicationScoreStatus}
        if not set(table["status"]).issubset(allowed_status):
            raise ValueError("communication score status is not recognized")
        for column in (
            "availability",
            "downstream_activity",
            "sender_component",
            "prior_quality",
            "abundance_component",
            "comm_strength",
        ):
            numeric = pd.to_numeric(table[column], errors="coerce")
            invalid_type = table[column].notna() & numeric.isna()
            if invalid_type.any():
                raise ValueError(f"{column} must contain numeric values or NA")
            present = numeric.dropna()
            if ((present < 0) | (present > 1)).any():
                raise ValueError(f"{column} values must lie in [0, 1]")
        ok = table["status"] == CommunicationScoreStatus.OK.value
        if table.loc[ok, "comm_strength"].isna().any():
            raise ValueError("status=ok requires a finite comm_strength")
        if table.loc[~ok, "comm_strength"].notna().any():
            raise ValueError("missing score status requires comm_strength=NA")
        if table.loc[ok, "reason_code"].notna().any():
            raise ValueError("status=ok must not carry a missing-evidence reason")
        if table.loc[~ok, "reason_code"].isna().any():
            raise ValueError("missing score status requires an explicit reason_code")
        validate_common_functional(table)
        object.__setattr__(self, "table", table)

    @property
    def score_semantics(self) -> str:
        return "strength_not_probability"

    def for_mode(self, mode: str) -> pd.DataFrame:
        """Return an independent copy of state or ecosystem score rows."""

        if mode not in {"state", "ecosystem"}:
            raise ValueError("mode must be 'state' or 'ecosystem'")
        return self.table.loc[self.table["mode"] == mode].copy()
