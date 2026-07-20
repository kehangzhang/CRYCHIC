"""Producer-owned semantic score views over one intact cross-fit run.

The four tables in this module deliberately keep different grains.  They are
descriptive views of already-produced fold artifacts and never refit a model or
add inferential claims.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from crychic.core import ContractError, canonical_json, stable_id
from crychic.core._validation import record_validation, validation_is_cached

from .crossfit import CrossFitArtifacts

SEMANTIC_AVAILABILITY_COLUMNS = (
    "crossfit_id",
    "crossfit_spec_id",
    "repeat_id",
    "fold_id",
    "sample_id",
    "subject_id",
    "context_id",
    "sender",
    "receiver",
    "interaction_id",
    "mode",
    "availability_score",
    "status",
    "reason_code",
    "training_artifact_id",
    "availability_application_id",
    "filter_universe_id",
    "source_table_digest",
    "formal_inference_allowed",
)

SEMANTIC_RECEIVER_PROGRAM_COLUMNS = (
    "crossfit_id",
    "crossfit_spec_id",
    "repeat_id",
    "fold_id",
    "contrast_id",
    "contrast",
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "family_id",
    "receiver_program_score",
    "status",
    "reason_code",
    "receiver_program_training_artifact_id",
    "receiver_program_application_id",
    "source_table_digest",
    "formal_inference_allowed",
)

SEMANTIC_INTEGRATED_LR_COLUMNS = (
    "crossfit_id",
    "crossfit_spec_id",
    "repeat_id",
    "fold_id",
    "contrast_id",
    "contrast",
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "family_id",
    "driver_id",
    "interaction_id",
    "mode",
    "integrated_lr_score",
    "status",
    "reason_code",
    "source_component",
    "family_common_functional_id",
    "family_common_application_id",
    "family_common_binding_id",
    "source_table_digest",
    "score_version",
    "formal_inference_allowed",
)

SEMANTIC_DIFFERENTIAL_EFFECT_COLUMNS = (
    "crossfit_id",
    "crossfit_spec_id",
    "repeat_id",
    "fold_id",
    "contrast_id",
    "contrast",
    "receiver",
    "subject_id",
    "family_id",
    "differential_effect",
    "status",
    "reason_code",
    "effect_semantics",
    "family_common_functional_id",
    "family_common_application_id",
    "family_common_binding_id",
    "source_table_digest",
    "score_version",
    "formal_inference_allowed",
)

SEMANTIC_SCORE_OUTPUTS = (
    "availability_score",
    "receiver_program_score",
    "integrated_lr_score",
    "differential_effect",
)

_TABLE_COLUMNS = {
    "availability_score": SEMANTIC_AVAILABILITY_COLUMNS,
    "receiver_program_score": SEMANTIC_RECEIVER_PROGRAM_COLUMNS,
    "integrated_lr_score": SEMANTIC_INTEGRATED_LR_COLUMNS,
    "differential_effect": SEMANTIC_DIFFERENTIAL_EFFECT_COLUMNS,
}
_VALUE_COLUMNS = {
    "availability_score": "availability_score",
    "receiver_program_score": "receiver_program_score",
    "integrated_lr_score": "integrated_lr_score",
    "differential_effect": "differential_effect",
}
_GRAINS = {
    "availability_score": ("fold_x_sample_x_sender_x_receiver_x_interaction_x_mode"),
    "receiver_program_score": ("fold_x_contrast_x_sample_x_receiver_x_family"),
    "integrated_lr_score": ("fold_x_contrast_x_sample_x_receiver_x_family_x_lr_x_mode"),
    "differential_effect": ("fold_x_contrast_x_subject_x_receiver_x_family"),
}
_KEY_COLUMNS = {
    "availability_score": (
        "fold_id",
        "sample_id",
        "sender",
        "receiver",
        "interaction_id",
        "mode",
    ),
    "receiver_program_score": (
        "fold_id",
        "contrast_id",
        "sample_id",
        "receiver",
        "family_id",
    ),
    "integrated_lr_score": (
        "fold_id",
        "contrast_id",
        "sample_id",
        "receiver",
        "family_id",
        "driver_id",
        "interaction_id",
        "mode",
    ),
    "differential_effect": (
        "fold_id",
        "contrast_id",
        "receiver",
        "subject_id",
        "family_id",
    ),
}
_ROW_STATUSES = frozenset({"observed", "not_estimable", "structural_zero"})
_VIEW_STATUSES = frozenset({"produced", "not_estimable", "not_produced"})
_COLLECTION_MARKER = "crychic.semantic_score_collection.v1"
_SCHEMA_VERSION = "1"
_INTEGRATED_SOURCE_COMPONENT = "sender_unresolved_strength"
_DIFFERENTIAL_SEMANTICS = (
    "descriptive_subject_receiver_null_vs_single_family_loss_ratio_v1"
)
_NO_COMMON_SCORING_REASON = "family_common_scoring_not_requested_without_penalty_tuning"
_DIRECTIONAL_DEDICATED_REASON = (
    "directional_contrasts_require_dedicated_integrated_lr_collection"
)
_DIRECTIONAL_EXCLUDED_REASON = (
    "directional_contrasts_excluded_from_generic_semantic_view"
)


def _directional_contrast_names(artifacts: CrossFitArtifacts) -> frozenset[str]:
    return frozenset(
        contrast.name
        for pair in artifacts.spec.directional_pairs
        for contrast in (pair.forward_contrast, pair.reverse_contrast)
    )


def _cell_token(value: object) -> dict[str, object]:
    if value is None or value is pd.NA or value is pd.NaT:
        return {"type": "missing", "value": None}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return {"type": "missing", "value": None}
        if not math.isfinite(value):
            raise ValueError("semantic score tables cannot contain infinite values")
        return {"type": "float", "value": value.hex()}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": canonical_json(value),
    }


def _table_digest(name: str, table: pd.DataFrame) -> str:
    kind = "semantic_score_table"
    prefix = (
        f'{{"components":{{"columns":{canonical_json(list(table.columns))},"rows":['
    )
    suffix = (
        f'],"table_name":{canonical_json(name)}}},'
        f'"kind":"{kind}","schema_version":"{_SCHEMA_VERSION}"}}'
    )

    missing_token = '{"type":"missing","value":null}'
    false_token = '{"type":"bool","value":false}'
    true_token = '{"type":"bool","value":true}'
    string_tokens: dict[str, str] = {}
    integer_tokens: dict[int, str] = {}

    def encoded_cell(value: object) -> str:
        if value is None or value is pd.NA or value is pd.NaT:
            return missing_token
        if isinstance(value, np.generic):
            value = value.item()
        if type(value) is float:
            if math.isnan(value):
                return missing_token
            if not math.isfinite(value):
                raise ValueError("semantic score tables cannot contain infinite values")
            return f'{{"type":"float","value":"{value.hex()}"}}'
        if isinstance(value, bool):
            return true_token if value else false_token
        if type(value) is int:
            token = integer_tokens.get(value)
            if token is None:
                token = f'{{"type":"int","value":{value}}}'
                if len(integer_tokens) < 4_096:
                    integer_tokens[value] = token
            return token
        if type(value) is str:
            token = string_tokens.get(value)
            if token is None:
                encoded_value = json.dumps(
                    value,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                token = f'{{"type":"str","value":{encoded_value}}}'
                if len(string_tokens) < 65_536:
                    string_tokens[value] = token
            return token
        return json.dumps(
            _cell_token(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def encoded_rows() -> Iterator[str]:
        for row in table.itertuples(index=False, name=None):
            yield "[" + ",".join(encoded_cell(value) for value in row) + "]"

    digest = hashlib.sha256()
    digest.update(prefix.encode("ascii"))
    previous: str | None = None
    row_count = 0
    for encoded in encoded_rows():
        if previous is not None and encoded < previous:
            break
        if row_count:
            digest.update(b",")
        digest.update(encoded.encode("ascii"))
        previous = encoded
        row_count += 1
    else:
        digest.update(suffix.encode("ascii"))
        return f"{kind}_{digest.hexdigest()}"

    rows = sorted(encoded_rows())
    digest = hashlib.sha256()
    digest.update(prefix.encode("ascii"))
    digest.update(",".join(rows).encode("ascii"))
    digest.update(suffix.encode("ascii"))
    return f"{kind}_{digest.hexdigest()}"


def _optional_reason(value: object) -> str | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    result = str(value)
    return result if result else None


def _contrast_id(contrast: object) -> str:
    to_dict = getattr(contrast, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("contrast must expose to_dict()")
    return str(stable_id("contrast", to_dict()))


def _availability_table(artifacts: CrossFitArtifacts) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    identifiers = [
        "sample_id",
        "subject_id",
        "context_id",
        "sender",
        "receiver",
        "interaction_id",
    ]
    for fold in sorted(artifacts.folds, key=lambda item: item.fold_id):
        application = fold.application
        source = application.availability.sample_interactions
        missing = set(identifiers).difference(source.columns)
        if missing:
            raise ValueError(
                "availability source is missing semantic identifiers: "
                f"{sorted(missing)}"
            )
        source_digest = _table_digest("heldout_availability", source)
        for mode, score_column, status_column, reason_column in (
            (
                "state",
                "availability_state",
                "state_status",
                "state_reason_code",
            ),
            (
                "ecosystem",
                "availability_ecosystem",
                "ecosystem_status",
                "ecosystem_reason_code",
            ),
        ):
            required = {score_column, status_column, reason_column}
            if not required.issubset(source.columns):
                raise ValueError(
                    "availability source is missing semantic values: "
                    f"{sorted(required.difference(source.columns))}"
                )
            part = source.loc[:, identifiers].copy()
            score = pd.to_numeric(source[score_column], errors="coerce")
            raw_status = source[status_column].astype("object")
            observed = score.notna()
            if bool((observed & ~raw_status.eq("observed")).any()):
                raise ValueError(
                    "finite heldout availability must have observed source status"
                )
            raw_reasons = source[reason_column].astype("object").map(_optional_reason)
            reasons = raw_reasons.where(
                ~observed,
                None,
            ).where(
                observed | raw_reasons.notna(),
                "availability_not_estimable",
            )
            part["crossfit_id"] = artifacts.crossfit_id
            part["crossfit_spec_id"] = artifacts.spec.spec_id
            part["repeat_id"] = artifacts.spec.repeat_id
            part["fold_id"] = fold.fold_id
            part["mode"] = mode
            part["availability_score"] = score.where(observed, np.nan)
            part["status"] = np.where(observed, "observed", "not_estimable")
            part["reason_code"] = reasons
            part["training_artifact_id"] = fold.training.training_artifact_id
            part["availability_application_id"] = application.application_id
            part["filter_universe_id"] = application.availability.filter_universe_id
            part["source_table_digest"] = source_digest
            part["formal_inference_allowed"] = False
            parts.append(part.loc[:, list(SEMANTIC_AVAILABILITY_COLUMNS)])
    if not parts:
        return pd.DataFrame(columns=SEMANTIC_AVAILABILITY_COLUMNS)
    return pd.concat(parts, ignore_index=True).sort_values(
        list(_KEY_COLUMNS["availability_score"]),
        kind="stable",
        ignore_index=True,
    )


def _receiver_program_table(artifacts: CrossFitArtifacts) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fold in sorted(artifacts.folds, key=lambda item: item.fold_id):
        contrasts = {
            encoder.contrast.name: (
                _contrast_id(encoder.contrast),
                encoder.contrast.name,
            )
            for encoder in fold.design_encoders
        }
        for application in fold.receiver_program_applications:
            source = application.to_table()
            contrast_name = application.training_artifact.contrast_name
            try:
                contrast_id, contrast = contrasts[contrast_name]
            except KeyError as error:  # pragma: no cover - fold contract guards this
                raise ValueError(
                    "receiver-program source references an unknown contrast"
                ) from error
            source_digest = _table_digest("receiver_program_score", source)
            part = source.copy(deep=True)
            part["crossfit_id"] = artifacts.crossfit_id
            part["crossfit_spec_id"] = artifacts.spec.spec_id
            part["repeat_id"] = artifacts.spec.repeat_id
            part["fold_id"] = fold.fold_id
            part["contrast_id"] = contrast_id
            part["contrast"] = contrast
            part["source_table_digest"] = source_digest
            part["formal_inference_allowed"] = False
            parts.append(part.loc[:, list(SEMANTIC_RECEIVER_PROGRAM_COLUMNS)])
    if not parts:
        return pd.DataFrame(columns=SEMANTIC_RECEIVER_PROGRAM_COLUMNS)
    return pd.concat(parts, ignore_index=True).sort_values(
        list(_KEY_COLUMNS["receiver_program_score"]),
        kind="stable",
        ignore_index=True,
    )


def _integrated_lr_table(artifacts: CrossFitArtifacts) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    directional_names = _directional_contrast_names(artifacts)
    for fold in sorted(artifacts.folds, key=lambda item: item.fold_id):
        for functional, application, binding in zip(
            fold.family_common_functionals,
            fold.family_common_applications,
            fold.family_common_bindings,
            strict=True,
        ):
            if functional.contrast_name in directional_names:
                continue
            source = application.member_scores
            part = source.loc[
                :,
                [
                    "sample_id",
                    "subject_id",
                    "context_id",
                    "receiver",
                    "family_id",
                    "driver_id",
                    "interaction_id",
                    "mode",
                    "sender_unresolved_strength",
                    "status",
                    "reason_code",
                    "score_version",
                ],
            ].copy()
            part.rename(
                columns={"sender_unresolved_strength": "integrated_lr_score"},
                inplace=True,
            )
            part["status"] = part["status"].replace({"ok": "observed"})
            part["crossfit_id"] = artifacts.crossfit_id
            part["crossfit_spec_id"] = artifacts.spec.spec_id
            part["repeat_id"] = artifacts.spec.repeat_id
            part["fold_id"] = fold.fold_id
            part["contrast_id"] = functional.contrast_manifest_id
            part["contrast"] = functional.contrast_name
            part["source_component"] = _INTEGRATED_SOURCE_COMPONENT
            part["family_common_functional_id"] = functional.family_common_functional_id
            part["family_common_application_id"] = application.application_id
            part["family_common_binding_id"] = binding.binding_id
            part["source_table_digest"] = application.member_scores_digest
            part["formal_inference_allowed"] = False
            parts.append(part.loc[:, list(SEMANTIC_INTEGRATED_LR_COLUMNS)])
    if not parts:
        return pd.DataFrame(columns=SEMANTIC_INTEGRATED_LR_COLUMNS)
    return pd.concat(parts, ignore_index=True).sort_values(
        list(_KEY_COLUMNS["integrated_lr_score"]),
        kind="stable",
        ignore_index=True,
    )


def _differential_effect_table(artifacts: CrossFitArtifacts) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    directional_names = _directional_contrast_names(artifacts)
    for fold in sorted(artifacts.folds, key=lambda item: item.fold_id):
        for functional, application, binding in zip(
            fold.family_common_functionals,
            fold.family_common_applications,
            fold.family_common_bindings,
            strict=True,
        ):
            if functional.contrast_name in directional_names:
                continue
            source = application.subject_differential
            part = source.loc[
                :,
                [
                    "subject_id",
                    "family_id",
                    "differential_effect",
                    "status",
                    "reason_code",
                ],
            ].copy()
            part["crossfit_id"] = artifacts.crossfit_id
            part["crossfit_spec_id"] = artifacts.spec.spec_id
            part["repeat_id"] = artifacts.spec.repeat_id
            part["fold_id"] = fold.fold_id
            part["contrast_id"] = functional.contrast_manifest_id
            part["contrast"] = functional.contrast_name
            part["receiver"] = functional.receiver
            part["effect_semantics"] = _DIFFERENTIAL_SEMANTICS
            part["family_common_functional_id"] = functional.family_common_functional_id
            part["family_common_application_id"] = application.application_id
            part["family_common_binding_id"] = binding.binding_id
            part["source_table_digest"] = application.subject_differential_digest
            part["score_version"] = functional.score_version
            part["formal_inference_allowed"] = False
            parts.append(part.loc[:, list(SEMANTIC_DIFFERENTIAL_EFFECT_COLUMNS)])
    if not parts:
        return pd.DataFrame(columns=SEMANTIC_DIFFERENTIAL_EFFECT_COLUMNS)
    return pd.concat(parts, ignore_index=True).sort_values(
        list(_KEY_COLUMNS["differential_effect"]),
        kind="stable",
        ignore_index=True,
    )


def _validate_table(
    name: str,
    table: pd.DataFrame,
    *,
    crossfit_id: str,
    crossfit_spec_id: str,
    repeat_id: str,
) -> None:
    columns = _TABLE_COLUMNS[name]
    if tuple(table.columns) != columns:
        raise ValueError(f"{name} columns do not match the semantic contract")
    if table.empty:
        return
    expected_lineage = {
        "crossfit_id": crossfit_id,
        "crossfit_spec_id": crossfit_spec_id,
        "repeat_id": repeat_id,
    }
    for column, expected in expected_lineage.items():
        if not bool(table[column].astype(str).eq(expected).all()):
            raise ValueError(f"{name}.{column} changed collection lineage")
    if bool(table.duplicated(list(_KEY_COLUMNS[name])).any()):
        raise ValueError(f"{name} contains duplicate semantic grain rows")
    inference_flags = table["formal_inference_allowed"]
    if inference_flags.dtype != np.dtype(bool) and any(
        type(value) is not bool for value in inference_flags.tolist()
    ):
        raise ValueError(f"{name} cannot claim formal inference")
    if bool(inference_flags.any()):
        raise ValueError(f"{name} cannot claim formal inference")

    value_column = _VALUE_COLUMNS[name]
    statuses = table["status"].astype(str)
    if not bool(statuses.isin(_ROW_STATUSES).all()):
        raise ValueError(f"{name} contains an unsupported row status")

    raw_values = table[value_column]
    raw_reasons = table["reason_code"]
    fast_numeric_values = (
        pd.api.types.is_numeric_dtype(raw_values.dtype)
        and not pd.api.types.is_complex_dtype(raw_values.dtype)
        and not pd.api.types.is_datetime64_any_dtype(raw_values.dtype)
        and not pd.api.types.is_timedelta64_dtype(raw_values.dtype)
    )
    fast_reasons = all(
        value is None
        or value is pd.NA
        or value is pd.NaT
        or type(value) in {str, float, np.float64}
        for value in raw_reasons.to_numpy(dtype="object", copy=False)
    )
    if not fast_numeric_values or not fast_reasons:
        for raw_value, raw_status, raw_reason in table.loc[
            :, [value_column, "status", "reason_code"]
        ].itertuples(index=False, name=None):
            status = str(raw_status)
            reason = _optional_reason(raw_reason)
            missing = raw_value is None or bool(pd.isna(cast(Any, raw_value)))
            value = None if missing else float(cast(Any, raw_value))
            if status not in _ROW_STATUSES:
                raise ValueError(f"{name} contains an unsupported row status")
            if status == "observed" and (value is None or reason is not None):
                raise ValueError(f"{name} observed rows require value and no reason")
            if status == "not_estimable" and (value is not None or reason is None):
                raise ValueError(
                    f"{name} not-estimable rows require NA and a reason"
                )
            if status == "structural_zero" and (value != 0.0 or reason is None):
                raise ValueError(
                    f"{name} structural-zero rows require zero and a reason"
                )
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} values must be finite")
            if (
                name != "differential_effect"
                and value is not None
                and not 0 <= value <= 1
            ):
                raise ValueError(f"{name} values must lie in [0, 1]")
        return

    missing_values = raw_values.isna().to_numpy(dtype=bool, copy=False)
    numeric_values: np.ndarray = np.full(len(table), np.nan, dtype=float)
    try:
        numeric_values[~missing_values] = pd.to_numeric(
            raw_values.loc[~missing_values], errors="raise"
        ).to_numpy(dtype=float, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} values must be numeric") from error

    missing_reasons = np.logical_or(
        raw_reasons.isna().to_numpy(dtype=bool, copy=False),
        raw_reasons.eq("").fillna(False).to_numpy(dtype=bool, copy=False),
    )
    observed = statuses.eq("observed").to_numpy(dtype=bool, copy=False)
    not_estimable = statuses.eq("not_estimable").to_numpy(dtype=bool, copy=False)
    structural_zero = statuses.eq("structural_zero").to_numpy(
        dtype=bool, copy=False
    )

    if bool(np.any(observed & (missing_values | ~missing_reasons))):
        raise ValueError(f"{name} observed rows require value and no reason")
    if bool(np.any(not_estimable & (~missing_values | missing_reasons))):
        raise ValueError(f"{name} not-estimable rows require NA and a reason")
    if bool(
        np.any(
            structural_zero
            & ((numeric_values != 0.0) | missing_reasons)
        )
    ):
        raise ValueError(f"{name} structural-zero rows require zero and a reason")

    finite_values = numeric_values[~missing_values]
    if not bool(np.isfinite(finite_values).all()):
        raise ValueError(f"{name} values must be finite")
    if name != "differential_effect" and bool(
        np.any((finite_values < 0.0) | (finite_values > 1.0))
    ):
        raise ValueError(f"{name} values must lie in [0, 1]")


def _view_statuses(
    artifacts: CrossFitArtifacts,
    tables: dict[str, pd.DataFrame],
) -> tuple[tuple[str, str, str | None], ...]:
    rows: list[tuple[str, str, str | None]] = []
    directional_names = _directional_contrast_names(artifacts)
    has_directional_pair = bool(directional_names)
    for name in SEMANTIC_SCORE_OUTPUTS:
        table = tables[name]
        if not table.empty:
            reason = (
                _DIRECTIONAL_EXCLUDED_REASON
                if has_directional_pair
                and name in {"integrated_lr_score", "differential_effect"}
                else None
            )
            rows.append((name, "produced", reason))
        elif has_directional_pair and name in {
            "integrated_lr_score",
            "differential_effect",
        }:
            rows.append((name, "not_produced", _DIRECTIONAL_DEDICATED_REASON))
        elif name in {"integrated_lr_score", "differential_effect"} and (
            artifacts.spec.penalty_tuning_spec is None
        ):
            rows.append((name, "not_produced", _NO_COMMON_SCORING_REASON))
        else:
            rows.append((name, "not_estimable", f"no_{name}_rows"))
    return tuple(rows)


@dataclass(frozen=True, slots=True, init=False)
class SemanticScoreCollection:
    """Four independent, lineage-bound descriptive cross-fit score views."""

    collection_id: str
    crossfit_id: str
    crossfit_spec_id: str
    repeat_id: str
    table_digests: tuple[tuple[str, str], ...]
    table_row_counts: tuple[tuple[str, int], ...]
    view_statuses: tuple[tuple[str, str, str | None], ...]
    formal_inference_allowed: bool
    _availability_score: pd.DataFrame = field(repr=False)
    _receiver_program_score: pd.DataFrame = field(repr=False)
    _integrated_lr_score: pd.DataFrame = field(repr=False)
    _differential_effect: pd.DataFrame = field(repr=False)
    _source_artifacts: CrossFitArtifacts = field(repr=False)
    _producer_marker: str = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "SemanticScoreCollection is producer-owned; use "
            "build_crossfit_semantic_scores()"
        )

    @classmethod
    def _from_tables(
        cls,
        artifacts: CrossFitArtifacts,
        tables: dict[str, pd.DataFrame],
    ) -> SemanticScoreCollection:
        owned_tables = {
            name: tables[name].copy(deep=True) for name in SEMANTIC_SCORE_OUTPUTS
        }
        for name, table in owned_tables.items():
            _validate_table(
                name,
                table,
                crossfit_id=artifacts.crossfit_id,
                crossfit_spec_id=artifacts.spec.spec_id,
                repeat_id=artifacts.spec.repeat_id,
            )
        digests = tuple(
            (name, _table_digest(name, owned_tables[name]))
            for name in SEMANTIC_SCORE_OUTPUTS
        )
        row_counts = tuple(
            (name, len(owned_tables[name])) for name in SEMANTIC_SCORE_OUTPUTS
        )
        self = object.__new__(cls)
        values: dict[str, object] = {
            "crossfit_id": artifacts.crossfit_id,
            "crossfit_spec_id": artifacts.spec.spec_id,
            "repeat_id": artifacts.spec.repeat_id,
            "table_digests": digests,
            "table_row_counts": row_counts,
            "view_statuses": _view_statuses(artifacts, owned_tables),
            "formal_inference_allowed": False,
            "_availability_score": owned_tables["availability_score"],
            "_receiver_program_score": owned_tables["receiver_program_score"],
            "_integrated_lr_score": owned_tables["integrated_lr_score"],
            "_differential_effect": owned_tables["differential_effect"],
            "_source_artifacts": artifacts,
            "_producer_marker": _COLLECTION_MARKER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "collection_id",
            stable_id(
                "crossfit_semantic_score_collection",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        record_validation(self)
        return self

    def _tables(self) -> dict[str, pd.DataFrame]:
        return {
            "availability_score": self._availability_score,
            "receiver_program_score": self._receiver_program_score,
            "integrated_lr_score": self._integrated_lr_score,
            "differential_effect": self._differential_effect,
        }

    def _identity_payload(self) -> dict[str, object]:
        return {
            "crossfit_id": self.crossfit_id,
            "crossfit_spec_id": self.crossfit_spec_id,
            "formal_inference_allowed": self.formal_inference_allowed,
            "repeat_id": self.repeat_id,
            "table_digests": [list(item) for item in self.table_digests],
            "table_row_counts": [list(item) for item in self.table_row_counts],
            "view_statuses": [list(item) for item in self.view_statuses],
        }

    def _require_intact(self) -> None:
        if validation_is_cached(self):
            return
        try:
            self._source_artifacts._require_intact()
            tables = self._tables()
            for name, table in tables.items():
                _validate_table(
                    name,
                    table,
                    crossfit_id=self.crossfit_id,
                    crossfit_spec_id=self.crossfit_spec_id,
                    repeat_id=self.repeat_id,
                )
            observed_digests = tuple(
                (name, _table_digest(name, tables[name]))
                for name in SEMANTIC_SCORE_OUTPUTS
            )
            observed_counts = tuple(
                (name, len(tables[name])) for name in SEMANTIC_SCORE_OUTPUTS
            )
            expected_statuses = _view_statuses(self._source_artifacts, tables)
            valid = (
                self._producer_marker == _COLLECTION_MARKER
                and self.crossfit_id == self._source_artifacts.crossfit_id
                and self.crossfit_spec_id == self._source_artifacts.spec.spec_id
                and self.repeat_id == self._source_artifacts.spec.repeat_id
                and self.table_digests == observed_digests
                and self.table_row_counts == observed_counts
                and self.view_statuses == expected_statuses
                and all(
                    name in SEMANTIC_SCORE_OUTPUTS
                    and status in _VIEW_STATUSES
                    and (
                        (status == "produced" and reason is None)
                        or (
                            status == "produced"
                            and reason == _DIRECTIONAL_EXCLUDED_REASON
                        )
                        or (status != "produced" and reason is not None)
                    )
                    for name, status, reason in self.view_statuses
                )
                and self.formal_inference_allowed is False
                and stable_id(
                    "crossfit_semantic_score_collection",
                    self._identity_payload(),
                    schema_version=_SCHEMA_VERSION,
                )
                == self.collection_id
            )
        except (
            AttributeError,
            ContractError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ContractError(
                "Semantic score collection failed integrity validation",
                code="semantic_score_collection_integrity_violation",
                field="collection_id",
                remediation="Rebuild it from intact in-memory CrossFitArtifacts",
            ) from error
        if not valid:
            raise ContractError(
                "Semantic score collection failed integrity validation",
                code="semantic_score_collection_integrity_violation",
                field="collection_id",
                remediation="Rebuild it from intact in-memory CrossFitArtifacts",
            )
        record_validation(self)

    def table(self, semantic_output: str) -> pd.DataFrame:
        """Return one named semantic view as a defensive copy."""

        if semantic_output not in SEMANTIC_SCORE_OUTPUTS:
            raise ValueError(
                "semantic_output must be availability_score, receiver_program_score, "
                "integrated_lr_score, or differential_effect"
            )
        self._require_intact()
        return self._tables()[semantic_output].copy(deep=True)

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return all semantic views after one integrity validation pass."""

        self._require_intact()
        return {name: table.copy(deep=True) for name, table in self._tables().items()}

    def snapshot(self) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
        """Return defensive table copies and their manifest after one validation."""

        self._require_intact()
        return (
            {name: table.copy(deep=True) for name, table in self._tables().items()},
            self._manifest_payload(),
        )

    @property
    def availability_score(self) -> pd.DataFrame:
        """Return interaction-level held-out availability by mode."""

        return self.table("availability_score")

    @property
    def receiver_program_score(self) -> pd.DataFrame:
        """Return source-, receptor-, and sender-agnostic receiver programs."""

        return self.table("receiver_program_score")

    @property
    def integrated_lr_score(self) -> pd.DataFrame:
        """Return family-allocated LR evidence before sender allocation."""

        return self.table("integrated_lr_score")

    @property
    def differential_effect(self) -> pd.DataFrame:
        """Return subject-family effects under each common scoring functional."""

        return self.table("differential_effect")

    @property
    def view_manifest(self) -> pd.DataFrame:
        """Return status, grain, lineage digest, and size for all four views."""

        self._require_intact()
        digest_by_name = dict(self.table_digests)
        count_by_name = dict(self.table_row_counts)
        return pd.DataFrame(
            [
                {
                    "semantic_output": name,
                    "grain": _GRAINS[name],
                    "status": status,
                    "reason_code": reason,
                    "row_count": count_by_name[name],
                    "table_digest": digest_by_name[name],
                    "dedicated_view": (
                        name in {"integrated_lr_score", "differential_effect"}
                        and bool(self._source_artifacts.spec.directional_pairs)
                    ),
                    "formal_inference_allowed": False,
                }
                for name, status, reason in self.view_statuses
            ]
        )

    def _manifest_payload(self) -> dict[str, object]:
        digest_by_name = dict(self.table_digests)
        count_by_name = dict(self.table_row_counts)
        return {
            "collection_id": self.collection_id,
            **self._identity_payload(),
            "views": [
                {
                    "semantic_output": name,
                    "grain": _GRAINS[name],
                    "status": status,
                    "reason_code": reason,
                    "row_count": count_by_name[name],
                    "table_digest": digest_by_name[name],
                    "dedicated_view": (
                        name in {"integrated_lr_score", "differential_effect"}
                        and bool(self._source_artifacts.spec.directional_pairs)
                    ),
                    "formal_inference_allowed": False,
                }
                for name, status, reason in self.view_statuses
            ],
            "excluded_output_kinds": [
                "p_value",
                "q_value",
                "posterior_probability",
                "communication_probability",
            ],
        }

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return self._manifest_payload()


def build_crossfit_semantic_scores(
    artifacts: CrossFitArtifacts,
) -> SemanticScoreCollection:
    """Build four independent descriptive views without refitting or inference."""

    if type(artifacts) is not CrossFitArtifacts:
        raise TypeError("artifacts must be producer-owned CrossFitArtifacts")
    artifacts._require_intact()
    tables = {
        "availability_score": _availability_table(artifacts),
        "receiver_program_score": _receiver_program_table(artifacts),
        "integrated_lr_score": _integrated_lr_table(artifacts),
        "differential_effect": _differential_effect_table(artifacts),
    }
    return SemanticScoreCollection._from_tables(artifacts, tables)


__all__ = [
    "SEMANTIC_AVAILABILITY_COLUMNS",
    "SEMANTIC_DIFFERENTIAL_EFFECT_COLUMNS",
    "SEMANTIC_INTEGRATED_LR_COLUMNS",
    "SEMANTIC_RECEIVER_PROGRAM_COLUMNS",
    "SEMANTIC_SCORE_OUTPUTS",
    "SemanticScoreCollection",
    "build_crossfit_semantic_scores",
]
