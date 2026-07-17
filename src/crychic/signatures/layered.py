"""Post-fit receiver, LR, and sender-LR-receiver signatures.

The builder in this module only reconciles explicit fitted component records.
It does not fit, select, or reallocate communication components.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar, cast

import pandas as pd

from crychic.core import ContractError, canonical_json, stable_id

from .contracts import DirectionAgreement

LAYERED_SIGNATURE_SCHEMA_VERSION = "0.2.0"
LAYERED_SIGNATURE_ID_SCHEMA_VERSION = "1"
SIGNED_DIRECTION_RULE_VERSION = "signed_component_agreement_v1"


class SignatureComponentLevel(StrEnum):
    """Input grain of an already-fitted signature component."""

    FAMILY = "family"
    LR = "lr"
    SENDER_LR = "sender_lr"


class SignatureFeatureKind(StrEnum):
    """Feature type represented by a signature row."""

    GENE = "gene"
    TF = "tf"
    PROGRAM = "program"


class LayeredSignatureStatus(StrEnum):
    """Estimability of a response or fitted signature component."""

    OK = "ok"
    STRUCTURAL_ZERO = "structural_zero"
    NOT_ESTIMABLE = "not_estimable"
    FAILED = "failed"
    FAMILY_HIGH_ENTROPY = "family_high_entropy"


COMPONENT_RECORD_COLUMNS = (
    "component_level",
    "source_record_id",
    "context_id",
    "context_json",
    "receiver",
    "contrast",
    "feature_id",
    "feature_kind",
    "feature_namespace",
    "family_id",
    "interaction_id",
    "sender",
    "observed",
    "predicted",
    "residual",
    "attributed_contribution",
    "response_status",
    "component_status",
    "reason_code",
    "within_family_entropy",
    "uncertainty",
    "uncertainty_kind",
    "source_artifact_id",
    "scoring_functional_id",
    "fold_id",
    "repeat_id",
    "provenance_json",
)

RECEIVER_CONTEXT_SIGNATURE_COLUMNS = (
    "signature_id",
    "context_id",
    "context_json",
    "receiver",
    "contrast",
    "feature_id",
    "feature_kind",
    "feature_namespace",
    "observed",
    "predicted",
    "residual",
    "direction_consistent_predicted",
    "observed_direction",
    "predicted_direction",
    "direction_agreement",
    "status",
    "reason_code",
    "component_coverage_status",
    "component_coverage_reason",
    "family_component_count",
    "estimable_family_component_count",
    "uncertainty_json",
    "source_record_ids_json",
    "source_artifact_ids_json",
    "scoring_functional_ids_json",
    "fold_ids_json",
    "repeat_ids_json",
    "provenance_json",
    "direction_rule_version",
    "signature_rank",
)

LR_ATTRIBUTED_SIGNATURE_COLUMNS = (
    "signature_id",
    "receiver_signature_id",
    "source_record_id",
    "context_id",
    "context_json",
    "receiver",
    "contrast",
    "feature_id",
    "feature_kind",
    "feature_namespace",
    "family_id",
    "interaction_id",
    "observed",
    "predicted",
    "residual",
    "attributed_contribution",
    "direction_consistent_attributed",
    "observed_direction",
    "attributed_direction",
    "direction_agreement",
    "response_status",
    "status",
    "reason_code",
    "within_family_entropy",
    "entropy_threshold",
    "uncertainty",
    "uncertainty_kind",
    "source_artifact_id",
    "scoring_functional_id",
    "fold_id",
    "repeat_id",
    "provenance_json",
    "direction_rule_version",
    "signature_rank",
)

SENDER_LR_RECEIVER_SIGNATURE_COLUMNS = (
    "signature_id",
    "receiver_signature_id",
    "lr_signature_id",
    "source_record_id",
    "context_id",
    "context_json",
    "sender",
    "receiver",
    "contrast",
    "feature_id",
    "feature_kind",
    "feature_namespace",
    "family_id",
    "interaction_id",
    "observed",
    "predicted",
    "residual",
    "attributed_contribution",
    "direction_consistent_attributed",
    "observed_direction",
    "attributed_direction",
    "direction_agreement",
    "response_status",
    "status",
    "reason_code",
    "within_family_entropy",
    "entropy_threshold",
    "uncertainty",
    "uncertainty_kind",
    "source_artifact_id",
    "scoring_functional_id",
    "fold_id",
    "repeat_id",
    "provenance_json",
    "direction_rule_version",
    "signature_rank",
)

_RESPONSE_STATUSES = {
    LayeredSignatureStatus.OK,
    LayeredSignatureStatus.NOT_ESTIMABLE,
    LayeredSignatureStatus.FAILED,
}
_INPUT_COMPONENT_STATUSES = {
    LayeredSignatureStatus.OK,
    LayeredSignatureStatus.STRUCTURAL_ZERO,
    LayeredSignatureStatus.NOT_ESTIMABLE,
    LayeredSignatureStatus.FAILED,
}
_ESTIMABLE_COMPONENT_STATUSES = {
    LayeredSignatureStatus.OK,
    LayeredSignatureStatus.STRUCTURAL_ZERO,
}
_UNAVAILABLE_OUTPUT_STATUSES = {
    LayeredSignatureStatus.NOT_ESTIMABLE.value,
    LayeredSignatureStatus.FAILED.value,
    LayeredSignatureStatus.FAMILY_HIGH_ENTROPY.value,
}

EnumT = TypeVar("EnumT", bound=StrEnum)
AnchorKey = tuple[str, str, str, str, str, str]
FamilyKey = tuple[str, str, str, str, str, str, str]
LRKey = tuple[str, str, str, str, str, str, str, str]


def _error(
    message: str,
    *,
    code: str,
    field: str | None = None,
    remediation: str | None = None,
) -> ContractError:
    return ContractError(
        message,
        code=code,
        field=field,
        remediation=remediation,
    )


def _missing(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float) and math.isnan(value)


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(
            f"{field} must be a non-empty string",
            code="invalid_layered_signature_record",
            field=field,
        )
    return value.strip()


def _optional_string(value: object, field: str) -> str | None:
    if _missing(value):
        return None
    return _required_string(value, field)


def _optional_float(value: object, field: str) -> float | None:
    if _missing(value):
        return None
    if isinstance(value, bool):
        raise _error(
            f"{field} must be numeric or missing",
            code="invalid_layered_signature_numeric",
            field=field,
        )
    try:
        result = float(cast(float, value))
    except (TypeError, ValueError) as exc:
        raise _error(
            f"{field} must be numeric or missing",
            code="invalid_layered_signature_numeric",
            field=field,
        ) from exc
    if not math.isfinite(result):
        raise _error(
            f"{field} must be finite when present",
            code="invalid_layered_signature_numeric",
            field=field,
        )
    return result


def _enum(value: object, enum_type: type[EnumT], field: str) -> EnumT:
    if not isinstance(value, str):
        raise _error(
            f"{field} must be a string enum value",
            code="invalid_layered_signature_enum",
            field=field,
        )
    try:
        return enum_type(value)
    except ValueError as exc:
        raise _error(
            f"Invalid {field}: {value!r}",
            code="invalid_layered_signature_enum",
            field=field,
        ) from exc


def _forbidden_name(name: str) -> bool:
    normalized = name.strip().lower()
    return (
        normalized in {"p", "q", "pvalue", "qvalue", "p_value", "q_value"}
        or "probability" in normalized
        or "posterior" in normalized
    )


def _check_json_keys(value: object, field: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _error(
                    f"{field} must use string JSON keys",
                    code="invalid_layered_signature_json",
                    field=field,
                )
            if _forbidden_name(key):
                raise _error(
                    f"{field} cannot contain inferential field {key!r}",
                    code="forbidden_layered_signature_inference",
                    field=field,
                    remediation="Keep p/q/probability/posterior fields in inference",
                )
            _check_json_keys(item, field)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _check_json_keys(item, field)


def _json_object(value: object, field: str) -> str:
    decoded: object
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise _error(
                f"{field} must be valid JSON",
                code="invalid_layered_signature_json",
                field=field,
            ) from exc
    else:
        decoded = value
    if not isinstance(decoded, Mapping):
        raise _error(
            f"{field} must encode a JSON object",
            code="invalid_layered_signature_json",
            field=field,
        )
    _check_json_keys(decoded, field)
    try:
        return canonical_json(decoded)
    except (TypeError, ValueError) as exc:
        raise _error(
            f"{field} is not canonically serializable",
            code="invalid_layered_signature_json",
            field=field,
        ) from exc


def _check_semantics(value: str, field: str) -> None:
    if _forbidden_name(value):
        raise _error(
            f"{field} cannot describe an inferential probability or p/q quantity",
            code="forbidden_layered_signature_inference",
            field=field,
        )


def _same_float(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


@dataclass(frozen=True, slots=True)
class FittedSignatureComponentRecord:
    """One fitted family, LR, or sender-LR contribution to one feature."""

    component_level: SignatureComponentLevel
    source_record_id: str
    context_id: str
    context_json: str
    receiver: str
    contrast: str
    feature_id: str
    feature_kind: SignatureFeatureKind
    feature_namespace: str
    family_id: str
    interaction_id: str | None
    sender: str | None
    observed: float | None
    predicted: float | None
    residual: float | None
    attributed_contribution: float | None
    response_status: LayeredSignatureStatus
    component_status: LayeredSignatureStatus
    reason_code: str | None
    within_family_entropy: float
    uncertainty: float | None
    uncertainty_kind: str | None
    source_artifact_id: str
    scoring_functional_id: str | None
    fold_id: str | None
    repeat_id: str | None
    provenance_json: str

    def __post_init__(self) -> None:
        for field in (
            "source_record_id",
            "context_id",
            "receiver",
            "contrast",
            "feature_id",
            "feature_namespace",
            "family_id",
            "source_artifact_id",
        ):
            object.__setattr__(
                self, field, _required_string(getattr(self, field), field)
            )
        object.__setattr__(
            self,
            "component_level",
            _enum(self.component_level, SignatureComponentLevel, "component_level"),
        )
        object.__setattr__(
            self,
            "feature_kind",
            _enum(self.feature_kind, SignatureFeatureKind, "feature_kind"),
        )
        object.__setattr__(
            self,
            "response_status",
            _enum(self.response_status, LayeredSignatureStatus, "response_status"),
        )
        object.__setattr__(
            self,
            "component_status",
            _enum(self.component_status, LayeredSignatureStatus, "component_status"),
        )
        if self.response_status not in _RESPONSE_STATUSES:
            raise _error(
                "response_status cannot be structural-zero or entropy-derived",
                code="invalid_layered_signature_status",
                field="response_status",
            )
        if self.component_status not in _INPUT_COMPONENT_STATUSES:
            raise _error(
                "component_status cannot be entropy-derived at input",
                code="invalid_layered_signature_status",
                field="component_status",
            )
        for field in (
            "interaction_id",
            "sender",
            "reason_code",
            "uncertainty_kind",
            "scoring_functional_id",
            "fold_id",
            "repeat_id",
        ):
            object.__setattr__(
                self, field, _optional_string(getattr(self, field), field)
            )
        if self.component_level is SignatureComponentLevel.FAMILY:
            if self.interaction_id is not None or self.sender is not None:
                raise _error(
                    "Family records cannot name an interaction or sender",
                    code="invalid_layered_signature_grain",
                    field="component_level",
                )
        elif self.component_level is SignatureComponentLevel.LR:
            if self.interaction_id is None or self.sender is not None:
                raise _error(
                    "LR records require interaction_id and no sender",
                    code="invalid_layered_signature_grain",
                    field="component_level",
                )
        elif self.interaction_id is None or self.sender is None:
            raise _error(
                "sender-LR records require both interaction_id and sender",
                code="invalid_layered_signature_grain",
                field="component_level",
            )
        object.__setattr__(
            self, "context_json", _json_object(self.context_json, "context_json")
        )
        object.__setattr__(
            self,
            "provenance_json",
            _json_object(self.provenance_json, "provenance_json"),
        )
        for field in (
            "observed",
            "predicted",
            "residual",
            "attributed_contribution",
            "within_family_entropy",
            "uncertainty",
        ):
            object.__setattr__(
                self, field, _optional_float(getattr(self, field), field)
            )
        entropy = self.within_family_entropy
        if entropy is None or not 0.0 <= entropy <= 1.0:
            raise _error(
                "within_family_entropy must be finite and in [0, 1]",
                code="invalid_layered_signature_entropy",
                field="within_family_entropy",
            )
        if self.uncertainty is None and self.uncertainty_kind is not None:
            raise _error(
                "uncertainty_kind requires a numeric uncertainty",
                code="invalid_layered_signature_uncertainty",
                field="uncertainty_kind",
            )
        if self.uncertainty is not None:
            if self.uncertainty < 0 or self.uncertainty_kind is None:
                raise _error(
                    "Uncertainty must be non-negative and explicitly typed",
                    code="invalid_layered_signature_uncertainty",
                    field="uncertainty",
                )
            _check_semantics(self.uncertainty_kind, "uncertainty_kind")
        if self.response_status is LayeredSignatureStatus.OK:
            if self.observed is None or self.predicted is None or self.residual is None:
                raise _error(
                    "Estimable responses require observed, predicted, and residual",
                    code="missing_layered_signature_response",
                    field="observed,predicted,residual",
                )
            if not math.isclose(
                self.observed,
                self.predicted + self.residual,
                rel_tol=1e-9,
                abs_tol=1e-10,
            ):
                raise _error(
                    "predicted plus full signed residual must reconstruct observed",
                    code="invalid_layered_signature_reconstruction",
                    field="residual",
                )
        else:
            if any(
                value is not None
                for value in (self.observed, self.predicted, self.residual)
            ):
                raise _error(
                    "Unavailable responses must preserve missing response quantities",
                    code="unavailable_layered_signature_has_value",
                    field="observed,predicted,residual",
                )
            if self.reason_code is None:
                raise _error(
                    "Unavailable responses require a reason",
                    code="missing_layered_signature_reason",
                    field="reason_code",
                )
        if self.component_status is LayeredSignatureStatus.OK:
            if self.attributed_contribution is None or self.reason_code is not None:
                raise _error(
                    "OK components require a contribution and no failure reason",
                    code="invalid_layered_signature_component",
                    field="attributed_contribution",
                )
        elif self.component_status is LayeredSignatureStatus.STRUCTURAL_ZERO:
            if self.attributed_contribution != 0.0 or self.reason_code is None:
                raise _error(
                    "Structural-zero components require zero and an explicit reason",
                    code="invalid_layered_signature_component",
                    field="attributed_contribution",
                )
        elif self.attributed_contribution is not None or self.reason_code is None:
            raise _error(
                "Unavailable components require a missing contribution and reason",
                code="invalid_layered_signature_component",
                field="attributed_contribution",
                remediation="Do not encode a not-estimable component as zero",
            )
        if (
            self.response_status is not LayeredSignatureStatus.OK
            and self.component_status
            not in {LayeredSignatureStatus.NOT_ESTIMABLE, LayeredSignatureStatus.FAILED}
        ):
            raise _error(
                "Unavailable responses cannot have an estimable component",
                code="invalid_layered_signature_status",
                field="component_status",
            )

    @property
    def anchor_key(self) -> AnchorKey:
        """Return the receiver-context-feature identity."""

        return (
            self.context_id,
            self.receiver,
            self.contrast,
            self.feature_id,
            self.feature_kind.value,
            self.feature_namespace,
        )

    @property
    def family_key(self) -> FamilyKey:
        """Return the parent family identity."""

        return (*self.anchor_key, self.family_id)

    @property
    def lr_key(self) -> LRKey:
        """Return the parent LR identity."""

        if self.interaction_id is None:
            raise RuntimeError("Family records do not have an LR key")
        return (*self.family_key, self.interaction_id)


def _record_from_mapping(row: Mapping[str, object]) -> FittedSignatureComponentRecord:
    return FittedSignatureComponentRecord(
        component_level=cast(SignatureComponentLevel, row["component_level"]),
        source_record_id=cast(str, row["source_record_id"]),
        context_id=cast(str, row["context_id"]),
        context_json=cast(str, row["context_json"]),
        receiver=cast(str, row["receiver"]),
        contrast=cast(str, row["contrast"]),
        feature_id=cast(str, row["feature_id"]),
        feature_kind=cast(SignatureFeatureKind, row["feature_kind"]),
        feature_namespace=cast(str, row["feature_namespace"]),
        family_id=cast(str, row["family_id"]),
        interaction_id=cast(str | None, row["interaction_id"]),
        sender=cast(str | None, row["sender"]),
        observed=cast(float | None, row["observed"]),
        predicted=cast(float | None, row["predicted"]),
        residual=cast(float | None, row["residual"]),
        attributed_contribution=cast(float | None, row["attributed_contribution"]),
        response_status=cast(LayeredSignatureStatus, row["response_status"]),
        component_status=cast(LayeredSignatureStatus, row["component_status"]),
        reason_code=cast(str | None, row["reason_code"]),
        within_family_entropy=cast(float, row["within_family_entropy"]),
        uncertainty=cast(float | None, row["uncertainty"]),
        uncertainty_kind=cast(str | None, row["uncertainty_kind"]),
        source_artifact_id=cast(str, row["source_artifact_id"]),
        scoring_functional_id=cast(str | None, row["scoring_functional_id"]),
        fold_id=cast(str | None, row["fold_id"]),
        repeat_id=cast(str | None, row["repeat_id"]),
        provenance_json=cast(str, row["provenance_json"]),
    )


def _coerce_records(
    records: pd.DataFrame | Sequence[FittedSignatureComponentRecord],
    expected_level: SignatureComponentLevel,
) -> list[FittedSignatureComponentRecord]:
    if isinstance(records, pd.DataFrame):
        columns = set(map(str, records.columns))
        forbidden = sorted(column for column in columns if _forbidden_name(column))
        if forbidden:
            raise _error(
                "Fitted signature records cannot contain inferential columns: "
                + repr(forbidden),
                code="forbidden_layered_signature_inference",
                field="columns",
            )
        missing = set(COMPONENT_RECORD_COLUMNS).difference(columns)
        unknown = columns.difference(COMPONENT_RECORD_COLUMNS)
        if missing or unknown:
            raise _error(
                "Fitted signature records must use the explicit component schema",
                code="invalid_layered_signature_schema",
                field="columns",
                remediation=f"missing={sorted(missing)!r}; unknown={sorted(unknown)!r}",
            )
        rows = cast(list[dict[str, object]], records.to_dict(orient="records"))
        result = [_record_from_mapping(row) for row in rows]
    else:
        result = list(records)
        if not all(
            isinstance(record, FittedSignatureComponentRecord) for record in result
        ):
            raise TypeError("All component records must be typed fitted records")
    if any(record.component_level is not expected_level for record in result):
        raise _error(
            f"Expected only {expected_level.value} component records",
            code="invalid_layered_signature_grain",
            field="component_level",
        )
    return result


def _validate_anchor(
    left: FittedSignatureComponentRecord, right: FittedSignatureComponentRecord
) -> None:
    if (
        left.context_json != right.context_json
        or left.response_status is not right.response_status
        or not _same_float(left.observed, right.observed)
        or not _same_float(left.predicted, right.predicted)
        or not _same_float(left.residual, right.residual)
    ):
        raise _error(
            "Component records disagree on their receiver response anchor",
            code="inconsistent_layered_signature_anchor",
            field="observed,predicted,residual",
        )


def _conserves(
    parent: FittedSignatureComponentRecord,
    children: Sequence[FittedSignatureComponentRecord],
    *,
    relation: str,
) -> None:
    if not children or parent.component_status not in _ESTIMABLE_COMPONENT_STATUSES:
        return
    if not all(
        child.component_status in _ESTIMABLE_COMPONENT_STATUSES for child in children
    ):
        return
    parent_value = cast(float, parent.attributed_contribution)
    child_total = sum(cast(float, child.attributed_contribution) for child in children)
    if not math.isclose(parent_value, child_total, rel_tol=1e-9, abs_tol=1e-10):
        raise _error(
            f"{relation} child contributions do not conserve the fitted parent",
            code="invalid_layered_signature_conservation",
            field="attributed_contribution",
        )


def _validate_child_status(
    parent: FittedSignatureComponentRecord,
    child: FittedSignatureComponentRecord,
) -> None:
    if (
        parent.component_status not in _ESTIMABLE_COMPONENT_STATUSES
        and child.component_status in _ESTIMABLE_COMPONENT_STATUSES
    ):
        raise _error(
            "An unavailable fitted parent cannot have an estimable child component",
            code="invalid_layered_signature_status",
            field="component_status",
        )


def _direction(value: float | None, tolerance: float) -> str:
    if value is None:
        return DirectionAgreement.NOT_AVAILABLE.value
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "zero"


def _agreement(
    observed: float | None, attributed: float | None, tolerance: float
) -> str:
    if observed is None or attributed is None:
        return DirectionAgreement.NOT_AVAILABLE.value
    if abs(attributed) <= tolerance:
        return DirectionAgreement.NO_PREDICTION.value
    if (observed > tolerance and attributed > tolerance) or (
        observed < -tolerance and attributed < -tolerance
    ):
        return DirectionAgreement.AGREES.value
    return DirectionAgreement.DISCORDANT.value


def _consistent(
    observed: float | None, attributed: float | None, tolerance: float
) -> float | None:
    agreement = _agreement(observed, attributed, tolerance)
    if agreement == DirectionAgreement.NOT_AVAILABLE.value:
        return None
    if agreement == DirectionAgreement.AGREES.value:
        return attributed
    return 0.0


def _stable_id(kind: str, components: object) -> str:
    return stable_id(
        kind,
        components,
        schema_version=LAYERED_SIGNATURE_ID_SCHEMA_VERSION,
    )


def _json_list(values: Sequence[object]) -> str:
    return canonical_json(list(values))


def _rank(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_columns: Sequence[str],
) -> None:
    ranks: dict[int, int] = {}
    eligible = (
        frame["status"].isin(
            [
                LayeredSignatureStatus.OK.value,
                LayeredSignatureStatus.STRUCTURAL_ZERO.value,
            ]
        )
        & frame[value_column].notna()
    )
    selected = frame.loc[eligible]
    for _, group in selected.groupby(list(group_columns), sort=True, dropna=False):
        ordered = sorted(
            group.index,
            key=lambda index: (
                -abs(float(cast(float, frame.at[index, value_column]))),
                str(frame.at[index, "feature_id"]),
                str(frame.at[index, "signature_id"]),
            ),
        )
        ranks.update({index: rank for rank, index in enumerate(ordered, start=1)})
    frame["signature_rank"] = pd.array(
        [ranks.get(index, pd.NA) for index in frame.index], dtype="Int64"
    )


def _receiver_row(
    anchor: AnchorKey,
    records: Sequence[FittedSignatureComponentRecord],
    *,
    tolerance: float,
) -> dict[str, object]:
    first = records[0]
    source_ids = sorted(record.source_record_id for record in records)
    signature_id = _stable_id(
        "receiver_context_signature",
        {"anchor": list(anchor), "source_record_ids": source_ids},
    )
    estimable_count = sum(
        record.component_status in _ESTIMABLE_COMPONENT_STATUSES for record in records
    )
    complete = estimable_count == len(records)
    uncertainty_bundle = [
        {
            "family_id": record.family_id,
            "source_record_id": record.source_record_id,
            "uncertainty": record.uncertainty,
            "uncertainty_kind": record.uncertainty_kind,
        }
        for record in sorted(records, key=lambda item: item.family_id)
    ]
    provenance_bundle = [
        {
            "family_id": record.family_id,
            "source_record_id": record.source_record_id,
            "provenance": json.loads(record.provenance_json),
        }
        for record in sorted(records, key=lambda item: item.family_id)
    ]
    reason = (
        first.reason_code
        if first.response_status is not LayeredSignatureStatus.OK
        else None
    )
    return {
        "signature_id": signature_id,
        "context_id": first.context_id,
        "context_json": first.context_json,
        "receiver": first.receiver,
        "contrast": first.contrast,
        "feature_id": first.feature_id,
        "feature_kind": first.feature_kind.value,
        "feature_namespace": first.feature_namespace,
        "observed": first.observed,
        "predicted": first.predicted,
        "residual": first.residual,
        "direction_consistent_predicted": _consistent(
            first.observed, first.predicted, tolerance
        ),
        "observed_direction": _direction(first.observed, tolerance),
        "predicted_direction": _direction(first.predicted, tolerance),
        "direction_agreement": _agreement(first.observed, first.predicted, tolerance),
        "status": first.response_status.value,
        "reason_code": reason,
        "component_coverage_status": "complete" if complete else "incomplete",
        "component_coverage_reason": (
            None if complete else "one_or_more_family_components_not_estimable"
        ),
        "family_component_count": len(records),
        "estimable_family_component_count": estimable_count,
        "uncertainty_json": _json_list(uncertainty_bundle),
        "source_record_ids_json": _json_list(source_ids),
        "source_artifact_ids_json": _json_list(
            sorted({record.source_artifact_id for record in records})
        ),
        "scoring_functional_ids_json": _json_list(
            sorted(
                {
                    record.scoring_functional_id
                    for record in records
                    if record.scoring_functional_id is not None
                }
            )
        ),
        "fold_ids_json": _json_list(
            sorted({record.fold_id for record in records if record.fold_id is not None})
        ),
        "repeat_ids_json": _json_list(
            sorted(
                {record.repeat_id for record in records if record.repeat_id is not None}
            )
        ),
        "provenance_json": _json_list(provenance_bundle),
        "direction_rule_version": SIGNED_DIRECTION_RULE_VERSION,
        "signature_rank": pd.NA,
    }


def _resolved_component(
    record: FittedSignatureComponentRecord,
    entropy_threshold: float,
) -> tuple[LayeredSignatureStatus, float | None, str | None]:
    if record.within_family_entropy >= entropy_threshold:
        return (
            LayeredSignatureStatus.FAMILY_HIGH_ENTROPY,
            None,
            "within_family_entropy_at_or_above_threshold",
        )
    return record.component_status, record.attributed_contribution, record.reason_code


def _lr_row(
    record: FittedSignatureComponentRecord,
    *,
    receiver_signature_id: str,
    entropy_threshold: float,
    tolerance: float,
) -> dict[str, object]:
    status, contribution, reason = _resolved_component(record, entropy_threshold)
    signature_id = _stable_id(
        "lr_attributed_signature",
        {
            "family_id": record.family_id,
            "interaction_id": record.interaction_id,
            "receiver_signature_id": receiver_signature_id,
            "source_record_id": record.source_record_id,
        },
    )
    return {
        "signature_id": signature_id,
        "receiver_signature_id": receiver_signature_id,
        "source_record_id": record.source_record_id,
        "context_id": record.context_id,
        "context_json": record.context_json,
        "receiver": record.receiver,
        "contrast": record.contrast,
        "feature_id": record.feature_id,
        "feature_kind": record.feature_kind.value,
        "feature_namespace": record.feature_namespace,
        "family_id": record.family_id,
        "interaction_id": record.interaction_id,
        "observed": record.observed,
        "predicted": record.predicted,
        "residual": record.residual,
        "attributed_contribution": contribution,
        "direction_consistent_attributed": _consistent(
            record.observed, contribution, tolerance
        ),
        "observed_direction": _direction(record.observed, tolerance),
        "attributed_direction": _direction(contribution, tolerance),
        "direction_agreement": _agreement(record.observed, contribution, tolerance),
        "response_status": record.response_status.value,
        "status": status.value,
        "reason_code": reason,
        "within_family_entropy": record.within_family_entropy,
        "entropy_threshold": entropy_threshold,
        "uncertainty": record.uncertainty,
        "uncertainty_kind": record.uncertainty_kind,
        "source_artifact_id": record.source_artifact_id,
        "scoring_functional_id": record.scoring_functional_id,
        "fold_id": record.fold_id,
        "repeat_id": record.repeat_id,
        "provenance_json": record.provenance_json,
        "direction_rule_version": SIGNED_DIRECTION_RULE_VERSION,
        "signature_rank": pd.NA,
    }


def _sender_row(
    record: FittedSignatureComponentRecord,
    *,
    receiver_signature_id: str,
    lr_signature_id: str,
    entropy_threshold: float,
    tolerance: float,
) -> dict[str, object]:
    status, contribution, reason = _resolved_component(record, entropy_threshold)
    signature_id = _stable_id(
        "sender_lr_receiver_signature",
        {
            "lr_signature_id": lr_signature_id,
            "sender": record.sender,
            "source_record_id": record.source_record_id,
        },
    )
    return {
        "signature_id": signature_id,
        "receiver_signature_id": receiver_signature_id,
        "lr_signature_id": lr_signature_id,
        "source_record_id": record.source_record_id,
        "context_id": record.context_id,
        "context_json": record.context_json,
        "sender": record.sender,
        "receiver": record.receiver,
        "contrast": record.contrast,
        "feature_id": record.feature_id,
        "feature_kind": record.feature_kind.value,
        "feature_namespace": record.feature_namespace,
        "family_id": record.family_id,
        "interaction_id": record.interaction_id,
        "observed": record.observed,
        "predicted": record.predicted,
        "residual": record.residual,
        "attributed_contribution": contribution,
        "direction_consistent_attributed": _consistent(
            record.observed, contribution, tolerance
        ),
        "observed_direction": _direction(record.observed, tolerance),
        "attributed_direction": _direction(contribution, tolerance),
        "direction_agreement": _agreement(record.observed, contribution, tolerance),
        "response_status": record.response_status.value,
        "status": status.value,
        "reason_code": reason,
        "within_family_entropy": record.within_family_entropy,
        "entropy_threshold": entropy_threshold,
        "uncertainty": record.uncertainty,
        "uncertainty_kind": record.uncertainty_kind,
        "source_artifact_id": record.source_artifact_id,
        "scoring_functional_id": record.scoring_functional_id,
        "fold_id": record.fold_id,
        "repeat_id": record.repeat_id,
        "provenance_json": record.provenance_json,
        "direction_rule_version": SIGNED_DIRECTION_RULE_VERSION,
        "signature_rank": pd.NA,
    }


def _validate_output_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    name: str,
    *,
    contribution_column: str,
) -> pd.DataFrame:
    result = frame.copy(deep=True)
    if tuple(result.columns) != columns:
        raise _error(
            f"{name} columns do not match the released layered signature schema",
            code="invalid_layered_signature_schema",
            field=name,
        )
    forbidden = [column for column in result.columns if _forbidden_name(str(column))]
    if forbidden:
        raise _error(
            f"{name} cannot contain inferential columns",
            code="forbidden_layered_signature_inference",
            field=name,
        )
    if result["signature_id"].duplicated().any():
        raise _error(
            f"{name} signature IDs must be unique",
            code="duplicate_layered_signature_id",
            field="signature_id",
        )
    unavailable = result["status"].isin(_UNAVAILABLE_OUTPUT_STATUSES)
    if result.loc[unavailable, contribution_column].notna().any():
        raise _error(
            "Unavailable layered signatures cannot expose fitted contributions",
            code="unavailable_layered_signature_has_value",
            field=contribution_column,
        )
    if result.loc[unavailable, "signature_rank"].notna().any():
        raise _error(
            "Unavailable layered signatures cannot be ranked",
            code="unavailable_layered_signature_ranked",
            field="signature_rank",
        )
    return result


@dataclass(frozen=True, slots=True)
class LayeredSignatureTable:
    """Three Parquet-ready post-fit signature grains."""

    receiver_context: pd.DataFrame
    lr_attributed: pd.DataFrame
    sender_lr_receiver: pd.DataFrame
    entropy_threshold: float
    direction_rule_version: str = SIGNED_DIRECTION_RULE_VERSION
    schema_version: str = LAYERED_SIGNATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.entropy_threshold)
            or not 0 <= self.entropy_threshold <= 1
        ):
            raise _error(
                "entropy_threshold must lie in [0, 1]",
                code="invalid_layered_signature_entropy",
                field="entropy_threshold",
            )
        receiver = _validate_output_frame(
            self.receiver_context,
            RECEIVER_CONTEXT_SIGNATURE_COLUMNS,
            "receiver_context",
            contribution_column="predicted",
        )
        lr = _validate_output_frame(
            self.lr_attributed,
            LR_ATTRIBUTED_SIGNATURE_COLUMNS,
            "lr_attributed",
            contribution_column="attributed_contribution",
        )
        sender = _validate_output_frame(
            self.sender_lr_receiver,
            SENDER_LR_RECEIVER_SIGNATURE_COLUMNS,
            "sender_lr_receiver",
            contribution_column="attributed_contribution",
        )
        for row in receiver.loc[
            receiver["status"] == LayeredSignatureStatus.OK.value
        ].itertuples(index=False):
            if not math.isclose(
                cast(float, row.observed),
                cast(float, row.predicted) + cast(float, row.residual),
                rel_tol=1e-9,
                abs_tol=1e-10,
            ):
                raise _error(
                    "Receiver signature does not preserve the full signed "
                    "residual identity",
                    code="invalid_layered_signature_reconstruction",
                    field="residual",
                )
        object.__setattr__(self, "receiver_context", receiver)
        object.__setattr__(self, "lr_attributed", lr)
        object.__setattr__(self, "sender_lr_receiver", sender)

    @property
    def inference_eligible(self) -> bool:
        """Layered signatures are descriptive and never formal inference."""

        return False

    @property
    def post_fit_only(self) -> bool:
        """Layered signatures only reconcile already-fitted records."""

        return True

    def query_receiver_context(
        self, *, context_id: str | None = None, receiver: str | None = None
    ) -> pd.DataFrame:
        """Query receiver-context signatures."""

        return _query(self.receiver_context, context_id=context_id, receiver=receiver)

    def query_lr(
        self,
        *,
        context_id: str | None = None,
        receiver: str | None = None,
        interaction_id: str | None = None,
    ) -> pd.DataFrame:
        """Query LR-attributed signatures."""

        return _query(
            self.lr_attributed,
            context_id=context_id,
            receiver=receiver,
            interaction_id=interaction_id,
        )

    def query_sender_lr(
        self,
        *,
        context_id: str | None = None,
        sender: str | None = None,
        receiver: str | None = None,
        interaction_id: str | None = None,
    ) -> pd.DataFrame:
        """Query complete sender-LR-receiver signatures."""

        return _query(
            self.sender_lr_receiver,
            context_id=context_id,
            sender=sender,
            receiver=receiver,
            interaction_id=interaction_id,
        )


def _query(frame: pd.DataFrame, **filters: str | None) -> pd.DataFrame:
    selected = pd.Series(True, index=frame.index, dtype=bool)
    for column, value in filters.items():
        if value is not None:
            selected &= frame[column].astype(str).eq(value)
    return frame.loc[selected].reset_index(drop=True).copy(deep=True)


def build_layered_signature_table(
    family_components: pd.DataFrame | Sequence[FittedSignatureComponentRecord],
    lr_components: pd.DataFrame | Sequence[FittedSignatureComponentRecord],
    sender_lr_components: pd.DataFrame | Sequence[FittedSignatureComponentRecord],
    *,
    entropy_threshold: float = 0.8,
    direction_tolerance: float = 1e-12,
) -> LayeredSignatureTable:
    """Reconcile fitted components into three deterministic signature grains."""

    if not math.isfinite(entropy_threshold) or not 0 <= entropy_threshold <= 1:
        raise _error(
            "entropy_threshold must be finite and in [0, 1]",
            code="invalid_layered_signature_entropy",
            field="entropy_threshold",
        )
    if not math.isfinite(direction_tolerance) or direction_tolerance < 0:
        raise _error(
            "direction_tolerance must be finite and non-negative",
            code="invalid_direction_tolerance",
            field="direction_tolerance",
        )
    families = _coerce_records(family_components, SignatureComponentLevel.FAMILY)
    lrs = _coerce_records(lr_components, SignatureComponentLevel.LR)
    senders = _coerce_records(sender_lr_components, SignatureComponentLevel.SENDER_LR)
    if not families:
        raise _error(
            "At least one fitted family component is required",
            code="empty_layered_signature_components",
            field="family_components",
        )
    all_records = [*families, *lrs, *senders]
    source_ids = [record.source_record_id for record in all_records]
    if len(source_ids) != len(set(source_ids)):
        raise _error(
            "source_record_id must be unique across all component levels",
            code="duplicate_layered_signature_source_record",
            field="source_record_id",
        )

    family_by_key: dict[FamilyKey, FittedSignatureComponentRecord] = {}
    families_by_anchor: dict[AnchorKey, list[FittedSignatureComponentRecord]] = {}
    for record in families:
        if record.family_key in family_by_key:
            raise _error(
                "Family component grain must be unique",
                code="duplicate_layered_signature_component",
                field="family_id",
            )
        family_by_key[record.family_key] = record
        families_by_anchor.setdefault(record.anchor_key, []).append(record)
    for records in families_by_anchor.values():
        first = records[0]
        for record in records[1:]:
            _validate_anchor(first, record)
        if first.response_status is LayeredSignatureStatus.OK and all(
            record.component_status in _ESTIMABLE_COMPONENT_STATUSES
            for record in records
        ):
            total = sum(
                cast(float, record.attributed_contribution) for record in records
            )
            if not math.isclose(
                cast(float, first.predicted), total, rel_tol=1e-9, abs_tol=1e-10
            ):
                raise _error(
                    "Family contributions do not reconstruct full prediction",
                    code="invalid_layered_signature_conservation",
                    field="predicted",
                )

    lr_by_key: dict[LRKey, FittedSignatureComponentRecord] = {}
    lrs_by_family: dict[FamilyKey, list[FittedSignatureComponentRecord]] = {}
    for record in lrs:
        parent = family_by_key.get(record.family_key)
        if parent is None:
            raise _error(
                "LR component has no matching fitted family parent",
                code="orphan_layered_signature_component",
                field="family_id",
            )
        _validate_anchor(parent, record)
        _validate_child_status(parent, record)
        if not _same_float(parent.within_family_entropy, record.within_family_entropy):
            raise _error(
                "LR and family records disagree on within-family entropy",
                code="inconsistent_layered_signature_entropy",
                field="within_family_entropy",
            )
        if record.lr_key in lr_by_key:
            raise _error(
                "LR component grain must be unique",
                code="duplicate_layered_signature_component",
                field="interaction_id",
            )
        lr_by_key[record.lr_key] = record
        lrs_by_family.setdefault(record.family_key, []).append(record)
    for family_key, children in lrs_by_family.items():
        _conserves(family_by_key[family_key], children, relation="LR")

    senders_by_lr: dict[LRKey, list[FittedSignatureComponentRecord]] = {}
    sender_keys: set[tuple[*LRKey, str]] = set()
    for record in senders:
        parent = lr_by_key.get(record.lr_key)
        if parent is None:
            raise _error(
                "Sender-LR component has no matching LR parent",
                code="orphan_layered_signature_component",
                field="interaction_id",
            )
        _validate_anchor(parent, record)
        _validate_child_status(parent, record)
        if not _same_float(parent.within_family_entropy, record.within_family_entropy):
            raise _error(
                "Sender-LR and family records disagree on within-family entropy",
                code="inconsistent_layered_signature_entropy",
                field="within_family_entropy",
            )
        sender_key = (*record.lr_key, cast(str, record.sender))
        if sender_key in sender_keys:
            raise _error(
                "Sender-LR component grain must be unique",
                code="duplicate_layered_signature_component",
                field="sender",
            )
        sender_keys.add(sender_key)
        senders_by_lr.setdefault(record.lr_key, []).append(record)
    for lr_key, children in senders_by_lr.items():
        _conserves(lr_by_key[lr_key], children, relation="sender")

    receiver_rows = [
        _receiver_row(
            anchor,
            sorted(records, key=lambda item: item.family_id),
            tolerance=direction_tolerance,
        )
        for anchor, records in sorted(families_by_anchor.items())
    ]
    receiver = pd.DataFrame(receiver_rows, columns=RECEIVER_CONTEXT_SIGNATURE_COLUMNS)
    receiver_ids = {
        anchor: cast(str, row["signature_id"])
        for anchor, row in zip(sorted(families_by_anchor), receiver_rows, strict=True)
    }
    lr_rows = [
        _lr_row(
            record,
            receiver_signature_id=receiver_ids[record.anchor_key],
            entropy_threshold=entropy_threshold,
            tolerance=direction_tolerance,
        )
        for record in sorted(lrs, key=lambda item: item.lr_key)
    ]
    lr = pd.DataFrame(lr_rows, columns=LR_ATTRIBUTED_SIGNATURE_COLUMNS)
    lr_ids = {
        record.lr_key: cast(str, row["signature_id"])
        for record, row in zip(
            sorted(lrs, key=lambda item: item.lr_key), lr_rows, strict=True
        )
    }
    sender_rows = [
        _sender_row(
            record,
            receiver_signature_id=receiver_ids[record.anchor_key],
            lr_signature_id=lr_ids[record.lr_key],
            entropy_threshold=entropy_threshold,
            tolerance=direction_tolerance,
        )
        for record in sorted(
            senders, key=lambda item: (*item.lr_key, cast(str, item.sender))
        )
    ]
    sender = pd.DataFrame(sender_rows, columns=SENDER_LR_RECEIVER_SIGNATURE_COLUMNS)
    _rank(
        receiver,
        value_column="predicted",
        group_columns=("context_id", "receiver", "contrast"),
    )
    _rank(
        lr,
        value_column="attributed_contribution",
        group_columns=(
            "context_id",
            "receiver",
            "contrast",
            "family_id",
            "interaction_id",
        ),
    )
    _rank(
        sender,
        value_column="attributed_contribution",
        group_columns=(
            "context_id",
            "sender",
            "receiver",
            "contrast",
            "family_id",
            "interaction_id",
        ),
    )
    return LayeredSignatureTable(
        receiver_context=receiver.sort_values("signature_id", ignore_index=True),
        lr_attributed=lr.sort_values("signature_id", ignore_index=True),
        sender_lr_receiver=sender.sort_values("signature_id", ignore_index=True),
        entropy_threshold=entropy_threshold,
    )


__all__ = [
    "COMPONENT_RECORD_COLUMNS",
    "LAYERED_SIGNATURE_SCHEMA_VERSION",
    "LR_ATTRIBUTED_SIGNATURE_COLUMNS",
    "RECEIVER_CONTEXT_SIGNATURE_COLUMNS",
    "SENDER_LR_RECEIVER_SIGNATURE_COLUMNS",
    "SIGNED_DIRECTION_RULE_VERSION",
    "FittedSignatureComponentRecord",
    "LayeredSignatureStatus",
    "LayeredSignatureTable",
    "SignatureComponentLevel",
    "SignatureFeatureKind",
    "build_layered_signature_table",
]
