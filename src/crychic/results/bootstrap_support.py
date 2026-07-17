"""Versioned persistence contract for authenticated bootstrap support."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from crychic.core import canonical_json, stable_id

from ._schema import (
    RESULT_SCHEMA_VERSION,
    ResultExtensionContract,
    load_schema_document,
)
from .errors import ResultValidationError

BOOTSTRAP_SUPPORT_EXTENSION_NAME = "bootstrap_support"
BOOTSTRAP_SUPPORT_EXTENSION_VERSION = "1.0.0"
BOOTSTRAP_SUPPORT_REGISTRY_FILENAME = "bootstrap_support_registry.json"
BOOTSTRAP_SUPPORT_REGISTRY_SCHEMA = "bootstrap_support.schema.json"
SPECIFICITY_SUPPORT_TABLE = "specificity_support"
SELECTION_FREQUENCY_TABLE = "selection_frequency"
_PRIMARY_ENDPOINT = "driver_family_receiver_context_omnibus_v1"
_SECONDARY_ENDPOINT = "family_common_integrated_lr_context_effect_v1"
_REGISTRY_KIND = "frozen_bootstrap_support_lineage_v1"
_REGISTRY_KEYS = {
    "extension_schema_version",
    "result_schema_version",
    "registry_kind",
    "hypothesis_universe",
    "resampling_lineage",
    "bootstrap_plan_ids",
    "specificity_sources",
    "selection_sources",
    "registry_id",
}
_UNIVERSE_KEYS = {
    "universe_id",
    "universe_name",
    "declaration_ids",
    "hypothesis_ids",
    "selection_hypothesis_ids",
    "specificity_hypothesis_ids",
    "declarations",
}
_DECLARATION_KEYS = {
    "declaration_id",
    "hypothesis_id",
    "hypothesis_key",
    "endpoint",
    "contrast_name",
    "receiver",
    "family_id",
    "mode",
    "role",
    "multiplicity_family",
    "parent_key",
    "filter_stage",
    "prefilter_policy",
    "prefilter_status",
    "filter_reason_code",
}
_RESAMPLING_KEYS = {
    "resampling_result_id",
    "exchangeability_id",
    "source_input_identity_id",
    "source_input_digest",
    "source_snapshot_id",
    "config_digest",
    "crossfit_spec_id",
    "resource_bundle_content_id",
    "target_prior_content_id",
    "root_seed_lineage",
}
_SPECIFICITY_SOURCE_KEYS = {
    "result_id",
    "hypothesis_id",
    "target_id",
    "score_target_id",
    "bootstrap_source_binding_id",
    "bootstrap_plan_ids",
    "workflow_record_ids",
    "effect_record_ids",
}
_SELECTION_SOURCE_KEYS = {
    "result_id",
    "hypothesis_id",
    "target_id",
    "source_resampled_attribution_collection_id",
    "source_binding_id",
    "bootstrap_plan_ids",
    "event_ids",
    "numeric_record_ids",
}
_SPECIFICITY_OBSERVED_LINEAGE_FIELDS = (
    "minimum_effect",
    "specificity_direction",
    "target_id",
    "score_target_id",
    "effect_scale_id",
    "point_effect_result_id",
    "bootstrap_source_binding_id",
    "bootstrap_plan_set_id",
    "effect_spec_id",
    "distribution_spec_id",
    "source_distribution_id",
    "specificity_support_spec_id",
    "numeric_result_id",
)


def _error(message: str, *, code: str, field: str) -> ResultValidationError:
    return ResultValidationError(
        message,
        code=code,
        field=field,
        remediation=(
            "Rebuild the bootstrap-support extension from frozen workflow results"
        ),
    )


@dataclass(frozen=True, slots=True)
class BootstrapSupportExtensionContract:
    """Three-artifact bootstrap-support extension contract."""

    name: str
    extension_schema_version: str
    specificity: ResultExtensionContract
    selection: ResultExtensionContract
    registry_filename: str
    registry_schema_filename: str
    linked_tables: tuple[str, ...]


def _constant(properties: Mapping[str, Any], name: str, schema: str) -> Any:
    value = properties.get(name)
    if not isinstance(value, Mapping) or "const" not in value:
        raise _error(
            f"Extension schema {schema} lacks {name!r}",
            code="invalid_result_schema",
            field=name,
        )
    return value["const"]


def _table_contract(
    schema_filename: str,
    *,
    table_name: str,
    filename: str,
) -> ResultExtensionContract:
    document = load_schema_document(schema_filename)
    properties = document.get("properties")
    if not isinstance(properties, Mapping):
        raise _error(
            f"Extension schema {schema_filename} has no properties",
            code="invalid_result_schema",
            field="properties",
        )
    columns = properties.get("columns")
    column_properties = (
        columns.get("properties") if isinstance(columns, Mapping) else None
    )
    required_columns = columns.get("required") if isinstance(columns, Mapping) else None
    primary_key = _constant(properties, "primary_key", schema_filename)
    linked_tables = _constant(properties, "linked_tables", schema_filename)
    if (
        _constant(properties, "extension_schema_version", schema_filename)
        != BOOTSTRAP_SUPPORT_EXTENSION_VERSION
        or _constant(properties, "result_schema_version", schema_filename)
        != RESULT_SCHEMA_VERSION
        or _constant(properties, "table", schema_filename) != table_name
        or _constant(properties, "filename", schema_filename) != filename
        or not isinstance(primary_key, list)
        or not isinstance(linked_tables, list)
        or linked_tables != ["differential"]
        or not isinstance(column_properties, Mapping)
        or not isinstance(required_columns, list)
        or set(required_columns) != set(column_properties)
    ):
        raise _error(
            f"Extension schema {schema_filename} has incompatible metadata",
            code="invalid_result_schema",
            field="extension_schema_version",
        )
    return ResultExtensionContract(
        name=table_name,
        extension_schema_version=BOOTSTRAP_SUPPORT_EXTENSION_VERSION,
        filename=filename,
        primary_key=tuple(primary_key),
        linked_tables=tuple(linked_tables),
        columns=tuple(required_columns),
        schema_filename=schema_filename,
    )


def bootstrap_support_contract() -> BootstrapSupportExtensionContract:
    """Return the released multi-artifact extension contract."""

    return BootstrapSupportExtensionContract(
        name=BOOTSTRAP_SUPPORT_EXTENSION_NAME,
        extension_schema_version=BOOTSTRAP_SUPPORT_EXTENSION_VERSION,
        specificity=_table_contract(
            "specificity_support.schema.json",
            table_name=SPECIFICITY_SUPPORT_TABLE,
            filename="specificity_support.parquet",
        ),
        selection=_table_contract(
            "selection_frequency.schema.json",
            table_name=SELECTION_FREQUENCY_TABLE,
            filename="selection_frequency.parquet",
        ),
        registry_filename=BOOTSTRAP_SUPPORT_REGISTRY_FILENAME,
        registry_schema_filename=BOOTSTRAP_SUPPORT_REGISTRY_SCHEMA,
        linked_tables=("differential",),
    )


def _resolved_columns(schema_filename: str) -> dict[str, Mapping[str, Any]]:
    document = load_schema_document(schema_filename)
    properties = document["properties"]
    raw = properties["columns"]["properties"]
    definitions = document.get("$defs", {})
    values: dict[str, Mapping[str, Any]] = {}
    for name, contract in raw.items():
        reference = contract.get("$ref")
        if reference is None:
            values[name] = contract
            continue
        prefix = "#/$defs/"
        if not isinstance(reference, str) or not reference.startswith(prefix):
            raise _error(
                "Bootstrap-support schema has an invalid column reference",
                code="invalid_result_schema",
                field=name,
            )
        resolved = definitions.get(reference.removeprefix(prefix))
        if not isinstance(resolved, Mapping):
            raise _error(
                "Bootstrap-support schema references an unknown definition",
                code="invalid_result_schema",
                field=name,
            )
        values[name] = resolved
    return values


def _is_null(value: object) -> bool:
    try:
        return bool(pd.isna(cast(Any, value)))
    except (TypeError, ValueError):
        return False


def _validate_scalar(value: object, contract: Mapping[str, Any], *, field: str) -> None:
    nullable = contract.get("nullable")
    if _is_null(value):
        if nullable is not True:
            raise _error(
                f"Column {field!r} contains a forbidden null",
                code="invalid_bootstrap_support_table",
                field=field,
            )
        return
    dtype = contract.get("dtype")
    if dtype == "string":
        valid = isinstance(value, str) and bool(value) and value == value.strip()
    elif dtype == "boolean":
        valid = isinstance(value, (bool, np.bool_))
    elif dtype == "integer":
        valid = isinstance(value, (int, np.integer)) and not isinstance(
            value, (bool, np.bool_)
        )
    elif dtype == "float":
        valid = isinstance(
            value, (int, float, np.integer, np.floating)
        ) and not isinstance(value, (bool, np.bool_))
        valid = valid and math.isfinite(float(cast(Any, value)))
    else:
        valid = False
    if not valid:
        raise _error(
            f"Column {field!r} violates its persisted dtype",
            code="invalid_bootstrap_support_table",
            field=field,
        )
    enum = contract.get("enum")
    if isinstance(enum, Sequence) and value not in enum:
        raise _error(
            f"Column {field!r} contains an unsupported value",
            code="invalid_bootstrap_support_table",
            field=field,
        )
    numeric = float(cast(Any, value)) if dtype in {"integer", "float"} else None
    minimum = contract.get("minimum")
    maximum = contract.get("maximum")
    if numeric is not None and minimum is not None and numeric < float(minimum):
        raise _error(
            f"Column {field!r} is below its minimum",
            code="invalid_bootstrap_support_table",
            field=field,
        )
    if numeric is not None and maximum is not None and numeric > float(maximum):
        raise _error(
            f"Column {field!r} exceeds its maximum",
            code="invalid_bootstrap_support_table",
            field=field,
        )


def _validate_frame(
    frame: pd.DataFrame,
    contract: ResultExtensionContract,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{contract.name} must be a pandas DataFrame")
    if frame.empty:
        raise _error(
            f"{contract.name} cannot be empty",
            code="invalid_bootstrap_support_table",
            field=contract.name,
        )
    if tuple(frame.columns) != contract.columns:
        raise _error(
            f"{contract.name} columns do not match the released schema",
            code="invalid_bootstrap_support_table",
            field="columns",
        )
    definitions = _resolved_columns(contract.schema_filename)
    for name in contract.columns:
        for value in frame[name].tolist():
            _validate_scalar(value, definitions[name], field=name)
    if frame.loc[:, list(contract.primary_key)].isna().any(axis=None):
        raise _error(
            f"{contract.name} primary key contains null values",
            code="invalid_bootstrap_support_table",
            field="primary_key",
        )
    if frame.duplicated(list(contract.primary_key)).any():
        raise _error(
            f"{contract.name} primary key is not unique",
            code="duplicate_bootstrap_support_key",
            field="primary_key",
        )
    return frame.sort_values(list(contract.primary_key), kind="mergesort").reset_index(
        drop=True
    )


def _validate_specificity_semantics(frame: pd.DataFrame) -> None:
    for row in frame.itertuples(index=False):
        total = int(cast(Any, row.n_bootstrap_total))
        observed = int(cast(Any, row.n_bootstrap_observed))
        not_estimable = int(cast(Any, row.n_bootstrap_not_estimable))
        failed = int(cast(Any, row.n_bootstrap_failed))
        if total != observed + not_estimable + failed:
            raise _error(
                "Specificity bootstrap counts do not conserve the plan total",
                code="bootstrap_support_count_mismatch",
                field="n_bootstrap_total",
            )
        minimum_effect_missing = _is_null(row.minimum_effect)
        direction_missing = _is_null(row.specificity_direction)
        target_missing = _is_null(row.target_id)
        score_target_missing = _is_null(row.score_target_id)
        if (
            minimum_effect_missing != direction_missing
            or target_missing != score_target_missing
        ):
            raise _error(
                "Specificity specification and target lineage must be paired",
                code="bootstrap_support_lineage_mismatch",
                field="minimum_effect,specificity_direction,target_id,score_target_id",
            )
        if row.status == "observed":
            valid = (
                not _is_null(row.specificity_support)
                and _is_null(row.reason_code)
                and all(
                    not _is_null(getattr(row, field))
                    for field in _SPECIFICITY_OBSERVED_LINEAGE_FIELDS
                )
                and total >= 1_000
                and observed == total
                and not_estimable == 0
                and failed == 0
                and bool(row.specificity_support_release_allowed)
            )
        else:
            valid = (
                _is_null(row.specificity_support)
                and not _is_null(row.reason_code)
                and not bool(row.specificity_support_release_allowed)
            )
        if not valid:
            raise _error(
                "Specificity value/status/reason/release fields are inconsistent",
                code="bootstrap_support_status_mismatch",
                field="specificity_support",
            )


def _validate_selection_semantics(frame: pd.DataFrame) -> None:
    for row in frame.itertuples(index=False):
        plan_total = int(cast(Any, row.n_bootstrap_plans_total))
        plan_observed = int(cast(Any, row.n_bootstrap_plans_observed))
        plan_ne = int(cast(Any, row.n_bootstrap_plans_not_estimable))
        plan_failed = int(cast(Any, row.n_bootstrap_plans_failed))
        event_total = int(cast(Any, row.n_event_rows_total))
        event_observed = int(cast(Any, row.n_observed_event_rows))
        event_ne = int(cast(Any, row.n_not_estimable_event_rows))
        event_failed = int(cast(Any, row.n_failed_event_rows))
        selected = int(cast(Any, row.n_selected_event_rows))
        if (
            plan_total != plan_observed + plan_ne + plan_failed
            or event_total != event_observed + event_ne + event_failed
            or selected > event_observed
        ):
            raise _error(
                "Selection-frequency counts do not conserve plans/events",
                code="bootstrap_support_count_mismatch",
                field="n_bootstrap_plans_total,n_event_rows_total",
            )
        if row.status == "observed":
            valid = (
                not _is_null(row.selection_frequency)
                and _is_null(row.reason_code)
                and plan_total >= 1_000
                and plan_observed == plan_total
                and event_observed == event_total
                and plan_ne == plan_failed == event_ne == event_failed == 0
                and bool(row.selection_frequency_release_allowed)
            )
        else:
            valid = (
                _is_null(row.selection_frequency)
                and not _is_null(row.reason_code)
                and not bool(row.selection_frequency_release_allowed)
            )
        if not valid:
            raise _error(
                "Selection value/status/reason/release fields are inconsistent",
                code="bootstrap_support_status_mismatch",
                field="selection_frequency",
            )


def validate_specificity_support(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize the specificity-support aggregate table."""

    contract = bootstrap_support_contract().specificity
    validated = _validate_frame(frame, contract)
    _validate_specificity_semantics(validated)
    return validated


def validate_selection_frequency(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize the equal-bootstrap selection table."""

    contract = bootstrap_support_contract().selection
    validated = _validate_frame(frame, contract)
    _validate_selection_semantics(validated)
    return validated


def _string(value: object, *, field: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise _error(
            "Bootstrap-support registry contains an invalid identifier",
            code="invalid_bootstrap_support_registry",
            field=field,
        )
    return value


def _string_list(
    value: object,
    *,
    field: str,
    allow_empty: bool = False,
) -> list[str]:
    if (
        not isinstance(value, list)
        or (not value and not allow_empty)
        or any(_string(item, field=field) is None for item in value)
        or len(value) != len(set(value))
    ):
        raise _error(
            "Bootstrap-support registry contains an invalid ID collection",
            code="invalid_bootstrap_support_registry",
            field=field,
        )
    return value


def _source_array(
    value: object,
    *,
    field: str,
    keys: set[str],
    list_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise _error(
            "Bootstrap-support registry source collection is empty",
            code="invalid_bootstrap_support_registry",
            field=field,
        )
    sources: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != keys:
            raise _error(
                "Bootstrap-support source mapping has invalid fields",
                code="invalid_bootstrap_support_registry",
                field=field,
            )
        _string(item["result_id"], field=f"{field}.result_id")
        _string(item["hypothesis_id"], field=f"{field}.hypothesis_id")
        _string(item["target_id"], field=f"{field}.target_id", nullable=True)
        scalar_fields = keys.difference(
            {"result_id", "hypothesis_id", "target_id"}
        ).difference(list_fields)
        for name in scalar_fields:
            _string(item[name], field=f"{field}.{name}", nullable=True)
        for name in list_fields:
            _string_list(
                item[name],
                field=f"{field}.{name}",
                allow_empty=True,
            )
        sources.append(item)
    identifiers = [(item["result_id"], item["hypothesis_id"]) for item in sources]
    if len(identifiers) != len(set(identifiers)):
        raise _error(
            "Bootstrap-support source mappings are not unique",
            code="invalid_bootstrap_support_registry",
            field=field,
        )
    return sources


def validate_bootstrap_support_registry(
    registry: Mapping[str, object],
) -> dict[str, object]:
    """Validate registry structure, stable identity, and source coverage."""

    try:
        value = json.loads(canonical_json(registry))
    except (TypeError, ValueError) as error:
        raise _error(
            "Bootstrap-support registry is not canonically serializable",
            code="invalid_bootstrap_support_registry",
            field="registry",
        ) from error
    if not isinstance(value, dict) or set(value) != _REGISTRY_KEYS:
        raise _error(
            "Bootstrap-support registry has invalid top-level fields",
            code="invalid_bootstrap_support_registry",
            field="registry",
        )
    if (
        value["extension_schema_version"] != BOOTSTRAP_SUPPORT_EXTENSION_VERSION
        or value["result_schema_version"] != RESULT_SCHEMA_VERSION
        or value["registry_kind"] != _REGISTRY_KIND
    ):
        raise _error(
            "Bootstrap-support registry version or kind is invalid",
            code="invalid_bootstrap_support_registry",
            field="extension_schema_version,registry_kind",
        )
    universe = value["hypothesis_universe"]
    if not isinstance(universe, dict) or set(universe) != _UNIVERSE_KEYS:
        raise _error(
            "Bootstrap-support universe registry is invalid",
            code="invalid_bootstrap_support_registry",
            field="hypothesis_universe",
        )
    universe_id = _string(universe["universe_id"], field="universe_id")
    _string(universe["universe_name"], field="universe_name")
    declaration_ids = _string_list(
        universe["declaration_ids"], field="declaration_ids"
    )
    hypothesis_ids = _string_list(universe["hypothesis_ids"], field="hypothesis_ids")
    selection_ids = _string_list(
        universe["selection_hypothesis_ids"], field="selection_hypothesis_ids"
    )
    specificity_ids = _string_list(
        universe["specificity_hypothesis_ids"], field="specificity_hypothesis_ids"
    )
    declarations = universe["declarations"]
    if not isinstance(declarations, list) or not declarations or any(
        not isinstance(item, dict) for item in declarations
    ):
        raise _error(
            "Bootstrap-support universe declarations are invalid",
            code="invalid_bootstrap_support_registry",
            field="declarations",
        )
    for index, item in enumerate(declarations):
        if set(item) != _DECLARATION_KEYS:
            raise _error(
                "Bootstrap-support declaration has invalid fields",
                code="invalid_bootstrap_support_registry",
                field=f"declarations[{index}]",
            )
        for name in _DECLARATION_KEYS.difference(
            {"parent_key", "filter_reason_code"}
        ):
            _string(item[name], field=f"declarations[{index}].{name}")
        parent_key = _string(
            item["parent_key"],
            field=f"declarations[{index}].parent_key",
            nullable=True,
        )
        filter_reason = _string(
            item["filter_reason_code"],
            field=f"declarations[{index}].filter_reason_code",
            nullable=True,
        )
        role = item["role"]
        status = item["prefilter_status"]
        if (
            item["mode"] not in {"state", "ecosystem"}
            or role not in {"primary", "secondary"}
            or status not in {"included", "filtered_pre_fit"}
            or (role == "primary") != (parent_key is None)
            or (status == "included") != (filter_reason is None)
        ):
            raise _error(
                "Bootstrap-support declaration semantics are invalid",
                code="bootstrap_support_universe_mismatch",
                field=f"declarations[{index}]",
            )
    selection_from_declarations = [
        item["hypothesis_id"]
        for item in declarations
        if item["role"] == "primary"
        and item["endpoint"] == _PRIMARY_ENDPOINT
        and item["mode"] == "state"
    ]
    specificity_from_declarations = [
        item["hypothesis_id"]
        for item in declarations
        if item["role"] == "secondary"
        and item["endpoint"] == _SECONDARY_ENDPOINT
        and item["mode"] == "state"
    ]
    primary_keys = {
        item["hypothesis_key"] for item in declarations if item["role"] == "primary"
    }
    declarations_by_key = {item["hypothesis_key"]: item for item in declarations}
    if (
        [item.get("declaration_id") for item in declarations] != declaration_ids
        or [item.get("hypothesis_id") for item in declarations] != hypothesis_ids
        or selection_from_declarations != selection_ids
        or specificity_from_declarations != specificity_ids
        or set(selection_ids).intersection(specificity_ids)
        or len(declarations_by_key) != len(declarations)
        or any(
            item["parent_key"] not in primary_keys
            or (
                item["receiver"],
                item["family_id"],
                item["mode"],
            )
            != (
                declarations_by_key[item["parent_key"]]["receiver"],
                declarations_by_key[item["parent_key"]]["family_id"],
                declarations_by_key[item["parent_key"]]["mode"],
            )
            for item in declarations
            if item["role"] == "secondary"
        )
    ):
        raise _error(
            "Bootstrap-support universe IDs do not match declarations",
            code="bootstrap_support_universe_mismatch",
            field="hypothesis_universe",
        )
    resampling = value["resampling_lineage"]
    if not isinstance(resampling, dict) or set(resampling) != _RESAMPLING_KEYS:
        raise _error(
            "Bootstrap-support resampling lineage is invalid",
            code="invalid_bootstrap_support_registry",
            field="resampling_lineage",
        )
    for name in _RESAMPLING_KEYS.difference({"root_seed_lineage"}):
        _string(resampling[name], field=f"resampling_lineage.{name}")
    seed = resampling["root_seed_lineage"]
    if (
        not isinstance(seed, dict)
        or set(seed) != {"root_seed", "path", "derived_seed"}
        or isinstance(seed["root_seed"], bool)
        or not isinstance(seed["root_seed"], int)
        or isinstance(seed["derived_seed"], bool)
        or not isinstance(seed["derived_seed"], int)
        or not isinstance(seed["path"], list)
        or any(not isinstance(item, str) or not item for item in seed["path"])
    ):
        raise _error(
            "Bootstrap-support root seed lineage is invalid",
            code="invalid_bootstrap_support_registry",
            field="root_seed_lineage",
        )
    plan_ids = _string_list(value["bootstrap_plan_ids"], field="bootstrap_plan_ids")
    specificity_sources = _source_array(
        value["specificity_sources"],
        field="specificity_sources",
        keys=_SPECIFICITY_SOURCE_KEYS,
        list_fields=(
            "bootstrap_plan_ids",
            "workflow_record_ids",
            "effect_record_ids",
        ),
    )
    selection_sources = _source_array(
        value["selection_sources"],
        field="selection_sources",
        keys=_SELECTION_SOURCE_KEYS,
        list_fields=("bootstrap_plan_ids", "event_ids", "numeric_record_ids"),
    )
    if (
        {item["hypothesis_id"] for item in specificity_sources}
        != set(specificity_ids)
        or {item["hypothesis_id"] for item in selection_sources}
        != set(selection_ids)
        or any(
            item["bootstrap_plan_ids"] not in ([], plan_ids)
            for item in (*specificity_sources, *selection_sources)
        )
    ):
        raise _error(
            "Bootstrap-support source coverage differs from the frozen universe",
            code="bootstrap_support_source_coverage_mismatch",
            field="specificity_sources,selection_sources",
        )
    persisted_id = _string(value["registry_id"], field="registry_id")
    payload = {key: item for key, item in value.items() if key != "registry_id"}
    expected_id = stable_id(
        "bootstrap_support_registry",
        payload,
        schema_version=BOOTSTRAP_SUPPORT_EXTENSION_VERSION,
    )
    if persisted_id != expected_id or universe_id is None:
        raise _error(
            "Bootstrap-support registry identity is inconsistent",
            code="bootstrap_support_registry_identity_mismatch",
            field="registry_id",
        )
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class BootstrapSupportDocument:
    """Validated specificity, selection, and exact lineage registry."""

    specificity_support: pd.DataFrame
    selection_frequency: pd.DataFrame
    registry: Mapping[str, object]

    def __post_init__(self) -> None:
        specificity = validate_specificity_support(self.specificity_support)
        selection = validate_selection_frequency(self.selection_frequency)
        registry = validate_bootstrap_support_registry(self.registry)
        universe = registry["hypothesis_universe"]
        resampling = registry["resampling_lineage"]
        specificity_sources = registry["specificity_sources"]
        selection_sources = registry["selection_sources"]
        if (
            not isinstance(universe, dict)
            or not isinstance(resampling, dict)
            or not isinstance(specificity_sources, list)
            or not isinstance(selection_sources, list)
        ):
            raise _error(
                "Bootstrap-support registry has invalid lineage containers",
                code="bootstrap_support_registry_table_mismatch",
                field="registry",
            )
        if (
            set(specificity["hypothesis_id"])
            != {item["hypothesis_id"] for item in specificity_sources}
            or set(selection["hypothesis_id"])
            != {item["hypothesis_id"] for item in selection_sources}
            or set(specificity["result_id"])
            != {item["result_id"] for item in specificity_sources}
            or set(selection["result_id"])
            != {item["result_id"] for item in selection_sources}
            or set(specificity["hypothesis_universe_id"])
            != {universe["universe_id"]}
            or set(selection["hypothesis_universe_id"])
            != {universe["universe_id"]}
            or set(specificity["crossfit_spec_id"])
            != {resampling["crossfit_spec_id"]}
            or set(selection["crossfit_spec_id"])
            != {resampling["crossfit_spec_id"]}
        ):
            raise _error(
                "Bootstrap-support tables do not match their lineage registry",
                code="bootstrap_support_registry_table_mismatch",
                field="registry",
            )
        declarations = universe["declarations"]
        if not isinstance(declarations, list):
            raise _error(
                "Bootstrap-support hypothesis declarations must be a list",
                code="bootstrap_support_registry_table_mismatch",
                field="registry.hypothesis_universe.declarations",
            )
        declaration_by_hypothesis = {
            item["hypothesis_id"]: item for item in declarations
        }
        specificity_source_by_key = {
            (item["result_id"], item["hypothesis_id"]): item
            for item in specificity_sources
        }
        selection_source_by_key = {
            (item["result_id"], item["hypothesis_id"]): item
            for item in selection_sources
        }
        for row in specificity.itertuples(index=False):
            declaration = declaration_by_hypothesis[row.hypothesis_id]
            source = specificity_source_by_key[(row.result_id, row.hypothesis_id)]
            if (
                declaration["role"] != "secondary"
                or row.hypothesis_role != "secondary"
                or row.declaration_id != declaration["declaration_id"]
                or row.contrast != declaration["contrast_name"]
                or row.mode != declaration["mode"]
                or row.receiver != declaration["receiver"]
                or row.family_id != declaration["family_id"]
                or (None if _is_null(row.target_id) else row.target_id)
                != source["target_id"]
                or (None if _is_null(row.score_target_id) else row.score_target_id)
                != source["score_target_id"]
                or (
                    None
                    if _is_null(row.bootstrap_source_binding_id)
                    else row.bootstrap_source_binding_id
                )
                != source["bootstrap_source_binding_id"]
            ):
                raise _error(
                    "Specificity table row differs from declaration/source registry",
                    code="bootstrap_support_registry_table_mismatch",
                    field="specificity_support",
                )
        for row in selection.itertuples(index=False):
            declaration = declaration_by_hypothesis[row.hypothesis_id]
            source = selection_source_by_key[(row.result_id, row.hypothesis_id)]
            if (
                declaration["role"] != "primary"
                or row.hypothesis_role != "primary"
                or row.declaration_id != declaration["declaration_id"]
                or row.contrast != declaration["contrast_name"]
                or row.mode != declaration["mode"]
                or row.receiver != declaration["receiver"]
                or row.family_id != declaration["family_id"]
                or row.resampling_result_id != resampling["resampling_result_id"]
                or (None if _is_null(row.target_id) else row.target_id)
                != source["target_id"]
                or (
                    None
                    if _is_null(row.source_resampled_attribution_collection_id)
                    else row.source_resampled_attribution_collection_id
                )
                != source["source_resampled_attribution_collection_id"]
                or (None if _is_null(row.source_binding_id) else row.source_binding_id)
                != source["source_binding_id"]
            ):
                raise _error(
                    "Selection table row differs from declaration/source registry",
                    code="bootstrap_support_registry_table_mismatch",
                    field="selection_frequency",
                )
        object.__setattr__(self, "specificity_support", specificity.copy(deep=True))
        object.__setattr__(self, "selection_frequency", selection.copy(deep=True))
        object.__setattr__(self, "registry", registry)

    @property
    def registry_id(self) -> str:
        return str(self.registry["registry_id"])


def validate_bootstrap_support_links(
    document: BootstrapSupportDocument,
    differential: pd.DataFrame,
) -> None:
    """Require both aggregate grains to resolve to base differential keys."""

    if not isinstance(document, BootstrapSupportDocument):
        raise TypeError("document must be BootstrapSupportDocument")
    required = ("hypothesis_level", "hypothesis_id", "contrast", "mode", "view")
    if not isinstance(differential, pd.DataFrame) or not set(required).issubset(
        differential.columns
    ):
        raise _error(
            "Differential table cannot link bootstrap support",
            code="bootstrap_support_differential_link_mismatch",
            field="differential",
        )
    base = {
        tuple(row)
        for row in differential.loc[:, list(required)].itertuples(
            index=False, name=None
        )
    }
    for frame in (document.specificity_support, document.selection_frequency):
        keys = {
            tuple(row)
            for row in frame.loc[:, list(required)].itertuples(index=False, name=None)
        }
        if not keys.issubset(base):
            raise _error(
                "Bootstrap-support hypothesis keys are absent from differential",
                code="bootstrap_support_differential_link_mismatch",
                field="hypothesis_id",
            )


__all__ = [
    "BOOTSTRAP_SUPPORT_EXTENSION_NAME",
    "BOOTSTRAP_SUPPORT_EXTENSION_VERSION",
    "BOOTSTRAP_SUPPORT_REGISTRY_FILENAME",
    "BOOTSTRAP_SUPPORT_REGISTRY_SCHEMA",
    "SELECTION_FREQUENCY_TABLE",
    "SPECIFICITY_SUPPORT_TABLE",
    "BootstrapSupportDocument",
    "BootstrapSupportExtensionContract",
    "bootstrap_support_contract",
    "validate_bootstrap_support_links",
    "validate_bootstrap_support_registry",
    "validate_selection_frequency",
    "validate_specificity_support",
]
