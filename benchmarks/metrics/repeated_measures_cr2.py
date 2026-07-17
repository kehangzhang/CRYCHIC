"""Formal-ready CR2 effects for checksum-bound benchmark score tables.

This adapter freezes the core CRYCHIC repeated-measures design once and batches
all edges for a method/receiver combination through the reviewed CR2 backend.
It deliberately exposes no analytic p-values or q-values; release calibration
is a separate contract.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from crychic.core import stable_id
from crychic.design.contrasts import ContrastSpec
from crychic.design.repeated_measures import (
    RepeatedMeasuresDesignSpec,
    freeze_repeated_measures_design,
)
from crychic.response.repeated_cr2 import (
    fit_repeated_measures_cr2_receiver_effect,
)

_SCHEMA_VERSION = "1.0.0"
_FORBIDDEN_INFERENCE_FIELDS = frozenset({"p", "p_value", "q", "q_value"})


def _field_names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(values)
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in result
    ):
        raise ValueError(f"{field_name} must contain canonical non-empty names")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique names")
    return result


def _require_complete_keys(
    table: pd.DataFrame,
    keys: Sequence[str],
    *,
    table_name: str,
) -> None:
    missing = set(keys).difference(table.columns)
    if missing:
        raise ValueError(f"{table_name} is missing columns: {sorted(missing)}")
    incomplete = [key for key in keys if table[key].isna().any()]
    if incomplete:
        raise ValueError(
            f"{table_name} identity fields must not contain missing values: "
            f"{incomplete}"
        )


def _plain_identity_value(value: object, *, field_name: str) -> object:
    plain = value.item() if isinstance(value, np.generic) else value
    if isinstance(plain, float) and not math.isfinite(plain):
        raise ValueError(f"{field_name} identity values must be finite")
    if not isinstance(plain, (bool, int, float, str)):
        raise TypeError(f"{field_name} identity values must be JSON-compatible scalars")
    return plain


def _edge_feature_id(edge: dict[str, object]) -> str:
    payload = {
        "edge_fields": [
            {
                "name": name,
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "value": value,
            }
            for name, value in edge.items()
        ]
    }
    return cast(
        str,
        stable_id(
            "benchmark_repeated_measures_cr2_feature",
            payload,
            schema_version=_SCHEMA_VERSION,
        ),
    )


def repeated_measures_cr2_effects(
    table: pd.DataFrame,
    sample_design: pd.DataFrame,
    *,
    design_spec: RepeatedMeasuresDesignSpec,
    contrast: ContrastSpec,
    identity_keys: Sequence[str],
    edge_keys: Sequence[str],
    value_key: str = "value",
    status_key: str | None = None,
    observed_statuses: Sequence[str] = ("observed", "not_predicted"),
    receiver_key: str = "receiver",
) -> pd.DataFrame:
    """Fit one CR2 effect per edge while batching edges by method and receiver.

    ``sample_design`` is expected to have been checksum-validated by the
    finalizer. Rows with statuses outside ``observed_statuses`` become missing
    feature values and therefore fail closed independently of peer edges.
    """

    identities = _field_names(identity_keys, field_name="identity_keys")
    edges = _field_names(edge_keys, field_name="edge_keys")
    if not identities:
        raise ValueError("identity_keys must contain at least one field")
    if not edges:
        raise ValueError("edge_keys must contain at least one field")
    if set(identities).intersection(edges):
        raise ValueError("identity_keys and edge_keys must be disjoint")
    if receiver_key not in edges:
        raise ValueError("receiver_key must be declared in edge_keys")
    required = {*identities, *edges, design_spec.sample_key, value_key}
    if status_key is not None:
        required.add(status_key)
    _require_complete_keys(
        table,
        (*identities, *edges, design_spec.sample_key),
        table_name="CR2 benchmark score table",
    )
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"CR2 benchmark score table is missing columns: {sorted(missing)}"
        )
    unique_key = (*identities, *edges, design_spec.sample_key)
    if table.duplicated(list(unique_key)).any():
        raise ValueError("CR2 benchmark score table contains repeated edge/sample rows")

    design = freeze_repeated_measures_design(
        sample_design,
        contrast=contrast,
        spec=design_spec,
    )
    unknown_samples = set(table[design_spec.sample_key].astype(str)).difference(
        design.sample_ids
    )
    if unknown_samples:
        raise ValueError(
            "CR2 benchmark scores contain samples outside the frozen design: "
            + ",".join(sorted(unknown_samples))
        )
    allowed = frozenset(str(value) for value in observed_statuses)
    if not allowed:
        raise ValueError("observed_statuses must contain at least one status")

    rows: list[dict[str, object]] = []
    identity_grouper: str | list[str] = (
        identities[0] if len(identities) == 1 else list(identities)
    )
    for raw_identity, identity_group in table.groupby(
        identity_grouper,
        observed=True,
        sort=False,
    ):
        identity_values = (
            (raw_identity,)
            if len(identities) == 1
            else cast(tuple[object, ...], raw_identity)
        )
        identity: dict[str, object] = dict(
            zip(identities, identity_values, strict=True)
        )
        for raw_receiver, receiver_group in identity_group.groupby(
            receiver_key,
            observed=True,
            sort=False,
        ):
            receiver = str(raw_receiver)
            edge_records: list[tuple[str, dict[str, object]]] = []
            unique_edges = receiver_group.loc[:, list(edges)].drop_duplicates()
            for raw_edge in unique_edges.to_dict(orient="records"):
                edge = {
                    key: _plain_identity_value(raw_edge[key], field_name=key)
                    for key in edges
                }
                edge_records.append((_edge_feature_id(edge), edge))
            edge_records.sort(key=lambda item: item[0])
            feature_ids = tuple(feature_id for feature_id, _ in edge_records)
            if len(set(feature_ids)) != len(feature_ids):
                raise RuntimeError("CR2 benchmark edge feature IDs collided")
            feature_index = {
                tuple(edge[key] for key in edges): index
                for index, (_, edge) in enumerate(edge_records)
            }
            sample_index = {
                sample_id: index for index, sample_id in enumerate(design.sample_ids)
            }
            values: np.ndarray = np.full(
                (len(design.sample_ids), len(feature_ids)),
                np.nan,
                dtype=np.float64,
            )
            for score_record in receiver_group.to_dict(orient="records"):
                if (
                    status_key is not None
                    and str(score_record[status_key]) not in allowed
                ):
                    continue
                raw_value = pd.to_numeric(
                    pd.Series([score_record[value_key]]), errors="coerce"
                ).iloc[0]
                if pd.isna(raw_value) or not math.isfinite(float(raw_value)):
                    raise ValueError(
                        f"eligible CR2 benchmark {value_key} values must be finite"
                    )
                sample_id = str(score_record[design_spec.sample_key])
                edge_identity = tuple(
                    _plain_identity_value(score_record[key], field_name=key)
                    for key in edges
                )
                values[sample_index[sample_id], feature_index[edge_identity]] = float(
                    raw_value
                )

            fitted = fit_repeated_measures_cr2_receiver_effect(
                design,
                values,
                sample_ids=design.sample_ids,
                feature_ids=feature_ids,
                receiver=receiver,
            )
            by_feature: dict[str, dict[str, object]] = dict(edge_records)
            for raw_record in fitted.to_frame().to_dict(orient="records"):
                output_record = cast(dict[str, Any], raw_record)
                feature_id = str(output_record.pop("feature_id"))
                fitted_receiver = str(output_record.pop("receiver"))
                fitted_contrast = str(output_record.pop("contrast"))
                if fitted_receiver != receiver or fitted_contrast != contrast.name:
                    raise RuntimeError("CR2 benchmark fit lineage mismatch")
                output_record["repeated_measures_cr2_artifact_id"] = output_record.pop(
                    "artifact_id"
                )
                output_record["repeated_measures_cr2_response_input_id"] = (
                    output_record.pop("response_input_id")
                )
                output_record["repeated_measures_cr2_design_id"] = output_record.pop(
                    "design_id"
                )
                output_record["repeated_measures_cr2_feature_id"] = feature_id
                rows.append(identity | by_feature[feature_id] | output_record)

    result = pd.DataFrame.from_records(rows)
    forbidden = _FORBIDDEN_INFERENCE_FIELDS.intersection(result.columns)
    if forbidden:
        raise RuntimeError(
            "CR2 benchmark backend emitted forbidden inferential fields: "
            f"{sorted(forbidden)}"
        )
    if not result.empty and not result["formal_inference_allowed"].eq(False).all():
        raise RuntimeError("CR2 benchmark backend cannot authorize formal inference")
    return result


__all__ = ["repeated_measures_cr2_effects"]
