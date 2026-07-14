"""Repeated full train/apply cross-fit diagnostics without formal inference."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np
import pandas as pd
from anndata import AnnData

from crychic.core import ContractError, CrychicConfig, stable_id
from crychic.resources import ResourceBundle, TargetPrior
from crychic.scoring import (
    FamilyCommonScoringApplication,
    FamilyCommonScoringFunctional,
)

from .crossfit import CrossFitArtifacts, CrossFitSpec, _run_subject_crossfit
from .training import (
    SanitizedRawInputIdentity,
    _resource_bundle_content_id,
    _sanitized_raw_input_snapshot,
    _target_prior_content_id,
)

_SCHEMA_VERSION = "1.0.0"
_PRODUCER_MARKER = "crychic.workflow.repeated_crossfit.v1"
_INFERENCE_STATUS = "not_computed_repeated_crossfit_diagnostic_only"
_OBSERVED_STATUS = "observed_descriptive_repeat_stability"
_PARTIAL_STATUS = "partially_observed_descriptive_repeat_stability"
_NO_PARTITION_STATUS = "not_estimable_insufficient_distinct_repeat_partitions"
_NO_FAMILY_STATUS = "not_estimable_family_common_stage_not_connected"
_NO_ESTIMABLE_STATUS = "not_estimable_no_complete_family_repeat_stability"

_REPEAT_REGISTRY_COLUMNS = (
    "repeat_index",
    "repeat_id",
    "crossfit_id",
    "fold_plan_id",
    "partition_id",
    "effective_n_splits",
    "n_subjects",
    "n_folds",
    "oof_coverage_complete",
    "family_common_stage_connected",
    "child_certification_status",
    "complete_pipeline_oof_certified",
)
_FAMILY_FOLD_EVENT_COLUMNS = (
    "repeat_index",
    "repeat_id",
    "fold_id",
    "contrast_name",
    "receiver",
    "family_id",
    "driver_ids",
    "n_heldout_subjects",
    "family_available",
    "family_estimable",
    "family_selected",
    "family_coefficient",
    "family_gain",
    "raw_family_gain",
    "selection_status",
    "reason_code",
)
_SUBJECT_FAMILY_REPEAT_COLUMNS = (
    "repeat_index",
    "repeat_id",
    "fold_id",
    "subject_id",
    "contrast_name",
    "receiver",
    "family_id",
    "driver_ids",
    "family_available",
    "family_estimable",
    "family_selected",
    "selection_status",
    "selection_reason_code",
    "null_loss",
    "family_loss",
    "differential_effect",
    "bounded_incremental_gain",
    "effect_status",
    "effect_reason_code",
)
_FAMILY_STABILITY_COLUMNS = (
    "contrast_name",
    "receiver",
    "family_id",
    "driver_ids",
    "n_repeats_requested",
    "n_distinct_partitions_overall",
    "n_distinct_complete_selection_partitions",
    "n_subject_repeat_opportunities",
    "n_family_available_opportunities",
    "family_availability_fraction",
    "n_family_estimable_opportunities",
    "family_estimable_fraction",
    "n_selection_observed_opportunities",
    "selection_observed_fraction",
    "n_structurally_determined_opportunities",
    "structurally_determined_fraction",
    "n_selected_opportunities",
    "diagnostic_conditional_subject_exposure_selection_fraction",
    "n_fit_opportunities",
    "n_fit_available",
    "fit_availability_fraction",
    "n_fit_estimable",
    "fit_estimable_fraction",
    "n_fit_selection_observed",
    "n_selected_fits",
    "diagnostic_conditional_fit_selection_fraction",
    "n_complete_selection_repeats",
    "complete_selection_repeat_fraction",
    "n_repeats_with_observed_selection",
    "repeat_subject_exposure_selection_fraction_median",
    "repeat_subject_exposure_selection_fraction_minimum",
    "repeat_subject_exposure_selection_fraction_maximum",
    "repeat_fit_selection_fraction_median",
    "repeat_fit_selection_fraction_minimum",
    "repeat_fit_selection_fraction_maximum",
    "n_repeats_with_observed_effect",
    "n_effect_observed_opportunities",
    "effect_observed_fraction",
    "n_structurally_determined_effect_opportunities",
    "structurally_determined_effect_fraction",
    "n_complete_effect_repeats",
    "complete_effect_repeat_fraction",
    "n_distinct_complete_effect_partitions",
    "repeat_mean_bounded_gain_median",
    "repeat_mean_bounded_gain_minimum",
    "repeat_mean_bounded_gain_maximum",
    "effect_stability_status",
    "effect_stability_reason_code",
    "status",
    "reason_code",
    "formal_inference_status",
)


def _canonical_cell(value: object) -> object:
    if isinstance(value, tuple):
        return [_canonical_cell(item) for item in value]
    if isinstance(value, list):
        return [_canonical_cell(item) for item in value]
    if isinstance(value, np.generic):
        return _canonical_cell(value.item())
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ValueError("repeated cross-fit tables cannot contain infinity")
        return {"float_hex": value.hex()}
    if isinstance(value, (str, bool, int)):
        return value
    return str(value)


def _table_digest(name: str, table: pd.DataFrame, columns: tuple[str, ...]) -> str:
    if tuple(table.columns) != columns:
        raise ValueError(f"{name} columns do not match the released contract")
    rows = [
        [_canonical_cell(value) for value in row]
        for row in table.itertuples(index=False, name=None)
    ]
    result: str = stable_id(
        name,
        {"columns": list(columns), "rows": rows},
        schema_version="1",
        digest_length=64,
    )
    return result


def _nullable_float(value: object) -> float | None:
    if value is None or value is pd.NA:
        return None
    numeric = float(cast(Any, value))
    if math.isnan(numeric):
        return None
    if not math.isfinite(numeric):
        raise ValueError("repeated cross-fit diagnostics require finite values")
    return numeric


def _nullable_bool(value: object) -> bool | None:
    if (
        value is None
        or value is pd.NA
        or (isinstance(value, (float, np.floating)) and math.isnan(float(value)))
    ):
        return None
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("family_selected must be boolean or missing")
    return bool(value)


def _nullable_string(value: object) -> str | None:
    if (
        value is None
        or value is pd.NA
        or (isinstance(value, (float, np.floating)) and math.isnan(float(value)))
    ):
        return None
    result = str(value)
    if not result:
        raise ValueError("diagnostic strings must be non-empty when present")
    return result


def _partition_id(artifacts: CrossFitArtifacts) -> str:
    partitions = sorted(
        (tuple(sorted(fold.test_subject_ids)) for fold in artifacts.fold_plan.folds),
        key=lambda values: (len(values), values),
    )
    result: str = stable_id(
        "subject_crossfit_partition",
        {"test_subject_partitions": [list(values) for values in partitions]},
        schema_version="1",
    )
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class RepeatedCrossFitSpec:
    """Frozen policy for repeated outer subject cross-fitting."""

    crossfit_spec: CrossFitSpec
    n_repeats: int
    schema_version: str = _SCHEMA_VERSION
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.crossfit_spec, CrossFitSpec):
            raise TypeError("crossfit_spec must be a CrossFitSpec")
        self.crossfit_spec._require_intact()
        if self.crossfit_spec.repeat_index != 0:
            raise ValueError("crossfit_spec must use repeat_index=0 as the base policy")
        if (
            isinstance(self.n_repeats, bool)
            or not isinstance(self.n_repeats, int)
            or self.n_repeats < 2
        ):
            raise ValueError("n_repeats must be an integer >= 2")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(
                f"RepeatedCrossFitSpec schema_version must be {_SCHEMA_VERSION}"
            )
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "repeated_subject_crossfit_spec",
                {
                    "crossfit_spec_id": self.crossfit_spec.spec_id,
                    "n_repeats": self.n_repeats,
                    "schema_version": self.schema_version,
                },
                schema_version=self.schema_version,
            ),
        )

    def _require_intact(self) -> None:
        try:
            repeated = RepeatedCrossFitSpec(
                crossfit_spec=self.crossfit_spec,
                n_repeats=self.n_repeats,
                schema_version=self.schema_version,
            )
            valid = self.spec_id == repeated.spec_id
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Repeated cross-fit specification failed integrity validation",
                code="repeated_crossfit_spec_integrity_violation",
                field="spec_id",
                remediation="Rebuild the repeat policy from an intact CrossFitSpec",
            ) from error
        if not valid:
            raise ContractError(
                "Repeated cross-fit specification failed integrity validation",
                code="repeated_crossfit_spec_integrity_violation",
                field="spec_id",
                remediation="Rebuild the repeat policy from an intact CrossFitSpec",
            )

    def to_dict(self) -> dict[str, object]:
        """Return the repeat policy and base algorithm manifest."""

        self._require_intact()
        return {
            "spec_id": self.spec_id,
            "schema_version": self.schema_version,
            "n_repeats": self.n_repeats,
            "crossfit_spec": self.crossfit_spec.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _FoldFamilyView:
    repeat_index: int
    repeat_id: str
    fold_id: str
    heldout_subject_ids: tuple[str, ...]
    contrast_name: str
    receiver: str
    functional: FamilyCommonScoringFunctional
    application: FamilyCommonScoringApplication


def _collect_fold_views(
    repeats: tuple[CrossFitArtifacts, ...],
) -> tuple[
    dict[tuple[int, str, str, str], _FoldFamilyView],
    dict[tuple[str, str], dict[str, tuple[str, ...]]],
]:
    views: dict[tuple[int, str, str, str], _FoldFamilyView] = {}
    families: dict[tuple[str, str], dict[str, tuple[str, ...]]] = defaultdict(dict)
    for artifacts in repeats:
        manifests = {fold.fold_id: fold for fold in artifacts.fold_plan.folds}
        for fold in artifacts.folds:
            manifest = manifests[fold.fold_id]
            if len(fold.family_common_functionals) != len(
                fold.family_common_applications
            ):
                raise ValueError("family-common functionals and applications misalign")
            for functional, application in zip(
                fold.family_common_functionals,
                fold.family_common_applications,
                strict=True,
            ):
                if tuple(application.heldout_subject_ids) != tuple(
                    manifest.test_subject_ids
                ):
                    raise ValueError(
                        "family-common application lacks exact heldout subject coverage"
                    )
                key = (
                    artifacts.spec.repeat_index,
                    fold.fold_id,
                    functional.contrast_name,
                    functional.receiver,
                )
                if key in views:
                    raise ValueError("duplicate repeated family-common fold key")
                views[key] = _FoldFamilyView(
                    repeat_index=artifacts.spec.repeat_index,
                    repeat_id=artifacts.spec.repeat_id,
                    fold_id=fold.fold_id,
                    heldout_subject_ids=tuple(manifest.test_subject_ids),
                    contrast_name=functional.contrast_name,
                    receiver=functional.receiver,
                    functional=functional,
                    application=application,
                )
                definitions = functional.receiver_family.family_basis.family_definitions
                member_map = {
                    family.family_id: set(family.driver_ids) for family in definitions
                }
                if set(member_map) != set(functional.family_ids):
                    raise ValueError("family-common definitions are incomplete")
                for interaction in functional.interactions:
                    if interaction.driver_id not in member_map.get(
                        interaction.family_id, set()
                    ):
                        raise ValueError(
                            "family-common interaction maps outside its family"
                        )
                family_key = (functional.contrast_name, functional.receiver)
                for family_id, driver_ids in member_map.items():
                    members = tuple(sorted(driver_ids))
                    previous = families[family_key].get(family_id)
                    if previous is not None and previous != members:
                        raise ValueError("stable family ID maps to different members")
                    families[family_key][family_id] = members
    return views, {key: dict(value) for key, value in families.items()}


def _build_repeat_registry(
    repeats: tuple[CrossFitArtifacts, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for artifacts in repeats:
        rows.append(
            {
                "repeat_index": artifacts.spec.repeat_index,
                "repeat_id": artifacts.spec.repeat_id,
                "crossfit_id": artifacts.crossfit_id,
                "fold_plan_id": artifacts.fold_plan.plan_id,
                "partition_id": _partition_id(artifacts),
                "effective_n_splits": artifacts.fold_plan.effective_n_splits,
                "n_subjects": len(artifacts.fold_plan.subject_ids),
                "n_folds": len(artifacts.fold_plan.folds),
                "oof_coverage_complete": artifacts.completed_stage_oof_verified,
                "family_common_stage_connected": all(
                    len(fold.family_common_functionals)
                    == len(fold.receiver_incremental_models)
                    for fold in artifacts.folds
                ),
                "child_certification_status": artifacts.certification_status,
                "complete_pipeline_oof_certified": artifacts.is_oof_certified,
            }
        )
    return pd.DataFrame(rows, columns=_REPEAT_REGISTRY_COLUMNS).sort_values(
        "repeat_index", kind="stable", ignore_index=True
    )


def _absent_event(
    *,
    repeat_index: int,
    repeat_id: str,
    fold_id: str,
    contrast_name: str,
    receiver: str,
    family_id: str,
    driver_ids: tuple[str, ...],
    heldout_subject_ids: tuple[str, ...],
    reason_code: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    event = {
        "repeat_index": repeat_index,
        "repeat_id": repeat_id,
        "fold_id": fold_id,
        "contrast_name": contrast_name,
        "receiver": receiver,
        "family_id": family_id,
        "driver_ids": driver_ids,
        "n_heldout_subjects": len(heldout_subject_ids),
        "family_available": False,
        "family_estimable": False,
        "family_selected": None,
        "family_coefficient": None,
        "family_gain": None,
        "raw_family_gain": None,
        "selection_status": "not_estimable",
        "reason_code": reason_code,
    }
    subjects: list[dict[str, object]] = [
        {
            "repeat_index": repeat_index,
            "repeat_id": repeat_id,
            "fold_id": fold_id,
            "subject_id": subject_id,
            "contrast_name": contrast_name,
            "receiver": receiver,
            "family_id": family_id,
            "driver_ids": driver_ids,
            "family_available": False,
            "family_estimable": False,
            "family_selected": None,
            "selection_status": "not_estimable",
            "selection_reason_code": reason_code,
            "null_loss": None,
            "family_loss": None,
            "differential_effect": None,
            "bounded_incremental_gain": None,
            "effect_status": "not_estimable",
            "effect_reason_code": reason_code,
        }
        for subject_id in heldout_subject_ids
    ]
    return event, subjects


def _build_family_tables(
    repeats: tuple[CrossFitArtifacts, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    views, family_universe = _collect_fold_views(repeats)
    event_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    for artifacts in repeats:
        for manifest in artifacts.fold_plan.folds:
            for (contrast_name, receiver), family_members in sorted(
                family_universe.items()
            ):
                view = views.get(
                    (
                        artifacts.spec.repeat_index,
                        manifest.fold_id,
                        contrast_name,
                        receiver,
                    )
                )
                if view is None:
                    for family_id, driver_ids in sorted(family_members.items()):
                        event, absent_subject_rows = _absent_event(
                            repeat_index=artifacts.spec.repeat_index,
                            repeat_id=artifacts.spec.repeat_id,
                            fold_id=manifest.fold_id,
                            contrast_name=contrast_name,
                            receiver=receiver,
                            family_id=family_id,
                            driver_ids=driver_ids,
                            heldout_subject_ids=manifest.test_subject_ids,
                            reason_code=(
                                "receiver_contrast_not_in_training_fold_universe"
                            ),
                        )
                        event_rows.append(event)
                        subject_rows.extend(absent_subject_rows)
                    continue

                application_tables = view.application._validated_tables()
                attribution = application_tables[0].set_index("family_id", drop=False)
                differential = application_tables[1].set_index(
                    ["subject_id", "family_id"], drop=False
                )
                expected_differential = {
                    (subject_id, family_id)
                    for subject_id in manifest.test_subject_ids
                    for family_id in view.functional.family_ids
                }
                if set(differential.index.tolist()) != expected_differential:
                    raise ValueError(
                        "subject-family differential table lacks exact coverage"
                    )
                for family_id, driver_ids in sorted(family_members.items()):
                    if family_id not in attribution.index:
                        event, absent_subject_rows = _absent_event(
                            repeat_index=artifacts.spec.repeat_index,
                            repeat_id=artifacts.spec.repeat_id,
                            fold_id=manifest.fold_id,
                            contrast_name=contrast_name,
                            receiver=receiver,
                            family_id=family_id,
                            driver_ids=driver_ids,
                            heldout_subject_ids=manifest.test_subject_ids,
                            reason_code="family_not_in_training_fold_universe",
                        )
                        event_rows.append(event)
                        subject_rows.extend(absent_subject_rows)
                        continue
                    row = attribution.loc[family_id]
                    if isinstance(row, pd.DataFrame):
                        raise ValueError("family attribution contains duplicate rows")
                    selected = _nullable_bool(row["family_selected"])
                    selection_status = str(row["selection_status"])
                    selection_reason = _nullable_string(row["reason_code"])
                    family_estimable = bool(row["family_estimable"])
                    event_rows.append(
                        {
                            "repeat_index": artifacts.spec.repeat_index,
                            "repeat_id": artifacts.spec.repeat_id,
                            "fold_id": manifest.fold_id,
                            "contrast_name": contrast_name,
                            "receiver": receiver,
                            "family_id": family_id,
                            "driver_ids": driver_ids,
                            "n_heldout_subjects": len(manifest.test_subject_ids),
                            "family_available": True,
                            "family_estimable": family_estimable,
                            "family_selected": selected,
                            "family_coefficient": _nullable_float(
                                row["family_coefficient"]
                            ),
                            "family_gain": _nullable_float(row["family_gain"]),
                            "raw_family_gain": _nullable_float(row["raw_family_gain"]),
                            "selection_status": selection_status,
                            "reason_code": selection_reason,
                        }
                    )
                    for subject_id in manifest.test_subject_ids:
                        effect_row = differential.loc[(subject_id, family_id)]
                        if isinstance(effect_row, pd.DataFrame):
                            raise ValueError(
                                "subject-family differential contains duplicates"
                            )
                        effect = cast(pd.Series, effect_row)
                        subject_rows.append(
                            {
                                "repeat_index": artifacts.spec.repeat_index,
                                "repeat_id": artifacts.spec.repeat_id,
                                "fold_id": manifest.fold_id,
                                "subject_id": subject_id,
                                "contrast_name": contrast_name,
                                "receiver": receiver,
                                "family_id": family_id,
                                "driver_ids": driver_ids,
                                "family_available": True,
                                "family_estimable": family_estimable,
                                "family_selected": selected,
                                "selection_status": selection_status,
                                "selection_reason_code": selection_reason,
                                "null_loss": _nullable_float(effect["null_loss"]),
                                "family_loss": _nullable_float(effect["family_loss"]),
                                "differential_effect": _nullable_float(
                                    effect["differential_effect"]
                                ),
                                "bounded_incremental_gain": _nullable_float(
                                    effect["bounded_incremental_gain"]
                                ),
                                "effect_status": str(effect["status"]),
                                "effect_reason_code": _nullable_string(
                                    effect["reason_code"]
                                ),
                            }
                        )
    events = pd.DataFrame(event_rows, columns=_FAMILY_FOLD_EVENT_COLUMNS)
    subject_table = pd.DataFrame(subject_rows, columns=_SUBJECT_FAMILY_REPEAT_COLUMNS)
    if not events.empty:
        events = events.sort_values(
            [
                "repeat_index",
                "fold_id",
                "contrast_name",
                "receiver",
                "family_id",
            ],
            kind="stable",
            ignore_index=True,
        )
    if not subject_table.empty:
        subject_table = subject_table.sort_values(
            [
                "repeat_index",
                "subject_id",
                "contrast_name",
                "receiver",
                "family_id",
            ],
            kind="stable",
            ignore_index=True,
        )
    return events, subject_table


def _repeat_distribution(
    group: pd.DataFrame,
    *,
    value_column: str,
    reducer: str,
) -> list[float]:
    values: list[float] = []
    for _, repeat in group.groupby("repeat_index", sort=True, observed=True):
        observed = pd.to_numeric(repeat[value_column], errors="coerce").dropna()
        if observed.empty:
            continue
        if reducer == "mean":
            values.append(float(observed.mean()))
        elif reducer == "selected_fraction":
            values.append(float(observed.astype(bool).mean()))
        else:
            raise RuntimeError(f"unknown repeat reducer: {reducer}")
    return values


def _build_family_stability(
    subjects: pd.DataFrame,
    *,
    repeats: tuple[CrossFitArtifacts, ...],
    partition_by_repeat: dict[int, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if subjects.empty:
        return pd.DataFrame(columns=_FAMILY_STABILITY_COLUMNS)
    subject_ids = repeats[0].fold_plan.subject_ids
    expected_opportunities = len(repeats) * len(subject_ids)
    if set(partition_by_repeat) != set(range(len(repeats))):
        raise ValueError("partition_by_repeat must exactly cover repeat indexes")
    distinct_partitions = len(set(partition_by_repeat.values()))
    group_columns = ["contrast_name", "receiver", "family_id"]
    for key, group in subjects.groupby(group_columns, sort=True, observed=True):
        contrast_name, receiver, family_id = map(str, key)
        if (
            len(group) != expected_opportunities
            or group.duplicated(["repeat_index", "subject_id"]).any()
        ):
            raise ValueError(
                "repeated subject-family diagnostics lack exact repeat coverage"
            )
        driver_sets = {tuple(value) for value in group["driver_ids"]}
        if len(driver_sets) != 1:
            raise ValueError("stable family ID maps to inconsistent driver members")
        driver_ids = next(iter(driver_sets))
        available = group["family_available"].astype(bool)
        estimable = group["family_estimable"].astype(bool)
        selection_observed = estimable & group["family_selected"].notna()
        structurally_determined = (~estimable) & group["family_selected"].notna()
        selected = selection_observed & group["family_selected"].eq(True)
        complete_repeats = 0
        complete_repeat_indexes: list[int] = []
        for repeat_index, repeat in group.groupby(
            "repeat_index", sort=True, observed=True
        ):
            if (
                len(repeat) == len(subject_ids)
                and repeat["family_available"].astype(bool).all()
                and repeat["family_estimable"].astype(bool).all()
                and repeat["family_selected"].notna().all()
            ):
                complete_repeats += 1
                complete_repeat_indexes.append(int(cast(Any, repeat_index)))
        distinct_complete_partitions = len(
            {partition_by_repeat[index] for index in complete_repeat_indexes}
        )
        complete_group = group.loc[group["repeat_index"].isin(complete_repeat_indexes)]
        selection_distribution = _repeat_distribution(
            complete_group,
            value_column="family_selected",
            reducer="selected_fraction",
        )
        fit_events = group.loc[
            :,
            [
                "repeat_index",
                "fold_id",
                "family_available",
                "family_estimable",
                "family_selected",
            ],
        ].drop_duplicates()
        if fit_events.duplicated(["repeat_index", "fold_id"]).any():
            raise ValueError("family selection is inconsistent within a heldout fold")
        fit_available = fit_events["family_available"].astype(bool)
        fit_estimable = fit_events["family_estimable"].astype(bool)
        fit_selection_observed = fit_estimable & fit_events["family_selected"].notna()
        selected_fits = fit_selection_observed & fit_events["family_selected"].eq(True)
        complete_fit_events = fit_events.loc[
            fit_events["repeat_index"].isin(complete_repeat_indexes)
        ]
        fit_selection_distribution = _repeat_distribution(
            complete_fit_events,
            value_column="family_selected",
            reducer="selected_fraction",
        )
        effect_observed = group["effect_status"].eq("observed") & group[
            "bounded_incremental_gain"
        ].notna()
        structurally_determined_effect = group["effect_status"].eq(
            "structural_zero"
        ) & group["bounded_incremental_gain"].notna()
        complete_effect_indexes: list[int] = []
        for repeat_index, repeat in group.groupby(
            "repeat_index", sort=True, observed=True
        ):
            if (
                len(repeat) == len(subject_ids)
                and repeat["effect_status"].eq("observed").all()
                and repeat["bounded_incremental_gain"].notna().all()
            ):
                complete_effect_indexes.append(int(cast(Any, repeat_index)))
        distinct_complete_effect_partitions = len(
            {partition_by_repeat[index] for index in complete_effect_indexes}
        )
        complete_effect_group = group.loc[
            group["repeat_index"].isin(complete_effect_indexes)
        ]
        effect_distribution = _repeat_distribution(
            complete_effect_group,
            value_column="bounded_incremental_gain",
            reducer="mean",
        )
        if distinct_partitions < 2:
            status = "not_estimable"
            reason_code: str | None = "insufficient_distinct_repeat_partitions"
        elif complete_repeats < 2:
            status = "not_estimable"
            reason_code = "insufficient_complete_estimable_family_coverage"
        elif distinct_complete_partitions < 2:
            status = "not_estimable"
            reason_code = "insufficient_distinct_complete_family_partitions"
        elif len(selection_distribution) < 2:
            status = "not_estimable"
            reason_code = "insufficient_estimable_repeat_selection"
        else:
            status = "observed"
            reason_code = None
        if distinct_partitions < 2:
            effect_status = "not_estimable"
            effect_reason: str | None = "insufficient_distinct_repeat_partitions"
        elif len(complete_effect_indexes) < 2:
            effect_status = "not_estimable"
            effect_reason = "insufficient_complete_repeat_effect_coverage"
        elif distinct_complete_effect_partitions < 2:
            effect_status = "not_estimable"
            effect_reason = "insufficient_distinct_complete_effect_partitions"
        else:
            effect_status = "observed"
            effect_reason = None
        rows.append(
            {
                "contrast_name": contrast_name,
                "receiver": receiver,
                "family_id": family_id,
                "driver_ids": driver_ids,
                "n_repeats_requested": len(repeats),
                "n_distinct_partitions_overall": distinct_partitions,
                "n_distinct_complete_selection_partitions": (
                    distinct_complete_partitions
                ),
                "n_subject_repeat_opportunities": expected_opportunities,
                "n_family_available_opportunities": int(available.sum()),
                "family_availability_fraction": float(available.mean()),
                "n_family_estimable_opportunities": int(estimable.sum()),
                "family_estimable_fraction": float(estimable.mean()),
                "n_selection_observed_opportunities": int(selection_observed.sum()),
                "selection_observed_fraction": float(selection_observed.mean()),
                "n_structurally_determined_opportunities": int(
                    structurally_determined.sum()
                ),
                "structurally_determined_fraction": float(
                    structurally_determined.mean()
                ),
                "n_selected_opportunities": int(selected.sum()),
                "diagnostic_conditional_subject_exposure_selection_fraction": (
                    None
                    if not selection_observed.any()
                    else float(selected.sum() / selection_observed.sum())
                ),
                "n_fit_opportunities": len(fit_events),
                "n_fit_available": int(fit_available.sum()),
                "fit_availability_fraction": float(fit_available.mean()),
                "n_fit_estimable": int(fit_estimable.sum()),
                "fit_estimable_fraction": float(fit_estimable.mean()),
                "n_fit_selection_observed": int(fit_selection_observed.sum()),
                "n_selected_fits": int(selected_fits.sum()),
                "diagnostic_conditional_fit_selection_fraction": (
                    None
                    if not fit_selection_observed.any()
                    else float(selected_fits.sum() / fit_selection_observed.sum())
                ),
                "n_complete_selection_repeats": complete_repeats,
                "complete_selection_repeat_fraction": (complete_repeats / len(repeats)),
                "n_repeats_with_observed_selection": len(selection_distribution),
                "repeat_subject_exposure_selection_fraction_median": (
                    None
                    if not selection_distribution
                    else float(np.median(selection_distribution))
                ),
                "repeat_subject_exposure_selection_fraction_minimum": (
                    None if not selection_distribution else min(selection_distribution)
                ),
                "repeat_subject_exposure_selection_fraction_maximum": (
                    None if not selection_distribution else max(selection_distribution)
                ),
                "repeat_fit_selection_fraction_median": (
                    None
                    if not fit_selection_distribution
                    else float(np.median(fit_selection_distribution))
                ),
                "repeat_fit_selection_fraction_minimum": (
                    None
                    if not fit_selection_distribution
                    else min(fit_selection_distribution)
                ),
                "repeat_fit_selection_fraction_maximum": (
                    None
                    if not fit_selection_distribution
                    else max(fit_selection_distribution)
                ),
                "n_repeats_with_observed_effect": len(effect_distribution),
                "n_effect_observed_opportunities": int(effect_observed.sum()),
                "effect_observed_fraction": float(effect_observed.mean()),
                "n_structurally_determined_effect_opportunities": int(
                    structurally_determined_effect.sum()
                ),
                "structurally_determined_effect_fraction": float(
                    structurally_determined_effect.mean()
                ),
                "n_complete_effect_repeats": len(complete_effect_indexes),
                "complete_effect_repeat_fraction": (
                    len(complete_effect_indexes) / len(repeats)
                ),
                "n_distinct_complete_effect_partitions": (
                    distinct_complete_effect_partitions
                ),
                "repeat_mean_bounded_gain_median": (
                    None
                    if not effect_distribution
                    else float(np.median(effect_distribution))
                ),
                "repeat_mean_bounded_gain_minimum": (
                    None if not effect_distribution else min(effect_distribution)
                ),
                "repeat_mean_bounded_gain_maximum": (
                    None if not effect_distribution else max(effect_distribution)
                ),
                "effect_stability_status": effect_status,
                "effect_stability_reason_code": effect_reason,
                "status": status,
                "reason_code": reason_code,
                "formal_inference_status": _INFERENCE_STATUS,
            }
        )
    return pd.DataFrame(rows, columns=_FAMILY_STABILITY_COLUMNS).sort_values(
        group_columns, kind="stable", ignore_index=True
    )


@dataclass(frozen=True, slots=True, init=False)
class RepeatedCrossFitDiagnostics:
    """Producer-owned split-stability diagnostics from complete pipeline refits."""

    spec: RepeatedCrossFitSpec
    repeats: tuple[CrossFitArtifacts, ...]
    root_input_identity: SanitizedRawInputIdentity
    resource_bundle_content_id: str
    target_prior_content_id: str
    formal_inference_status: str
    diagnostic_status: str
    repeated_crossfit_id: str
    repeat_registry_digest: str
    family_fold_events_digest: str
    subject_family_repeat_values_digest: str
    family_repeat_stability_digest: str
    _repeat_registry: pd.DataFrame = field(repr=False)
    _family_fold_events: pd.DataFrame = field(repr=False)
    _subject_family_repeat_values: pd.DataFrame = field(repr=False)
    _family_repeat_stability: pd.DataFrame = field(repr=False)
    _producer_marker: str = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "RepeatedCrossFitDiagnostics are producer-owned; use "
            "run_repeated_subject_crossfit()"
        )

    @classmethod
    def _from_workflow(
        cls,
        *,
        spec: RepeatedCrossFitSpec,
        repeats: tuple[CrossFitArtifacts, ...],
        root_input_identity: SanitizedRawInputIdentity,
        resource_bundle_content_id: str,
        target_prior_content_id: str,
    ) -> RepeatedCrossFitDiagnostics:
        self = object.__new__(cls)
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "repeats", repeats)
        object.__setattr__(self, "root_input_identity", root_input_identity)
        object.__setattr__(
            self, "resource_bundle_content_id", resource_bundle_content_id
        )
        object.__setattr__(self, "target_prior_content_id", target_prior_content_id)
        object.__setattr__(self, "formal_inference_status", _INFERENCE_STATUS)
        object.__setattr__(self, "_producer_marker", _PRODUCER_MARKER)
        self.__post_init__()
        return self

    def __post_init__(self) -> None:
        if self._producer_marker != _PRODUCER_MARKER:
            raise TypeError("repeated diagnostics were not produced by the workflow")
        if not isinstance(self.spec, RepeatedCrossFitSpec):
            raise TypeError("spec must be a RepeatedCrossFitSpec")
        self.spec._require_intact()
        if not isinstance(self.root_input_identity, SanitizedRawInputIdentity):
            raise TypeError("root_input_identity must be a SanitizedRawInputIdentity")
        self.root_input_identity._require_intact()
        for field_name in (
            "resource_bundle_content_id",
            "target_prior_content_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty identifier")
        repeats = tuple(self.repeats)
        if len(repeats) != self.spec.n_repeats or any(
            not isinstance(item, CrossFitArtifacts) for item in repeats
        ):
            raise ValueError("repeats must exactly match n_repeats")
        base_spec = self.spec.crossfit_spec
        expected_subject_ids: tuple[str, ...] | None = None
        for repeat_index, artifacts in enumerate(repeats):
            artifacts._require_intact()
            if (
                artifacts.spec.spec_id != base_spec.spec_id
                or artifacts.spec.repeat_index != repeat_index
            ):
                raise ValueError("repeat child does not match the frozen base policy")
            if (
                artifacts.root_input_identity.identity_id
                != self.root_input_identity.identity_id
            ):
                raise ValueError("repeat children do not share the bound root input")
            if expected_subject_ids is None:
                expected_subject_ids = artifacts.fold_plan.subject_ids
            elif artifacts.fold_plan.subject_ids != expected_subject_ids:
                raise ValueError("repeat children use different subject universes")
            for fold in artifacts.folds:
                if (
                    fold.training.config.digest
                    != self.root_input_identity.config_digest
                    or fold.training.resource_bundle_content_id
                    != self.resource_bundle_content_id
                    or fold.training.target_prior_content_id
                    != self.target_prior_content_id
                ):
                    raise ValueError(
                        "repeat child config or resource provenance is incompatible"
                    )
        if len({item.spec.repeat_id for item in repeats}) != len(repeats):
            raise ValueError("repeat IDs must be unique")
        if len({item.fold_plan.plan_id for item in repeats}) != len(repeats):
            raise ValueError("fold plan IDs must be unique")
        if len({item.crossfit_id for item in repeats}) != len(repeats):
            raise ValueError("cross-fit child IDs must be unique")

        registry = _build_repeat_registry(repeats)
        events, subject_values = _build_family_tables(repeats)
        distinct_partitions = int(registry["partition_id"].nunique())
        partition_by_repeat = {
            int(cast(Any, row.repeat_index)): str(row.partition_id)
            for row in registry.itertuples(index=False)
        }
        stability = _build_family_stability(
            subject_values,
            repeats=repeats,
            partition_by_repeat=partition_by_repeat,
        )
        if stability.empty:
            diagnostic_status = _NO_FAMILY_STATUS
        elif distinct_partitions < 2:
            diagnostic_status = _NO_PARTITION_STATUS
        elif not (
            stability["status"].eq("observed").any()
            or stability["effect_stability_status"].eq("observed").any()
        ):
            diagnostic_status = _NO_ESTIMABLE_STATUS
        elif (
            stability["status"].eq("observed").all()
            and stability["effect_stability_status"].eq("observed").all()
        ):
            diagnostic_status = _OBSERVED_STATUS
        else:
            diagnostic_status = _PARTIAL_STATUS
        digests = (
            _table_digest(
                "repeated_crossfit_repeat_registry",
                registry,
                _REPEAT_REGISTRY_COLUMNS,
            ),
            _table_digest(
                "repeated_crossfit_family_fold_events",
                events,
                _FAMILY_FOLD_EVENT_COLUMNS,
            ),
            _table_digest(
                "repeated_crossfit_subject_family_repeat_values",
                subject_values,
                _SUBJECT_FAMILY_REPEAT_COLUMNS,
            ),
            _table_digest(
                "repeated_crossfit_family_repeat_stability",
                stability,
                _FAMILY_STABILITY_COLUMNS,
            ),
        )
        payload = {
            "config_digest": self.root_input_identity.config_digest,
            "diagnostic_status": diagnostic_status,
            "family_fold_events_digest": digests[1],
            "family_repeat_stability_digest": digests[3],
            "formal_inference_status": self.formal_inference_status,
            "input_digest": self.root_input_identity.input_digest,
            "root_input_identity_id": self.root_input_identity.identity_id,
            "repeat_crossfit_ids": [item.crossfit_id for item in repeats],
            "repeat_registry_digest": digests[0],
            "resource_bundle_content_id": self.resource_bundle_content_id,
            "spec_id": self.spec.spec_id,
            "subject_family_repeat_values_digest": digests[2],
            "target_prior_content_id": self.target_prior_content_id,
        }
        object.__setattr__(self, "repeats", repeats)
        object.__setattr__(self, "diagnostic_status", diagnostic_status)
        object.__setattr__(self, "repeat_registry_digest", digests[0])
        object.__setattr__(self, "family_fold_events_digest", digests[1])
        object.__setattr__(self, "subject_family_repeat_values_digest", digests[2])
        object.__setattr__(self, "family_repeat_stability_digest", digests[3])
        object.__setattr__(self, "_repeat_registry", registry)
        object.__setattr__(self, "_family_fold_events", events)
        object.__setattr__(self, "_subject_family_repeat_values", subject_values)
        object.__setattr__(self, "_family_repeat_stability", stability)
        object.__setattr__(
            self,
            "repeated_crossfit_id",
            stable_id(
                "repeated_subject_crossfit_diagnostics",
                payload,
                schema_version="1",
            ),
        )

    @property
    def is_inference_eligible(self) -> bool:
        """Repeated split diagnostics never constitute formal inference."""

        return False

    @property
    def input_digest(self) -> str:
        self.root_input_identity._require_intact()
        return self.root_input_identity.input_digest

    @property
    def config_digest(self) -> str:
        self.root_input_identity._require_intact()
        return self.root_input_identity.config_digest

    @property
    def repeat_registry(self) -> pd.DataFrame:
        return self._repeat_registry.copy(deep=True)

    @property
    def family_fold_events(self) -> pd.DataFrame:
        return self._family_fold_events.copy(deep=True)

    @property
    def subject_family_repeat_values(self) -> pd.DataFrame:
        return self._subject_family_repeat_values.copy(deep=True)

    @property
    def family_repeat_stability(self) -> pd.DataFrame:
        return self._family_repeat_stability.copy(deep=True)

    def _require_intact(self) -> None:
        try:
            observed_digests = (
                _table_digest(
                    "repeated_crossfit_repeat_registry",
                    self._repeat_registry,
                    _REPEAT_REGISTRY_COLUMNS,
                ),
                _table_digest(
                    "repeated_crossfit_family_fold_events",
                    self._family_fold_events,
                    _FAMILY_FOLD_EVENT_COLUMNS,
                ),
                _table_digest(
                    "repeated_crossfit_subject_family_repeat_values",
                    self._subject_family_repeat_values,
                    _SUBJECT_FAMILY_REPEAT_COLUMNS,
                ),
                _table_digest(
                    "repeated_crossfit_family_repeat_stability",
                    self._family_repeat_stability,
                    _FAMILY_STABILITY_COLUMNS,
                ),
            )
            repeated = RepeatedCrossFitDiagnostics._from_workflow(
                spec=self.spec,
                repeats=self.repeats,
                root_input_identity=self.root_input_identity,
                resource_bundle_content_id=self.resource_bundle_content_id,
                target_prior_content_id=self.target_prior_content_id,
            )
            valid = (
                self._producer_marker == _PRODUCER_MARKER
                and self.formal_inference_status == _INFERENCE_STATUS
                and not self.is_inference_eligible
                and self.root_input_identity.identity_id
                == repeated.root_input_identity.identity_id
                and self.resource_bundle_content_id
                == repeated.resource_bundle_content_id
                and self.target_prior_content_id == repeated.target_prior_content_id
                and self.diagnostic_status == repeated.diagnostic_status
                and self.repeated_crossfit_id == repeated.repeated_crossfit_id
                and observed_digests
                == (
                    self.repeat_registry_digest,
                    self.family_fold_events_digest,
                    self.subject_family_repeat_values_digest,
                    self.family_repeat_stability_digest,
                )
                and self.repeat_registry_digest == repeated.repeat_registry_digest
                and self.family_fold_events_digest == repeated.family_fold_events_digest
                and self.subject_family_repeat_values_digest
                == repeated.subject_family_repeat_values_digest
                and self.family_repeat_stability_digest
                == repeated.family_repeat_stability_digest
            )
        except (
            AttributeError,
            ContractError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            raise ContractError(
                "Repeated cross-fit diagnostics failed integrity validation",
                code="repeated_crossfit_diagnostics_integrity_violation",
                field="repeated_crossfit_id",
                remediation="Rerun every repeat from intact raw-count inputs",
            ) from error
        if not valid:
            raise ContractError(
                "Repeated cross-fit diagnostics failed integrity validation",
                code="repeated_crossfit_diagnostics_integrity_violation",
                field="repeated_crossfit_id",
                remediation="Rerun every repeat from intact raw-count inputs",
            )

    def to_manifest(self) -> dict[str, object]:
        """Return provenance and explicit non-inferential release boundaries."""

        self._require_intact()
        selection_status_counts = {
            str(status): int(count)
            for status, count in sorted(
                self._family_repeat_stability["status"].value_counts().items()
            )
        }
        effect_status_counts = {
            str(status): int(count)
            for status, count in sorted(
                self._family_repeat_stability["effect_stability_status"]
                .value_counts()
                .items()
            )
        }
        return {
            "repeated_crossfit_id": self.repeated_crossfit_id,
            "spec": self.spec.to_dict(),
            "diagnostic_status": self.diagnostic_status,
            "input_digest": self.input_digest,
            "config_digest": self.config_digest,
            "root_input_identity_id": self.root_input_identity.identity_id,
            "resource_bundle_content_id": self.resource_bundle_content_id,
            "target_prior_content_id": self.target_prior_content_id,
            "formal_inference_status": self.formal_inference_status,
            "is_inference_eligible": self.is_inference_eligible,
            "family_universe_policy": "union_of_observed_fold_family_ids_v1",
            "permanently_absent_family_semantics": (
                "outside_observed_union_not_materialized"
            ),
            "child_retention_policy": "full_children_in_memory_development_v1",
            "streaming_full_pipeline_resampling_ready": False,
            "selection_stability_status_counts": selection_status_counts,
            "effect_stability_status_counts": effect_status_counts,
            "inferential_fields_available": [],
            "inferential_fields_unavailable": [
                "p_value",
                "q_value",
                "comm_probability",
                "specificity_support",
                "confidence_interval",
            ],
            "repeat_registry_digest": self.repeat_registry_digest,
            "family_fold_events_digest": self.family_fold_events_digest,
            "subject_family_repeat_values_digest": (
                self.subject_family_repeat_values_digest
            ),
            "family_repeat_stability_digest": (self.family_repeat_stability_digest),
            "repeat_crossfit_ids": [item.crossfit_id for item in self.repeats],
            "n_repeat_registry_rows": len(self._repeat_registry),
            "n_family_fold_event_rows": len(self._family_fold_events),
            "n_subject_family_repeat_rows": len(self._subject_family_repeat_values),
            "n_family_repeat_stability_rows": len(self._family_repeat_stability),
        }


def run_repeated_subject_crossfit(
    adata: AnnData,
    config: CrychicConfig,
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
    *,
    spec: RepeatedCrossFitSpec,
) -> RepeatedCrossFitDiagnostics:
    """Refit the public subject cross-fit pipeline for every declared repeat."""

    if not isinstance(spec, RepeatedCrossFitSpec):
        raise TypeError("spec must be a RepeatedCrossFitSpec")
    spec._require_intact()
    sanitized_snapshot = _sanitized_raw_input_snapshot(
        adata,
        config,
    )
    root_input_identity = sanitized_snapshot.identity
    resource_bundle_content_id = _resource_bundle_content_id(resource_bundle)
    target_prior_content_id = _target_prior_content_id(target_prior)
    repeats = tuple(
        _run_subject_crossfit(
            sanitized_snapshot,
            config,
            resource_bundle,
            target_prior,
            spec=replace(spec.crossfit_spec, repeat_index=repeat_index),
        )
        for repeat_index in range(spec.n_repeats)
    )
    return RepeatedCrossFitDiagnostics._from_workflow(
        spec=spec,
        repeats=repeats,
        root_input_identity=root_input_identity,
        resource_bundle_content_id=resource_bundle_content_id,
        target_prior_content_id=target_prior_content_id,
    )


__all__ = [
    "RepeatedCrossFitDiagnostics",
    "RepeatedCrossFitSpec",
    "run_repeated_subject_crossfit",
]
