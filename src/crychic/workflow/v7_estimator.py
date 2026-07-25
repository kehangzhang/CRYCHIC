"""Unified post-cross-fit estimator for the v7 sample-level score path."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crychic.core import canonical_digest, stable_id
from crychic.inference import (
    DesignAwareDifferentialResult,
    DesignAwareHypergraphShrinkageV2Result,
    DifferentialDesignKind,
    DifferentialDesignSpec,
    TwoPartOccurrenceV2Result,
    TwoPartOccurrenceV2Spec,
    build_sample_edge_differential_input,
    fit_design_aware_differential,
    fit_design_aware_hypergraph_shrinkage_v2,
    fit_two_part_occurrence_v2,
)
from crychic.scoring import (
    FrozenHypergraphPrior,
    UncertaintyAwareHypergraphShrinkageV2Spec,
)

from .crossfit import CrossFitArtifacts
from .occurrence_v2 import CrossFitTwoPartOccurrenceV2Result

V7_CROSSFIT_ESTIMATOR_VERSION = "unified_sample_level_crossfit_estimator_v7"
_SCHEMA_VERSION = "1.0.0"
_SUPPORTED_SCORE_HEADS = {
    "sender_detection_raw",
    "parent_peak_raw",
    "parent_total_raw",
    "parent_mean_raw",
    "program_signed",
}


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _sample_metadata_columns(design: DifferentialDesignSpec) -> tuple[str, ...]:
    columns = {
        design.subject_column,
        design.condition_column,
        *design.batch_columns,
        *design.continuous_covariates,
        *design.categorical_covariates,
    }
    if design.cohort_column is not None:
        columns.add(design.cohort_column)
    return (design.sample_column, *sorted(columns.difference({design.sample_column})))


def _required_design_columns(design: DifferentialDesignSpec) -> tuple[str, ...]:
    columns = set(_sample_metadata_columns(design))
    if design.precision_weight_column is not None:
        columns.add(design.precision_weight_column)
    return tuple(sorted(columns))


def _canonical_metadata_digest(table: pd.DataFrame, columns: tuple[str, ...]) -> str:
    records: list[dict[str, object]] = []
    for row in (
        table.loc[:, list(columns)]
        .sort_values(columns[0], kind="stable", ignore_index=True)
        .itertuples(index=False, name=None)
    ):
        record: dict[str, object] = {}
        for column, value in zip(columns, row, strict=True):
            if value is pd.NA or value is pd.NaT or pd.isna(value):
                record[column] = None
            elif isinstance(value, np.generic):
                record[column] = value.item()
            else:
                record[column] = value
        records.append(record)
    return str(canonical_digest(records))


def _frame_digest(table: pd.DataFrame) -> str:
    records: list[dict[str, object]] = []
    for row in table.itertuples(index=False, name=None):
        record: dict[str, object] = {}
        for column, value in zip(table.columns, row, strict=True):
            if value is None or value is pd.NA or value is pd.NaT:
                normalized: object = None
            elif isinstance(value, (float, np.floating)):
                normalized = (
                    None
                    if not np.isfinite(float(value))
                    else {"float_hex": float(value).hex()}
                )
            elif isinstance(value, (bool, np.bool_)):
                normalized = bool(value)
            elif isinstance(value, (int, np.integer)):
                normalized = int(value)
            else:
                normalized = str(value)
            record[str(column)] = normalized
        records.append(record)
    return str(
        canonical_digest(
            {
                "columns": list(map(str, table.columns)),
                "records": records,
            }
        )
    )


def _merge_required_metadata(
    score_table: pd.DataFrame,
    sample_metadata: pd.DataFrame | None,
    design: DifferentialDesignSpec,
) -> tuple[pd.DataFrame, str]:
    required = _required_design_columns(design)
    result = score_table.copy(deep=True)
    missing = tuple(column for column in required if column not in result.columns)
    if missing:
        if sample_metadata is None:
            raise ValueError(
                f"v7 design requires sample metadata columns: {list(missing)}"
            )
        if not isinstance(sample_metadata, pd.DataFrame):
            raise TypeError("sample_metadata must be a pandas DataFrame or None")
        if "sample_id" not in sample_metadata.columns:
            raise ValueError("sample_metadata requires the canonical sample_id column")
        metadata = sample_metadata.copy(deep=True)
        if (
            metadata["sample_id"].isna().any()
            or metadata["sample_id"].duplicated().any()
        ):
            raise ValueError("sample_metadata requires unique non-missing sample IDs")
        absent = set(missing).difference(metadata.columns)
        if absent:
            raise ValueError(
                f"sample_metadata is missing v7 design fields: {sorted(absent)}"
            )
        result = result.merge(
            metadata.loc[:, ["sample_id", *missing]],
            on="sample_id",
            how="left",
            validate="many_to_one",
            sort=False,
        )
    missing_values = result.loc[:, list(required)].isna().any()
    if missing_values.any():
        raise ValueError(
            "v7 design metadata contains missing values in: "
            f"{sorted(missing_values.index[missing_values].tolist())}"
        )
    metadata_columns = _sample_metadata_columns(design)
    metadata_frame = result.loc[:, list(metadata_columns)]
    consistency = metadata_frame.groupby(
        design.sample_column,
        observed=True,
        sort=False,
    ).nunique(dropna=False)
    if consistency.gt(1).any(axis=None):
        raise ValueError("v7 design metadata is not constant within sample")
    sample_rows = metadata_frame.drop_duplicates(subset=[design.sample_column])
    return result, _canonical_metadata_digest(sample_rows, metadata_columns)


def _build_oof_input(
    crossfit: CrossFitArtifacts,
    *,
    score_head: str,
    design: DifferentialDesignSpec,
    sample_metadata: pd.DataFrame | None,
) -> tuple[pd.DataFrame, tuple[str, ...], str]:
    parts: list[pd.DataFrame] = []
    provenance_ids: list[str] = []
    for fold in crossfit.folds:
        scores = fold.sample_edge_scores_v2
        if scores is None:
            raise ValueError("v7 estimator requires M0 v2 scores in every fold")
        provenance_ids.append(scores.provenance.provenance_id)
        parts.append(
            build_sample_edge_differential_input(
                scores,
                score_head=score_head,
            )
        )
    if not parts:
        raise ValueError("v7 estimator requires at least one cross-fit fold")
    table = pd.concat(parts, ignore_index=True)
    if table.duplicated(["event_id", "sample_id"]).any():
        raise ValueError("v7 OOF score rows overlap across outer folds")
    observed_samples = set(table["sample_id"].astype(str))
    expected_samples = set(crossfit.oof_sample_edge_scores_v2["sample_id"].astype(str))
    if observed_samples != expected_samples:
        raise ValueError("v7 estimator does not cover the exact OOF sample axis")
    merged, metadata_digest = _merge_required_metadata(
        table,
        sample_metadata,
        design,
    )
    return merged, tuple(sorted(provenance_ids)), metadata_digest


@dataclass(frozen=True, slots=True, kw_only=True)
class V7EstimatorSpec:
    """Pre-registered score head and optional M4/M5 post-cross-fit stages."""

    design: DifferentialDesignSpec
    score_head: str = "parent_mean_raw"
    occurrence_spec: TwoPartOccurrenceV2Spec | None = None
    hypergraph_spec: UncertaintyAwareHypergraphShrinkageV2Spec | None = None
    hypergraph_contrast_name: str | None = None
    schema_version: str = _SCHEMA_VERSION
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.design, DifferentialDesignSpec):
            raise TypeError("design must be DifferentialDesignSpec")
        score_head = _name(self.score_head, field_name="score_head")
        if score_head not in _SUPPORTED_SCORE_HEADS:
            raise ValueError(
                f"score_head must be one of {sorted(_SUPPORTED_SCORE_HEADS)}"
            )
        occurrence = self.occurrence_spec
        if occurrence is not None:
            if not isinstance(occurrence, TwoPartOccurrenceV2Spec):
                raise TypeError(
                    "occurrence_spec must be TwoPartOccurrenceV2Spec or None"
                )
            if occurrence.design.spec_id != self.design.spec_id:
                raise ValueError("M4 and continuous v7 designs must be identical")
        hypergraph = self.hypergraph_spec
        contrast = self.hypergraph_contrast_name
        if (hypergraph is None) != (contrast is None):
            raise ValueError(
                "hypergraph_spec and hypergraph_contrast_name must coexist"
            )
        if hypergraph is not None:
            if not isinstance(hypergraph, UncertaintyAwareHypergraphShrinkageV2Spec):
                raise TypeError(
                    "hypergraph_spec must be "
                    "UncertaintyAwareHypergraphShrinkageV2Spec or None"
                )
            contrast = _name(contrast, field_name="hypergraph_contrast_name")
            declared = (
                {f"slope:{self.design.condition_column}"}
                if DifferentialDesignKind(self.design.design_kind)
                is DifferentialDesignKind.CONTINUOUS
                else {item.name for item in self.design.contrasts}
            )
            if contrast not in declared:
                raise ValueError("M5 contrast must be pre-registered in the design")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")
        object.__setattr__(self, "score_head", score_head)
        object.__setattr__(self, "hypergraph_contrast_name", contrast)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "v7_crossfit_estimator_spec",
                self._identity_payload(),
                schema_version=self.schema_version,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "design_spec_id": self.design.spec_id,
            "hypergraph_contrast_name": self.hypergraph_contrast_name,
            "hypergraph_spec_id": (
                None if self.hypergraph_spec is None else self.hypergraph_spec.spec_id
            ),
            "occurrence_spec_id": (
                None if self.occurrence_spec is None else self.occurrence_spec.spec_id
            ),
            "schema_version": self.schema_version,
            "score_head": self.score_head,
            "version": V7_CROSSFIT_ESTIMATOR_VERSION,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "spec_id": self.spec_id,
            **self._identity_payload(),
            "design": self.design.to_dict(),
            "occurrence_spec": (
                None if self.occurrence_spec is None else self.occurrence_spec.to_dict()
            ),
            "hypergraph_spec": (
                None if self.hypergraph_spec is None else self.hypergraph_spec.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossFitV7EstimatorResult:
    """M0/M1 continuous, M4 occurrence, and M5 results from one cross-fit."""

    crossfit_id: str
    spec: V7EstimatorSpec
    differential: DesignAwareDifferentialResult
    occurrence: CrossFitTwoPartOccurrenceV2Result | None
    hypergraph: DesignAwareHypergraphShrinkageV2Result | None
    score_provenance_ids: tuple[str, ...]
    sample_metadata_digest: str
    output_digest: str = field(init=False)
    estimator_id: str = field(init=False)

    def __post_init__(self) -> None:
        crossfit_id = _name(self.crossfit_id, field_name="crossfit_id")
        if not isinstance(self.spec, V7EstimatorSpec):
            raise TypeError("spec must be V7EstimatorSpec")
        if not isinstance(self.differential, DesignAwareDifferentialResult):
            raise TypeError("differential must be DesignAwareDifferentialResult")
        if self.differential.spec.spec_id != self.spec.design.spec_id:
            raise ValueError("v7 differential design differs from estimator spec")
        occurrence = self.occurrence
        if (occurrence is None) != (self.spec.occurrence_spec is None):
            raise ValueError("v7 occurrence result does not match estimator spec")
        if occurrence is not None:
            occurrence_spec = self.spec.occurrence_spec
            assert occurrence_spec is not None
            if (
                not isinstance(occurrence, CrossFitTwoPartOccurrenceV2Result)
                or occurrence.crossfit_id != crossfit_id
                or occurrence.occurrence.spec.spec_id != occurrence_spec.spec_id
            ):
                raise ValueError("v7 occurrence result lineage is invalid")
        hypergraph = self.hypergraph
        if (hypergraph is None) != (self.spec.hypergraph_spec is None):
            raise ValueError("v7 hypergraph result does not match estimator spec")
        if hypergraph is not None:
            hypergraph_spec = self.spec.hypergraph_spec
            assert hypergraph_spec is not None
            if (
                not isinstance(hypergraph, DesignAwareHypergraphShrinkageV2Result)
                or hypergraph.design_spec_id != self.spec.design.spec_id
                or hypergraph.contrast_name != self.spec.hypergraph_contrast_name
                or hypergraph.fit.spec.spec_id != hypergraph_spec.spec_id
            ):
                raise ValueError("v7 hypergraph result lineage is invalid")
        provenance = tuple(sorted(self.score_provenance_ids))
        if not provenance or len(provenance) != len(set(provenance)):
            raise ValueError("score_provenance_ids must be non-empty and unique")
        metadata_digest = _name(
            self.sample_metadata_digest,
            field_name="sample_metadata_digest",
        )
        if len(metadata_digest) != 64:
            raise ValueError("sample_metadata_digest must be a canonical digest")
        output_digest = str(
            canonical_digest(
                {
                    "differential_effects": _frame_digest(self.differential.effects),
                    "differential_omnibus": _frame_digest(self.differential.omnibus),
                    "hypergraph": (
                        None
                        if hypergraph is None
                        else _frame_digest(hypergraph.shrinkage)
                    ),
                    "occurrence_effects": (
                        None
                        if occurrence is None
                        else _frame_digest(occurrence.occurrence.effects)
                    ),
                    "occurrence_subject_events": (
                        None
                        if occurrence is None
                        else _frame_digest(occurrence.occurrence.subject_events)
                    ),
                }
            )
        )
        payload = {
            "crossfit_id": crossfit_id,
            "differential_effect_ids": list(self.differential.effects["effect_id"]),
            "differential_omnibus_ids": list(self.differential.omnibus["omnibus_id"]),
            "hypergraph_result_id": (
                None if hypergraph is None else hypergraph.result_id
            ),
            "occurrence_result_id": (
                None if occurrence is None else occurrence.result_id
            ),
            "output_digest": output_digest,
            "sample_metadata_digest": metadata_digest,
            "score_provenance_ids": list(provenance),
            "spec_id": self.spec.spec_id,
            "version": V7_CROSSFIT_ESTIMATOR_VERSION,
        }
        object.__setattr__(self, "crossfit_id", crossfit_id)
        object.__setattr__(self, "score_provenance_ids", provenance)
        object.__setattr__(self, "sample_metadata_digest", metadata_digest)
        object.__setattr__(self, "output_digest", output_digest)
        object.__setattr__(
            self,
            "estimator_id",
            stable_id(
                "crossfit_v7_estimator_result",
                payload,
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _require_intact(self) -> None:
        expected_output_digest = str(
            canonical_digest(
                {
                    "differential_effects": _frame_digest(self.differential.effects),
                    "differential_omnibus": _frame_digest(self.differential.omnibus),
                    "hypergraph": (
                        None
                        if self.hypergraph is None
                        else _frame_digest(self.hypergraph.shrinkage)
                    ),
                    "occurrence_effects": (
                        None
                        if self.occurrence is None
                        else _frame_digest(self.occurrence.occurrence.effects)
                    ),
                    "occurrence_subject_events": (
                        None
                        if self.occurrence is None
                        else _frame_digest(self.occurrence.occurrence.subject_events)
                    ),
                }
            )
        )
        expected_id = stable_id(
            "crossfit_v7_estimator_result",
            {
                "crossfit_id": self.crossfit_id,
                "differential_effect_ids": list(self.differential.effects["effect_id"]),
                "differential_omnibus_ids": list(
                    self.differential.omnibus["omnibus_id"]
                ),
                "hypergraph_result_id": (
                    None if self.hypergraph is None else self.hypergraph.result_id
                ),
                "occurrence_result_id": (
                    None if self.occurrence is None else self.occurrence.result_id
                ),
                "output_digest": expected_output_digest,
                "sample_metadata_digest": self.sample_metadata_digest,
                "score_provenance_ids": list(self.score_provenance_ids),
                "spec_id": self.spec.spec_id,
                "version": V7_CROSSFIT_ESTIMATOR_VERSION,
            },
            schema_version=_SCHEMA_VERSION,
        )
        if (
            self.output_digest != expected_output_digest
            or self.estimator_id != expected_id
        ):
            raise ValueError("v7 estimator output integrity validation failed")

    def to_manifest(self) -> dict[str, object]:
        self._require_intact()
        return {
            "estimator_id": self.estimator_id,
            "crossfit_id": self.crossfit_id,
            "spec": self.spec.to_dict(),
            "score_provenance_ids": list(self.score_provenance_ids),
            "sample_metadata_digest": self.sample_metadata_digest,
            "output_digest": self.output_digest,
            "continuous_effect_rows": len(self.differential.effects),
            "continuous_omnibus_rows": len(self.differential.omnibus),
            "occurrence": (
                None if self.occurrence is None else self.occurrence.to_manifest()
            ),
            "hypergraph": (
                None if self.hypergraph is None else self.hypergraph.to_manifest()
            ),
            "formal_inference_allowed": False,
            "required_formal_next_stage": "v7_full_pipeline_subject_resampling",
        }


def fit_crossfit_v7_estimator(
    crossfit: CrossFitArtifacts,
    spec: V7EstimatorSpec,
    *,
    sample_metadata: pd.DataFrame | None = None,
    hypergraph_prior: FrozenHypergraphPrior | None = None,
) -> CrossFitV7EstimatorResult:
    """Fit all configured v7 post-cross-fit stages on exact held-out rows."""

    if not isinstance(crossfit, CrossFitArtifacts):
        raise TypeError("crossfit must be CrossFitArtifacts")
    crossfit._require_intact()
    if not isinstance(spec, V7EstimatorSpec):
        raise TypeError("spec must be V7EstimatorSpec")
    if (hypergraph_prior is None) != (spec.hypergraph_spec is None):
        raise ValueError(
            "hypergraph_prior must be supplied exactly when M5 is configured"
        )
    score_input, provenance_ids, metadata_digest = _build_oof_input(
        crossfit,
        score_head=spec.score_head,
        design=spec.design,
        sample_metadata=sample_metadata,
    )
    differential = fit_design_aware_differential(score_input, spec.design)

    occurrence: CrossFitTwoPartOccurrenceV2Result | None = None
    if spec.occurrence_spec is not None:
        occurrence_input, occurrence_provenance, occurrence_metadata_digest = (
            _build_oof_input(
                crossfit,
                score_head=spec.occurrence_spec.activity_head,
                design=spec.design,
                sample_metadata=sample_metadata,
            )
        )
        if (
            occurrence_provenance != provenance_ids
            or occurrence_metadata_digest != metadata_digest
        ):
            raise ValueError("M4 and continuous v7 inputs have different lineage")
        occurrence_result: TwoPartOccurrenceV2Result = fit_two_part_occurrence_v2(
            occurrence_input,
            spec.occurrence_spec,
        )
        occurrence = CrossFitTwoPartOccurrenceV2Result(
            crossfit_id=crossfit.crossfit_id,
            occurrence=occurrence_result,
            fold_score_provenance_ids=provenance_ids,
        )

    hypergraph: DesignAwareHypergraphShrinkageV2Result | None = None
    if spec.hypergraph_spec is not None:
        assert hypergraph_prior is not None
        assert spec.hypergraph_contrast_name is not None
        hypergraph = fit_design_aware_hypergraph_shrinkage_v2(
            differential,
            prior=hypergraph_prior,
            contrast_name=spec.hypergraph_contrast_name,
            spec=spec.hypergraph_spec,
        )
    return CrossFitV7EstimatorResult(
        crossfit_id=crossfit.crossfit_id,
        spec=spec,
        differential=differential,
        occurrence=occurrence,
        hypergraph=hypergraph,
        score_provenance_ids=provenance_ids,
        sample_metadata_digest=metadata_digest,
    )


__all__ = [
    "V7_CROSSFIT_ESTIMATOR_VERSION",
    "CrossFitV7EstimatorResult",
    "V7EstimatorSpec",
    "fit_crossfit_v7_estimator",
]
