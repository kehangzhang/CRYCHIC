"""Train-only formula encoding reparameterized around one EMM contrast."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import patsy  # type: ignore[import-untyped]

from crychic.core import canonical_digest, canonical_json, stable_id

from .audit import audit_sample_design, default_design_formula
from .contrasts import ContrastSpec

_PRODUCER_MARKER = "crychic.design.frozen_encoder.v2"
_NUMERIC_SCALE_FLOOR = 1e-12


def _plain(value: object) -> Hashable:
    result = value.item() if isinstance(value, np.generic) else value
    if not isinstance(result, Hashable):
        raise TypeError("design values must be hashable")
    return result


def _typed_key(value: object) -> str:
    plain = _plain(value)
    serialized: str = canonical_json(
        {
            "type": f"{type(plain).__module__}.{type(plain).__qualname__}",
            "value": plain,
        }
    )
    return serialized


def _names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(values)
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in result
    ):
        raise ValueError(f"{field_name} must contain non-empty canonical names")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique names")
    return result


def _array_digest(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8", order="C")
    if np.any(~np.isfinite(canonical)):
        raise ValueError("design array digest requires finite values")
    digest = hashlib.sha256()
    digest.update(canonical_json({"shape": list(canonical.shape)}).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _immutable_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).copy(order="C")
    if np.any(~np.isfinite(array)):
        raise ValueError("frozen design arrays must be finite")
    array.setflags(write=False)
    return cast(np.ndarray, array)


def _is_categorical(series: pd.Series, *, declared: bool) -> bool:
    return bool(
        declared
        or pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or pd.api.types.is_bool_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _levels(series: pd.Series) -> tuple[Hashable, ...]:
    by_key: dict[str, Hashable] = {}
    for value in series.tolist():
        by_key.setdefault(_typed_key(value), _plain(value))
    return tuple(by_key[key] for key in sorted(by_key))


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenCovariateEncoding:
    """Training-derived type and support registry for one covariate."""

    name: str
    kind: str
    levels: tuple[Hashable, ...] = ()
    center: float | None = None
    scale: float | None = None
    column_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("covariate encoding name must be non-empty")
        if self.kind not in {"categorical", "continuous", "constant_continuous"}:
            raise ValueError("unsupported frozen covariate encoding kind")
        columns = _names(self.column_ids, field_name="column_ids")
        levels = tuple(self.levels)
        if self.kind == "categorical":
            if not levels or len({_typed_key(value) for value in levels}) != len(
                levels
            ):
                raise ValueError("categorical levels must be non-empty and unique")
            if len(columns) != max(0, len(levels) - 1):
                raise ValueError("categorical columns must use reference-level coding")
            if self.center is not None or self.scale is not None:
                raise ValueError("categorical encoding cannot declare center or scale")
        else:
            if levels:
                raise ValueError("continuous encoding cannot declare levels")
            if (
                self.center is None
                or self.scale is None
                or not math.isfinite(self.center)
                or not math.isfinite(self.scale)
                or self.scale <= 0
            ):
                raise ValueError("continuous center and scale must be finite")
            expected_columns = 1 if self.kind == "continuous" else 0
            if len(columns) != expected_columns:
                raise ValueError("continuous column count does not match its kind")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "column_ids", columns)

    def to_dict(self) -> dict[str, object]:
        """Return canonical encoding provenance."""

        return {
            "name": self.name,
            "kind": self.kind,
            "levels": list(self.levels),
            "center": self.center,
            "scale": self.scale,
            "column_ids": list(self.column_ids),
        }


@dataclass(frozen=True, slots=True, init=False)
class FrozenDesignEncoder:
    """Producer-owned formula design learned exclusively from training samples."""

    contrast: ContrastSpec
    formula: str
    context_keys: tuple[str, ...]
    covariates: tuple[str, ...]
    categorical_covariates: tuple[str, ...]
    sample_key: str
    subject_key: str
    training_sample_ids: tuple[str, ...]
    training_subject_ids: tuple[str, ...]
    covariate_encodings: tuple[FrozenCovariateEncoding, ...]
    factor_levels: tuple[tuple[str, tuple[Hashable, ...]], ...]
    formula_column_ids: tuple[str, ...]
    nuisance_column_ids: tuple[str, ...]
    training_nuisance_matrix: np.ndarray
    training_context_regressor: np.ndarray
    training_sample_manifest_digest: str
    training_design_digest: str
    coefficient_contrast_digest: str
    reparameterization_digest: str
    context_regressor_id: str
    nuisance_design_id: str
    encoder_id: str
    _null_basis: np.ndarray
    _contrast_direction: np.ndarray
    _design_info: Any
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenDesignEncoder is producer-owned; use fit_frozen_design_encoder()"
        )

    @classmethod
    def _from_training(
        cls,
        *,
        contrast: ContrastSpec,
        formula: str,
        context_keys: tuple[str, ...],
        covariates: tuple[str, ...],
        categorical_covariates: tuple[str, ...],
        sample_key: str,
        subject_key: str,
        training_sample_ids: tuple[str, ...],
        training_subject_ids: tuple[str, ...],
        covariate_encodings: tuple[FrozenCovariateEncoding, ...],
        factor_levels: tuple[tuple[str, tuple[Hashable, ...]], ...],
        formula_column_ids: tuple[str, ...],
        nuisance_column_ids: tuple[str, ...],
        training_nuisance_matrix: np.ndarray,
        training_context_regressor: np.ndarray,
        full_design_matrix: np.ndarray,
        coefficient_contrast: np.ndarray,
        null_basis: np.ndarray,
        contrast_direction: np.ndarray,
        training_sample_manifest: tuple[dict[str, object], ...],
        design_info: Any,
    ) -> FrozenDesignEncoder:
        nuisance = _immutable_array(training_nuisance_matrix)
        regressor = _immutable_array(training_context_regressor)
        full_design = np.asarray(full_design_matrix, dtype=np.float64)
        coefficient = _immutable_array(coefficient_contrast)
        frozen_null = _immutable_array(null_basis)
        frozen_direction = _immutable_array(contrast_direction)
        n_samples = len(training_sample_ids)
        n_columns = len(formula_column_ids)
        if nuisance.shape != (n_samples, len(nuisance_column_ids)):
            raise ValueError("training nuisance matrix has incompatible shape")
        if regressor.shape != (n_samples,):
            raise ValueError("training context regressor has incompatible shape")
        if full_design.shape != (n_samples, n_columns):
            raise ValueError("training formula design has incompatible shape")
        if coefficient.shape != (n_columns,):
            raise ValueError("coefficient contrast has incompatible shape")
        if frozen_null.shape != (n_columns, n_columns - 1):
            raise ValueError("contrast-null basis has incompatible shape")
        if frozen_direction.shape != (n_columns,):
            raise ValueError("contrast direction has incompatible shape")
        if tuple(map(str, design_info.column_names)) != formula_column_ids:
            raise ValueError("Patsy design information does not match formula columns")
        transformed = np.column_stack((nuisance, regressor))
        if np.linalg.matrix_rank(transformed) != n_columns:
            raise ValueError("reparameterized training design is rank deficient")
        if not np.allclose(
            coefficient @ frozen_null,
            0.0,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("nuisance basis is not in the contrast null space")
        if not math.isclose(
            float(coefficient @ frozen_direction),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError("contrast direction does not identify the EMM contrast")

        manifest_digest = canonical_digest(training_sample_manifest)
        training_design_digest = _array_digest(full_design)
        coefficient_digest = _array_digest(coefficient)
        reparameterization_digest = canonical_digest(
            {
                "contrast_direction": _array_digest(frozen_direction),
                "null_basis": _array_digest(frozen_null),
            }
        )
        common_identity = {
            "categorical_covariates": list(categorical_covariates),
            "coefficient_contrast_digest": coefficient_digest,
            "contrast": contrast.to_dict(),
            "context_keys": list(context_keys),
            "covariates": list(covariates),
            "factor_levels": [
                {"name": name, "levels": list(levels)}
                for name, levels in factor_levels
            ],
            "formula": formula,
            "formula_column_ids": list(formula_column_ids),
            "reparameterization_digest": reparameterization_digest,
            "sample_key": sample_key,
            "subject_key": subject_key,
            "training_design_digest": training_design_digest,
            "training_sample_manifest_digest": manifest_digest,
        }
        context_regressor_id = stable_id(
            "frozen_context_regressor",
            {
                **common_identity,
                "training_context_regressor_digest": _array_digest(regressor),
            },
            schema_version="2",
        )
        nuisance_design_id = stable_id(
            "frozen_nuisance_design",
            {
                **common_identity,
                "covariate_encodings": [
                    encoding.to_dict() for encoding in covariate_encodings
                ],
                "nuisance_column_ids": list(nuisance_column_ids),
                "training_nuisance_digest": _array_digest(nuisance),
            },
            schema_version="2",
        )
        encoder_id = stable_id(
            "frozen_design_encoder",
            {
                "context_regressor_id": context_regressor_id,
                "nuisance_design_id": nuisance_design_id,
            },
            schema_version="2",
        )
        self = object.__new__(cls)
        values: dict[str, Any] = {
            "contrast": contrast,
            "formula": formula,
            "context_keys": context_keys,
            "covariates": covariates,
            "categorical_covariates": categorical_covariates,
            "sample_key": sample_key,
            "subject_key": subject_key,
            "training_sample_ids": training_sample_ids,
            "training_subject_ids": training_subject_ids,
            "covariate_encodings": covariate_encodings,
            "factor_levels": factor_levels,
            "formula_column_ids": formula_column_ids,
            "nuisance_column_ids": nuisance_column_ids,
            "training_nuisance_matrix": nuisance,
            "training_context_regressor": regressor,
            "training_sample_manifest_digest": manifest_digest,
            "training_design_digest": training_design_digest,
            "coefficient_contrast_digest": coefficient_digest,
            "reparameterization_digest": reparameterization_digest,
            "context_regressor_id": context_regressor_id,
            "nuisance_design_id": nuisance_design_id,
            "encoder_id": encoder_id,
            "_null_basis": frozen_null,
            "_contrast_direction": frozen_direction,
            "_design_info": design_info,
            "_producer_marker": _PRODUCER_MARKER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        return self

    def _require_producer_owned(self) -> None:
        if self._producer_marker != _PRODUCER_MARKER:
            raise TypeError("FrozenDesignEncoder was not produced by this workflow")

    def to_dict(self) -> dict[str, object]:
        """Return manifest provenance without expanding training matrices."""

        return {
            "encoder_id": self.encoder_id,
            "context_regressor_id": self.context_regressor_id,
            "nuisance_design_id": self.nuisance_design_id,
            "contrast": self.contrast.to_dict(),
            "formula": self.formula,
            "context_keys": list(self.context_keys),
            "covariates": list(self.covariates),
            "categorical_covariates": list(self.categorical_covariates),
            "sample_key": self.sample_key,
            "subject_key": self.subject_key,
            "training_sample_ids": list(self.training_sample_ids),
            "training_subject_ids": list(self.training_subject_ids),
            "training_sample_manifest_digest": self.training_sample_manifest_digest,
            "training_design_digest": self.training_design_digest,
            "coefficient_contrast_digest": self.coefficient_contrast_digest,
            "reparameterization_digest": self.reparameterization_digest,
            "covariate_encodings": [
                encoding.to_dict() for encoding in self.covariate_encodings
            ],
            "factor_levels": [
                {"name": name, "levels": list(levels)}
                for name, levels in self.factor_levels
            ],
            "formula_column_ids": list(self.formula_column_ids),
            "nuisance_column_ids": list(self.nuisance_column_ids),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenDesignApplication:
    """One frozen held-out design encoding or an explicit unavailable result."""

    encoder_id: str
    sample_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    nuisance_matrix: np.ndarray
    context_regressor: np.ndarray
    status: str
    reason_code: str | None

    def __post_init__(self) -> None:
        if not self.encoder_id or not self.sample_ids or not self.subject_ids:
            raise ValueError("design application IDs must be non-empty")
        if self.status not in {"observed", "not_estimable"}:
            raise ValueError("design application status is invalid")
        if (self.status == "observed") == (self.reason_code is not None):
            raise ValueError("reason_code is required exactly when not estimable")
        matrix = np.asarray(self.nuisance_matrix, dtype=np.float64).copy()
        regressor = np.asarray(self.context_regressor, dtype=np.float64).copy()
        if matrix.shape[0] != len(self.sample_ids) or regressor.shape != (
            len(self.sample_ids),
        ):
            raise ValueError("encoded held-out rows must align with sample IDs")
        if self.status == "observed":
            if np.any(~np.isfinite(matrix)) or np.any(~np.isfinite(regressor)):
                raise ValueError("observed held-out design must be finite")
        elif not np.isnan(matrix).all() or not np.isnan(regressor).all():
            raise ValueError("not-estimable held-out design must contain only NaN")
        matrix.setflags(write=False)
        regressor.setflags(write=False)
        object.__setattr__(self, "nuisance_matrix", matrix)
        object.__setattr__(self, "context_regressor", regressor)


def _sample_table(
    metadata: pd.DataFrame,
    *,
    context_keys: tuple[str, ...],
    covariates: tuple[str, ...],
    sample_key: str,
    subject_key: str,
) -> pd.DataFrame:
    required = tuple(
        dict.fromkeys((sample_key, subject_key, *context_keys, *covariates))
    )
    missing = set(required).difference(metadata.columns)
    if missing:
        raise ValueError(f"sample metadata is missing columns: {sorted(missing)}")
    table = metadata.loc[:, list(required)].copy(deep=True)
    if table.empty or table.isna().any().any():
        raise ValueError("sample design metadata must be non-empty and complete")
    for key in (sample_key, subject_key):
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in table[key]
        ):
            raise ValueError(f"{key} must contain non-empty canonical strings")
    for sample_id, group in table.groupby(sample_key, observed=True, sort=False):
        for column in (subject_key, *context_keys, *covariates):
            if len({_typed_key(value) for value in group[column]}) != 1:
                raise ValueError(
                    f"sample {sample_id!r} maps to multiple {column!r} values"
                )
    table = table.drop_duplicates(sample_key, keep="first")
    table["__sample_order"] = table[sample_key].map(_typed_key)
    return (
        table.sort_values("__sample_order", kind="stable")
        .drop(columns="__sample_order")
        .reset_index(drop=True)
    )


def _fit_type_registry(
    table: pd.DataFrame,
    *,
    context_keys: tuple[str, ...],
    covariates: tuple[str, ...],
    categorical_covariates: tuple[str, ...],
) -> tuple[
    tuple[FrozenCovariateEncoding, ...],
    tuple[tuple[str, tuple[Hashable, ...]], ...],
]:
    categorical = set(categorical_covariates)
    factor_levels: list[tuple[str, tuple[Hashable, ...]]] = []
    for name in context_keys:
        factor_levels.append((name, _levels(table[name])))
    encodings: list[FrozenCovariateEncoding] = []
    for name in covariates:
        series = table[name]
        if _is_categorical(series, declared=name in categorical):
            levels = _levels(series)
            factor_levels.append((name, levels))
            encodings.append(
                FrozenCovariateEncoding(
                    name=name,
                    kind="categorical",
                    levels=levels,
                    column_ids=tuple(
                        f"{name}:"
                        + stable_id(
                            "design_level", {"value": _typed_key(level)}
                        )
                        for level in levels[1:]
                    ),
                )
            )
            continue
        numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(numeric)):
            raise ValueError(f"numeric covariate {name!r} must be finite")
        center = float(np.mean(numeric))
        scale = float(np.std(numeric, ddof=0))
        if scale <= _NUMERIC_SCALE_FLOOR:
            encodings.append(
                FrozenCovariateEncoding(
                    name=name,
                    kind="constant_continuous",
                    center=center,
                    scale=1.0,
                )
            )
        else:
            encodings.append(
                FrozenCovariateEncoding(
                    name=name,
                    kind="continuous",
                    center=center,
                    scale=scale,
                    column_ids=(f"{name}:raw_formula_v1",),
                )
            )
    return tuple(encodings), tuple(factor_levels)


def _contrast_reparameterization(
    coefficient_contrast: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coefficient = np.asarray(coefficient_contrast, dtype=np.float64)
    if coefficient.ndim != 1 or np.any(~np.isfinite(coefficient)):
        raise ValueError("coefficient contrast must be a finite vector")
    norm_squared = float(coefficient @ coefficient)
    if norm_squared <= _NUMERIC_SCALE_FLOOR:
        raise ValueError("coefficient contrast must be non-zero")
    direction = coefficient / norm_squared
    pivot = int(np.argmax(np.abs(coefficient)))
    pivot_value = float(coefficient[pivot])
    columns: list[np.ndarray] = []
    for index in range(len(coefficient)):
        if index == pivot:
            continue
        column: np.ndarray = np.zeros(len(coefficient), dtype=np.float64)
        column[index] = 1.0
        column[pivot] = -float(coefficient[index]) / pivot_value
        column /= float(np.linalg.norm(column))
        columns.append(column)
    null_basis = (
        np.column_stack(columns)
        if columns
        else np.empty((len(coefficient), 0), dtype=np.float64)
    )
    return null_basis, direction


def _training_manifest(
    table: pd.DataFrame,
    *,
    context_keys: tuple[str, ...],
    covariates: tuple[str, ...],
    sample_key: str,
    subject_key: str,
) -> tuple[dict[str, object], ...]:
    fields = (*context_keys, *covariates)
    return tuple(
        {
            "sample_id": _typed_key(row[sample_key]),
            "subject_id": _typed_key(row[subject_key]),
            "design_values": [
                {"field": name, "value": _typed_key(row[name])} for name in fields
            ],
        }
        for _, row in table.iterrows()
    )


def fit_frozen_design_encoder(
    sample_metadata: pd.DataFrame,
    *,
    contrast: ContrastSpec,
    context_keys: Sequence[str],
    covariates: Sequence[str] = (),
    categorical_covariates: Sequence[str] = (),
    formula: str | None = None,
    sample_key: str = "sample_id",
    subject_key: str = "subject_id",
) -> FrozenDesignEncoder:
    """Fit one full formula and isolate its declared EMM contrast coefficient."""

    if not isinstance(contrast, ContrastSpec):
        raise TypeError("contrast must be a ContrastSpec")
    contexts = _names(context_keys, field_name="context_keys")
    covariate_names = _names(covariates, field_name="covariates")
    categorical_names = _names(
        categorical_covariates, field_name="categorical_covariates"
    )
    if not contexts:
        raise ValueError("context_keys must not be empty")
    if set(contexts).intersection(covariate_names):
        raise ValueError("context and covariate names must be disjoint")
    if set(categorical_names).difference(covariate_names):
        raise ValueError("categorical_covariates must be declared covariates")
    for name, value in (("sample_key", sample_key), ("subject_key", subject_key)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be non-empty")
    table = _sample_table(
        sample_metadata,
        context_keys=contexts,
        covariates=covariate_names,
        sample_key=sample_key,
        subject_key=subject_key,
    )
    declared_formula = formula or default_design_formula(contexts, covariate_names)
    audit = audit_sample_design(
        table,
        context_keys=contexts,
        covariates=covariate_names,
        categorical_covariates=categorical_names,
        formula=declared_formula,
        sample_key=sample_key,
    )
    if not audit.ready:
        raise ValueError(
            "training_design_not_estimable:" + ",".join(audit.reason_codes)
        )
    if not audit.contrast_estimable(contrast):
        raise ValueError("training_contrast_not_estimable")
    full_design = audit.design_matrix.to_numpy(dtype=np.float64)
    coefficient = audit.coefficient_contrast(contrast)
    null_basis, direction = _contrast_reparameterization(coefficient)
    nuisance = full_design @ null_basis
    regressor = full_design @ direction
    encodings, factor_levels = _fit_type_registry(
        table,
        context_keys=contexts,
        covariates=covariate_names,
        categorical_covariates=categorical_names,
    )
    nuisance_ids = tuple(
        f"contrast_null:{index:04d}:{_array_digest(null_basis[:, index])[:16]}"
        for index in range(null_basis.shape[1])
    )
    return FrozenDesignEncoder._from_training(
        contrast=contrast,
        formula=declared_formula,
        context_keys=contexts,
        covariates=covariate_names,
        categorical_covariates=categorical_names,
        sample_key=sample_key,
        subject_key=subject_key,
        training_sample_ids=tuple(table[sample_key]),
        training_subject_ids=tuple(sorted(table[subject_key].unique())),
        covariate_encodings=encodings,
        factor_levels=factor_levels,
        formula_column_ids=audit.column_names,
        nuisance_column_ids=nuisance_ids,
        training_nuisance_matrix=nuisance,
        training_context_regressor=regressor,
        full_design_matrix=full_design,
        coefficient_contrast=coefficient,
        null_basis=null_basis,
        contrast_direction=direction,
        training_sample_manifest=_training_manifest(
            table,
            context_keys=contexts,
            covariates=covariate_names,
            sample_key=sample_key,
            subject_key=subject_key,
        ),
        design_info=audit.design_info,
    )


def _not_estimable(
    encoder: FrozenDesignEncoder,
    table: pd.DataFrame,
    *,
    reason_code: str,
) -> FrozenDesignApplication:
    shape = (len(table), len(encoder.nuisance_column_ids))
    return FrozenDesignApplication(
        encoder_id=encoder.encoder_id,
        sample_ids=tuple(table[encoder.sample_key]),
        subject_ids=tuple(sorted(table[encoder.subject_key].unique())),
        nuisance_matrix=np.full(shape, np.nan, dtype=np.float64),
        context_regressor=np.full(len(table), np.nan, dtype=np.float64),
        status="not_estimable",
        reason_code=reason_code,
    )


def _heldout_formula_table(
    encoder: FrozenDesignEncoder,
    table: pd.DataFrame,
) -> tuple[pd.DataFrame | None, str | None]:
    result = table.loc[:, [*encoder.context_keys, *encoder.covariates]].copy(deep=True)
    level_registry = dict(encoder.factor_levels)
    encoding_registry = {
        encoding.name: encoding for encoding in encoder.covariate_encodings
    }
    for name, levels in encoder.factor_levels:
        known_by_key = {_typed_key(level): level for level in levels}
        observed_keys = result[name].map(_typed_key)
        if not set(observed_keys).issubset(known_by_key):
            return None, f"unseen_heldout_level:{name}"
        canonical_values = observed_keys.map(known_by_key)
        result[name] = pd.Categorical(canonical_values, categories=list(levels))
    for name in encoder.covariates:
        if name in level_registry:
            continue
        encoding = encoding_registry[name]
        numeric = pd.to_numeric(result[name], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if np.any(~np.isfinite(numeric)):
            return None, f"invalid_heldout_numeric:{name}"
        if encoding.kind == "constant_continuous":
            assert encoding.center is not None
            if not np.allclose(
                numeric,
                encoding.center,
                rtol=0.0,
                atol=_NUMERIC_SCALE_FLOOR,
            ):
                return (
                    None,
                    f"heldout_value_outside_constant_training_support:{name}",
                )
        result[name] = numeric
    return result, None


def apply_frozen_design_encoder(
    encoder: FrozenDesignEncoder,
    sample_metadata: pd.DataFrame,
) -> FrozenDesignApplication:
    """Apply the training formula and fixed reparameterization without refitting."""

    if not isinstance(encoder, FrozenDesignEncoder):
        raise TypeError("encoder must be a FrozenDesignEncoder")
    encoder._require_producer_owned()
    table = _sample_table(
        sample_metadata,
        context_keys=encoder.context_keys,
        covariates=encoder.covariates,
        sample_key=encoder.sample_key,
        subject_key=encoder.subject_key,
    )
    overlap = set(table[encoder.subject_key]).intersection(encoder.training_subject_ids)
    if overlap:
        raise ValueError(
            "held-out design overlaps training subjects: " + ", ".join(sorted(overlap))
        )
    formula_table, reason = _heldout_formula_table(encoder, table)
    if reason is not None or formula_table is None:
        return _not_estimable(encoder, table, reason_code=str(reason))
    try:
        full_design = np.asarray(
            patsy.build_design_matrices(
                [encoder._design_info], formula_table, return_type="dataframe"
            )[0],
            dtype=np.float64,
        )
    except (patsy.PatsyError, ValueError) as error:
        return _not_estimable(
            encoder,
            table,
            reason_code=f"heldout_formula_encoding_failed:{type(error).__name__}",
        )
    if full_design.shape != (len(table), len(encoder.formula_column_ids)):
        raise RuntimeError("frozen formula encoder emitted the wrong matrix shape")
    nuisance = full_design @ encoder._null_basis
    regressor = full_design @ encoder._contrast_direction
    return FrozenDesignApplication(
        encoder_id=encoder.encoder_id,
        sample_ids=tuple(table[encoder.sample_key]),
        subject_ids=tuple(sorted(table[encoder.subject_key].unique())),
        nuisance_matrix=nuisance,
        context_regressor=regressor,
        status="observed",
        reason_code=None,
    )


__all__ = [
    "FrozenCovariateEncoding",
    "FrozenDesignApplication",
    "FrozenDesignEncoder",
    "apply_frozen_design_encoder",
    "fit_frozen_design_encoder",
]
