from __future__ import annotations

import numpy as np
import pytest

from crychic.attribution import (
    PrecisionTransformResult,
    winsorized_normalized_precision,
)
from crychic.core import ContractError


def _fit(raw: np.ndarray, **kwargs: object) -> PrecisionTransformResult:
    parameters: dict[str, object] = {
        "feature_ids": tuple(f"G{index}" for index in range(len(raw))),
        "receiver": "Receiver",
        "contrast_name": "stim_vs_ctrl",
        "fold_id": "fold-1",
    }
    parameters.update(kwargs)
    return winsorized_normalized_precision(raw, **parameters)  # type: ignore[arg-type]


def test_precision_is_winsorized_and_positive_median_normalized() -> None:
    result = _fit(
        np.asarray([1.0, 4.0, 100.0, np.nan, 0.0]),
        lower_quantile=0.0,
        upper_quantile=1.0,
    )

    np.testing.assert_allclose(result.values, [0.25, 1.0, 25.0, 0.0, 0.0])
    assert np.median(result.values[result.values > 0]) == pytest.approx(1.0)
    assert result.n_positive_features == 3
    assert result.estimable
    assert result.reason_code is None


def test_precision_transform_records_insufficient_support() -> None:
    result = _fit(np.asarray([np.nan, 2.0, 0.0]), min_positive_features=2)

    np.testing.assert_allclose(result.values, [0.0, 1.0, 0.0])
    assert not result.estimable
    assert result.reason_code == "insufficient_response_precision_support"


def test_precision_transform_id_changes_with_fitted_bounds() -> None:
    first = _fit(np.asarray([1.0, 2.0, 100.0]))
    second = _fit(np.asarray([1.0, 2.0, 10.0]))

    assert first.precision_transform_id != second.precision_transform_id


def test_precision_quantile_types_are_canonicalized() -> None:
    raw = np.asarray([1.0, 2.0, 3.0])
    python = _fit(raw, lower_quantile=0.0, upper_quantile=1.0)
    numpy = _fit(
        raw,
        lower_quantile=np.int64(0),
        upper_quantile=np.float32(1.0),
    )

    assert numpy.precision_transform_id == python.precision_transform_id
    assert type(numpy.lower_quantile) is float
    assert type(numpy.upper_quantile) is float


def test_precision_quantiles_reject_boolean_values() -> None:
    with pytest.raises(ValueError, match="not boolean"):
        _fit(
            np.asarray([1.0, 2.0]),
            lower_quantile=False,
            upper_quantile=True,
        )


def test_precision_preserves_exact_whitespace_bearing_identifiers() -> None:
    result = winsorized_normalized_precision(
        np.asarray([1.0, 2.0]),
        feature_ids=(" G1 ", "G2"),
        receiver=" Receiver ",
        contrast_name=" contrast ",
        fold_id=" fold ",
    )

    assert result.feature_ids == (" G1 ", "G2")
    assert result.receiver == " Receiver "


def test_precision_transform_id_binds_values_when_summary_bounds_match() -> None:
    first = _fit(
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        lower_quantile=0.0,
        upper_quantile=1.0,
    )
    second = _fit(
        np.asarray([1.0, 2.5, 3.0, 3.5, 5.0]),
        lower_quantile=0.0,
        upper_quantile=1.0,
    )

    assert (first.lower_bound, first.upper_bound, first.normalization_median) == (
        second.lower_bound,
        second.upper_bound,
        second.normalization_median,
    )
    assert first.precision_transform_id != second.precision_transform_id
    assert first.raw_precision_digest != second.raw_precision_digest
    assert first.transformed_precision_digest != second.transformed_precision_digest


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("feature_ids", ("X0", "G1", "G2")),
        ("receiver", "OtherReceiver"),
        ("contrast_name", "other_contrast"),
        ("fold_id", "fold-2"),
    ],
)
def test_precision_transform_id_binds_scope(
    field_name: str, replacement: object
) -> None:
    raw = np.asarray([1.0, 2.0, 3.0])
    first = _fit(raw)
    second = _fit(raw, **{field_name: replacement})

    assert first.precision_transform_id != second.precision_transform_id


def test_precision_values_are_backed_by_immutable_storage() -> None:
    result = _fit(np.asarray([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        result.values.setflags(write=True)
    with pytest.raises(ValueError, match="read-only"):
        result.values[0] = 99.0


def test_precision_transform_detects_post_serialization_mutation() -> None:
    result = _fit(np.asarray([1.0, 2.0, 3.0]))
    object.__setattr__(result, "values", result.values.copy())
    result.values[0] = 99.0

    with pytest.raises(ContractError) as error:
        result.to_dict()

    assert error.value.details.code == "precision_transform_integrity_violation"


def test_precision_transform_is_producer_owned() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        PrecisionTransformResult()


def test_precision_transform_rejects_feature_misalignment() -> None:
    with pytest.raises(ValueError, match="align"):
        winsorized_normalized_precision(
            np.asarray([1.0, 2.0]),
            feature_ids=("G1",),
            receiver="Receiver",
            contrast_name="stim_vs_ctrl",
            fold_id="fold-1",
        )


def test_precision_transform_rejects_incompatible_consumption_scope() -> None:
    result = _fit(np.asarray([1.0, 2.0, 3.0]))

    with pytest.raises(ContractError) as error:
        result.require_compatible(
            feature_ids=("G1", "G0", "G2"),
            receiver="Receiver",
            contrast_name="stim_vs_ctrl",
            fold_id="fold-1",
        )

    assert error.value.details.code == "precision_transform_scope_mismatch"


@pytest.mark.parametrize("minimum", [True, 1.5, 0])
def test_precision_transform_rejects_invalid_minimum_support(
    minimum: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _fit(
            np.asarray([1.0, 2.0]),
            min_positive_features=minimum,
        )
