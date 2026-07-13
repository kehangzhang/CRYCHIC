from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from benchmarks.metrics.repeated_measures import (
    FrozenRepeatedMeasuresDesign,
    RepeatedMeasuresSpec,
    fit_repeated_measures_contrast,
    freeze_repeated_measures_design,
    repeated_measures_effects,
)


def _mixed_design() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for index in range(4):
        subject = f"p{index}"
        for context in ("control", "case"):
            rows.append(
                {
                    "sample_id": f"{subject}-{context}",
                    "subject_id": subject,
                    "context": context,
                }
            )
    for index in range(4):
        rows.extend(
            [
                {
                    "sample_id": f"c{index}",
                    "subject_id": f"c{index}",
                    "context": "control",
                },
                {
                    "sample_id": f"t{index}",
                    "subject_id": f"t{index}",
                    "context": "case",
                },
            ]
        )
    return pd.DataFrame(rows)


def _scores(design: pd.DataFrame, *, effect: float = 0.25) -> pd.DataFrame:
    values: list[dict[str, object]] = []
    for row in design.itertuples(index=False):
        subject_index = int(str(row.subject_id)[1:])
        baseline = 0.1 + 0.01 * subject_index
        values.append(
            {
                "sample_id": row.sample_id,
                "value": baseline + (effect if row.context == "case" else 0.0),
            }
        )
    return pd.DataFrame(values)


def _spec(**kwargs: object) -> RepeatedMeasuresSpec:
    return RepeatedMeasuresSpec(
        context_key="context",
        reference="control",
        target="case",
        min_subjects_per_context=3,
        min_subject_clusters=6,
        **kwargs,
    )


def test_mixed_paired_and_unpaired_subjects_contribute_to_one_effect() -> None:
    sample_design = _mixed_design()
    design = freeze_repeated_measures_design(sample_design, spec=_spec())
    fit = fit_repeated_measures_contrast(design, _scores(sample_design))

    assert design.estimable
    assert fit.status == "exploratory"
    assert fit.reason_code is None
    assert fit.design_kind == "mixed_paired_unpaired_subject_cluster_ols"
    assert fit.effect == pytest.approx(0.25)
    assert np.isfinite(fit.diagnostic_standard_error)
    assert fit.n_reference_subjects == 8
    assert fit.n_target_subjects == 8
    assert fit.n_paired_subjects == 4
    assert fit.n_subject_clusters == 12
    assert fit.n_contrast_subject_clusters == 12
    assert fit.n_repeated_subject_clusters == 4


def test_replicate_samples_are_averaged_within_subject_design_cell() -> None:
    sample_design = _mixed_design()
    duplicate = sample_design.loc[
        sample_design["sample_id"].eq("p0-control")
    ].copy()
    duplicate["sample_id"] = "p0-control-replicate"
    augmented_design = pd.concat([sample_design, duplicate], ignore_index=True)
    scores = _scores(sample_design)
    scores = pd.concat(
        [
            scores,
            pd.DataFrame(
                [{"sample_id": "p0-control-replicate", "value": 0.2}]
            ),
        ],
        ignore_index=True,
    )
    scores.loc[scores["sample_id"].eq("p0-control"), "value"] = 0.0
    design = freeze_repeated_measures_design(augmented_design, spec=_spec())
    fit = fit_repeated_measures_contrast(design, scores)

    # The two p0/control values average back to the original baseline 0.1.
    assert fit.effect == pytest.approx(0.25)
    assert fit.n_observations == len(sample_design)


def test_separate_region_and_batch_adjustments_are_explicit() -> None:
    rows: list[dict[str, object]] = []
    for context in ("control", "case"):
        for index in range(12):
            region = "core" if index % 2 else "border"
            batch = "b2" if index % 3 else "b1"
            rows.append(
                {
                    "sample_id": f"{context}-{index}",
                    "subject_id": f"{context}-{index}",
                    "context": context,
                    "region": region,
                    "batch": batch,
                    "value": (
                        0.2
                        + (0.3 if context == "case" else 0.0)
                        + (0.4 if region == "core" else 0.0)
                        + (0.1 if batch == "b2" else 0.0)
                    ),
                }
            )
    table = pd.DataFrame(rows)
    spec = _spec(region_key="region", batch_keys=("batch",))
    design = freeze_repeated_measures_design(table, spec=spec)
    fit = fit_repeated_measures_contrast(design, table)

    assert design.estimable
    assert "adjustment:region=core" in design.model_columns
    assert "adjustment:batch=b2" in design.model_columns
    assert fit.effect == pytest.approx(0.3)


def test_region_may_be_the_context_for_kuppe_style_pairwise_contrast() -> None:
    sample_design = _mixed_design().rename(columns={"context": "region"})
    spec = RepeatedMeasuresSpec(
        context_key="region",
        region_key="region",
        reference="control",
        target="case",
        min_subjects_per_context=3,
        min_subject_clusters=6,
    )
    design = freeze_repeated_measures_design(sample_design, spec=spec)
    scores = _scores(_mixed_design())
    fit = fit_repeated_measures_contrast(design, scores)

    assert design.factor_levels[0][0] == "region"
    assert not any(
        column.startswith("adjustment:region") for column in design.model_columns
    )
    assert fit.effect == pytest.approx(0.25)


def test_kuppe_shape_retains_partial_pairs_and_collapses_replicates() -> None:
    subject_regions = {
        "p2": ("RZ", "BZ"),
        "p3": ("BZ", "IZ", "FZ"),
        "p9": ("RZ", "BZ", "IZ", "FZ"),
        "rz1": ("RZ",),
        "rz2": ("RZ",),
        "bz1": ("BZ",),
        "bz2": ("BZ",),
        "ctrl1": ("CTRL",),
        "ctrl2": ("CTRL",),
    }
    rows: list[dict[str, object]] = []
    region_effect = {"CTRL": -0.1, "RZ": 0.1, "BZ": 0.3, "IZ": 0.4, "FZ": 0.5}
    for subject, regions in subject_regions.items():
        for region in regions:
            rows.append(
                {
                    "sample_id": f"{subject}-{region}",
                    "subject_id": subject,
                    "region": region,
                    "value": 0.2 + region_effect[region],
                }
            )
    rows.append(
        {
            "sample_id": "p9-BZ-replicate",
            "subject_id": "p9",
            "region": "BZ",
            "value": 0.5,
        }
    )
    table = pd.DataFrame(rows)
    spec = RepeatedMeasuresSpec(
        context_key="region",
        region_key="region",
        reference="RZ",
        target="BZ",
        min_subjects_per_context=3,
        min_subject_clusters=6,
    )
    design = freeze_repeated_measures_design(table, spec=spec)
    fit = fit_repeated_measures_contrast(design, table)

    assert design.model_columns == (
        "intercept",
        "context:region=BZ",
        "context:region=CTRL",
        "context:region=FZ",
        "context:region=IZ",
    )
    assert fit.status == "exploratory"
    assert fit.design_kind == "mixed_paired_unpaired_subject_cluster_ols"
    assert fit.effect == pytest.approx(0.2)
    assert fit.n_paired_subjects == 2
    assert fit.n_observations == len(table) - 1


def test_context_batch_confounding_fails_closed_without_effect_or_se() -> None:
    sample_design = _mixed_design()
    sample_design["batch"] = np.where(
        sample_design["context"].eq("control"), "b1", "b2"
    )
    spec = _spec(batch_keys=("batch",))
    design = freeze_repeated_measures_design(sample_design, spec=spec)
    fit = fit_repeated_measures_contrast(design, _scores(sample_design))

    assert not design.estimable
    assert design.reason_code == "rank_deficient_declared_design"
    assert fit.status == "not_estimable"
    assert fit.reason_code == "rank_deficient_declared_design"
    assert np.isnan(fit.effect)
    assert np.isnan(fit.diagnostic_standard_error)


def test_edge_specific_missing_region_fails_rank_check() -> None:
    sample_design = _mixed_design()
    extra = pd.DataFrame(
        [
            {
                "sample_id": f"other-{index}",
                "subject_id": f"other-{index}",
                "context": "other",
            }
            for index in range(4)
        ]
    )
    sample_design = pd.concat([sample_design, extra], ignore_index=True)
    design = freeze_repeated_measures_design(sample_design, spec=_spec())
    scores = _scores(sample_design.loc[sample_design["context"].ne("other")])
    fit = fit_repeated_measures_contrast(design, scores)

    assert design.estimable
    assert fit.status == "not_estimable"
    assert fit.reason_code == "rank_deficient_edge_design"
    assert np.isnan(fit.effect)


def test_grouped_backend_reuses_design_and_never_emits_p_or_q() -> None:
    sample_design = _mixed_design()
    base = _scores(sample_design)
    table = pd.concat(
        [
            base.assign(method="m1", edge="e1", status="observed"),
            base.assign(method="m2", edge="e1", status="observed"),
            base.assign(method="m1", edge="e2", status="missing"),
        ],
        ignore_index=True,
    )
    result = repeated_measures_effects(
        table,
        sample_design,
        spec=_spec(),
        group_keys=("method", "edge"),
        status_key="status",
    )

    observed = result.loc[result["edge"].eq("e1")]
    unavailable = result.loc[result["edge"].eq("e2")].iloc[0]
    assert set(observed["status"]) == {"exploratory"}
    assert observed["repeated_measures_design_id"].nunique() == 1
    assert unavailable["status"] == "not_estimable"
    assert unavailable["reason_code"] == "insufficient_subjects_per_context"
    assert result["formal_inference_allowed"].eq(False).all()
    assert not {"p", "p_value", "q", "q_value"}.intersection(result.columns)


def test_design_identifier_is_invariant_to_sample_row_order() -> None:
    sample_design = _mixed_design()
    first = freeze_repeated_measures_design(sample_design, spec=_spec())
    second = freeze_repeated_measures_design(
        sample_design.sample(frac=1.0, random_state=17), spec=_spec()
    )

    assert first.design_id == second.design_id


def test_frozen_design_copies_input_and_uses_readonly_arrays() -> None:
    sample_design = _mixed_design()
    expected = sample_design.copy(deep=True)
    design = freeze_repeated_measures_design(sample_design, spec=_spec())
    sample_design.loc[:, "context"] = "poison"

    assert set(design.sample_table["context"]) == set(expected["context"])
    assert not design.design_matrix.flags.writeable
    assert not design.contrast_vector.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        design.design_matrix[0, 0] = 7.0
    with pytest.raises(ValueError, match="read-only"):
        design.contrast_vector[1] = -3.0
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        design.design_matrix.setflags(write=True)


@pytest.mark.parametrize("table_name", ["sample_table", "cell_table"])
def test_public_table_mutation_fails_closed_before_fit(table_name: str) -> None:
    sample_design = _mixed_design()
    design = freeze_repeated_measures_design(sample_design, spec=_spec())
    original_id = design.design_id
    table = getattr(design, table_name)
    table.loc[table.index[0], "context"] = "poison"

    fit = fit_repeated_measures_contrast(design, _scores(sample_design))

    assert design.design_id == original_id
    assert fit.status == "not_estimable"
    assert fit.reason_code == "frozen_design_integrity_violation"
    assert np.isnan(fit.effect)
    assert np.isnan(fit.diagnostic_standard_error)


def test_public_table_structure_mutation_fails_closed_before_fit() -> None:
    sample_design = _mixed_design()
    design = freeze_repeated_measures_design(sample_design, spec=_spec())
    design.sample_table.drop(columns="context", inplace=True)

    fit = fit_repeated_measures_contrast(design, _scores(sample_design))

    assert fit.status == "not_estimable"
    assert fit.reason_code == "frozen_design_integrity_violation"
    assert np.isnan(fit.effect)


@pytest.mark.parametrize("array_name", ["design_matrix", "contrast_vector"])
def test_forced_array_replacement_fails_closed(array_name: str) -> None:
    sample_design = _mixed_design()
    design = freeze_repeated_measures_design(sample_design, spec=_spec())
    poisoned = getattr(design, array_name).copy()
    poisoned.flat[0] += 1.0
    object.__setattr__(design, array_name, poisoned)

    fit = fit_repeated_measures_contrast(design, _scores(sample_design))

    assert fit.status == "not_estimable"
    assert fit.reason_code == "frozen_design_integrity_violation"
    assert np.isnan(fit.effect)


def test_frozen_design_rejects_direct_construction() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        FrozenRepeatedMeasuresDesign()


@pytest.mark.parametrize(
    "missing_label",
    [pd.NaT, np.datetime64("NaT")],
    ids=["pandas_nat", "numpy_datetime64_nat"],
)
def test_design_rejects_nat_labels_before_grouping(missing_label: object) -> None:
    sample_design = _mixed_design().astype({"context": object})
    sample_design.loc[0, "context"] = missing_label

    with pytest.raises(ValueError, match="group keys must not contain missing"):
        freeze_repeated_measures_design(sample_design, spec=_spec())


@pytest.mark.parametrize("nonfinite", [np.inf, -np.inf])
def test_design_rejects_nonfinite_numeric_labels(nonfinite: float) -> None:
    sample_design = _mixed_design().astype({"context": object})
    sample_design.loc[0, "context"] = nonfinite

    with pytest.raises(ValueError, match="numeric labels must be finite"):
        freeze_repeated_measures_design(sample_design, spec=_spec())


@pytest.mark.parametrize("key", ["subject_id", "context", "batch"])
def test_design_rejects_missing_group_keys_before_grouping(key: str) -> None:
    sample_design = _mixed_design().assign(batch="b1")
    sample_design.loc[0, key] = pd.NA

    with pytest.raises(ValueError, match=rf"missing values: .*{key}"):
        freeze_repeated_measures_design(
            sample_design,
            spec=_spec(batch_keys=("batch",)),
        )


@pytest.mark.parametrize("key", ["method", "edge"])
def test_grouped_backend_rejects_missing_identity_before_grouping(key: str) -> None:
    sample_design = _mixed_design()
    table = _scores(sample_design).assign(method="m1", edge="e1")
    table.loc[0, key] = pd.NA

    with pytest.raises(ValueError, match=rf"missing values: .*{key}"):
        repeated_measures_effects(
            table,
            sample_design,
            spec=_spec(),
            group_keys=("method", "edge"),
        )


def test_rejects_repeated_group_sample_rows() -> None:
    sample_design = _mixed_design()
    table = _scores(sample_design).assign(method="m1", edge="e1")
    repeated = pd.concat([table, table.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="repeated group/sample"):
        repeated_measures_effects(
            repeated,
            sample_design,
            spec=_spec(),
            group_keys=("method", "edge"),
        )
