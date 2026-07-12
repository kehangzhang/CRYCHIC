from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from crychic.data import InputMode, InputSchema, validate_anndata
from crychic.design import (
    ContextGraph,
    audit_sample_design,
    balanced_contrast,
    factorial_interaction_contrast,
)
from crychic.pseudobulk import aggregate_pseudobulk
from crychic.response import (
    ResponseMethod,
    ResponseStatus,
    estimate_gene_response,
)


def _aggregate(
    values: Sequence[Sequence[float]],
    *,
    samples: Sequence[str],
    subjects: Sequence[str],
    contexts: Sequence[str],
    cell_types: Sequence[str] | None = None,
    counts: bool = False,
    min_cells: int = 1,
):
    array = np.asarray(values, dtype=float)
    obs = pd.DataFrame(
        {
            "sample_id": samples,
            "subject_id": subjects,
            "cell_type": cell_types or ["R"] * len(samples),
            "condition": contexts,
        },
        index=[f"cell-{index}" for index in range(len(samples))],
    )
    adata = ad.AnnData(
        X=np.zeros_like(array) if counts else array,
        obs=obs,
        var=pd.DataFrame(index=[f"G{index + 1}" for index in range(array.shape[1])]),
    )
    if counts:
        adata.layers["counts"] = array
        schema = InputSchema(context_keys=("condition",))
    else:
        schema = InputSchema(
            context_keys=("condition",),
            expression_source="published normalized expression",
            expression_transform="linear_normalized",
        )
    return aggregate_pseudobulk(validate_anndata(adata, schema), min_cells=min_cells)


def _two_context_graph() -> ContextGraph:
    return ContextGraph.chain(["control", "treated"])


def _treated_vs_control():
    return balanced_contrast(["treated"], ["control"], name="treated-v-control")


def _contrast_rows(response):
    return response.contrasts.loc[
        response.contrasts["contrast"] == "treated-v-control"
    ].set_index("gene")


def _one_factor_audit(
    samples: Sequence[str],
    contexts: Sequence[str],
    *,
    batches: Sequence[str] | None = None,
):
    metadata = pd.DataFrame(
        {
            "sample_id": samples,
            "condition": contexts,
        }
    ).drop_duplicates("sample_id", keep="first")
    covariates: tuple[str, ...] = ()
    formula = "~ condition"
    if batches is not None:
        metadata["batch"] = batches
        covariates = ("batch",)
        formula = "~ batch + condition"
    return audit_sample_design(
        metadata,
        context_keys=("condition",),
        covariates=covariates,
        formula=formula,
        sample_key="sample_id",
    )


def test_counts_are_transformed_to_log1p_cpm_before_contrast() -> None:
    aggregate = _aggregate(
        [[1, 1], [2, 2], [3, 1], [6, 2]],
        samples=["c1", "c2", "t1", "t2"],
        subjects=["p1", "p2", "p3", "p4"],
        contexts=["control", "control", "treated", "treated"],
        counts=True,
    )
    response = estimate_gene_response(
        aggregate,
        _two_context_graph(),
        contrasts=[_treated_vs_control()],
        cpm_scale=100.0,
    )

    assert response.input_mode is InputMode.COUNTS
    assert response.value_scale == "log1p_cpm"
    expected_control = np.log1p([50.0, 50.0])
    expected_treated = np.log1p([75.0, 25.0])
    rows = _contrast_rows(response)
    assert rows.loc["G1", "effect"] == pytest.approx(
        expected_treated[0] - expected_control[0]
    )
    assert rows.loc["G2", "effect"] == pytest.approx(
        expected_treated[1] - expected_control[1]
    )
    assert rows.loc["G1", "status"] == ResponseStatus.OK.value
    assert set(response.feature_ids) == set(rows.index)
    assert not {"p_value", "q_value", "p", "q"}.intersection(response.contrasts.columns)


def test_normalized_values_keep_declared_scale_and_hand_contrast() -> None:
    aggregate = _aggregate(
        [[1, 10], [3, 8], [5, 6], [7, 4]],
        samples=["c1", "c2", "t1", "t2"],
        subjects=["p1", "p2", "p3", "p4"],
        contexts=["control", "control", "treated", "treated"],
    )
    response = estimate_gene_response(
        aggregate,
        _two_context_graph(),
        contrasts=[_treated_vs_control()],
    )

    assert response.input_mode is InputMode.NORMALIZED_ONLY
    assert response.value_scale == "linear_normalized"
    assert response.reason_codes == ("normalized_only",)
    assert not response.inference_eligible
    rows = _contrast_rows(response)
    assert rows.loc["G1", "effect"] == pytest.approx(4.0)
    assert rows.loc["G2", "effect"] == pytest.approx(-4.0)
    assert rows.loc["G1", "standard_error"] == pytest.approx(np.sqrt(2.0))
    assert rows.loc["G1", "z_score"] == pytest.approx(4.0 / np.sqrt(2.0))
    assert rows.loc["G1", "method"] == ResponseMethod.INDEPENDENT.value
    assert rows.loc["G1", "n_samples"] == 4
    assert rows.loc["G1", "n_subjects"] == 4
    np.testing.assert_allclose(
        sorted(response.sample_vector("R", "G1").dropna()), [1, 3, 5, 7]
    )


def test_formula_emm_adjusts_an_unbalanced_batch_design() -> None:
    samples = ["c1", "c2", "c3", "c4", "t1", "t2", "t3", "t4"]
    contexts = ["control"] * 4 + ["treated"] * 4
    batches = ["a", "a", "a", "b", "a", "b", "b", "b"]
    aggregate = _aggregate(
        [[0], [0], [0], [10], [2], [12], [12], [12]],
        samples=samples,
        subjects=[f"p{index}" for index in range(8)],
        contexts=contexts,
    )
    audit = _one_factor_audit(samples, contexts, batches=batches)

    response = estimate_gene_response(
        aggregate,
        _two_context_graph(),
        contrasts=[_treated_vs_control()],
        design_audit=audit,
    )

    row = _contrast_rows(response).loc["G1"]
    assert row["effect"] == pytest.approx(2.0)
    assert row["effect"] != pytest.approx(7.0)
    assert row["method"] == ResponseMethod.EMM_INDEPENDENT.value
    means = response.context_means.set_index("context")["mean_response"]
    assert means["control"] == pytest.approx(5.0)
    assert means["treated"] == pytest.approx(7.0)


def test_formula_paired_estimate_matches_hand_difference_and_standard_error() -> None:
    samples = ["p1-c", "p1-t", "p2-c", "p2-t", "p3-c", "p3-t"]
    contexts = ["control", "treated"] * 3
    aggregate = _aggregate(
        [[1], [3], [2], [5], [4], [8]],
        samples=samples,
        subjects=["p1", "p1", "p2", "p2", "p3", "p3"],
        contexts=contexts,
    )
    audit = _one_factor_audit(samples, contexts)

    response = estimate_gene_response(
        aggregate,
        _two_context_graph(),
        contrasts=[_treated_vs_control()],
        design_audit=audit,
    )

    row = _contrast_rows(response).loc["G1"]
    assert row["effect"] == pytest.approx(3.0)
    assert row["standard_error"] == pytest.approx(1.0 / np.sqrt(3.0))
    assert row["method"] == ResponseMethod.EMM_PAIRED.value
    assert row["paired"]


@pytest.mark.parametrize("use_formula", [False, True])
def test_independent_response_gives_each_subject_equal_weight(
    use_formula: bool,
) -> None:
    samples = ["p1-c1", "p1-c2", "p2-c", "p3-t", "p4-t"]
    contexts = ["control", "control", "control", "treated", "treated"]
    aggregate = _aggregate(
        [[0], [10], [2], [5], [7]],
        samples=samples,
        subjects=["p1", "p1", "p2", "p3", "p4"],
        contexts=contexts,
    )
    audit = _one_factor_audit(samples, contexts)

    response = estimate_gene_response(
        aggregate,
        _two_context_graph(),
        contrasts=[_treated_vs_control()],
        design_audit=audit if use_formula else None,
    )

    row = _contrast_rows(response).loc["G1"]
    assert row["effect"] == pytest.approx(2.5)
    assert row["effect"] != pytest.approx(2.0)
    assert row["n_samples"] == 5
    assert row["n_subjects"] == 4
    expected_method = (
        ResponseMethod.EMM_INDEPENDENT if use_formula else ResponseMethod.INDEPENDENT
    )
    assert row["method"] == expected_method.value


def test_balanced_null_retains_zero_effect_without_inferential_fields() -> None:
    aggregate = _aggregate(
        [[1], [3], [1], [3]],
        samples=["c1", "c2", "t1", "t2"],
        subjects=["p1", "p2", "p3", "p4"],
        contexts=["control", "control", "treated", "treated"],
    )
    response = estimate_gene_response(
        aggregate,
        _two_context_graph(),
        contrasts=[_treated_vs_control()],
    )

    row = _contrast_rows(response).loc["G1"]
    assert row["effect"] == pytest.approx(0.0)
    assert row["z_score"] == pytest.approx(0.0)
    assert row["direction"] == "zero"
    assert row["status"] == ResponseStatus.OK.value
    assert not {"p_value", "q_value"}.intersection(response.contrasts)


def test_complete_two_condition_design_uses_paired_subject_differences() -> None:
    aggregate = _aggregate(
        [[1], [3], [2], [5], [4], [8]],
        samples=["p1-c", "p1-t", "p2-c", "p2-t", "p3-c", "p3-t"],
        subjects=["p1", "p1", "p2", "p2", "p3", "p3"],
        contexts=["control", "treated"] * 3,
    )
    response = estimate_gene_response(
        aggregate,
        _two_context_graph(),
        contrasts=[_treated_vs_control()],
    )

    row = _contrast_rows(response).loc["G1"]
    expected_se = 1.0 / np.sqrt(3.0)
    assert row["effect"] == pytest.approx(3.0)
    assert row["standard_error"] == pytest.approx(expected_se)
    assert row["z_score"] == pytest.approx(3.0 / expected_se)
    assert row["paired"]
    assert row["method"] == ResponseMethod.PAIRED.value
    assert row["n_samples"] == 6
    assert row["n_subjects"] == 3


def test_context_means_use_equal_sample_not_cell_weights() -> None:
    aggregate = _aggregate(
        [[1], *([[9]] * 3)],
        samples=["s1", "s2", "s2", "s2"],
        subjects=["p1", "p2", "p2", "p2"],
        contexts=["control"] * 4,
    )
    response = estimate_gene_response(
        aggregate,
        ContextGraph.chain(["control"]),
        contrasts=[],
    )

    context_row = response.context_means.iloc[0]
    assert context_row["mean_response"] == pytest.approx(5.0)
    assert context_row["n_samples"] == 2
    assert context_row["n_subjects"] == 2
    assert response.contrasts.empty


def test_default_response_contains_all_global_and_local_continuous_vectors() -> None:
    aggregate = _aggregate(
        [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]],
        samples=["a1", "a2", "b1", "b2", "c1", "c2"],
        subjects=["p1", "p2", "p3", "p4", "p5", "p6"],
        contexts=["a", "a", "b", "b", "c", "c"],
    )
    response = estimate_gene_response(aggregate, ContextGraph.chain(["a", "b", "c"]))

    assert len(response.contrast_specs) == 6
    assert set(response.contrasts["mode"]) == {
        "global_one_vs_rest",
        "local_neighbor",
    }
    assert len(response.contrasts) == 6 * 2
    assert set(response.contrasts["gene"]) == {"G1", "G2"}


def test_product_graph_matches_canonical_multifactor_contexts() -> None:
    values = np.asarray([[1], [3], [5], [7]], dtype=float)
    obs = pd.DataFrame(
        {
            "sample_id": ["r1-a", "r1-b", "r2-a", "r2-b"],
            "subject_id": ["p1", "p2", "p3", "p4"],
            "cell_type": ["R"] * 4,
            "treatment": ["control"] * 4,
            "region": ["r1", "r1", "r2", "r2"],
        },
        index=[f"cell-{index}" for index in range(4)],
    )
    adata = ad.AnnData(
        X=values,
        obs=obs,
        var=pd.DataFrame(index=["G1"]),
    )
    validated = validate_anndata(
        adata,
        InputSchema(
            context_keys=("treatment", "region"),
            expression_source="normalized",
            expression_transform="linear_normalized",
        ),
    )
    aggregate = aggregate_pseudobulk(validated, min_cells=1)
    graph = ContextGraph.product(
        {
            "treatment": ContextGraph.chain(["control"]),
            "region": ContextGraph.chain(["r1", "r2"]),
        }
    )
    r1, r2 = graph.nodes
    contrast = balanced_contrast([r2], [r1], name="r2-v-r1")

    response = estimate_gene_response(aggregate, graph, contrasts=[contrast])

    assert set(response.sample_metadata["context_node"]) == set(graph.nodes)
    assert response.contrasts.iloc[0]["effect"] == pytest.approx(4.0)


def test_additive_formula_marks_factorial_interaction_not_estimable() -> None:
    rows: list[list[float]] = []
    observations: list[dict[str, str]] = []
    for treatment in ("control", "treated"):
        for region in ("core", "edge"):
            for replicate in range(2):
                rows.append([float(treatment == "treated" and region == "edge")])
                observations.append(
                    {
                        "sample_id": f"{treatment}-{region}-{replicate}",
                        "subject_id": f"p-{treatment}-{region}-{replicate}",
                        "cell_type": "R",
                        "treatment": treatment,
                        "region": region,
                    }
                )
    adata = ad.AnnData(
        X=np.asarray(rows),
        obs=pd.DataFrame(
            observations,
            index=[f"cell-{index}" for index in range(len(rows))],
        ),
        var=pd.DataFrame(index=["G1"]),
    )
    validated = validate_anndata(
        adata,
        InputSchema(
            context_keys=("treatment", "region"),
            expression_source="normalized",
            expression_transform="linear_normalized",
        ),
    )
    audit = audit_sample_design(
        validated.report.sample_metadata,
        context_keys=("treatment", "region"),
        formula="~ treatment + region",
        sample_key="sample_id",
    )
    graph = ContextGraph.complete(audit.context_nodes)
    interaction = factorial_interaction_contrast(
        graph,
        "treatment",
        "treated",
        "control",
        "region",
        "edge",
        "core",
        name="treatment-x-region",
    )

    response = estimate_gene_response(
        aggregate_pseudobulk(validated, min_cells=1),
        graph,
        contrasts=[interaction],
        design_audit=audit,
    )

    row = response.contrasts.iloc[0]
    assert np.isnan(row["effect"])
    assert row["status"] == ResponseStatus.NOT_ESTIMABLE.value
    assert row["reason_code"] == "receiver_formula_contrast_not_estimable"


def test_structural_missingness_and_low_cells_stay_nan_and_not_estimable() -> None:
    aggregate = _aggregate(
        [[1], [1], [2], [2], [0], [3], [0]],
        samples=["c1", "c1", "c2", "c2", "t1", "t2", "t2"],
        subjects=["p1", "p1", "p2", "p2", "p3", "p4", "p4"],
        contexts=["control"] * 4 + ["treated"] * 3,
        cell_types=["R", "R", "R", "R", "A", "R", "A"],
        min_cells=2,
    )
    response = estimate_gene_response(
        aggregate,
        _two_context_graph(),
        receivers=["R"],
        contrasts=[_treated_vs_control()],
    )

    vector = response.sample_vector("R", "G1")
    assert vector.notna().sum() == 2
    assert vector.isna().sum() == 2
    treated_mean = response.context_means.loc[
        response.context_means["context"] == "treated"
    ].iloc[0]
    assert np.isnan(treated_mean["mean_response"])
    assert treated_mean["status"] == ResponseStatus.NOT_ESTIMABLE.value
    row = _contrast_rows(response).loc["G1"]
    assert np.isnan(row["effect"])
    assert row["status"] == ResponseStatus.NOT_ESTIMABLE.value
    assert row["reason_code"] == "missing_context_support"


def test_insufficient_support_returns_na_instead_of_a_one_sample_effect() -> None:
    aggregate = _aggregate(
        [[1], [5]],
        samples=["c1", "t1"],
        subjects=["p1", "p2"],
        contexts=["control", "treated"],
    )
    response = estimate_gene_response(
        aggregate,
        _two_context_graph(),
        contrasts=[_treated_vs_control()],
    )

    row = _contrast_rows(response).loc["G1"]
    assert np.isnan(row["effect"])
    assert row["reason_code"] == "insufficient_sample_support"
    assert row["status"] == ResponseStatus.NOT_ESTIMABLE.value


def test_subject_fixed_effects_refuse_between_subject_context_effect() -> None:
    aggregate = _aggregate(
        [[1], [2], [5], [6]],
        samples=["c1", "c2", "t1", "t2"],
        subjects=["p1", "p2", "p3", "p4"],
        contexts=["control", "control", "treated", "treated"],
    )
    response = estimate_gene_response(
        aggregate,
        _two_context_graph(),
        contrasts=[_treated_vs_control()],
        subject_fixed_effects=True,
    )

    row = _contrast_rows(response).loc["G1"]
    assert np.isnan(row["effect"])
    assert row["reason_code"] == "rank_deficient_subject_fixed_effects"
    assert row["status"] == ResponseStatus.NOT_ESTIMABLE.value


def test_mixed_paired_and_unpaired_study_is_not_estimable() -> None:
    aggregate = _aggregate(
        [[1], [3], [2], [5]],
        samples=["p1-c", "p1-t", "p2-c", "p3-t"],
        subjects=["p1", "p1", "p2", "p3"],
        contexts=["control", "treated", "control", "treated"],
    )
    response = estimate_gene_response(
        aggregate,
        _two_context_graph(),
        contrasts=[_treated_vs_control()],
    )

    row = _contrast_rows(response).loc["G1"]
    assert np.isnan(row["effect"])
    assert row["reason_code"] == "mixed_paired_unpaired_design_not_supported"
    assert row["status"] == ResponseStatus.NOT_ESTIMABLE.value


def test_partial_pairing_uses_only_explicit_complete_case_differences() -> None:
    receiver_values = {
        "p1-c": 1.0,
        "p1-t": 3.0,
        "p2-c": 2.0,
        "p2-t": 5.0,
        "p3-c": 4.0,
        "p3-t": 8.0,
        "p4-c": 100.0,
        "p5-t": 200.0,
    }
    values: list[list[float]] = []
    samples: list[str] = []
    subjects: list[str] = []
    contexts: list[str] = []
    cell_types: list[str] = []
    study_samples: list[str] = []
    study_contexts: list[str] = []
    for subject in ("p1", "p2", "p3", "p4", "p5"):
        for suffix, context in (("c", "control"), ("t", "treated")):
            sample = f"{subject}-{suffix}"
            study_samples.append(sample)
            study_contexts.append(context)
            if sample in receiver_values:
                values.append([receiver_values[sample]])
                samples.append(sample)
                subjects.append(subject)
                contexts.append(context)
                cell_types.append("R")
            values.append([0.0])
            samples.append(sample)
            subjects.append(subject)
            contexts.append(context)
            cell_types.append("Carrier")
    aggregate = _aggregate(
        values,
        samples=samples,
        subjects=subjects,
        contexts=contexts,
        cell_types=cell_types,
    )
    audit = _one_factor_audit(study_samples, study_contexts)
    response = estimate_gene_response(
        aggregate,
        _two_context_graph(),
        receivers=["R"],
        contrasts=[_treated_vs_control()],
        design_audit=audit,
    )

    row = _contrast_rows(response).loc["G1"]
    assert row["effect"] == pytest.approx(3.0)
    assert row["standard_error"] == pytest.approx(1.0 / np.sqrt(3.0))
    assert row["method"] == ResponseMethod.EMM_PAIRED_COMPLETE_CASE.value
    assert row["n_subjects"] == 3
    assert row["n_samples"] == 6
    assert row["paired"]


def test_zero_library_count_unit_is_nan_not_zero() -> None:
    aggregate = _aggregate(
        [[0, 0], [1, 1], [1, 1], [1, 1]],
        samples=["c1", "c2", "t1", "t2"],
        subjects=["p1", "p2", "p3", "p4"],
        contexts=["control", "control", "treated", "treated"],
        counts=True,
    )
    response = estimate_gene_response(
        aggregate,
        _two_context_graph(),
        contrasts=[_treated_vs_control()],
    )

    zero_row = response.sample_metadata["sample_id"] == "c1"
    assert np.isnan(response.sample_values[zero_row.to_numpy(), :]).all()
    assert response.sample_metadata.loc[zero_row, "response_reason"].item() == (
        "zero_library_size"
    )
    assert _contrast_rows(response).loc["G1", "reason_code"] == (
        "insufficient_sample_support"
    )
