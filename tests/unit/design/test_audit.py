from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crychic.design import (
    ContextGraph,
    DesignAuditError,
    DesignStatus,
    SubjectDesign,
    audit_sample_design,
    audit_subject_design,
    factorial_interaction_contrast,
    marginal_factor_contrast,
)


def _factorial_metadata(*, duplicate_by_cells: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for treatment in ("control", "treated"):
        for region in ("core", "edge"):
            for replicate in range(2):
                row = {
                    "sample_id": f"{treatment}-{region}-{replicate}",
                    "treatment": treatment,
                    "region": region,
                    "batch": replicate,
                }
                repetitions = 20 if duplicate_by_cells and treatment == "treated" else 1
                rows.extend([row] * repetitions)
    return pd.DataFrame(rows).drop_duplicates("sample_id", keep="first")


def _nodes() -> tuple[tuple[tuple[str, str], ...], ...]:
    return tuple(
        (("region", region), ("treatment", treatment))
        for treatment in ("control", "treated")
        for region in ("core", "edge")
    )


def test_factorial_audit_matches_hand_emm_and_is_not_cell_weighted() -> None:
    audit = audit_sample_design(
        _factorial_metadata(),
        context_keys=("treatment", "region"),
        covariates=("batch",),
        formula="~ batch + treatment * region",
    )
    duplicated = audit_sample_design(
        _factorial_metadata(duplicate_by_cells=True),
        context_keys=("treatment", "region"),
        covariates=("batch",),
        formula="~ batch + treatment * region",
    )

    assert audit.status is DesignStatus.READY
    assert audit.rank == audit.n_columns == 5
    assert audit.context_nodes == duplicated.context_nodes
    np.testing.assert_allclose(audit.emm_matrix, duplicated.emm_matrix)

    graph = ContextGraph.complete(_nodes())
    main = marginal_factor_contrast(
        graph, "treatment", "treated", "control", name="treatment"
    )
    interaction = factorial_interaction_contrast(
        graph,
        "treatment",
        "treated",
        "control",
        "region",
        "edge",
        "core",
        name="treatment_x_region",
    )
    table = audit.contrast_table((main, interaction)).set_index("contrast")
    assert table["estimable"].all()
    main_vector = pd.Series(audit.coefficient_contrast(main), index=audit.column_names)
    interaction_vector = pd.Series(
        audit.coefficient_contrast(interaction), index=audit.column_names
    )
    assert main_vector["treatment[T.treated]"] == pytest.approx(1.0)
    assert main_vector["treatment[T.treated]:region[T.edge]"] == pytest.approx(0.5)
    assert interaction_vector["treatment[T.treated]:region[T.edge]"] == pytest.approx(
        1.0
    )
    assert np.count_nonzero(interaction_vector) == 1


def test_rank_deficiency_and_missing_factor_are_explicitly_blocked() -> None:
    metadata = _factorial_metadata()
    metadata["batch"] = (metadata["treatment"] == "treated").astype(int)
    confounded = audit_sample_design(
        metadata,
        context_keys=("treatment", "region"),
        covariates=("batch",),
        formula="~ batch + treatment * region",
    )
    omitted = audit_sample_design(
        metadata,
        context_keys=("treatment", "region"),
        covariates=("batch",),
        formula="~ batch + region",
    )

    assert confounded.status is DesignStatus.BLOCKED
    assert "design_rank_deficient" in confounded.reason_codes
    assert confounded.aliased_columns
    assert omitted.status is DesignStatus.BLOCKED
    assert "context_factor_omitted" in omitted.reason_codes


@pytest.mark.parametrize(
    "formula, match",
    [
        ("outcome ~ treatment", "one-sided"),
        ("~ np.log(batch) + treatment", "unsupported expression"),
        ("~ missing + treatment", "unsupported expression"),
    ],
)
def test_formula_grammar_rejects_outcomes_and_expressions(
    formula: str, match: str
) -> None:
    with pytest.raises(DesignAuditError, match=match):
        audit_sample_design(
            _factorial_metadata(),
            context_keys=("treatment", "region"),
            covariates=("batch",),
            formula=formula,
        )


def test_factorial_contrast_requires_complete_cells() -> None:
    nodes = _nodes()[:-1]
    with pytest.raises(ValueError, match="lacks cell"):
        factorial_interaction_contrast(
            nodes,
            "treatment",
            "treated",
            "control",
            "region",
            "edge",
            "core",
        )


def test_additive_formula_does_not_make_interaction_estimable() -> None:
    audit = audit_sample_design(
        _factorial_metadata(),
        context_keys=("treatment", "region"),
        covariates=("batch",),
        formula="~ batch + treatment + region",
    )
    interaction = factorial_interaction_contrast(
        _nodes(),
        "treatment",
        "treated",
        "control",
        "region",
        "edge",
        "core",
    )

    assert np.linalg.norm(audit.coefficient_contrast(interaction)) == pytest.approx(0)
    assert not audit.contrast_estimable(interaction)


def test_factorial_helpers_reject_same_levels_and_unbalanced_support() -> None:
    with pytest.raises(ValueError, match="levels must differ"):
        factorial_interaction_contrast(
            _nodes(),
            "treatment",
            "treated",
            "treated",
            "region",
            "edge",
            "core",
        )
    incomplete = (
        (("region", "core"), ("treatment", "control")),
        (("region", "core"), ("treatment", "treated")),
        (("region", "edge"), ("treatment", "treated")),
    )
    with pytest.raises(ValueError, match="common nuisance-factor support"):
        marginal_factor_contrast(incomplete, "treatment", "treated", "control")


@pytest.mark.parametrize(
    ("subjects", "expected", "ready"),
    [
        (("p1", "p2", "p3", "p4"), SubjectDesign.INDEPENDENT, True),
        (("p1", "p2", "p1", "p2"), SubjectDesign.PAIRED, True),
        (("p1", "p2", "p1", "p3"), SubjectDesign.MIXED, False),
    ],
)
def test_subject_design_audit_uses_the_complete_sample_allocation(
    subjects: tuple[str, ...], expected: SubjectDesign, ready: bool
) -> None:
    metadata = pd.DataFrame(
        {
            "condition": ["control", "control", "treated", "treated"],
            "subject_id": subjects,
        }
    )

    audit = audit_subject_design(
        metadata,
        ("control", "treated"),
        context_key="condition",
        subject_key="subject_id",
    )

    assert audit.design is expected
    assert audit.ready is ready
    assert audit.paired is (expected is SubjectDesign.PAIRED)
    assert bool(audit.reason_code) is (not ready)
