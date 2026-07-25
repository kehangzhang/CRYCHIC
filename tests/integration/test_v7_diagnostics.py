from __future__ import annotations

from dataclasses import replace

from tests.integration.test_subject_crossfit import (
    _adata,
    _bundle,
    _config,
    _persistable_spec,
    _prior,
)

from crychic.scoring import AbsoluteActivityV2Spec
from crychic.workflow import (
    GATE_STAGE_LEDGER_COLUMNS,
    GATE_STAGES,
    build_v7_diagnostics,
    build_v7_gate_stage_ledger,
    run_subject_crossfit,
)


def test_legacy_parent_gates_propagate_to_the_exact_frozen_sender_axis() -> None:
    crossfit = run_subject_crossfit(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=replace(
            _persistable_spec(),
            absolute_activity_v2_spec=AbsoluteActivityV2Spec(),
        ),
    )
    assert all(fold.family_common_applications for fold in crossfit.folds)

    result = build_v7_diagnostics(crossfit, dataset_id="legacy-connected")
    overall = result.gate_attrition.loc[
        result.gate_attrition["fold_id"].eq("__all__")
        & result.gate_attrition["stratum_kind"].eq("overall")
    ].set_index("stage")

    assert tuple(overall.index) == GATE_STAGES
    assert int(overall.loc["common_candidate_universe", "n_universe"]) == 32
    for stage in GATE_STAGES[2:-1]:
        row = overall.loc[stage]
        assert int(row["n_not_computed"]) == 0
        assert (
            int(row["n_stage_passed"])
            + int(row["n_stage_failed"])
            + int(row["n_not_estimable"])
        ) == 32
    assert int(overall.loc["receptor_eligible", "n_stage_passed"]) == 16
    assert int(overall.loc["sender_assignment_available", "n_stage_passed"]) == 32

    ledger = build_v7_gate_stage_ledger(
        crossfit,
        dataset_id="legacy-connected",
    )
    assert tuple(ledger.columns) == GATE_STAGE_LEDGER_COLUMNS
    assert len(ledger) == 32
    assert not ledger.duplicated(
        [
            "contrast_name",
            "fold_id",
            "sample_id",
            "context_id",
            "sender",
            "receiver",
            "interaction_id",
        ]
    ).any()
    assert ledger["common_candidate_universe_status"].eq("passed").all()
    assert ledger["receptor_eligible_status"].eq("passed").sum() == 16
    assert ledger["sender_assignment_available_status"].eq("passed").sum() == 32
