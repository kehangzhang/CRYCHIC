from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "tutorials/developer_subject_crossfit.ipynb"


def test_developer_notebook_is_clean_valid_v4_and_code_compiles() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert payload["nbformat"] == 4
    assert payload["nbformat_minor"] >= 5
    assert payload["metadata"]["kernelspec"]["name"] == "python3"
    cells = payload["cells"]
    assert cells
    cell_ids = [cell["id"] for cell in cells]
    assert len(cell_ids) == len(set(cell_ids))
    for cell in cells:
        assert isinstance(cell["source"], list)
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"{NOTEBOOK}:{cell['id']}", "exec")


def test_developer_notebook_preserves_claim_and_track_boundaries() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")

    for required in (
        "not a deep-learning model",
        "complete_pipeline_oof_certified",
        "formal_inference_status",
        "comm_probability",
        "CrossFitResult.load",
        "read_components",
        "read_descriptive_differential",
        "query_family_scores",
        "query_integrated_lr_scores",
        "query_lr_pairs",
        "query_sender_lr_pairs",
        "RepeatedCrossFitSpec",
        "fit_repeated_crossfit",
        "resample_crossfit",
        "full_pipeline_refit_per_resample",
        "not_released_full_pipeline_resampling_diagnostic_only",
        "Track B",
        "freeze_crossfit_directional_target_program_universe",
        "score_crossfit_directional_target_programs",
        "method_superiority",
    ):
        assert required in text


def test_developer_notebook_tiny_fixture_builds_public_specs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Execute the portable setup cells without running the full fit."""

    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    setup_cell_ids = {
        "imports-and-options",
        "synthetic-input",
        "configuration-and-validation",
        "resource-builders",
        "crossfit-spec",
    }
    namespace: dict[str, object] = {"__name__": "__tutorial_smoke__"}
    monkeypatch.chdir(tmp_path)

    for cell in payload["cells"]:
        if cell["cell_type"] == "code" and cell["id"] in setup_cell_ids:
            exec(
                compile(
                    "".join(cell["source"]),
                    f"{NOTEBOOK}:{cell['id']}",
                    "exec",
                ),
                namespace,
            )

    assert namespace["USE_SYNTHETIC"] is True
    assert namespace["validated"].report.n_subjects == 8
    assert namespace["validated"].report.n_samples == 16
    assert len(namespace["lr_resource"].interactions) == 2
    assert namespace["target_prior"].driver_kind == "interaction"
    assert {contrast.name for contrast in namespace["crossfit_spec"].contrasts} == {
        "stim_vs_control",
        "control_vs_stim",
    }
    assert len(namespace["crossfit_spec"].directional_pairs) == 1
    assert namespace["tuning"].inner_allowed_n_splits == (2,)
