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
        "incremental_official_status_counts",
        "n_effectively_positive_family_coefficients",
        "Track B",
        "method_superiority",
    ):
        assert required in text
