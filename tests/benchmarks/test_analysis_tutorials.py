from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIALS = REPO_ROOT / "tutorials"
NOTEBOOKS = (
    TUTORIALS / "single_sample_communication.ipynb",
    TUTORIALS / "multigroup_differential_communication.ipynb",
    TUTORIALS / "literature_benchmark_results.ipynb",
)


def _notebook(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_analysis_notebook_is_valid_v4_and_code_compiles(path: Path) -> None:
    payload = _notebook(path)

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
        source = "".join(cell["source"])
        if path in NOTEBOOKS[:2]:
            assert source.startswith("%%time\n")
            source = source.removeprefix("%%time\n")
        compile(source, f"{path}:{cell['id']}", "exec")
        assert not any(
            output.get("output_type") == "error"
            for output in cell.get("outputs", [])
        )


def test_single_group_notebook_preserves_estimand_boundaries() -> None:
    text = NOTEBOOKS[0].read_text(encoding="utf-8")

    assert "/media/" not in text
    for required in (
        "CRYCHIC_SINGLE_GROUP_H5AD",
        "availability_only",
        "no_estimable_response_contrast",
        "comm_strength",
        "comm_probability",
        "rank_interactions",
        "persist_edge_evidence=True",
    ):
        assert required in text


def test_multigroup_notebook_downsamples_with_subject_structure() -> None:
    text = NOTEBOOKS[1].read_text(encoding="utf-8")

    assert "/media/" not in text
    for required in (
        "CRYCHIC_MULTIGROUP_H5AD",
        "stratified_downsample",
        "STRATA_KEYS",
        "small_subjects == parent_subjects",
        "multigroup_downsample_example.h5ad",
        "crossfit_specs",
        "fit_descriptive",
        "formal_inference_status",
        "not_estimable",
    ):
        assert required in text
    assert "MultiGroupDifferentialSpec" not in text
    assert "fit_multigroup_descriptive" not in text


def test_benchmark_notebook_embeds_audited_complete_results() -> None:
    payload = _notebook(NOTEBOOKS[2])
    text = NOTEBOOKS[2].read_text(encoding="utf-8")

    assert "/media/" not in text
    for required in (
        "citeseq_datasets_complete",
        "cytosig_datasets_complete",
        "ipf_tasks_complete",
        "H-common/resource_fixed",
        "controlled_unique_ligand_target",
        "patient_bootstrap.tsv",
        "method_robustness_summary.tsv",
        "sha256_file",
        "not_evaluable",
    ):
        assert required in text
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell["execution_count"] is not None for cell in code_cells)
    embedded_images = [
        output
        for cell in code_cells
        for output in cell["outputs"]
        if "image/png" in output.get("data", {})
    ]
    assert len(embedded_images) >= 7
