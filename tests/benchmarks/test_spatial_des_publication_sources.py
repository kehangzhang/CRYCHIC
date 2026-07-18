from __future__ import annotations

import json
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MACHINE_PATH = re.compile(r"(?:^|[=\"'\s])(?:/[A-Za-z0-9._~+-]+){2,}")


def _assert_path_safe_text(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert not MACHINE_PATH.search(text), f"absolute machine path in {path}"


def test_spatial_des_protocol_and_notebook_are_portable() -> None:
    protocol = (
        REPOSITORY_ROOT
        / "docs/benchmarks/multigroup_spatial_20260717/PROTOCOL.md"
    )
    notebook = REPOSITORY_ROOT / "tutorials/spatial_des_benchmark.ipynb"
    _assert_path_safe_text(protocol)
    _assert_path_safe_text(notebook)

    payload = json.loads(notebook.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    assert payload["cells"]
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(not cell.get("outputs") for cell in code_cells)
    source = "\n".join(
        "".join(cell["source"]) for cell in code_cells
    )
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
    assert "docs/benchmarks/multigroup_spatial_20260717/results" in source
    assert "benchmark_work" not in source


def test_compact_summaries_declare_analysis_units_and_are_path_safe() -> None:
    scdiffcom_path = (
        REPOSITORY_ROOT
        / "benchmarks/results/scdiffcom_condition_aware_s4_v1_summary.json"
    )
    scseq_path = (
        REPOSITORY_ROOT
        / "benchmarks/results/scseqcommdiff_paper_s4_v1_summary.json"
    )
    for path in (scdiffcom_path, scseq_path):
        _assert_path_safe_text(path)

    scdiffcom = json.loads(scdiffcom_path.read_text(encoding="utf-8"))
    assert {
        dataset["analysis_unit"] for dataset in scdiffcom["datasets"].values()
    } == {"condition_level"}

    scseq = json.loads(scseq_path.read_text(encoding="utf-8"))
    for dataset in scseq["datasets"].values():
        assert dataset["condition_aware"]["analysis_unit"] == "condition_level"
        assert (
            dataset["multi_sample_subject_primary"]["analysis_unit"]
            == "subject_id"
        )
