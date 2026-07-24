from __future__ import annotations

import ast
from pathlib import Path

import nbformat
from benchmarks.report.generate_suggest_v5_under100k_notebook import (
    build_notebook,
    generate,
)


def test_every_code_cell_records_wall_time_and_has_no_machine_path() -> None:
    notebook = build_notebook()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert len(code_cells) >= 10
    assert all(cell.source.startswith("%%time\n") for cell in code_cells)
    assert "/media/" not in "\n".join(cell.source for cell in notebook.cells)
    for cell in code_cells:
        ast.parse("\n".join(cell.source.splitlines()[1:]))


def test_generate_writes_valid_unexecuted_notebook(tmp_path: Path) -> None:
    output = tmp_path / "report.ipynb"

    generate(output, execute=False, workspace_root=None, timeout_seconds=30)

    observed = nbformat.read(output, as_version=4)
    assert observed.cells[0].cell_type == "markdown"
    assert "under-100k" in observed.cells[0].source
