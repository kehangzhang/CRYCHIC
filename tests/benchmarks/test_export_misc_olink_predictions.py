from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.comprehensive.evaluate_three_group import RunRecord
from benchmarks.comprehensive.export_misc_olink_predictions import (
    _aggregate_ligands,
    _materialize_effects,
    _validate_misc_run_binding,
)


def test_ligand_aggregation_uses_maximum_signed_observed_event() -> None:
    events = pd.DataFrame(
        {
            "ligand": ["L1", "L1", "L2"],
            "effect": [-0.4, 0.2, np.nan],
            "status": ["exploratory", "observed", "not_estimable"],
        }
    )

    result = _aggregate_ligands(events, ["L1", "L2"]).set_index("ligand")

    assert result.loc["L1", "score"] == 0.2
    assert result.loc["L1", "status"] == "observed"
    assert result.loc["L1", "n_observed_events"] == 2
    assert np.isnan(result.loc["L2", "score"])
    assert result.loc["L2", "status"] == "not_estimable"


def test_event_materialization_types_missing_reason_codes() -> None:
    universe = pd.DataFrame(
        {
            "sender": ["A", "A"],
            "receiver": ["B", "B"],
            "interaction_id": ["i1", "i2"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
        }
    )
    effects = universe.iloc[:1].copy()
    effects["effect"] = 0.2
    effects["status"] = "observed"
    effects["reason_code"] = np.nan

    result = _materialize_effects(effects, universe, method="test")

    assert result.loc[1, "status"] == "not_estimable"
    assert result.loc[1, "reason_code"] == "event_not_returned"


def test_misc_binding_rejects_raw_count_cellchat_input() -> None:
    manifest = {
        "method": {"id": "cellchat", "version": "2.1.2"},
        "input": {"sha256": "input"},
        "resource": {"mode": "H-common", "payload_sha256": "resource"},
        "code": {"commit": "commit", "dirty": False},
        "parameters": {"layer": "counts", "nboot": 100, "min_cells": 10},
    }
    record = RunRecord(
        method="cellchat",
        dataset_id="misc_olink",
        contrast=None,
        directory=Path("cellchat"),
        manifest=manifest,
        kind="external",
    )

    with pytest.raises(ValueError, match="CellChat MIS-C protocol mismatch"):
        _validate_misc_run_binding(
            record,
            expected_input_sha256="input",
            expected_resource_sha256="resource",
            expected_code_commit="commit",
        )
