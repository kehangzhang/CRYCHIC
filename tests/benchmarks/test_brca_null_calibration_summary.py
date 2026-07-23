from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from benchmarks.adapters.common import sha256_file
from benchmarks.comprehensive.summarize_brca_null_calibration import summarize


def test_randomized_null_summary_passes_complete_conservative_campaign(
    tmp_path: Path,
) -> None:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    effects = pd.DataFrame.from_records(
        [
            {
                "dataset_id": f"null_r{index:02d}",
                "seed": 1000 + index,
                "method_variant_id": f"crychic::run_{index}::native_raw_mean",
                "base_method_id": "crychic",
                "view_label": "mechanistic_sender_lr_score",
                "differential_engine": "native_raw_mean",
                "formal_did_p_value": True,
                "truth_label": 0,
                "truth_direction": 0,
                "status": "observed",
                "p_value": 1.0,
                "q_value": 1.0,
                "ci_low_95": -0.1,
                "ci_high_95": 0.1,
            }
            for index in range(30)
        ]
    )
    effects_path = evaluation / "event_effects.tsv.gz"
    effects.to_csv(effects_path, sep="\t", index=False, compression="gzip")
    metrics_path = evaluation / "metrics.tsv"
    effects.loc[
        :,
        [
            "dataset_id",
            "base_method_id",
            "view_label",
            "differential_engine",
        ],
    ].to_csv(metrics_path, sep="\t", index=False)
    manifest = {
        "status": "complete",
        "outputs": {
            path.name: {"sha256": sha256_file(path)}
            for path in (effects_path, metrics_path)
        },
    }
    (evaluation / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    output = tmp_path / "summary"
    result = summarize((evaluation,), output)

    assert result["gate"]["status"] == "PASS"
    assert result["gate"]["estimates"]["bh_fwer_events"] == 0
    assert (output / "replicate_calibration.tsv").is_file()
