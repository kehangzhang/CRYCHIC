from __future__ import annotations

import hashlib
import json
from pathlib import Path

from crychic.scoring import FAMILY_COMMON_SCORE_VERSION

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "benchmarks" / "configs" / "v7_legacy_baselines_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v7_legacy_baselines_are_frozen_diagnostics() -> None:
    payload = json.loads(FREEZE.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "crychic-v7-legacy-baseline-freeze-v1"
    assert payload["policy"] == {
        "benchmark_diagnostic_only": True,
        "formal_inference_eligible": False,
        "public_default_replaced": False,
        "further_rc_development_allowed": False,
    }
    assert len(payload["baselines"]) == 3
    assert all(item["benchmark_diagnostic_only"] for item in payload["baselines"])
    assert not any(item["formal_inference_eligible"] for item in payload["baselines"])


def test_v7_legacy_baseline_sources_match_frozen_payloads() -> None:
    payload = json.loads(FREEZE.read_text(encoding="utf-8"))
    by_name = {item["name"]: item for item in payload["baselines"]}

    assert (
        by_name["legacy_family_common_v2"]["score_version"]
        == FAMILY_COMMON_SCORE_VERSION
    )
    for name in ("RC12_sender_response_ranker", "RC14_pair_prioritized_event_ranker"):
        item = by_name[name]
        assert _sha256(ROOT / item["config"]) == item["config_sha256"]
