"""Summarize paired BRCA v7 effects as a descriptive response interaction."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from benchmarks.adapters.common import (
    git_metadata,
    json_safe,
    prepare_output,
    sha256_file,
    write_json,
)
from benchmarks.literature.run_v7_intervention_validation import (
    DEFAULT_CONFIG,
    InterventionDatasetContract,
    load_intervention_config,
)
from benchmarks.literature.run_v7_intervention_validation import (
    SCHEMA_VERSION as RUN_SCHEMA_VERSION,
)

SCHEMA_VERSION = "crychic-suggest-next2-v7-brca-response-summary-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVENT_KEYS = (
    "generator_id",
    "score_view",
    "estimand",
    "resolution",
    "contrast_scope",
    "sender",
    "receiver",
    "interaction_id",
    "inference_id",
)


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _input_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_effects(
    root: Path,
    *,
    contract: InterventionDatasetContract,
    protocol_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, object]]:
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != RUN_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("dataset") != contract.slug
        or manifest.get("dataset_id") != contract.dataset_id
        or manifest.get("formal_inference_allowed") is not False
    ):
        raise ValueError(f"BRCA {contract.slug} input is not a completed v7 paired run")
    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    release = manifest.get("release")
    if (
        not isinstance(inputs, Mapping)
        or not isinstance(outputs, Mapping)
        or not isinstance(release, Mapping)
        or release.get("brca_cross_subtype_effect_is_descriptive_only") is not True
    ):
        raise ValueError("BRCA paired run lacks its release or provenance contract")
    protocol = inputs.get("protocol")
    effects_record = outputs.get("effects.parquet")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("sha256") != protocol_sha256
        or not isinstance(effects_record, Mapping)
        or effects_record.get("filename") != "effects.parquet"
    ):
        raise ValueError("BRCA paired run protocol or effect record changed")
    effects_path = root / "effects.parquet"
    if sha256_file(effects_path) != effects_record.get("sha256"):
        raise ValueError("BRCA paired effect checksum mismatch")
    effects = pd.read_parquet(effects_path)
    required = {
        *EVENT_KEYS,
        "effect",
        "standard_error",
        "statistic",
        "status",
        "reason_code",
        "p_value",
        "q_value",
        "formal_inference_allowed",
    }
    missing = required.difference(effects.columns)
    if missing:
        raise ValueError(f"BRCA paired effects are missing: {sorted(missing)}")
    if effects["formal_inference_allowed"].astype(bool).any() or effects[
        ["p_value", "q_value"]
    ].notna().any(axis=None):
        raise ValueError("BRCA response summary requires withheld formal p/q")
    effects = effects.loc[effects["inference_id"].eq("I1")].copy()
    if effects.empty or effects.duplicated(list(EVENT_KEYS)).any():
        raise ValueError("BRCA paired I1 effect axis is empty or duplicated")
    return (
        effects,
        manifest,
        {
            "manifest": _input_record(manifest_path),
            "effects": _input_record(effects_path),
        },
    )


def build_descriptive_response_interaction(
    expander: pd.DataFrame,
    nonexpander: pd.DataFrame,
) -> pd.DataFrame:
    """Compute E(On-Pre) minus NE(On-Pre) without inferential relabeling."""

    required = {
        *EVENT_KEYS,
        "effect",
        "standard_error",
        "statistic",
        "status",
        "reason_code",
        "p_value",
        "q_value",
        "formal_inference_allowed",
    }
    for name, table in (("expander", expander), ("nonexpander", nonexpander)):
        missing = required.difference(table.columns)
        if missing or table.duplicated(list(EVENT_KEYS)).any():
            raise ValueError(f"{name} effect table violates the event contract")
        if table["formal_inference_allowed"].astype(bool).any() or table[
            ["p_value", "q_value"]
        ].notna().any(axis=None):
            raise ValueError("BRCA descriptive response cannot consume formal tests")
    value_columns = (
        "effect",
        "standard_error",
        "statistic",
        "status",
        "reason_code",
    )
    left = expander.loc[:, [*EVENT_KEYS, *value_columns]].rename(
        columns={column: f"{column}_e" for column in value_columns}
    )
    right = nonexpander.loc[:, [*EVENT_KEYS, *value_columns]].rename(
        columns={column: f"{column}_ne" for column in value_columns}
    )
    result = left.merge(
        right,
        on=list(EVENT_KEYS),
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    effect_e = pd.to_numeric(result["effect_e"], errors="coerce")
    effect_ne = pd.to_numeric(result["effect_ne"], errors="coerce")
    observed = (
        result["_merge"].eq("both")
        & result["status_e"].eq("observed")
        & result["status_ne"].eq("observed")
        & np.isfinite(effect_e)
        & np.isfinite(effect_ne)
    )
    result["expander_on_minus_pre_effect"] = effect_e
    result["nonexpander_on_minus_pre_effect"] = effect_ne
    result["descriptive_response_interaction"] = (effect_e - effect_ne).where(observed)
    result["absolute_response_interaction"] = result[
        "descriptive_response_interaction"
    ].abs()
    result["comparison_status"] = np.where(
        observed, "observed", "not_estimable_in_one_or_both_subtypes"
    )
    result["direction_pattern"] = np.select(
        (
            observed & effect_e.gt(0.0) & effect_ne.le(0.0),
            observed & effect_e.lt(0.0) & effect_ne.ge(0.0),
            observed & effect_e.gt(effect_ne),
            observed & effect_e.lt(effect_ne),
            observed,
        ),
        (
            "expander_specific_increase",
            "expander_specific_decrease",
            "stronger_in_expander",
            "stronger_in_nonexpander",
            "tied",
        ),
        default="not_estimable",
    )
    result["formal_inference_allowed"] = False
    result["p_value"] = np.nan
    result["q_value"] = np.nan
    result["claim_scope"] = (
        "descriptive_difference_of_independent_subtype_paired_effects_only"
    )
    return result.drop(columns="_merge").sort_values(
        [*EVENT_KEYS, "comparison_status"], kind="stable", ignore_index=True
    )


def summarize_response_interaction(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize shared-event coverage and descriptive interaction geometry."""

    rows: list[dict[str, object]] = []
    grouping = ["generator_id", "score_view", "estimand", "resolution"]
    for key, group in table.groupby(grouping, observed=True, sort=True):
        observed = group.loc[group["comparison_status"].eq("observed")]
        e_effect = pd.to_numeric(
            observed["expander_on_minus_pre_effect"], errors="coerce"
        )
        ne_effect = pd.to_numeric(
            observed["nonexpander_on_minus_pre_effect"], errors="coerce"
        )
        interaction = pd.to_numeric(
            observed["descriptive_response_interaction"], errors="coerce"
        )
        correlation = math.nan
        if len(observed) >= 3 and e_effect.nunique() > 1 and ne_effect.nunique() > 1:
            correlation = float(spearmanr(e_effect, ne_effect).statistic)
        rows.append(
            {
                **dict(zip(grouping, key, strict=True)),
                "rows_union": len(group),
                "rows_observed_both": len(observed),
                "observed_both_fraction": len(observed) / len(group),
                "median_response_interaction": (
                    math.nan if interaction.empty else float(interaction.median())
                ),
                "median_absolute_response_interaction": (
                    math.nan if interaction.empty else float(interaction.abs().median())
                ),
                "positive_response_interaction_fraction": (
                    math.nan if interaction.empty else float(interaction.gt(0.0).mean())
                ),
                "e_ne_effect_spearman": correlation,
                "formal_inference_allowed": False,
                "claim_scope": (
                    "descriptive_difference_of_independent_subtype_paired_effects_only"
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def _top_events(table: pd.DataFrame, *, budget: int = 100) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    grouping = ["generator_id", "score_view", "estimand", "resolution"]
    for _, group in table.loc[table["comparison_status"].eq("observed")].groupby(
        grouping, observed=True, sort=True
    ):
        local = group.sort_values(
            ["absolute_response_interaction", "sender", "receiver", "interaction_id"],
            ascending=[False, True, True, True],
            kind="stable",
        ).head(budget)
        local = local.copy()
        local["descriptive_rank"] = np.arange(1, len(local) + 1)
        selected.append(local)
    return (
        pd.concat(selected, ignore_index=True)
        if selected
        else table.head(0).assign(descriptive_rank=pd.Series(dtype=int))
    )


def run(
    *,
    expander_root: Path,
    nonexpander_root: Path,
    output_dir: Path,
    config_path: Path = DEFAULT_CONFIG,
    overwrite: bool = False,
) -> dict[str, Any]:
    protocol, contracts = load_intervention_config(config_path)
    protocol_sha256 = sha256_file(config_path)
    expander, _, expander_provenance = _load_effects(
        expander_root,
        contract=contracts["brca_e"],
        protocol_sha256=protocol_sha256,
    )
    nonexpander, _, nonexpander_provenance = _load_effects(
        nonexpander_root,
        contract=contracts["brca_ne"],
        protocol_sha256=protocol_sha256,
    )
    response = build_descriptive_response_interaction(expander, nonexpander)
    summary = summarize_response_interaction(response)
    top = _top_events(response)
    output = prepare_output(output_dir, overwrite=overwrite)
    tables = {
        "descriptive_response_interactions.parquet": response,
        "descriptive_response_summary.tsv": summary,
        "descriptive_top100_events.tsv": top,
    }
    records: dict[str, object] = {}
    for filename, table in tables.items():
        path = output / filename
        if path.suffix == ".parquet":
            table.to_parquet(path, index=False, compression="zstd")
        else:
            table.to_csv(path, sep="\t", index=False)
        records[filename] = {
            "filename": filename,
            "bytes": path.stat().st_size,
            "rows": len(table),
            "sha256": sha256_file(path),
        }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "estimand": "expander_on_minus_pre_minus_nonexpander_on_minus_pre",
        "formal_inference_allowed": False,
        "claim_scope": (
            "descriptive_difference_of_independent_subtype_paired_effects_only"
        ),
        "inputs": {
            "protocol": _input_record(config_path),
            "brca_e": expander_provenance,
            "brca_ne": nonexpander_provenance,
        },
        "release": dict(protocol["release"]),
        "code": git_metadata(REPOSITORY_ROOT),
        "outputs": records,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expander-root", type=Path, required=True)
    parser.add_argument("--nonexpander-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = run(
        expander_root=arguments.expander_root.resolve(),
        nonexpander_root=arguments.nonexpander_root.resolve(),
        output_dir=arguments.output_dir.resolve(),
        config_path=arguments.config.resolve(),
        overwrite=arguments.overwrite,
    )
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
