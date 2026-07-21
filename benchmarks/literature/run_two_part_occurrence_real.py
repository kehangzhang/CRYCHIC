"""Run the frozen RC6 two-part hurdle head on Kuppe or MS."""

from __future__ import annotations

import argparse
import json
import resource as process_resource
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import betainc

from benchmarks.adapters.common import (
    git_metadata,
    prepare_output,
    sha256_file,
    write_json,
)
from benchmarks.literature.component_swap_benchmark import _evaluate
from benchmarks.literature.run_receiver_program_soft_real import (
    METHOD_STRICT as RC3_METHOD_STRICT,
)
from benchmarks.literature.run_receiver_program_soft_real import (
    SCHEMA_VERSION as RC3_SCHEMA_VERSION,
)
from benchmarks.literature.two_part_occurrence import (
    fit_hurdle_channel_reliability,
    fit_subject_hurdle_effects,
    hurdle_directional_weights,
)

SCHEMA_VERSION = "crychic-two-part-occurrence-real-v1"
METHOD_STRICT = "CRYCHIC_RC6_two_part_strict"
METHOD_FALLBACK = "CRYCHIC_RC6_two_part_fallback"
METHOD_VERSION = "two-part-occurrence-rc6-v1"
RESOURCE_ID = "ConnectomeDB2020_Hou_2020_human"
CONFIG_SHA256 = "3596c29fa0732296077218425ff53970070c001819db942b416efaaff4689623"
ACCEPTANCE_SHA256 = "51efab810accc1b0c8c90a1d30d3bf5fd2f5149e88f846ae710a6494f29f4667"
FROZEN_SHA256 = "625d1aba84dec8ee33ab49f38dca041060f379c21983b26182e779982ee0e2d0"
RC3_SHA256 = {
    "kuppe": {
        "manifest.json": (
            "617f6814a79e9b832398f0f9e33fde5052a27eae37a2e2f73d968501d010dd01"
        ),
        "directed_edge_program_weights.parquet": (
            "739634cac2a49aad71a05413799555faae5cae4b00b39547418ee71500ee0026"
        ),
        "condition_cell_pair_rankings.tsv": (
            "78ef42d18063c01f18d554c1200d28116ede89c210a36f0694c60e3ed2990ea6"
        ),
    },
    "ms": {
        "manifest.json": (
            "d9c9c582af3be2613b0ef93251ebf1bc1281c2073128c8986cce4205ba74db54"
        ),
        "directed_edge_program_weights.parquet": (
            "e275f7ab4587c7e5ac911500d2321758b2e96c1e6eff6cd13862aab1f21113fa"
        ),
        "condition_cell_pair_rankings.tsv": (
            "cc3388a40ab7288f99df909d821e868e86a76f8a6a22995f1d4e514d748892d2"
        ),
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return cast(dict[str, Any], value)


def _input_record(path: Path) -> dict[str, object]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _output_record(path: Path, table: pd.DataFrame) -> dict[str, object]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "columns": list(table.columns),
        "sha256": sha256_file(path),
    }


def _bound_output(
    root: Path,
    manifest: Mapping[str, Any],
    filename: str,
    *,
    expected_sha256: str | None = None,
) -> Path:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(
        outputs.get(filename), Mapping
    ):
        raise ValueError(f"manifest lacks bound output: {filename}")
    record = cast(Mapping[str, Any], outputs[filename])
    path = root / filename
    observed = sha256_file(path)
    if record.get("filename") != filename or record.get("sha256") != observed:
        raise ValueError(f"manifest output checksum mismatch: {filename}")
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(f"frozen input checksum mismatch: {filename}")
    return path


def _validate_frozen_candidate(
    config_path: Path, acceptance_path: Path, frozen_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = {
        config_path: CONFIG_SHA256,
        acceptance_path: ACCEPTANCE_SHA256,
        frozen_path: FROZEN_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"frozen RC6 checksum mismatch: {path.name}")
    acceptance = _read_json(acceptance_path)
    frozen = _read_json(frozen_path)
    if acceptance.get("accepted") is not True:
        raise ValueError("RC6 simulation candidate did not pass acceptance")
    candidate = frozen.get("candidate")
    if not isinstance(candidate, Mapping) or dict(candidate) != {
        "magnitude_alpha": 0.5,
        "name": "two_part_20_05",
        "occurrence_alpha": 2.0,
    }:
        raise ValueError("unexpected RC6 frozen candidate")
    return frozen, _read_json(config_path)


def _load_rc3(
    dataset: str, rc3_run: Path, expected_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    manifest_path = rc3_run / "manifest.json"
    if sha256_file(manifest_path) != RC3_SHA256[dataset]["manifest.json"]:
        raise ValueError("frozen RC3 manifest checksum mismatch")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != RC3_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("dataset") != dataset
    ):
        raise ValueError("RC3 run schema, status, or dataset mismatch")
    edge_path = _bound_output(
        rc3_run,
        manifest,
        "directed_edge_program_weights.parquet",
        expected_sha256=RC3_SHA256[dataset]["directed_edge_program_weights.parquet"],
    )
    ranking_path = _bound_output(
        rc3_run,
        manifest,
        "condition_cell_pair_rankings.tsv",
        expected_sha256=RC3_SHA256[dataset]["condition_cell_pair_rankings.tsv"],
    )
    expected_record = manifest.get("inputs", {}).get("rc2", {}).get("expected_sets")
    if not isinstance(expected_record, Mapping) or expected_record.get(
        "sha256"
    ) != sha256_file(expected_path):
        raise ValueError("expected spatial sets do not match the frozen RC3 run")
    rankings = pd.read_csv(ranking_path, sep="\t")
    strict = rankings.loc[rankings["method"].eq(RC3_METHOD_STRICT)]
    if strict.empty:
        raise ValueError("RC3 strict ranking template is missing")
    return (
        pd.read_parquet(edge_path),
        rankings,
        pd.read_csv(expected_path, sep="\t"),
        {
            "manifest": _input_record(manifest_path),
            "edges": _input_record(edge_path),
            "rankings": _input_record(ranking_path),
            "expected_sets": _input_record(expected_path),
        },
    )


def _dense(matrix: Any) -> np.ndarray:
    return (
        matrix.toarray().astype(float, copy=False)
        if sparse.issparse(matrix)
        else np.asarray(matrix, dtype=float)
    )


def _subject_pseudobulk(
    pdata: ad.AnnData,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    required_obs = {
        "replicate_id",
        "subject_id",
        "condition",
        "cell_type",
        "psbulk_n_cells",
    }
    missing = required_obs.difference(pdata.obs.columns)
    if missing or "psbulk_props" not in pdata.layers:
        raise ValueError(f"pseudobulk hurdle inputs are missing: {sorted(missing)}")
    obs = pdata.obs.reset_index(names="pseudobulk_id").copy()
    for column in ("replicate_id", "subject_id", "condition", "cell_type"):
        obs[column] = obs[column].astype(str)
    design = obs.loc[:, ["subject_id", "condition"]].drop_duplicates()
    if design["subject_id"].duplicated().any():
        raise ValueError("one subject maps to multiple pseudobulk conditions")
    design = design.sort_values(
        ["condition", "subject_id"], kind="stable", ignore_index=True
    )
    groups = (
        obs.loc[:, ["subject_id", "condition", "cell_type"]]
        .drop_duplicates()
        .sort_values(
            ["condition", "subject_id", "cell_type"],
            kind="stable",
            ignore_index=True,
        )
    )
    group_lookup = {
        tuple(row): index
        for index, row in enumerate(
            groups.loc[:, ["subject_id", "condition", "cell_type"]].to_numpy()
        )
    }
    group_index = np.array(
        [
            group_lookup[tuple(row)]
            for row in obs.loc[:, ["subject_id", "condition", "cell_type"]].to_numpy()
        ],
        dtype=int,
    )
    aggregation = sparse.csr_matrix(
        (
            np.ones(len(group_index), dtype=float),
            (group_index, np.arange(len(group_index))),
        ),
        shape=(len(groups), len(obs)),
    )
    counts = _dense(pdata.X)
    proportions = _dense(pdata.layers["psbulk_props"])
    n_cells = pd.to_numeric(obs["psbulk_n_cells"], errors="coerce").to_numpy(
        dtype=float
    )
    if (
        not np.isfinite(counts).all()
        or (counts < 0.0).any()
        or not np.isfinite(proportions).all()
        or ((proportions < 0.0) | (proportions > 1.0)).any()
        or not np.isfinite(n_cells).all()
        or (n_cells <= 0.0).any()
    ):
        raise ValueError("pseudobulk counts, proportions, or cell counts are invalid")
    subject_counts = np.asarray(aggregation @ counts, dtype=float)
    detected = np.asarray(aggregation @ (n_cells[:, None] * proportions), dtype=float)
    subject_cells = np.asarray(aggregation @ n_cells).ravel()
    detected = np.clip(detected, 0.0, subject_cells[:, None])
    posterior_presence = 1.0 - betainc(
        detected + 0.5,
        subject_cells[:, None] - detected + 0.5,
        0.1,
    )
    library = subject_counts.sum(axis=1)
    normalized_expression = np.log1p(
        np.divide(
            subject_counts * 1e4,
            library[:, None],
            out=np.zeros_like(subject_counts),
            where=library[:, None] > 0.0,
        )
    )
    support = groups.copy()
    support["pseudobulk_samples"] = (
        np.asarray(aggregation.sum(axis=1)).ravel().astype(int)
    )
    support["subject_cell_count"] = subject_cells
    support["subject_library_size"] = library
    return posterior_presence, normalized_expression, design, support


def derive_subject_hurdle_inputs(
    pdata: ad.AnnData,
    edges: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map subject pseudobulks to aligned directed LR hurdle observations."""

    edge_columns = {"sender", "receiver", "interaction_id", "ligand", "receptor"}
    missing = edge_columns.difference(edges.columns)
    if missing or edges.duplicated(["sender", "receiver", "interaction_id"]).any():
        raise ValueError(
            f"RC3 edge identities are missing or duplicated: {sorted(missing)}"
        )
    posterior, expression, design, support = _subject_pseudobulk(pdata)
    genes = pd.Index(pdata.var_names.astype(str))
    ligand_position = genes.get_indexer(edges["ligand"].astype(str))
    receptor_position = genes.get_indexer(edges["receptor"].astype(str))
    if (ligand_position < 0).any() or (receptor_position < 0).any():
        raise ValueError("RC3 edges contain genes absent from LIANA pseudobulk")
    group_position = {
        (str(row.subject_id), str(row.cell_type)): index
        for index, row in support.iterrows()
    }
    sender = edges["sender"].astype(str).to_numpy()
    receiver = edges["receiver"].astype(str).to_numpy()
    presence = np.full((len(edges), len(design)), np.nan, dtype=float)
    magnitude = np.full((len(edges), len(design)), np.nan, dtype=float)
    sample_tables = []
    identifiers = ["sender", "receiver", "interaction_id", "ligand", "receptor"]
    for subject_index, subject in enumerate(design.itertuples(index=False)):
        sender_row = np.array(
            [
                group_position.get((str(subject.subject_id), value), -1)
                for value in sender
            ]
        )
        receiver_row = np.array(
            [
                group_position.get((str(subject.subject_id), value), -1)
                for value in receiver
            ]
        )
        observed = (sender_row >= 0) & (receiver_row >= 0)
        ligand_presence = np.zeros(len(edges), dtype=float)
        receptor_presence = np.zeros(len(edges), dtype=float)
        ligand_expression = np.zeros(len(edges), dtype=float)
        receptor_expression = np.zeros(len(edges), dtype=float)
        ligand_presence[observed] = posterior[
            sender_row[observed], ligand_position[observed]
        ]
        receptor_presence[observed] = posterior[
            receiver_row[observed], receptor_position[observed]
        ]
        ligand_expression[observed] = expression[
            sender_row[observed], ligand_position[observed]
        ]
        receptor_expression[observed] = expression[
            receiver_row[observed], receptor_position[observed]
        ]
        presence[observed, subject_index] = (
            ligand_presence[observed] * receptor_presence[observed]
        )
        positive = observed & (ligand_expression > 0.0) & (receptor_expression > 0.0)
        magnitude[positive, subject_index] = np.sqrt(
            ligand_expression[positive] * receptor_expression[positive]
        )
        subject_table = edges.loc[:, identifiers].copy()
        subject_table["subject_id"] = str(subject.subject_id)
        subject_table["condition"] = str(subject.condition)
        subject_table["presence_probability"] = presence[:, subject_index]
        subject_table["positive_magnitude"] = magnitude[:, subject_index]
        subject_table["status"] = np.where(observed, "observed", "structural_absence")
        sample_tables.append(subject_table)
    return (
        presence,
        magnitude,
        design,
        support,
        pd.concat(sample_tables, ignore_index=True),
    )


def _load_liana_hurdle(
    liana_run: Path,
    edges: pd.DataFrame,
    *,
    target: str,
    reference: str,
) -> tuple[
    np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]
]:
    manifest_path = liana_run / "run_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("contrast") != {
        "reference": reference,
        "target": target,
    }:
        raise ValueError("LIANA run status or contrast differs from RC3")
    analysis_unit = manifest.get("analysis_unit")
    if not isinstance(analysis_unit, Mapping) or (
        analysis_unit.get("replicate_key") != "sample_id"
        or analysis_unit.get("subject_key") != "subject_id"
        or analysis_unit.get("primary_panel") is not True
    ):
        raise ValueError("LIANA run is not the paper-matched sample primary panel")
    pseudobulk_path = _bound_output(liana_run, manifest, "pseudobulk_counts.h5ad")
    lr_path = _bound_output(liana_run, manifest, "liana_lr_results.tsv.gz")
    lr = pd.read_csv(lr_path, sep="\t").rename(
        columns={"source": "sender", "target": "receiver"}
    )
    keys = ["sender", "receiver", "interaction_id"]
    if lr.duplicated(keys).any() or set(map(tuple, lr[keys].to_numpy())) != set(
        map(tuple, edges[keys].to_numpy())
    ):
        raise ValueError("LIANA and RC3 directed LR universes differ")
    aligned_lr = edges.loc[:, keys].merge(
        lr.loc[:, [*keys, "interaction_stat", "interaction_pvalue"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    np.testing.assert_array_equal(
        aligned_lr["interaction_stat"].to_numpy(dtype=float),
        edges["interaction_stat"].to_numpy(dtype=float),
    )
    np.testing.assert_array_equal(
        aligned_lr["interaction_pvalue"].to_numpy(dtype=float),
        edges["interaction_pvalue"].to_numpy(dtype=float),
    )
    pdata = ad.read_h5ad(pseudobulk_path)
    try:
        presence, magnitude, design, support, edge_subject = (
            derive_subject_hurdle_inputs(pdata, edges)
        )
    finally:
        if getattr(pdata, "file", None) is not None:
            pdata.file.close()
    return (
        presence,
        magnitude,
        design,
        support,
        edge_subject,
        {
            "manifest": _input_record(manifest_path),
            "pseudobulk": _input_record(pseudobulk_path),
            "lr_results": _input_record(lr_path),
        },
    )


def _weighted_rankings(
    edges: pd.DataFrame,
    template: pd.DataFrame,
    weights: np.ndarray,
    *,
    target: str,
    reference: str,
    method: str,
    semantics: str,
) -> tuple[pd.DataFrame, int]:
    if len(edges) != len(weights):
        raise ValueError("edge and hurdle weight lengths differ")
    p_value = pd.to_numeric(edges["interaction_pvalue"], errors="coerce").to_numpy()
    sign = pd.to_numeric(edges["final_sign_statistic"], errors="coerce").to_numpy()
    selected = (
        np.isfinite(p_value) & np.isfinite(sign) & (p_value < 0.05) & (sign != 0.0)
    )
    sender = edges["sender"].astype(str).to_numpy()
    receiver = edges["receiver"].astype(str).to_numpy()
    calls = pd.DataFrame(
        {
            "condition": np.where(sign[selected] > 0.0, target, reference),
            "sender": np.minimum(sender[selected], receiver[selected]),
            "receiver": np.maximum(sender[selected], receiver[selected]),
            "weighted_strength": weights[selected],
        }
    )
    scores = (
        calls.groupby(["condition", "sender", "receiver"], observed=True, sort=True)[
            "weighted_strength"
        ]
        .sum()
        .reset_index()
    )
    result = template.merge(
        scores,
        on=["condition", "sender", "receiver"],
        how="left",
        validate="one_to_one",
    )
    observed = result["status"].eq("observed")
    result.loc[observed & result["weighted_strength"].isna(), "weighted_strength"] = 0.0
    result.loc[~observed, "weighted_strength"] = np.nan
    result["ranked_strength"] = result["weighted_strength"].astype("Float64")
    result["condition_specific_directed_lr"] = result["ranked_strength"]
    result["method"] = method
    result["method_version"] = METHOD_VERSION
    result["ranking_semantics"] = semantics
    return (
        result.drop(columns="weighted_strength").sort_values(
            ["dataset", "method", "condition", "sender", "receiver"],
            kind="stable",
            ignore_index=True,
        ),
        int(selected.sum()),
    )


def _assert_rank_parity(
    rebuilt: pd.DataFrame, original: pd.DataFrame
) -> dict[str, object]:
    identity = [
        "condition",
        "sender",
        "receiver",
        "estimable_directed_lr",
        "status",
        "reason_code",
    ]
    numeric = ["ranked_strength", "condition_specific_directed_lr"]
    order = ["condition", "sender", "receiver"]
    left = rebuilt.sort_values(order, kind="stable").reset_index(drop=True)
    right = original.sort_values(order, kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        left.loc[:, identity],
        right.loc[:, identity],
        check_dtype=False,
        check_exact=True,
    )
    maximum_absolute_difference = 0.0
    for column in numeric:
        left_value = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
        right_value = pd.to_numeric(right[column], errors="coerce").to_numpy(
            dtype=float
        )
        np.testing.assert_array_equal(np.isnan(left_value), np.isnan(right_value))
        finite = np.isfinite(left_value) & np.isfinite(right_value)
        if finite.any():
            maximum_absolute_difference = max(
                maximum_absolute_difference,
                float(np.max(np.abs(left_value[finite] - right_value[finite]))),
            )
        np.testing.assert_allclose(
            left_value,
            right_value,
            rtol=1e-13,
            atol=1e-13,
            equal_nan=True,
        )
    rank_order = ["condition", "ranked_strength", "sender", "receiver"]
    ascending = [True, False, True, True]
    left_order = (
        left.loc[left["ranked_strength"].notna()]
        .sort_values(rank_order, ascending=ascending, kind="stable")
        .loc[:, ["condition", "sender", "receiver"]]
        .reset_index(drop=True)
    )
    right_order = (
        right.loc[right["ranked_strength"].notna()]
        .sort_values(rank_order, ascending=ascending, kind="stable")
        .loc[:, ["condition", "sender", "receiver"]]
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(left_order, right_order, check_exact=True)
    return {
        "rankings_numerically_equal": True,
        "rank_order_exact": True,
        "ranking_rows": len(left),
        "rtol": 1e-13,
        "atol": 1e-13,
        "maximum_absolute_difference": maximum_absolute_difference,
    }


def _coverage_fallback(
    strict: pd.DataFrame, all_rankings: pd.DataFrame
) -> pd.DataFrame:
    result = strict.copy()
    observed = result["status"].eq("observed")
    result["_strict_rank"] = np.nan
    result.loc[observed, "_strict_rank"] = (
        result.loc[observed]
        .groupby("condition", observed=True)["ranked_strength"]
        .rank(pct=True, method="average")
    )
    fallback = all_rankings.loc[
        all_rankings["method"].eq("C"),
        [
            "condition",
            "sender",
            "receiver",
            "ranked_strength",
            "estimable_directed_lr",
            "status",
        ],
    ].copy()
    fallback_observed = fallback["status"].eq("observed")
    fallback["_fallback_rank"] = np.nan
    fallback.loc[fallback_observed, "_fallback_rank"] = (
        fallback.loc[fallback_observed]
        .groupby("condition", observed=True)["ranked_strength"]
        .rank(pct=True, method="average")
    )
    fallback = fallback.rename(columns={"estimable_directed_lr": "_fallback_estimable"})
    result = result.merge(
        fallback.loc[
            :,
            [
                "condition",
                "sender",
                "receiver",
                "_fallback_rank",
                "_fallback_estimable",
            ],
        ],
        on=["condition", "sender", "receiver"],
        how="left",
        validate="one_to_one",
    )
    use_fallback = ~observed & result["_fallback_rank"].notna()
    result["ranked_strength"] = result["_strict_rank"].where(
        observed, result["_fallback_rank"]
    )
    result.loc[use_fallback, "estimable_directed_lr"] = result.loc[
        use_fallback, "_fallback_estimable"
    ]
    result.loc[use_fallback, "condition_specific_directed_lr"] = np.nan
    result["status"] = np.where(
        result["ranked_strength"].notna(), "observed", "not_estimable"
    )
    result["reason_code"] = np.where(
        observed,
        None,
        np.where(
            use_fallback,
            "two_part_not_estimable_used_crychic_wald_rank_fallback",
            "two_part_and_crychic_fallback_not_estimable",
        ),
    )
    result["method"] = METHOD_FALLBACK
    result["method_version"] = METHOD_VERSION
    result["ranking_semantics"] = (
        "subject_two_part_program_soft_cardinality_percentile_with_crychic_"
        "wald_percentile_only_when_liana_pair_not_estimable"
    )
    return result.drop(
        columns=["_strict_rank", "_fallback_rank", "_fallback_estimable"]
    )


def run(
    *,
    dataset: str,
    rc3_run: Path,
    liana_run: Path,
    expected_path: Path,
    config_path: Path,
    acceptance_path: Path,
    frozen_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    frozen, config = _validate_frozen_candidate(
        config_path, acceptance_path, frozen_path
    )
    output = prepare_output(output_dir, overwrite=overwrite)
    edges, rc3_rankings, expected, rc3_provenance = _load_rc3(
        dataset, rc3_run.resolve(), expected_path.resolve()
    )
    contrast = cast(
        Mapping[str, Any], _read_json(rc3_run / "manifest.json")["contrast"]
    )
    target = str(contrast["target"])
    reference = str(contrast["reference"])
    strict_template = rc3_rankings.loc[
        rc3_rankings["method"].eq(RC3_METHOD_STRICT)
    ].copy()
    reconstructed_rc3, rc3_calls = _weighted_rankings(
        edges,
        strict_template,
        edges["program_weight"].to_numpy(dtype=float),
        target=target,
        reference=reference,
        method=RC3_METHOD_STRICT,
        semantics=str(strict_template["ranking_semantics"].iloc[0]),
    )
    rc3_parity = _assert_rank_parity(reconstructed_rc3, strict_template)
    rc3_parity["selected_calls"] = rc3_calls

    presence, magnitude, design, support, edge_subject, liana_provenance = (
        _load_liana_hurdle(
            liana_run.resolve(), edges, target=target, reference=reference
        )
    )
    hurdle_policy = cast(Mapping[str, Any], frozen["hurdle_policy"])
    effects = fit_subject_hurdle_effects(
        presence,
        magnitude,
        design["condition"].astype(str).to_numpy(),
        reference=reference,
        target=target,
        beta_prior=float(hurdle_policy["beta_prior"]),
        minimum_subjects=int(hurdle_policy["minimum_subjects"]),
        minimum_active_subjects=int(hurdle_policy["minimum_active_subjects"]),
        conditional_presence_threshold=float(
            hurdle_policy["conditional_presence_threshold"]
        ),
        magnitude_log_sd_floor=float(hurdle_policy["magnitude_log_sd_floor"]),
    )
    candidate = cast(Mapping[str, Any], frozen["candidate"])
    baseline = edges["interaction_stat"].to_numpy(dtype=float)
    occurrence_fit = fit_hurdle_channel_reliability(
        baseline,
        effects["occurrence_effect"].to_numpy(dtype=float),
        channel="occurrence",
        maximum_alpha=float(candidate["occurrence_alpha"]),
        alpha_cap=float(hurdle_policy["channel_alpha_cap"]),
        minimum_edges=int(hurdle_policy["minimum_concordance_edges"]),
    )
    magnitude_fit = fit_hurdle_channel_reliability(
        baseline,
        effects["magnitude_effect"].to_numpy(dtype=float),
        channel="magnitude",
        maximum_alpha=float(candidate["magnitude_alpha"]),
        alpha_cap=float(hurdle_policy["channel_alpha_cap"]),
        minimum_edges=int(hurdle_policy["minimum_concordance_edges"]),
    )
    final_sign = edges["final_sign_statistic"].to_numpy(dtype=float)
    direction = np.where(final_sign >= 0.0, 1.0, -1.0)
    hurdle_weight, occurrence_evidence, magnitude_evidence = hurdle_directional_weights(
        direction,
        effects["occurrence_z"].to_numpy(dtype=float),
        effects["magnitude_z"].to_numpy(dtype=float),
        occurrence_fit,
        magnitude_fit,
        log_weight_cap=float(hurdle_policy["log_weight_cap"]),
    )
    combined_weight = edges["program_weight"].to_numpy(dtype=float) * hurdle_weight
    strict, selected_calls = _weighted_rankings(
        edges,
        strict_template,
        combined_weight,
        target=target,
        reference=reference,
        method=METHOD_STRICT,
        semantics=(
            "exact_rc3_calls_weighted_by_subject_occurrence_and_positive_"
            "magnitude_hurdle;structural_absence_missing"
        ),
    )
    if selected_calls != rc3_calls:
        raise AssertionError("RC6 changed the frozen RC3 call set")
    fallback = _coverage_fallback(strict, rc3_rankings)
    all_rankings = pd.concat([rc3_rankings, strict, fallback], ignore_index=True)
    scores, coverage, summary = _evaluate(all_rankings, expected, dataset=dataset)

    identifiers = ["sender", "receiver", "interaction_id", "ligand", "receptor"]
    edge_effects = pd.concat(
        [edges.loc[:, identifiers].reset_index(drop=True), effects], axis=1
    )
    edge_effects["target_condition"] = target
    edge_effects["reference_condition"] = reference
    edge_effects["formal_inference_allowed"] = False
    edge_output = pd.concat(
        [edges.reset_index(drop=True), effects.add_prefix("hurdle_")], axis=1
    )
    edge_output["occurrence_evidence"] = occurrence_evidence
    edge_output["magnitude_evidence"] = magnitude_evidence
    edge_output["hurdle_weight"] = hurdle_weight
    edge_output["combined_program_hurdle_weight"] = combined_weight
    edge_output["selected_rc3_call"] = pd.to_numeric(
        edge_output["interaction_pvalue"], errors="coerce"
    ).lt(0.05) & pd.to_numeric(edge_output["final_sign_statistic"], errors="coerce").ne(
        0.0
    )
    edge_output["formal_inference_allowed"] = False

    paths: dict[str, tuple[Path, pd.DataFrame]] = {
        "edge_subject_hurdle.parquet": (
            output / "edge_subject_hurdle.parquet",
            edge_subject,
        ),
        "edge_occurrence_effects.parquet": (
            output / "edge_occurrence_effects.parquet",
            edge_effects,
        ),
        "directed_edge_two_part_weights.parquet": (
            output / "directed_edge_two_part_weights.parquet",
            edge_output,
        ),
        "subject_cell_type_support.tsv": (
            output / "subject_cell_type_support.tsv",
            support,
        ),
        "condition_cell_pair_rankings.tsv": (
            output / "condition_cell_pair_rankings.tsv",
            all_rankings,
        ),
        "spatial_des_scores.tsv": (output / "spatial_des_scores.tsv", scores),
        "spatial_des_coverage.tsv": (
            output / "spatial_des_coverage.tsv",
            coverage,
        ),
        "spatial_des_summary.tsv": (
            output / "spatial_des_summary.tsv",
            summary,
        ),
    }
    for filename, (path, table) in paths.items():
        if filename.endswith(".parquet"):
            table.to_parquet(path, index=False, compression="zstd")
        else:
            table.to_csv(path, sep="\t", index=False, lineterminator="\n")

    occurrence_observed = effects["occurrence_status"].eq("observed")
    magnitude_observed = effects["magnitude_status"].eq("observed")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "dataset": dataset,
        "contrast": {"target": target, "reference": reference},
        "frozen_candidate": frozen,
        "rc3_parity": rc3_parity,
        "occurrence_fit": occurrence_fit.to_dict(),
        "magnitude_fit": magnitude_fit.to_dict(),
        "alignment": {
            "rc3_edges": len(edges),
            "selected_rc3_calls": selected_calls,
            "subjects": int(design["subject_id"].nunique()),
            "subjects_by_condition": {
                str(key): int(value)
                for key, value in design.groupby("condition", observed=True)[
                    "subject_id"
                ]
                .nunique()
                .items()
            },
            "occurrence_observed_edges": int(occurrence_observed.sum()),
            "occurrence_coverage": float(occurrence_observed.mean()),
            "magnitude_observed_edges": int(magnitude_observed.sum()),
            "magnitude_coverage": float(magnitude_observed.mean()),
            "structural_absence_is_missing": True,
            "rc3_call_set_modified": False,
        },
        "backbone_modified": False,
        "benchmark_head_modified": True,
        "formal_release_allowed": False,
        "limitations": [
            "working_beta_and_normal_probabilities_are_benchmark_weights",
            "spatial_colocalization_is_an_indirect_proxy",
            "receiver_program_source_remains_partial_pipeline_evidence",
        ],
        "inputs": {
            "rc3": rc3_provenance,
            "liana": liana_provenance,
            "config": _input_record(config_path),
            "acceptance": _input_record(acceptance_path),
            "frozen_candidate": _input_record(frozen_path),
        },
        "performance": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_kib": int(
                process_resource.getrusage(process_resource.RUSAGE_SELF).ru_maxrss
            ),
        },
        "code": git_metadata(Path(__file__).resolve().parents[2]),
        "outputs": {
            filename: _output_record(path, table)
            for filename, (path, table) in paths.items()
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("kuppe", "ms"), required=True)
    parser.add_argument("--rc3-run", type=Path, required=True)
    parser.add_argument("--liana-run", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        dataset=args.dataset,
        rc3_run=args.rc3_run,
        liana_run=args.liana_run,
        expected_path=args.expected,
        config_path=args.config,
        acceptance_path=args.acceptance,
        frozen_path=args.frozen,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"],
                "occurrence_fit": result["occurrence_fit"],
                "magnitude_fit": result["magnitude_fit"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
