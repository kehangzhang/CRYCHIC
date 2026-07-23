"""Build event-level DES ranking variants without changing the CRYCHIC core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

PAIR_KEYS = ("condition", "sender", "receiver")
EVENT_BUDGETS = (100, 250, 500, 1000)
MECHANISMS = ("contact", "ecm_receptor", "secreted")


def _canonical_pair(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sender = table["sender"].astype(str).to_numpy()
    receiver = table["receiver"].astype(str).to_numpy()
    return np.minimum(sender, receiver), np.maximum(sender, receiver)


def _validate_pair_axes(table: pd.DataFrame, *, label: str) -> None:
    required = {*PAIR_KEYS, "ranked_strength", "status"}
    missing = required.difference(table.columns)
    if missing or table.empty:
        raise ValueError(f"{label} is empty or missing columns: {sorted(missing)}")
    if table.duplicated(list(PAIR_KEYS)).any():
        raise ValueError(f"{label} pair axes must be unique")
    status = table["status"].astype(str)
    score = pd.to_numeric(table["ranked_strength"], errors="coerce")
    if score.loc[status.eq("observed")].isna().any():
        raise ValueError(f"{label} observed pair axes require finite scores")


def mechanism_annotations(resource: pd.DataFrame) -> pd.DataFrame:
    """Return an unambiguous ligand/receptor mechanism lookup."""

    required = {"ligand", "receptor", "ligand_location"}
    missing = required.difference(resource.columns)
    if missing or resource.empty:
        raise ValueError(f"resource is empty or missing columns: {sorted(missing)}")
    lookup = resource.loc[:, ["ligand", "receptor", "ligand_location"]].copy()
    if lookup.duplicated(["ligand", "receptor"]).any():
        raise ValueError("resource ligand/receptor keys must be unique")
    mapping = {
        "plasma membrane": "contact",
        "ECM": "ecm_receptor",
        "secreted": "secreted",
        "plasma membrane; secreted": "ambiguous_contact_secreted",
    }
    lookup["mechanism"] = lookup["ligand_location"].map(mapping).fillna(
        "unclassified"
    )
    return lookup


def bounded_pair_gate(pair_rankings: pd.DataFrame, *, floor: float) -> pd.DataFrame:
    """Map within-condition RC11 pair ranks to a bounded multiplicative gate."""

    if not np.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError("pair gate floor must lie in [0, 1]")
    _validate_pair_axes(pair_rankings, label="RC11 ranking")
    result = pair_rankings.loc[:, [*PAIR_KEYS, "ranked_strength", "status"]].copy()
    pair_sender, pair_receiver = _canonical_pair(result)
    result["sender"] = pair_sender
    result["receiver"] = pair_receiver
    if result.duplicated(list(PAIR_KEYS)).any():
        raise ValueError("RC11 ranking contains duplicate unordered pair axes")
    observed = result["status"].astype(str).eq("observed")
    result["pair_percentile"] = np.nan
    result.loc[observed, "pair_percentile"] = result.loc[observed].groupby(
        "condition", sort=False, observed=True
    )["ranked_strength"].rank(method="average", pct=True, ascending=True)
    result["pair_gate"] = floor + (1.0 - floor) * result["pair_percentile"]
    result.loc[~observed, ["pair_percentile", "pair_gate"]] = np.nan
    if not result.loc[observed, "pair_gate"].between(floor, 1.0).all():
        raise AssertionError("bounded pair gate escaped its declared interval")
    return result.loc[:, [*PAIR_KEYS, "pair_percentile", "pair_gate", "status"]]


def _join_mechanism(
    events: pd.DataFrame, mechanism_lookup: pd.DataFrame
) -> pd.DataFrame:
    result = events.merge(
        mechanism_lookup,
        on=["ligand", "receptor"],
        how="left",
        validate="many_to_one",
    )
    result["ligand_location"] = result["ligand_location"].fillna("unmapped")
    result["mechanism"] = result["mechanism"].fillna("unclassified")
    return result


def prepare_crychic_event_ledger(
    events: pd.DataFrame,
    pair_rankings: pd.DataFrame,
    mechanism_lookup: pd.DataFrame,
    *,
    pair_gate_floor: float,
) -> pd.DataFrame:
    """Normalize CRYCHIC directed effects into a condition-assigned event ledger."""

    required = {
        "sender",
        "receiver",
        "interaction_id",
        "ligand",
        "receptor",
        "reference_condition",
        "target_condition",
        "effect_target_minus_reference",
        "effect_signal_to_noise",
        "one_standard_error_stable",
        "status",
        "reason_code",
    }
    missing = required.difference(events.columns)
    if missing or events.empty:
        raise ValueError(
            f"CRYCHIC events are empty or missing columns: {sorted(missing)}"
        )
    event_keys = ["sender", "receiver", "interaction_id"]
    if events.duplicated(event_keys).any():
        raise ValueError("CRYCHIC directed event keys must be unique")
    conditions = events.loc[
        :, ["reference_condition", "target_condition"]
    ].drop_duplicates()
    if len(conditions) != 1:
        raise ValueError("CRYCHIC event ledger must contain exactly one contrast")
    reference = str(conditions.iloc[0]["reference_condition"])
    target = str(conditions.iloc[0]["target_condition"])
    result = events.loc[:, sorted(required)].copy()
    effect = pd.to_numeric(result["effect_target_minus_reference"], errors="coerce")
    snr = pd.to_numeric(result["effect_signal_to_noise"], errors="coerce")
    observed = result["status"].astype(str).eq("observed")
    if effect.loc[observed].isna().any() or snr.loc[observed].isna().any():
        raise ValueError("observed CRYCHIC events require finite effect and SNR")
    if np.isinf(effect.loc[observed]).any() or np.isinf(snr.loc[observed]).any():
        raise ValueError("observed CRYCHIC event statistics must be finite")
    result["condition"] = np.select(
        (effect.gt(0.0), effect.lt(0.0)), (target, reference), default="tied"
    )
    result["abs_effect"] = effect.abs()
    result["event_evidence"] = snr.abs()
    result["native_selected"] = (
        observed
        & effect.ne(0.0)
        & result["one_standard_error_stable"].fillna(False).astype(bool)
    )
    pair_sender, pair_receiver = _canonical_pair(result)
    result["pair_sender"] = pair_sender
    result["pair_receiver"] = pair_receiver
    gate = bounded_pair_gate(pair_rankings, floor=pair_gate_floor).rename(
        columns={"sender": "pair_sender", "receiver": "pair_receiver"}
    )
    result = result.merge(
        gate.loc[
            :,
            [
                "condition",
                "pair_sender",
                "pair_receiver",
                "pair_percentile",
                "pair_gate",
            ],
        ],
        on=["condition", "pair_sender", "pair_receiver"],
        how="left",
        validate="many_to_one",
    )
    active = observed & effect.ne(0.0)
    if result.loc[active, "pair_gate"].isna().any():
        raise ValueError("an observed nonzero CRYCHIC event lacks its RC11 pair gate")
    result["bounded_event_evidence"] = result["event_evidence"] * result["pair_gate"]
    result["bounded_abs_effect"] = result["abs_effect"] * result["pair_gate"]
    return _join_mechanism(result, mechanism_lookup).sort_values(
        ["condition", "sender", "receiver", "interaction_id"],
        kind="stable",
        ignore_index=True,
    )


def prepare_scseqcommdiff_event_ledger(
    events: pd.DataFrame,
    mechanism_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize an exported scSeqCommDiff event ledger."""

    required = {
        "sender",
        "receiver",
        "ligand",
        "receptor",
        "score_target",
        "score_reference",
        "effect_target_minus_reference",
        "logFC",
        "p_value",
        "max_S_intra",
        "reference_condition",
        "target_condition",
        "status",
        "reason_code",
    }
    missing = required.difference(events.columns)
    if missing or events.empty:
        raise ValueError(
            f"scSeqCommDiff events are empty or missing columns: {sorted(missing)}"
        )
    keys = ["sender", "receiver", "ligand", "receptor"]
    if events.duplicated(keys).any():
        raise ValueError("scSeqCommDiff directed event keys must be unique")
    conditions = events.loc[
        :, ["reference_condition", "target_condition"]
    ].drop_duplicates()
    if len(conditions) != 1:
        raise ValueError("scSeqCommDiff event ledger must contain one contrast")
    reference = str(conditions.iloc[0]["reference_condition"])
    target = str(conditions.iloc[0]["target_condition"])
    result = events.copy()
    effect = pd.to_numeric(result["effect_target_minus_reference"], errors="coerce")
    log_fc = pd.to_numeric(result["logFC"], errors="coerce")
    p_value = pd.to_numeric(result["p_value"], errors="coerce")
    observed = result["status"].astype(str).eq("observed")
    if log_fc.loc[observed].isna().any() or p_value.loc[observed].isna().any():
        raise ValueError("observed scSeqCommDiff events require native statistics")
    if np.isinf(p_value.loc[observed]).any():
        raise ValueError("observed scSeqCommDiff p-values must be finite")
    intracellular = pd.to_numeric(result["max_S_intra"], errors="coerce")
    gate_eligible = intracellular.isna() | intracellular.gt(0.5)
    result["condition"] = np.select(
        (log_fc.gt(0.0), log_fc.lt(0.0)), (target, reference), default="tied"
    )
    result["abs_effect"] = effect.abs()
    tiny = np.finfo(float).tiny
    result["event_evidence"] = -np.log10(p_value.clip(lower=tiny))
    result["native_selected"] = (
        observed & gate_eligible & p_value.lt(0.05) & log_fc.ne(0.0)
    )
    result["top_k_eligible"] = observed & gate_eligible & log_fc.ne(0.0)
    continuous_estimable = gate_eligible & effect.notna() & np.isfinite(effect)
    result["continuous_weight"] = effect.abs().where(continuous_estimable)
    pair_sender, pair_receiver = _canonical_pair(result)
    result["pair_sender"] = pair_sender
    result["pair_receiver"] = pair_receiver
    return _join_mechanism(result, mechanism_lookup).sort_values(
        ["condition", "sender", "receiver", "ligand", "receptor"],
        kind="stable",
        ignore_index=True,
    )


def select_top_k_events(
    ledger: pd.DataFrame,
    *,
    budget: int,
    evidence_column: str,
    eligible: pd.Series,
) -> pd.Series:
    """Select one deterministic global event budget across both directions."""

    if isinstance(budget, bool) or budget < 1:
        raise ValueError("event budget must be a positive integer")
    if evidence_column not in ledger:
        raise ValueError(f"event evidence column is missing: {evidence_column}")
    eligible = eligible.reindex(ledger.index, fill_value=False).astype(bool)
    evidence = pd.to_numeric(ledger[evidence_column], errors="coerce")
    active = eligible & ledger["condition"].astype(str).ne("tied") & evidence.notna()
    if np.isinf(evidence.loc[active]).any():
        raise ValueError("top-K event evidence must be finite")
    if int(active.sum()) < budget:
        raise ValueError(
            f"top-K event pool has {int(active.sum())} rows, fewer than K={budget}"
        )
    tie_columns = [
        "condition",
        "sender",
        "receiver",
        "ligand",
        "receptor",
    ]
    if "interaction_id" in ledger:
        tie_columns.append("interaction_id")
    ranked = ledger.loc[active].assign(_evidence=evidence.loc[active]).sort_values(
        ["_evidence", "abs_effect", *tie_columns],
        ascending=[False, False, *([True] * len(tie_columns))],
        kind="stable",
    )
    selected = pd.Series(False, index=ledger.index, dtype=bool)
    selected.loc[ranked.index[:budget]] = True
    return selected


def pair_rankings_from_events(
    ledger: pd.DataFrame,
    pair_axes: pd.DataFrame,
    *,
    selected: pd.Series,
    weight_column: str | None,
    metadata: Mapping[str, str],
) -> pd.DataFrame:
    """Collapse selected directed LR events to paper-compatible unordered pairs."""

    _validate_pair_axes(pair_axes, label="pair axes")
    required_metadata = {
        "dataset",
        "method",
        "method_version",
        "resource",
        "ranking_semantics",
    }
    missing_metadata = required_metadata.difference(metadata)
    if missing_metadata:
        raise ValueError(f"ranking metadata is missing: {sorted(missing_metadata)}")
    selected = selected.reindex(ledger.index, fill_value=False).astype(bool)
    if weight_column is None:
        weight = pd.Series(1.0, index=ledger.index)
    else:
        if weight_column not in ledger:
            raise ValueError(f"event weight column is missing: {weight_column}")
        weight = pd.to_numeric(ledger[weight_column], errors="coerce")
    active = selected & ledger["condition"].astype(str).ne("tied")
    if weight.loc[active].isna().any() or np.isinf(weight.loc[active]).any():
        raise ValueError("selected event weights must be finite")
    if (weight.loc[active] < 0.0).any():
        raise ValueError("selected event weights must be non-negative")
    working = ledger.loc[
        active, ["condition", "pair_sender", "pair_receiver"]
    ].copy()
    working["event_weight"] = weight.loc[active].to_numpy(dtype=float)
    grouped = (
        working.groupby(
            ["condition", "pair_sender", "pair_receiver"],
            sort=True,
            observed=True,
            as_index=False,
        )
        .agg(
            ranked_strength=("event_weight", "sum"),
            condition_specific_directed_lr=("event_weight", "size"),
        )
        .rename(columns={"pair_sender": "sender", "pair_receiver": "receiver"})
    )
    axes = pair_axes.loc[:, [*PAIR_KEYS, "status"]].copy()
    pair_sender, pair_receiver = _canonical_pair(axes)
    axes["sender"] = pair_sender
    axes["receiver"] = pair_receiver
    if axes.duplicated(list(PAIR_KEYS)).any():
        raise ValueError("pair axes duplicate after unordered canonicalization")
    result = axes.merge(
        grouped,
        on=list(PAIR_KEYS),
        how="left",
        validate="one_to_one",
    )
    observed = result["status"].astype(str).eq("observed")
    result.loc[observed, "ranked_strength"] = result.loc[
        observed, "ranked_strength"
    ].fillna(0.0)
    result.loc[observed, "condition_specific_directed_lr"] = result.loc[
        observed, "condition_specific_directed_lr"
    ].fillna(0).astype(int)
    missing_columns = ["ranked_strength", "condition_specific_directed_lr"]
    result.loc[~observed, missing_columns] = np.nan
    result["reason_code"] = np.where(observed, None, "pair_not_estimable_in_source")
    for index, column in enumerate(
        ("dataset", "method", "method_version", "resource", "ranking_semantics")
    ):
        result.insert(index, column, metadata[column])
    return result.sort_values(
        ["condition", "sender", "receiver"], kind="stable", ignore_index=True
    )


def assert_original_count_parity(
    rebuilt: pd.DataFrame, native: pd.DataFrame
) -> None:
    """Require exact pair-count parity with a persisted native ranking."""

    _validate_pair_axes(rebuilt, label="rebuilt original count")
    _validate_pair_axes(native, label="native original count")
    columns = [*PAIR_KEYS, "ranked_strength", "status"]
    left = rebuilt.loc[:, columns].sort_values(list(PAIR_KEYS), kind="stable")
    right = native.loc[:, columns].sort_values(list(PAIR_KEYS), kind="stable")
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    if not left.loc[:, list(PAIR_KEYS)].equals(right.loc[:, list(PAIR_KEYS)]):
        raise ValueError("rebuilt and native original-count axes disagree")
    if not left["status"].astype(str).equals(right["status"].astype(str)):
        raise ValueError("rebuilt and native original-count statuses disagree")
    left_score = pd.to_numeric(left["ranked_strength"], errors="coerce").to_numpy()
    right_score = pd.to_numeric(right["ranked_strength"], errors="coerce").to_numpy()
    if not np.allclose(left_score, right_score, rtol=0, atol=0, equal_nan=True):
        raise ValueError("rebuilt and native original-count values disagree")


def selected_event_diagnostics(
    ledger: pd.DataFrame,
    selections: Mapping[str, pd.Series],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for variant, selected in selections.items():
        selected = selected.reindex(ledger.index, fill_value=False).astype(bool)
        for condition, group in ledger.loc[selected].groupby(
            "condition", sort=True, observed=True
        ):
            records.append(
                {
                    "variant": variant,
                    "condition": condition,
                    "selected_events": len(group),
                    "selected_unordered_pairs": group.loc[
                        :, ["pair_sender", "pair_receiver"]
                    ].drop_duplicates().shape[0],
                    "median_abs_effect": float(group["abs_effect"].median()),
                    "median_event_evidence": float(group["event_evidence"].median()),
                }
            )
    return pd.DataFrame.from_records(records).sort_values(
        ["variant", "condition"], kind="stable", ignore_index=True
    )


__all__: Sequence[str] = (
    "EVENT_BUDGETS",
    "MECHANISMS",
    "assert_original_count_parity",
    "bounded_pair_gate",
    "mechanism_annotations",
    "pair_rankings_from_events",
    "prepare_crychic_event_ledger",
    "prepare_scseqcommdiff_event_ledger",
    "select_top_k_events",
    "selected_event_diagnostics",
)
