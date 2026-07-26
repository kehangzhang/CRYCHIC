"""Sparse Visium geometry diagnostics for indirect cell-pair validation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import norm, rankdata


@dataclass(frozen=True, slots=True)
class DistanceBand:
    name: str
    lower_exclusive: float
    upper_inclusive: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name != self.name.strip()
        ):
            raise ValueError("distance-band name must be canonical")
        lower = float(self.lower_exclusive)
        upper = float(self.upper_inclusive)
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or not 0 <= lower < upper
        ):
            raise ValueError("distance-band bounds must satisfy 0 <= lower < upper")
        object.__setattr__(self, "lower_exclusive", lower)
        object.__setattr__(self, "upper_inclusive", upper)


@dataclass(frozen=True, slots=True)
class BandGeometry:
    band: DistanceBand
    weights: sparse.csr_matrix
    unordered_spot_pairs: int
    mean_distance: float
    connected_components: int
    nonisolated_spots: int


@dataclass(frozen=True, slots=True)
class SectionGeometryResult:
    dataset: str
    sample_id: str
    subject_id: str
    condition: str
    cell_types: tuple[str, ...]
    bands: tuple[str, ...]
    observed: np.ndarray
    coordinate_null: np.ndarray
    pair_left: np.ndarray
    pair_right: np.ndarray
    band_audit: pd.DataFrame


def distance_bands(records: Sequence[Mapping[str, object]]) -> tuple[DistanceBand, ...]:
    bands = tuple(
        DistanceBand(
            name=str(record["name"]),
            lower_exclusive=float(record["lower_exclusive"]),
            upper_inclusive=float(record["upper_inclusive"]),
        )
        for record in records
    )
    if not bands or bands[0].name != "contact" or bands[0].lower_exclusive != 0.0:
        raise ValueError("distance bands must begin with the contact band at zero")
    if len({band.name for band in bands}) != len(bands):
        raise ValueError("distance-band names must be unique")
    for previous, current in pairwise(bands):
        if not math.isclose(
            previous.upper_inclusive, current.lower_exclusive, abs_tol=1e-12
        ):
            raise ValueError("distance bands must be contiguous and non-overlapping")
    return bands


def visium_hex_coordinates(table: pd.DataFrame) -> np.ndarray:
    """Convert Visium array indices to nearest-neighbor spot-pitch coordinates."""

    required = {"array_row", "array_col"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Visium coordinates are missing: {sorted(missing)}")
    numeric = table.loc[:, ["array_row", "array_col"]].apply(
        pd.to_numeric, errors="coerce"
    )
    values = numeric.to_numpy(dtype=float)
    if (
        len(values) < 3
        or not np.isfinite(values).all()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError("Visium array coordinates must be finite integer indices")
    integer = values.astype(np.int64)
    if len(np.unique(integer, axis=0)) != len(integer):
        raise ValueError("Visium array coordinates must be unique within section")
    if np.mod(integer[:, 0] + integer[:, 1], 2).any():
        raise ValueError("Visium row/column parity is invalid for a hex lattice")
    return np.column_stack([integer[:, 1] / 2.0, np.sqrt(3.0) * integer[:, 0] / 2.0])


def rank_normal_abundance(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Apply a robust within-section rank-normal transform by cell type."""

    if not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError("abundance must be a non-empty DataFrame")
    numeric = table.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0.0).any():
        raise ValueError("cell-type abundance must be finite and non-negative")
    transformed = np.full(numeric.shape, np.nan, dtype=np.float32)
    variable = np.zeros(numeric.shape[1], dtype=bool)
    for index in range(numeric.shape[1]):
        values = numeric[:, index]
        if np.ptp(values) <= 0.0:
            continue
        ranks = rankdata(values, method="average")
        quantiles = (ranks - 0.5) / len(values)
        scores = norm.ppf(quantiles)
        scale = float(np.std(scores, ddof=0))
        if not math.isfinite(scale) or scale <= 0.0:
            continue
        transformed[:, index] = ((scores - float(np.mean(scores))) / scale).astype(
            np.float32
        )
        variable[index] = True
    return transformed, variable


def _symmetric_adjacency(
    pairs: np.ndarray,
    *,
    spots: int,
) -> sparse.csr_matrix:
    if pairs.size == 0:
        return sparse.csr_matrix((spots, spots), dtype=np.float32)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    columns = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(spots, spots),
    )


def build_band_geometry(
    coordinates: np.ndarray,
    bands: Sequence[DistanceBand],
) -> tuple[BandGeometry, ...]:
    """Build sparse row-normalized bands without a dense spot-distance matrix."""

    coordinate_array = np.asarray(coordinates, dtype=float)
    if (
        coordinate_array.ndim != 2
        or coordinate_array.shape[1] != 2
        or len(coordinate_array) < 3
        or not np.isfinite(coordinate_array).all()
    ):
        raise ValueError("coordinates must be a finite n-by-2 matrix")
    declared = tuple(bands)
    if not declared or declared[0].name != "contact":
        raise ValueError("band geometry requires a leading contact band")
    maximum = max(band.upper_inclusive for band in declared)
    pairs = cKDTree(coordinate_array).query_pairs(maximum, output_type="ndarray")
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    distances = (
        np.linalg.norm(
            coordinate_array[pairs[:, 0]] - coordinate_array[pairs[:, 1]], axis=1
        )
        if len(pairs)
        else np.asarray([], dtype=float)
    )
    contact = declared[0]
    contact_pairs = pairs[
        (distances > contact.lower_exclusive) & (distances <= contact.upper_inclusive)
    ]
    contact_graph = _symmetric_adjacency(contact_pairs, spots=len(coordinate_array))
    component_count, labels = connected_components(
        contact_graph, directed=False, return_labels=True
    )
    results: list[BandGeometry] = []
    for band in declared:
        selected = (
            (distances > band.lower_exclusive)
            & (distances <= band.upper_inclusive)
            & (labels[pairs[:, 0]] == labels[pairs[:, 1]])
        )
        band_pairs = pairs[selected]
        adjacency = _symmetric_adjacency(band_pairs, spots=len(coordinate_array))
        row_mass = np.asarray(adjacency.sum(axis=1)).reshape(-1)
        inverse = np.divide(
            1.0,
            row_mass,
            out=np.zeros_like(row_mass, dtype=np.float32),
            where=row_mass > 0.0,
        )
        weights = (sparse.diags(inverse, format="csr") @ adjacency).tocsr()
        results.append(
            BandGeometry(
                band=band,
                weights=weights,
                unordered_spot_pairs=len(band_pairs),
                mean_distance=(
                    math.nan
                    if not selected.any()
                    else float(np.mean(distances[selected]))
                ),
                connected_components=int(component_count),
                nonisolated_spots=int(np.count_nonzero(row_mass)),
            )
        )
    return tuple(results)


def _cross_moran(z_scores: np.ndarray, geometry: BandGeometry) -> np.ndarray:
    mass = float(geometry.weights.sum())
    if mass <= 0.0:
        return np.full((z_scores.shape[1], z_scores.shape[1]), np.nan, dtype=float)
    lagged = geometry.weights @ z_scores
    raw = np.asarray(z_scores.T @ lagged, dtype=float) / mass
    return 0.5 * (raw + raw.T)


def _seed(root_seed: int, *parts: str) -> int:
    digest = hashlib.sha256(
        "\x1f".join((str(root_seed), *parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _canonical_cell_pair(
    cell_types: Sequence[str], left: int | np.integer, right: int | np.integer
) -> tuple[str, str]:
    first = str(cell_types[int(left)])
    second = str(cell_types[int(right)])
    return min(first, second), max(first, second)


def cell_type_permutations(
    cell_types: Sequence[str],
    *,
    replicates: int,
    seed: int,
    dataset: str,
) -> np.ndarray:
    names = tuple(map(str, cell_types))
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("cell-type permutation requires a unique axis")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 1
    ):
        raise ValueError("cell-label permutation replicates must be positive")
    rng = np.random.default_rng(_seed(seed, dataset, "cell_label"))
    return np.vstack([rng.permutation(len(names)) for _ in range(replicates)])


def compute_section_geometry(
    *,
    dataset: str,
    sample_id: str,
    subject_id: str,
    condition: str,
    coordinates: pd.DataFrame,
    abundance: pd.DataFrame,
    bands: Sequence[DistanceBand],
    coordinate_permutations: int,
    seed: int,
) -> SectionGeometryResult:
    """Compute one section and synchronized within-section coordinate nulls."""

    names = tuple(map(str, abundance.columns))
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("section cell-type names must be unique")
    if len(coordinates) != len(abundance):
        raise ValueError("section coordinates and abundance must share spot rows")
    if (
        isinstance(coordinate_permutations, bool)
        or not isinstance(coordinate_permutations, int)
        or coordinate_permutations < 1
    ):
        raise ValueError("coordinate_permutations must be positive")
    xy = visium_hex_coordinates(coordinates)
    z_scores, variable = rank_normal_abundance(abundance)
    geometries = build_band_geometry(xy, bands)
    pair_left, pair_right = np.triu_indices(len(names), k=1)
    observed = np.stack(
        [_cross_moran(z_scores, geometry) for geometry in geometries], axis=0
    )[:, pair_left, pair_right]
    invalid_pairs = ~(variable[pair_left] & variable[pair_right])
    observed[:, invalid_pairs] = np.nan
    coordinate_null = np.full(
        (coordinate_permutations, len(geometries), len(pair_left)),
        np.nan,
        dtype=np.float32,
    )
    rng = np.random.default_rng(_seed(seed, dataset, sample_id, "coordinate"))
    for replicate in range(coordinate_permutations):
        permuted = z_scores[rng.permutation(len(z_scores))]
        for band_index, geometry in enumerate(geometries):
            matrix = _cross_moran(permuted, geometry)
            coordinate_null[replicate, band_index] = matrix[pair_left, pair_right]
        coordinate_null[replicate, :, invalid_pairs] = np.nan
    audit = pd.DataFrame.from_records(
        {
            "dataset": dataset,
            "sample_id": sample_id,
            "subject_id": subject_id,
            "condition": condition,
            "band": geometry.band.name,
            "lower_exclusive": geometry.band.lower_exclusive,
            "upper_inclusive": geometry.band.upper_inclusive,
            "unordered_spot_pairs": geometry.unordered_spot_pairs,
            "mean_distance": geometry.mean_distance,
            "contact_connected_components": geometry.connected_components,
            "nonisolated_spots": geometry.nonisolated_spots,
            "spots": len(z_scores),
            "variable_cell_types": int(variable.sum()),
        }
        for geometry in geometries
    )
    return SectionGeometryResult(
        dataset=dataset,
        sample_id=sample_id,
        subject_id=subject_id,
        condition=condition,
        cell_types=names,
        bands=tuple(geometry.band.name for geometry in geometries),
        observed=observed,
        coordinate_null=coordinate_null,
        pair_left=pair_left.astype(np.int16),
        pair_right=pair_right.astype(np.int16),
        band_audit=audit,
    )


def section_association_table(result: SectionGeometryResult) -> pd.DataFrame:
    """Materialize sample-level geometry and coordinate-null diagnostics."""

    records: list[dict[str, object]] = []
    for band_index, band in enumerate(result.bands):
        null = result.coordinate_null[:, band_index, :].astype(float)
        observed = result.observed[band_index].astype(float)
        maximum = np.nanmax(np.abs(null), axis=1)
        for pair_index, (left, right) in enumerate(
            zip(result.pair_left, result.pair_right, strict=True)
        ):
            sender, receiver = _canonical_cell_pair(result.cell_types, left, right)
            value = observed[pair_index]
            draws = null[:, pair_index]
            finite = draws[np.isfinite(draws)]
            estimable = math.isfinite(value) and len(finite) == len(draws)
            null_sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan
            records.append(
                {
                    "dataset": result.dataset,
                    "sample_id": result.sample_id,
                    "subject_id": result.subject_id,
                    "condition": result.condition,
                    "band": band,
                    "sender": sender,
                    "receiver": receiver,
                    "association": value if estimable else math.nan,
                    "coordinate_null_mean": (
                        math.nan if not estimable else float(np.mean(finite))
                    ),
                    "coordinate_null_sd": null_sd if estimable else math.nan,
                    "coordinate_null_z": (
                        (value - float(np.mean(finite))) / null_sd
                        if estimable and math.isfinite(null_sd) and null_sd > 0.0
                        else math.nan
                    ),
                    "coordinate_empirical_p_value": (
                        math.nan
                        if not estimable
                        else (1 + int(np.sum(np.abs(finite) >= abs(value))))
                        / (len(finite) + 1)
                    ),
                    "coordinate_max_t_p_value": (
                        math.nan
                        if not estimable
                        else (1 + int(np.sum(maximum >= abs(value))))
                        / (len(maximum) + 1)
                    ),
                    "coordinate_permutations": len(draws),
                    "status": "observed" if estimable else "not_estimable",
                    "reason_code": None
                    if estimable
                    else "geometry_or_null_not_estimable",
                    "claim_scope": "indirect_spot_geometry_diagnostic_only",
                }
            )
    return pd.DataFrame.from_records(records)


def _aggregate_units(
    observed: np.ndarray,
    null: np.ndarray,
    metadata: pd.DataFrame,
    *,
    unit: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if unit not in {"sample_id", "subject_id"}:
        raise ValueError("geometry analysis unit must be sample_id or subject_id")
    groups = metadata.groupby(unit, observed=True, sort=True).indices
    unit_observed: list[np.ndarray] = []
    unit_null: list[np.ndarray] = []
    records: list[dict[str, str]] = []
    for unit_id, indices in groups.items():
        index = np.asarray(indices, dtype=int)
        conditions = set(metadata.iloc[index]["condition"].astype(str))
        if len(conditions) != 1:
            raise ValueError("one spatial analysis unit cannot span conditions")
        unit_observed.append(np.nanmean(observed[index], axis=0))
        unit_null.append(np.nanmean(null[index], axis=0))
        records.append({"unit_id": str(unit_id), "condition": conditions.pop()})
    return (
        np.stack(unit_observed),
        np.stack(unit_null),
        pd.DataFrame.from_records(records),
    )


def condition_geometry_effects(
    results: Sequence[SectionGeometryResult],
    *,
    reference: str,
    target: str,
    analysis_unit: str,
    label_permutations: np.ndarray,
    minimum_units_per_condition: int,
) -> pd.DataFrame:
    """Compare conditions after section-to-unit aggregation with synchronized nulls."""

    items = tuple(results)
    if not items:
        raise ValueError("condition geometry requires section results")
    first = items[0]
    for item in items[1:]:
        if (
            item.cell_types != first.cell_types
            or item.bands != first.bands
            or not np.array_equal(item.pair_left, first.pair_left)
            or not np.array_equal(item.pair_right, first.pair_right)
            or item.coordinate_null.shape[0] != first.coordinate_null.shape[0]
        ):
            raise ValueError("section geometry axes differ within dataset")
    observed = np.stack([item.observed for item in items])
    null = np.stack([item.coordinate_null for item in items])
    metadata = pd.DataFrame.from_records(
        {
            "sample_id": item.sample_id,
            "subject_id": item.subject_id,
            "condition": item.condition,
        }
        for item in items
    )
    unit_observed, unit_null, units = _aggregate_units(
        observed, null, metadata, unit=analysis_unit
    )
    reference_mask = units["condition"].eq(reference).to_numpy()
    target_mask = units["condition"].eq(target).to_numpy()
    n_reference = int(reference_mask.sum())
    n_target = int(target_mask.sum())
    if min(n_reference, n_target) < minimum_units_per_condition:
        raise ValueError("insufficient spatial units per condition")
    reference_mean = np.nanmean(unit_observed[reference_mask], axis=0)
    target_mean = np.nanmean(unit_observed[target_mask], axis=0)
    effect = target_mean - reference_mean
    null_effect = np.nanmean(unit_null[target_mask], axis=0) - np.nanmean(
        unit_null[reference_mask], axis=0
    )
    label_permutations = np.asarray(label_permutations, dtype=int)
    if label_permutations.ndim != 2 or label_permutations.shape[1] != len(
        first.cell_types
    ):
        raise ValueError("cell-label permutation axis differs from cell types")
    label_null = np.full(
        (len(label_permutations), len(first.bands), len(first.pair_left)),
        np.nan,
        dtype=float,
    )
    for band_index in range(len(first.bands)):
        matrix = np.full(
            (len(first.cell_types), len(first.cell_types)), np.nan, dtype=float
        )
        matrix[first.pair_left, first.pair_right] = effect[band_index]
        matrix[first.pair_right, first.pair_left] = effect[band_index]
        for replicate, permutation in enumerate(label_permutations):
            permuted = matrix[np.ix_(permutation, permutation)]
            label_null[replicate, band_index] = permuted[
                first.pair_left, first.pair_right
            ]
    records: list[dict[str, object]] = []
    for band_index, band in enumerate(first.bands):
        coordinate_maximum = np.nanmax(np.abs(null_effect[:, band_index]), axis=1)
        label_maximum = np.nanmax(np.abs(label_null[:, band_index]), axis=1)
        for pair_index, (left, right) in enumerate(
            zip(first.pair_left, first.pair_right, strict=True)
        ):
            sender, receiver = _canonical_cell_pair(first.cell_types, left, right)
            value = float(effect[band_index, pair_index])
            coordinate_draws = null_effect[:, band_index, pair_index].astype(float)
            label_draws = label_null[:, band_index, pair_index].astype(float)
            coordinate_finite = coordinate_draws[np.isfinite(coordinate_draws)]
            label_finite = label_draws[np.isfinite(label_draws)]
            observed_row = (
                math.isfinite(value)
                and len(coordinate_finite) == len(coordinate_draws)
                and len(label_finite) == len(label_draws)
            )
            coordinate_sd = (
                float(np.std(coordinate_finite, ddof=1))
                if len(coordinate_finite) > 1
                else math.nan
            )
            records.append(
                {
                    "dataset": first.dataset,
                    "analysis_unit": analysis_unit,
                    "band": band,
                    "sender": sender,
                    "receiver": receiver,
                    "reference": reference,
                    "target": target,
                    "reference_mean": float(reference_mean[band_index, pair_index]),
                    "target_mean": float(target_mean[band_index, pair_index]),
                    "effect_target_minus_reference": value
                    if observed_row
                    else math.nan,
                    "absolute_effect": abs(value) if observed_row else math.nan,
                    "coordinate_null_mean": (
                        math.nan
                        if not observed_row
                        else float(np.mean(coordinate_finite))
                    ),
                    "coordinate_null_sd": coordinate_sd if observed_row else math.nan,
                    "coordinate_null_z": (
                        (value - float(np.mean(coordinate_finite))) / coordinate_sd
                        if observed_row
                        and math.isfinite(coordinate_sd)
                        and coordinate_sd > 0.0
                        else math.nan
                    ),
                    "coordinate_empirical_p_value": (
                        math.nan
                        if not observed_row
                        else (1 + int(np.sum(np.abs(coordinate_finite) >= abs(value))))
                        / (len(coordinate_finite) + 1)
                    ),
                    "coordinate_max_t_p_value": (
                        math.nan
                        if not observed_row
                        else (1 + int(np.sum(coordinate_maximum >= abs(value))))
                        / (len(coordinate_maximum) + 1)
                    ),
                    "cell_label_empirical_p_value": (
                        math.nan
                        if not observed_row
                        else (1 + int(np.sum(np.abs(label_finite) >= abs(value))))
                        / (len(label_finite) + 1)
                    ),
                    "cell_label_max_t_p_value": (
                        math.nan
                        if not observed_row
                        else (1 + int(np.sum(label_maximum >= abs(value))))
                        / (len(label_maximum) + 1)
                    ),
                    "n_reference_units": n_reference,
                    "n_target_units": n_target,
                    "coordinate_permutations": len(coordinate_draws),
                    "cell_label_permutations": len(label_draws),
                    "status": "observed" if observed_row else "not_estimable",
                    "reason_code": (
                        None
                        if observed_row
                        else "condition_geometry_or_null_not_estimable"
                    ),
                    "formal_inference_allowed": False,
                    "claim_scope": "indirect_spot_geometry_diagnostic_only",
                }
            )
    result = pd.DataFrame.from_records(records)
    result["geometry_rank"] = pd.NA
    observed_mask = result["status"].eq("observed")
    result.loc[observed_mask, "geometry_rank"] = (
        result.loc[observed_mask]
        .groupby(["analysis_unit", "band"], observed=True)["absolute_effect"]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )
    result["geometry_rank"] = result["geometry_rank"].astype("Int64")
    return result.sort_values(
        ["analysis_unit", "band", "geometry_rank", "sender", "receiver"],
        kind="stable",
        na_position="last",
        ignore_index=True,
    )


def distance_decay_table(
    associations: pd.DataFrame,
    band_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Fit an indirect association-decay diagnostic across frozen distance bands."""

    means = band_audit.loc[:, ["sample_id", "band", "mean_distance"]]
    merged = associations.merge(
        means,
        on=["sample_id", "band"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    records: list[dict[str, object]] = []
    keys = ["dataset", "sample_id", "subject_id", "condition", "sender", "receiver"]
    for key, group in merged.groupby(keys, observed=True, sort=True):
        observed = group.loc[
            group["status"].eq("observed")
            & np.isfinite(pd.to_numeric(group["mean_distance"], errors="coerce"))
        ].sort_values("mean_distance")
        slope = excess_auc = contact_minus_farthest = math.nan
        if len(observed) >= 3:
            distance = np.log1p(observed["mean_distance"].to_numpy(dtype=float))
            magnitude = observed["association"].abs().to_numpy(dtype=float)
            slope = float(np.polyfit(distance, magnitude, 1)[0])
            z_value = pd.to_numeric(
                observed["coordinate_null_z"], errors="coerce"
            ).to_numpy(dtype=float)
            if np.isfinite(z_value).all():
                excess_auc = float(np.trapezoid(np.maximum(z_value, 0.0), distance))
            contact_minus_farthest = float(magnitude[0] - magnitude[-1])
        records.append(
            {
                **dict(zip(keys, key, strict=True)),
                "bands_observed": len(observed),
                "absolute_association_slope_per_log_distance": slope,
                "contact_minus_farthest_absolute_association": contact_minus_farthest,
                "positive_coordinate_excess_auc": excess_auc,
                "status": "observed" if math.isfinite(slope) else "not_estimable",
                "claim_scope": "indirect_spot_geometry_diagnostic_only",
            }
        )
    return pd.DataFrame.from_records(records)


def expected_geometry_sets(
    effects: pd.DataFrame,
    *,
    top_fractions: Sequence[float],
) -> pd.DataFrame:
    """Encode a full fixed pair universe for downstream spatial DES evaluation."""

    fractions = tuple(sorted(set(map(float, top_fractions))))
    if not fractions or any(not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("geometry top fractions must lie in (0, 1]")
    records: list[pd.DataFrame] = []
    grouping = ["dataset", "analysis_unit", "band"]
    for _, group in effects.groupby(grouping, observed=True, sort=True):
        local = group.copy()
        eligible = local["status"].eq("observed") & local[
            "effect_target_minus_reference"
        ].ne(0.0)
        eligible_rows = local.loc[eligible].sort_values(
            ["absolute_effect", "sender", "receiver"],
            ascending=[False, True, True],
            kind="stable",
        )
        for fraction in fractions:
            count = int(math.floor(fraction * len(eligible_rows)))
            selected = set(eligible_rows.head(count).index)
            encoded = local.copy()
            encoded["top_fraction"] = fraction
            encoded["is_expected"] = encoded.index.isin(selected)
            encoded["expected_condition"] = np.select(
                (
                    encoded["effect_target_minus_reference"].gt(0.0),
                    encoded["effect_target_minus_reference"].lt(0.0),
                ),
                (encoded["target"], encoded["reference"]),
                default="tied",
            )
            records.append(encoded)
    return pd.concat(records, ignore_index=True) if records else effects.head(0)


__all__ = [
    "BandGeometry",
    "DistanceBand",
    "SectionGeometryResult",
    "build_band_geometry",
    "cell_type_permutations",
    "compute_section_geometry",
    "condition_geometry_effects",
    "distance_bands",
    "distance_decay_table",
    "expected_geometry_sets",
    "rank_normal_abundance",
    "section_association_table",
    "visium_hex_coordinates",
]
