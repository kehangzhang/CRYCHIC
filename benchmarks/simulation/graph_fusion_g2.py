"""Scalable paired simulation for the preregistered graph-fusion G2 gate."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse

from benchmarks.metrics.graph_fusion_g2 import (
    FUSED_METHOD,
    METHODS,
    NO_TOPOLOGY_METHOD,
    PRIMARY_METRIC,
    UNFUSED_METHOD,
    WRONG_TOPOLOGY_METHOD,
    G2GateSpecification,
    driver_family_context_macro_auprc,
    driver_family_context_macro_auroc,
    graph_jump_localization_auprc,
)
from crychic.attribution import solve_graph_fused_nonnegative_elastic_net
from crychic.core import canonical_digest, stable_id
from crychic.design import ContextGraph

G2_BENCHMARK_SCHEMA_VERSION = "crychic-g2-graph-fusion-benchmark-v1"
G2_RESULT_SCHEMA_VERSION = "crychic-g2-graph-fusion-results-v1"
FROZEN_TOPOLOGIES = ("chain", "product")
FROZEN_SIGNALS = ("smooth_gradient", "single_local_jump")
FROZEN_SNR_NAMES = ("low", "high")
FROZEN_COLLINEARITY_NAMES = ("low", "high")
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "graph_fusion_g2_v1.json"
)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an array")
    return cast(Sequence[object], value)


def _required_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class G2FactorLevel:
    """One named numerical level in the frozen scenario factorial."""

    name: str
    value: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("factor level name must be non-empty")
        if not math.isfinite(self.value):
            raise ValueError("factor level value must be finite")


@dataclass(frozen=True, slots=True)
class G2ScenarioCell:
    """One equal-weight cell in the frozen 2 x 2 x 2 x 2 manifest."""

    topology: str
    signal: str
    snr_name: str
    snr: float
    collinearity_name: str
    collinearity: float
    scenario_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class G2BenchmarkManifest:
    """Validated benchmark manifest, including solver and decision policy."""

    source_path: Path
    source_sha256: str
    benchmark_id: str
    root_seed: int
    default_profile: str
    smoke_replicates: int
    formal_replicates: int
    scenarios: tuple[G2ScenarioCell, ...]
    context_count: int
    family_count: int
    feature_count: int
    active_coefficient_floor: float
    lambda1: float
    lambda2: float
    lambda_f: float
    tolerance: float
    max_iterations: int
    gate_specification: G2GateSpecification

    def __post_init__(self) -> None:
        if len(self.scenarios) != 16:
            raise ValueError("G2 manifest must contain exactly 16 scenario cells")
        if self.context_count != 4:
            raise ValueError("G2 v1 freezes four contexts per topology")
        if self.family_count < 4 or self.feature_count <= self.family_count:
            raise ValueError(
                "G2 simulation needs multiple families and excess features"
            )
        if self.active_coefficient_floor < 0:
            raise ValueError("active_coefficient_floor must be non-negative")
        if any(value < 0 for value in (self.lambda1, self.lambda2, self.lambda_f)):
            raise ValueError("G2 solver penalties must be non-negative")
        if self.lambda_f <= 0:
            raise ValueError("G2 fused candidate requires positive lambda_f")
        if self.tolerance <= 0:
            raise ValueError("G2 solver tolerance must be positive")
        if self.gate_specification.scenario_ids != tuple(
            scenario.scenario_id for scenario in self.scenarios
        ):
            raise ValueError("G2 gate scenarios must match the frozen manifest")


@dataclass(frozen=True, slots=True, kw_only=True)
class G2SyntheticProblem:
    """One simulated problem reused by every paired method variant."""

    scenario: G2ScenarioCell
    replicate: int
    root_seed: int
    scenario_seed: int
    problem_id: str
    manifest_sha256: str
    graph: ContextGraph
    no_topology_graph: ContextGraph
    wrong_topology_graph: ContextGraph
    graph_id: str
    no_topology_graph_id: str
    wrong_topology_graph_id: str
    family_ids: tuple[str, ...]
    matrices: Mapping[Hashable, sparse.csc_matrix]
    responses: Mapping[Hashable, np.ndarray]
    precision_weights: Mapping[Hashable, np.ndarray]
    truth_coefficients: np.ndarray
    biological_edge_indices: tuple[tuple[int, int], ...]


def _factor_levels(
    value: object,
    *,
    field: str,
    expected_names: tuple[str, str],
) -> tuple[G2FactorLevel, G2FactorLevel]:
    raw_levels = _sequence(value, field=field)
    levels = tuple(
        G2FactorLevel(
            name=_required_string(
                _mapping(raw, field=f"{field}[]").get("name"),
                field=f"{field}[].name",
            ),
            value=_finite_float(
                _mapping(raw, field=f"{field}[]").get("value"),
                field=f"{field}[].value",
            ),
        )
        for raw in raw_levels
    )
    if tuple(level.name for level in levels) != expected_names:
        raise ValueError(f"{field} names must equal {expected_names}")
    if len(levels) != 2:
        raise ValueError(f"{field} must contain exactly two levels")
    return levels[0], levels[1]


def _scenario_id(
    topology: str,
    signal: str,
    snr_name: str,
    collinearity_name: str,
) -> str:
    return (
        f"topology={topology}__signal={signal}__snr={snr_name}__"
        f"family_collinearity={collinearity_name}"
    )


def load_g2_manifest(path: str | Path = DEFAULT_CONFIG) -> G2BenchmarkManifest:
    """Load and strictly validate the preregistered G2 scenario manifest."""

    source = Path(path).resolve()
    raw_bytes = source.read_bytes()
    value = json.loads(raw_bytes)
    config = _mapping(value, field="config")
    if config.get("schema_version") != G2_BENCHMARK_SCHEMA_VERSION:
        raise ValueError("unsupported G2 benchmark schema")
    if config.get("public_api") != (
        "crychic.attribution.solve_graph_fused_nonnegative_elastic_net"
    ):
        raise ValueError("G2 manifest must call the released graph-fused public API")
    scenario_manifest = _mapping(
        config.get("scenario_manifest"), field="scenario_manifest"
    )
    topologies = tuple(
        _required_string(item, field="scenario_manifest.topologies[]")
        for item in _sequence(
            scenario_manifest.get("topologies"),
            field="scenario_manifest.topologies",
        )
    )
    signals = tuple(
        _required_string(item, field="scenario_manifest.signals[]")
        for item in _sequence(
            scenario_manifest.get("signals"), field="scenario_manifest.signals"
        )
    )
    if topologies != FROZEN_TOPOLOGIES or signals != FROZEN_SIGNALS:
        raise ValueError("G2 topology and signal factors changed from the frozen order")
    snr_levels = _factor_levels(
        scenario_manifest.get("snr_levels"),
        field="scenario_manifest.snr_levels",
        expected_names=FROZEN_SNR_NAMES,
    )
    collinearity_levels = _factor_levels(
        scenario_manifest.get("family_collinearity_levels"),
        field="scenario_manifest.family_collinearity_levels",
        expected_names=FROZEN_COLLINEARITY_NAMES,
    )
    if any(level.value <= 0 for level in snr_levels):
        raise ValueError("G2 SNR levels must be positive")
    if any(not 0 <= level.value < 1 for level in collinearity_levels):
        raise ValueError("G2 family collinearity levels must lie in [0, 1)")
    scenarios = tuple(
        G2ScenarioCell(
            topology=topology,
            signal=signal,
            snr_name=snr.name,
            snr=snr.value,
            collinearity_name=collinearity.name,
            collinearity=collinearity.value,
            scenario_id=_scenario_id(
                topology, signal, snr.name, collinearity.name
            ),
        )
        for topology, signal, snr, collinearity in product(
            topologies, signals, snr_levels, collinearity_levels
        )
    )
    if _positive_int(
        scenario_manifest.get("scenario_cell_count"),
        field="scenario_manifest.scenario_cell_count",
    ) != len(scenarios):
        raise ValueError("scenario_cell_count does not match the frozen factorial")
    if scenario_manifest.get("aggregation") != "equal_weight_per_scenario_cell":
        raise ValueError("G2 primary aggregation must remain scenario-cell equal")

    profiles = _mapping(config.get("profiles"), field="profiles")
    smoke = _mapping(profiles.get("smoke"), field="profiles.smoke")
    formal = _mapping(profiles.get("formal"), field="profiles.formal")
    if smoke.get("publication_eligible") is not False:
        raise ValueError("G2 smoke profile cannot be publication eligible")
    simulation = _mapping(config.get("simulation"), field="simulation")
    expected_simulation_semantics = {
        "response_channel": "positive_rectified",
        "snr_definition": (
            "pre_rectification_signal_rms_over_gaussian_noise_sd"
        ),
        "family_collinearity_definition": "shared_gamma_mixture_weight",
        "wrong_topology_policy": "node_order_permutation_0_2_1_3_chain",
    }
    for field_name, expected in expected_simulation_semantics.items():
        if simulation.get(field_name) != expected:
            raise ValueError(
                f"simulation.{field_name} changed from the frozen semantics"
            )
    solver = _mapping(config.get("solver"), field="solver")
    primary = _mapping(config.get("primary_endpoint"), field="primary_endpoint")
    sensitivity = _mapping(
        config.get("topology_sensitivity"), field="topology_sensitivity"
    )
    if (
        primary.get("unit") != "driver_family_x_context"
        or primary.get("metric") != PRIMARY_METRIC
        or primary.get("pairing") != "same_scenario_seed_fused_vs_unfused"
        or primary.get("aggregation")
        != "paired_seed_within_scenario_then_equal_scenario_cell"
        or primary.get("interval_type")
        != "paired_seed_stratified_percentile_bootstrap"
    ):
        raise ValueError("G2 primary endpoint changed from the preregistration")
    controls = tuple(
        str(item)
        for item in _sequence(
            sensitivity.get("controls"), field="topology_sensitivity.controls"
        )
    )
    if controls != ("no_topology", "wrong_topology"):
        raise ValueError("G2 topology controls changed from the preregistration")
    gate = G2GateSpecification(
        scenario_ids=tuple(scenario.scenario_id for scenario in scenarios),
        minimum_paired_replicates_per_scenario=_positive_int(
            primary.get("minimum_paired_replicates_per_scenario"),
            field="primary_endpoint.minimum_paired_replicates_per_scenario",
        ),
        bootstrap_replicates=_positive_int(
            primary.get("bootstrap_replicates"),
            field="primary_endpoint.bootstrap_replicates",
        ),
        bootstrap_seed=_nonnegative_int(
            primary.get("bootstrap_seed"),
            field="primary_endpoint.bootstrap_seed",
        ),
        confidence_level=_finite_float(
            primary.get("confidence_level"),
            field="primary_endpoint.confidence_level",
        ),
        improvement_margin=_finite_float(
            primary.get("improvement_ci_lower_margin"),
            field="primary_endpoint.improvement_ci_lower_margin",
        ),
        noninferiority_margin=_finite_float(
            sensitivity.get("noninferiority_ci_lower_margin"),
            field="topology_sensitivity.noninferiority_ci_lower_margin",
        ),
    )
    return G2BenchmarkManifest(
        source_path=source,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        benchmark_id=_required_string(
            config.get("benchmark_id"), field="benchmark_id"
        ),
        root_seed=_nonnegative_int(config.get("root_seed"), field="root_seed"),
        default_profile=_required_string(
            profiles.get("default"), field="profiles.default"
        ),
        smoke_replicates=_positive_int(
            smoke.get("replicates_per_scenario"),
            field="profiles.smoke.replicates_per_scenario",
        ),
        formal_replicates=_positive_int(
            formal.get("replicates_per_scenario"),
            field="profiles.formal.replicates_per_scenario",
        ),
        scenarios=scenarios,
        context_count=_positive_int(
            simulation.get("context_count"), field="simulation.context_count"
        ),
        family_count=_positive_int(
            simulation.get("family_count"), field="simulation.family_count"
        ),
        feature_count=_positive_int(
            simulation.get("feature_count"), field="simulation.feature_count"
        ),
        active_coefficient_floor=_finite_float(
            simulation.get("active_coefficient_floor"),
            field="simulation.active_coefficient_floor",
        ),
        lambda1=_finite_float(solver.get("lambda1"), field="solver.lambda1"),
        lambda2=_finite_float(solver.get("lambda2"), field="solver.lambda2"),
        lambda_f=_finite_float(solver.get("lambda_f"), field="solver.lambda_f"),
        tolerance=_finite_float(
            solver.get("tolerance"), field="solver.tolerance"
        ),
        max_iterations=_positive_int(
            solver.get("max_iterations"), field="solver.max_iterations"
        ),
        gate_specification=gate,
    )


def _context_graph(topology: str, context_count: int) -> ContextGraph:
    if topology == "chain":
        return ContextGraph.chain(tuple(f"context_{index}" for index in range(4)))
    if topology == "product":
        if context_count != 4:
            raise ValueError("G2 product topology requires four contexts")
        return ContextGraph.product(
            {
                "region": ContextGraph.chain(("proximal", "distal")),
                "time": ContextGraph.chain(("early", "late")),
            }
        )
    raise ValueError(f"unsupported G2 topology: {topology}")


def _wrong_topology(graph: ContextGraph) -> ContextGraph:
    nodes = tuple(graph.nodes)
    if len(nodes) != 4:
        raise ValueError("G2 wrong-topology construction requires four nodes")
    order = (nodes[0], nodes[2], nodes[1], nodes[3])
    wrong = ContextGraph.chain(order)
    return ContextGraph(nodes=wrong.nodes, edges=wrong.edges, kind="wrong_topology")


def _distances(graph: ContextGraph) -> dict[Hashable, int]:
    root = graph.nodes[0]
    distance: dict[Hashable, int] = {root: 0}
    frontier = [root]
    while frontier:
        node = frontier.pop(0)
        for neighbor, _ in graph.neighbors(node):
            if neighbor in distance:
                continue
            distance[neighbor] = distance[node] + 1
            frontier.append(neighbor)
    if set(distance) != set(graph.nodes):
        raise ValueError("G2 biological topology must be connected")
    return distance


def _truth_coefficients(
    graph: ContextGraph,
    signal: str,
    family_count: int,
) -> np.ndarray:
    distances = _distances(graph)
    maximum = max(distances.values())
    coefficients: np.ndarray = np.zeros(
        (len(graph.nodes), family_count), dtype=np.float64
    )
    for index, node in enumerate(graph.nodes):
        position = float(distances[node]) / float(maximum)
        coefficients[index, 0] = 0.9
        if signal == "smooth_gradient":
            coefficients[index, 1] = 1.1 * (1.0 - position)
            coefficients[index, 2] = 1.1 * position
        elif signal == "single_local_jump":
            if distances[node] == maximum:
                coefficients[index, 3] = 1.2
            else:
                coefficients[index, 1] = 0.8
        else:
            raise ValueError(f"unsupported G2 signal: {signal}")
    return coefficients


def _scenario_seed(
    manifest: G2BenchmarkManifest,
    scenario: G2ScenarioCell,
    replicate: int,
) -> int:
    payload = (
        f"{manifest.benchmark_id}:{manifest.root_seed}:{scenario.scenario_id}:"
        f"{replicate}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _matrix_digest(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8", order="C")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _graph_id(graph: ContextGraph) -> str:
    return str(stable_id("g2_context_graph", graph.to_dict(), schema_version="1"))


def _biological_edge_indices(
    graph: ContextGraph,
) -> tuple[tuple[int, int], ...]:
    index = {node: position for position, node in enumerate(graph.nodes)}
    return tuple((index[edge.left], index[edge.right]) for edge in graph.edges)


def simulate_g2_problem(
    manifest: G2BenchmarkManifest,
    scenario: G2ScenarioCell,
    replicate: int,
) -> G2SyntheticProblem:
    """Generate one deterministic problem shared by all paired method variants."""

    if not isinstance(manifest, G2BenchmarkManifest):
        raise TypeError("manifest must be a G2BenchmarkManifest")
    if scenario not in manifest.scenarios:
        raise ValueError("scenario is outside the frozen G2 manifest")
    if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
        raise ValueError("replicate must be a non-negative integer")
    scenario_seed = _scenario_seed(manifest, scenario, replicate)
    rng = np.random.default_rng(scenario_seed)
    graph = _context_graph(scenario.topology, manifest.context_count)
    no_topology = ContextGraph(nodes=graph.nodes, edges=(), kind="no_topology")
    wrong_topology = _wrong_topology(graph)
    shared = rng.gamma(shape=2.0, scale=1.0, size=(manifest.feature_count, 1))
    independent = rng.gamma(
        shape=2.0,
        scale=1.0,
        size=(manifest.feature_count, manifest.family_count),
    )
    basis = (
        math.sqrt(scenario.collinearity) * shared
        + math.sqrt(1.0 - scenario.collinearity) * independent
    )
    norms = np.linalg.norm(basis, axis=0)
    basis = basis / norms
    truth = _truth_coefficients(graph, scenario.signal, manifest.family_count)
    matrices: dict[Hashable, sparse.csc_matrix] = {}
    responses: dict[Hashable, np.ndarray] = {}
    precision: dict[Hashable, np.ndarray] = {}
    response_digests: list[str] = []
    for context_index, node in enumerate(graph.nodes):
        signal = basis @ truth[context_index]
        signal_rms = float(np.sqrt(np.mean(np.square(signal))))
        noise_scale = signal_rms / scenario.snr
        response = np.maximum(
            signal + rng.normal(0.0, noise_scale, size=manifest.feature_count),
            0.0,
        )
        matrices[node] = sparse.csc_matrix(basis)
        responses[node] = response.astype(np.float64)
        precision[node] = np.ones(manifest.feature_count, dtype=np.float64)
        response_digests.append(_matrix_digest(response))
    graph_id = _graph_id(graph)
    no_topology_id = _graph_id(no_topology)
    wrong_topology_id = _graph_id(wrong_topology)
    problem_id = stable_id(
        "g2_synthetic_problem",
        {
            "basis_digest": _matrix_digest(basis),
            "graph_id": graph_id,
            "manifest_sha256": manifest.source_sha256,
            "replicate": replicate,
            "response_digests": response_digests,
            "scenario_id": scenario.scenario_id,
            "scenario_seed": scenario_seed,
            "truth_digest": _matrix_digest(truth),
        },
        schema_version="1",
    )
    return G2SyntheticProblem(
        scenario=scenario,
        replicate=replicate,
        root_seed=manifest.root_seed,
        scenario_seed=scenario_seed,
        problem_id=problem_id,
        manifest_sha256=manifest.source_sha256,
        graph=graph,
        no_topology_graph=no_topology,
        wrong_topology_graph=wrong_topology,
        graph_id=graph_id,
        no_topology_graph_id=no_topology_id,
        wrong_topology_graph_id=wrong_topology_id,
        family_ids=tuple(
            f"driver_family_{index}" for index in range(manifest.family_count)
        ),
        matrices=matrices,
        responses=responses,
        precision_weights=precision,
        truth_coefficients=truth,
        biological_edge_indices=_biological_edge_indices(graph),
    )


def _method_graphs(
    problem: G2SyntheticProblem,
    manifest: G2BenchmarkManifest,
) -> tuple[tuple[str, ContextGraph, str, float], ...]:
    return (
        (UNFUSED_METHOD, problem.graph, problem.graph_id, 0.0),
        (FUSED_METHOD, problem.graph, problem.graph_id, manifest.lambda_f),
        (
            NO_TOPOLOGY_METHOD,
            problem.no_topology_graph,
            problem.no_topology_graph_id,
            manifest.lambda_f,
        ),
        (
            WRONG_TOPOLOGY_METHOD,
            problem.wrong_topology_graph,
            problem.wrong_topology_graph_id,
            manifest.lambda_f,
        ),
    )


def run_g2_problem(
    manifest: G2BenchmarkManifest,
    problem: G2SyntheticProblem,
) -> pd.DataFrame:
    """Run all paired variants through the released graph-fused public API."""

    truth_active = problem.truth_coefficients > manifest.active_coefficient_floor
    records: list[dict[str, object]] = []
    for method, graph, graph_id, lambda_f in _method_graphs(problem, manifest):
        try:
            solution = solve_graph_fused_nonnegative_elastic_net(
                problem.matrices,
                problem.responses,
                graph,
                precision_weights=problem.precision_weights,
                lambda1=manifest.lambda1,
                lambda2=manifest.lambda2,
                lambda_f=lambda_f,
                tolerance=manifest.tolerance,
                max_iterations=manifest.max_iterations,
            )
            converged = solution.diagnostics.converged
            coefficients = solution.coefficients
            macro_auprc: float | None = None
            macro_auroc: float | None = None
            coefficient_rmse: float | None = None
            jump_auprc: float | None = None
            if converged:
                macro_auprc = driver_family_context_macro_auprc(
                    truth_active, coefficients
                )
                macro_auroc = driver_family_context_macro_auroc(
                    truth_active, coefficients
                )
                coefficient_rmse = float(
                    np.sqrt(
                        np.mean(
                            np.square(coefficients - problem.truth_coefficients)
                        )
                    )
                )
                jump_auprc = graph_jump_localization_auprc(
                    problem.truth_coefficients,
                    coefficients,
                    problem.biological_edge_indices,
                )
            status = "observed" if converged else "failed"
            reason_code = solution.diagnostics.failure_reason
            backend = solution.diagnostics.backend
            iterations = solution.diagnostics.iterations
            optimality = solution.diagnostics.optimality
            constraint_violation = solution.diagnostics.constraint_violation
        except Exception as error:  # benchmark records failures; it never passes them
            macro_auprc = None
            macro_auroc = None
            coefficient_rmse = None
            jump_auprc = None
            status = "failed"
            reason_code = f"{type(error).__name__}:{error}"
            backend = None
            iterations = None
            optimality = None
            constraint_violation = None
        records.append(
            {
                "schema_version": G2_RESULT_SCHEMA_VERSION,
                "manifest_sha256": manifest.source_sha256,
                "benchmark_id": manifest.benchmark_id,
                "problem_id": problem.problem_id,
                "scenario_id": problem.scenario.scenario_id,
                "topology": problem.scenario.topology,
                "signal": problem.scenario.signal,
                "snr_name": problem.scenario.snr_name,
                "snr": problem.scenario.snr,
                "family_collinearity_name": (
                    problem.scenario.collinearity_name
                ),
                "family_collinearity": problem.scenario.collinearity,
                "replicate": problem.replicate,
                "root_seed": problem.root_seed,
                "scenario_seed": problem.scenario_seed,
                "method": method,
                "graph_id": graph_id,
                "biological_graph_id": problem.graph_id,
                "lambda1": manifest.lambda1,
                "lambda2": manifest.lambda2,
                "lambda_f": lambda_f,
                "primary_metric": PRIMARY_METRIC,
                "macro_auprc": macro_auprc,
                "macro_auroc": macro_auroc,
                "coefficient_rmse": coefficient_rmse,
                "jump_localization_auprc": jump_auprc,
                "solver_backend": backend,
                "solver_iterations": iterations,
                "solver_optimality": optimality,
                "solver_constraint_violation": constraint_violation,
                "status": status,
                "reason_code": reason_code,
            }
        )
    return pd.DataFrame.from_records(records)


def run_g2_simulation(
    manifest: G2BenchmarkManifest,
    *,
    replicates_per_scenario: int | None = None,
) -> pd.DataFrame:
    """Run the full frozen factorial with configurable paired replicates."""

    replicates = (
        manifest.smoke_replicates
        if replicates_per_scenario is None
        else replicates_per_scenario
    )
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 1
    ):
        raise ValueError("replicates_per_scenario must be a positive integer")
    parts = [
        run_g2_problem(
            manifest,
            simulate_g2_problem(manifest, scenario, replicate),
        )
        for scenario in manifest.scenarios
        for replicate in range(replicates)
    ]
    result = pd.concat(parts, ignore_index=True)
    expected_rows = len(manifest.scenarios) * replicates * len(METHODS)
    if len(result) != expected_rows or result.duplicated(
        ["scenario_id", "replicate", "method"]
    ).any():
        raise RuntimeError("G2 runner did not produce the exact paired result grain")
    return result


def manifest_provenance(manifest: G2BenchmarkManifest) -> dict[str, Any]:
    """Return compact scenario/topology/seed provenance for reports."""

    return {
        "schema_version": G2_BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": manifest.benchmark_id,
        "config_path": manifest.source_path.name,
        "config_sha256": manifest.source_sha256,
        "root_seed": manifest.root_seed,
        "scenario_count": len(manifest.scenarios),
        "scenario_ids": [scenario.scenario_id for scenario in manifest.scenarios],
        "topologies": list(FROZEN_TOPOLOGIES),
        "signals": list(FROZEN_SIGNALS),
        "public_api": (
            "crychic.attribution.solve_graph_fused_nonnegative_elastic_net"
        ),
        "solver_policy_digest": canonical_digest(
            {
                "lambda1": manifest.lambda1,
                "lambda2": manifest.lambda2,
                "lambda_f": manifest.lambda_f,
                "max_iterations": manifest.max_iterations,
                "tolerance": manifest.tolerance,
            }
        ),
    }


__all__ = [
    "DEFAULT_CONFIG",
    "FROZEN_COLLINEARITY_NAMES",
    "FROZEN_SIGNALS",
    "FROZEN_SNR_NAMES",
    "FROZEN_TOPOLOGIES",
    "G2_BENCHMARK_SCHEMA_VERSION",
    "G2_RESULT_SCHEMA_VERSION",
    "G2BenchmarkManifest",
    "G2ScenarioCell",
    "G2SyntheticProblem",
    "load_g2_manifest",
    "manifest_provenance",
    "run_g2_problem",
    "run_g2_simulation",
    "simulate_g2_problem",
]
