"""Raw-count DGP families for the integrated suggest-next2 v7 benchmark."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from benchmarks.simulation.v7_protocol import DESIGN_KINDS, REQUIRED_DGP_FAMILIES
from crychic.attribution import PenaltyTuningSpec
from crychic.core import CrychicConfig, canonical_digest, stable_id
from crychic.design import ContrastSpec, balanced_contrast
from crychic.inference import DifferentialContrastSpec, DifferentialDesignSpec
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
    TargetPrior,
)
from crychic.scoring import AbsoluteActivityV2Spec, SignedProgramV2Spec
from crychic.sender import (
    ContrastCommonSenderParameters,
    EBShrunkenCouplingV2Spec,
    SenderAttributionV2Spec,
)
from crychic.workflow import CrossFitSpec, FoldTrainingSpec

SCHEMA_VERSION = "crychic-suggest-next2-v7-raw-dgp-v1"
CONDITIONS_BY_DESIGN = {
    "independent_two_group": ("A", "B"),
    "independent_multi_group": ("A", "B", "C"),
    "paired": ("A", "B"),
    "repeated": ("A", "B", "C"),
    "multi_cohort": ("A", "B"),
    "continuous": ("A", "B"),
}
CANDIDATE_SENDER_COUNTS = (2, 5, 10, 20)

GENES = (
    "HK1",
    "HK2",
    "L1",
    "R1",
    "L2",
    "R2",
    "L3A",
    "L3B",
    "R3A",
    "R3B",
    "L4A",
    "L4B",
    "R4",
    "L5",
    "R5",
    "L6",
    "R6",
    "L7",
    "R7",
    "T_ACT",
    "T_ALT",
    "T_INH",
    "T_WEAK",
    "G_STATE",
    "BATCH_G",
)
_GENE_INDEX = {gene: index for index, gene in enumerate(GENES)}


@dataclass(frozen=True, slots=True)
class _InteractionDefinition:
    interaction_id: str
    ligand_name: str
    receptor_name: str
    ligand_subunits: tuple[str, ...]
    receptor_subunits: tuple[str, ...]
    pathway: str
    annotation: str
    target: str
    target_weight: float


INTERACTIONS = (
    _InteractionDefinition(
        "lr_signal",
        "L1",
        "R1",
        ("L1",),
        ("R1",),
        "signal_pathway",
        "secreted",
        "T_ACT",
        1.0,
    ),
    _InteractionDefinition(
        "lr_decoy",
        "L2",
        "R2",
        ("L2",),
        ("R2",),
        "decoy_pathway",
        "secreted",
        "T_WEAK",
        0.5,
    ),
    _InteractionDefinition(
        "lr_complex",
        "L3A_L3B",
        "R3A_R3B",
        ("L3A", "L3B"),
        ("R3A", "R3B"),
        "complex_pathway",
        "secreted",
        "T_ACT",
        0.8,
    ),
    _InteractionDefinition(
        "lr_alternative",
        "L4A_or_L4B",
        "R4",
        ("L4A", "L4B"),
        ("R4",),
        "alternative_pathway",
        "secreted",
        "T_ALT",
        1.0,
    ),
    _InteractionDefinition(
        "lr_inhibitory",
        "L5",
        "R5",
        ("L5",),
        ("R5",),
        "inhibitory_pathway",
        "secreted",
        "T_INH",
        -1.0,
    ),
    _InteractionDefinition(
        "lr_contact",
        "L6",
        "R6",
        ("L6",),
        ("R6",),
        "contact_pathway",
        "contact",
        "T_WEAK",
        0.6,
    ),
    _InteractionDefinition(
        "lr_ecm",
        "L7",
        "R7",
        ("L7",),
        ("R7",),
        "ecm_pathway",
        "ECM",
        "T_WEAK",
        0.6,
    ),
)
_INTERACTION_BY_ID = {item.interaction_id: item for item in INTERACTIONS}

TRUTH_COLUMNS = (
    "schema_version",
    "dataset_id",
    "dgp_family",
    "design_kind",
    "contrast_name",
    "sender",
    "receiver",
    "interaction_id",
    "ligand",
    "receptor",
    "pathway",
    "mechanism_class",
    "receptor_complex_cardinality",
    "truth_score_effect",
    "truth_parent_effect",
    "truth_program_effect",
    "truth_occurrence_effect",
    "truth_causal_sender",
    "truth_causal_parent",
    "truth_direction",
)

_NULL_FAMILIES = {
    "global_null",
    "scoring_functional_null",
    "fixed_subject_increasing_cell_null",
    "legal_condition_permutation_null",
    "generic_state_null",
    "batch_context_confounding_null",
    "composition_only_null",
    "receiver_autonomous_null",
    "abundance_only_null",
}
_WEAK_TOPOLOGY_FAMILIES = {
    "graph_smooth",
    "hypergraph_weak_effects",
    "prior_corruption",
    "wrong_topology",
    "disconnected_graph",
    "annotation_perturbation",
    "prior_replacement",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class V7DGPFixture:
    """In-memory raw input, frozen resources, designs, and truth for one DGP."""

    dataset_id: str
    dgp_family: str
    design_kind: str
    seed: int
    candidate_sender_count: int
    adata: AnnData = field(repr=False)
    raw_input_digest: str
    config: CrychicConfig
    resource: ResourceBundle
    target_prior: TargetPrior
    crossfit_spec: CrossFitSpec
    differential_design: DifferentialDesignSpec
    sample_metadata: pd.DataFrame
    truth: pd.DataFrame
    cell_counts: pd.DataFrame
    fixture_id: str

    def __post_init__(self) -> None:
        if self.dgp_family not in REQUIRED_DGP_FAMILIES:
            raise ValueError("fixture dgp_family is not registered")
        if self.design_kind not in DESIGN_KINDS:
            raise ValueError("fixture design_kind is not registered")
        if self.candidate_sender_count not in CANDIDATE_SENDER_COUNTS:
            raise ValueError("fixture candidate sender count is unsupported")
        if len(self.raw_input_digest) != 64:
            raise ValueError("fixture raw_input_digest must be a SHA-256 digest")
        if tuple(self.truth.columns) != TRUTH_COLUMNS or self.truth.empty:
            raise ValueError("fixture truth contract is invalid")
        if self.sample_metadata["sample_id"].duplicated().any():
            raise ValueError("fixture sample metadata IDs must be unique")
        observed_samples = set(self.adata.obs["sample_id"].astype(str))
        if observed_samples != set(self.sample_metadata["sample_id"].astype(str)):
            raise ValueError("fixture AnnData and sample metadata axes differ")
        if self.cell_counts["cell_count"].lt(0).any():
            raise ValueError("fixture cell counts cannot be negative")

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": self.fixture_id,
            "dataset_id": self.dataset_id,
            "dgp_family": self.dgp_family,
            "design_kind": self.design_kind,
            "seed": self.seed,
            "candidate_sender_count": self.candidate_sender_count,
            "shape": [int(self.adata.n_obs), int(self.adata.n_vars)],
            "raw_input_digest": self.raw_input_digest,
            "subjects": int(self.adata.obs["subject_id"].astype(str).nunique()),
            "samples": int(self.adata.obs["sample_id"].astype(str).nunique()),
            "resource_id": self.resource.resource_id,
            "target_prior_id": self.target_prior.resource_id,
            "crossfit_spec_id": self.crossfit_spec.spec_id,
            "differential_design_id": self.differential_design.spec_id,
            "truth_rows": len(self.truth),
        }


def _resource_bundle(dgp_family: str) -> ResourceBundle:
    interactions: list[Interaction] = []
    for definition in INTERACTIONS:
        annotation = definition.annotation
        if dgp_family == "annotation_perturbation":
            annotation = {
                "contact": "secreted",
                "secreted": "contact",
                "ECM": "secreted",
            }.get(annotation, annotation)
        interactions.append(
            Interaction(
                interaction_id=definition.interaction_id,
                source_interaction_id=definition.interaction_id,
                ligand_name=definition.ligand_name,
                receptor_name=definition.receptor_name,
                ligand_subunits=definition.ligand_subunits,
                receptor_subunits=definition.receptor_subunits,
                ligand_is_complex=(
                    len(definition.ligand_subunits) > 1
                    and definition.interaction_id != "lr_alternative"
                ),
                receptor_is_complex=len(definition.receptor_subunits) > 1,
                direction="Ligand-Receptor",
                source="crychic-v7-synthetic",
                version="1",
                species=Species.HUMAN,
                gene_namespace=GeneNamespace.HGNC_SYMBOL,
                pathway=definition.pathway,
                annotation=annotation,
            )
        )
    manifest_digest = str(
        canonical_digest(
            [
                {
                    "annotation": item.annotation,
                    "interaction_id": item.interaction_id,
                    "ligand_is_complex": item.ligand_is_complex,
                    "ligand_subunits": list(item.ligand_subunits),
                    "pathway": item.pathway,
                    "receptor_is_complex": item.receptor_is_complex,
                    "receptor_subunits": list(item.receptor_subunits),
                }
                for item in interactions
            ]
        )
    )
    return ResourceBundle(
        resource_id="crychic-v7-dgp-resource",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=tuple(interactions),
        mapping_report=MappingReport(
            source_rows=len(interactions),
            loaded_rows=len(interactions),
            mapped_entities=len(
                {
                    gene
                    for item in INTERACTIONS
                    for gene in (*item.ligand_subunits, *item.receptor_subunits)
                }
            ),
        ),
        manifest_digest=manifest_digest,
        source_files=("synthetic:v7_dgp",),
        license="CC0-1.0",
        citation="CRYCHIC v7 synthetic benchmark fixture",
    )


def _target_prior(dgp_family: str) -> TargetPrior:
    target_ids = ("T_ACT", "T_ALT", "T_INH", "T_WEAK")
    target_index = {target: index for index, target in enumerate(target_ids)}
    definitions = sorted(INTERACTIONS, key=lambda item: item.interaction_id)
    if dgp_family == "prior_replacement":
        definitions = sorted(
            [
                _InteractionDefinition(
                    item.interaction_id,
                    item.ligand_name,
                    item.receptor_name,
                    item.ligand_subunits,
                    item.receptor_subunits,
                    item.pathway,
                    item.annotation,
                    "T_WEAK" if item.target != "T_WEAK" else "T_ACT",
                    item.target_weight,
                )
                for item in definitions
            ],
            key=lambda item: item.interaction_id,
        )
    indptr = [0]
    indices: list[int] = []
    weights: list[float] = []
    for item in definitions:
        indices.append(target_index[item.target])
        weights.append(abs(float(item.target_weight)))
        indptr.append(len(indices))
    direction = 1
    manifest_digest = str(
        canonical_digest(
            {
                "direction": direction,
                "driver_ids": [item.interaction_id for item in definitions],
                "target_ids": list(target_ids),
                "target_indices": indices,
                "weights": weights,
            }
        )
    )
    return TargetPrior(
        resource_id="crychic-v7-dgp-target-prior",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="interaction",
        target_ids=target_ids,
        driver_ids=tuple(item.interaction_id for item in definitions),
        indptr=tuple(indptr),
        target_indices=tuple(indices),
        weights=tuple(weights),
        ranks=None,
        direction=direction,
        evidence="synthetic planted mechanism; never used as evaluation truth",
        mapping_report=MappingReport(
            source_rows=len(definitions),
            loaded_rows=len(definitions),
            mapped_entities=len(indices),
        ),
        manifest_digest=manifest_digest,
    )


def _sample_layout(
    design_kind: str,
    *,
    subjects_per_level: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    conditions = CONDITIONS_BY_DESIGN[design_kind]
    records: list[dict[str, object]] = []
    if design_kind in {"paired", "repeated"}:
        for subject_index in range(subjects_per_level):
            subject = f"P{subject_index + 1:02d}"
            responder = bool(rng.random() < 0.5)
            for condition_index, condition in enumerate(conditions):
                records.append(
                    {
                        "sample_id": f"{subject}:{condition}",
                        "subject_id": subject,
                        "condition": condition,
                        "batch": f"batch{1 + ((subject_index + condition_index) % 2)}",
                        "cohort": "cohort1",
                        "dose": float(condition_index),
                        "responder": responder,
                    }
                )
    elif design_kind == "continuous":
        total = max(2 * subjects_per_level, 12)
        doses = np.linspace(-1.0, 1.0, total)
        for subject_index, dose in enumerate(doses):
            subject = f"D{subject_index + 1:02d}"
            condition = "A" if dose < 0.0 else "B"
            records.append(
                {
                    "sample_id": subject,
                    "subject_id": subject,
                    "condition": condition,
                    "batch": f"batch{1 + (subject_index % 2)}",
                    "cohort": "cohort1",
                    "dose": float(dose),
                    "responder": bool(rng.random() < 0.5),
                }
            )
    else:
        for condition_index, condition in enumerate(conditions):
            for subject_index in range(subjects_per_level):
                subject = f"{condition}:S{subject_index + 1:02d}"
                cohort = (
                    f"cohort{1 + (subject_index % 2)}"
                    if design_kind == "multi_cohort"
                    else "cohort1"
                )
                records.append(
                    {
                        "sample_id": subject,
                        "subject_id": subject,
                        "condition": condition,
                        "batch": f"batch{1 + ((subject_index + condition_index) % 2)}",
                        "cohort": cohort,
                        "dose": float(condition_index),
                        "responder": bool(rng.random() < 0.5),
                    }
                )
    result = pd.DataFrame.from_records(records)
    if result["sample_id"].duplicated().any():
        raise RuntimeError("DGP sample layout generated duplicate samples")
    return result


def _condition_scale(condition: str, dose: float, design_kind: str) -> float:
    if design_kind == "continuous":
        return float(dose)
    return {"A": 0.0, "B": 1.0, "C": 2.0}[condition]


def _active_event_amplitudes(
    dgp_family: str,
    *,
    condition: str,
    scale: float,
) -> dict[str, float]:
    if dgp_family in _NULL_FAMILIES or math.isclose(scale, 0.0):
        return {}
    if dgp_family == "topology_jump":
        if condition == "B":
            return {"lr_signal": 1.0}
        if condition == "C":
            return {"lr_decoy": 1.0}
        return {}
    if dgp_family == "complex_and":
        return {"lr_complex": 1.4 * scale}
    if dgp_family == "alternative_or":
        return {"lr_alternative": 1.4 * scale}
    if dgp_family == "inhibitory_program":
        return {"lr_inhibitory": 1.2 * scale}
    if dgp_family == "spatial_range":
        return {
            "lr_signal": 1.0 * scale,
            "lr_contact": 0.9 * scale,
            "lr_ecm": 0.7 * scale,
        }
    if dgp_family in _WEAK_TOPOLOGY_FAMILIES:
        return {
            "lr_signal": 0.45 * scale,
            "lr_complex": 0.35 * scale,
            "lr_alternative": 0.30 * scale,
            "lr_contact": 0.25 * scale,
        }
    return {"lr_signal": 1.0 * scale}


def _family_mode(dgp_family: str) -> str:
    if dgp_family == "ligand_only":
        return "ligand"
    if dgp_family == "receptor_only":
        return "receptor"
    if dgp_family == "program_only":
        return "program"
    return "joint"


def _true_senders(dgp_family: str, sender_ids: tuple[str, ...]) -> tuple[str, ...]:
    if dgp_family in {"receptor_only", "program_only"} or not sender_ids:
        return ()
    if dgp_family == "multi_sender" and len(sender_ids) >= 2:
        return sender_ids[:2]
    return sender_ids[:1]


def _base_profile(cell_type: str, *, dgp_family: str) -> np.ndarray:
    values: np.ndarray = np.full(len(GENES), 0.04, dtype=float)
    values[_GENE_INDEX["HK1"]] = 4.0
    values[_GENE_INDEX["HK2"]] = 3.0
    if cell_type == "Receiver":
        for gene in ("R1", "R2", "R3A", "R3B", "R4", "R5", "R6", "R7"):
            values[_GENE_INDEX[gene]] = 1.4
        for gene in ("T_ACT", "T_ALT", "T_INH", "T_WEAK", "G_STATE"):
            values[_GENE_INDEX[gene]] = 0.7
    else:
        for gene in ("L1", "L2", "L3A", "L3B", "L4A", "L4B", "L5", "L6", "L7"):
            values[_GENE_INDEX[gene]] = 1.0
        if (
            dgp_family in {"sender_decoy", "candidate_cardinality"}
            and cell_type != "S01"
        ):
            values[_GENE_INDEX["L1"]] = 2.5
    return values


def _gene_log2_effects(
    dgp_family: str,
    *,
    cell_type: str,
    condition: str,
    scale: float,
    responder: bool,
    batch: str,
    sender_ids: tuple[str, ...],
) -> dict[str, float]:
    effects: dict[str, float] = {}
    if dgp_family in {"generic_state_null", "receiver_autonomous_null"}:
        if cell_type == "Receiver":
            effects["G_STATE"] = 1.8 * scale
        return effects
    if dgp_family == "batch_context_confounding_null":
        if batch == "batch2":
            if cell_type == "S01":
                effects["L2"] = 1.5
            if cell_type == "Receiver":
                effects["R2"] = 1.5
                effects["BATCH_G"] = 2.0
        return effects
    amplitudes = _active_event_amplitudes(dgp_family, condition=condition, scale=scale)
    if dgp_family == "occurrence_heterogeneity" and not responder:
        amplitudes = {}
    mode = _family_mode(dgp_family)
    true_senders = _true_senders(dgp_family, sender_ids)
    for interaction_id, amplitude in amplitudes.items():
        definition = _INTERACTION_BY_ID[interaction_id]
        if mode in {"joint", "ligand"} and cell_type in true_senders:
            ligand_coefficient = 1.5 * amplitude
            ligand_subunits = (
                definition.ligand_subunits[:1]
                if interaction_id == "lr_alternative"
                else definition.ligand_subunits
            )
            for gene in ligand_subunits:
                effects[gene] = effects.get(gene, 0.0) + ligand_coefficient
        if mode in {"joint", "receptor"} and cell_type == "Receiver":
            receptor_coefficient = 1.0 * amplitude
            for gene in definition.receptor_subunits:
                effects[gene] = effects.get(gene, 0.0) + receptor_coefficient
        if mode in {"joint", "program"} and cell_type == "Receiver":
            program_coefficient = 1.2 * amplitude
            if dgp_family == "inhibitory_program":
                program_coefficient = -abs(program_coefficient)
            effects[definition.target] = (
                effects.get(definition.target, 0.0) + program_coefficient
            )
    if dgp_family == "collinear_lr":
        if cell_type in true_senders:
            effects["L2"] = effects.get("L1", 0.0)
        if cell_type == "Receiver":
            effects["R2"] = effects.get("R1", 0.0)
    return effects


def _cell_count(
    dgp_family: str,
    *,
    cell_type: str,
    condition: str,
    subject_index: int,
    cells_per_type: int,
) -> int:
    count = cells_per_type + (subject_index % 2)
    if (
        dgp_family in {"composition_only_null", "abundance_only_null"}
        and condition != "A"
    ):
        if cell_type == "S01":
            count *= 3
        elif cell_type == "Receiver":
            count = max(1, count // 2)
    if dgp_family == "fixed_subject_increasing_cell_null" and condition != "A":
        count *= 4
    if dgp_family == "structural_absence" and condition != "A" and cell_type == "S01":
        return 0
    if (
        dgp_family == "context_correlated_sampling_missingness"
        and condition != "A"
        and cell_type == "Receiver"
        and subject_index % 3 == 0
    ):
        return 0
    return count


def _simulate_counts(
    metadata: pd.DataFrame,
    *,
    dgp_family: str,
    design_kind: str,
    candidate_sender_count: int,
    cells_per_type: int,
    rng: np.random.Generator,
) -> tuple[AnnData, pd.DataFrame, pd.DataFrame]:
    sender_ids = tuple(f"S{index:02d}" for index in range(1, candidate_sender_count))
    cell_types = (*sender_ids, "Receiver")
    subject_ids = tuple(sorted(metadata["subject_id"].astype(str).unique()))
    subject_index = {subject: index for index, subject in enumerate(subject_ids)}
    subject_factors = {
        (subject, cell_type): rng.normal(0.0, 0.12, size=len(GENES))
        for subject in subject_ids
        for cell_type in cell_types
    }
    count_parts: list[np.ndarray] = []
    obs_parts: list[pd.DataFrame] = []
    cell_count_records: list[dict[str, object]] = []
    sample_totals: dict[str, int] = {}
    receiver_counts: dict[str, int] = {}

    for row in metadata.itertuples(index=False):
        sample_id = str(row.sample_id)
        subject_id = str(row.subject_id)
        condition = str(row.condition)
        scale = _condition_scale(condition, float(cast(Any, row.dose)), design_kind)
        total = 0
        for cell_type in cell_types:
            n_cells = _cell_count(
                dgp_family,
                cell_type=cell_type,
                condition=condition,
                subject_index=subject_index[subject_id],
                cells_per_type=cells_per_type,
            )
            cell_count_records.append(
                {
                    "sample_id": sample_id,
                    "cell_type": cell_type,
                    "cell_count": n_cells,
                }
            )
            total += n_cells
            if cell_type == "Receiver":
                receiver_counts[sample_id] = n_cells
            if n_cells == 0:
                continue
            profile = _base_profile(cell_type, dgp_family=dgp_family)
            effects = _gene_log2_effects(
                dgp_family,
                cell_type=cell_type,
                condition=condition,
                scale=scale,
                responder=bool(row.responder),
                batch=str(row.batch),
                sender_ids=sender_ids,
            )
            log2_effect: np.ndarray = np.zeros(len(GENES), dtype=float)
            for gene, value in effects.items():
                log2_effect[_GENE_INDEX[gene]] += float(value)
            shared = subject_factors[(subject_id, cell_type)]
            mean = profile * np.exp2(log2_effect + shared) * 6.0
            cell_depth = rng.lognormal(0.0, 0.20, size=n_cells)
            expected = cell_depth[:, np.newaxis] * mean[np.newaxis, :]
            dispersion = 0.18
            latent = rng.gamma(
                shape=1.0 / dispersion,
                scale=np.maximum(expected * dispersion, 1.0e-8),
            )
            counts = np.asarray(rng.poisson(latent), dtype=np.int32)
            count_parts.append(counts)
            obs_parts.append(
                pd.DataFrame(
                    {
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "condition": condition,
                        "cell_type": cell_type,
                        "batch": str(row.batch),
                        "cohort": str(row.cohort),
                        "dose": float(cast(Any, row.dose)),
                    },
                    index=[
                        f"{sample_id}:{cell_type}:{index:04d}"
                        for index in range(n_cells)
                    ],
                )
            )
        sample_totals[sample_id] = total
    if not count_parts:
        raise RuntimeError("DGP generated no cells")
    counts = sparse.csr_matrix(np.vstack(count_parts), dtype=np.int32)
    observations = pd.concat(obs_parts, axis=0)
    result = AnnData(
        X=sparse.csr_matrix(counts.shape, dtype=float),
        obs=observations,
        var=pd.DataFrame(index=pd.Index(GENES, name="gene_symbol")),
    )
    result.layers["counts"] = counts
    result.uns["dgp_contract"] = {
        "schema_version": SCHEMA_VERSION,
        "dgp_family": dgp_family,
        "design_kind": design_kind,
        "candidate_sender_count": candidate_sender_count,
        "x_semantics": "empty_placeholder_use_counts_layer",
    }
    enriched = metadata.copy()
    enriched["total_cells"] = enriched["sample_id"].map(sample_totals).astype(int)
    enriched["receiver_cells"] = (
        enriched["sample_id"].map(receiver_counts).fillna(0).astype(int)
    )
    enriched["receiver_fraction"] = np.where(
        enriched["total_cells"].gt(0),
        enriched["receiver_cells"] / enriched["total_cells"],
        np.nan,
    )
    return result, enriched, pd.DataFrame.from_records(cell_count_records)


def _raw_input_digest(adata: AnnData) -> str:
    counts = sparse.csr_matrix(adata.layers["counts"])
    digest = hashlib.sha256()
    digest.update(np.asarray(counts.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(counts.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(counts.indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(counts.data, dtype=np.int64).tobytes())
    metadata = adata.obs.reset_index(names="cell_id").to_dict(orient="records")
    digest.update(str(canonical_digest(metadata)).encode("ascii"))
    digest.update(
        str(canonical_digest(list(map(str, adata.var_names)))).encode("ascii")
    )
    return digest.hexdigest()


def _crossfit_contrasts(conditions: tuple[str, ...]) -> tuple[ContrastSpec, ...]:
    records: list[ContrastSpec] = []
    for target_index in range(1, len(conditions)):
        target = conditions[target_index]
        for reference in conditions[:target_index]:
            records.append(
                balanced_contrast(
                    (target,),
                    (reference,),
                    name=f"{target}_vs_{reference}",
                )
            )
    return tuple(records)


def _differential_design(
    design_kind: str,
    conditions: tuple[str, ...],
    *,
    dgp_family: str,
) -> DifferentialDesignSpec:
    batch_columns = ("batch",) if dgp_family == "batch_context_confounding_null" else ()
    if design_kind == "continuous":
        return DifferentialDesignSpec(
            design_kind=design_kind,
            condition_column="dose",
            batch_columns=batch_columns,
            continuous_covariates=("receiver_fraction",),
            precision_weight_column=None,
            minimum_subjects_per_level=4,
            minimum_clusters_for_cr2=6,
        )
    contrasts = tuple(
        DifferentialContrastSpec(
            name=f"{target}_vs_{reference}",
            weights=((reference, -1.0), (target, 1.0)),
        )
        for target_index, target in enumerate(conditions[1:], start=1)
        for reference in conditions[:target_index]
    )
    return DifferentialDesignSpec(
        design_kind=design_kind,
        condition_column="condition",
        condition_levels=conditions,
        batch_columns=batch_columns,
        continuous_covariates=("receiver_fraction",),
        cohort_column="cohort" if design_kind == "multi_cohort" else None,
        contrasts=contrasts,
        precision_weight_column=None,
        minimum_subjects_per_level=4,
        minimum_clusters_for_cr2=6,
    )


def _crossfit_spec(
    conditions: tuple[str, ...],
    *,
    dgp_family: str,
    seed: int,
    candidate_sender_count: int,
) -> CrossFitSpec:
    contrasts = _crossfit_contrasts(conditions)
    primary_contrast = "B_vs_A"
    observed_cell_types = tuple(
        sorted(
            (
                "Receiver",
                *(f"S{index:02d}" for index in range(1, candidate_sender_count)),
            )
        )
    )
    return CrossFitSpec(
        contrasts=contrasts,
        predeclared_receiver_ids=observed_cell_types,
        outer_fold_partition_seed=seed,
        training_spec=FoldTrainingSpec(
            min_cells=1,
            min_pooled_availability=0.0,
            max_interactions=None,
            sender_parameters=ContrastCommonSenderParameters(min_subjects=2),
        ),
        allowed_n_splits=(2,),
        min_train_subjects_per_context=2,
        min_test_subjects_per_context=1,
        penalty_tuning_spec=PenaltyTuningSpec(
            lambda1_fractions=(1.0, 0.1),
            lambda2_fractions=(0.0,),
            inner_allowed_n_splits=(2,),
            min_inner_train_subjects_per_context=1,
            min_inner_validation_subjects_per_context=1,
            root_seed=seed,
        ),
        absolute_activity_v2_spec=AbsoluteActivityV2Spec(),
        sender_attribution_v2_spec=SenderAttributionV2Spec(
            minimum_calibration_subjects=4
        ),
        signed_program_v2_spec=SignedProgramV2Spec(
            generic_state_feature_ids=("G_STATE",),
            mechanism_direction_overrides=(
                (("lr_inhibitory", "attenuation"),)
                if dgp_family == "inhibitory_program"
                else ()
            ),
            minimum_training_samples=4,
            minimum_training_subjects=3,
        ),
        eb_shrunken_coupling_v2_spec=EBShrunkenCouplingV2Spec(
            contrast_name=primary_contrast,
            minimum_subjects=4,
            minimum_observed_edges_for_eb=2,
            attribution_coupling_weight=0.25,
        ),
    )


def _expected_effects(
    dgp_family: str,
    *,
    design_kind: str,
    conditions: tuple[str, ...],
    sender_ids: tuple[str, ...],
    contrast_name: str,
) -> dict[tuple[str, str], tuple[float, float, bool, bool]]:
    if design_kind == "continuous":
        weights = {"unit_slope": 1.0}
    else:
        target, _, reference = contrast_name.partition("_vs_")
        weights = {reference: -1.0, target: 1.0}
    mode = _family_mode(dgp_family)
    true_senders = _true_senders(dgp_family, sender_ids)
    result: dict[tuple[str, str], tuple[float, float, bool, bool]] = {}
    for sender in (*sender_ids, "Receiver"):
        for definition in INTERACTIONS:
            ligand_effect = 0.0
            receptor_effect = 0.0
            program_effect = 0.0
            causal_parent = False
            for condition, weight in weights.items():
                if condition == "unit_slope":
                    scale = 1.0
                    label = "B"
                else:
                    scale = _condition_scale(condition, 0.0, design_kind)
                    label = condition
                amplitudes = _active_event_amplitudes(
                    dgp_family, condition=label, scale=scale
                )
                amplitude = float(amplitudes.get(definition.interaction_id, 0.0))
                if dgp_family == "occurrence_heterogeneity":
                    amplitude *= 0.5
                if mode in {"joint", "ligand"} and sender in true_senders:
                    ligand_effect += weight * 1.5 * amplitude
                if mode in {"joint", "receptor"}:
                    receptor_effect += weight * 1.0 * amplitude
                if mode in {"joint", "program"}:
                    raw_program = 1.2 * amplitude
                    if dgp_family == "inhibitory_program":
                        raw_program = -abs(raw_program)
                    program_effect += weight * definition.target_weight * raw_program
                causal_parent = causal_parent or not math.isclose(amplitude, 0.0)
            score_effect = 0.5 * (ligand_effect + receptor_effect)
            causal_sender = sender in true_senders and causal_parent
            if dgp_family in _NULL_FAMILIES:
                causal_parent = False
                causal_sender = False
            result[(sender, definition.interaction_id)] = (
                score_effect,
                program_effect,
                causal_sender,
                causal_parent,
            )
    return result


def _truth_table(
    *,
    dataset_id: str,
    dgp_family: str,
    design_kind: str,
    conditions: tuple[str, ...],
    sender_ids: tuple[str, ...],
    differential_design: DifferentialDesignSpec,
) -> pd.DataFrame:
    contrast_names = (
        ("slope:dose",)
        if design_kind == "continuous"
        else tuple(item.name for item in differential_design.contrasts)
    )
    records: list[dict[str, object]] = []
    for contrast_name in contrast_names:
        effects = _expected_effects(
            dgp_family,
            design_kind=design_kind,
            conditions=conditions,
            sender_ids=sender_ids,
            contrast_name=contrast_name,
        )
        parent_effects = {
            interaction.interaction_id: float(
                np.mean(
                    [
                        effects[(sender, interaction.interaction_id)][0]
                        for sender in (*sender_ids, "Receiver")
                    ]
                )
            )
            for interaction in INTERACTIONS
        }
        for sender in (*sender_ids, "Receiver"):
            for interaction in INTERACTIONS:
                score, program, causal_sender, causal_parent = effects[
                    (sender, interaction.interaction_id)
                ]
                occurrence = (
                    0.5
                    if dgp_family == "occurrence_heterogeneity" and causal_parent
                    else float(causal_parent)
                )
                parent_effect = parent_effects[interaction.interaction_id]
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "dataset_id": dataset_id,
                        "dgp_family": dgp_family,
                        "design_kind": design_kind,
                        "contrast_name": contrast_name,
                        "sender": sender,
                        "receiver": "Receiver",
                        "interaction_id": interaction.interaction_id,
                        "ligand": interaction.ligand_name,
                        "receptor": interaction.receptor_name,
                        "pathway": interaction.pathway,
                        "mechanism_class": interaction.annotation.lower(),
                        "receptor_complex_cardinality": len(
                            interaction.receptor_subunits
                        ),
                        "truth_score_effect": score,
                        "truth_parent_effect": parent_effect,
                        "truth_program_effect": program,
                        "truth_occurrence_effect": occurrence,
                        "truth_causal_sender": causal_sender,
                        "truth_causal_parent": causal_parent,
                        "truth_direction": int(np.sign(parent_effect or program)),
                    }
                )
    return pd.DataFrame.from_records(records, columns=TRUTH_COLUMNS).sort_values(
        ["contrast_name", "sender", "receiver", "interaction_id"],
        kind="stable",
        ignore_index=True,
    )


def generate_v7_dgp(
    *,
    dataset_id: str,
    dgp_family: str,
    design_kind: str,
    seed: int,
    candidate_sender_count: int = 5,
    cells_per_type: int = 4,
    subjects_per_level: int = 6,
) -> V7DGPFixture:
    """Generate one raw-count DGP without exposing planted truth to methods."""

    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string")
    if dgp_family not in REQUIRED_DGP_FAMILIES:
        raise ValueError("dgp_family is not registered")
    if design_kind not in DESIGN_KINDS:
        raise ValueError("design_kind is not registered")
    if candidate_sender_count not in CANDIDATE_SENDER_COUNTS:
        raise ValueError("candidate_sender_count must be 2, 5, 10, or 20")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("seed must be a non-negative 31-bit integer")
    if cells_per_type < 1 or subjects_per_level < 4:
        raise ValueError("cells_per_type>=1 and subjects_per_level>=4 are required")
    conditions = CONDITIONS_BY_DESIGN[design_kind]
    rng = np.random.default_rng(seed)
    metadata = _sample_layout(
        design_kind,
        subjects_per_level=subjects_per_level,
        rng=rng,
    )
    if dgp_family == "batch_context_confounding_null":
        metadata["batch"] = metadata["condition"].map(
            {"A": "batch1", "B": "batch2", "C": "batch2"}
        )
    adata, metadata, cell_counts = _simulate_counts(
        metadata,
        dgp_family=dgp_family,
        design_kind=design_kind,
        candidate_sender_count=candidate_sender_count,
        cells_per_type=cells_per_type,
        rng=rng,
    )
    resource = _resource_bundle(dgp_family)
    target_prior = _target_prior(dgp_family)
    config = CrychicConfig(
        context_keys=("condition",),
        counts_layer="counts",
        design="~ condition",
        random_seed=seed,
    )
    crossfit_spec = _crossfit_spec(
        conditions,
        dgp_family=dgp_family,
        seed=seed,
        candidate_sender_count=candidate_sender_count,
    )
    differential_design = _differential_design(
        design_kind, conditions, dgp_family=dgp_family
    )
    sender_ids = tuple(f"S{index:02d}" for index in range(1, candidate_sender_count))
    truth = _truth_table(
        dataset_id=dataset_id,
        dgp_family=dgp_family,
        design_kind=design_kind,
        conditions=conditions,
        sender_ids=sender_ids,
        differential_design=differential_design,
    )
    identity_payload: dict[str, Any] = {
        "candidate_sender_count": candidate_sender_count,
        "cells_per_type": cells_per_type,
        "dataset_id": dataset_id,
        "design_kind": design_kind,
        "dgp_family": dgp_family,
        "resource_id": resource.resource_id,
        "resource_manifest_digest": resource.manifest_digest,
        "raw_input_digest": _raw_input_digest(adata),
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "subjects_per_level": subjects_per_level,
        "target_prior_manifest_digest": target_prior.manifest_digest,
        "crossfit_spec_id": crossfit_spec.spec_id,
        "differential_design_id": differential_design.spec_id,
        "truth_digest": str(canonical_digest(truth.to_dict(orient="records"))),
    }
    fixture_id = stable_id("v7_raw_dgp_fixture", identity_payload, schema_version="1")
    return V7DGPFixture(
        dataset_id=dataset_id,
        dgp_family=dgp_family,
        design_kind=design_kind,
        seed=seed,
        candidate_sender_count=candidate_sender_count,
        adata=adata,
        raw_input_digest=str(identity_payload["raw_input_digest"]),
        config=config,
        resource=resource,
        target_prior=target_prior,
        crossfit_spec=crossfit_spec,
        differential_design=differential_design,
        sample_metadata=metadata,
        truth=truth,
        cell_counts=cell_counts,
        fixture_id=fixture_id,
    )


__all__ = [
    "CANDIDATE_SENDER_COUNTS",
    "CONDITIONS_BY_DESIGN",
    "GENES",
    "INTERACTIONS",
    "SCHEMA_VERSION",
    "TRUTH_COLUMNS",
    "V7DGPFixture",
    "generate_v7_dgp",
]
