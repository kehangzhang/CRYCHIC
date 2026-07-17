"""Opt-in LR integration over graph-fused held-out family effects.

This module deliberately defines a new score collection.  It reuses only the
availability, hard-gate, and within-family allocation evidence from intact
family-common applications.  Their incremental coefficients, selected flags,
cores, and statuses are never used because graph-joint leave-one-family-out
gain is a different estimand.

The output domain is the actual held-out family-common sample grid in each
context.  Graph effects retain a complete subject x context diagnostic grid,
but an independent-group sample need not exist in every context.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from crychic.core import ContractError, canonical_json, stable_id
from crychic.design import ContrastSpec, node_context_fields
from crychic.scoring import pair_softmin

from .crossfit import CrossFitArtifacts, CrossFitFoldArtifacts
from .graph_fused_effects import GraphFusedFamilyEffectCollection

GRAPH_FUSED_INTEGRATED_SEMANTICS = (
    "graph_joint_conditional_gain_x_frozen_mechanistic_availability_v1"
)
GRAPH_FUSED_INTEGRATED_SCORE_VERSION = "graph_fused_integrated_lr_v1"

GRAPH_FUSED_INTEGRATED_FAMILY_COLUMNS = (
    "integrated_spec_id",
    "crossfit_id",
    "graph_effect_collection_id",
    "graph_registry_id",
    "graph_id",
    "graph_record_id",
    "graph_workflow_id",
    "graph_application_id",
    "graph_problem_id",
    "graph_fit_id",
    "coefficient_digest",
    "fixed_precision_digest",
    "heldout_response_digest",
    "context_loss_digest",
    "family_common_functional_id",
    "family_common_application_id",
    "family_common_family_scores_digest",
    "family_common_member_scores_digest",
    "edge_evidence_digest",
    "fold_id",
    "contrast_id",
    "contrast_name",
    "context_id",
    "receiver",
    "sample_id",
    "subject_id",
    "family_id",
    "mode",
    "receiver_family_training_artifact_id",
    "family_basis_id",
    "family_coefficient",
    "joint_full_loss",
    "joint_loss_without_family",
    "raw_conditional_gain",
    "bounded_conditional_gain",
    "graph_effect_status",
    "graph_effect_reason_code",
    "family_availability",
    "receptor_eligible",
    "ligand_contrast_gate_status",
    "ligand_contrast_supported_interaction_count",
    "ligand_contrast_not_estimable_interaction_count",
    "within_family_entropy",
    "lr_identifiability_status",
    "family_selected",
    "integrated_family_score",
    "status",
    "reason_code",
    "score_version",
    "semantics",
    "experimental",
    "cross_receiver_comparable",
    "cross_context_comparable",
    "cross_mode_comparable",
    "formal_inference_allowed",
)

GRAPH_FUSED_INTEGRATED_LR_COLUMNS = (
    "integrated_spec_id",
    "crossfit_id",
    "graph_effect_collection_id",
    "graph_registry_id",
    "graph_id",
    "graph_record_id",
    "graph_workflow_id",
    "graph_application_id",
    "graph_problem_id",
    "graph_fit_id",
    "coefficient_digest",
    "fixed_precision_digest",
    "heldout_response_digest",
    "context_loss_digest",
    "family_common_functional_id",
    "family_common_application_id",
    "family_common_family_scores_digest",
    "family_common_member_scores_digest",
    "edge_evidence_digest",
    "fold_id",
    "contrast_id",
    "contrast_name",
    "context_id",
    "receiver",
    "sample_id",
    "subject_id",
    "family_id",
    "driver_id",
    "interaction_id",
    "mode",
    "receiver_family_training_artifact_id",
    "family_basis_id",
    "family_coefficient",
    "raw_conditional_gain",
    "bounded_conditional_gain",
    "graph_effect_status",
    "graph_effect_reason_code",
    "family_availability",
    "family_selected",
    "integrated_family_score",
    "family_status",
    "family_reason_code",
    "availability",
    "receptor_gate",
    "receptor_eligible",
    "ligand_contrast_gate",
    "ligand_contrast_gate_id",
    "ligand_contrast_gate_status",
    "ligand_contrast_gate_reason_code",
    "ligand_availability",
    "prior_quality",
    "subject_prevalence",
    "resource_evidence",
    "member_evidence_score",
    "within_family_lr_weight",
    "within_family_entropy",
    "lr_identifiability_status",
    "integrated_lr_score",
    "status",
    "reason_code",
    "score_version",
    "semantics",
    "experimental",
    "cross_receiver_comparable",
    "cross_context_comparable",
    "cross_mode_comparable",
    "formal_inference_allowed",
)

_FAMILY_KEY = (
    "fold_id",
    "contrast_id",
    "context_id",
    "receiver",
    "sample_id",
    "family_id",
    "mode",
)
_LR_KEY = (*_FAMILY_KEY, "driver_id", "interaction_id")
_SCHEMA_VERSION = "1.0.0"
_IDENTITY_SCHEMA_VERSION = "1"
_PRODUCER = "crychic.workflow.graph_fused_integrated_lr.v1"
_COLLECTION_MARKER = "crychic.graph_fused_integrated_lr_collection.v1"
_STATUSES = frozenset({"observed", "structural_zero", "not_estimable"})
_SUPPORTED_GATE = "supported"
_UNSUPPORTED_GATE = "unsupported"
_NOT_ESTIMABLE_GATE = "not_estimable"
_RECEPTOR_INELIGIBLE_GATE = "not_applicable_receptor_ineligible"


def _required_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _optional_text(value: object) -> str | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    result = str(value)
    return result if result else None


def _optional_float(value: object, *, field_name: str) -> float | None:
    if value is None or value is pd.NA:
        return None
    numeric = float(cast(Any, value))
    if math.isnan(numeric):
        return None
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite or missing")
    return numeric


def _unit_float(value: object, *, field_name: str) -> float | None:
    numeric = _optional_float(value, field_name=field_name)
    if numeric is not None and not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field_name} must lie in [0, 1] or be missing")
    return numeric


def _source_mismatch(message: str, *, field_name: str) -> ContractError:
    return ContractError(
        message,
        code="graph_fused_integrated_source_mismatch",
        field=field_name,
        remediation=(
            "Use one intact graph-effect collection and the matching cross-fit "
            "family-common availability/allocation parents"
        ),
    )


def _cell_token(value: object) -> dict[str, object]:
    if value is None or value is pd.NA or value is pd.NaT:
        return {"type": "missing", "value": None}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return {"type": "missing", "value": None}
        if not math.isfinite(value):
            raise ValueError("integrated graph score tables cannot contain infinity")
        return {"type": "float", "value": value.hex()}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": canonical_json(value),
    }


def _table_digest(name: str, table: pd.DataFrame) -> str:
    rows = [
        [_cell_token(value) for value in row]
        for row in table.itertuples(index=False, name=None)
    ]
    rows.sort(key=canonical_json)
    return str(
        stable_id(
            name,
            {"columns": list(table.columns), "rows": rows},
            schema_version=_IDENTITY_SCHEMA_VERSION,
            digest_length=64,
        )
    )


def _contrast_id(contrast: ContrastSpec) -> str:
    return str(stable_id("contrast_manifest", contrast.to_dict()))


@dataclass(frozen=True, slots=True)
class GraphFusedIntegratedLRSpec:
    """Explicit graph-context to global-one-vs-rest gate mapping."""

    context_contrasts: Mapping[str, str] | tuple[tuple[str, str], ...]
    family_selection_threshold: float = 0.0
    softmin_power: float = 4.0
    epsilon: float = 1.0e-12
    schema_version: str = _SCHEMA_VERSION
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        raw_items = (
            tuple(self.context_contrasts.items())
            if isinstance(self.context_contrasts, Mapping)
            else tuple(self.context_contrasts)
        )
        normalized = tuple(
            sorted(
                (
                    _required_name(context, field_name="context_id"),
                    _required_name(contrast, field_name="contrast_name"),
                )
                for context, contrast in raw_items
            )
        )
        if (
            not normalized
            or len({item[0] for item in normalized}) != len(normalized)
            or len({item[1] for item in normalized}) != len(normalized)
        ):
            raise ValueError("context_contrasts must be a non-empty one-to-one mapping")
        threshold = float(self.family_selection_threshold)
        power = float(self.softmin_power)
        epsilon = float(self.epsilon)
        if not math.isfinite(threshold) or threshold != 0.0:
            raise ValueError(
                "family_selection_threshold is fixed at 0.0; reporting "
                "thresholds must not alter the fitted universe"
            )
        if not math.isfinite(power) or power <= 0.0:
            raise ValueError("softmin_power must be finite and positive")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION}")
        object.__setattr__(self, "context_contrasts", normalized)
        object.__setattr__(self, "family_selection_threshold", threshold)
        object.__setattr__(self, "softmin_power", power)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "graph_fused_integrated_lr_spec",
                self._identity_payload(),
                schema_version=_IDENTITY_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        context_contrasts = cast(tuple[tuple[str, str], ...], self.context_contrasts)
        return {
            "context_contrasts": [list(item) for item in context_contrasts],
            "epsilon": self.epsilon,
            "experimental": True,
            "family_selection_threshold": self.family_selection_threshold,
            "formal_inference_allowed": False,
            "schema_version": self.schema_version,
            "softmin_power": self.softmin_power,
        }

    def _require_intact(self) -> None:
        try:
            context_contrasts = cast(
                tuple[tuple[str, str], ...], self.context_contrasts
            )
            repeated = GraphFusedIntegratedLRSpec(
                context_contrasts=context_contrasts,
                family_selection_threshold=self.family_selection_threshold,
                softmin_power=self.softmin_power,
                epsilon=self.epsilon,
                schema_version=self.schema_version,
            )
            valid = repeated.spec_id == self.spec_id
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Graph-fused integrated LR spec failed integrity validation",
                code="graph_fused_integrated_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the explicit context/contrast mapping",
            ) from error
        if not valid:
            raise ContractError(
                "Graph-fused integrated LR spec failed integrity validation",
                code="graph_fused_integrated_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the explicit context/contrast mapping",
            )

    @property
    def contrast_by_context(self) -> dict[str, str]:
        self._require_intact()
        return dict(cast(tuple[tuple[str, str], ...], self.context_contrasts))

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"spec_id": self.spec_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True)
class _ParentBinding:
    fold_id: str
    context_id: str
    contrast_id: str
    contrast_name: str
    receiver: str
    functional: Any
    application: Any
    family_scores: pd.DataFrame
    member_scores: pd.DataFrame

    def identity_payload(self) -> dict[str, object]:
        return {
            "contrast_id": self.contrast_id,
            "contrast_name": self.contrast_name,
            "context_id": self.context_id,
            "edge_evidence_digest": self.application.edge_evidence_digest,
            "family_common_application_id": self.application.application_id,
            "family_common_functional_id": (
                self.functional.family_common_functional_id
            ),
            "family_scores_digest": self.application.family_scores_digest,
            "fold_id": self.fold_id,
            "member_scores_digest": self.application.member_scores_digest,
            "receiver": self.receiver,
            "receiver_family_training_artifact_id": (
                self.functional.receiver_family.training_artifact_id
            ),
        }


@dataclass(frozen=True, slots=True)
class _ValidatedSources:
    effect_table: pd.DataFrame
    bindings: tuple[_ParentBinding, ...]
    context_contrasts: tuple[tuple[str, str, str], ...]


def _folds(crossfit: CrossFitArtifacts) -> dict[str, CrossFitFoldArtifacts]:
    result = {fold.fold_id: fold for fold in crossfit.folds}
    if len(result) != len(crossfit.folds):
        raise _source_mismatch(
            "Cross-fit folds do not have unique identities",
            field_name="crossfit.folds",
        )
    return result


def _sample_manifest(
    table: pd.DataFrame, *, field_name: str
) -> dict[str, tuple[str, str]]:
    """Return one immutable sample-to-subject/context axis for a parent table."""

    manifest: dict[str, tuple[str, str]] = {}
    for sample_id, group in table.groupby("sample_id", observed=True, sort=False):
        pairs = {
            (str(subject_id), str(context_id))
            for subject_id, context_id in zip(
                group["subject_id"], group["context_id"], strict=True
            )
        }
        if len(pairs) != 1:
            raise _source_mismatch(
                "Parent table maps one sample to multiple subject/context rows",
                field_name=field_name,
            )
        manifest[str(sample_id)] = next(iter(pairs))
    return manifest


def _validated_contrasts(
    crossfit: CrossFitArtifacts,
    effects: GraphFusedFamilyEffectCollection,
    spec: GraphFusedIntegratedLRSpec,
    effect_table: pd.DataFrame,
) -> tuple[tuple[str, str, str], ...]:
    context_ids = tuple(sorted(set(effect_table["context_id"].astype(str))))
    mapping = spec.contrast_by_context
    if set(mapping) != set(context_ids):
        raise _source_mismatch(
            "Integrated spec must map every graph context exactly once",
            field_name="spec.context_contrasts",
        )
    contrasts = {contrast.name: contrast for contrast in crossfit.spec.contrasts}
    if len(contrasts) != len(crossfit.spec.contrasts):
        raise _source_mismatch(
            "Cross-fit contrast names are not unique",
            field_name="crossfit.spec.contrasts",
        )
    first_fold = crossfit.folds[0]
    context_keys = tuple(first_fold.training.config.context_keys)
    graph = effects._source_registry.graph
    graph_contexts = tuple(
        str(node_context_fields(node, context_keys)[0]) for node in graph.nodes
    )
    if set(graph_contexts) != set(context_ids):
        raise _source_mismatch(
            "Graph contexts differ from the family-effect table",
            field_name="effects.context_id",
        )
    result: list[tuple[str, str, str]] = []
    expected_rest_weight = -1.0 / float(len(context_ids) - 1)
    for context_id, contrast_name in sorted(mapping.items()):
        contrast = contrasts.get(contrast_name)
        if contrast is None:
            raise _source_mismatch(
                "Mapped contrast is absent from the cross-fit specification",
                field_name="spec.context_contrasts",
            )
        weights: dict[str, float] = {}
        for node, weight in contrast.weights.items():
            normalized = str(node_context_fields(node, context_keys)[0])
            if normalized in weights:
                raise _source_mismatch(
                    "Mapped contrast has duplicate normalized contexts",
                    field_name="contrast.weights",
                )
            weights[normalized] = float(weight)
        expected = {
            observed: (1.0 if observed == context_id else expected_rest_weight)
            for observed in context_ids
        }
        if (
            contrast.mode != "global_one_vs_rest"
            or not contrast.estimable
            or set(weights) != set(expected)
            or any(
                not math.isclose(weights[observed], value, rel_tol=0.0, abs_tol=1.0e-12)
                for observed, value in expected.items()
            )
        ):
            raise _source_mismatch(
                "Each mapped contrast must be the focal global-one-vs-rest contrast",
                field_name="spec.context_contrasts",
            )
        result.append((context_id, contrast.name, _contrast_id(contrast)))
    return tuple(result)


def _validated_bindings(
    crossfit: CrossFitArtifacts,
    effects: GraphFusedFamilyEffectCollection,
    context_contrasts: tuple[tuple[str, str, str], ...],
    effect_table: pd.DataFrame,
) -> tuple[_ParentBinding, ...]:
    contrast_by_context = {
        context_id: (contrast_name, contrast_id)
        for context_id, contrast_name, contrast_id in context_contrasts
    }
    effect_keys = set(
        zip(
            effect_table["fold_id"].astype(str),
            effect_table["context_id"].astype(str),
            effect_table["receiver"].astype(str),
            strict=True,
        )
    )
    bindings: list[_ParentBinding] = []
    for fold in crossfit.folds:
        applications: dict[tuple[str, str], tuple[Any, Any]] = {}
        for functional, application in zip(
            fold.family_common_functionals,
            fold.family_common_applications,
            strict=True,
        ):
            functional._require_intact()
            application._require_intact()
            key = (functional.contrast_name, functional.receiver)
            if key in applications:
                raise _source_mismatch(
                    "Family-common applications duplicate contrast/receiver",
                    field_name="fold.family_common_applications",
                )
            if (
                application.functional.family_common_functional_id
                != functional.family_common_functional_id
            ):
                raise _source_mismatch(
                    "Family-common application functional lineage changed",
                    field_name="family_common_application_id",
                )
            applications[key] = (functional, application)
        fold_effects = effect_table.loc[effect_table["fold_id"].eq(fold.fold_id)]
        receivers = tuple(sorted(set(fold_effects["receiver"].astype(str))))
        for context_id, (contrast_name, contrast_id) in sorted(
            contrast_by_context.items()
        ):
            for receiver in receivers:
                key = (contrast_name, receiver)
                if key not in applications:
                    raise _source_mismatch(
                        "Mapped graph context lacks a family-common application",
                        field_name="fold.family_common_applications",
                    )
                functional, application = applications[key]
                if (
                    functional.fold_id != fold.fold_id
                    or functional.receiver != receiver
                    or functional.contrast_manifest_id != contrast_id
                    or set(functional.context_ids) != set(contrast_by_context)
                ):
                    raise _source_mismatch(
                        "Family-common functional does not match the exact focal "
                        "contrast manifest or graph context axis",
                        field_name="family_common_functional_id",
                    )
                family_scores = application.family_scores
                member_scores = application.member_scores
                local_family = family_scores.loc[
                    family_scores["context_id"].astype(str).eq(context_id)
                    & family_scores["receiver"].astype(str).eq(receiver)
                ].copy(deep=True)
                local_member = member_scores.loc[
                    member_scores["context_id"].astype(str).eq(context_id)
                    & member_scores["receiver"].astype(str).eq(receiver)
                ].copy(deep=True)
                local_effect = fold_effects.loc[
                    fold_effects["context_id"].astype(str).eq(context_id)
                    & fold_effects["receiver"].astype(str).eq(receiver)
                ]
                if local_effect.empty or local_family.empty or local_member.empty:
                    raise _source_mismatch(
                        "Graph/family-common sources lack exact focal-context rows",
                        field_name="context_id",
                    )
                family_sample_manifest = _sample_manifest(
                    local_family, field_name="family_common_family_scores"
                )
                member_sample_manifest = _sample_manifest(
                    local_member, field_name="family_common_member_scores"
                )
                if (
                    set(local_effect["receiver_family_training_artifact_id"])
                    != {functional.receiver_family.training_artifact_id}
                    or not set(local_family["family_id"].astype(str)).issubset(
                        set(local_effect["family_id"].astype(str))
                    )
                    or not set(local_family["subject_id"].astype(str)).issubset(
                        set(local_effect["subject_id"].astype(str))
                    )
                    or not set(local_family["subject_id"].astype(str)).issubset(
                        set(application.heldout_subject_ids)
                    )
                    or family_sample_manifest != member_sample_manifest
                    or local_family.duplicated(["sample_id", "family_id", "mode"]).any()
                    or local_member.duplicated(
                        ["sample_id", "interaction_id", "mode"]
                    ).any()
                ):
                    raise _source_mismatch(
                        "Graph effects and family-common evidence axes differ",
                        field_name="family_id,subject_id",
                    )
                bindings.append(
                    _ParentBinding(
                        fold_id=fold.fold_id,
                        context_id=context_id,
                        contrast_id=contrast_id,
                        contrast_name=contrast_name,
                        receiver=receiver,
                        functional=functional,
                        application=application,
                        family_scores=local_family,
                        member_scores=local_member,
                    )
                )
    observed_keys = {
        (item.fold_id, item.context_id, item.receiver) for item in bindings
    }
    if observed_keys != effect_keys:
        raise _source_mismatch(
            "Family-common bindings do not exactly cover graph effect records",
            field_name="family_common_applications",
        )
    return tuple(
        sorted(
            bindings,
            key=lambda item: (item.fold_id, item.context_id, item.receiver),
        )
    )


def _validate_sources(
    crossfit: CrossFitArtifacts,
    effects: GraphFusedFamilyEffectCollection,
    spec: GraphFusedIntegratedLRSpec,
) -> _ValidatedSources:
    if type(crossfit) is not CrossFitArtifacts:
        raise TypeError("crossfit must be producer-owned CrossFitArtifacts")
    if type(effects) is not GraphFusedFamilyEffectCollection:
        raise TypeError(
            "effects must be a producer-owned GraphFusedFamilyEffectCollection"
        )
    if not isinstance(spec, GraphFusedIntegratedLRSpec):
        raise TypeError("spec must be GraphFusedIntegratedLRSpec")
    crossfit._require_intact()
    effects._require_intact()
    spec._require_intact()
    if effects.crossfit_id != crossfit.crossfit_id:
        raise _source_mismatch(
            "Graph effects do not derive from this cross-fit run",
            field_name="effects.crossfit_id",
        )
    effect_table = effects.family_effects
    unsupported_receivers = effect_table[
        "receiver_training_support_status"
    ].astype(str).ne("observed")
    if bool(unsupported_receivers.any()):
        raise ContractError(
            "Graph-integrated LR scoring cannot manufacture sample/LR parents for "
            "a training-absent receiver",
            code="graph_fused_integrated_receiver_training_support_unavailable",
            field="effects.receiver_training_support_status",
            remediation=(
                "Use the complete graph family-effect opportunity table for typed "
                "receiver-family NE rows; integrate LR scores only when every "
                "receiver has real family-common application parents"
            ),
        )
    context_contrasts = _validated_contrasts(crossfit, effects, spec, effect_table)
    bindings = _validated_bindings(crossfit, effects, context_contrasts, effect_table)
    return _ValidatedSources(
        effect_table=effect_table,
        bindings=bindings,
        context_contrasts=context_contrasts,
    )


def _allocation_state(
    members: pd.DataFrame,
) -> tuple[bool, bool, float | None, str]:
    receptor = members["receptor_eligible"].astype(bool)
    gates = members["ligand_contrast_gate_status"].astype(str)
    supported = receptor & gates.eq(_SUPPORTED_GATE)
    eligible_ne = receptor & gates.eq(_NOT_ESTIMABLE_GATE)
    mixed_ne = bool(eligible_ne.any())
    weights = [
        _unit_float(value, field_name="within_family_lr_weight")
        for value in members.loc[supported, "within_family_lr_weight"]
    ]
    resolved = (
        bool(supported.any())
        and not mixed_ne
        and all(value is not None for value in weights)
        and math.isclose(
            sum(cast(list[float], weights)),
            1.0,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        )
    )
    entropy_values = {
        _optional_float(value, field_name="within_family_entropy")
        for value in members["within_family_entropy"]
    }
    if len(entropy_values) != 1:
        raise _source_mismatch(
            "Family-common member entropy is inconsistent within a family",
            field_name="within_family_entropy",
        )
    entropy = next(iter(entropy_values))
    identifiability_values = set(members["lr_identifiability_status"].astype(str))
    if len(identifiability_values) != 1:
        raise _source_mismatch(
            "Family-common member identifiability status is inconsistent",
            field_name="lr_identifiability_status",
        )
    return resolved, mixed_ne, entropy, next(iter(identifiability_values))


def _family_outcome(
    effect: pd.Series,
    source: pd.Series,
    members: pd.DataFrame,
    spec: GraphFusedIntegratedLRSpec,
) -> dict[str, object]:
    graph_status = str(effect["status"])
    coefficient = _optional_float(
        effect["family_coefficient"], field_name="family_coefficient"
    )
    bounded_gain = _unit_float(
        effect["bounded_conditional_gain"],
        field_name="bounded_conditional_gain",
    )
    if graph_status == "not_estimable":
        selected: bool | None = None
    elif graph_status == "structural_zero":
        selected = False
    else:
        if coefficient is None or bounded_gain is None:
            raise _source_mismatch(
                "Estimable graph effect lacks coefficient or gain",
                field_name="graph_effect",
            )
        # Selection belongs to the frozen fit.  A non-positive held-out gain
        # remains an observed validation score and must not redefine the fit
        # universe as a structural zero.
        selected = coefficient > 0.0
    availability = _unit_float(
        source["family_availability"], field_name="family_availability"
    )
    receptor_eligible = bool(source["receptor_eligible"])
    gate_status = str(source["ligand_contrast_gate_status"])
    ne_count = int(source["ligand_contrast_not_estimable_interaction_count"])
    _, _, entropy, identifiability = _allocation_state(members)
    status: str
    reason: str | None
    score: float | None
    if not receptor_eligible:
        status, reason, score = (
            "structural_zero",
            "receptor_family_ineligible",
            0.0,
        )
    elif gate_status in {_UNSUPPORTED_GATE, _RECEPTOR_INELIGIBLE_GATE}:
        status, reason, score = (
            "structural_zero",
            "ligand_contrast_not_supported",
            0.0,
        )
    elif availability == 0.0:
        status, reason, score = (
            "structural_zero",
            "family_availability_zero",
            0.0,
        )
    elif graph_status == "structural_zero":
        status, reason, score = (
            "structural_zero",
            "graph_family_coefficient_zero",
            0.0,
        )
    elif selected is False:
        status, reason, score = (
            "structural_zero",
            "graph_family_not_selected",
            0.0,
        )
    elif gate_status == _NOT_ESTIMABLE_GATE:
        status, reason, score = (
            "not_estimable",
            "ligand_contrast_not_estimable",
            None,
        )
    elif availability is None:
        status, reason, score = (
            "not_estimable",
            "family_availability_missing",
            None,
        )
    elif graph_status == "not_estimable":
        status, reason, score = (
            "not_estimable",
            _optional_text(effect["reason_code"])
            or "graph_family_effect_not_estimable",
            None,
        )
    else:
        assert bounded_gain is not None
        score = pair_softmin(
            availability,
            bounded_gain,
            power=spec.softmin_power,
            epsilon=spec.epsilon,
        )
        if score is None:  # pragma: no cover - guarded above
            raise RuntimeError("observed graph family integration returned missing")
        status, reason = "observed", None
    return {
        "family_availability": availability,
        "receptor_eligible": receptor_eligible,
        "ligand_contrast_gate_status": gate_status,
        "ligand_contrast_supported_interaction_count": int(
            source["ligand_contrast_supported_interaction_count"]
        ),
        "ligand_contrast_not_estimable_interaction_count": ne_count,
        "within_family_entropy": entropy,
        "lr_identifiability_status": identifiability,
        "family_selected": selected,
        "integrated_family_score": score,
        "status": status,
        "reason_code": reason,
    }


def _lineage(
    *,
    crossfit: CrossFitArtifacts,
    effects: GraphFusedFamilyEffectCollection,
    spec: GraphFusedIntegratedLRSpec,
    binding: _ParentBinding,
    effect: pd.Series,
) -> dict[str, object]:
    return {
        "integrated_spec_id": spec.spec_id,
        "crossfit_id": crossfit.crossfit_id,
        "graph_effect_collection_id": effects.collection_id,
        "graph_registry_id": effects.registry_id,
        "graph_id": effects.graph_id,
        "graph_record_id": effect["record_id"],
        "graph_workflow_id": effect["workflow_id"],
        "graph_application_id": effect["application_id"],
        "graph_problem_id": effect["problem_id"],
        "graph_fit_id": effect["fit_id"],
        "coefficient_digest": effect["coefficient_digest"],
        "fixed_precision_digest": effect["fixed_precision_digest"],
        "heldout_response_digest": effect["heldout_response_digest"],
        "context_loss_digest": effect["context_loss_digest"],
        "family_common_functional_id": (binding.functional.family_common_functional_id),
        "family_common_application_id": binding.application.application_id,
        "family_common_family_scores_digest": (
            binding.application.family_scores_digest
        ),
        "family_common_member_scores_digest": (
            binding.application.member_scores_digest
        ),
        "edge_evidence_digest": binding.application.edge_evidence_digest,
        "fold_id": binding.fold_id,
        "contrast_id": binding.contrast_id,
        "contrast_name": binding.contrast_name,
        "context_id": binding.context_id,
        "receiver": binding.receiver,
        "receiver_family_training_artifact_id": effect[
            "receiver_family_training_artifact_id"
        ],
        "family_basis_id": effect["family_basis_id"],
        "family_coefficient": effect["family_coefficient"],
        "raw_conditional_gain": effect["raw_conditional_gain"],
        "bounded_conditional_gain": effect["bounded_conditional_gain"],
        "graph_effect_status": effect["status"],
        "graph_effect_reason_code": effect["reason_code"],
        "score_version": GRAPH_FUSED_INTEGRATED_SCORE_VERSION,
        "semantics": GRAPH_FUSED_INTEGRATED_SEMANTICS,
        "experimental": True,
        "cross_receiver_comparable": False,
        "cross_context_comparable": False,
        "cross_mode_comparable": False,
        "formal_inference_allowed": False,
    }


def _member_outcome(
    member: pd.Series,
    family: dict[str, object],
) -> tuple[float | None, str, str | None]:
    receptor_eligible = bool(member["receptor_eligible"])
    gate_status = str(member["ligand_contrast_gate_status"])
    family_status = str(family["status"])
    family_score = cast(float | None, family["integrated_family_score"])
    weight = _unit_float(
        member["within_family_lr_weight"],
        field_name="within_family_lr_weight",
    )
    if not receptor_eligible:
        return 0.0, "structural_zero", "receptor_interaction_ineligible"
    if gate_status == _UNSUPPORTED_GATE:
        return 0.0, "structural_zero", "ligand_contrast_not_supported"
    if family_status == "structural_zero":
        return 0.0, "structural_zero", cast(str, family["reason_code"])
    if gate_status == _NOT_ESTIMABLE_GATE:
        return (
            None,
            "not_estimable",
            _optional_text(member["ligand_contrast_gate_reason_code"])
            or "ligand_contrast_not_estimable",
        )
    if family_status == "not_estimable":
        return None, "not_estimable", cast(str, family["reason_code"])
    if weight is None:
        return None, "not_estimable", "within_family_weight_missing"
    if weight == 0.0:
        return 0.0, "structural_zero", "within_family_weight_zero"
    if family_score is None:  # pragma: no cover - family contract guards this
        raise RuntimeError("observed family score is missing")
    return family_score * weight, "observed", None


def _build_tables(
    crossfit: CrossFitArtifacts,
    effects: GraphFusedFamilyEffectCollection,
    spec: GraphFusedIntegratedLRSpec,
    sources: _ValidatedSources,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effect_lookup = {
        (
            str(row["fold_id"]),
            str(row["context_id"]),
            str(row["receiver"]),
            str(row["subject_id"]),
            str(row["family_id"]),
        ): pd.Series(row, dtype=object)
        for row in sources.effect_table.to_dict(orient="records")
    }
    family_rows: list[dict[str, object]] = []
    lr_rows: list[dict[str, object]] = []
    for binding in sources.bindings:
        member_groups = {
            (str(sample_id), str(family_id), str(mode)): group.copy(deep=True)
            for (sample_id, family_id, mode), group in binding.member_scores.groupby(
                ["sample_id", "family_id", "mode"],
                observed=True,
                sort=False,
            )
        }
        for source_record in binding.family_scores.to_dict(orient="records"):
            source = pd.Series(source_record, dtype=object)
            effect_key = (
                binding.fold_id,
                binding.context_id,
                binding.receiver,
                str(source["subject_id"]),
                str(source["family_id"]),
            )
            try:
                effect = effect_lookup[effect_key]
            except KeyError as error:
                raise _source_mismatch(
                    "Family-common sample lacks its graph family effect",
                    field_name="subject_id,context_id,family_id",
                ) from error
            member_key = (
                str(source["sample_id"]),
                str(source["family_id"]),
                str(source["mode"]),
            )
            members = member_groups.get(member_key)
            if members is None or members.empty:
                raise _source_mismatch(
                    "Family-common family row lacks member allocation rows",
                    field_name="family_common_member_scores",
                )
            outcome = _family_outcome(effect, source, members, spec)
            lineage = _lineage(
                crossfit=crossfit,
                effects=effects,
                spec=spec,
                binding=binding,
                effect=effect,
            )
            family_row = {
                **lineage,
                "sample_id": source["sample_id"],
                "subject_id": source["subject_id"],
                "family_id": source["family_id"],
                "mode": source["mode"],
                "joint_full_loss": effect["full_loss"],
                "joint_loss_without_family": effect["loss_without"],
                **outcome,
            }
            family_rows.append(family_row)
            for member_record in members.to_dict(orient="records"):
                member = pd.Series(member_record, dtype=object)
                integrated, status, reason = _member_outcome(member, outcome)
                lr_rows.append(
                    {
                        **lineage,
                        "sample_id": member["sample_id"],
                        "subject_id": member["subject_id"],
                        "family_id": member["family_id"],
                        "driver_id": member["driver_id"],
                        "interaction_id": member["interaction_id"],
                        "mode": member["mode"],
                        "family_availability": outcome["family_availability"],
                        "family_selected": outcome["family_selected"],
                        "integrated_family_score": outcome["integrated_family_score"],
                        "family_status": outcome["status"],
                        "family_reason_code": outcome["reason_code"],
                        "availability": member["availability"],
                        "receptor_gate": member["receptor_gate"],
                        "receptor_eligible": bool(member["receptor_eligible"]),
                        "ligand_contrast_gate": member["ligand_contrast_gate"],
                        "ligand_contrast_gate_id": member["ligand_contrast_gate_id"],
                        "ligand_contrast_gate_status": member[
                            "ligand_contrast_gate_status"
                        ],
                        "ligand_contrast_gate_reason_code": member[
                            "ligand_contrast_gate_reason_code"
                        ],
                        "ligand_availability": member["ligand_availability"],
                        "prior_quality": member["prior_quality"],
                        "subject_prevalence": member["subject_prevalence"],
                        "resource_evidence": member["resource_evidence"],
                        "member_evidence_score": member["member_evidence_score"],
                        "within_family_lr_weight": member["within_family_lr_weight"],
                        "within_family_entropy": outcome["within_family_entropy"],
                        "lr_identifiability_status": outcome[
                            "lr_identifiability_status"
                        ],
                        "integrated_lr_score": integrated,
                        "status": status,
                        "reason_code": reason,
                    }
                )
    family_table = pd.DataFrame(
        family_rows, columns=GRAPH_FUSED_INTEGRATED_FAMILY_COLUMNS
    ).sort_values(list(_FAMILY_KEY), kind="stable", ignore_index=True)
    lr_table = pd.DataFrame(
        lr_rows, columns=GRAPH_FUSED_INTEGRATED_LR_COLUMNS
    ).sort_values(list(_LR_KEY), kind="stable", ignore_index=True)
    return family_table, lr_table


def _missing(value: object) -> bool:
    return value is None or bool(pd.isna(cast(Any, value)))


def _validate_status_values(
    table: pd.DataFrame,
    *,
    value_column: str,
    status_column: str = "status",
    reason_column: str = "reason_code",
) -> None:
    for value, raw_status, raw_reason in table.loc[
        :, [value_column, status_column, reason_column]
    ].itertuples(index=False, name=None):
        status = str(raw_status)
        reason = _optional_text(raw_reason)
        missing = _missing(value)
        if status not in _STATUSES:
            raise ValueError("integrated graph table has an unsupported status")
        if status == "observed" and (missing or reason is not None):
            raise ValueError("observed integrated graph rows require value/no reason")
        if status == "structural_zero" and (
            missing or float(value) != 0.0 or reason is None
        ):
            raise ValueError("structural-zero integrated rows require zero/reason")
        if status == "not_estimable" and (not missing or reason is None):
            raise ValueError("not-estimable integrated rows require missing/reason")
        if not missing and (
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError("integrated graph scores must lie in [0, 1]")


def _validate_tables(
    family: pd.DataFrame,
    lr: pd.DataFrame,
    *,
    crossfit_id: str,
    effects_id: str,
    spec_id: str,
) -> None:
    if tuple(family.columns) != GRAPH_FUSED_INTEGRATED_FAMILY_COLUMNS:
        raise ValueError("graph integrated family columns changed")
    if tuple(lr.columns) != GRAPH_FUSED_INTEGRATED_LR_COLUMNS:
        raise ValueError("graph integrated LR columns changed")
    if family.empty or lr.empty:
        raise ValueError("graph integrated collection must contain family/LR rows")
    if family.duplicated(list(_FAMILY_KEY)).any() or lr.duplicated(list(_LR_KEY)).any():
        raise ValueError("graph integrated tables contain duplicate primary keys")
    for table in (family, lr):
        for column, expected in (
            ("crossfit_id", crossfit_id),
            ("graph_effect_collection_id", effects_id),
            ("integrated_spec_id", spec_id),
            ("score_version", GRAPH_FUSED_INTEGRATED_SCORE_VERSION),
            ("semantics", GRAPH_FUSED_INTEGRATED_SEMANTICS),
        ):
            if set(table[column].astype(str)) != {expected}:
                raise ValueError(f"{column} changed integrated source lineage")
        if (
            any(type(value) is not bool for value in table["experimental"])
            or not bool(table["experimental"].all())
            or any(
                type(value) is not bool for value in table["cross_receiver_comparable"]
            )
            or bool(table["cross_receiver_comparable"].any())
            or any(
                type(value) is not bool for value in table["cross_context_comparable"]
            )
            or bool(table["cross_context_comparable"].any())
            or any(
                type(value) is not bool for value in table["cross_mode_comparable"]
            )
            or bool(table["cross_mode_comparable"].any())
            or any(
                type(value) is not bool for value in table["formal_inference_allowed"]
            )
            or bool(table["formal_inference_allowed"].any())
        ):
            raise ValueError("graph integrated scope flags are invalid")
    _validate_status_values(family, value_column="integrated_family_score")
    _validate_status_values(lr, value_column="integrated_lr_score")
    family_lookup = {
        tuple(str(row[column]) for column in _FAMILY_KEY): pd.Series(row, dtype=object)
        for row in family.to_dict(orient="records")
    }
    for key, group in lr.groupby(list(_FAMILY_KEY), observed=True, sort=False):
        normalized_key = tuple(str(value) for value in cast(tuple[Any, ...], key))
        family_row = family_lookup[normalized_key]
        if set(group["family_status"].astype(str)) != {str(family_row["status"])}:
            raise ValueError("member rows changed their family status")
        family_score = family_row["integrated_family_score"]
        if not _missing(family_score):
            if group["integrated_lr_score"].isna().any():
                if group["status"].eq("observed").any():
                    raise ValueError(
                        "incomplete member allocation cannot emit observed rows"
                    )
            elif not math.isclose(
                float(group["integrated_lr_score"].sum()),
                float(family_score),
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            ):
                raise ValueError("integrated LR members do not conserve family score")


def _status_counts(table: pd.DataFrame) -> tuple[tuple[str, int], ...]:
    return tuple(
        (status, int(table["status"].eq(status).sum()))
        for status in ("observed", "structural_zero", "not_estimable")
    )


@dataclass(frozen=True, slots=True, init=False)
class GraphFusedIntegratedLRCollection:
    """Producer-owned descriptive graph-integrated family and LR tables."""

    collection_id: str
    spec: GraphFusedIntegratedLRSpec
    crossfit_id: str
    graph_effect_collection_id: str
    graph_registry_id: str
    graph_id: str
    parent_bindings: tuple[dict[str, object], ...]
    family_scores_digest: str
    integrated_lr_scores_digest: str
    table_row_counts: tuple[int, int]
    family_status_counts: tuple[tuple[str, int], ...]
    lr_status_counts: tuple[tuple[str, int], ...]
    experimental: bool
    cross_receiver_comparable: bool
    cross_context_comparable: bool
    cross_mode_comparable: bool
    formal_inference_allowed: bool
    _family_scores: pd.DataFrame = field(repr=False)
    _integrated_lr_scores: pd.DataFrame = field(repr=False)
    _source_crossfit: CrossFitArtifacts = field(repr=False)
    _source_effects: GraphFusedFamilyEffectCollection = field(repr=False)
    _producer_marker: str = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "GraphFusedIntegratedLRCollection is producer-owned; use "
            "build_graph_fused_integrated_lr_scores()"
        )

    @classmethod
    def _from_tables(
        cls,
        crossfit: CrossFitArtifacts,
        effects: GraphFusedFamilyEffectCollection,
        spec: GraphFusedIntegratedLRSpec,
        sources: _ValidatedSources,
        family: pd.DataFrame,
        lr: pd.DataFrame,
    ) -> GraphFusedIntegratedLRCollection:
        self = object.__new__(cls)
        values: dict[str, object] = {
            "spec": spec,
            "crossfit_id": crossfit.crossfit_id,
            "graph_effect_collection_id": effects.collection_id,
            "graph_registry_id": effects.registry_id,
            "graph_id": effects.graph_id,
            "parent_bindings": tuple(
                binding.identity_payload() for binding in sources.bindings
            ),
            "family_scores_digest": _table_digest(
                "graph_fused_integrated_family_scores", family
            ),
            "integrated_lr_scores_digest": _table_digest(
                "graph_fused_integrated_lr_scores", lr
            ),
            "table_row_counts": (len(family), len(lr)),
            "family_status_counts": _status_counts(family),
            "lr_status_counts": _status_counts(lr),
            "experimental": True,
            "cross_receiver_comparable": False,
            "cross_context_comparable": False,
            "cross_mode_comparable": False,
            "formal_inference_allowed": False,
            "_family_scores": family.copy(deep=True),
            "_integrated_lr_scores": lr.copy(deep=True),
            "_source_crossfit": crossfit,
            "_source_effects": effects,
            "_producer_marker": _COLLECTION_MARKER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "collection_id",
            stable_id(
                "graph_fused_integrated_lr_collection",
                self._identity_payload(),
                schema_version=_IDENTITY_SCHEMA_VERSION,
            ),
        )
        self._require_intact()
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "cross_receiver_comparable": self.cross_receiver_comparable,
            "cross_context_comparable": self.cross_context_comparable,
            "cross_mode_comparable": self.cross_mode_comparable,
            "crossfit_id": self.crossfit_id,
            "experimental": self.experimental,
            "family_scores_digest": self.family_scores_digest,
            "family_status_counts": [list(item) for item in self.family_status_counts],
            "formal_inference_allowed": self.formal_inference_allowed,
            "graph_effect_collection_id": self.graph_effect_collection_id,
            "graph_id": self.graph_id,
            "graph_registry_id": self.graph_registry_id,
            "integrated_lr_scores_digest": self.integrated_lr_scores_digest,
            "lr_status_counts": [list(item) for item in self.lr_status_counts],
            "parent_bindings": list(self.parent_bindings),
            "score_version": GRAPH_FUSED_INTEGRATED_SCORE_VERSION,
            "semantics": GRAPH_FUSED_INTEGRATED_SEMANTICS,
            "spec_id": self.spec.spec_id,
            "table_row_counts": list(self.table_row_counts),
        }

    def _require_intact(self) -> None:
        try:
            self.spec._require_intact()
            sources = _validate_sources(
                self._source_crossfit, self._source_effects, self.spec
            )
            expected_family, expected_lr = _build_tables(
                self._source_crossfit,
                self._source_effects,
                self.spec,
                sources,
            )
            _validate_tables(
                self._family_scores,
                self._integrated_lr_scores,
                crossfit_id=self.crossfit_id,
                effects_id=self.graph_effect_collection_id,
                spec_id=self.spec.spec_id,
            )
            valid = (
                self._producer_marker == _COLLECTION_MARKER
                and self.crossfit_id == self._source_crossfit.crossfit_id
                and self.graph_effect_collection_id
                == self._source_effects.collection_id
                and self.graph_registry_id == self._source_effects.registry_id
                and self.graph_id == self._source_effects.graph_id
                and self.parent_bindings
                == tuple(binding.identity_payload() for binding in sources.bindings)
                and self.family_scores_digest
                == _table_digest(
                    "graph_fused_integrated_family_scores", self._family_scores
                )
                == _table_digest(
                    "graph_fused_integrated_family_scores", expected_family
                )
                and self.integrated_lr_scores_digest
                == _table_digest(
                    "graph_fused_integrated_lr_scores", self._integrated_lr_scores
                )
                == _table_digest("graph_fused_integrated_lr_scores", expected_lr)
                and self._family_scores.equals(expected_family)
                and self._integrated_lr_scores.equals(expected_lr)
                and self.table_row_counts
                == (len(self._family_scores), len(self._integrated_lr_scores))
                and self.family_status_counts == _status_counts(self._family_scores)
                and self.lr_status_counts == _status_counts(self._integrated_lr_scores)
                and self.experimental is True
                and self.cross_receiver_comparable is False
                and self.cross_context_comparable is False
                and self.cross_mode_comparable is False
                and self.formal_inference_allowed is False
                and self.collection_id
                == stable_id(
                    "graph_fused_integrated_lr_collection",
                    self._identity_payload(),
                    schema_version=_IDENTITY_SCHEMA_VERSION,
                )
            )
        except (
            AttributeError,
            ContractError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ContractError(
                "Graph-fused integrated LR collection failed integrity validation",
                code="graph_fused_integrated_collection_integrity_violation",
                field="collection_id",
                remediation="Rebuild it from intact graph and family-common parents",
            ) from error
        if not valid:
            raise ContractError(
                "Graph-fused integrated LR collection failed integrity validation",
                code="graph_fused_integrated_collection_integrity_violation",
                field="collection_id",
                remediation="Rebuild it from intact graph and family-common parents",
            )

    @property
    def family_scores(self) -> pd.DataFrame:
        self._require_intact()
        return self._family_scores.copy(deep=True)

    @property
    def integrated_lr_scores(self) -> pd.DataFrame:
        self._require_intact()
        return self._integrated_lr_scores.copy(deep=True)

    def to_manifest(self) -> dict[str, object]:
        self._require_intact()
        return {
            "collection_id": self.collection_id,
            "schema_version": _SCHEMA_VERSION,
            "producer": _PRODUCER,
            **self._identity_payload(),
            "family_grain": (
                "outer_fold_x_focal_contrast_x_receiver_x_sample_x_family_x_mode"
            ),
            "lr_grain": (
                "outer_fold_x_focal_contrast_x_receiver_x_sample_x_family_x_lr_x_mode"
            ),
            "status": "complete_descriptive",
            "comparability_scope": "within_same_spec_receiver_context_mode_only",
            "p_value": None,
            "q_value": None,
            "comm_probability": None,
        }


def build_graph_fused_integrated_lr_scores(
    crossfit: CrossFitArtifacts,
    effects: GraphFusedFamilyEffectCollection,
    *,
    spec: GraphFusedIntegratedLRSpec,
) -> GraphFusedIntegratedLRCollection:
    """Build an opt-in graph-integrated LR collection without refitting."""

    sources = _validate_sources(crossfit, effects, spec)
    family, lr = _build_tables(crossfit, effects, spec, sources)
    _validate_tables(
        family,
        lr,
        crossfit_id=crossfit.crossfit_id,
        effects_id=effects.collection_id,
        spec_id=spec.spec_id,
    )
    return GraphFusedIntegratedLRCollection._from_tables(
        crossfit, effects, spec, sources, family, lr
    )


__all__ = [
    "GRAPH_FUSED_INTEGRATED_FAMILY_COLUMNS",
    "GRAPH_FUSED_INTEGRATED_LR_COLUMNS",
    "GRAPH_FUSED_INTEGRATED_SCORE_VERSION",
    "GRAPH_FUSED_INTEGRATED_SEMANTICS",
    "GraphFusedIntegratedLRCollection",
    "GraphFusedIntegratedLRSpec",
    "build_graph_fused_integrated_lr_scores",
]
