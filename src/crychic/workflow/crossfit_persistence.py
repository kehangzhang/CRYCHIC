"""Atomic, verifiable persistence for producer-owned cross-fit diagnostics."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise, repeat
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from crychic.attribution.gain_calibration import GAIN_CALIBRATION_PERCENTILE_POLICY
from crychic.core import canonical_digest, canonical_json, stable_id
from crychic.core._validation import validation_scope
from crychic.resources import (
    MECHANISTIC_VARIANT_POLICY_ID,
    MOLECULAR_LR_EQUIVALENCE_POLICY_ID,
)
from crychic.results.errors import (
    IncompleteResultError,
    ResultValidationError,
    ResultWriteError,
)
from crychic.scoring import (
    GLOBAL_COMMON_LR_SCORE_COLUMNS,
    GLOBAL_COMMON_SENDER_SCORE_COLUMNS,
    SCORING_COLLECTION_EXTENSION_VERSION,
    SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
    FrozenLatentNuisanceSpec,
    ScoringCollectionDocument,
)

from .certification import (
    CROSSFIT_OOF_DESCRIPTIVE_SCOPE,
    FORMAL_INFERENCE_DISABLED_REASON,
    CrossFitOOFCertificationAudit,
)
from .crossfit import CrossFitArtifacts
from .semantic_scores import (
    _DIRECTIONAL_DEDICATED_REASON,
    _DIRECTIONAL_EXCLUDED_REASON,
    _NO_COMMON_SCORING_REASON,
    SEMANTIC_AVAILABILITY_COLUMNS,
    SEMANTIC_DIFFERENTIAL_EFFECT_COLUMNS,
    SEMANTIC_INTEGRATED_LR_COLUMNS,
    SEMANTIC_RECEIVER_PROGRAM_COLUMNS,
    SEMANTIC_SCORE_OUTPUTS,
    build_crossfit_semantic_scores,
)
from .semantic_scores import (
    _GRAINS as _SEMANTIC_GRAINS,
)
from .semantic_scores import (
    _table_digest as _semantic_table_digest,
)
from .semantic_scores import (
    _validate_table as _validate_semantic_table_contract,
)

CROSSFIT_RESULT_SCHEMA_VERSION = "9.0.0"
_V8_CROSSFIT_RESULT_SCHEMA_VERSION = "8.0.0"
_V7_CROSSFIT_RESULT_SCHEMA_VERSION = "7.0.0"
_V6_CROSSFIT_RESULT_SCHEMA_VERSION = "6.0.0"
_V5_CROSSFIT_RESULT_SCHEMA_VERSION = "5.0.0"
_V4_CROSSFIT_RESULT_SCHEMA_VERSION = "4.0.0"
_V3_CROSSFIT_RESULT_SCHEMA_VERSION = "3.0.0"
_V2_CROSSFIT_RESULT_SCHEMA_VERSION = "2.0.0"
_V1_CROSSFIT_RESULT_SCHEMA_VERSION = "1.0.0"
_SUPPORTED_CROSSFIT_RESULT_SCHEMA_VERSIONS = frozenset(
    {
        CROSSFIT_RESULT_SCHEMA_VERSION,
        _V8_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V7_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V6_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V5_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V4_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V3_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V2_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V1_CROSSFIT_RESULT_SCHEMA_VERSION,
    }
)
_V7_PLUS_CROSSFIT_RESULT_SCHEMA_VERSIONS = frozenset(
    {
        CROSSFIT_RESULT_SCHEMA_VERSION,
        _V8_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V7_CROSSFIT_RESULT_SCHEMA_VERSION,
    }
)
_V8_PLUS_CROSSFIT_RESULT_SCHEMA_VERSIONS = frozenset(
    {CROSSFIT_RESULT_SCHEMA_VERSION, _V8_CROSSFIT_RESULT_SCHEMA_VERSION}
)
CROSSFIT_COMPONENT_TABLE = "family_common_components"
CROSSFIT_DIFFERENTIAL_TABLE = "descriptive_differential"
CROSSFIT_CONTRAST_COMMON_LR_TABLE = "contrast_common_lr_scores"
CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE = "contrast_common_sender_lr_scores"
CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE = "directional_channel_registry"
CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE = "receiver_training_support"
CROSSFIT_SEMANTIC_AVAILABILITY_TABLE = "semantic_availability_scores"
CROSSFIT_SEMANTIC_RECEIVER_PROGRAM_TABLE = "semantic_receiver_program_scores"
CROSSFIT_SEMANTIC_INTEGRATED_LR_TABLE = "semantic_integrated_lr_scores"
CROSSFIT_SEMANTIC_DIFFERENTIAL_EFFECT_TABLE = "semantic_differential_effects"
CROSSFIT_RECEIVER_TABLE_NAMES = (
    CROSSFIT_COMPONENT_TABLE,
    CROSSFIT_DIFFERENTIAL_TABLE,
)
_V4_CROSSFIT_TABLE_NAMES = (
    *CROSSFIT_RECEIVER_TABLE_NAMES,
    CROSSFIT_CONTRAST_COMMON_LR_TABLE,
    CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
)
_V6_CROSSFIT_TABLE_NAMES = (
    *_V4_CROSSFIT_TABLE_NAMES,
    CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE,
)
_V7_CROSSFIT_TABLE_NAMES = (
    *_V6_CROSSFIT_TABLE_NAMES,
    CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE,
)
CROSSFIT_SEMANTIC_TABLE_NAMES = (
    CROSSFIT_SEMANTIC_AVAILABILITY_TABLE,
    CROSSFIT_SEMANTIC_RECEIVER_PROGRAM_TABLE,
    CROSSFIT_SEMANTIC_INTEGRATED_LR_TABLE,
    CROSSFIT_SEMANTIC_DIFFERENTIAL_EFFECT_TABLE,
)
CROSSFIT_TABLE_NAMES = (*_V7_CROSSFIT_TABLE_NAMES, *CROSSFIT_SEMANTIC_TABLE_NAMES)

_SEMANTIC_OUTPUT_TO_TABLE = dict(
    zip(SEMANTIC_SCORE_OUTPUTS, CROSSFIT_SEMANTIC_TABLE_NAMES, strict=True)
)
_SEMANTIC_TABLE_TO_OUTPUT = {
    table_name: semantic_output
    for semantic_output, table_name in _SEMANTIC_OUTPUT_TO_TABLE.items()
}
_SEMANTIC_COLLECTION_FIELDS = frozenset(
    {
        "collection_id",
        "crossfit_id",
        "crossfit_spec_id",
        "formal_inference_allowed",
        "repeat_id",
        "table_digests",
        "table_row_counts",
        "view_statuses",
        "views",
        "excluded_output_kinds",
    }
)
_SEMANTIC_VIEW_FIELDS = frozenset(
    {
        "semantic_output",
        "grain",
        "status",
        "reason_code",
        "row_count",
        "table_digest",
        "dedicated_view",
        "formal_inference_allowed",
    }
)
_SEMANTIC_VIEW_STATUSES = frozenset({"produced", "not_estimable", "not_produced"})
_SEMANTIC_EXCLUDED_OUTPUT_KINDS = (
    "p_value",
    "q_value",
    "posterior_probability",
    "communication_probability",
)
_RESPONSE_BACKEND_MANIFEST_FIELDS = frozenset(
    {
        "response_backend_manifest_id",
        "artifact_kind",
        "contrast_name",
        "feature_diagnostics",
        "feature_diagnostics_digest",
        "feature_ids",
        "fold_id",
        "formal_inference_allowed",
        "method",
        "n_backend_eligible_features",
        "n_features",
        "n_precision_supported_features",
        "precision_parent_raw_digest",
        "precision_supported_feature_indices",
        "reason_code",
        "receiver",
        "repeated_design_manifest",
        "response_artifact_id",
        "response_identity",
        "schema_version",
        "status",
    }
)
_RESPONSE_FEATURE_DIAGNOSTIC_FIELDS = frozenset(
    {
        "backend",
        "cluster_df",
        "effect_id",
        "feature_id",
        "feature_index",
        "formal_backend_eligible",
        "n_effective_clusters",
        "precision_supported",
        "reason_code",
        "status",
    }
)
_PRECISION_AUDIT_MANIFEST_FIELDS = frozenset(
    {
        "precision_audit_manifest_id",
        "numeric_values_persisted",
        "ordered_values_digest",
        "provenance",
        "replay_scope",
        "schema_version",
    }
)
_PRECISION_PROVENANCE_FIELDS = frozenset(
    {
        "precision_transform_id",
        "method",
        "receiver",
        "contrast_name",
        "fold_id",
        "feature_ids",
        "raw_precision_digest",
        "working_precision_digest",
        "transformed_precision_digest",
        "feature_scale_digest",
        "feature_scale_source",
        "residual_df",
        "residual_df_source",
        "low_df_threshold",
        "weight_mode",
        "lineage_mode",
        "response_artifact_id",
        "training_row_manifest_id",
        "training_subject_ids",
        "encoder_id",
        "lower_quantile",
        "upper_quantile",
        "lower_bound",
        "upper_bound",
        "normalization_median",
        "n_positive_features",
        "min_positive_features",
        "estimable",
        "reason_code",
    }
)
_CR1_REPEATED_RESPONSE_METHOD = "formula_ols_subject_cluster_cr1_exploratory_v1"
_CR2_REPEATED_RESPONSE_METHOD = "subject_equal_repeated_measures_cr2_outer_fold_v1"
_CR1_REPEATED_PRECISION_METHOD = (
    "repeated_cr1_diagnostic_standardized_inverse_variance_v1"
)
_CR2_REPEATED_PRECISION_METHOD = "repeated_cr2_standardized_inverse_variance_v1"
_FOLD_RESPONSE_PRECISION_METHOD = (
    "standardized_inverse_variance_with_low_df_equal_support_guardrail_v1"
)
_CR1_REPEATED_PRECISION_LINEAGE = "repeated_measures_fold_response_parented_v1"
_CR2_REPEATED_PRECISION_LINEAGE = "repeated_measures_cr2_fold_response_parented_v1"
_FOLD_RESPONSE_PRECISION_LINEAGE = "fold_gene_response_parented_v1"
_CR1_REPEATED_OFFICIAL_REASON = (
    "repeated_measures_cr1_diagnostic_only_no_formal_inference"
)
_RECEIVER_PROGRAM_SOURCE_COLUMNS = (
    "receiver_program_training_artifact_id",
    "receiver_program_application_id",
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "family_id",
    "receiver_program_score",
    "status",
    "reason_code",
)
CROSSFIT_INTEGRATED_LR_QUERY_COLUMNS = (
    *(
        column
        for column in SEMANTIC_INTEGRATED_LR_COLUMNS
        if column != "source_table_digest"
    ),
    "spec_id",
    "semantic_output",
    "directional_contrasts_excluded",
)

_STATUS_FILENAME = "_status.json"
_MANIFEST_FILENAME = "crossfit_manifest.json"
_INCOMPLETE_STATUS = "incomplete"
_COMPLETE_STATUS = "complete"
_FORMAL_INFERENCE_STATUS = "not_available_descriptive_only"
_UNCERTIFIED_CLAIM_SCOPE = "heldout_family_common_diagnostic_not_complete_oof_certified"
_CONTRAST_COMMON_UNCERTIFIED_CLAIM_SCOPE = (
    "heldout_cross_receiver_common_diagnostic_not_complete_oof_certified"
)
_FORBIDDEN_INFERENCE_FIELDS = frozenset(
    {
        "p",
        "p_value",
        "q",
        "q_value",
        "fdr",
        "posterior",
        "comm_probability",
        "communication_probability",
        "confidence_interval",
        "standard_error",
    }
)

CROSSFIT_COMPONENT_COLUMNS = (
    "crossfit_id",
    "spec_id",
    "repeat_id",
    "fold_id",
    "contrast_id",
    "contrast",
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "family_id",
    "driver_id",
    "interaction_id",
    "mode",
    "sender",
    "component_scope",
    "component",
    "component_value",
    "status",
    "reason_code",
    "row_status",
    "row_reason_code",
    "lr_identifiability_status",
    "ligand_contrast_gate_status",
    "score_version",
    "family_common_functional_id",
    "family_common_application_id",
    "family_common_binding_id",
    "sender_functional_id",
    "certification_status",
    "is_oof_certified",
    "formal_inference_status",
    "claim_scope",
    "source_table",
    "source_row_id",
)

CROSSFIT_DIFFERENTIAL_COLUMNS = (
    "crossfit_id",
    "spec_id",
    "repeat_id",
    "fold_id",
    "contrast_id",
    "contrast",
    "receiver",
    "subject_id",
    "family_id",
    "differential_effect",
    "status",
    "reason_code",
    "effect_semantics",
    "score_version",
    "family_common_functional_id",
    "family_common_application_id",
    "family_common_binding_id",
    "certification_status",
    "is_oof_certified",
    "formal_inference_status",
    "claim_scope",
    "source_row_id",
)

_CONTRAST_COMMON_PREFIX_COLUMNS = (
    "crossfit_id",
    "spec_id",
    "repeat_id",
    "fold_id",
    "contrast_id",
    "contrast",
    "contrast_common_collection_id",
    "global_common_application_id",
)
_CONTRAST_COMMON_SUFFIX_COLUMNS = (
    "certification_status",
    "is_oof_certified",
    "formal_inference_status",
    "claim_scope",
    "source_row_id",
)
CROSSFIT_CONTRAST_COMMON_LR_COLUMNS = (
    *_CONTRAST_COMMON_PREFIX_COLUMNS,
    *GLOBAL_COMMON_LR_SCORE_COLUMNS,
    *_CONTRAST_COMMON_SUFFIX_COLUMNS,
)
CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS = (
    *_CONTRAST_COMMON_PREFIX_COLUMNS,
    *GLOBAL_COMMON_SENDER_SCORE_COLUMNS,
    *_CONTRAST_COMMON_SUFFIX_COLUMNS,
)
CROSSFIT_DIRECTIONAL_CHANNEL_COLUMNS = (
    "crossfit_id",
    "spec_id",
    "repeat_id",
    "fold_id",
    "receiver",
    "pair_spec_id",
    "binding_id",
    "channel_role",
    "channel",
    "contrast_id",
    "contrast",
    "response_id",
    "training_artifact_id",
    "application_id",
    "response_pair_id",
    "response_channel_id",
    "status",
    "reason_code",
    "combination_rule",
    "active_inhibition_allowed",
    "supports_active_inhibition_claim",
    "paired_score_comparison_allowed",
    "formal_inference_allowed",
    "certification_status",
    "is_oof_certified",
    "formal_inference_status",
    "claim_scope",
)
CROSSFIT_RECEIVER_TRAINING_SUPPORT_COLUMNS = (
    "crossfit_id",
    "spec_id",
    "repeat_id",
    "receiver_universe_id",
    "receiver_axis_id",
    "fold_id",
    "receiver",
    "receiver_training_support_id",
    "receiver_training_support_status",
    "receiver_training_support_reason_code",
    "training_cell_type_ids",
)
CROSSFIT_DIRECTIONAL_OPPORTUNITY_COLUMNS = (
    "crossfit_id",
    "spec_id",
    "repeat_id",
    "pair_opportunity_id",
    "channel_opportunity_id",
    "receiver_universe_id",
    "receiver_axis_id",
    "fold_id",
    "receiver",
    "receiver_training_support_id",
    "receiver_training_support_status",
    "receiver_training_support_reason_code",
    "pair_spec_id",
    "binding_id",
    "channel_role",
    "channel",
    "contrast_id",
    "contrast",
    "response_id",
    "training_artifact_id",
    "application_id",
    "response_pair_id",
    "response_channel_id",
    "status",
    "reason_code",
    "combination_rule",
    "active_inhibition_allowed",
    "supports_active_inhibition_claim",
    "paired_score_comparison_allowed",
    "formal_inference_allowed",
    "certification_status",
    "is_oof_certified",
    "formal_inference_status",
    "claim_scope",
)

# Versions 4 and 5 predate conserved sender allocation. Keep their exact sender
# layout private so old bundles continue to validate raw-evidence multiplication.
_V5_GLOBAL_COMMON_SENDER_SCORE_COLUMNS = (
    "global_common_functional_id",
    "source_family_common_functional_id",
    "source_family_common_application_id",
    "source_sender_functional_id",
    "source_sender_application_id",
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "family_id",
    "driver_id",
    "interaction_id",
    "mode",
    "sender",
    "global_lr_score",
    "gain_calibration_binding_id",
    "gain_calibration_artifact_id",
    "gain_calibration_status",
    "gain_calibration_reason_code",
    "ligand_availability",
    "training_prevalence_prior",
    "raw_sender_evidence",
    "global_sender_lr_score",
    "status",
    "reason_code",
    "score_version",
)
_V5_CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS = (
    *_CONTRAST_COMMON_PREFIX_COLUMNS,
    *_V5_GLOBAL_COMMON_SENDER_SCORE_COLUMNS,
    *_CONTRAST_COMMON_SUFFIX_COLUMNS,
)

# Version 3 persisted the retired coefficient-quantile shrink heuristic.  Keep
# its exact source/table layouts private so released bundles remain readable
# without exposing those columns as the current contract.
_V3_GLOBAL_COMMON_LR_SCORE_COLUMNS = (
    "global_common_functional_id",
    "source_family_common_functional_id",
    "source_family_common_application_id",
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "family_id",
    "driver_id",
    "interaction_id",
    "mode",
    "receptor_eligible",
    "ligand_contrast_gate_status",
    "availability",
    "receiver_relative_family_gain",
    "prior_quality",
    "training_family_coefficient",
    "training_receiver_scale_factor",
    "global_lr_core_strength",
    "global_lr_score",
    "status",
    "reason_code",
    "score_version",
)
_V3_GLOBAL_COMMON_SENDER_SCORE_COLUMNS = (
    "global_common_functional_id",
    "source_family_common_functional_id",
    "source_family_common_application_id",
    "source_sender_functional_id",
    "source_sender_application_id",
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "family_id",
    "driver_id",
    "interaction_id",
    "mode",
    "sender",
    "global_lr_score",
    "training_receiver_scale_factor",
    "ligand_availability",
    "training_prevalence_prior",
    "raw_sender_evidence",
    "global_sender_lr_score",
    "status",
    "reason_code",
    "score_version",
)
_V3_CROSSFIT_CONTRAST_COMMON_LR_COLUMNS = (
    *_CONTRAST_COMMON_PREFIX_COLUMNS,
    *_V3_GLOBAL_COMMON_LR_SCORE_COLUMNS,
    *_CONTRAST_COMMON_SUFFIX_COLUMNS,
)
_V3_CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS = (
    *_CONTRAST_COMMON_PREFIX_COLUMNS,
    *_V3_GLOBAL_COMMON_SENDER_SCORE_COLUMNS,
    *_CONTRAST_COMMON_SUFFIX_COLUMNS,
)

_TABLE_FILENAMES = {
    CROSSFIT_COMPONENT_TABLE: "family_common_components.parquet",
    CROSSFIT_DIFFERENTIAL_TABLE: "descriptive_differential.parquet",
    CROSSFIT_CONTRAST_COMMON_LR_TABLE: "contrast_common_lr_scores.parquet",
    CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE: (
        "contrast_common_sender_lr_scores.parquet"
    ),
    CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE: (
        "directional_channel_registry.parquet"
    ),
    CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE: ("receiver_training_support.parquet"),
    CROSSFIT_SEMANTIC_AVAILABILITY_TABLE: "semantic_availability_scores.parquet",
    CROSSFIT_SEMANTIC_RECEIVER_PROGRAM_TABLE: (
        "semantic_receiver_program_scores.parquet"
    ),
    CROSSFIT_SEMANTIC_INTEGRATED_LR_TABLE: "semantic_integrated_lr_scores.parquet",
    CROSSFIT_SEMANTIC_DIFFERENTIAL_EFFECT_TABLE: (
        "semantic_differential_effects.parquet"
    ),
}
_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    CROSSFIT_COMPONENT_TABLE: CROSSFIT_COMPONENT_COLUMNS,
    CROSSFIT_DIFFERENTIAL_TABLE: CROSSFIT_DIFFERENTIAL_COLUMNS,
    CROSSFIT_CONTRAST_COMMON_LR_TABLE: CROSSFIT_CONTRAST_COMMON_LR_COLUMNS,
    CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE: (
        CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS
    ),
    CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE: (CROSSFIT_DIRECTIONAL_CHANNEL_COLUMNS),
    CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE: (
        CROSSFIT_RECEIVER_TRAINING_SUPPORT_COLUMNS
    ),
    CROSSFIT_SEMANTIC_AVAILABILITY_TABLE: SEMANTIC_AVAILABILITY_COLUMNS,
    CROSSFIT_SEMANTIC_RECEIVER_PROGRAM_TABLE: SEMANTIC_RECEIVER_PROGRAM_COLUMNS,
    CROSSFIT_SEMANTIC_INTEGRATED_LR_TABLE: SEMANTIC_INTEGRATED_LR_COLUMNS,
    CROSSFIT_SEMANTIC_DIFFERENTIAL_EFFECT_TABLE: (SEMANTIC_DIFFERENTIAL_EFFECT_COLUMNS),
}
_TABLE_SCHEMA_VERSIONS = {
    CROSSFIT_COMPONENT_TABLE: "2.0.0",
    CROSSFIT_DIFFERENTIAL_TABLE: "2.0.0",
    CROSSFIT_CONTRAST_COMMON_LR_TABLE: "2.0.0",
    CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE: "3.0.0",
    CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE: "1.0.0",
    CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE: "1.0.0",
    CROSSFIT_SEMANTIC_AVAILABILITY_TABLE: "1.0.0",
    CROSSFIT_SEMANTIC_RECEIVER_PROGRAM_TABLE: "1.0.0",
    CROSSFIT_SEMANTIC_INTEGRATED_LR_TABLE: "1.0.0",
    CROSSFIT_SEMANTIC_DIFFERENTIAL_EFFECT_TABLE: "1.0.0",
}
_V7_TABLE_SCHEMA_VERSIONS = {
    name: version
    for name, version in _TABLE_SCHEMA_VERSIONS.items()
    if name not in CROSSFIT_SEMANTIC_TABLE_NAMES
}
_V7_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    name: columns
    for name, columns in _TABLE_COLUMNS.items()
    if name not in CROSSFIT_SEMANTIC_TABLE_NAMES
}
_V6_TABLE_SCHEMA_VERSIONS = {
    name: version
    for name, version in _V7_TABLE_SCHEMA_VERSIONS.items()
    if name != CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE
}
_V6_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    name: columns
    for name, columns in _V7_TABLE_COLUMNS.items()
    if name != CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE
}
_V5_TABLE_SCHEMA_VERSIONS = {
    CROSSFIT_COMPONENT_TABLE: "2.0.0",
    CROSSFIT_DIFFERENTIAL_TABLE: "2.0.0",
    CROSSFIT_CONTRAST_COMMON_LR_TABLE: "2.0.0",
    CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE: "2.0.0",
    CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE: "1.0.0",
}
_V5_TABLE_COLUMNS = {
    CROSSFIT_COMPONENT_TABLE: CROSSFIT_COMPONENT_COLUMNS,
    CROSSFIT_DIFFERENTIAL_TABLE: CROSSFIT_DIFFERENTIAL_COLUMNS,
    CROSSFIT_CONTRAST_COMMON_LR_TABLE: CROSSFIT_CONTRAST_COMMON_LR_COLUMNS,
    CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE: (
        _V5_CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS
    ),
    CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE: (CROSSFIT_DIRECTIONAL_CHANNEL_COLUMNS),
}
_V3_TABLE_SCHEMA_VERSIONS = {
    CROSSFIT_COMPONENT_TABLE: "2.0.0",
    CROSSFIT_DIFFERENTIAL_TABLE: "2.0.0",
    CROSSFIT_CONTRAST_COMMON_LR_TABLE: "1.0.0",
    CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE: "1.0.0",
}
_V3_TABLE_COLUMNS = {
    CROSSFIT_COMPONENT_TABLE: CROSSFIT_COMPONENT_COLUMNS,
    CROSSFIT_DIFFERENTIAL_TABLE: CROSSFIT_DIFFERENTIAL_COLUMNS,
    CROSSFIT_CONTRAST_COMMON_LR_TABLE: _V3_CROSSFIT_CONTRAST_COMMON_LR_COLUMNS,
    CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE: (
        _V3_CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS
    ),
}
_LEGACY_TABLE_SCHEMA_VERSIONS = {
    CROSSFIT_COMPONENT_TABLE: "1.0.0",
    CROSSFIT_DIFFERENTIAL_TABLE: "1.0.0",
}

_FAMILY_COMPONENTS = (
    "availability_score",
    "receiver_program_score",
    "incremental_downstream_gain",
    "differential_effect",
    "family_coefficient",
    "within_family_entropy",
    "family_core_strength",
    "integrated_lr_score",
)
_MEMBER_COMPONENTS = (
    "availability",
    "receptor_gate",
    "receptor_eligible",
    "ligand_contrast_gate",
    "ligand_availability",
    "prior_quality",
    "subject_prevalence",
    "resource_evidence",
    "member_evidence_score",
    "within_family_lr_weight",
    "within_family_entropy",
    "family_core_strength",
    "sender_unresolved_strength",
)
_SENDER_COMPONENTS = (
    "assignment_weight",
    "sender_resolved_strength",
)
_SCOPE_COMPONENTS = {
    "family": frozenset(_FAMILY_COMPONENTS),
    "lr_member": frozenset(_MEMBER_COMPONENTS),
    "sender_lr_member": frozenset(_SENDER_COMPONENTS),
}
_TERMINAL_COMPONENTS = frozenset(
    {
        "family_core_strength",
        "integrated_lr_score",
        "sender_unresolved_strength",
        "sender_resolved_strength",
    }
)
_EFFECT_SEMANTICS = "descriptive_subject_receiver_null_vs_single_family_loss_ratio_v1"
_COMPONENT_STATUSES = frozenset({"observed", "not_estimable", "structural_zero"})
_DIFFERENTIAL_STATUSES = frozenset({"observed", "not_estimable", "structural_zero"})
_CONTRAST_COMMON_STATUSES = frozenset({"observed", "not_estimable", "structural_zero"})
_DIRECTIONAL_CHANNEL_STATUSES = frozenset({"observed", "not_estimable"})
_DIRECTIONAL_CHANNELS = {
    "forward": "increased_activation_compatible",
    "reverse": "reduced_activation_compatible",
}
_DIRECTIONAL_COMBINATION_RULE = "independent_contrast_views_not_additive_v1"
_CONTRAST_COMMON_COLLECTION_FIELDS = frozenset(
    {
        "contrast_common_collection_id",
        "crossfit_id",
        "spec_id",
        "repeat_id",
        "contrast_id",
        "contrast",
        "score_version",
        "estimand",
        "common_functional_across_receivers",
        "receiver_balanced_descriptive_collection",
        "formal_inference_allowed",
        "certification_status",
        "is_oof_certified",
        "claim_scope",
        "fold_applications",
    }
)
_V3_CONTRAST_COMMON_APPLICATION_FIELDS = frozenset(
    {
        "fold_id",
        "global_common_functional_id",
        "global_common_application_id",
        "functional_spec_id",
        "functional_schema_version",
        "filter_universe_id",
        "context_ids",
        "receiver_ids",
        "training_subject_ids",
        "heldout_subject_ids",
        "receiver_children",
        "sender_functional_id",
        "sender_lineages",
        "sender_application_digests",
        "calibration_policy",
        "calibration_quantile",
        "global_training_anchor",
        "receiver_training_anchors",
        "receiver_scale_factors",
        "training_only_receiver_calibration",
        "receiver_scale_amplification",
        "common_functional_across_receivers",
        "receiver_balanced_descriptive_collection",
        "interaction_mapping_digest",
        "functional_certification_status",
        "lr_scores_digest",
        "sender_scores_digest",
        "n_lr_rows",
        "n_sender_rows",
    }
)
_CONTRAST_COMMON_APPLICATION_FIELDS = frozenset(
    {
        "fold_id",
        "global_common_functional_id",
        "global_common_application_id",
        "functional_spec_id",
        "functional_schema_version",
        "filter_universe_id",
        "context_ids",
        "receiver_ids",
        "training_subject_ids",
        "heldout_subject_ids",
        "receiver_children",
        "sender_functional_id",
        "sender_lineages",
        "sender_application_digests",
        "calibration_policy",
        "scale_policy",
        "sender_policy",
        "softmin_power",
        "epsilon",
        "receiver_gain_calibration_bindings",
        "all_receivers_gain_calibrated",
        "cross_receiver_percentile_rank_eligible",
        "training_only_receiver_calibration",
        "receiver_scale_amplification",
        "common_functional_across_receivers",
        "receiver_balanced_descriptive_collection",
        "interaction_mapping_digest",
        "functional_certification_status",
        "lr_scores_digest",
        "sender_scores_digest",
        "n_lr_rows",
        "n_sender_rows",
    }
)
_GAIN_CALIBRATION_BINDING_FIELDS = frozenset(
    {
        "gain_calibration_binding_id",
        "receiver",
        "source_family_common_functional_id",
        "gain_calibration_artifact_id",
        "gain_calibration_spec_id",
        "gain_calibration_status",
        "gain_calibration_reason_code",
        "percentile_policy",
        "positive_gain_source_knots",
        "positive_gain_percentile_knots",
        "n_supported_families",
        "n_positive_observations",
        "n_distinct_positive_gains",
        "tuning_id",
        "outer_incremental_functional_id",
        "outer_selected_resolved_penalty_id",
    }
)
_CONTRAST_COMMON_CHILD_FIELDS = frozenset(
    {
        "receiver",
        "family_common_functional_id",
        "family_common_application_id",
    }
)
_CONTRAST_COMMON_SENDER_LINEAGE_FIELDS = frozenset(
    {
        "receiver",
        "sender_functional_id",
        "sender_application_id",
    }
)
_CONTRAST_COMMON_SENDER_DIGEST_FIELDS = frozenset(
    {"receiver", "sender_application_digest"}
)
_APPLICATION_LINEAGE_STRING_FIELDS = (
    "fold_id",
    "contrast_id",
    "contrast",
    "receiver",
    "family_common_functional_id",
    "family_common_application_id",
    "family_common_binding_id",
    "sender_functional_id",
    "score_version",
    "certification_status",
    "claim_scope",
)


@dataclass(frozen=True, slots=True)
class _OOFPersistenceProjection:
    """Validated certification values shared by a persisted result bundle."""

    is_oof_certified: bool
    certification_status: str
    claim_scope: str
    audit_id: str


def _oof_persistence_projection(
    artifacts: CrossFitArtifacts,
) -> _OOFPersistenceProjection:
    audit = artifacts.oof_certification_audit
    audit._require_intact()
    if (
        audit.source_crossfit_id != artifacts.crossfit_id
        or audit.formal_inference_allowed is not False
        or audit.formal_inference_reason_code != FORMAL_INFERENCE_DISABLED_REASON
    ):
        raise ValueError("cross-fit OOF certification audit lineage is invalid")
    certified = audit.is_oof_descriptive_certified
    return _OOFPersistenceProjection(
        is_oof_certified=certified,
        certification_status=(
            audit.certification_scope if certified else artifacts.certification_status
        ),
        claim_scope=(
            audit.certification_scope if certified else _UNCERTIFIED_CLAIM_SCOPE
        ),
        audit_id=audit.audit_id,
    )


def _expected_claim_scope(is_oof_certified: bool) -> str:
    return (
        CROSSFIT_OOF_DESCRIPTIVE_SCOPE if is_oof_certified else _UNCERTIFIED_CLAIM_SCOPE
    )


def _expected_contrast_common_claim_scope(is_oof_certified: bool) -> str:
    return (
        CROSSFIT_OOF_DESCRIPTIVE_SCOPE
        if is_oof_certified
        else _CONTRAST_COMMON_UNCERTIFIED_CLAIM_SCOPE
    )


def _bundle_table_names(schema_version: object) -> tuple[str, ...]:
    if schema_version in _V8_PLUS_CROSSFIT_RESULT_SCHEMA_VERSIONS:
        return CROSSFIT_TABLE_NAMES
    if schema_version == _V7_CROSSFIT_RESULT_SCHEMA_VERSION:
        return _V7_CROSSFIT_TABLE_NAMES
    if schema_version in {
        _V6_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V5_CROSSFIT_RESULT_SCHEMA_VERSION,
    }:
        return _V6_CROSSFIT_TABLE_NAMES
    if schema_version in {
        _V4_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V3_CROSSFIT_RESULT_SCHEMA_VERSION,
    }:
        return _V4_CROSSFIT_TABLE_NAMES
    if schema_version in {
        _V2_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V1_CROSSFIT_RESULT_SCHEMA_VERSION,
    }:
        return CROSSFIT_RECEIVER_TABLE_NAMES
    raise ValueError("cross-fit result schema version is unsupported")


def _table_schema_version(schema_version: object, table_name: str) -> str:
    if schema_version == _V1_CROSSFIT_RESULT_SCHEMA_VERSION:
        return _LEGACY_TABLE_SCHEMA_VERSIONS[table_name]
    if schema_version == _V3_CROSSFIT_RESULT_SCHEMA_VERSION:
        return _V3_TABLE_SCHEMA_VERSIONS[table_name]
    if schema_version == _V6_CROSSFIT_RESULT_SCHEMA_VERSION:
        return _V6_TABLE_SCHEMA_VERSIONS[table_name]
    if schema_version == _V7_CROSSFIT_RESULT_SCHEMA_VERSION:
        return _V7_TABLE_SCHEMA_VERSIONS[table_name]
    if schema_version in {
        _V4_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V5_CROSSFIT_RESULT_SCHEMA_VERSION,
    }:
        return _V5_TABLE_SCHEMA_VERSIONS[table_name]
    return _TABLE_SCHEMA_VERSIONS[table_name]


def _table_columns(schema_version: object, table_name: str) -> tuple[str, ...]:
    if schema_version == _V3_CROSSFIT_RESULT_SCHEMA_VERSION:
        return _V3_TABLE_COLUMNS[table_name]
    if schema_version == _V6_CROSSFIT_RESULT_SCHEMA_VERSION:
        return _V6_TABLE_COLUMNS[table_name]
    if schema_version == _V7_CROSSFIT_RESULT_SCHEMA_VERSION:
        return _V7_TABLE_COLUMNS[table_name]
    if schema_version in {
        _V4_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V5_CROSSFIT_RESULT_SCHEMA_VERSION,
    }:
        return _V5_TABLE_COLUMNS[table_name]
    return _TABLE_COLUMNS[table_name]


def _canonical_table_scalar(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if hasattr(value, "item"):
        try:
            value = cast(Any, value).item()
        except ValueError:
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _global_source_table_digest(
    name: str,
    table: pd.DataFrame,
    columns: tuple[str, ...],
) -> str:
    if tuple(table.columns) != columns:
        raise ValueError(f"{name} columns do not match the producer contract")
    recognized_prefix: tuple[str, ...] | None = None
    if name == "cross_receiver_common_lr_scores" and columns == (
        GLOBAL_COMMON_LR_SCORE_COLUMNS
    ):
        recognized_prefix = columns[:11]
    elif name == "cross_receiver_common_sender_scores" and columns == (
        GLOBAL_COMMON_SENDER_SCORE_COLUMNS
    ):
        recognized_prefix = columns[:14]
    if recognized_prefix is not None:
        prefix_table = table.loc[:, list(recognized_prefix)]
        standard_prefix = all(
            isinstance(prefix_table[column].dtype, pd.StringDtype)
            for column in recognized_prefix
        )
        if (
            standard_prefix
            and not bool(prefix_table.isna().any(axis=None))
            and not bool(prefix_table.duplicated().any())
        ):
            from crychic.scoring.global_common import _table_digest

            return _table_digest(
                name,
                table,
                columns,
                unique_text_prefix=recognized_prefix,
            )
    stable_id(name, (), schema_version="1", digest_length=64)
    prefix = f'{{"components":{{"columns":{canonical_json(list(columns))},"rows":['
    suffix = f']}},"kind":{canonical_json(name)},"schema_version":"1"}}'

    string_tokens: dict[str, str] = {}
    integer_tokens: dict[int, str] = {}

    def encoded_scalar(value: object) -> str:
        if value is None or value is pd.NA:
            return "null"
        if type(value) is str:
            token = string_tokens.get(value)
            if token is None:
                token = json.dumps(
                    value,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                if len(string_tokens) < 65_536:
                    string_tokens[value] = token
            return token
        if type(value) is bool:
            return "true" if value else "false"
        if type(value) is int:
            token = integer_tokens.get(value)
            if token is None:
                token = str(value)
                if len(integer_tokens) < 4_096:
                    integer_tokens[value] = token
            return token
        if type(value) is float:
            if math.isnan(value):
                return "null"
            if math.isfinite(value):
                return repr(value)
        return json.dumps(
            _canonical_table_scalar(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        )

    def encoded_rows() -> Any:
        for row in table.itertuples(index=False, name=None):
            yield "[" + ",".join(encoded_scalar(value) for value in row) + "]"

    digest = hashlib.sha256()
    digest.update(prefix.encode("ascii"))
    previous: str | None = None
    row_count = 0
    for encoded in encoded_rows():
        if previous is not None and encoded < previous:
            break
        if row_count:
            digest.update(b",")
        digest.update(encoded.encode("ascii"))
        previous = encoded
        row_count += 1
    else:
        digest.update(suffix.encode("ascii"))
        return f"{name}_{digest.hexdigest()}"

    rows = sorted(encoded_rows())
    digest = hashlib.sha256()
    digest.update(prefix.encode("ascii"))
    digest.update(",".join(rows).encode("ascii"))
    digest.update(suffix.encode("ascii"))
    return f"{name}_{digest.hexdigest()}"


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(f"{canonical_json(value)}\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return cast(dict[str, Any], value)


def _mark_incomplete(path: Path, error_type: str | None = None) -> None:
    marker: dict[str, object] = {
        "schema_version": CROSSFIT_RESULT_SCHEMA_VERSION,
        "status": _INCOMPLETE_STATUS,
    }
    if error_type is not None:
        marker["error_type"] = error_type
    try:
        _write_json(path / _STATUS_FILENAME, marker)
    except OSError:
        pass


def _optional_string(value: object) -> str | None:
    if value is None or value is pd.NA or bool(pd.isna(cast(Any, value))):
        return None
    result = str(value)
    if not result:
        raise ValueError("persisted optional identifiers must be non-empty")
    return result


def _optional_float(value: object, *, field_name: str) -> float | None:
    if value is None or value is pd.NA or bool(pd.isna(cast(Any, value))):
        return None
    result = float(cast(Any, value))
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite or missing")
    return result


_SOURCE_ROW_ID_KIND = "persisted_crossfit_source_row"
_SOURCE_ROW_ID_COMPONENT_KEYS = (
    "application_id",
    "context_id",
    "driver_id",
    "family_id",
    "interaction_id",
    "mode",
    "receiver",
    "sample_id",
    "sender",
    "source_table",
    "subject_id",
)


def _fast_json_string(value: object) -> str | None:
    if value is None:
        return "null"
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isprintable()
        or '"' in value
        or "\\" in value
    ):
        return None
    return f'"{value}"'


def _source_row_id(
    *,
    source_table: str,
    application_id: str,
    sample_id: str | None = None,
    subject_id: str | None = None,
    context_id: str | None = None,
    receiver: str,
    family_id: str,
    driver_id: str | None = None,
    interaction_id: str | None = None,
    mode: str | None = None,
    sender: str | None = None,
) -> str:
    components = {
        "application_id": application_id,
        "context_id": context_id,
        "driver_id": driver_id,
        "family_id": family_id,
        "interaction_id": interaction_id,
        "mode": mode,
        "receiver": receiver,
        "sample_id": sample_id,
        "sender": sender,
        "source_table": source_table,
        "subject_id": subject_id,
    }
    encoded_values = {
        key: _fast_json_string(components[key]) for key in _SOURCE_ROW_ID_COMPONENT_KEYS
    }
    if all(value is not None for value in encoded_values.values()):
        encoded_components = ",".join(
            f'"{key}":{encoded_values[key]}' for key in _SOURCE_ROW_ID_COMPONENT_KEYS
        )
        payload = (
            f'{{"components":{{{encoded_components}}},'
            f'"kind":"{_SOURCE_ROW_ID_KIND}","schema_version":"1"}}'
        )
        digest = hashlib.sha256(payload.encode("ascii")).hexdigest()[:32]
        return f"{_SOURCE_ROW_ID_KIND}_{digest}"
    return stable_id(
        _SOURCE_ROW_ID_KIND,
        components,
        schema_version="1",
    )


def _source_row_ids(
    *,
    source_table: str,
    application_ids: np.ndarray,
    sample_ids: np.ndarray,
    subject_ids: np.ndarray,
    context_ids: np.ndarray,
    receivers: np.ndarray,
    family_ids: np.ndarray,
    driver_ids: np.ndarray,
    interaction_ids: np.ndarray,
    modes: np.ndarray,
    senders: np.ndarray,
) -> np.ndarray:
    arrays = (
        application_ids,
        context_ids,
        driver_ids,
        family_ids,
        interaction_ids,
        modes,
        receivers,
        sample_ids,
        senders,
        subject_ids,
    )
    row_count = len(application_ids)
    if any(len(values) != row_count for values in arrays):
        raise ValueError("source-row identity columns must have equal lengths")

    def legacy() -> np.ndarray:
        return cast(
            np.ndarray,
            np.fromiter(
                (
                    _source_row_id(
                        source_table=source_table,
                        application_id=cast(str, application_ids[index]),
                        sample_id=cast(str | None, sample_ids[index]),
                        subject_id=cast(str | None, subject_ids[index]),
                        context_id=cast(str | None, context_ids[index]),
                        receiver=cast(str, receivers[index]),
                        family_id=cast(str, family_ids[index]),
                        driver_id=cast(str | None, driver_ids[index]),
                        interaction_id=cast(str | None, interaction_ids[index]),
                        mode=cast(str | None, modes[index]),
                        sender=cast(str | None, senders[index]),
                    )
                    for index in range(row_count)
                ),
                dtype=object,
                count=row_count,
            ),
        )

    encoded_source_table = _fast_json_string(source_table)
    if encoded_source_table is None:
        return legacy()
    encoded_columns: list[np.ndarray] = []
    for key, values in zip(
        (
            "application_id",
            "context_id",
            "driver_id",
            "family_id",
            "interaction_id",
            "mode",
            "receiver",
            "sample_id",
            "sender",
            "subject_id",
        ),
        arrays,
        strict=True,
    ):
        codes, uniques = pd.factorize(values, sort=False)
        fragments: list[str] = []
        for value in uniques:
            encoded = _fast_json_string(value)
            if encoded is None:
                return legacy()
            fragments.append(f'"{key}":{encoded}')
        if bool(np.any(codes < 0)):
            missing_index = len(fragments)
            fragments.append(f'"{key}":null')
            codes = codes.copy()
            codes[codes < 0] = missing_index
        encoded_columns.append(np.asarray(fragments, dtype=object)[codes])

    source_fragment = f'"source_table":{encoded_source_table}'
    prefix = '{"components":{'
    suffix = f'}},"kind":"{_SOURCE_ROW_ID_KIND}","schema_version":"1"}}'
    row_fragments = zip(
        *encoded_columns[:9],
        repeat(source_fragment, row_count),
        encoded_columns[9],
        strict=True,
    )
    return cast(
        np.ndarray,
        np.fromiter(
            (
                f"{_SOURCE_ROW_ID_KIND}_"
                + hashlib.sha256(
                    (prefix + ",".join(fragments) + suffix).encode("ascii")
                ).hexdigest()[:32]
                for fragments in row_fragments
            ),
            dtype=object,
            count=row_count,
        ),
    )


def _component_status(
    *,
    component: str,
    value: float | None,
    row_status: str,
    row_reason_code: str | None,
    explicit_status: str | None = None,
    explicit_reason_code: str | None = None,
) -> tuple[str, str | None]:
    if explicit_status is not None:
        if explicit_status == "ok":
            return "observed", None
        return explicit_status, explicit_reason_code
    if value is None:
        return "not_estimable", row_reason_code or "source_component_value_missing"
    if component in _TERMINAL_COMPONENTS and row_status == "structural_zero":
        if value != 0.0:
            raise ValueError("structural-zero terminal component must equal zero")
        return "structural_zero", row_reason_code
    return "observed", None


def _base_component_row(
    artifacts: CrossFitArtifacts,
    *,
    projection: _OOFPersistenceProjection,
    fold_id: str,
    functional: Any,
    application: Any,
    binding: Any,
    source_table: str,
    source_row_id: str,
    sample_id: str,
    subject_id: str,
    context_id: str,
    receiver: str,
    family_id: str,
    driver_id: str | None,
    interaction_id: str | None,
    mode: str,
    sender: str | None,
    component_scope: str,
    component: str,
    component_value: float | None,
    status: str,
    reason_code: str | None,
    row_status: str,
    row_reason_code: str | None,
    lr_identifiability_status: str | None,
    ligand_contrast_gate_status: str | None,
) -> dict[str, object]:
    return {
        "crossfit_id": artifacts.crossfit_id,
        "spec_id": artifacts.spec.spec_id,
        "repeat_id": artifacts.spec.repeat_id,
        "fold_id": fold_id,
        "contrast_id": functional.contrast_manifest_id,
        "contrast": functional.contrast_name,
        "sample_id": sample_id,
        "subject_id": subject_id,
        "context_id": context_id,
        "receiver": receiver,
        "family_id": family_id,
        "driver_id": driver_id,
        "interaction_id": interaction_id,
        "mode": mode,
        "sender": sender,
        "component_scope": component_scope,
        "component": component,
        "component_value": component_value,
        "status": status,
        "reason_code": reason_code,
        "row_status": row_status,
        "row_reason_code": row_reason_code,
        "lr_identifiability_status": lr_identifiability_status,
        "ligand_contrast_gate_status": ligand_contrast_gate_status,
        "score_version": functional.score_version,
        "family_common_functional_id": functional.family_common_functional_id,
        "family_common_application_id": application.application_id,
        "family_common_binding_id": binding.binding_id,
        "sender_functional_id": functional.sender_functional.sender_functional_id,
        "certification_status": projection.certification_status,
        "is_oof_certified": projection.is_oof_certified,
        "formal_inference_status": _FORMAL_INFERENCE_STATUS,
        "claim_scope": projection.claim_scope,
        "source_table": source_table,
        "source_row_id": source_row_id,
    }


def _standard_required_string_array(values: pd.Series) -> np.ndarray | None:
    if isinstance(values.dtype, pd.StringDtype):
        if bool(values.isna().any()):
            return None
        return cast(np.ndarray, values.to_numpy(dtype=object, copy=False))
    if isinstance(values.dtype, pd.CategoricalDtype):
        if bool(values.isna().any()) or not all(
            type(value) is str for value in values.cat.categories
        ):
            return None
        return cast(np.ndarray, values.to_numpy(dtype=object, copy=False))
    return None


def _required_string_array(values: pd.Series) -> np.ndarray:
    """Match ``str(value)`` while keeping ordinary string columns vectorized."""

    standard = _standard_required_string_array(values)
    if standard is not None:
        return standard
    return cast(
        np.ndarray,
        np.fromiter(
            (str(value) for value in values), dtype=object, count=len(values)
        ),
    )


def _optional_string_array(values: pd.Series, *, field_name: str) -> np.ndarray:
    missing = values.isna().to_numpy(dtype=bool, copy=False)
    result: np.ndarray
    if _standard_optional_string_presence(values) is None:
        result = np.fromiter(
            (_optional_string(value) for value in values),
            dtype=object,
            count=len(values),
        )
    else:
        result = values.to_numpy(dtype=object, copy=True)
        result[missing] = None
    if any(value == "" for value in result[~missing]):
        raise ValueError("persisted optional identifiers must be non-empty")
    return result


def _finite_float_matrix(
    source: pd.DataFrame,
    columns: tuple[str, ...],
) -> np.ndarray:
    values: list[np.ndarray] = []
    for column in columns:
        numeric = pd.to_numeric(source[column], errors="raise").to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        if np.isinf(numeric).any():
            raise ValueError(f"{column} must be finite or missing")
        values.append(numeric)
    if not values:
        return np.empty((len(source), 0), dtype=float)
    return cast(np.ndarray, np.column_stack(values))


def _component_source_row_ids(
    source: pd.DataFrame,
    *,
    source_table: str,
    application_id: str,
) -> np.ndarray:
    sample_ids = _required_string_array(source["sample_id"])
    subject_ids = _required_string_array(source["subject_id"])
    context_ids = _required_string_array(source["context_id"])
    receivers = _required_string_array(source["receiver"])
    family_ids = _required_string_array(source["family_id"])
    modes = _required_string_array(source["mode"])
    has_member_ids = source_table != "family_scores"
    driver_ids = (
        _required_string_array(source["driver_id"])
        if has_member_ids
        else np.full(len(source), None, dtype=object)
    )
    interaction_ids = (
        _required_string_array(source["interaction_id"])
        if has_member_ids
        else np.full(len(source), None, dtype=object)
    )
    senders = (
        _required_string_array(source["sender"])
        if source_table == "sender_scores"
        else np.full(len(source), None, dtype=object)
    )
    return _source_row_ids(
        source_table=source_table,
        application_ids=np.full(len(source), application_id, dtype=object),
        sample_ids=sample_ids,
        subject_ids=subject_ids,
        context_ids=context_ids,
        receivers=receivers,
        family_ids=family_ids,
        driver_ids=driver_ids,
        interaction_ids=interaction_ids,
        modes=modes,
        senders=senders,
    )


def _component_frame(
    artifacts: CrossFitArtifacts,
    *,
    projection: _OOFPersistenceProjection,
    fold_id: str,
    functional: Any,
    application: Any,
    binding: Any,
    source_table: str,
    source: pd.DataFrame,
    component_scope: str,
    components: tuple[str, ...],
) -> pd.DataFrame:
    if source.empty:
        return pd.DataFrame(columns=CROSSFIT_COMPONENT_COLUMNS)

    row_count = len(source)
    component_count = len(components)
    output_count = row_count * component_count
    source_row_ids = _component_source_row_ids(
        source,
        source_table=source_table,
        application_id=application.application_id,
    )
    component_names = np.tile(np.asarray(components, dtype=object), row_count)
    component_values = _finite_float_matrix(source, components).reshape(-1)
    row_status = _required_string_array(source["status"])
    row_reason = _optional_string_array(
        source["reason_code"],
        field_name="reason_code",
    )
    repeated_row_status = np.repeat(row_status, component_count)
    repeated_row_reason = np.repeat(row_reason, component_count)

    status = np.full(output_count, "observed", dtype=object)
    reason = np.full(output_count, None, dtype=object)
    missing = np.isnan(component_values)
    status[missing] = "not_estimable"
    reason[missing] = np.where(
        pd.notna(repeated_row_reason[missing]),
        repeated_row_reason[missing],
        "source_component_value_missing",
    )

    terminal = (
        np.isin(component_names, tuple(_TERMINAL_COMPONENTS))
        & (repeated_row_status == "structural_zero")
        & ~missing
    )
    if np.any(terminal & (component_values != 0.0)):
        raise ValueError("structural-zero terminal component must equal zero")
    status[terminal] = "structural_zero"
    reason[terminal] = repeated_row_reason[terminal]

    if source_table == "family_scores":
        receiver_program = component_names == "receiver_program_score"
        explicit_status = np.repeat(
            _required_string_array(source["receiver_program_status"]),
            component_count,
        )
        explicit_reason = np.repeat(
            _optional_string_array(
                source["receiver_program_reason_code"],
                field_name="receiver_program_reason_code",
            ),
            component_count,
        )
        explicit_ok = receiver_program & (explicit_status == "ok")
        explicit_other = receiver_program & ~explicit_ok
        status[explicit_ok] = "observed"
        reason[explicit_ok] = None
        status[explicit_other] = explicit_status[explicit_other]
        reason[explicit_other] = explicit_reason[explicit_other]

    def repeated(column: str) -> np.ndarray:
        return np.repeat(_required_string_array(source[column]), component_count)

    if source_table == "family_scores":
        driver_id = np.full(output_count, None, dtype=object)
        interaction_id = np.full(output_count, None, dtype=object)
        sender = np.full(output_count, None, dtype=object)
        lr_status = np.repeat(
            _optional_string_array(
                source["lr_identifiability_status"],
                field_name="lr_identifiability_status",
            ),
            component_count,
        )
        ligand_status = np.repeat(
            _optional_string_array(
                source["ligand_contrast_gate_status"],
                field_name="ligand_contrast_gate_status",
            ),
            component_count,
        )
    elif source_table == "member_scores":
        driver_id = repeated("driver_id")
        interaction_id = repeated("interaction_id")
        sender = np.full(output_count, None, dtype=object)
        lr_status = np.repeat(
            _optional_string_array(
                source["lr_identifiability_status"],
                field_name="lr_identifiability_status",
            ),
            component_count,
        )
        ligand_status = np.repeat(
            _optional_string_array(
                source["ligand_contrast_gate_status"],
                field_name="ligand_contrast_gate_status",
            ),
            component_count,
        )
    elif source_table == "sender_scores":
        driver_id = repeated("driver_id")
        interaction_id = repeated("interaction_id")
        sender = repeated("sender")
        lr_status = np.full(output_count, None, dtype=object)
        ligand_status = np.full(output_count, None, dtype=object)
    else:  # pragma: no cover - private caller invariant
        raise KeyError(source_table)

    frame = pd.DataFrame(
        {
            "crossfit_id": np.full(output_count, artifacts.crossfit_id, dtype=object),
            "spec_id": np.full(output_count, artifacts.spec.spec_id, dtype=object),
            "repeat_id": np.full(output_count, artifacts.spec.repeat_id, dtype=object),
            "fold_id": np.full(output_count, fold_id, dtype=object),
            "contrast_id": np.full(
                output_count,
                functional.contrast_manifest_id,
                dtype=object,
            ),
            "contrast": np.full(
                output_count,
                functional.contrast_name,
                dtype=object,
            ),
            "sample_id": repeated("sample_id"),
            "subject_id": repeated("subject_id"),
            "context_id": repeated("context_id"),
            "receiver": repeated("receiver"),
            "family_id": repeated("family_id"),
            "driver_id": driver_id,
            "interaction_id": interaction_id,
            "mode": repeated("mode"),
            "sender": sender,
            "component_scope": np.full(
                output_count,
                component_scope,
                dtype=object,
            ),
            "component": component_names,
            "component_value": component_values,
            "status": status,
            "reason_code": reason,
            "row_status": repeated_row_status,
            "row_reason_code": repeated_row_reason,
            "lr_identifiability_status": lr_status,
            "ligand_contrast_gate_status": ligand_status,
            "score_version": np.full(
                output_count,
                functional.score_version,
                dtype=object,
            ),
            "family_common_functional_id": np.full(
                output_count,
                functional.family_common_functional_id,
                dtype=object,
            ),
            "family_common_application_id": np.full(
                output_count,
                application.application_id,
                dtype=object,
            ),
            "family_common_binding_id": np.full(
                output_count,
                binding.binding_id,
                dtype=object,
            ),
            "sender_functional_id": np.full(
                output_count,
                functional.sender_functional.sender_functional_id,
                dtype=object,
            ),
            "certification_status": np.full(
                output_count,
                projection.certification_status,
                dtype=object,
            ),
            "is_oof_certified": np.full(
                output_count,
                projection.is_oof_certified,
                dtype=bool,
            ),
            "formal_inference_status": np.full(
                output_count,
                _FORMAL_INFERENCE_STATUS,
                dtype=object,
            ),
            "claim_scope": np.full(
                output_count,
                projection.claim_scope,
                dtype=object,
            ),
            "source_table": np.full(output_count, source_table, dtype=object),
            "source_row_id": np.repeat(source_row_ids, component_count),
        },
        columns=CROSSFIT_COMPONENT_COLUMNS,
        dtype=object,
    )
    frame["component_value"] = component_values
    frame["is_oof_certified"] = np.full(
        output_count,
        projection.is_oof_certified,
        dtype=bool,
    )
    return frame


def _family_component_rows(
    artifacts: CrossFitArtifacts,
    *,
    projection: _OOFPersistenceProjection,
    fold_id: str,
    functional: Any,
    application: Any,
    binding: Any,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in application.family_scores.itertuples(index=False):
        row_status = str(source.status)
        row_reason = _optional_string(source.reason_code)
        row_id = _source_row_id(
            source_table="family_scores",
            application_id=application.application_id,
            sample_id=str(source.sample_id),
            subject_id=str(source.subject_id),
            context_id=str(source.context_id),
            receiver=str(source.receiver),
            family_id=str(source.family_id),
            mode=str(source.mode),
        )
        for component in _FAMILY_COMPONENTS:
            value = _optional_float(getattr(source, component), field_name=component)
            explicit_status = None
            explicit_reason = None
            if component == "receiver_program_score":
                explicit_status = str(source.receiver_program_status)
                explicit_reason = _optional_string(source.receiver_program_reason_code)
            status, reason = _component_status(
                component=component,
                value=value,
                row_status=row_status,
                row_reason_code=row_reason,
                explicit_status=explicit_status,
                explicit_reason_code=explicit_reason,
            )
            rows.append(
                _base_component_row(
                    artifacts,
                    projection=projection,
                    fold_id=fold_id,
                    functional=functional,
                    application=application,
                    binding=binding,
                    source_table="family_scores",
                    source_row_id=row_id,
                    sample_id=str(source.sample_id),
                    subject_id=str(source.subject_id),
                    context_id=str(source.context_id),
                    receiver=str(source.receiver),
                    family_id=str(source.family_id),
                    driver_id=None,
                    interaction_id=None,
                    mode=str(source.mode),
                    sender=None,
                    component_scope="family",
                    component=component,
                    component_value=value,
                    status=status,
                    reason_code=reason,
                    row_status=row_status,
                    row_reason_code=row_reason,
                    lr_identifiability_status=_optional_string(
                        source.lr_identifiability_status
                    ),
                    ligand_contrast_gate_status=_optional_string(
                        source.ligand_contrast_gate_status
                    ),
                )
            )
    return rows


def _member_component_rows(
    artifacts: CrossFitArtifacts,
    *,
    projection: _OOFPersistenceProjection,
    fold_id: str,
    functional: Any,
    application: Any,
    binding: Any,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in application.member_scores.itertuples(index=False):
        row_status = str(source.status)
        row_reason = _optional_string(source.reason_code)
        row_id = _source_row_id(
            source_table="member_scores",
            application_id=application.application_id,
            sample_id=str(source.sample_id),
            subject_id=str(source.subject_id),
            context_id=str(source.context_id),
            receiver=str(source.receiver),
            family_id=str(source.family_id),
            driver_id=str(source.driver_id),
            interaction_id=str(source.interaction_id),
            mode=str(source.mode),
        )
        for component in _MEMBER_COMPONENTS:
            value = _optional_float(getattr(source, component), field_name=component)
            status, reason = _component_status(
                component=component,
                value=value,
                row_status=row_status,
                row_reason_code=row_reason,
            )
            rows.append(
                _base_component_row(
                    artifacts,
                    projection=projection,
                    fold_id=fold_id,
                    functional=functional,
                    application=application,
                    binding=binding,
                    source_table="member_scores",
                    source_row_id=row_id,
                    sample_id=str(source.sample_id),
                    subject_id=str(source.subject_id),
                    context_id=str(source.context_id),
                    receiver=str(source.receiver),
                    family_id=str(source.family_id),
                    driver_id=str(source.driver_id),
                    interaction_id=str(source.interaction_id),
                    mode=str(source.mode),
                    sender=None,
                    component_scope="lr_member",
                    component=component,
                    component_value=value,
                    status=status,
                    reason_code=reason,
                    row_status=row_status,
                    row_reason_code=row_reason,
                    lr_identifiability_status=_optional_string(
                        source.lr_identifiability_status
                    ),
                    ligand_contrast_gate_status=_optional_string(
                        source.ligand_contrast_gate_status
                    ),
                )
            )
    return rows


def _sender_component_rows(
    artifacts: CrossFitArtifacts,
    *,
    projection: _OOFPersistenceProjection,
    fold_id: str,
    functional: Any,
    application: Any,
    binding: Any,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in application.sender_scores.itertuples(index=False):
        row_status = str(source.status)
        row_reason = _optional_string(source.reason_code)
        row_id = _source_row_id(
            source_table="sender_scores",
            application_id=application.application_id,
            sample_id=str(source.sample_id),
            subject_id=str(source.subject_id),
            context_id=str(source.context_id),
            receiver=str(source.receiver),
            family_id=str(source.family_id),
            driver_id=str(source.driver_id),
            interaction_id=str(source.interaction_id),
            mode=str(source.mode),
            sender=str(source.sender),
        )
        for component in _SENDER_COMPONENTS:
            value = _optional_float(getattr(source, component), field_name=component)
            status, reason = _component_status(
                component=component,
                value=value,
                row_status=row_status,
                row_reason_code=row_reason,
            )
            rows.append(
                _base_component_row(
                    artifacts,
                    projection=projection,
                    fold_id=fold_id,
                    functional=functional,
                    application=application,
                    binding=binding,
                    source_table="sender_scores",
                    source_row_id=row_id,
                    sample_id=str(source.sample_id),
                    subject_id=str(source.subject_id),
                    context_id=str(source.context_id),
                    receiver=str(source.receiver),
                    family_id=str(source.family_id),
                    driver_id=str(source.driver_id),
                    interaction_id=str(source.interaction_id),
                    mode=str(source.mode),
                    sender=str(source.sender),
                    component_scope="sender_lr_member",
                    component=component,
                    component_value=value,
                    status=status,
                    reason_code=reason,
                    row_status=row_status,
                    row_reason_code=row_reason,
                    lr_identifiability_status=None,
                    ligand_contrast_gate_status=None,
                )
            )
    return rows


def _differential_rows(
    artifacts: CrossFitArtifacts,
    *,
    projection: _OOFPersistenceProjection,
    fold_id: str,
    functional: Any,
    application: Any,
    binding: Any,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in application.subject_differential.itertuples(index=False):
        row_id = _source_row_id(
            source_table="subject_differential",
            application_id=application.application_id,
            subject_id=str(source.subject_id),
            receiver=functional.receiver,
            family_id=str(source.family_id),
        )
        rows.append(
            {
                "crossfit_id": artifacts.crossfit_id,
                "spec_id": artifacts.spec.spec_id,
                "repeat_id": artifacts.spec.repeat_id,
                "fold_id": fold_id,
                "contrast_id": functional.contrast_manifest_id,
                "contrast": functional.contrast_name,
                "receiver": functional.receiver,
                "subject_id": str(source.subject_id),
                "family_id": str(source.family_id),
                "differential_effect": _optional_float(
                    source.differential_effect,
                    field_name="differential_effect",
                ),
                "status": str(source.status),
                "reason_code": _optional_string(source.reason_code),
                "effect_semantics": _EFFECT_SEMANTICS,
                "score_version": functional.score_version,
                "family_common_functional_id": (functional.family_common_functional_id),
                "family_common_application_id": application.application_id,
                "family_common_binding_id": binding.binding_id,
                "certification_status": projection.certification_status,
                "is_oof_certified": projection.is_oof_certified,
                "formal_inference_status": _FORMAL_INFERENCE_STATUS,
                "claim_scope": projection.claim_scope,
                "source_row_id": row_id,
            }
        )
    return rows


def _differential_frame(
    artifacts: CrossFitArtifacts,
    *,
    projection: _OOFPersistenceProjection,
    fold_id: str,
    functional: Any,
    application: Any,
    binding: Any,
) -> pd.DataFrame:
    source = application.subject_differential
    if source.empty:
        return pd.DataFrame(columns=CROSSFIT_DIFFERENTIAL_COLUMNS)
    row_count = len(source)
    subject_ids = _required_string_array(source["subject_id"])
    family_ids = _required_string_array(source["family_id"])
    values = _finite_float_matrix(source, ("differential_effect",))[:, 0]
    source_row_ids = np.fromiter(
        (
            _source_row_id(
                source_table="subject_differential",
                application_id=application.application_id,
                subject_id=subject_ids[index],
                receiver=functional.receiver,
                family_id=family_ids[index],
            )
            for index in range(row_count)
        ),
        dtype=object,
        count=row_count,
    )
    frame = pd.DataFrame(
        {
            "crossfit_id": np.full(row_count, artifacts.crossfit_id, dtype=object),
            "spec_id": np.full(row_count, artifacts.spec.spec_id, dtype=object),
            "repeat_id": np.full(row_count, artifacts.spec.repeat_id, dtype=object),
            "fold_id": np.full(row_count, fold_id, dtype=object),
            "contrast_id": np.full(
                row_count,
                functional.contrast_manifest_id,
                dtype=object,
            ),
            "contrast": np.full(
                row_count,
                functional.contrast_name,
                dtype=object,
            ),
            "receiver": np.full(row_count, functional.receiver, dtype=object),
            "subject_id": subject_ids,
            "family_id": family_ids,
            "differential_effect": values,
            "status": _required_string_array(source["status"]),
            "reason_code": _optional_string_array(
                source["reason_code"],
                field_name="reason_code",
            ),
            "effect_semantics": np.full(
                row_count,
                _EFFECT_SEMANTICS,
                dtype=object,
            ),
            "score_version": np.full(
                row_count,
                functional.score_version,
                dtype=object,
            ),
            "family_common_functional_id": np.full(
                row_count,
                functional.family_common_functional_id,
                dtype=object,
            ),
            "family_common_application_id": np.full(
                row_count,
                application.application_id,
                dtype=object,
            ),
            "family_common_binding_id": np.full(
                row_count,
                binding.binding_id,
                dtype=object,
            ),
            "certification_status": np.full(
                row_count,
                projection.certification_status,
                dtype=object,
            ),
            "is_oof_certified": np.full(
                row_count,
                projection.is_oof_certified,
                dtype=bool,
            ),
            "formal_inference_status": np.full(
                row_count,
                _FORMAL_INFERENCE_STATUS,
                dtype=object,
            ),
            "claim_scope": np.full(
                row_count,
                projection.claim_scope,
                dtype=object,
            ),
            "source_row_id": source_row_ids,
        },
        columns=CROSSFIT_DIFFERENTIAL_COLUMNS,
        dtype=object,
    )
    frame["differential_effect"] = values
    frame["is_oof_certified"] = np.full(
        row_count,
        projection.is_oof_certified,
        dtype=bool,
    )
    return frame


@dataclass(frozen=True, slots=True)
class _ContrastCommonSource:
    fold_id: str
    functional: Any
    application: Any
    lr_scores: pd.DataFrame
    sender_scores: pd.DataFrame
    application_record: dict[str, object]


def _contrast_common_application_record(
    *,
    fold_id: str,
    functional: Any,
    application: Any,
    lr_scores: pd.DataFrame,
    sender_scores: pd.DataFrame,
) -> dict[str, object]:
    if tuple(lr_scores.columns) != GLOBAL_COMMON_LR_SCORE_COLUMNS:
        raise ValueError("global common LR source columns are incompatible")
    if tuple(sender_scores.columns) != GLOBAL_COMMON_SENDER_SCORE_COLUMNS:
        raise ValueError("global common sender-LR source columns are incompatible")
    children_by_receiver = {
        child.receiver: child for child in functional.child_functionals
    }
    applications_by_receiver = {
        child.functional.receiver: child for child in application.child_applications
    }
    if set(children_by_receiver) != set(applications_by_receiver):
        raise ValueError("global common receiver child lineage is incomplete")
    receiver_children = [
        {
            "receiver": receiver,
            "family_common_functional_id": (
                children_by_receiver[receiver].family_common_functional_id
            ),
            "family_common_application_id": (
                applications_by_receiver[receiver].application_id
            ),
        }
        for receiver in sorted(children_by_receiver)
    ]
    sender_digests = dict(application.source_sender_application_digests)
    sender_lineage_frame = sender_scores.loc[
        :,
        [
            "receiver",
            "source_sender_functional_id",
            "source_sender_application_id",
        ],
    ].drop_duplicates()
    sender_lineages = [
        {
            "receiver": str(row.receiver),
            "sender_functional_id": str(row.source_sender_functional_id),
            "sender_application_id": str(row.source_sender_application_id),
        }
        for row in sender_lineage_frame.sort_values(
            [
                "receiver",
                "source_sender_functional_id",
                "source_sender_application_id",
            ],
            kind="stable",
        ).itertuples(index=False)
    ]
    if {str(item["receiver"]) for item in sender_lineages} != set(sender_digests):
        raise ValueError("global common sender lineage lacks exact receiver coverage")
    sender_application_digests = [
        {
            "receiver": receiver,
            "sender_application_digest": digest,
        }
        for receiver, digest in sorted(sender_digests.items())
    ]
    functional_payload = functional.to_dict()
    return {
        "fold_id": fold_id,
        "global_common_functional_id": functional.global_common_functional_id,
        "global_common_application_id": application.application_id,
        "functional_spec_id": functional.spec.spec_id,
        "functional_schema_version": functional.spec.schema_version,
        "filter_universe_id": functional.filter_universe_id,
        "context_ids": list(functional.context_ids),
        "receiver_ids": list(functional.receiver_ids),
        "training_subject_ids": list(functional.training_subject_ids),
        "heldout_subject_ids": list(application.heldout_subject_ids),
        "receiver_children": receiver_children,
        "sender_functional_id": functional.sender_functional_id,
        "sender_lineages": sender_lineages,
        "sender_application_digests": sender_application_digests,
        "calibration_policy": functional_payload["calibration_policy"],
        "scale_policy": functional_payload["scale_policy"],
        "sender_policy": functional_payload["sender_policy"],
        "softmin_power": functional.spec.softmin_power,
        "epsilon": functional.spec.epsilon,
        "receiver_gain_calibration_bindings": copy.deepcopy(
            functional_payload["receiver_gain_calibration_bindings"]
        ),
        "all_receivers_gain_calibrated": (functional.all_receivers_gain_calibrated),
        "cross_receiver_percentile_rank_eligible": (
            functional.cross_receiver_percentile_rank_eligible
        ),
        "training_only_receiver_calibration": True,
        "receiver_scale_amplification": False,
        "common_functional_across_receivers": False,
        "receiver_balanced_descriptive_collection": True,
        "interaction_mapping_digest": canonical_digest(
            [list(item) for item in functional.interaction_mapping]
        ),
        "functional_certification_status": functional.certification_status,
        "lr_scores_digest": application.lr_scores_digest,
        "sender_scores_digest": application.sender_scores_digest,
        "n_lr_rows": application.table_row_counts[0],
        "n_sender_rows": application.table_row_counts[1],
    }


def _contrast_common_sources(
    artifacts: CrossFitArtifacts,
) -> list[_ContrastCommonSource]:
    sources: list[_ContrastCommonSource] = []
    for fold in sorted(artifacts.folds, key=lambda item: item.fold_id):
        for functional, application in zip(
            fold.cross_receiver_common_functionals,
            fold.cross_receiver_common_applications,
            strict=True,
        ):
            functional._require_intact()
            application._require_intact()
            lr_scores = application.global_lr_scores
            sender_scores = application.global_sender_lr_scores
            record = _contrast_common_application_record(
                fold_id=fold.fold_id,
                functional=functional,
                application=application,
                lr_scores=lr_scores,
                sender_scores=sender_scores,
            )
            sources.append(
                _ContrastCommonSource(
                    fold_id=fold.fold_id,
                    functional=functional,
                    application=application,
                    lr_scores=lr_scores,
                    sender_scores=sender_scores,
                    application_record=record,
                )
            )
    return sources


def _contrast_common_collections(
    artifacts: CrossFitArtifacts,
    sources: list[_ContrastCommonSource],
    projection: _OOFPersistenceProjection,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    grouped: dict[tuple[str, str, str, str], list[_ContrastCommonSource]] = {}
    for source in sources:
        functional = source.functional
        estimand = str(functional.to_dict()["estimand"])
        key = (
            functional.contrast_manifest_id,
            functional.contrast_name,
            functional.score_version,
            estimand,
        )
        grouped.setdefault(key, []).append(source)
    collections: list[dict[str, object]] = []
    application_to_collection: dict[str, str] = {}
    for (contrast_id, contrast, score_version, estimand), members in sorted(
        grouped.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        fold_records = [
            copy.deepcopy(source.application_record)
            for source in sorted(members, key=lambda item: item.fold_id)
        ]
        fold_ids = [str(record["fold_id"]) for record in fold_records]
        if len(fold_ids) != len(set(fold_ids)):
            raise ValueError("contrast-common collection has duplicate fold scope")
        payload: dict[str, object] = {
            "crossfit_id": artifacts.crossfit_id,
            "spec_id": artifacts.spec.spec_id,
            "repeat_id": artifacts.spec.repeat_id,
            "contrast_id": contrast_id,
            "contrast": contrast,
            "score_version": score_version,
            "estimand": estimand,
            "common_functional_across_receivers": False,
            "receiver_balanced_descriptive_collection": True,
            "formal_inference_allowed": False,
            "certification_status": projection.certification_status,
            "is_oof_certified": projection.is_oof_certified,
            "claim_scope": _expected_contrast_common_claim_scope(
                projection.is_oof_certified
            ),
            "fold_applications": fold_records,
        }
        collection_id = stable_id(
            "contrast_common_oof_score_collection",
            payload,
            schema_version="1",
        )
        collection = {
            "contrast_common_collection_id": collection_id,
            **payload,
        }
        collections.append(collection)
        for record in fold_records:
            application_id = str(record["global_common_application_id"])
            if application_id in application_to_collection:
                raise ValueError(
                    "global common application belongs to multiple collections"
                )
            application_to_collection[application_id] = collection_id
    return collections, application_to_collection


def _contrast_common_persisted_rows(
    artifacts: CrossFitArtifacts,
    *,
    projection: _OOFPersistenceProjection,
    source: _ContrastCommonSource,
    collection_id: str,
    source_table: str,
) -> list[dict[str, object]]:
    columns: tuple[str, ...]
    if source_table == CROSSFIT_CONTRAST_COMMON_LR_TABLE:
        table = source.lr_scores
        columns = GLOBAL_COMMON_LR_SCORE_COLUMNS
    elif source_table == CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE:
        table = source.sender_scores
        columns = GLOBAL_COMMON_SENDER_SCORE_COLUMNS
    else:  # pragma: no cover - private caller invariant
        raise KeyError(source_table)
    rows: list[dict[str, object]] = []
    for values in table.itertuples(index=False, name=None):
        source_values = dict(zip(columns, values, strict=True))
        sender = (
            str(source_values["sender"])
            if source_table == CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE
            else None
        )
        row_id = _source_row_id(
            source_table=source_table,
            application_id=source.application.application_id,
            sample_id=str(source_values["sample_id"]),
            subject_id=str(source_values["subject_id"]),
            context_id=str(source_values["context_id"]),
            receiver=str(source_values["receiver"]),
            family_id=str(source_values["family_id"]),
            driver_id=str(source_values["driver_id"]),
            interaction_id=str(source_values["interaction_id"]),
            mode=str(source_values["mode"]),
            sender=sender,
        )
        rows.append(
            {
                "crossfit_id": artifacts.crossfit_id,
                "spec_id": artifacts.spec.spec_id,
                "repeat_id": artifacts.spec.repeat_id,
                "fold_id": source.fold_id,
                "contrast_id": source.functional.contrast_manifest_id,
                "contrast": source.functional.contrast_name,
                "contrast_common_collection_id": collection_id,
                "global_common_application_id": source.application.application_id,
                **source_values,
                "certification_status": projection.certification_status,
                "is_oof_certified": projection.is_oof_certified,
                "formal_inference_status": _FORMAL_INFERENCE_STATUS,
                "claim_scope": _expected_contrast_common_claim_scope(
                    projection.is_oof_certified
                ),
                "source_row_id": row_id,
            }
        )
    return rows


def _contrast_common_persisted_frame(
    artifacts: CrossFitArtifacts,
    *,
    projection: _OOFPersistenceProjection,
    source: _ContrastCommonSource,
    collection_id: str,
    source_table: str,
) -> pd.DataFrame:
    output_columns: tuple[str, ...]
    if source_table == CROSSFIT_CONTRAST_COMMON_LR_TABLE:
        table = source.lr_scores
        output_columns = CROSSFIT_CONTRAST_COMMON_LR_COLUMNS
        sender = np.full(len(table), None, dtype=object)
    elif source_table == CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE:
        table = source.sender_scores
        output_columns = CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS
        sender = _required_string_array(table["sender"])
    else:  # pragma: no cover - private caller invariant
        raise KeyError(source_table)
    if table.empty:
        return pd.DataFrame(columns=output_columns)

    row_count = len(table)
    sample_ids = _required_string_array(table["sample_id"])
    subject_ids = _required_string_array(table["subject_id"])
    context_ids = _required_string_array(table["context_id"])
    receivers = _required_string_array(table["receiver"])
    family_ids = _required_string_array(table["family_id"])
    driver_ids = _required_string_array(table["driver_id"])
    interaction_ids = _required_string_array(table["interaction_id"])
    modes = _required_string_array(table["mode"])
    source_row_ids = _source_row_ids(
        source_table=source_table,
        application_ids=np.full(
            row_count, source.application.application_id, dtype=object
        ),
        sample_ids=sample_ids,
        subject_ids=subject_ids,
        context_ids=context_ids,
        receivers=receivers,
        family_ids=family_ids,
        driver_ids=driver_ids,
        interaction_ids=interaction_ids,
        modes=modes,
        senders=sender,
    )
    prefix = pd.DataFrame(
        {
            "crossfit_id": np.full(row_count, artifacts.crossfit_id, dtype=object),
            "spec_id": np.full(row_count, artifacts.spec.spec_id, dtype=object),
            "repeat_id": np.full(row_count, artifacts.spec.repeat_id, dtype=object),
            "fold_id": np.full(row_count, source.fold_id, dtype=object),
            "contrast_id": np.full(
                row_count,
                source.functional.contrast_manifest_id,
                dtype=object,
            ),
            "contrast": np.full(
                row_count,
                source.functional.contrast_name,
                dtype=object,
            ),
            "contrast_common_collection_id": np.full(
                row_count,
                collection_id,
                dtype=object,
            ),
            "global_common_application_id": np.full(
                row_count,
                source.application.application_id,
                dtype=object,
            ),
        },
        dtype=object,
    )
    suffix = pd.DataFrame(
        {
            "certification_status": np.full(
                row_count,
                projection.certification_status,
                dtype=object,
            ),
            "is_oof_certified": np.full(
                row_count,
                projection.is_oof_certified,
                dtype=bool,
            ),
            "formal_inference_status": np.full(
                row_count,
                _FORMAL_INFERENCE_STATUS,
                dtype=object,
            ),
            "claim_scope": np.full(
                row_count,
                _expected_contrast_common_claim_scope(projection.is_oof_certified),
                dtype=object,
            ),
            "source_row_id": source_row_ids,
        },
        dtype=object,
    )
    suffix["is_oof_certified"] = np.full(
        row_count,
        projection.is_oof_certified,
        dtype=bool,
    )
    source_frame = table.reset_index(drop=True).copy(deep=False)
    result = pd.concat(
        [prefix, source_frame, suffix],
        axis="columns",
    )
    return result.loc[:, output_columns]


_DIRECTIONAL_BINDING_FIELDS = frozenset(
    {
        "binding_id",
        "combination_rule",
        "active_inhibition_allowed",
        "feature_ids",
        "fold_id",
        "formal_inference_allowed",
        "forward_application_id",
        "forward_channel_name",
        "forward_response_id",
        "forward_training_artifact_id",
        "heldout_subject_ids",
        "paired_score_comparison_allowed",
        "pair_spec_id",
        "reason_code",
        "receiver",
        "response_pair_id",
        "reverse_application_id",
        "reverse_channel_name",
        "reverse_response_id",
        "reverse_training_artifact_id",
        "status",
        "supports_active_inhibition_claim",
        "training_subject_ids",
        "pair_spec",
        "response_pair",
    }
)
_DIRECTIONAL_PAIR_SPEC_FIELDS = frozenset(
    {
        "pair_spec_id",
        "schema_version",
        "forward_contrast_id",
        "reverse_contrast_id",
        "forward_contrast",
        "reverse_contrast",
        "forward_channel_name",
        "reverse_channel_name",
        "supports_active_inhibition_claim",
        "combination_rule",
    }
)
_DIRECTIONAL_CONTRAST_FIELDS = frozenset(
    {"name", "family", "mode", "estimable", "reason_code", "weights"}
)
_DIRECTIONAL_RESPONSE_PAIR_FIELDS = frozenset(
    {
        "response_pair_id",
        "forward_channel_id",
        "reverse_channel_id",
        "forward_contrast_id",
        "reverse_contrast_id",
        "forward_response_id",
        "reverse_response_id",
        "feature_ids",
        "prior_semantics",
        "forward_biological_claim",
        "reverse_biological_claim",
        "supports_active_inhibition_claim",
        "combination_rule",
    }
)


def _directional_mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return dict(value)


def _directional_identifier_array(value: object, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty identifier array")
    result = [_manifest_string(item, field_name=field_name) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must contain unique identifiers")
    return result


def _directional_contrast_weights(
    contrast: dict[str, Any],
    *,
    field_name: str,
) -> dict[str, float]:
    raw_weights = contrast.get("weights")
    if not isinstance(raw_weights, list) or not raw_weights:
        raise ValueError(f"{field_name}.weights must be a non-empty array")
    weights: dict[str, float] = {}
    for item in raw_weights:
        record = _directional_mapping(item, field_name=f"{field_name}.weights")
        if set(record) != {"context", "weight"}:
            raise ValueError(f"{field_name}.weights fields are invalid")
        context_key = canonical_json(record["context"])
        weight = float(record["weight"])
        if context_key in weights or not math.isfinite(weight) or weight == 0.0:
            raise ValueError(f"{field_name}.weights are invalid")
        weights[context_key] = weight
    if not math.isclose(sum(weights.values()), 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{field_name}.weights are not balanced")
    return weights


def _directional_pair_spec_projection(
    value: object,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pair_spec = _directional_mapping(value, field_name="pair_spec")
    if set(pair_spec) != _DIRECTIONAL_PAIR_SPEC_FIELDS:
        raise ValueError("directional pair specification fields are invalid")
    if pair_spec["schema_version"] != "1.0.0":
        raise ValueError("directional pair specification schema is invalid")
    if (
        pair_spec["forward_channel_name"] != _DIRECTIONAL_CHANNELS["forward"]
        or pair_spec["reverse_channel_name"] != _DIRECTIONAL_CHANNELS["reverse"]
        or pair_spec["supports_active_inhibition_claim"] is not False
        or pair_spec["combination_rule"] != _DIRECTIONAL_COMBINATION_RULE
    ):
        raise ValueError("directional pair specification semantics are invalid")
    contrasts: dict[str, dict[str, Any]] = {}
    weights: dict[str, dict[str, float]] = {}
    for role in ("forward", "reverse"):
        contrast = _directional_mapping(
            pair_spec[f"{role}_contrast"],
            field_name=f"pair_spec.{role}_contrast",
        )
        if set(contrast) != _DIRECTIONAL_CONTRAST_FIELDS:
            raise ValueError("directional contrast fields are invalid")
        for field_name in ("name", "family", "mode"):
            _manifest_string(
                contrast[field_name],
                field_name=f"pair_spec.{role}_contrast.{field_name}",
            )
        if contrast["estimable"] is not True or contrast["reason_code"] is not None:
            raise ValueError("directional contrasts must be estimable")
        contrast_id = _manifest_string(
            pair_spec[f"{role}_contrast_id"],
            field_name=f"pair_spec.{role}_contrast_id",
        )
        if contrast_id != stable_id("contrast", contrast):
            raise ValueError("directional contrast identity is invalid")
        contrasts[role] = contrast
        weights[role] = _directional_contrast_weights(
            contrast,
            field_name=f"pair_spec.{role}_contrast",
        )
    if (
        contrasts["forward"]["name"] == contrasts["reverse"]["name"]
        or contrasts["forward"]["family"] != contrasts["reverse"]["family"]
        or contrasts["forward"]["mode"] != contrasts["reverse"]["mode"]
        or set(weights["forward"]) != set(weights["reverse"])
        or any(
            weights["reverse"][context] != -weight
            for context, weight in weights["forward"].items()
        )
    ):
        raise ValueError("directional contrasts are not exact reverses")
    pair_spec_id = _manifest_string(
        pair_spec["pair_spec_id"], field_name="pair_spec.pair_spec_id"
    )
    expected_pair_spec_id = stable_id(
        "directional_contrast_pair_spec",
        {
            "combination_rule": _DIRECTIONAL_COMBINATION_RULE,
            "forward_channel_name": _DIRECTIONAL_CHANNELS["forward"],
            "forward_contrast_id": pair_spec["forward_contrast_id"],
            "reverse_channel_name": _DIRECTIONAL_CHANNELS["reverse"],
            "reverse_contrast_id": pair_spec["reverse_contrast_id"],
            "schema_version": "1.0.0",
            "supports_active_inhibition_claim": False,
        },
        schema_version="1",
    )
    if pair_spec_id != expected_pair_spec_id:
        raise ValueError("directional pair specification identity is invalid")
    return pair_spec, contrasts


def _directional_binding_rows(
    value: object,
    *,
    crossfit_id: str,
    spec_id: str,
    repeat_id: str,
    fold_id: str,
    certification_status: str,
    is_oof_certified: bool,
    claim_scope: str,
) -> list[dict[str, object]]:
    binding = _directional_mapping(value, field_name="directional binding")
    if set(binding) != _DIRECTIONAL_BINDING_FIELDS:
        raise ValueError("directional binding fields are invalid")
    pair_spec, contrasts = _directional_pair_spec_projection(binding["pair_spec"])
    string_fields = (
        "binding_id",
        "fold_id",
        "forward_application_id",
        "forward_channel_name",
        "forward_response_id",
        "forward_training_artifact_id",
        "pair_spec_id",
        "receiver",
        "reverse_application_id",
        "reverse_channel_name",
        "reverse_response_id",
        "reverse_training_artifact_id",
        "status",
    )
    for field_name in string_fields:
        _manifest_string(binding[field_name], field_name=f"binding.{field_name}")
    if binding["fold_id"] != fold_id:
        raise ValueError("directional binding fold lineage is invalid")
    if (
        binding["pair_spec_id"] != pair_spec["pair_spec_id"]
        or binding["forward_channel_name"] != _DIRECTIONAL_CHANNELS["forward"]
        or binding["reverse_channel_name"] != _DIRECTIONAL_CHANNELS["reverse"]
        or binding["combination_rule"] != _DIRECTIONAL_COMBINATION_RULE
        or binding["active_inhibition_allowed"] is not False
        or binding["supports_active_inhibition_claim"] is not False
        or binding["paired_score_comparison_allowed"] is not False
        or binding["formal_inference_allowed"] is not False
    ):
        raise ValueError("directional binding semantics are invalid")
    status = str(binding["status"])
    if status not in _DIRECTIONAL_CHANNEL_STATUSES:
        raise ValueError("directional binding status is invalid")
    reason_code = _manifest_optional_string(
        binding["reason_code"], field_name="binding.reason_code"
    )
    response_pair_id = _manifest_optional_string(
        binding["response_pair_id"], field_name="binding.response_pair_id"
    )
    if (status == "observed") != (reason_code is None and response_pair_id is not None):
        raise ValueError("directional binding status/reason semantics are invalid")
    feature_ids = _directional_identifier_array(
        binding["feature_ids"], field_name="binding.feature_ids"
    )
    training_subject_ids = _manifest_string_array(
        binding["training_subject_ids"], field_name="binding.training_subject_ids"
    )
    heldout_subject_ids = _manifest_string_array(
        binding["heldout_subject_ids"], field_name="binding.heldout_subject_ids"
    )
    if set(training_subject_ids).intersection(heldout_subject_ids):
        raise ValueError("directional binding train/heldout subjects overlap")
    response_pair: dict[str, Any] | None
    if status == "observed":
        response_pair = _directional_mapping(
            binding["response_pair"], field_name="binding.response_pair"
        )
        if set(response_pair) != _DIRECTIONAL_RESPONSE_PAIR_FIELDS:
            raise ValueError("directional response pair fields are invalid")
        for field_name in (
            "response_pair_id",
            "forward_channel_id",
            "reverse_channel_id",
            "forward_contrast_id",
            "reverse_contrast_id",
            "forward_response_id",
            "reverse_response_id",
        ):
            _manifest_string(
                response_pair[field_name],
                field_name=f"binding.response_pair.{field_name}",
            )
        if (
            response_pair["response_pair_id"] != response_pair_id
            or response_pair["forward_contrast_id"] != pair_spec["forward_contrast_id"]
            or response_pair["reverse_contrast_id"] != pair_spec["reverse_contrast_id"]
            or response_pair["forward_response_id"] != binding["forward_response_id"]
            or response_pair["reverse_response_id"] != binding["reverse_response_id"]
            or response_pair["feature_ids"] != feature_ids
            or response_pair["prior_semantics"] != "nonnegative_activation_prior_v1"
            or response_pair["forward_biological_claim"]
            != "direction_compatible_activation"
            or response_pair["reverse_biological_claim"]
            != "attenuation_of_activation_from_explicit_reverse_contrast"
            or response_pair["supports_active_inhibition_claim"] is not False
            or response_pair["combination_rule"] != _DIRECTIONAL_COMBINATION_RULE
        ):
            raise ValueError("directional response pair lineage is invalid")
    else:
        if binding["response_pair"] is not None or response_pair_id is not None:
            raise ValueError("not-estimable directional binding has a response pair")
        response_pair = None
    identity_payload = {
        key: binding[key]
        for key in _DIRECTIONAL_BINDING_FIELDS.difference(
            {"binding_id", "pair_spec", "response_pair"}
        )
    }
    if binding["binding_id"] != stable_id(
        "directional_crossfit_binding", identity_payload, schema_version="1"
    ):
        raise ValueError("directional binding identity is invalid")
    rows: list[dict[str, object]] = []
    for role in ("forward", "reverse"):
        rows.append(
            {
                "crossfit_id": crossfit_id,
                "spec_id": spec_id,
                "repeat_id": repeat_id,
                "fold_id": fold_id,
                "receiver": binding["receiver"],
                "pair_spec_id": binding["pair_spec_id"],
                "binding_id": binding["binding_id"],
                "channel_role": role,
                "channel": binding[f"{role}_channel_name"],
                "contrast_id": pair_spec[f"{role}_contrast_id"],
                "contrast": contrasts[role]["name"],
                "response_id": binding[f"{role}_response_id"],
                "training_artifact_id": binding[f"{role}_training_artifact_id"],
                "application_id": binding[f"{role}_application_id"],
                "response_pair_id": response_pair_id,
                "response_channel_id": (
                    None
                    if response_pair is None
                    else response_pair[f"{role}_channel_id"]
                ),
                "status": status,
                "reason_code": reason_code,
                "combination_rule": _DIRECTIONAL_COMBINATION_RULE,
                "active_inhibition_allowed": False,
                "supports_active_inhibition_claim": False,
                "paired_score_comparison_allowed": False,
                "formal_inference_allowed": False,
                "certification_status": certification_status,
                "is_oof_certified": is_oof_certified,
                "formal_inference_status": _FORMAL_INFERENCE_STATUS,
                "claim_scope": claim_scope,
            }
        )
    return rows


def _result_tables(
    artifacts: CrossFitArtifacts,
) -> tuple[
    dict[str, pd.DataFrame],
    list[dict[str, object]],
    list[dict[str, object]],
    _OOFPersistenceProjection,
]:
    projection = _oof_persistence_projection(artifacts)
    component_frames: list[pd.DataFrame] = []
    differential_frames: list[pd.DataFrame] = []
    applications: list[dict[str, object]] = []
    directional_rows: list[dict[str, object]] = []
    receiver_support_rows: list[dict[str, object]] = []
    for fold in sorted(artifacts.folds, key=lambda item: item.fold_id):
        for support in fold.receiver_training_support:
            support._require_intact()
            receiver_support_rows.append(
                {
                    "crossfit_id": artifacts.crossfit_id,
                    "spec_id": artifacts.spec.spec_id,
                    "repeat_id": artifacts.spec.repeat_id,
                    "receiver_universe_id": (artifacts.receiver_universe.universe_id),
                    "receiver_axis_id": artifacts.receiver_universe.receiver_axis_id,
                    "fold_id": fold.fold_id,
                    "receiver": support.receiver_id,
                    "receiver_training_support_id": support.support_record_id,
                    "receiver_training_support_status": support.status.value,
                    "receiver_training_support_reason_code": support.reason_code,
                    "training_cell_type_ids": canonical_json(
                        list(support.training_cell_type_ids)
                    ),
                }
            )
        for directional_binding in sorted(
            fold.directional_response_bindings,
            key=lambda item: (item.pair_spec_id, item.receiver),
        ):
            directional_binding._require_intact()
            directional_rows.extend(
                _directional_binding_rows(
                    directional_binding.to_dict(),
                    crossfit_id=artifacts.crossfit_id,
                    spec_id=artifacts.spec.spec_id,
                    repeat_id=artifacts.spec.repeat_id,
                    fold_id=fold.fold_id,
                    certification_status=projection.certification_status,
                    is_oof_certified=projection.is_oof_certified,
                    claim_scope=projection.claim_scope,
                )
            )
        for functional, application, family_binding in zip(
            fold.family_common_functionals,
            fold.family_common_applications,
            fold.family_common_bindings,
            strict=True,
        ):
            functional._require_intact()
            application._require_intact()
            family_binding._require_intact()
            family_rows = _component_frame(
                artifacts,
                projection=projection,
                fold_id=fold.fold_id,
                functional=functional,
                application=application,
                binding=family_binding,
                source_table="family_scores",
                source=application.family_scores,
                component_scope="family",
                components=_FAMILY_COMPONENTS,
            )
            member_rows = _component_frame(
                artifacts,
                projection=projection,
                fold_id=fold.fold_id,
                functional=functional,
                application=application,
                binding=family_binding,
                source_table="member_scores",
                source=application.member_scores,
                component_scope="lr_member",
                components=_MEMBER_COMPONENTS,
            )
            sender_rows = _component_frame(
                artifacts,
                projection=projection,
                fold_id=fold.fold_id,
                functional=functional,
                application=application,
                binding=family_binding,
                source_table="sender_scores",
                source=application.sender_scores,
                component_scope="sender_lr_member",
                components=_SENDER_COMPONENTS,
            )
            effects = _differential_frame(
                artifacts,
                projection=projection,
                fold_id=fold.fold_id,
                functional=functional,
                application=application,
                binding=family_binding,
            )
            component_frames.extend((family_rows, member_rows, sender_rows))
            differential_frames.append(effects)
            applications.append(
                {
                    "fold_id": fold.fold_id,
                    "contrast_id": functional.contrast_manifest_id,
                    "contrast": functional.contrast_name,
                    "receiver": functional.receiver,
                    "family_common_functional_id": (
                        functional.family_common_functional_id
                    ),
                    "family_common_application_id": application.application_id,
                    "family_common_binding_id": family_binding.binding_id,
                    "sender_functional_id": (
                        functional.sender_functional.sender_functional_id
                    ),
                    "score_version": functional.score_version,
                    "certification_status": projection.certification_status,
                    "is_oof_certified": projection.is_oof_certified,
                    "source_table_digests": {
                        "family_attribution": application.family_attribution_digest,
                        "subject_differential": application.subject_differential_digest,
                        "family_scores": application.family_scores_digest,
                        "member_scores": application.member_scores_digest,
                        "sender_scores": application.sender_scores_digest,
                    },
                    "source_table_row_counts": {
                        "family_attribution": application.table_row_counts[0],
                        "subject_differential": application.table_row_counts[1],
                        "family_scores": application.table_row_counts[2],
                        "member_scores": application.table_row_counts[3],
                        "sender_scores": application.table_row_counts[4],
                    },
                    "persisted_component_rows": (
                        len(family_rows) + len(member_rows) + len(sender_rows)
                    ),
                    "persisted_differential_rows": len(effects),
                }
            )
    common_sources = _contrast_common_sources(artifacts)
    common_collections, application_to_collection = _contrast_common_collections(
        artifacts,
        common_sources,
        projection,
    )
    common_lr_frames: list[pd.DataFrame] = []
    common_sender_frames: list[pd.DataFrame] = []
    for source in common_sources:
        collection_id = application_to_collection[source.application.application_id]
        common_lr_frames.append(
            _contrast_common_persisted_frame(
                artifacts,
                projection=projection,
                source=source,
                collection_id=collection_id,
                source_table=CROSSFIT_CONTRAST_COMMON_LR_TABLE,
            )
        )
        common_sender_frames.append(
            _contrast_common_persisted_frame(
                artifacts,
                projection=projection,
                source=source,
                collection_id=collection_id,
                source_table=CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
            )
        )
    components = (
        pd.concat(component_frames, ignore_index=True)
        if component_frames
        else pd.DataFrame(columns=CROSSFIT_COMPONENT_COLUMNS)
    )
    differential = (
        pd.concat(differential_frames, ignore_index=True)
        if differential_frames
        else pd.DataFrame(columns=CROSSFIT_DIFFERENTIAL_COLUMNS)
    )
    common_lr = (
        pd.concat(common_lr_frames, ignore_index=True)
        if common_lr_frames
        else pd.DataFrame(columns=CROSSFIT_CONTRAST_COMMON_LR_COLUMNS)
    )
    common_sender = (
        pd.concat(common_sender_frames, ignore_index=True)
        if common_sender_frames
        else pd.DataFrame(columns=CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS)
    )
    directional_registry = pd.DataFrame(
        directional_rows,
        columns=CROSSFIT_DIRECTIONAL_CHANNEL_COLUMNS,
    )
    receiver_training_support = pd.DataFrame(
        receiver_support_rows,
        columns=CROSSFIT_RECEIVER_TRAINING_SUPPORT_COLUMNS,
    )
    if not components.empty:
        components = components.sort_values(
            [
                "fold_id",
                "contrast",
                "receiver",
                "sample_id",
                "family_id",
                "component_scope",
                "interaction_id",
                "sender",
                "component",
            ],
            kind="stable",
            na_position="first",
            ignore_index=True,
        )
    if not differential.empty:
        differential = differential.sort_values(
            ["fold_id", "contrast", "receiver", "subject_id", "family_id"],
            kind="stable",
            ignore_index=True,
        )
    if not common_lr.empty:
        common_lr = common_lr.sort_values(
            [
                "contrast",
                "fold_id",
                "sample_id",
                "receiver",
                "interaction_id",
                "mode",
            ],
            kind="stable",
            ignore_index=True,
        )
    if not common_sender.empty:
        common_sender = common_sender.sort_values(
            [
                "contrast",
                "fold_id",
                "sample_id",
                "sender",
                "receiver",
                "interaction_id",
                "mode",
            ],
            kind="stable",
            ignore_index=True,
        )
    if not directional_registry.empty:
        directional_registry = directional_registry.sort_values(
            ["pair_spec_id", "fold_id", "receiver", "channel_role"],
            kind="stable",
            ignore_index=True,
        )
    receiver_training_support = receiver_training_support.sort_values(
        ["fold_id", "receiver"],
        kind="stable",
        ignore_index=True,
    )
    tables = {
        CROSSFIT_COMPONENT_TABLE: components,
        CROSSFIT_DIFFERENTIAL_TABLE: differential,
        CROSSFIT_CONTRAST_COMMON_LR_TABLE: common_lr,
        CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE: common_sender,
        CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE: directional_registry,
        CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE: receiver_training_support,
    }
    return tables, applications, common_collections, projection


def _require_exact_columns(
    table: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    table_name: str,
) -> None:
    if tuple(table.columns) != columns:
        raise ValueError(f"{table_name} columns do not match the released schema")
    normalized = {str(column).lower() for column in table.columns}
    forbidden = normalized.intersection(_FORBIDDEN_INFERENCE_FIELDS)
    if forbidden:
        raise ValueError(
            f"{table_name} contains forbidden inferential fields: {sorted(forbidden)}"
        )


def _require_identifiers(
    table: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    table_name: str,
) -> None:
    for column in columns:
        values = table[column]
        if values.isna().any():
            raise ValueError(f"{table_name}.{column} requires non-empty identifiers")
        standard = _standard_required_string_array(values)
        if standard is None:
            has_empty = any(not str(value) for value in values)
        else:
            has_empty = bool(np.equal(standard, "").any())
        if has_empty:
            raise ValueError(f"{table_name}.{column} requires non-empty identifiers")


def _standard_optional_string_presence(values: pd.Series) -> np.ndarray | None:
    if isinstance(values.dtype, pd.StringDtype):
        missing = values.isna().to_numpy(dtype=bool, copy=False)
        empty = values.eq("").fillna(False).to_numpy(dtype=bool, copy=False)
    elif isinstance(values.dtype, pd.CategoricalDtype):
        if not all(type(value) is str for value in values.cat.categories):
            return None
        missing = values.isna().to_numpy(dtype=bool, copy=False)
        empty = values.eq("").fillna(False).to_numpy(dtype=bool, copy=False)
    elif values.dtype == np.dtype("object"):
        raw = values.to_numpy(dtype="object", copy=False)
        if not all(
            value is None
            or value is pd.NA
            or value is pd.NaT
            or type(value) is str
            or type(value) in {float, np.float64}
            for value in raw
        ):
            return None
        missing = values.isna().to_numpy(dtype=bool, copy=False)
        empty = values.eq("").fillna(False).to_numpy(dtype=bool, copy=False)
    else:
        return None
    if bool(empty.any()):
        raise ValueError("persisted optional identifiers must be non-empty")
    return ~missing


def _validate_component_rows_legacy(result: pd.DataFrame) -> None:
    for row in result.itertuples(index=False):
        scope = str(row.component_scope)
        component = str(row.component)
        if scope not in _SCOPE_COMPONENTS or component not in _SCOPE_COMPONENTS[scope]:
            raise ValueError("component does not belong to its declared scope")
        if scope == "family" and any(
            _optional_string(value) is not None
            for value in (row.driver_id, row.interaction_id, row.sender)
        ):
            raise ValueError("family components cannot claim LR or sender identity")
        if scope == "lr_member" and (
            _optional_string(row.driver_id) is None
            or _optional_string(row.interaction_id) is None
            or _optional_string(row.sender) is not None
        ):
            raise ValueError("LR-member components require LR but no sender identity")
        if scope == "sender_lr_member" and any(
            _optional_string(value) is None
            for value in (row.driver_id, row.interaction_id, row.sender)
        ):
            raise ValueError(
                "sender LR-member components require LR and sender identity"
            )
        value = _optional_float(row.component_value, field_name="component_value")
        status = str(row.status)
        reason = _optional_string(row.reason_code)
        if status not in _COMPONENT_STATUSES:
            raise ValueError("component status is not recognized")
        if status == "not_estimable" and (value is not None or reason is None):
            raise ValueError("not-estimable components require NA value and reason")
        if status == "structural_zero" and (value != 0.0 or reason is None):
            raise ValueError("structural-zero components require zero value and reason")
        if status == "observed" and (value is None or reason is not None):
            raise ValueError("observed components require a value and no reason")
        if type(row.is_oof_certified) is not bool:
            raise ValueError("component OOF certification flag must be boolean")
        is_oof_certified = bool(row.is_oof_certified)
        if str(row.formal_inference_status) != _FORMAL_INFERENCE_STATUS:
            raise ValueError("component rows cannot claim formal inference")
        if str(row.claim_scope) != _expected_claim_scope(is_oof_certified):
            raise ValueError("component claim scope is invalid")


def _validate_component_table(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy(deep=True)
    _require_exact_columns(
        result,
        CROSSFIT_COMPONENT_COLUMNS,
        table_name=CROSSFIT_COMPONENT_TABLE,
    )
    _require_identifiers(
        result,
        (
            "crossfit_id",
            "spec_id",
            "repeat_id",
            "fold_id",
            "contrast_id",
            "contrast",
            "sample_id",
            "subject_id",
            "context_id",
            "receiver",
            "family_id",
            "mode",
            "component_scope",
            "component",
            "status",
            "row_status",
            "score_version",
            "family_common_functional_id",
            "family_common_application_id",
            "family_common_binding_id",
            "sender_functional_id",
            "certification_status",
            "formal_inference_status",
            "claim_scope",
            "source_table",
            "source_row_id",
        ),
        table_name=CROSSFIT_COMPONENT_TABLE,
    )
    presence = {
        column: _standard_optional_string_presence(result[column])
        for column in ("driver_id", "interaction_id", "sender", "reason_code")
    }
    value_dtype = result["component_value"].dtype
    fast_values = (
        pd.api.types.is_numeric_dtype(value_dtype)
        and not pd.api.types.is_complex_dtype(value_dtype)
        and not pd.api.types.is_datetime64_any_dtype(value_dtype)
        and not pd.api.types.is_timedelta64_dtype(value_dtype)
    )
    fast_flags = result["is_oof_certified"].dtype == np.dtype(bool)
    if any(value is None for value in presence.values()) or not (
        fast_values and fast_flags
    ):
        _validate_component_rows_legacy(result)
    else:
        scopes = pd.Series(
            _required_string_array(result["component_scope"]), index=result.index
        )
        components = pd.Series(
            _required_string_array(result["component"]), index=result.index
        )
        valid_scope_component: np.ndarray = np.zeros(len(result), dtype=bool)
        for scope, allowed in _SCOPE_COMPONENTS.items():
            valid_scope_component |= scopes.eq(scope).to_numpy(
                dtype=bool, copy=False
            ) & components.isin(allowed).to_numpy(dtype=bool, copy=False)
        if not bool(valid_scope_component.all()):
            raise ValueError("component does not belong to its declared scope")

        driver_present = cast(np.ndarray, presence["driver_id"])
        interaction_present = cast(np.ndarray, presence["interaction_id"])
        sender_present = cast(np.ndarray, presence["sender"])
        family = scopes.eq("family").to_numpy(dtype=bool, copy=False)
        lr_member = scopes.eq("lr_member").to_numpy(dtype=bool, copy=False)
        sender_member = scopes.eq("sender_lr_member").to_numpy(
            dtype=bool, copy=False
        )
        if bool(
            np.any(family & (driver_present | interaction_present | sender_present))
        ):
            raise ValueError("family components cannot claim LR or sender identity")
        if bool(
            np.any(
                lr_member
                & (~driver_present | ~interaction_present | sender_present)
            )
        ):
            raise ValueError("LR-member components require LR but no sender identity")
        if bool(
            np.any(
                sender_member
                & (~driver_present | ~interaction_present | ~sender_present)
            )
        ):
            raise ValueError(
                "sender LR-member components require LR and sender identity"
            )

        raw_values = result["component_value"]
        missing_values = raw_values.isna().to_numpy(dtype=bool, copy=False)
        values = raw_values.to_numpy(dtype=float, na_value=np.nan, copy=False)
        if not bool(np.isfinite(values[~missing_values]).all()):
            raise ValueError("component_value must be finite or missing")
        statuses = pd.Series(
            _required_string_array(result["status"]), index=result.index
        )
        if not bool(statuses.isin(_COMPONENT_STATUSES).all()):
            raise ValueError("component status is not recognized")
        reason_present = cast(np.ndarray, presence["reason_code"])
        observed = statuses.eq("observed").to_numpy(dtype=bool, copy=False)
        not_estimable = statuses.eq("not_estimable").to_numpy(
            dtype=bool, copy=False
        )
        structural_zero = statuses.eq("structural_zero").to_numpy(
            dtype=bool, copy=False
        )
        if bool(np.any(not_estimable & (~missing_values | ~reason_present))):
            raise ValueError("not-estimable components require NA value and reason")
        if bool(
            np.any(structural_zero & ((values != 0.0) | ~reason_present))
        ):
            raise ValueError("structural-zero components require zero value and reason")
        if bool(np.any(observed & (missing_values | reason_present))):
            raise ValueError("observed components require a value and no reason")

        flags = result["is_oof_certified"].to_numpy(dtype=bool, copy=False)
        if not bool(
            np.equal(
                _required_string_array(result["formal_inference_status"]),
                _FORMAL_INFERENCE_STATUS,
            ).all()
        ):
            raise ValueError("component rows cannot claim formal inference")
        expected_claims = np.where(
            flags,
            _expected_claim_scope(True),
            _expected_claim_scope(False),
        )
        if not bool(
            np.equal(
                _required_string_array(result["claim_scope"]), expected_claims
            ).all()
        ):
            raise ValueError("component claim scope is invalid")
    if result.duplicated(["source_row_id", "component"]).any():
        raise ValueError("component table contains duplicate source-row components")
    return result


def _validate_differential_table(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy(deep=True)
    _require_exact_columns(
        result,
        CROSSFIT_DIFFERENTIAL_COLUMNS,
        table_name=CROSSFIT_DIFFERENTIAL_TABLE,
    )
    _require_identifiers(
        result,
        tuple(
            column
            for column in CROSSFIT_DIFFERENTIAL_COLUMNS
            if column
            not in {
                "differential_effect",
                "reason_code",
                "is_oof_certified",
            }
        ),
        table_name=CROSSFIT_DIFFERENTIAL_TABLE,
    )
    for row in result.itertuples(index=False):
        effect = _optional_float(
            row.differential_effect,
            field_name="differential_effect",
        )
        status = str(row.status)
        reason = _optional_string(row.reason_code)
        if status not in _DIFFERENTIAL_STATUSES:
            raise ValueError("differential status is not recognized")
        if status == "observed" and (effect is None or reason is not None):
            raise ValueError("observed descriptive effects require value and no reason")
        if status == "not_estimable" and (effect is not None or reason is None):
            raise ValueError("not-estimable descriptive effects require NA and reason")
        if status == "structural_zero" and (effect != 0.0 or reason is None):
            raise ValueError(
                "structural-zero descriptive effects require zero and reason"
            )
        if row.effect_semantics != _EFFECT_SEMANTICS:
            raise ValueError("descriptive effect semantics are invalid")
        if type(row.is_oof_certified) is not bool:
            raise ValueError("differential OOF certification flag must be boolean")
        is_oof_certified = bool(row.is_oof_certified)
        if row.formal_inference_status != _FORMAL_INFERENCE_STATUS:
            raise ValueError(
                "descriptive differential rows cannot claim formal inference"
            )
        if row.claim_scope != _expected_claim_scope(is_oof_certified):
            raise ValueError("descriptive differential claim scope is invalid")
    keys = [
        "crossfit_id",
        "fold_id",
        "contrast_id",
        "receiver",
        "subject_id",
        "family_id",
    ]
    if result.duplicated(keys).any():
        raise ValueError("descriptive differential keys must be unique")
    return result


def _optional_unit_interval(value: object, *, field_name: str) -> float | None:
    result = _optional_float(value, field_name=field_name)
    if result is not None and not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must lie in [0, 1] or be missing")
    return result


def _validate_v3_contrast_common_table(
    table: pd.DataFrame,
    *,
    table_name: str,
    columns: tuple[str, ...],
    score_column: str,
    sender_grain: bool,
) -> pd.DataFrame:
    result = table.copy(deep=True)
    _require_exact_columns(result, columns, table_name=table_name)
    required = {
        "crossfit_id",
        "spec_id",
        "repeat_id",
        "fold_id",
        "contrast_id",
        "contrast",
        "contrast_common_collection_id",
        "global_common_application_id",
        "global_common_functional_id",
        "source_family_common_functional_id",
        "source_family_common_application_id",
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "family_id",
        "driver_id",
        "interaction_id",
        "mode",
        "ligand_contrast_gate_status" if not sender_grain else "sender",
        "status",
        "score_version",
        "certification_status",
        "formal_inference_status",
        "claim_scope",
        "source_row_id",
    }
    if sender_grain:
        required.update(
            {
                "source_sender_functional_id",
                "source_sender_application_id",
            }
        )
    _require_identifiers(result, tuple(sorted(required)), table_name=table_name)
    unit_fields = (
        (
            "availability",
            "receiver_relative_family_gain",
            "prior_quality",
            "training_receiver_scale_factor",
            "global_lr_core_strength",
            "global_lr_score",
        )
        if not sender_grain
        else (
            "global_lr_score",
            "training_receiver_scale_factor",
            "ligand_availability",
            "training_prevalence_prior",
            "raw_sender_evidence",
            "global_sender_lr_score",
        )
    )
    for row in result.itertuples(index=False):
        if str(row.mode) not in {"state", "ecosystem"}:
            raise ValueError("contrast-common score mode is not recognized")
        if not sender_grain and type(row.receptor_eligible) is not bool:
            raise ValueError("contrast-common receptor eligibility must be boolean")
        unit_values = {
            field_name: _optional_unit_interval(
                getattr(row, field_name), field_name=field_name
            )
            for field_name in unit_fields
        }
        score = unit_values[score_column]
        receiver_scale_factor = unit_values["training_receiver_scale_factor"]
        if receiver_scale_factor is None or receiver_scale_factor <= 0.0:
            raise ValueError("training_receiver_scale_factor must lie in (0, 1]")
        status = str(row.status)
        reason = _optional_string(row.reason_code)
        if status not in _CONTRAST_COMMON_STATUSES:
            raise ValueError("contrast-common score status is not recognized")
        if status == "observed" and (score is None or reason is not None):
            raise ValueError(
                "observed contrast-common scores require a value and no reason"
            )
        if status == "not_estimable" and (score is not None or reason is None):
            raise ValueError(
                "not-estimable contrast-common scores require NA and a reason"
            )
        if status == "structural_zero" and (score != 0.0 or reason is None):
            raise ValueError(
                "structural-zero contrast-common scores require zero and a reason"
            )
        if sender_grain and status == "observed":
            lr_score = unit_values["global_lr_score"]
            sender_evidence = unit_values["raw_sender_evidence"]
            if (
                lr_score is None
                or sender_evidence is None
                or score is None
                or not math.isclose(
                    score,
                    lr_score * sender_evidence,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "observed sender-LR score violates its multiplicative contract"
                )
        if type(row.is_oof_certified) is not bool:
            raise ValueError("contrast-common OOF certification flag must be boolean")
        if str(row.formal_inference_status) != _FORMAL_INFERENCE_STATUS:
            raise ValueError("contrast-common rows cannot claim formal inference")
        if str(row.claim_scope) != _expected_contrast_common_claim_scope(
            bool(row.is_oof_certified)
        ):
            raise ValueError("contrast-common row claim scope is invalid")
        expected_row_id = _source_row_id(
            source_table=table_name,
            application_id=str(row.global_common_application_id),
            sample_id=str(row.sample_id),
            subject_id=str(row.subject_id),
            context_id=str(row.context_id),
            receiver=str(row.receiver),
            family_id=str(row.family_id),
            driver_id=str(row.driver_id),
            interaction_id=str(row.interaction_id),
            mode=str(row.mode),
            sender=str(row.sender) if sender_grain else None,
        )
        if str(row.source_row_id) != expected_row_id:
            raise ValueError("contrast-common source-row identity is invalid")
    keys = [
        "crossfit_id",
        "fold_id",
        "contrast_id",
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "interaction_id",
        "mode",
    ]
    if sender_grain:
        keys.insert(-2, "sender")
    if result.duplicated(keys).any() or result["source_row_id"].duplicated().any():
        raise ValueError("contrast-common score keys must be unique")
    return result


def _validate_v3_contrast_common_lr_table(table: pd.DataFrame) -> pd.DataFrame:
    return _validate_v3_contrast_common_table(
        table,
        table_name=CROSSFIT_CONTRAST_COMMON_LR_TABLE,
        columns=_V3_CROSSFIT_CONTRAST_COMMON_LR_COLUMNS,
        score_column="global_lr_score",
        sender_grain=False,
    )


def _validate_v3_contrast_common_sender_lr_table(
    table: pd.DataFrame,
) -> pd.DataFrame:
    return _validate_v3_contrast_common_table(
        table,
        table_name=CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
        columns=_V3_CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS,
        score_column="global_sender_lr_score",
        sender_grain=True,
    )


def _same_optional_float(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _math_isclose_array(
    left: np.ndarray,
    right: np.ndarray,
    *,
    rel_tol: float,
    abs_tol: float,
) -> np.ndarray:
    difference = np.abs(left - right)
    tolerance = np.maximum(
        abs_tol,
        rel_tol * np.maximum(np.abs(left), np.abs(right)),
    )
    return cast(np.ndarray, difference <= tolerance)


def _validate_contrast_common_rows_fast(
    result: pd.DataFrame,
    *,
    table_name: str,
    unit_fields: tuple[str, ...],
    score_column: str,
    sender_grain: bool,
    conserved_sender: bool,
) -> bool:
    if not sender_grain:
        return False
    if any(
        not pd.api.types.is_numeric_dtype(result[column].dtype)
        or pd.api.types.is_complex_dtype(result[column].dtype)
        or pd.api.types.is_datetime64_any_dtype(result[column].dtype)
        or pd.api.types.is_timedelta64_dtype(result[column].dtype)
        for column in unit_fields
    ) or result["is_oof_certified"].dtype != np.dtype(bool):
        return False
    optional_presence = {
        column: _standard_optional_string_presence(result[column])
        for column in (
            "gain_calibration_binding_id",
            "gain_calibration_artifact_id",
            "gain_calibration_reason_code",
            "reason_code",
        )
    }
    if any(value is None for value in optional_presence.values()):
        return False

    modes = pd.Series(_required_string_array(result["mode"]), index=result.index)
    if not bool(modes.isin({"state", "ecosystem"}).all()):
        raise ValueError("contrast-common score mode is not recognized")
    unit_values = _finite_float_matrix(result, unit_fields)
    for index, field_name in enumerate(unit_fields):
        values = unit_values[:, index]
        finite = ~np.isnan(values)
        if bool(np.any((values[finite] < 0.0) | (values[finite] > 1.0))):
            raise ValueError(f"{field_name} must lie in [0, 1] or be missing")
    unit_by_name = {
        field_name: unit_values[:, index]
        for index, field_name in enumerate(unit_fields)
    }

    binding_present = cast(
        np.ndarray, optional_presence["gain_calibration_binding_id"]
    )
    artifact_present = cast(
        np.ndarray, optional_presence["gain_calibration_artifact_id"]
    )
    calibration_reason_present = cast(
        np.ndarray, optional_presence["gain_calibration_reason_code"]
    )
    calibration_status = pd.Series(
        _required_string_array(result["gain_calibration_status"]), index=result.index
    )
    calibration_observed = calibration_status.eq("observed").to_numpy(
        dtype=bool, copy=False
    )
    calibration_not_estimable = calibration_status.eq("not_estimable").to_numpy(
        dtype=bool, copy=False
    )
    if bool(
        np.any(
            ~binding_present
            | ~(calibration_observed | calibration_not_estimable)
        )
    ):
        raise ValueError("gain calibration row lineage is invalid")
    if bool(
        np.any(
            calibration_observed
            & (~artifact_present | calibration_reason_present)
        )
    ):
        raise ValueError("observed gain calibration row lineage is invalid")
    if bool(np.any(calibration_not_estimable & ~calibration_reason_present)):
        raise ValueError("not-estimable gain calibration requires a reason")

    score = unit_by_name[score_column]
    score_present = ~np.isnan(score)
    statuses = pd.Series(
        _required_string_array(result["status"]), index=result.index
    )
    if not bool(statuses.isin(_CONTRAST_COMMON_STATUSES).all()):
        raise ValueError("contrast-common score status is not recognized")
    reason_present = cast(np.ndarray, optional_presence["reason_code"])
    observed = statuses.eq("observed").to_numpy(dtype=bool, copy=False)
    not_estimable = statuses.eq("not_estimable").to_numpy(dtype=bool, copy=False)
    structural_zero = statuses.eq("structural_zero").to_numpy(
        dtype=bool, copy=False
    )
    if bool(np.any(observed & (~score_present | reason_present))):
        raise ValueError(
            "observed contrast-common scores require a value and no reason"
        )
    if bool(np.any(not_estimable & (score_present | ~reason_present))):
        raise ValueError(
            "not-estimable contrast-common scores require NA and a reason"
        )
    if bool(
        np.any(structural_zero & ((score != 0.0) | ~reason_present))
    ):
        raise ValueError(
            "structural-zero contrast-common scores require zero and a reason"
        )

    ligand = unit_by_name["ligand_availability"]
    prevalence = unit_by_name["training_prevalence_prior"]
    raw_sender = unit_by_name["raw_sender_evidence"]
    expected_raw_present = ~np.isnan(ligand) & ~np.isnan(prevalence)
    raw_present = ~np.isnan(raw_sender)
    if bool(np.any(raw_present != expected_raw_present)) or (
        bool(expected_raw_present.any())
        and not bool(
            _math_isclose_array(
                raw_sender[expected_raw_present],
                ligand[expected_raw_present] * prevalence[expected_raw_present],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ).all()
        )
    ):
        raise ValueError(
            "raw sender evidence violates ligand-by-prevalence multiplication"
        )
    if not conserved_sender and bool(observed.any()):
        lr_score = unit_by_name["global_lr_score"]
        valid_formula = (
            ~np.isnan(lr_score[observed])
            & ~np.isnan(raw_sender[observed])
            & _math_isclose_array(
                score[observed],
                lr_score[observed] * raw_sender[observed],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        if not bool(valid_formula.all()):
            raise ValueError(
                "observed sender-LR score violates its multiplicative contract"
            )

    flags = result["is_oof_certified"].to_numpy(dtype=bool, copy=False)
    if not bool(
        np.equal(
            _required_string_array(result["formal_inference_status"]),
            _FORMAL_INFERENCE_STATUS,
        ).all()
    ):
        raise ValueError("contrast-common rows cannot claim formal inference")
    expected_claims = np.where(
        flags,
        _expected_contrast_common_claim_scope(True),
        _expected_contrast_common_claim_scope(False),
    )
    if not bool(
        np.equal(
            _required_string_array(result["claim_scope"]), expected_claims
        ).all()
    ):
        raise ValueError("contrast-common row claim scope is invalid")

    application_ids = _required_string_array(result["global_common_application_id"])
    sample_ids = _required_string_array(result["sample_id"])
    subject_ids = _required_string_array(result["subject_id"])
    context_ids = _required_string_array(result["context_id"])
    receivers = _required_string_array(result["receiver"])
    family_ids = _required_string_array(result["family_id"])
    driver_ids = _required_string_array(result["driver_id"])
    interaction_ids = _required_string_array(result["interaction_id"])
    senders = _required_string_array(result["sender"])
    mode_values = _required_string_array(result["mode"])
    expected_row_ids = _source_row_ids(
        source_table=table_name,
        application_ids=application_ids,
        sample_ids=sample_ids,
        subject_ids=subject_ids,
        context_ids=context_ids,
        receivers=receivers,
        family_ids=family_ids,
        driver_ids=driver_ids,
        interaction_ids=interaction_ids,
        modes=mode_values,
        senders=senders,
    )
    observed_row_ids = _required_string_array(result["source_row_id"])
    if not bool(np.equal(observed_row_ids, expected_row_ids).all()):
        raise ValueError("contrast-common source-row identity is invalid")
    return True


def _validate_contrast_common_table(
    table: pd.DataFrame,
    *,
    table_name: str,
    columns: tuple[str, ...],
    score_column: str,
    sender_grain: bool,
    conserved_sender: bool = False,
) -> pd.DataFrame:
    result = table.copy(deep=True)
    _require_exact_columns(result, columns, table_name=table_name)
    required = {
        "crossfit_id",
        "spec_id",
        "repeat_id",
        "fold_id",
        "contrast_id",
        "contrast",
        "contrast_common_collection_id",
        "global_common_application_id",
        "global_common_functional_id",
        "source_family_common_functional_id",
        "source_family_common_application_id",
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "family_id",
        "driver_id",
        "interaction_id",
        "mode",
        "sender" if sender_grain else "ligand_contrast_gate_status",
        "gain_calibration_binding_id",
        "gain_calibration_status",
        "status",
        "score_version",
        "certification_status",
        "formal_inference_status",
        "claim_scope",
        "source_row_id",
    }
    if sender_grain:
        required.update({"source_sender_functional_id", "source_sender_application_id"})
    _require_identifiers(result, tuple(sorted(required)), table_name=table_name)
    unit_fields = (
        (
            "availability",
            "receiver_relative_family_gain",
            "calibrated_family_gain_percentile",
            "prior_quality",
            "global_lr_core_strength",
            "global_lr_score",
        )
        if not sender_grain
        else (
            "global_lr_score",
            "ligand_availability",
            "training_prevalence_prior",
            "raw_sender_evidence",
            *(("assignment_weight", "normalized_entropy") if conserved_sender else ()),
            "global_sender_lr_score",
        )
    )
    fast_rows_validated = _validate_contrast_common_rows_fast(
        result,
        table_name=table_name,
        unit_fields=unit_fields,
        score_column=score_column,
        sender_grain=sender_grain,
        conserved_sender=conserved_sender,
    )
    for row in (() if fast_rows_validated else result.itertuples(index=False)):
        if str(row.mode) not in {"state", "ecosystem"}:
            raise ValueError("contrast-common score mode is not recognized")
        if not sender_grain and type(row.receptor_eligible) is not bool:
            raise ValueError("contrast-common receptor eligibility must be boolean")
        unit_values = {
            field_name: _optional_unit_interval(
                getattr(row, field_name), field_name=field_name
            )
            for field_name in unit_fields
        }
        coefficient = (
            None
            if sender_grain
            else _optional_float(
                row.training_family_coefficient,
                field_name="training_family_coefficient",
            )
        )
        if coefficient is not None and coefficient < 0.0:
            raise ValueError("training_family_coefficient cannot be negative")
        binding_id = _optional_string(row.gain_calibration_binding_id)
        artifact_id = _optional_string(row.gain_calibration_artifact_id)
        calibration_status = str(row.gain_calibration_status)
        calibration_reason = _optional_string(row.gain_calibration_reason_code)
        if binding_id is None or calibration_status not in {
            "observed",
            "not_estimable",
        }:
            raise ValueError("gain calibration row lineage is invalid")
        if calibration_status == "observed" and (
            artifact_id is None or calibration_reason is not None
        ):
            raise ValueError("observed gain calibration row lineage is invalid")
        if calibration_status == "not_estimable" and calibration_reason is None:
            raise ValueError("not-estimable gain calibration requires a reason")
        score = unit_values[score_column]
        status = str(row.status)
        reason = _optional_string(row.reason_code)
        if status not in _CONTRAST_COMMON_STATUSES:
            raise ValueError("contrast-common score status is not recognized")
        if status == "observed" and (score is None or reason is not None):
            raise ValueError(
                "observed contrast-common scores require a value and no reason"
            )
        if status == "not_estimable" and (score is not None or reason is None):
            raise ValueError(
                "not-estimable contrast-common scores require NA and a reason"
            )
        if status == "structural_zero" and (score != 0.0 or reason is None):
            raise ValueError(
                "structural-zero contrast-common scores require zero and a reason"
            )
        if sender_grain:
            ligand = unit_values["ligand_availability"]
            prevalence = unit_values["training_prevalence_prior"]
            raw_sender = unit_values["raw_sender_evidence"]
            expected_raw = (
                None if ligand is None or prevalence is None else ligand * prevalence
            )
            if not _same_optional_float(raw_sender, expected_raw):
                raise ValueError(
                    "raw sender evidence violates ligand-by-prevalence multiplication"
                )
            if status == "observed" and not conserved_sender:
                lr_score = unit_values["global_lr_score"]
                if (
                    lr_score is None
                    or raw_sender is None
                    or score is None
                    or not math.isclose(
                        score,
                        lr_score * raw_sender,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError(
                        "observed sender-LR score violates its multiplicative contract"
                    )
        if type(row.is_oof_certified) is not bool:
            raise ValueError("contrast-common OOF certification flag must be boolean")
        if str(row.formal_inference_status) != _FORMAL_INFERENCE_STATUS:
            raise ValueError("contrast-common rows cannot claim formal inference")
        if str(row.claim_scope) != _expected_contrast_common_claim_scope(
            bool(row.is_oof_certified)
        ):
            raise ValueError("contrast-common row claim scope is invalid")
        expected_row_id = _source_row_id(
            source_table=table_name,
            application_id=str(row.global_common_application_id),
            sample_id=str(row.sample_id),
            subject_id=str(row.subject_id),
            context_id=str(row.context_id),
            receiver=str(row.receiver),
            family_id=str(row.family_id),
            driver_id=str(row.driver_id),
            interaction_id=str(row.interaction_id),
            mode=str(row.mode),
            sender=str(row.sender) if sender_grain else None,
        )
        if str(row.source_row_id) != expected_row_id:
            raise ValueError("contrast-common source-row identity is invalid")
    keys = [
        "crossfit_id",
        "fold_id",
        "contrast_id",
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "interaction_id",
        "mode",
    ]
    if sender_grain:
        keys.insert(-2, "sender")
    if result.duplicated(keys).any() or result["source_row_id"].duplicated().any():
        raise ValueError("contrast-common score keys must be unique")
    if sender_grain and conserved_sender:
        _validate_conserved_sender_groups(result)
    return result


def _validate_conserved_sender_groups_legacy(table: pd.DataFrame) -> None:
    group_keys = [
        "global_common_application_id",
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "family_id",
        "driver_id",
        "interaction_id",
        "mode",
    ]
    for _, group in table.groupby(group_keys, observed=True, sort=False):
        parents = [
            _optional_unit_interval(value, field_name="global_lr_score")
            for value in group["global_lr_score"]
        ]
        parent = parents[0]
        if any(not _same_optional_float(value, parent) for value in parents[1:]):
            raise ValueError("conserved sender group has multiple LR parent scores")
        weights = [
            _optional_unit_interval(value, field_name="assignment_weight")
            for value in group["assignment_weight"]
        ]
        entropies = [
            _optional_unit_interval(value, field_name="normalized_entropy")
            for value in group["normalized_entropy"]
        ]
        scores = [
            _optional_unit_interval(value, field_name="global_sender_lr_score")
            for value in group["global_sender_lr_score"]
        ]
        statuses = [str(value) for value in group["status"]]
        reasons = [_optional_string(value) for value in group["reason_code"]]
        all_weights_missing = all(value is None for value in weights)
        if all_weights_missing:
            if any(value is not None for value in entropies):
                raise ValueError(
                    "missing sender weights require missing normalized entropy"
                )
        else:
            if any(value is None for value in weights) or any(
                value is None for value in entropies
            ):
                raise ValueError(
                    "sender weights and entropy must be complete or entirely missing"
                )
            numeric_weights = cast(list[float], weights)
            if not math.isclose(
                sum(numeric_weights), 1.0, rel_tol=1e-10, abs_tol=1e-12
            ):
                raise ValueError("sender assignment weights do not sum to one")
            expected_entropy = 0.0
            if len(numeric_weights) > 1:
                expected_entropy = -sum(
                    weight * math.log(weight)
                    for weight in numeric_weights
                    if weight > 0.0
                ) / math.log(len(numeric_weights))
            if any(
                not math.isclose(
                    cast(float, value),
                    expected_entropy,
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                )
                for value in entropies
            ):
                raise ValueError(
                    "normalized entropy does not match sender assignment weights"
                )
        for weight, score, status, reason in zip(
            weights, scores, statuses, reasons, strict=True
        ):
            if parent is None:
                valid = (
                    status == "not_estimable" and score is None and reason is not None
                )
            elif parent == 0.0:
                valid = (
                    status == "structural_zero" and score == 0.0 and reason is not None
                )
            elif weight is None:
                valid = (
                    status == "not_estimable" and score is None and reason is not None
                )
            elif weight == 0.0:
                valid = (
                    status == "structural_zero"
                    and score == 0.0
                    and reason == "sender_assignment_weight_zero"
                )
            else:
                valid = (
                    status == "observed"
                    and reason is None
                    and _same_optional_float(score, parent * weight)
                )
            if not valid:
                raise ValueError(
                    "conserved sender row violates LR or assignment precedence"
                )
        if parent is not None and parent > 0.0 and not all_weights_missing:
            if any(score is None for score in scores) or not math.isclose(
                sum(cast(list[float], scores)),
                parent,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError("global sender scores do not conserve LR strength")


def _validate_conserved_sender_groups(table: pd.DataFrame) -> None:
    numeric_columns = (
        "global_lr_score",
        "assignment_weight",
        "normalized_entropy",
        "global_sender_lr_score",
    )
    fast_numeric = all(
        pd.api.types.is_numeric_dtype(table[column].dtype)
        and not pd.api.types.is_complex_dtype(table[column].dtype)
        and not pd.api.types.is_datetime64_any_dtype(table[column].dtype)
        and not pd.api.types.is_timedelta64_dtype(table[column].dtype)
        for column in numeric_columns
    )
    reason_present = _standard_optional_string_presence(table["reason_code"])
    if not fast_numeric or reason_present is None:
        _validate_conserved_sender_groups_legacy(table)
        return

    numeric = _finite_float_matrix(table, numeric_columns)
    present = ~np.isnan(numeric)
    if bool(np.any((numeric[present] < 0.0) | (numeric[present] > 1.0))):
        raise ValueError("conserved sender values must lie in [0, 1] or be missing")
    parents, weights, entropies, scores = numeric.T
    parent_present, weight_present, entropy_present, score_present = present.T

    group_keys = [
        "global_common_application_id",
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "family_id",
        "driver_id",
        "interaction_id",
        "mode",
    ]
    codes = (
        table.groupby(group_keys, observed=True, sort=False, dropna=False)
        .ngroup()
        .to_numpy(dtype=np.intp, copy=False)
    )
    if bool(np.any(codes < 0)):  # pragma: no cover - identifiers are validated first
        raise ValueError("conserved sender group identifiers cannot be missing")
    n_groups = int(codes.max()) + 1 if len(codes) else 0
    sizes = np.bincount(codes, minlength=n_groups)
    parent_counts = np.bincount(
        codes, weights=parent_present.astype(np.int8), minlength=n_groups
    ).astype(np.intp)
    if bool(np.any((parent_counts != 0) & (parent_counts != sizes))):
        raise ValueError("conserved sender group has multiple LR parent scores")
    _, first_indices = np.unique(codes, return_index=True)
    parent_by_group = parents[first_indices]
    complete_parent_groups = parent_counts == sizes
    comparable_parent_rows = parent_present & complete_parent_groups[codes]
    if bool(comparable_parent_rows.any()):
        if not bool(
            _math_isclose_array(
                parents[comparable_parent_rows],
                parent_by_group[codes[comparable_parent_rows]],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ).all()
        ):
            raise ValueError("conserved sender group has multiple LR parent scores")

    weight_counts = np.bincount(
        codes, weights=weight_present.astype(np.int8), minlength=n_groups
    ).astype(np.intp)
    entropy_counts = np.bincount(
        codes, weights=entropy_present.astype(np.int8), minlength=n_groups
    ).astype(np.intp)
    all_weights_missing = weight_counts == 0
    complete_weights = weight_counts == sizes
    if bool(np.any(all_weights_missing & (entropy_counts != 0))):
        raise ValueError("missing sender weights require missing normalized entropy")
    if bool(np.any(~all_weights_missing & ~complete_weights)):
        raise ValueError(
            "sender weights and entropy must be complete or entirely missing"
        )
    if bool(np.any(complete_weights & (entropy_counts != sizes))):
        raise ValueError(
            "sender weights and entropy must be complete or entirely missing"
        )

    weight_sums = np.bincount(
        codes, weights=np.nan_to_num(weights, nan=0.0), minlength=n_groups
    )
    if bool(complete_weights.any()) and not bool(
        _math_isclose_array(
            weight_sums[complete_weights],
            np.ones(int(complete_weights.sum()), dtype=float),
            rel_tol=1e-10,
            abs_tol=1e-12,
        ).all()
    ):
        raise ValueError("sender assignment weights do not sum to one")
    entropy_terms: np.ndarray = np.zeros(len(table), dtype=float)
    positive_weights = weight_present & (weights > 0.0)
    entropy_terms[positive_weights] = -weights[positive_weights] * np.log(
        weights[positive_weights]
    )
    entropy_sums = np.bincount(codes, weights=entropy_terms, minlength=n_groups)
    expected_entropy: np.ndarray = np.zeros(n_groups, dtype=float)
    multi_sender = sizes > 1
    expected_entropy[multi_sender] = entropy_sums[multi_sender] / np.log(
        sizes[multi_sender]
    )
    complete_entropy_rows = complete_weights[codes]
    if bool(complete_entropy_rows.any()) and not bool(
        _math_isclose_array(
            entropies[complete_entropy_rows],
            expected_entropy[codes[complete_entropy_rows]],
            rel_tol=1e-10,
            abs_tol=1e-12,
        ).all()
    ):
        raise ValueError("normalized entropy does not match sender assignment weights")

    statuses = _required_string_array(table["status"])
    reasons = table["reason_code"]
    reason_present_array: np.ndarray = reason_present
    parent_missing = ~parent_present
    parent_zero = parent_present & (parents == 0.0)
    positive_parent = parent_present & (parents > 0.0)
    weight_missing = ~weight_present
    weight_zero = weight_present & (weights == 0.0)
    positive_weight = weight_present & (weights > 0.0)
    valid: np.ndarray = np.zeros(len(table), dtype=bool)
    valid[parent_missing] = (
        (statuses[parent_missing] == "not_estimable")
        & ~score_present[parent_missing]
        & reason_present_array[parent_missing]
    )
    valid[parent_zero] = (
        (statuses[parent_zero] == "structural_zero")
        & score_present[parent_zero]
        & (scores[parent_zero] == 0.0)
        & reason_present_array[parent_zero]
    )
    missing_assignment = positive_parent & weight_missing
    valid[missing_assignment] = (
        (statuses[missing_assignment] == "not_estimable")
        & ~score_present[missing_assignment]
        & reason_present_array[missing_assignment]
    )
    zero_assignment = positive_parent & weight_zero
    valid[zero_assignment] = (
        (statuses[zero_assignment] == "structural_zero")
        & score_present[zero_assignment]
        & (scores[zero_assignment] == 0.0)
        & reasons.eq("sender_assignment_weight_zero")
        .fillna(False)
        .to_numpy(dtype=bool, copy=False)[zero_assignment]
    )
    observed_assignment = positive_parent & positive_weight
    valid[observed_assignment] = (
        (statuses[observed_assignment] == "observed")
        & ~reason_present_array[observed_assignment]
        & score_present[observed_assignment]
        & _math_isclose_array(
            scores[observed_assignment],
            parents[observed_assignment] * weights[observed_assignment],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    )
    if not bool(valid.all()):
        raise ValueError("conserved sender row violates LR or assignment precedence")

    conserved_groups = (parent_by_group > 0.0) & complete_weights
    if bool(conserved_groups.any()):
        score_counts = np.bincount(
            codes, weights=score_present.astype(np.int8), minlength=n_groups
        ).astype(np.intp)
        score_sums = np.bincount(
            codes, weights=np.nan_to_num(scores, nan=0.0), minlength=n_groups
        )
        incomplete_scores = bool(
            np.any(score_counts[conserved_groups] != sizes[conserved_groups])
        )
        mismatched_sum = not bool(
            _math_isclose_array(
                score_sums[conserved_groups],
                parent_by_group[conserved_groups],
                rel_tol=1e-10,
                abs_tol=1e-12,
            ).all()
        )
        if incomplete_scores or mismatched_sum:
            raise ValueError("global sender scores do not conserve LR strength")


def _validate_contrast_common_lr_table(table: pd.DataFrame) -> pd.DataFrame:
    return _validate_contrast_common_table(
        table,
        table_name=CROSSFIT_CONTRAST_COMMON_LR_TABLE,
        columns=CROSSFIT_CONTRAST_COMMON_LR_COLUMNS,
        score_column="global_lr_score",
        sender_grain=False,
    )


def _validate_contrast_common_sender_lr_table(
    table: pd.DataFrame,
) -> pd.DataFrame:
    return _validate_contrast_common_table(
        table,
        table_name=CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
        columns=CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS,
        score_column="global_sender_lr_score",
        sender_grain=True,
        conserved_sender=True,
    )


def _validate_v5_contrast_common_sender_lr_table(
    table: pd.DataFrame,
) -> pd.DataFrame:
    return _validate_contrast_common_table(
        table,
        table_name=CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
        columns=_V5_CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS,
        score_column="global_sender_lr_score",
        sender_grain=True,
    )


def _validate_directional_channel_registry_table(
    table: pd.DataFrame,
) -> pd.DataFrame:
    result = table.copy(deep=True)
    _require_exact_columns(
        result,
        CROSSFIT_DIRECTIONAL_CHANNEL_COLUMNS,
        table_name=CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE,
    )
    identifier_columns = (
        "crossfit_id",
        "spec_id",
        "repeat_id",
        "fold_id",
        "receiver",
        "pair_spec_id",
        "binding_id",
        "channel_role",
        "channel",
        "contrast_id",
        "contrast",
        "response_id",
        "training_artifact_id",
        "application_id",
        "status",
        "combination_rule",
        "certification_status",
        "formal_inference_status",
        "claim_scope",
    )
    _require_identifiers(
        result,
        identifier_columns,
        table_name=CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE,
    )
    for column in identifier_columns:
        if any(str(value) != str(value).strip() for value in result[column]):
            raise ValueError(
                f"{CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE}.{column} "
                "must be canonical"
            )
    if result.duplicated(["binding_id", "channel_role"]).any():
        raise ValueError("directional channel registry keys must be unique")
    for row in result.itertuples(index=False):
        role = str(row.channel_role)
        status = str(row.status)
        reason_code = _optional_string(row.reason_code)
        response_pair_id = _optional_string(row.response_pair_id)
        response_channel_id = _optional_string(row.response_channel_id)
        if (
            role not in _DIRECTIONAL_CHANNELS
            or str(row.channel) != (_DIRECTIONAL_CHANNELS[role])
        ):
            raise ValueError("directional channel role/name is invalid")
        if status not in _DIRECTIONAL_CHANNEL_STATUSES:
            raise ValueError("directional channel status is invalid")
        if status == "observed" and (
            reason_code is not None
            or response_pair_id is None
            or response_channel_id is None
        ):
            raise ValueError("observed directional channel lineage is incomplete")
        if status == "not_estimable" and (
            reason_code is None
            or response_pair_id is not None
            or response_channel_id is not None
        ):
            raise ValueError("not-estimable directional channel lineage is invalid")
        if (
            row.combination_rule != _DIRECTIONAL_COMBINATION_RULE
            or type(row.active_inhibition_allowed) is not bool
            or row.active_inhibition_allowed is not False
            or type(row.supports_active_inhibition_claim) is not bool
            or row.supports_active_inhibition_claim is not False
            or type(row.paired_score_comparison_allowed) is not bool
            or row.paired_score_comparison_allowed is not False
            or type(row.formal_inference_allowed) is not bool
            or row.formal_inference_allowed is not False
            or type(row.is_oof_certified) is not bool
            or row.formal_inference_status != _FORMAL_INFERENCE_STATUS
        ):
            raise ValueError("directional channel claim semantics are invalid")
    common_fields = (
        "crossfit_id",
        "spec_id",
        "repeat_id",
        "fold_id",
        "receiver",
        "pair_spec_id",
        "binding_id",
        "response_pair_id",
        "status",
        "reason_code",
        "combination_rule",
        "active_inhibition_allowed",
        "supports_active_inhibition_claim",
        "paired_score_comparison_allowed",
        "formal_inference_allowed",
        "certification_status",
        "is_oof_certified",
        "formal_inference_status",
        "claim_scope",
    )
    for _, rows in result.groupby("binding_id", sort=False, dropna=False):
        if len(rows) != 2 or set(rows["channel_role"].astype(str)) != {
            "forward",
            "reverse",
        }:
            raise ValueError("each directional binding requires exactly two channels")
        for field_name in common_fields:
            values = {
                canonical_json(_canonical_table_scalar(value))
                for value in rows[field_name]
            }
            if len(values) != 1:
                raise ValueError(
                    "directional channel rows disagree on binding-level lineage"
                )
        for field_name in (
            "channel",
            "contrast_id",
            "contrast",
            "response_id",
            "training_artifact_id",
            "application_id",
        ):
            if rows[field_name].astype(str).nunique(dropna=False) != 2:
                raise ValueError(
                    "directional channel-specific lineage must remain distinct"
                )
        if rows["status"].iloc[0] == "observed" and (
            rows["response_channel_id"].astype(str).nunique(dropna=False) != 2
        ):
            raise ValueError("directional response channel identities must be distinct")
    return result


def _validate_receiver_training_support_table(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy(deep=True)
    _require_exact_columns(
        result,
        CROSSFIT_RECEIVER_TRAINING_SUPPORT_COLUMNS,
        table_name=CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE,
    )
    if result.empty:
        raise ValueError("receiver training support table must not be empty")
    _require_identifiers(
        result,
        (
            "crossfit_id",
            "spec_id",
            "repeat_id",
            "receiver_universe_id",
            "receiver_axis_id",
            "fold_id",
            "receiver",
            "receiver_training_support_id",
            "receiver_training_support_status",
            "training_cell_type_ids",
        ),
        table_name=CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE,
    )
    if result.duplicated(["fold_id", "receiver"]).any():
        raise ValueError("receiver training support fold/receiver keys must be unique")
    if result["receiver_training_support_id"].astype(str).duplicated().any():
        raise ValueError("receiver training support IDs must be unique")

    for index, row in result.iterrows():
        status = str(row["receiver_training_support_status"])
        if status not in {"observed", "not_estimable"}:
            raise ValueError("receiver training support status is invalid")
        reason = _optional_string(row["receiver_training_support_reason_code"])
        expected_reason = (
            None if status == "observed" else "receiver_absent_in_outer_training"
        )
        if reason != expected_reason:
            raise ValueError("receiver training support reason/status is invalid")
        raw_training_ids = row["training_cell_type_ids"]
        try:
            training_ids = json.loads(str(raw_training_ids))
        except json.JSONDecodeError as error:
            raise ValueError(
                "receiver training cell-type IDs are not canonical JSON"
            ) from error
        if (
            not isinstance(training_ids, list)
            or any(
                not isinstance(value, str) or not value or value != value.strip()
                for value in training_ids
            )
            or training_ids != sorted(set(training_ids))
            or str(raw_training_ids) != canonical_json(training_ids)
        ):
            raise ValueError(
                "receiver training cell-type IDs must be a canonical unique array"
            )
        receiver = str(row["receiver"])
        expected_status = "observed" if receiver in training_ids else "not_estimable"
        if status != expected_status:
            raise ValueError(
                "receiver training support status disagrees with the training axis"
            )
        payload = {
            "outer_fold_id": str(row["fold_id"]),
            "reason_code": reason,
            "receiver_id": receiver,
            "receiver_universe_id": str(row["receiver_universe_id"]),
            "status": status,
            "training_cell_type_ids": training_ids,
        }
        expected_id = stable_id(
            "receiver_training_support", payload, schema_version="1"
        )
        if str(row["receiver_training_support_id"]) != expected_id:
            raise ValueError("receiver training support identity is inconsistent")
        result.at[index, "receiver_training_support_reason_code"] = reason
    return result


def _validate_table(
    name: str,
    table: pd.DataFrame,
    *,
    schema_version: str = CROSSFIT_RESULT_SCHEMA_VERSION,
) -> pd.DataFrame:
    if name == CROSSFIT_COMPONENT_TABLE:
        return _validate_component_table(table)
    if name == CROSSFIT_DIFFERENTIAL_TABLE:
        return _validate_differential_table(table)
    if name == CROSSFIT_CONTRAST_COMMON_LR_TABLE:
        if schema_version == _V3_CROSSFIT_RESULT_SCHEMA_VERSION:
            return _validate_v3_contrast_common_lr_table(table)
        return _validate_contrast_common_lr_table(table)
    if name == CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE:
        if schema_version == _V3_CROSSFIT_RESULT_SCHEMA_VERSION:
            return _validate_v3_contrast_common_sender_lr_table(table)
        if schema_version in {
            _V4_CROSSFIT_RESULT_SCHEMA_VERSION,
            _V5_CROSSFIT_RESULT_SCHEMA_VERSION,
        }:
            return _validate_v5_contrast_common_sender_lr_table(table)
        return _validate_contrast_common_sender_lr_table(table)
    if name == CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE:
        if schema_version not in {
            CROSSFIT_RESULT_SCHEMA_VERSION,
            _V8_CROSSFIT_RESULT_SCHEMA_VERSION,
            _V7_CROSSFIT_RESULT_SCHEMA_VERSION,
            _V6_CROSSFIT_RESULT_SCHEMA_VERSION,
            _V5_CROSSFIT_RESULT_SCHEMA_VERSION,
        }:
            raise KeyError(name)
        return _validate_directional_channel_registry_table(table)
    if name == CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE:
        if schema_version not in {
            CROSSFIT_RESULT_SCHEMA_VERSION,
            _V8_CROSSFIT_RESULT_SCHEMA_VERSION,
            _V7_CROSSFIT_RESULT_SCHEMA_VERSION,
        }:
            raise KeyError(name)
        return _validate_receiver_training_support_table(table)
    if name in CROSSFIT_SEMANTIC_TABLE_NAMES:
        if schema_version not in _V8_PLUS_CROSSFIT_RESULT_SCHEMA_VERSIONS:
            raise KeyError(name)
        semantic_output = _SEMANTIC_TABLE_TO_OUTPUT[name]
        _require_exact_columns(table, _TABLE_COLUMNS[name], table_name=name)
        result = table.copy(deep=True)
        if result.empty:
            return result
        lineage = {
            field_name: set(result[field_name].astype(str))
            for field_name in ("crossfit_id", "crossfit_spec_id", "repeat_id")
        }
        if any(len(values) != 1 for values in lineage.values()):
            raise ValueError(f"{name} mixes semantic collection lineage")
        _validate_semantic_table_contract(
            semantic_output,
            result,
            crossfit_id=next(iter(lineage["crossfit_id"])),
            crossfit_spec_id=next(iter(lineage["crossfit_spec_id"])),
            repeat_id=next(iter(lineage["repeat_id"])),
        )
        return result
    raise KeyError(name)


def _application_registry_counts(
    table: pd.DataFrame,
    *,
    id_column: str,
) -> dict[str, int]:
    if table.empty:
        return {}
    return {
        str(identifier): int(count)
        for identifier, count in table[id_column].value_counts().items()
    }


def _application_lineage_projection(
    application: dict[str, Any],
) -> dict[str, str | bool]:
    projection: dict[str, str | bool] = {
        field_name: application[field_name]
        for field_name in _APPLICATION_LINEAGE_STRING_FIELDS
        if field_name != "claim_scope"
    }
    for field_name, value in projection.items():
        if not isinstance(value, str) or not value:
            raise ResultValidationError(
                "Application registry lineage contains an invalid identifier",
                code="invalid_crossfit_result_application_lineage",
                field=field_name,
                remediation="Reject the bundle and rerun its producer",
            )
    is_oof_certified = application["is_oof_certified"]
    if type(is_oof_certified) is not bool:
        raise ResultValidationError(
            "Application registry lineage contains an invalid OOF flag",
            code="invalid_crossfit_result_application_lineage",
            field="is_oof_certified",
            remediation="Reject the bundle and rerun its producer",
        )
    projection["is_oof_certified"] = is_oof_certified
    projection["claim_scope"] = _expected_claim_scope(is_oof_certified)
    return projection


def _validate_application_table_lineage(
    table_name: str,
    table: pd.DataFrame,
    applications_by_id: dict[str, dict[str, Any]],
) -> None:
    grouped = table.groupby(
        "family_common_application_id",
        observed=True,
        sort=False,
    )
    for raw_application_id, rows in grouped:
        application = applications_by_id.get(str(raw_application_id))
        if application is None:
            continue
        projection = _application_lineage_projection(application)
        for field_name, expected in projection.items():
            if (
                field_name == "sender_functional_id"
                and table_name == CROSSFIT_DIFFERENTIAL_TABLE
            ):
                continue
            observed = rows[field_name].tolist()
            if type(expected) is bool:
                matches = all(
                    type(value) is bool and value is expected for value in observed
                )
            else:
                matches = all(
                    isinstance(value, str) and value == expected for value in observed
                )
            if not matches:
                raise ResultValidationError(
                    "Persisted rows mix or misstate application lineage",
                    code="crossfit_result_application_lineage_mismatch",
                    field=f"{table_name}.{field_name}",
                    remediation="Reject the bundle and rerun its producer",
                )


def _validate_source_scoring_registry(
    manifest: dict[str, object],
    applications_by_id: dict[str, dict[str, Any]],
) -> None:
    schema_version = manifest.get("schema_version")
    legacy_schema = schema_version == _V1_CROSSFIT_RESULT_SCHEMA_VERSION
    source = manifest["source_crossfit_manifest"]
    if not isinstance(source, dict):  # guarded by _validate_manifest
        raise ResultValidationError(
            "Source cross-fit manifest is unavailable for registry validation",
            code="invalid_crossfit_receiver_scoring_registry",
            field="source_crossfit_manifest",
            remediation="Reject the bundle and rerun its producer",
        )
    raw_registry = source.get("receiver_scoring_registry")
    if not isinstance(raw_registry, dict):
        if legacy_schema and raw_registry is None:
            return
        raise ResultValidationError(
            "Source receiver scoring registry is missing or invalid",
            code="invalid_crossfit_receiver_scoring_registry",
            field="receiver_scoring_registry",
            remediation="Reject the bundle and rerun its producer",
        )
    try:
        registry = ScoringCollectionDocument.from_dict(raw_registry)
    except (TypeError, ValueError) as error:
        raise ResultValidationError(
            "Source receiver scoring registry violates its contract",
            code="invalid_crossfit_receiver_scoring_registry",
            field="receiver_scoring_registry",
            remediation="Reject the bundle and rerun its producer",
        ) from error

    if not legacy_schema:
        expected_registry_version = (
            SCORING_COLLECTION_EXTENSION_VERSION
            if schema_version in _V7_PLUS_CROSSFIT_RESULT_SCHEMA_VERSIONS
            else SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION
        )
        if registry.extension_schema_version != expected_registry_version:
            raise ResultValidationError(
                "Receiver scoring registry version is incompatible with the "
                "cross-fit result schema",
                code="invalid_crossfit_receiver_scoring_registry",
                field="receiver_scoring_registry.extension_schema_version",
                remediation="Reject the bundle and rerun its producer",
            )

    source_registry_id = source.get("receiver_scoring_registry_id")
    if (
        registry.registry_id is not None
        and (not legacy_schema or source_registry_id is not None)
        and registry.registry_id != source_registry_id
    ):
        raise ResultValidationError(
            "Source receiver scoring registry identity is inconsistent",
            code="crossfit_result_scoring_registry_identity_mismatch",
            field="receiver_scoring_registry_id",
            remediation="Reject the bundle and rerun its producer",
        )
    if not legacy_schema and not registry.is_authoritative_registry:
        raise ResultValidationError(
            "Current cross-fit results require an authoritative planned registry",
            code="invalid_crossfit_receiver_scoring_registry",
            field="receiver_scoring_registry",
            remediation="Reject the bundle and rerun its producer",
        )
    if not legacy_schema and registry.to_dict() != raw_registry:
        raise ResultValidationError(
            "Source receiver scoring registry is not canonically serialized",
            code="invalid_crossfit_receiver_scoring_registry",
            field="receiver_scoring_registry",
            remediation="Reject the bundle and rerun its producer",
        )

    expected_repeat_id = str(manifest["repeat_id"])
    if any(
        collection.repeat_id != expected_repeat_id
        for collection in registry.collections
    ) or any(
        plan.repeat_id != expected_repeat_id
        for plan in registry.planned_collections or ()
    ):
        raise ResultValidationError(
            "Receiver scoring registry repeat lineage is inconsistent",
            code="crossfit_result_scoring_registry_lineage_mismatch",
            field="repeat_id",
            remediation="Reject the bundle and rerun its producer",
        )

    planned_scopes = {
        (plan.fold_id, plan.contrast, receiver)
        for plan in registry.planned_collections or ()
        for receiver in plan.planned_receivers
    }
    planned_filters = {
        (plan.fold_id, plan.contrast): plan.filter_universe_id
        for plan in registry.planned_collections or ()
    }
    support_by_scope: dict[tuple[str, str], tuple[str, str, str | None]] = {}
    if schema_version in _V7_PLUS_CROSSFIT_RESULT_SCHEMA_VERSIONS:
        receiver_ids, expected_support = _validate_source_receiver_universe(manifest)
        support_by_scope = {
            (str(row.fold_id), str(row.receiver)): (
                str(row.receiver_training_support_id),
                str(row.receiver_training_support_status),
                _optional_string(row.receiver_training_support_reason_code),
            )
            for row in expected_support.itertuples(index=False)
        }
        plans = registry.planned_collections or ()
        if not plans or any(plan.planned_receivers != receiver_ids for plan in plans):
            raise ResultValidationError(
                "Scoring plans do not preserve the frozen receiver axis",
                code="crossfit_result_scoring_registry_coverage_mismatch",
                field="receiver_scoring_registry.planned_collections",
                remediation="Reject the bundle and rerun its producer",
            )
        support_folds = {fold_id for fold_id, _ in support_by_scope}
        if {plan.fold_id for plan in plans} != support_folds:
            raise ResultValidationError(
                "Scoring plans do not exactly cover receiver-support folds",
                code="crossfit_result_scoring_registry_coverage_mismatch",
                field="receiver_scoring_registry.planned_collections",
                remediation="Reject the bundle and rerun its producer",
            )
    child_scopes: set[tuple[str, str, str]] = set()
    registered_children: dict[tuple[str, str, str], Any] = {}
    for collection in registry.collections:
        for child in collection.children:
            scope = (collection.fold_id, collection.contrast, child.receiver)
            child_scopes.add(scope)
            if schema_version in _V7_PLUS_CROSSFIT_RESULT_SCHEMA_VERSIONS:
                expected_support_lineage = support_by_scope.get(
                    (collection.fold_id, child.receiver)
                )
                observed_support_lineage = (
                    child.receiver_training_support_id,
                    child.receiver_training_support_status,
                    child.receiver_training_support_reason_code,
                )
                if observed_support_lineage != expected_support_lineage:
                    raise ResultValidationError(
                        "Registry child receiver-support lineage is inconsistent",
                        code="crossfit_result_scoring_registry_lineage_mismatch",
                        field="receiver_training_support_id",
                        remediation="Reject the bundle and rerun its producer",
                    )
            planned_filter = planned_filters.get(
                (collection.fold_id, collection.contrast)
            )
            if (
                planned_filter is not None
                and child.filter_universe_id != planned_filter
            ):
                raise ResultValidationError(
                    "Registry child filter universe disagrees with its plan",
                    code="crossfit_result_scoring_registry_lineage_mismatch",
                    field="filter_universe_id",
                    remediation="Reject the bundle and rerun its producer",
                )
            if child.registry_status in {None, "functional_registered"}:
                registered_children[scope] = child
    if planned_scopes and child_scopes != planned_scopes:
        raise ResultValidationError(
            "Registry children do not exactly cover planned receiver scopes",
            code="crossfit_result_scoring_registry_coverage_mismatch",
            field="receiver_scoring_registry",
            remediation="Reject the bundle and rerun its producer",
        )

    applications_by_scope: dict[
        tuple[str, str, str], tuple[str, dict[str, str | bool]]
    ] = {}
    for application_id, application in applications_by_id.items():
        projection = _application_lineage_projection(application)
        scope = (
            str(projection["fold_id"]),
            str(projection["contrast"]),
            str(projection["receiver"]),
        )
        if scope in applications_by_scope:
            raise ResultValidationError(
                "Multiple applications claim one planned scoring registry child",
                code="crossfit_result_scoring_registry_coverage_mismatch",
                field="applications",
                remediation="Reject the bundle and rerun its producer",
            )
        applications_by_scope[scope] = (application_id, projection)

    application_scopes = set(applications_by_scope)
    registered_scopes = set(registered_children)
    if application_scopes != registered_scopes or (
        planned_scopes and not application_scopes.issubset(planned_scopes)
    ):
        raise ResultValidationError(
            "Applications do not exactly cover registered planned receiver children",
            code="crossfit_result_scoring_registry_coverage_mismatch",
            field="applications",
            remediation="Reject the bundle and rerun its producer",
        )

    for scope, (_, projection) in applications_by_scope.items():
        child = registered_children[scope]
        if projection["family_common_functional_id"] != child.scoring_functional_id or (
            child.score_version is not None
            and projection["score_version"] != child.score_version
        ):
            raise ResultValidationError(
                "Application functional lineage disagrees with its registry child",
                code="crossfit_result_scoring_registry_lineage_mismatch",
                field="family_common_functional_id",
                remediation="Reject the bundle and rerun its producer",
            )


def _manifest_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _manifest_string_array(
    value: object,
    *,
    field_name: str,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    result = [_manifest_string(item, field_name=field_name) for item in value]
    if (not allow_empty and not result) or result != sorted(set(result)):
        raise ValueError(f"{field_name} must be a sorted unique identifier array")
    return result


def _manifest_float(
    value: object,
    *,
    field_name: str,
    allow_none: bool = False,
) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _validate_v3_contrast_common_collections_manifest(
    manifest: dict[str, Any],
) -> None:
    raw_collections = manifest["contrast_common_collections"]
    if not isinstance(raw_collections, list):
        raise ValueError("contrast-common collection registry must be an array")
    collection_ids: list[str] = []
    collection_scopes: list[tuple[str, str]] = []
    application_ids: list[str] = []
    functional_ids: list[str] = []
    canonical_collection_order: list[tuple[str, str]] = []
    is_oof_certified = bool(manifest["complete_pipeline_oof_certified"])
    for collection in raw_collections:
        if (
            not isinstance(collection, dict)
            or set(collection) != _CONTRAST_COMMON_COLLECTION_FIELDS
        ):
            raise ValueError("contrast-common collection entry is invalid")
        collection_id = _manifest_string(
            collection["contrast_common_collection_id"],
            field_name="contrast_common_collection_id",
        )
        contrast_id = _manifest_string(
            collection["contrast_id"], field_name="contrast_id"
        )
        contrast = _manifest_string(collection["contrast"], field_name="contrast")
        for field_name in (
            "crossfit_id",
            "spec_id",
            "repeat_id",
            "score_version",
            "estimand",
            "certification_status",
            "claim_scope",
        ):
            _manifest_string(collection[field_name], field_name=field_name)
        if (
            collection["crossfit_id"] != manifest["crossfit_id"]
            or collection["spec_id"] != manifest["spec_id"]
            or collection["repeat_id"] != manifest["repeat_id"]
            or collection["certification_status"] != manifest["certification_status"]
            or collection["is_oof_certified"] is not is_oof_certified
            or collection["claim_scope"]
            != _expected_contrast_common_claim_scope(is_oof_certified)
            or collection["common_functional_across_receivers"] is not False
            or collection["receiver_balanced_descriptive_collection"] is not True
            or collection["formal_inference_allowed"] is not False
        ):
            raise ValueError("contrast-common collection scope is inconsistent")
        fold_applications = collection["fold_applications"]
        if not isinstance(fold_applications, list) or not fold_applications:
            raise ValueError("contrast-common collection requires fold applications")
        fold_ids: list[str] = []
        for application in fold_applications:
            if (
                not isinstance(application, dict)
                or set(application) != _V3_CONTRAST_COMMON_APPLICATION_FIELDS
            ):
                raise ValueError("contrast-common fold application entry is invalid")
            for field_name in (
                "fold_id",
                "global_common_functional_id",
                "global_common_application_id",
                "functional_spec_id",
                "functional_schema_version",
                "filter_universe_id",
                "sender_functional_id",
                "calibration_policy",
                "interaction_mapping_digest",
                "functional_certification_status",
                "lr_scores_digest",
                "sender_scores_digest",
            ):
                _manifest_string(application[field_name], field_name=field_name)
            fold_id = str(application["fold_id"])
            fold_ids.append(fold_id)
            application_ids.append(str(application["global_common_application_id"]))
            functional_ids.append(str(application["global_common_functional_id"]))
            context_ids = _manifest_string_array(
                application["context_ids"], field_name="context_ids"
            )
            receiver_ids = _manifest_string_array(
                application["receiver_ids"], field_name="receiver_ids"
            )
            calibration_quantile = _manifest_float(
                application["calibration_quantile"],
                field_name="calibration_quantile",
            )
            if calibration_quantile is None or not 0 < calibration_quantile <= 1:
                raise ValueError("calibration_quantile must lie in (0, 1]")
            global_anchor = _manifest_float(
                application["global_training_anchor"],
                field_name="global_training_anchor",
                allow_none=True,
            )
            if global_anchor is not None and global_anchor < 0:
                raise ValueError("global training anchor cannot be negative")
            receiver_anchors = application["receiver_training_anchors"]
            receiver_factors = application["receiver_scale_factors"]
            if not isinstance(receiver_anchors, list) or not isinstance(
                receiver_factors, list
            ):
                raise ValueError("receiver calibration must be encoded as arrays")
            anchor_receivers: list[str] = []
            anchor_values: list[float | None] = []
            for item in receiver_anchors:
                if not isinstance(item, list) or len(item) != 2:
                    raise ValueError("receiver training anchor entry is invalid")
                anchor_receivers.append(
                    _manifest_string(item[0], field_name="receiver_training_anchor")
                )
                anchor = _manifest_float(
                    item[1],
                    field_name="receiver_training_anchor",
                    allow_none=True,
                )
                if anchor is not None and anchor < 0:
                    raise ValueError("receiver training anchor cannot be negative")
                anchor_values.append(anchor)
            factor_receivers: list[str] = []
            factor_values: list[float] = []
            for item in receiver_factors:
                if not isinstance(item, list) or len(item) != 2:
                    raise ValueError("receiver scale factor entry is invalid")
                factor_receivers.append(
                    _manifest_string(item[0], field_name="receiver_scale_factor")
                )
                factor = _manifest_float(item[1], field_name="receiver_scale_factor")
                if factor is None or not 0 < factor <= 1:
                    raise ValueError("receiver scale factor must lie in (0, 1]")
                factor_values.append(factor)
            if anchor_receivers != receiver_ids or factor_receivers != receiver_ids:
                raise ValueError(
                    "receiver calibration lacks exact planned receiver coverage"
                )
            positive_anchors = [
                anchor
                for anchor in anchor_values
                if anchor is not None and anchor > 0.0
            ]
            if bool(positive_anchors) != (global_anchor is not None):
                raise ValueError("global training anchor availability is inconsistent")
            for anchor, factor in zip(anchor_values, factor_values, strict=True):
                expected_factor = (
                    1.0
                    if anchor is None or anchor <= 0.0 or global_anchor is None
                    else min(1.0, global_anchor / anchor)
                )
                if not math.isclose(
                    factor,
                    expected_factor,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "receiver scale factor disagrees with train-only anchors"
                    )
            if (
                application["training_only_receiver_calibration"] is not True
                or application["receiver_scale_amplification"] is not False
                or application["common_functional_across_receivers"] is not False
                or application["receiver_balanced_descriptive_collection"] is not True
            ):
                raise ValueError(
                    "receiver-balanced descriptive collection flags are invalid"
                )
            training_subject_ids = _manifest_string_array(
                application["training_subject_ids"],
                field_name="training_subject_ids",
            )
            heldout_subject_ids = _manifest_string_array(
                application["heldout_subject_ids"],
                field_name="heldout_subject_ids",
            )
            if set(training_subject_ids).intersection(heldout_subject_ids):
                raise ValueError(
                    "contrast-common training and held-out subjects overlap"
                )
            if not context_ids:
                raise ValueError("contrast-common contexts cannot be empty")
            children = application["receiver_children"]
            if not isinstance(children, list):
                raise ValueError("contrast-common receiver children must be an array")
            child_receivers: list[str] = []
            for child in children:
                if (
                    not isinstance(child, dict)
                    or set(child) != _CONTRAST_COMMON_CHILD_FIELDS
                ):
                    raise ValueError("contrast-common receiver child is invalid")
                for field_name in _CONTRAST_COMMON_CHILD_FIELDS:
                    _manifest_string(child[field_name], field_name=field_name)
                child_receivers.append(str(child["receiver"]))
            if child_receivers != receiver_ids:
                raise ValueError(
                    "contrast-common receiver children lack exact planned coverage"
                )
            sender_lineages = application["sender_lineages"]
            if not isinstance(sender_lineages, list) or not sender_lineages:
                raise ValueError("contrast-common sender lineage must be non-empty")
            lineage_keys: list[tuple[str, str, str]] = []
            for lineage in sender_lineages:
                if (
                    not isinstance(lineage, dict)
                    or set(lineage) != _CONTRAST_COMMON_SENDER_LINEAGE_FIELDS
                ):
                    raise ValueError("contrast-common sender lineage is invalid")
                for field_name in _CONTRAST_COMMON_SENDER_LINEAGE_FIELDS:
                    _manifest_string(lineage[field_name], field_name=field_name)
                if (
                    lineage["sender_functional_id"]
                    != application["sender_functional_id"]
                ):
                    raise ValueError(
                        "contrast-common sender functional lineage is inconsistent"
                    )
                lineage_keys.append(
                    (
                        str(lineage["receiver"]),
                        str(lineage["sender_functional_id"]),
                        str(lineage["sender_application_id"]),
                    )
                )
            if lineage_keys != sorted(set(lineage_keys)) or {
                key[0] for key in lineage_keys
            } != set(receiver_ids):
                raise ValueError(
                    "contrast-common sender lineage is noncanonical or incomplete"
                )
            sender_digests = application["sender_application_digests"]
            if not isinstance(sender_digests, list):
                raise ValueError("contrast-common sender digests must be an array")
            digest_receivers: list[str] = []
            for sender_digest in sender_digests:
                if (
                    not isinstance(sender_digest, dict)
                    or set(sender_digest) != _CONTRAST_COMMON_SENDER_DIGEST_FIELDS
                ):
                    raise ValueError("contrast-common sender digest is invalid")
                for field_name in _CONTRAST_COMMON_SENDER_DIGEST_FIELDS:
                    _manifest_string(sender_digest[field_name], field_name=field_name)
                digest_receivers.append(str(sender_digest["receiver"]))
            if digest_receivers != receiver_ids:
                raise ValueError(
                    "contrast-common sender digests lack exact receiver coverage"
                )
            for count_field in ("n_lr_rows", "n_sender_rows"):
                count = application[count_field]
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise ValueError(
                        "contrast-common application row counts must be positive"
                    )
        if fold_ids != sorted(set(fold_ids)):
            raise ValueError(
                "contrast-common fold applications must be canonically ordered"
            )
        payload = {
            key: value
            for key, value in collection.items()
            if key != "contrast_common_collection_id"
        }
        if collection_id != stable_id(
            "contrast_common_oof_score_collection",
            payload,
            schema_version="1",
        ):
            raise ValueError("contrast-common collection identity is inconsistent")
        collection_ids.append(collection_id)
        collection_scopes.append((contrast_id, contrast))
        canonical_collection_order.append((contrast, contrast_id))
    if (
        len(collection_ids) != len(set(collection_ids))
        or len(collection_scopes) != len(set(collection_scopes))
        or len(application_ids) != len(set(application_ids))
        or len(functional_ids) != len(set(functional_ids))
        or canonical_collection_order != sorted(canonical_collection_order)
    ):
        raise ValueError("contrast-common collection registry is not unique/canonical")
    if manifest["contrast_common_stage_connected"] is not bool(raw_collections):
        raise ValueError(
            "contrast-common stage flag disagrees with its collection registry"
        )


def _manifest_optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _manifest_string(value, field_name=field_name)


_RECEIVER_UNIVERSE_MANIFEST_FIELDS = frozenset(
    {
        "universe_id",
        "observed_cell_type_ids",
        "receiver_axis_id",
        "receiver_ids",
        "root_config_digest",
        "root_input_digest",
        "root_input_identity_id",
        "root_subject_ids",
        "source_policy",
    }
)
_RECEIVER_TRAINING_SUPPORT_MANIFEST_FIELDS = frozenset(
    {
        "support_record_id",
        "outer_fold_id",
        "reason_code",
        "receiver_id",
        "receiver_universe_id",
        "status",
        "training_cell_type_ids",
    }
)
_ROOT_INPUT_IDENTITY_MANIFEST_FIELDS = frozenset(
    {
        "identity_id",
        "config_digest",
        "input_digest",
        "subject_ids",
        "sample_ids",
        "subject_content_digests",
    }
)
_ROOT_OBSERVED_RECEIVER_POLICY = "root_observed_cell_type_ids_context_label_blind_v1"
_EXPLICIT_RECEIVER_POLICY = "explicit_predeclared_receiver_ids_v1"
_RECEIVER_ABSENT_REASON = "receiver_absent_in_outer_training"
_RECEIVER_FAMILY_UNIVERSE_POLICY = (
    "target_prior_root_feature_axis_context_label_blind_v1"
)
_RECEIVER_FAMILY_LR_UNIVERSE_POLICY = (
    "external_resource_target_prior_unique_driver_mapping_v1"
)
_RECEIVER_FAMILY_LR_MAPPING_POLICY = "resource_prior_unique_driver_match_v1"
_RECEIVER_FAMILY_LR_UNIVERSE_SCHEMA_VERSION = "1.0.0"
_RECEIVER_FAMILY_LR_UNIVERSE_POLICY_V2 = (
    "external_resource_molecular_equivalence_target_prior_mapping_v2"
)
_RECEIVER_FAMILY_LR_MAPPING_POLICY_V2 = (
    "resource_molecular_equivalence_prior_unique_driver_match_v2"
)
_RECEIVER_FAMILY_LR_UNIVERSE_SCHEMA_VERSION_V2 = "2.0.0"
_RECEIVER_FAMILY_UNIVERSE_MANIFEST_FIELDS = frozenset(
    {
        "universe_id",
        "cosine_threshold",
        "driver_ids",
        "family_axis_id",
        "family_definitions",
        "feature_axis_id",
        "opportunity_axis_id",
        "prior_manifest_digest",
        "prior_content_id",
        "prior_resource_id",
        "prior_version",
        "receiver_axis_id",
        "receiver_universe_id",
        "root_input_digest",
        "root_input_identity_id",
        "source_policy",
        "receiver_ids",
        "feature_ids",
        "family_ids",
        "opportunity_count",
        "opportunities",
    }
)
_RECEIVER_FAMILY_DEFINITION_MANIFEST_FIELDS = frozenset(
    {
        "assignment_uncertainty",
        "driver_ids",
        "family_id",
        "mean_pairwise_cosine",
    }
)
_RECEIVER_FAMILY_OPPORTUNITY_MANIFEST_FIELDS = frozenset(
    {"receiver", "family_id", "opportunity_id"}
)
_RECEIVER_FAMILY_LR_UNIVERSE_MANIFEST_FIELDS = frozenset(
    {
        "universe_id",
        "family_axis_id",
        "feature_axis_id",
        "hypothesis_axis_id",
        "mapping_policy",
        "mapping_report_id",
        "membership_axis_id",
        "mode_axis_id",
        "modes",
        "opportunity_axis_id",
        "receiver_axis_id",
        "receiver_family_universe_id",
        "receiver_universe_id",
        "root_input_digest",
        "root_input_identity_id",
        "resource_bundle_content_id",
        "resource_id",
        "resource_manifest_digest",
        "resource_version",
        "schema_version",
        "source_policy",
        "target_prior_content_id",
        "target_prior_manifest_digest",
        "target_prior_resource_id",
        "target_prior_version",
        "receiver_ids",
        "family_ids",
        "receiver_family_universe",
        "mapping_report",
        "membership_count",
        "memberships",
        "hypothesis_count",
        "hypotheses",
        "opportunity_count",
        "opportunities",
    }
)
_RECEIVER_FAMILY_LR_UNIVERSE_MANIFEST_FIELDS_V2 = frozenset(
    {
        *_RECEIVER_FAMILY_LR_UNIVERSE_MANIFEST_FIELDS,
        "molecular_lr_equivalence_universe_id",
        "molecular_lr_axis_id",
        "molecular_lr_equivalence_universe",
    }
)
_RECEIVER_FAMILY_LR_MAPPING_REPORT_MANIFEST_FIELDS = frozenset(
    {
        "report_id",
        "mapping_policy",
        "resource_bundle_content_id",
        "target_prior_content_id",
        "interaction_count",
        "mapped_count",
        "unmapped_count",
        "ambiguous_count",
        "interaction_ids",
        "mapped_interaction_ids",
        "unmapped_interaction_ids",
        "ambiguous_interactions",
    }
)
_RECEIVER_FAMILY_LR_AMBIGUOUS_MANIFEST_FIELDS = frozenset(
    {"interaction_id", "candidate_driver_ids"}
)
_RECEIVER_FAMILY_LR_MEMBERSHIP_MANIFEST_FIELDS = frozenset(
    {"interaction_id", "driver_id", "family_id", "membership_id"}
)
_RECEIVER_FAMILY_LR_MEMBERSHIP_MANIFEST_FIELDS_V2 = frozenset(
    {*_RECEIVER_FAMILY_LR_MEMBERSHIP_MANIFEST_FIELDS, "molecular_lr_equivalence_id"}
)
_RECEIVER_FAMILY_LR_HYPOTHESIS_MANIFEST_FIELDS = frozenset(
    {
        "interaction_id",
        "driver_id",
        "family_id",
        "membership_id",
        "mode",
        "hypothesis_id",
    }
)
_RECEIVER_FAMILY_LR_HYPOTHESIS_MANIFEST_FIELDS_V2 = frozenset(
    {*_RECEIVER_FAMILY_LR_HYPOTHESIS_MANIFEST_FIELDS, "molecular_lr_equivalence_id"}
)
_RECEIVER_FAMILY_LR_OPPORTUNITY_MANIFEST_FIELDS = frozenset(
    {
        "receiver",
        "interaction_id",
        "driver_id",
        "family_id",
        "membership_id",
        "mode",
        "hypothesis_id",
        "opportunity_id",
    }
)
_RECEIVER_FAMILY_LR_OPPORTUNITY_MANIFEST_FIELDS_V2 = frozenset(
    {*_RECEIVER_FAMILY_LR_OPPORTUNITY_MANIFEST_FIELDS, "molecular_lr_equivalence_id"}
)
_MOLECULAR_LR_UNIVERSE_MANIFEST_FIELDS = frozenset(
    {
        "universe_id",
        "policy_id",
        "species",
        "gene_namespace",
        "direction",
        "component_semantics",
        "molecular_lr_axis_id",
        "molecular_lr_equivalence_ids",
        "mechanistic_variant_axis_id",
        "mechanistic_variant_ids",
        "mapping_axis_id",
        "mapping_record_ids",
        "source_bindings",
        "equivalence_classes",
        "mechanistic_variants",
        "mapping_records",
        "mapped_count",
        "unsupported_direction_count",
    }
)
_MOLECULAR_LR_SOURCE_BINDING_MANIFEST_FIELDS = frozenset(
    {
        "source_binding_id",
        "gene_namespace",
        "interaction_ids",
        "manifest_digest",
        "mapping_report",
        "resource_bundle_content_id",
        "resource_id",
        "species",
        "version",
    }
)
_MOLECULAR_LR_SOURCE_MAPPING_REPORT_MANIFEST_FIELDS = frozenset(
    {
        "source_rows",
        "loaded_rows",
        "mapped_entities",
        "unmapped_entities",
        "ambiguous_entities",
        "skipped_source_ids",
        "notes",
    }
)
_MOLECULAR_LR_EQUIVALENCE_CLASS_MANIFEST_FIELDS = frozenset(
    {
        "molecular_lr_equivalence_id",
        "component_semantics",
        "direction",
        "gene_namespace",
        "ligand_subunits",
        "policy_id",
        "receptor_subunits",
        "species",
    }
)
_MOLECULAR_LR_VARIANT_MANIFEST_FIELDS = frozenset(
    {
        "mechanistic_variant_id",
        "activating_coreceptor_subunits",
        "agonist_subunits",
        "antagonist_subunits",
        "inhibiting_coreceptor_subunits",
        "mechanistic_variant_semantics",
        "molecular_lr_equivalence_id",
        "policy_id",
    }
)
_MOLECULAR_LR_MAPPING_RECORD_MANIFEST_FIELDS = frozenset(
    {
        "mapping_record_id",
        "interaction_id",
        "mechanistic_variant_id",
        "molecular_lr_equivalence_id",
        "policy_id",
        "reason_code",
        "resource_bundle_content_id",
        "source_direction",
        "source_interaction_id",
        "status",
    }
)
_MOLECULAR_LR_DIRECTION = "ligand_to_receptor"
_MOLECULAR_LR_COMPONENT_SEMANTICS = (
    "exact_unordered_unique_component_sets_no_stoichiometry_v1"
)
_MOLECULAR_LR_VARIANT_SEMANTICS = "exact_modifier_component_sets_v1"
_MOLECULAR_LR_SUPPORTED_SOURCE_DIRECTION = "Ligand-Receptor"
_MOLECULAR_LR_UNSUPPORTED_REASON = "unsupported_interaction_direction"
_COMMUNICATION_MODE_ORDER = ("state", "ecosystem")


def _validate_source_root_input_identity(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict) or set(raw) != _ROOT_INPUT_IDENTITY_MANIFEST_FIELDS:
        raise ValueError("source root input identity is invalid")
    root = cast(dict[str, object], raw)
    identity_id = _manifest_string(root["identity_id"], field_name="identity_id")
    config_digest = _manifest_string(root["config_digest"], field_name="config_digest")
    input_digest = _manifest_string(root["input_digest"], field_name="input_digest")
    subject_ids = _manifest_string_array(root["subject_ids"], field_name="subject_ids")
    sample_ids = _manifest_string_array(root["sample_ids"], field_name="sample_ids")
    raw_subject_digests = root["subject_content_digests"]
    if not isinstance(raw_subject_digests, list):
        raise ValueError("source subject content digests must be an array")
    subject_digests: list[list[str]] = []
    for item in raw_subject_digests:
        if not isinstance(item, dict) or set(item) != {"subject_id", "digest"}:
            raise ValueError("source subject content digest entry is invalid")
        subject_digests.append(
            [
                _manifest_string(item["subject_id"], field_name="subject_id"),
                _manifest_string(item["digest"], field_name="digest"),
            ]
        )
    if [item[0] for item in subject_digests] != subject_ids:
        raise ValueError("source subject content digests do not cover root subjects")
    expected_input_digest = stable_id(
        "sanitized_raw_fold_input",
        {
            "config_digest": config_digest,
            "subject_content_digests": subject_digests,
        },
        schema_version="2",
    )
    identity_payload = {
        "config_digest": config_digest,
        "input_digest": input_digest,
        "sample_ids": sample_ids,
        "subject_content_digests": subject_digests,
        "subject_ids": subject_ids,
    }
    if input_digest != expected_input_digest or identity_id != stable_id(
        "sanitized_raw_input_identity", identity_payload, schema_version="1"
    ):
        raise ValueError("source root input identity is inconsistent")
    return {
        "identity_id": identity_id,
        **identity_payload,
    }


def _validate_source_receiver_universe(
    manifest: dict[str, object],
) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Validate the v7 frozen axis and return its exact fold-support grid."""

    if manifest.get("schema_version") not in _V7_PLUS_CROSSFIT_RESULT_SCHEMA_VERSIONS:
        raise ValueError("receiver universe lineage is only available in schema v7+")
    source = manifest.get("source_crossfit_manifest")
    if not isinstance(source, dict):
        raise ValueError("source cross-fit manifest is invalid")
    raw_universe = source.get("receiver_universe")
    if (
        not isinstance(raw_universe, dict)
        or set(raw_universe) != _RECEIVER_UNIVERSE_MANIFEST_FIELDS
    ):
        raise ValueError("source receiver universe is invalid")
    universe = cast(dict[str, object], raw_universe)
    receiver_ids = tuple(
        _manifest_string_array(universe["receiver_ids"], field_name="receiver_ids")
    )
    observed_ids = tuple(
        _manifest_string_array(
            universe["observed_cell_type_ids"],
            field_name="observed_cell_type_ids",
        )
    )
    receiver_axis_id = _manifest_string(
        universe["receiver_axis_id"], field_name="receiver_axis_id"
    )
    universe_id = _manifest_string(universe["universe_id"], field_name="universe_id")
    source_policy = _manifest_string(
        universe["source_policy"], field_name="source_policy"
    )
    policy_is_valid = (
        source_policy == _ROOT_OBSERVED_RECEIVER_POLICY and receiver_ids == observed_ids
    ) or (
        source_policy == _EXPLICIT_RECEIVER_POLICY
        and set(observed_ids).issubset(receiver_ids)
    )
    if not policy_is_valid:
        raise ValueError("source receiver universe policy is inconsistent")
    expected_axis_id = stable_id(
        "receiver_axis",
        {"receiver_ids": list(receiver_ids)},
        schema_version="1",
    )
    if receiver_axis_id != expected_axis_id:
        raise ValueError("source receiver axis identity is inconsistent")

    root = _validate_source_root_input_identity(source.get("root_input_identity"))
    root_subject_ids = tuple(
        _manifest_string_array(
            universe["root_subject_ids"], field_name="root_subject_ids"
        )
    )
    universe_payload = {
        "observed_cell_type_ids": list(observed_ids),
        "receiver_axis_id": receiver_axis_id,
        "receiver_ids": list(receiver_ids),
        "root_config_digest": _manifest_string(
            universe["root_config_digest"], field_name="root_config_digest"
        ),
        "root_input_digest": _manifest_string(
            universe["root_input_digest"], field_name="root_input_digest"
        ),
        "root_input_identity_id": _manifest_string(
            universe["root_input_identity_id"], field_name="root_input_identity_id"
        ),
        "root_subject_ids": list(root_subject_ids),
        "source_policy": source_policy,
    }
    if (
        universe_payload["root_config_digest"] != root["config_digest"]
        or universe_payload["root_input_digest"] != root["input_digest"]
        or universe_payload["root_input_identity_id"] != root["identity_id"]
        or root_subject_ids != tuple(cast(list[str], root["subject_ids"]))
        or manifest.get("root_input_digest") != root["input_digest"]
        or universe_id
        != stable_id("frozen_receiver_universe", universe_payload, schema_version="1")
        or manifest.get("receiver_universe_id") != universe_id
        or manifest.get("receiver_axis_id") != receiver_axis_id
    ):
        raise ValueError("source receiver universe lineage is inconsistent")

    raw_folds = source.get("fold_artifacts")
    if not isinstance(raw_folds, list) or not raw_folds:
        raise ValueError("source receiver support folds are missing")
    fold_ids: list[str] = []
    support_ids: list[str] = []
    support_rows: list[dict[str, object]] = []
    for fold in raw_folds:
        if not isinstance(fold, dict):
            raise ValueError("source receiver support fold is invalid")
        fold_id = _manifest_string(fold.get("fold_id"), field_name="fold_id")
        fold_ids.append(fold_id)
        raw_support = fold.get("receiver_training_support")
        if not isinstance(raw_support, list) or len(raw_support) != len(receiver_ids):
            raise ValueError("source receiver support does not cover the frozen axis")
        fold_training_ids: tuple[str, ...] | None = None
        fold_receivers: list[str] = []
        for raw_record in raw_support:
            if (
                not isinstance(raw_record, dict)
                or set(raw_record) != _RECEIVER_TRAINING_SUPPORT_MANIFEST_FIELDS
            ):
                raise ValueError("source receiver support record is invalid")
            record = cast(dict[str, object], raw_record)
            receiver = _manifest_string(record["receiver_id"], field_name="receiver_id")
            training_ids = tuple(
                _manifest_string_array(
                    record["training_cell_type_ids"],
                    field_name="training_cell_type_ids",
                    allow_empty=True,
                )
            )
            if set(training_ids).difference(receiver_ids):
                raise ValueError("source training axis exceeds the receiver universe")
            if fold_training_ids is None:
                fold_training_ids = training_ids
            elif training_ids != fold_training_ids:
                raise ValueError("source receiver support training axes disagree")
            status = _manifest_string(record["status"], field_name="status")
            reason = _manifest_optional_string(
                record["reason_code"], field_name="reason_code"
            )
            expected_status = (
                "observed" if receiver in training_ids else "not_estimable"
            )
            expected_reason = (
                None if expected_status == "observed" else _RECEIVER_ABSENT_REASON
            )
            support_payload = {
                "outer_fold_id": fold_id,
                "reason_code": reason,
                "receiver_id": receiver,
                "receiver_universe_id": universe_id,
                "status": status,
                "training_cell_type_ids": list(training_ids),
            }
            support_id = _manifest_string(
                record["support_record_id"], field_name="support_record_id"
            )
            if (
                record["outer_fold_id"] != fold_id
                or record["receiver_universe_id"] != universe_id
                or status != expected_status
                or reason != expected_reason
                or support_id
                != stable_id(
                    "receiver_training_support",
                    support_payload,
                    schema_version="1",
                )
            ):
                raise ValueError("source receiver support lineage is inconsistent")
            fold_receivers.append(receiver)
            support_ids.append(support_id)
            support_rows.append(
                {
                    "crossfit_id": manifest["crossfit_id"],
                    "spec_id": manifest["spec_id"],
                    "repeat_id": manifest["repeat_id"],
                    "receiver_universe_id": universe_id,
                    "receiver_axis_id": receiver_axis_id,
                    "fold_id": fold_id,
                    "receiver": receiver,
                    "receiver_training_support_id": support_id,
                    "receiver_training_support_status": status,
                    "receiver_training_support_reason_code": reason,
                    "training_cell_type_ids": canonical_json(list(training_ids)),
                }
            )
        if tuple(fold_receivers) != receiver_ids:
            raise ValueError("source receiver support rows are not canonically ordered")
    if len(fold_ids) != len(set(fold_ids)) or len(support_ids) != len(set(support_ids)):
        raise ValueError("source receiver support grid is not unique")
    expected = pd.DataFrame(
        support_rows,
        columns=CROSSFIT_RECEIVER_TRAINING_SUPPORT_COLUMNS,
    )
    return receiver_ids, _validate_receiver_training_support_table(expected)


def _manifest_exact_object(
    value: object,
    *,
    fields: frozenset[str],
    field_name: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{field_name} fields do not match the released schema")
    return cast(dict[str, object], value)


def _manifest_ordered_string_array(
    value: object,
    *,
    field_name: str,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    result = [_manifest_string(item, field_name=field_name) for item in value]
    if (not allow_empty and not result) or len(result) != len(set(result)):
        raise ValueError(f"{field_name} must be an ordered unique identifier array")
    return result


def _validate_molecular_lr_equivalence_universe(
    raw: object,
    *,
    expected_resource_bundle_content_id: str,
    expected_resource_id: str,
    expected_resource_version: str,
    expected_resource_manifest_digest: str,
) -> dict[str, object]:
    molecular = _manifest_exact_object(
        raw,
        fields=_MOLECULAR_LR_UNIVERSE_MANIFEST_FIELDS,
        field_name="molecular LR equivalence universe",
    )
    species = _manifest_string(molecular["species"], field_name="species")
    gene_namespace = _manifest_string(
        molecular["gene_namespace"], field_name="gene_namespace"
    )
    if species not in {"human", "mouse"} or gene_namespace not in {
        "HGNC symbol",
        "MGI symbol",
    }:
        raise ValueError("molecular LR species or namespace is unsupported")
    if (
        molecular["policy_id"] != MOLECULAR_LR_EQUIVALENCE_POLICY_ID
        or molecular["direction"] != _MOLECULAR_LR_DIRECTION
        or molecular["component_semantics"] != _MOLECULAR_LR_COMPONENT_SEMANTICS
    ):
        raise ValueError("molecular LR equivalence policy is inconsistent")

    raw_bindings = molecular["source_bindings"]
    if not isinstance(raw_bindings, list) or len(raw_bindings) != 1:
        raise ValueError("molecular LR universe requires one exact source binding")
    binding = _manifest_exact_object(
        raw_bindings[0],
        fields=_MOLECULAR_LR_SOURCE_BINDING_MANIFEST_FIELDS,
        field_name="molecular LR source binding",
    )
    binding_interaction_ids = tuple(
        _manifest_string_array(binding["interaction_ids"], field_name="interaction_ids")
    )
    source_report = _manifest_exact_object(
        binding["mapping_report"],
        fields=_MOLECULAR_LR_SOURCE_MAPPING_REPORT_MANIFEST_FIELDS,
        field_name="molecular LR source mapping report",
    )
    source_rows = _manifest_nonnegative_int(
        source_report["source_rows"], field_name="source_rows"
    )
    loaded_rows = _manifest_nonnegative_int(
        source_report["loaded_rows"], field_name="loaded_rows"
    )
    mapped_entities = _manifest_nonnegative_int(
        source_report["mapped_entities"], field_name="mapped_entities"
    )
    source_report_payload = {
        "source_rows": source_rows,
        "loaded_rows": loaded_rows,
        "mapped_entities": mapped_entities,
        **{
            field_name: _manifest_string_array(
                source_report[field_name],
                field_name=field_name,
                allow_empty=True,
            )
            for field_name in (
                "unmapped_entities",
                "ambiguous_entities",
                "skipped_source_ids",
                "notes",
            )
        },
    }
    binding_payload = {
        "gene_namespace": gene_namespace,
        "interaction_ids": list(binding_interaction_ids),
        "manifest_digest": _manifest_string(
            binding["manifest_digest"], field_name="manifest_digest"
        ),
        "mapping_report": source_report_payload,
        "resource_bundle_content_id": _manifest_string(
            binding["resource_bundle_content_id"],
            field_name="resource_bundle_content_id",
        ),
        "resource_id": _manifest_string(
            binding["resource_id"], field_name="resource_id"
        ),
        "species": species,
        "version": _manifest_string(binding["version"], field_name="version"),
    }
    source_binding_id = _manifest_string(
        binding["source_binding_id"], field_name="source_binding_id"
    )
    if (
        loaded_rows != len(binding_interaction_ids)
        or loaded_rows > source_rows
        or binding["gene_namespace"] != gene_namespace
        or binding["species"] != species
        or binding_payload["resource_bundle_content_id"]
        != expected_resource_bundle_content_id
        or binding_payload["resource_id"] != expected_resource_id
        or binding_payload["version"] != expected_resource_version
        or binding_payload["manifest_digest"] != expected_resource_manifest_digest
        or source_binding_id
        != stable_id(
            "molecular_lr_source_bundle_binding",
            binding_payload,
            schema_version="1",
        )
    ):
        raise ValueError("molecular LR source binding lineage is inconsistent")

    raw_classes = molecular["equivalence_classes"]
    if not isinstance(raw_classes, list):
        raise ValueError("molecular LR equivalence classes must be an array")
    equivalence_ids: list[str] = []
    for raw_class in raw_classes:
        equivalence = _manifest_exact_object(
            raw_class,
            fields=_MOLECULAR_LR_EQUIVALENCE_CLASS_MANIFEST_FIELDS,
            field_name="molecular LR equivalence class",
        )
        ligand_subunits = _manifest_string_array(
            equivalence["ligand_subunits"], field_name="ligand_subunits"
        )
        receptor_subunits = _manifest_string_array(
            equivalence["receptor_subunits"], field_name="receptor_subunits"
        )
        identity_payload = {
            "component_semantics": _MOLECULAR_LR_COMPONENT_SEMANTICS,
            "direction": _MOLECULAR_LR_DIRECTION,
            "gene_namespace": gene_namespace,
            "ligand_subunits": ligand_subunits,
            "policy_id": MOLECULAR_LR_EQUIVALENCE_POLICY_ID,
            "receptor_subunits": receptor_subunits,
            "species": species,
        }
        equivalence_id = _manifest_string(
            equivalence["molecular_lr_equivalence_id"],
            field_name="molecular_lr_equivalence_id",
        )
        if (
            equivalence["component_semantics"] != _MOLECULAR_LR_COMPONENT_SEMANTICS
            or equivalence["direction"] != _MOLECULAR_LR_DIRECTION
            or equivalence["gene_namespace"] != gene_namespace
            or equivalence["policy_id"] != MOLECULAR_LR_EQUIVALENCE_POLICY_ID
            or equivalence["species"] != species
            or equivalence_id
            != stable_id(
                "molecular_lr_equivalence",
                identity_payload,
                schema_version="1",
            )
        ):
            raise ValueError("molecular LR equivalence class is inconsistent")
        equivalence_ids.append(equivalence_id)
    if equivalence_ids != sorted(set(equivalence_ids)):
        raise ValueError("molecular LR equivalence classes are not canonical")

    raw_variants = molecular["mechanistic_variants"]
    if not isinstance(raw_variants, list):
        raise ValueError("molecular LR mechanistic variants must be an array")
    variant_ids: list[str] = []
    variant_to_core: dict[str, str] = {}
    for raw_variant in raw_variants:
        variant = _manifest_exact_object(
            raw_variant,
            fields=_MOLECULAR_LR_VARIANT_MANIFEST_FIELDS,
            field_name="molecular LR mechanistic variant",
        )
        molecular_id = _manifest_string(
            variant["molecular_lr_equivalence_id"],
            field_name="molecular_lr_equivalence_id",
        )
        identity_payload = {
            field_name: _manifest_string_array(
                variant[field_name], field_name=field_name, allow_empty=True
            )
            for field_name in (
                "activating_coreceptor_subunits",
                "agonist_subunits",
                "antagonist_subunits",
                "inhibiting_coreceptor_subunits",
            )
        }
        identity_payload.update(
            {
                "mechanistic_variant_semantics": (_MOLECULAR_LR_VARIANT_SEMANTICS),
                "molecular_lr_equivalence_id": molecular_id,
                "policy_id": MECHANISTIC_VARIANT_POLICY_ID,
            }
        )
        variant_id = _manifest_string(
            variant["mechanistic_variant_id"], field_name="mechanistic_variant_id"
        )
        if (
            molecular_id not in equivalence_ids
            or variant["mechanistic_variant_semantics"]
            != _MOLECULAR_LR_VARIANT_SEMANTICS
            or variant["policy_id"] != MECHANISTIC_VARIANT_POLICY_ID
            or variant_id
            != stable_id(
                "molecular_lr_mechanistic_variant",
                identity_payload,
                schema_version="1",
            )
        ):
            raise ValueError("molecular LR mechanistic variant is inconsistent")
        variant_ids.append(variant_id)
        variant_to_core[variant_id] = molecular_id
    if variant_ids != sorted(set(variant_ids)):
        raise ValueError("molecular LR mechanistic variants are not canonical")

    raw_records = molecular["mapping_records"]
    if not isinstance(raw_records, list):
        raise ValueError("molecular LR mapping records must be an array")
    mapping_ids: list[str] = []
    mapping_pairs: set[tuple[str, str]] = set()
    mapped_by_interaction: dict[str, str] = {}
    unsupported_interaction_ids: set[str] = set()
    mapped_core_ids: set[str] = set()
    mapped_variant_ids: set[str] = set()
    for raw_record in raw_records:
        record = _manifest_exact_object(
            raw_record,
            fields=_MOLECULAR_LR_MAPPING_RECORD_MANIFEST_FIELDS,
            field_name="molecular LR mapping record",
        )
        interaction_id = _manifest_string(
            record["interaction_id"], field_name="interaction_id"
        )
        source_interaction_id = _manifest_string(
            record["source_interaction_id"], field_name="source_interaction_id"
        )
        source_direction = _manifest_string(
            record["source_direction"], field_name="source_direction"
        )
        content_id = _manifest_string(
            record["resource_bundle_content_id"],
            field_name="resource_bundle_content_id",
        )
        status = _manifest_string(record["status"], field_name="status")
        record_molecular_id = _manifest_optional_string(
            record["molecular_lr_equivalence_id"],
            field_name="molecular_lr_equivalence_id",
        )
        record_variant_id = _manifest_optional_string(
            record["mechanistic_variant_id"],
            field_name="mechanistic_variant_id",
        )
        reason_code = _manifest_optional_string(
            record["reason_code"], field_name="reason_code"
        )
        if status == "mapped":
            if (
                source_direction != _MOLECULAR_LR_SUPPORTED_SOURCE_DIRECTION
                or record_molecular_id is None
                or record_variant_id is None
                or record_molecular_id not in equivalence_ids
                or record_variant_id not in variant_to_core
                or variant_to_core[record_variant_id] != record_molecular_id
                or reason_code is not None
            ):
                raise ValueError("mapped molecular LR record is inconsistent")
            mapped_by_interaction[interaction_id] = record_molecular_id
            mapped_core_ids.add(record_molecular_id)
            mapped_variant_ids.add(record_variant_id)
        elif status == "unsupported_direction":
            if (
                source_direction == _MOLECULAR_LR_SUPPORTED_SOURCE_DIRECTION
                or record_molecular_id is not None
                or record_variant_id is not None
                or reason_code != _MOLECULAR_LR_UNSUPPORTED_REASON
            ):
                raise ValueError("unsupported molecular LR record is inconsistent")
            unsupported_interaction_ids.add(interaction_id)
        else:
            raise ValueError("molecular LR mapping status is unsupported")
        mapping_identity_payload: dict[str, object] = {
            "interaction_id": interaction_id,
            "mechanistic_variant_id": record_variant_id,
            "molecular_lr_equivalence_id": record_molecular_id,
            "policy_id": MOLECULAR_LR_EQUIVALENCE_POLICY_ID,
            "reason_code": reason_code,
            "resource_bundle_content_id": content_id,
            "source_direction": source_direction,
            "source_interaction_id": source_interaction_id,
            "status": status,
        }
        mapping_id = _manifest_string(
            record["mapping_record_id"], field_name="mapping_record_id"
        )
        pair = (content_id, interaction_id)
        if (
            record["policy_id"] != MOLECULAR_LR_EQUIVALENCE_POLICY_ID
            or content_id != expected_resource_bundle_content_id
            or pair in mapping_pairs
            or mapping_id
            != stable_id(
                "molecular_lr_mapping_record",
                mapping_identity_payload,
                schema_version="1",
            )
        ):
            raise ValueError("molecular LR mapping record identity is inconsistent")
        mapping_pairs.add(pair)
        mapping_ids.append(mapping_id)
    expected_pairs = {
        (expected_resource_bundle_content_id, interaction_id)
        for interaction_id in binding_interaction_ids
    }
    if (
        mapping_ids != sorted(set(mapping_ids))
        or mapping_pairs != expected_pairs
        or mapped_core_ids != set(equivalence_ids)
        or mapped_variant_ids != set(variant_ids)
    ):
        raise ValueError("molecular LR mapping grid is incomplete or noncanonical")

    expected_molecular_axis_id = stable_id(
        "molecular_lr_equivalence_axis",
        {
            "gene_namespace": gene_namespace,
            "molecular_lr_equivalence_ids": equivalence_ids,
            "policy_id": MOLECULAR_LR_EQUIVALENCE_POLICY_ID,
            "species": species,
        },
        schema_version="1",
    )
    expected_variant_axis_id = stable_id(
        "molecular_lr_mechanistic_variant_axis",
        {
            "mechanistic_variant_ids": variant_ids,
            "molecular_lr_axis_id": expected_molecular_axis_id,
            "policy_id": MECHANISTIC_VARIANT_POLICY_ID,
        },
        schema_version="1",
    )
    expected_mapping_axis_id = stable_id(
        "molecular_lr_mapping_axis",
        {
            "mapping_record_ids": mapping_ids,
            "mechanistic_variant_axis_id": expected_variant_axis_id,
            "molecular_lr_axis_id": expected_molecular_axis_id,
            "policy_id": MOLECULAR_LR_EQUIVALENCE_POLICY_ID,
            "source_binding_ids": [source_binding_id],
        },
        schema_version="1",
    )
    expected_universe_id = stable_id(
        "frozen_molecular_lr_equivalence_universe",
        {
            "gene_namespace": gene_namespace,
            "mapping_axis_id": expected_mapping_axis_id,
            "mechanistic_variant_axis_id": expected_variant_axis_id,
            "molecular_lr_axis_id": expected_molecular_axis_id,
            "policy_id": MOLECULAR_LR_EQUIVALENCE_POLICY_ID,
            "source_binding_ids": [source_binding_id],
            "species": species,
        },
        schema_version="1",
    )
    if (
        molecular["molecular_lr_equivalence_ids"] != equivalence_ids
        or molecular["mechanistic_variant_ids"] != variant_ids
        or molecular["mapping_record_ids"] != mapping_ids
        or molecular["molecular_lr_axis_id"] != expected_molecular_axis_id
        or molecular["mechanistic_variant_axis_id"] != expected_variant_axis_id
        or molecular["mapping_axis_id"] != expected_mapping_axis_id
        or molecular["universe_id"] != expected_universe_id
        or _manifest_nonnegative_int(
            molecular["mapped_count"], field_name="mapped_count"
        )
        != len(mapped_by_interaction)
        or _manifest_nonnegative_int(
            molecular["unsupported_direction_count"],
            field_name="unsupported_direction_count",
        )
        != len(unsupported_interaction_ids)
    ):
        raise ValueError("molecular LR universe axes or identity are inconsistent")
    return {
        "universe_id": expected_universe_id,
        "molecular_lr_axis_id": expected_molecular_axis_id,
        "interaction_ids": binding_interaction_ids,
        "mapped_by_interaction": mapped_by_interaction,
        "unsupported_interaction_ids": unsupported_interaction_ids,
    }


def _validate_receiver_family_opportunity_universe(
    manifest: dict[str, object],
    raw: object,
    *,
    field_name: str = "receiver-family opportunity universe",
) -> dict[str, object]:
    """Strictly replay one root receiver x family opportunity universe."""

    receiver_ids, _ = _validate_source_receiver_universe(manifest)
    source = manifest.get("source_crossfit_manifest")
    if not isinstance(source, dict):  # guarded by the receiver-universe validator
        raise ValueError("source cross-fit manifest is invalid")
    raw_receiver_universe = cast(dict[str, object], source["receiver_universe"])
    receiver_universe_id = _manifest_string(
        raw_receiver_universe["universe_id"], field_name="receiver_universe_id"
    )
    receiver_axis_id = _manifest_string(
        raw_receiver_universe["receiver_axis_id"], field_name="receiver_axis_id"
    )
    root = _validate_source_root_input_identity(source.get("root_input_identity"))
    universe = _manifest_exact_object(
        raw,
        fields=_RECEIVER_FAMILY_UNIVERSE_MANIFEST_FIELDS,
        field_name=field_name,
    )

    universe_receivers = tuple(
        _manifest_string_array(universe["receiver_ids"], field_name="receiver_ids")
    )
    feature_ids = tuple(
        _manifest_ordered_string_array(
            universe["feature_ids"], field_name="feature_ids"
        )
    )
    driver_ids = tuple(
        _manifest_string_array(universe["driver_ids"], field_name="driver_ids")
    )
    family_ids = tuple(
        _manifest_string_array(universe["family_ids"], field_name="family_ids")
    )
    cosine_threshold = cast(
        float,
        _manifest_float(universe["cosine_threshold"], field_name="cosine_threshold"),
    )
    if not 0.0 <= cosine_threshold <= 1.0:
        raise ValueError("receiver-family cosine threshold must lie in [0, 1]")
    raw_spec = source.get("spec")
    if not isinstance(raw_spec, dict):
        raise ValueError("receiver-family universe requires a cross-fit spec")
    spec_cosine_threshold = cast(
        float,
        _manifest_float(
            raw_spec.get("family_cosine_threshold"),
            field_name="spec.family_cosine_threshold",
        ),
    )
    if cosine_threshold != spec_cosine_threshold:
        raise ValueError("receiver-family threshold differs from the cross-fit spec")

    prior_resource_id = _manifest_string(
        universe["prior_resource_id"], field_name="prior_resource_id"
    )
    prior_version = _manifest_string(
        universe["prior_version"], field_name="prior_version"
    )
    prior_manifest_digest = _manifest_string(
        universe["prior_manifest_digest"], field_name="prior_manifest_digest"
    )
    prior_content_id = _manifest_string(
        universe["prior_content_id"], field_name="prior_content_id"
    )
    root_identity_id = _manifest_string(
        universe["root_input_identity_id"], field_name="root_input_identity_id"
    )
    root_digest = _manifest_string(
        universe["root_input_digest"], field_name="root_input_digest"
    )
    universe_receiver_id = _manifest_string(
        universe["receiver_universe_id"], field_name="receiver_universe_id"
    )
    universe_receiver_axis_id = _manifest_string(
        universe["receiver_axis_id"], field_name="receiver_axis_id"
    )
    feature_axis_id = _manifest_string(
        universe["feature_axis_id"], field_name="feature_axis_id"
    )
    family_axis_id = _manifest_string(
        universe["family_axis_id"], field_name="family_axis_id"
    )
    opportunity_axis_id = _manifest_string(
        universe["opportunity_axis_id"], field_name="opportunity_axis_id"
    )
    if universe["source_policy"] != _RECEIVER_FAMILY_UNIVERSE_POLICY:
        raise ValueError("receiver-family source policy is invalid")
    if (
        universe_receivers != receiver_ids
        or universe_receiver_id != receiver_universe_id
        or universe_receiver_axis_id != receiver_axis_id
        or root_identity_id != root["identity_id"]
        or root_digest != root["input_digest"]
        or root_digest != manifest.get("root_input_digest")
    ):
        raise ValueError("receiver-family root/receiver lineage is inconsistent")

    raw_definitions = universe["family_definitions"]
    if not isinstance(raw_definitions, list) or not raw_definitions:
        raise ValueError("receiver-family definitions are invalid")
    definitions: list[dict[str, object]] = []
    definition_ids: list[str] = []
    family_by_driver: dict[str, str] = {}
    for raw_definition in raw_definitions:
        definition = _manifest_exact_object(
            raw_definition,
            fields=_RECEIVER_FAMILY_DEFINITION_MANIFEST_FIELDS,
            field_name="receiver-family definition",
        )
        definition_driver_ids = tuple(
            _manifest_string_array(
                definition["driver_ids"], field_name="family.driver_ids"
            )
        )
        family_id = _manifest_string(definition["family_id"], field_name="family_id")
        mean_cosine = cast(
            float,
            _manifest_float(
                definition["mean_pairwise_cosine"],
                field_name="mean_pairwise_cosine",
            ),
        )
        uncertainty = cast(
            float,
            _manifest_float(
                definition["assignment_uncertainty"],
                field_name="assignment_uncertainty",
            ),
        )
        if not 0.0 <= mean_cosine <= 1.0 or not 0.0 <= uncertainty <= 1.0:
            raise ValueError("receiver-family diagnostics must lie in [0, 1]")
        expected_family_id = stable_id(
            "driver_family",
            {
                "driver_ids": list(definition_driver_ids),
                "prior_resource_id": prior_resource_id,
                "prior_version": prior_version,
            },
        )
        if family_id != expected_family_id:
            raise ValueError("receiver-family identity differs from its prior members")
        if any(driver_id in family_by_driver for driver_id in definition_driver_ids):
            raise ValueError("receiver-family definitions overlap")
        family_by_driver.update(dict.fromkeys(definition_driver_ids, family_id))
        definition_ids.append(family_id)
        definitions.append(
            {
                "assignment_uncertainty": uncertainty,
                "driver_ids": list(definition_driver_ids),
                "family_id": family_id,
                "mean_pairwise_cosine": mean_cosine,
            }
        )
    if (
        tuple(definition_ids) != family_ids
        or tuple(definition_ids) != tuple(sorted(set(definition_ids)))
        or tuple(sorted(family_by_driver)) != driver_ids
    ):
        raise ValueError("receiver-family definitions do not partition the driver axis")

    expected_feature_axis_id = stable_id(
        "root_feature_axis",
        {"feature_ids": list(feature_ids)},
        schema_version="1",
    )
    expected_family_axis_id = stable_id(
        "strict_driver_family_axis",
        {
            "cosine_threshold": cosine_threshold,
            "family_definitions": [
                {
                    "driver_ids": definition["driver_ids"],
                    "family_id": definition["family_id"],
                }
                for definition in definitions
            ],
            "feature_axis_id": expected_feature_axis_id,
            "prior_manifest_digest": prior_manifest_digest,
        },
        schema_version="1",
    )
    expected_opportunity_axis_id = stable_id(
        "receiver_family_opportunity_axis",
        {
            "family_axis_id": expected_family_axis_id,
            "receiver_axis_id": receiver_axis_id,
        },
        schema_version="1",
    )
    expected_opportunities = [
        {
            "receiver": receiver,
            "family_id": family_id,
            "opportunity_id": stable_id(
                "receiver_family_opportunity",
                {
                    "family_id": family_id,
                    "opportunity_axis_id": expected_opportunity_axis_id,
                    "receiver": receiver,
                },
                schema_version="1",
            ),
        }
        for receiver in receiver_ids
        for family_id in family_ids
    ]
    raw_opportunities = universe["opportunities"]
    if not isinstance(raw_opportunities, list) or any(
        not isinstance(item, dict)
        or set(item) != _RECEIVER_FAMILY_OPPORTUNITY_MANIFEST_FIELDS
        for item in raw_opportunities
    ):
        raise ValueError("receiver-family opportunities are invalid")
    if (
        feature_axis_id != expected_feature_axis_id
        or family_axis_id != expected_family_axis_id
        or opportunity_axis_id != expected_opportunity_axis_id
        or _manifest_nonnegative_int(
            universe["opportunity_count"], field_name="opportunity_count"
        )
        != len(expected_opportunities)
        or raw_opportunities != expected_opportunities
    ):
        raise ValueError("receiver-family opportunity grid is inconsistent")
    identity_payload = {
        "cosine_threshold": cosine_threshold,
        "driver_ids": list(driver_ids),
        "family_axis_id": expected_family_axis_id,
        "family_definitions": definitions,
        "feature_axis_id": expected_feature_axis_id,
        "opportunity_axis_id": expected_opportunity_axis_id,
        "prior_manifest_digest": prior_manifest_digest,
        "prior_content_id": prior_content_id,
        "prior_resource_id": prior_resource_id,
        "prior_version": prior_version,
        "receiver_axis_id": receiver_axis_id,
        "receiver_universe_id": receiver_universe_id,
        "root_input_digest": root_digest,
        "root_input_identity_id": root_identity_id,
        "source_policy": _RECEIVER_FAMILY_UNIVERSE_POLICY,
    }
    universe_id = _manifest_string(
        universe["universe_id"], field_name="receiver_family_opportunity_universe_id"
    )
    if universe_id != stable_id(
        "frozen_receiver_family_opportunity_universe",
        identity_payload,
        schema_version="1",
    ):
        raise ValueError("receiver-family universe identity is inconsistent")
    return {
        "universe": universe,
        "universe_id": universe_id,
        "receiver_ids": receiver_ids,
        "receiver_universe_id": receiver_universe_id,
        "receiver_axis_id": receiver_axis_id,
        "root_input_identity_id": root_identity_id,
        "root_input_digest": root_digest,
        "feature_ids": feature_ids,
        "feature_axis_id": expected_feature_axis_id,
        "driver_ids": driver_ids,
        "family_ids": family_ids,
        "family_by_driver": family_by_driver,
        "family_axis_id": expected_family_axis_id,
        "opportunity_axis_id": expected_opportunity_axis_id,
        "prior_resource_id": prior_resource_id,
        "prior_version": prior_version,
        "prior_manifest_digest": prior_manifest_digest,
        "prior_content_id": prior_content_id,
    }


def _validate_source_receiver_family_opportunity_universe(
    manifest: dict[str, object],
) -> dict[str, object]:
    """Require the v9 root universe and bind both persisted manifest copies."""

    if manifest.get("schema_version") != CROSSFIT_RESULT_SCHEMA_VERSION:
        raise ValueError(
            "mandatory receiver-family opportunity lineage requires schema v9"
        )
    source = manifest.get("source_crossfit_manifest")
    if not isinstance(source, dict):
        raise ValueError("source cross-fit manifest is invalid")
    raw_source = source.get("receiver_family_opportunity_universe")
    raw_persisted = manifest.get("receiver_family_opportunity_universe")
    if raw_source != raw_persisted:
        raise ValueError(
            "persisted receiver-family opportunity universe differs from its source"
        )
    contract = _validate_receiver_family_opportunity_universe(
        manifest,
        raw_source,
        field_name="source receiver-family opportunity universe",
    )
    if (
        manifest.get("receiver_family_opportunity_universe_id")
        != contract["universe_id"]
    ):
        raise ValueError(
            "receiver-family opportunity universe result identity is inconsistent"
        )
    if (
        source.get("receiver_family_opportunity_universe_id") != contract["universe_id"]
        or source.get("family_axis_id") != contract["family_axis_id"]
        or source.get("receiver_family_opportunity_axis_id")
        != contract["opportunity_axis_id"]
        or manifest.get("family_axis_id") != contract["family_axis_id"]
        or manifest.get("receiver_family_opportunity_axis_id")
        != contract["opportunity_axis_id"]
    ):
        raise ValueError("receiver-family result/source axis projections disagree")
    directional = source.get("directional_lr_hypothesis_universe")
    if directional is not None:
        if (
            not isinstance(directional, dict)
            or directional.get("receiver_family_universe") != raw_source
        ):
            raise ValueError(
                "directional LR and root receiver-family universes disagree"
            )
    return contract


def _validate_source_directional_lr_hypothesis_universe(
    manifest: dict[str, object],
) -> None:
    """Validate an optional v7 pre-fit directional family/LR universe."""

    if manifest.get("schema_version") not in _V7_PLUS_CROSSFIT_RESULT_SCHEMA_VERSIONS:
        raise ValueError(
            "directional LR hypothesis universe lineage is only available in schema v7+"
        )
    source = manifest.get("source_crossfit_manifest")
    if not isinstance(source, dict):
        raise ValueError("source cross-fit manifest is invalid")
    if "directional_lr_hypothesis_universe" not in source:
        return
    raw_universe = source["directional_lr_hypothesis_universe"]

    raw_spec = source.get("spec")
    if not isinstance(raw_spec, dict):
        raise ValueError(
            "source directional LR hypothesis universe requires a cross-fit spec"
        )
    raw_pairs = raw_spec.get("directional_pairs")
    if (
        not isinstance(raw_pairs, list)
        or not raw_pairs
        or any(not isinstance(item, dict) for item in raw_pairs)
    ):
        raise ValueError(
            "source directional LR hypothesis universe requires directional_pairs"
        )
    pair_ids = [
        _manifest_string(item.get("pair_spec_id"), field_name="pair_spec_id")
        for item in cast(list[dict[str, object]], raw_pairs)
    ]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("source directional pair identities must be unique")

    receiver_ids, _ = _validate_source_receiver_universe(manifest)
    raw_receiver_universe = cast(dict[str, object], source["receiver_universe"])
    receiver_universe_id = _manifest_string(
        raw_receiver_universe["universe_id"], field_name="receiver_universe_id"
    )
    receiver_axis_id = _manifest_string(
        raw_receiver_universe["receiver_axis_id"], field_name="receiver_axis_id"
    )
    if not isinstance(raw_universe, dict):
        raise ValueError("source directional LR hypothesis universe is invalid")
    universe_schema_version = raw_universe.get("schema_version")
    if universe_schema_version == _RECEIVER_FAMILY_LR_UNIVERSE_SCHEMA_VERSION:
        is_v2 = False
        universe_fields = _RECEIVER_FAMILY_LR_UNIVERSE_MANIFEST_FIELDS
        expected_mapping_policy = _RECEIVER_FAMILY_LR_MAPPING_POLICY
        expected_source_policy = _RECEIVER_FAMILY_LR_UNIVERSE_POLICY
        expected_universe_schema_version = _RECEIVER_FAMILY_LR_UNIVERSE_SCHEMA_VERSION
        identity_schema_version = "1"
    elif universe_schema_version == _RECEIVER_FAMILY_LR_UNIVERSE_SCHEMA_VERSION_V2:
        is_v2 = True
        universe_fields = _RECEIVER_FAMILY_LR_UNIVERSE_MANIFEST_FIELDS_V2
        expected_mapping_policy = _RECEIVER_FAMILY_LR_MAPPING_POLICY_V2
        expected_source_policy = _RECEIVER_FAMILY_LR_UNIVERSE_POLICY_V2
        expected_universe_schema_version = (
            _RECEIVER_FAMILY_LR_UNIVERSE_SCHEMA_VERSION_V2
        )
        identity_schema_version = "2"
    else:
        raise ValueError("directional LR hypothesis universe schema is unsupported")
    universe = _manifest_exact_object(
        raw_universe,
        fields=universe_fields,
        field_name="source directional LR hypothesis universe",
    )
    parent_contract = _validate_receiver_family_opportunity_universe(
        manifest,
        universe["receiver_family_universe"],
        field_name="embedded receiver-family universe",
    )
    driver_ids = cast(tuple[str, ...], parent_contract["driver_ids"])
    family_ids = cast(tuple[str, ...], parent_contract["family_ids"])
    family_by_driver = cast(dict[str, str], parent_contract["family_by_driver"])
    expected_feature_axis_id = cast(str, parent_contract["feature_axis_id"])
    expected_family_axis_id = cast(str, parent_contract["family_axis_id"])
    parent_universe_id = cast(str, parent_contract["universe_id"])
    parent_root_identity_id = cast(str, parent_contract["root_input_identity_id"])
    parent_root_digest = cast(str, parent_contract["root_input_digest"])
    prior_resource_id = cast(str, parent_contract["prior_resource_id"])
    prior_version = cast(str, parent_contract["prior_version"])
    prior_manifest_digest = cast(str, parent_contract["prior_manifest_digest"])
    prior_content_id = cast(str, parent_contract["prior_content_id"])

    mapping_policy = _manifest_string(
        universe["mapping_policy"], field_name="mapping_policy"
    )
    resource_content_id = _manifest_string(
        universe["resource_bundle_content_id"],
        field_name="resource_bundle_content_id",
    )
    target_prior_content_id = _manifest_string(
        universe["target_prior_content_id"],
        field_name="target_prior_content_id",
    )
    resource_id = _manifest_string(universe["resource_id"], field_name="resource_id")
    resource_version = _manifest_string(
        universe["resource_version"], field_name="resource_version"
    )
    resource_manifest_digest = _manifest_string(
        universe["resource_manifest_digest"],
        field_name="resource_manifest_digest",
    )
    molecular_contract: dict[str, object] | None = None
    if is_v2:
        molecular_contract = _validate_molecular_lr_equivalence_universe(
            universe["molecular_lr_equivalence_universe"],
            expected_resource_bundle_content_id=resource_content_id,
            expected_resource_id=resource_id,
            expected_resource_version=resource_version,
            expected_resource_manifest_digest=resource_manifest_digest,
        )
        if (
            universe["molecular_lr_equivalence_universe_id"]
            != molecular_contract["universe_id"]
            or universe["molecular_lr_axis_id"]
            != molecular_contract["molecular_lr_axis_id"]
        ):
            raise ValueError(
                "directional LR molecular universe lineage is inconsistent"
            )
    mapping_report = _manifest_exact_object(
        universe["mapping_report"],
        fields=_RECEIVER_FAMILY_LR_MAPPING_REPORT_MANIFEST_FIELDS,
        field_name="directional LR mapping report",
    )
    interaction_ids = tuple(
        _manifest_string_array(
            mapping_report["interaction_ids"], field_name="interaction_ids"
        )
    )
    mapped_interaction_ids = tuple(
        _manifest_string_array(
            mapping_report["mapped_interaction_ids"],
            field_name="mapped_interaction_ids",
            allow_empty=True,
        )
    )
    unmapped_interaction_ids = tuple(
        _manifest_string_array(
            mapping_report["unmapped_interaction_ids"],
            field_name="unmapped_interaction_ids",
            allow_empty=True,
        )
    )
    raw_ambiguous = mapping_report["ambiguous_interactions"]
    if not isinstance(raw_ambiguous, list):
        raise ValueError("directional LR ambiguous interactions must be an array")
    ambiguous_interactions: list[dict[str, object]] = []
    ambiguous_ids: list[str] = []
    for raw_entry in raw_ambiguous:
        entry = _manifest_exact_object(
            raw_entry,
            fields=_RECEIVER_FAMILY_LR_AMBIGUOUS_MANIFEST_FIELDS,
            field_name="directional LR ambiguous interaction",
        )
        interaction_id = _manifest_string(
            entry["interaction_id"], field_name="interaction_id"
        )
        candidate_driver_ids = _manifest_string_array(
            entry["candidate_driver_ids"], field_name="candidate_driver_ids"
        )
        if len(candidate_driver_ids) < 2 or not set(candidate_driver_ids).issubset(
            driver_ids
        ):
            raise ValueError("directional LR ambiguous candidates are invalid")
        ambiguous_ids.append(interaction_id)
        ambiguous_interactions.append(
            {
                "interaction_id": interaction_id,
                "candidate_driver_ids": candidate_driver_ids,
            }
        )
    if ambiguous_ids != sorted(set(ambiguous_ids)):
        raise ValueError("directional LR ambiguous interactions are not canonical")
    partition = (
        *mapped_interaction_ids,
        *unmapped_interaction_ids,
        *tuple(ambiguous_ids),
    )
    if len(partition) != len(set(partition)) or tuple(sorted(partition)) != (
        interaction_ids
    ):
        raise ValueError("directional LR mapping report is not an exact partition")
    if (
        mapping_policy != expected_mapping_policy
        or mapping_report["mapping_policy"] != mapping_policy
        or mapping_report["resource_bundle_content_id"] != resource_content_id
        or mapping_report["target_prior_content_id"] != target_prior_content_id
        or _manifest_nonnegative_int(
            mapping_report["interaction_count"], field_name="interaction_count"
        )
        != len(interaction_ids)
        or _manifest_nonnegative_int(
            mapping_report["mapped_count"], field_name="mapped_count"
        )
        != len(mapped_interaction_ids)
        or _manifest_nonnegative_int(
            mapping_report["unmapped_count"], field_name="unmapped_count"
        )
        != len(unmapped_interaction_ids)
        or _manifest_nonnegative_int(
            mapping_report["ambiguous_count"], field_name="ambiguous_count"
        )
        != len(ambiguous_interactions)
    ):
        raise ValueError("directional LR mapping report metadata is inconsistent")
    mapping_report_payload = {
        "ambiguous_interactions": [
            {
                "candidate_driver_ids": entry["candidate_driver_ids"],
                "interaction_id": entry["interaction_id"],
            }
            for entry in ambiguous_interactions
        ],
        "interaction_ids": list(interaction_ids),
        "mapped_interaction_ids": list(mapped_interaction_ids),
        "mapping_policy": mapping_policy,
        "resource_bundle_content_id": resource_content_id,
        "target_prior_content_id": target_prior_content_id,
        "unmapped_interaction_ids": list(unmapped_interaction_ids),
    }
    mapping_report_id = _manifest_string(
        mapping_report["report_id"], field_name="mapping_report_id"
    )
    if mapping_report_id != stable_id(
        "receiver_family_lr_mapping_report",
        mapping_report_payload,
        schema_version="1",
    ):
        raise ValueError("directional LR mapping report identity is inconsistent")
    if is_v2:
        if molecular_contract is None:  # pragma: no cover - version branch invariant
            raise RuntimeError("v2 molecular LR contract is missing")
        molecular_interaction_ids = tuple(
            cast(tuple[str, ...], molecular_contract["interaction_ids"])
        )
        molecular_mapped = cast(
            dict[str, str], molecular_contract["mapped_by_interaction"]
        )
        unsupported_interaction_ids = cast(
            set[str], molecular_contract["unsupported_interaction_ids"]
        )
        if (
            interaction_ids != molecular_interaction_ids
            or unsupported_interaction_ids.intersection(mapped_interaction_ids)
            or not unsupported_interaction_ids.issubset(unmapped_interaction_ids)
            or any(
                interaction_id not in molecular_mapped
                for interaction_id in mapped_interaction_ids
            )
        ):
            raise ValueError(
                "directional LR mapping report differs from molecular support"
            )

    raw_memberships = universe["memberships"]
    if not isinstance(raw_memberships, list) or not raw_memberships:
        raise ValueError("directional LR memberships are invalid")
    membership_rows: list[tuple[str, str | None, str, str]] = []
    membership_fields = (
        _RECEIVER_FAMILY_LR_MEMBERSHIP_MANIFEST_FIELDS_V2
        if is_v2
        else _RECEIVER_FAMILY_LR_MEMBERSHIP_MANIFEST_FIELDS
    )
    for raw_membership in raw_memberships:
        membership = _manifest_exact_object(
            raw_membership,
            fields=membership_fields,
            field_name="directional LR membership",
        )
        interaction_id = _manifest_string(
            membership["interaction_id"], field_name="interaction_id"
        )
        molecular_id = (
            _manifest_string(
                membership["molecular_lr_equivalence_id"],
                field_name="molecular_lr_equivalence_id",
            )
            if is_v2
            else None
        )
        driver_id = _manifest_string(membership["driver_id"], field_name="driver_id")
        family_id = _manifest_string(membership["family_id"], field_name="family_id")
        if family_by_driver.get(driver_id) != family_id:
            raise ValueError("directional LR membership family is inconsistent")
        if is_v2 and (
            molecular_contract is None
            or cast(dict[str, str], molecular_contract["mapped_by_interaction"]).get(
                interaction_id
            )
            != molecular_id
        ):
            raise ValueError(
                "directional LR membership molecular equivalence is inconsistent"
            )
        membership_rows.append((interaction_id, molecular_id, driver_id, family_id))
    if (
        membership_rows != sorted(set(membership_rows))
        or tuple(item[0] for item in membership_rows) != mapped_interaction_ids
    ):
        raise ValueError("directional LR memberships do not cover mapped interactions")
    mapped_memberships = [
        (
            [interaction_id, cast(str, molecular_id), driver_id, family_id]
            if is_v2
            else [interaction_id, driver_id, family_id]
        )
        for interaction_id, molecular_id, driver_id, family_id in membership_rows
    ]
    membership_axis_payload: dict[str, object] = {
        "family_axis_id": expected_family_axis_id,
        "mapped_memberships": mapped_memberships,
        "mapping_policy": mapping_policy,
        "mapping_report_id": mapping_report_id,
        "resource_bundle_content_id": resource_content_id,
        "target_prior_content_id": target_prior_content_id,
    }
    if is_v2:
        if molecular_contract is None:  # pragma: no cover - version branch invariant
            raise RuntimeError("v2 molecular LR contract is missing")
        membership_axis_payload["molecular_lr_axis_id"] = molecular_contract[
            "molecular_lr_axis_id"
        ]
    expected_membership_axis_id = stable_id(
        "receiver_family_lr_membership_axis",
        membership_axis_payload,
        schema_version=identity_schema_version,
    )
    expected_memberships: list[dict[str, object]] = []
    for interaction_id, molecular_id, driver_id, family_id in membership_rows:
        membership_identity: dict[str, object] = {
            "driver_id": driver_id,
            "family_id": family_id,
            "interaction_id": interaction_id,
            "membership_axis_id": expected_membership_axis_id,
        }
        if is_v2:
            membership_identity["molecular_lr_equivalence_id"] = molecular_id
        expected_memberships.append(
            {
                "interaction_id": interaction_id,
                **({"molecular_lr_equivalence_id": molecular_id} if is_v2 else {}),
                "driver_id": driver_id,
                "family_id": family_id,
                "membership_id": stable_id(
                    "receiver_family_lr_membership",
                    membership_identity,
                    schema_version=identity_schema_version,
                ),
            }
        )
    if (
        _manifest_nonnegative_int(
            universe["membership_count"], field_name="membership_count"
        )
        != len(expected_memberships)
        or raw_memberships != expected_memberships
    ):
        raise ValueError("directional LR membership grid is inconsistent")

    modes = tuple(_manifest_ordered_string_array(universe["modes"], field_name="modes"))
    if modes != tuple(mode for mode in _COMMUNICATION_MODE_ORDER if mode in modes):
        raise ValueError("directional LR communication modes are not canonical")
    expected_mode_axis_id = stable_id(
        "communication_mode_axis",
        {"modes": list(modes)},
        schema_version="1",
    )
    expected_hypothesis_axis_id = stable_id(
        "receiver_family_lr_hypothesis_axis",
        {
            "membership_axis_id": expected_membership_axis_id,
            "mode_axis_id": expected_mode_axis_id,
        },
        schema_version=identity_schema_version,
    )
    expected_hypotheses = [
        {
            "interaction_id": membership["interaction_id"],
            "driver_id": membership["driver_id"],
            "family_id": membership["family_id"],
            **(
                {
                    "molecular_lr_equivalence_id": membership[
                        "molecular_lr_equivalence_id"
                    ]
                }
                if is_v2
                else {}
            ),
            "membership_id": membership["membership_id"],
            "mode": mode,
            "hypothesis_id": stable_id(
                "receiver_family_lr_hypothesis",
                {
                    "hypothesis_axis_id": expected_hypothesis_axis_id,
                    "membership_id": membership["membership_id"],
                    "mode": mode,
                },
                schema_version=identity_schema_version,
            ),
        }
        for membership in expected_memberships
        for mode in modes
    ]
    raw_hypotheses = universe["hypotheses"]
    hypothesis_fields = (
        _RECEIVER_FAMILY_LR_HYPOTHESIS_MANIFEST_FIELDS_V2
        if is_v2
        else _RECEIVER_FAMILY_LR_HYPOTHESIS_MANIFEST_FIELDS
    )
    if not isinstance(raw_hypotheses, list) or any(
        not isinstance(item, dict) or set(item) != hypothesis_fields
        for item in raw_hypotheses
    ):
        raise ValueError("directional LR hypotheses are invalid")
    if (
        _manifest_nonnegative_int(
            universe["hypothesis_count"], field_name="hypothesis_count"
        )
        != len(expected_hypotheses)
        or raw_hypotheses != expected_hypotheses
    ):
        raise ValueError("directional LR hypothesis grid is inconsistent")

    expected_opportunity_axis_id = stable_id(
        "receiver_family_lr_opportunity_axis",
        {
            "hypothesis_axis_id": expected_hypothesis_axis_id,
            "receiver_axis_id": receiver_axis_id,
        },
        schema_version=identity_schema_version,
    )
    expected_opportunities = [
        {
            "receiver": receiver,
            "interaction_id": hypothesis["interaction_id"],
            "driver_id": hypothesis["driver_id"],
            "family_id": hypothesis["family_id"],
            **(
                {
                    "molecular_lr_equivalence_id": hypothesis[
                        "molecular_lr_equivalence_id"
                    ]
                }
                if is_v2
                else {}
            ),
            "membership_id": hypothesis["membership_id"],
            "mode": hypothesis["mode"],
            "hypothesis_id": hypothesis["hypothesis_id"],
            "opportunity_id": stable_id(
                "receiver_family_lr_opportunity",
                {
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "opportunity_axis_id": expected_opportunity_axis_id,
                    "receiver": receiver,
                },
                schema_version=identity_schema_version,
            ),
        }
        for receiver in receiver_ids
        for hypothesis in expected_hypotheses
    ]
    raw_opportunities = universe["opportunities"]
    opportunity_fields = (
        _RECEIVER_FAMILY_LR_OPPORTUNITY_MANIFEST_FIELDS_V2
        if is_v2
        else _RECEIVER_FAMILY_LR_OPPORTUNITY_MANIFEST_FIELDS
    )
    if not isinstance(raw_opportunities, list) or any(
        not isinstance(item, dict) or set(item) != opportunity_fields
        for item in raw_opportunities
    ):
        raise ValueError("directional LR opportunities are invalid")
    if (
        _manifest_nonnegative_int(
            universe["opportunity_count"], field_name="opportunity_count"
        )
        != len(expected_opportunities)
        or raw_opportunities != expected_opportunities
    ):
        raise ValueError("directional LR opportunity grid is inconsistent")

    child_identity_payload = {
        "family_axis_id": expected_family_axis_id,
        "feature_axis_id": expected_feature_axis_id,
        "hypothesis_axis_id": expected_hypothesis_axis_id,
        "mapping_policy": mapping_policy,
        "mapping_report_id": mapping_report_id,
        "membership_axis_id": expected_membership_axis_id,
        "mode_axis_id": expected_mode_axis_id,
        "modes": list(modes),
        "opportunity_axis_id": expected_opportunity_axis_id,
        "receiver_axis_id": receiver_axis_id,
        "receiver_family_universe_id": parent_universe_id,
        "receiver_universe_id": receiver_universe_id,
        "root_input_digest": parent_root_digest,
        "root_input_identity_id": parent_root_identity_id,
        "resource_bundle_content_id": resource_content_id,
        "resource_id": resource_id,
        "resource_manifest_digest": resource_manifest_digest,
        "resource_version": resource_version,
        "schema_version": universe["schema_version"],
        "source_policy": universe["source_policy"],
        "target_prior_content_id": target_prior_content_id,
        "target_prior_manifest_digest": _manifest_string(
            universe["target_prior_manifest_digest"],
            field_name="target_prior_manifest_digest",
        ),
        "target_prior_resource_id": _manifest_string(
            universe["target_prior_resource_id"],
            field_name="target_prior_resource_id",
        ),
        "target_prior_version": _manifest_string(
            universe["target_prior_version"], field_name="target_prior_version"
        ),
    }
    if is_v2:
        if molecular_contract is None:  # pragma: no cover - version branch invariant
            raise RuntimeError("v2 molecular LR contract is missing")
        child_identity_payload.update(
            {
                "molecular_lr_axis_id": molecular_contract["molecular_lr_axis_id"],
                "molecular_lr_equivalence_universe_id": molecular_contract[
                    "universe_id"
                ],
            }
        )
    if (
        universe["schema_version"] != expected_universe_schema_version
        or universe["source_policy"] != expected_source_policy
        or universe["mapping_report_id"] != mapping_report_id
        or universe["membership_axis_id"] != expected_membership_axis_id
        or universe["mode_axis_id"] != expected_mode_axis_id
        or universe["hypothesis_axis_id"] != expected_hypothesis_axis_id
        or universe["opportunity_axis_id"] != expected_opportunity_axis_id
        or universe["receiver_family_universe_id"] != parent_universe_id
        or universe["receiver_universe_id"] != receiver_universe_id
        or universe["receiver_axis_id"] != receiver_axis_id
        or universe["feature_axis_id"] != expected_feature_axis_id
        or universe["family_axis_id"] != expected_family_axis_id
        or universe["root_input_identity_id"] != parent_root_identity_id
        or universe["root_input_digest"] != parent_root_digest
        or universe["receiver_ids"] != list(receiver_ids)
        or universe["family_ids"] != list(family_ids)
        or child_identity_payload["target_prior_resource_id"] != prior_resource_id
        or child_identity_payload["target_prior_version"] != prior_version
        or child_identity_payload["target_prior_manifest_digest"]
        != prior_manifest_digest
        or target_prior_content_id != prior_content_id
        or (
            is_v2
            and (
                molecular_contract is None
                or universe["molecular_lr_equivalence_universe_id"]
                != molecular_contract["universe_id"]
                or universe["molecular_lr_axis_id"]
                != molecular_contract["molecular_lr_axis_id"]
            )
        )
    ):
        raise ValueError("directional LR hypothesis universe lineage is inconsistent")
    child_universe_id = _manifest_string(
        universe["universe_id"], field_name="directional_lr_hypothesis_universe_id"
    )
    if child_universe_id != stable_id(
        "frozen_receiver_family_lr_hypothesis_universe",
        child_identity_payload,
        schema_version=identity_schema_version,
    ):
        raise ValueError("directional LR hypothesis universe identity is inconsistent")


def _manifest_nonnegative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _manifest_unit_knots(value: object, *, field_name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    result: list[float] = []
    for raw in value:
        numeric = _manifest_float(raw, field_name=field_name)
        if numeric is None or not 0.0 <= numeric <= 1.0:
            raise ValueError(f"{field_name} values must lie in [0, 1]")
        result.append(numeric)
    return result


def _validate_gain_calibration_binding(
    raw: object,
    *,
    expected_receiver: str,
    expected_family_functional_id: str,
) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != _GAIN_CALIBRATION_BINDING_FIELDS:
        raise ValueError("receiver gain calibration binding is invalid")
    binding = cast(dict[str, Any], raw)
    binding_id = _manifest_string(
        binding["gain_calibration_binding_id"],
        field_name="gain_calibration_binding_id",
    )
    receiver = _manifest_string(binding["receiver"], field_name="receiver")
    family_functional_id = _manifest_string(
        binding["source_family_common_functional_id"],
        field_name="source_family_common_functional_id",
    )
    if (
        receiver != expected_receiver
        or family_functional_id != expected_family_functional_id
    ):
        raise ValueError("gain calibration binding parent lineage is inconsistent")
    artifact_id = _manifest_optional_string(
        binding["gain_calibration_artifact_id"],
        field_name="gain_calibration_artifact_id",
    )
    calibration_spec_id = _manifest_optional_string(
        binding["gain_calibration_spec_id"],
        field_name="gain_calibration_spec_id",
    )
    status = _manifest_string(
        binding["gain_calibration_status"], field_name="gain_calibration_status"
    )
    reason = _manifest_optional_string(
        binding["gain_calibration_reason_code"],
        field_name="gain_calibration_reason_code",
    )
    percentile_policy = _manifest_string(
        binding["percentile_policy"], field_name="percentile_policy"
    )
    if percentile_policy != GAIN_CALIBRATION_PERCENTILE_POLICY:
        raise ValueError("gain calibration percentile policy is inconsistent")
    source_knots = _manifest_unit_knots(
        binding["positive_gain_source_knots"],
        field_name="positive_gain_source_knots",
    )
    percentile_knots = _manifest_unit_knots(
        binding["positive_gain_percentile_knots"],
        field_name="positive_gain_percentile_knots",
    )
    if len(source_knots) != len(percentile_knots):
        raise ValueError("gain calibration knot arrays do not align")
    supported = _manifest_nonnegative_int(
        binding["n_supported_families"], field_name="n_supported_families"
    )
    positives = _manifest_nonnegative_int(
        binding["n_positive_observations"], field_name="n_positive_observations"
    )
    distinct = _manifest_nonnegative_int(
        binding["n_distinct_positive_gains"],
        field_name="n_distinct_positive_gains",
    )
    for field_name in (
        "tuning_id",
        "outer_incremental_functional_id",
        "outer_selected_resolved_penalty_id",
    ):
        _manifest_optional_string(binding[field_name], field_name=field_name)
    if status == "observed":
        if (
            reason is not None
            or artifact_id is None
            or calibration_spec_id is None
            or len(source_knots) < 2
            or source_knots[0] != 0.0
            or percentile_knots[0] != 0.0
            or not math.isclose(percentile_knots[-1], 1.0)
            or any(right <= left for left, right in pairwise(source_knots))
            or any(right <= left for left, right in pairwise(percentile_knots))
            or supported <= 0
            or positives < distinct
            or distinct != len(source_knots) - 1
        ):
            raise ValueError("observed gain calibration binding is invalid")
    elif status == "not_estimable":
        if (
            reason is None
            or source_knots
            or percentile_knots
            or (artifact_id is None) != (calibration_spec_id is None)
        ):
            raise ValueError("not-estimable gain calibration binding is invalid")
    else:
        raise ValueError("gain calibration binding status is not recognized")
    payload = {
        key: value
        for key, value in binding.items()
        if key != "gain_calibration_binding_id"
    }
    if binding_id != stable_id(
        "receiver_gain_calibration_binding", payload, schema_version="1"
    ):
        raise ValueError("gain calibration binding identity is inconsistent")
    return binding


def _validate_contrast_common_collections_manifest(
    manifest: dict[str, Any],
) -> None:
    schema_version = manifest.get("schema_version", CROSSFIT_RESULT_SCHEMA_VERSION)
    if schema_version == _V3_CROSSFIT_RESULT_SCHEMA_VERSION:
        _validate_v3_contrast_common_collections_manifest(manifest)
        return
    if schema_version not in {
        CROSSFIT_RESULT_SCHEMA_VERSION,
        _V8_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V7_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V6_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V5_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V4_CROSSFIT_RESULT_SCHEMA_VERSION,
    }:
        raise ValueError("contrast-common collection schema is unsupported")
    raw_collections = manifest["contrast_common_collections"]
    if not isinstance(raw_collections, list):
        raise ValueError("contrast-common collection registry must be an array")
    collection_ids: list[str] = []
    collection_scopes: list[tuple[str, str]] = []
    application_ids: list[str] = []
    functional_ids: list[str] = []
    canonical_collection_order: list[tuple[str, str]] = []
    is_oof_certified = bool(manifest["complete_pipeline_oof_certified"])
    for collection in raw_collections:
        if (
            not isinstance(collection, dict)
            or set(collection) != _CONTRAST_COMMON_COLLECTION_FIELDS
        ):
            raise ValueError("contrast-common collection entry is invalid")
        collection_id = _manifest_string(
            collection["contrast_common_collection_id"],
            field_name="contrast_common_collection_id",
        )
        contrast_id = _manifest_string(
            collection["contrast_id"], field_name="contrast_id"
        )
        contrast = _manifest_string(collection["contrast"], field_name="contrast")
        for field_name in (
            "crossfit_id",
            "spec_id",
            "repeat_id",
            "score_version",
            "estimand",
            "certification_status",
            "claim_scope",
        ):
            _manifest_string(collection[field_name], field_name=field_name)
        if (
            collection["crossfit_id"] != manifest["crossfit_id"]
            or collection["spec_id"] != manifest["spec_id"]
            or collection["repeat_id"] != manifest["repeat_id"]
            or collection["certification_status"] != manifest["certification_status"]
            or collection["is_oof_certified"] is not is_oof_certified
            or collection["claim_scope"]
            != _expected_contrast_common_claim_scope(is_oof_certified)
            or collection["common_functional_across_receivers"] is not False
            or collection["receiver_balanced_descriptive_collection"] is not True
            or collection["formal_inference_allowed"] is not False
        ):
            raise ValueError("contrast-common collection scope is inconsistent")
        fold_applications = collection["fold_applications"]
        if not isinstance(fold_applications, list) or not fold_applications:
            raise ValueError("contrast-common collection requires fold applications")
        fold_ids: list[str] = []
        for application in fold_applications:
            if (
                not isinstance(application, dict)
                or set(application) != _CONTRAST_COMMON_APPLICATION_FIELDS
            ):
                raise ValueError("contrast-common fold application entry is invalid")
            for field_name in (
                "fold_id",
                "global_common_functional_id",
                "global_common_application_id",
                "functional_spec_id",
                "functional_schema_version",
                "filter_universe_id",
                "sender_functional_id",
                "calibration_policy",
                "scale_policy",
                "sender_policy",
                "interaction_mapping_digest",
                "functional_certification_status",
                "lr_scores_digest",
                "sender_scores_digest",
            ):
                _manifest_string(application[field_name], field_name=field_name)
            fold_id = str(application["fold_id"])
            fold_ids.append(fold_id)
            application_ids.append(str(application["global_common_application_id"]))
            functional_ids.append(str(application["global_common_functional_id"]))
            _manifest_string_array(application["context_ids"], field_name="context_ids")
            receiver_ids = _manifest_string_array(
                application["receiver_ids"], field_name="receiver_ids"
            )
            softmin_power = _manifest_float(
                application["softmin_power"], field_name="softmin_power"
            )
            epsilon = _manifest_float(application["epsilon"], field_name="epsilon")
            if (
                softmin_power is None
                or softmin_power <= 0.0
                or epsilon is None
                or epsilon <= 0.0
            ):
                raise ValueError("soft-min parameters must be positive")
            training_subject_ids = _manifest_string_array(
                application["training_subject_ids"],
                field_name="training_subject_ids",
            )
            heldout_subject_ids = _manifest_string_array(
                application["heldout_subject_ids"], field_name="heldout_subject_ids"
            )
            if set(training_subject_ids).intersection(heldout_subject_ids):
                raise ValueError(
                    "contrast-common training and held-out subjects overlap"
                )
            children = application["receiver_children"]
            if not isinstance(children, list):
                raise ValueError("contrast-common receiver children must be an array")
            child_receivers: list[str] = []
            child_by_receiver: dict[str, dict[str, Any]] = {}
            for child in children:
                if (
                    not isinstance(child, dict)
                    or set(child) != _CONTRAST_COMMON_CHILD_FIELDS
                ):
                    raise ValueError("contrast-common receiver child is invalid")
                for field_name in _CONTRAST_COMMON_CHILD_FIELDS:
                    _manifest_string(child[field_name], field_name=field_name)
                receiver = str(child["receiver"])
                child_receivers.append(receiver)
                child_by_receiver[receiver] = cast(dict[str, Any], child)
            if child_receivers != receiver_ids:
                raise ValueError(
                    "contrast-common receiver children lack exact planned coverage"
                )
            raw_bindings = application["receiver_gain_calibration_bindings"]
            if not isinstance(raw_bindings, list):
                raise ValueError("receiver gain calibration bindings must be an array")
            bindings = [
                _validate_gain_calibration_binding(
                    raw_binding,
                    expected_receiver=receiver,
                    expected_family_functional_id=str(
                        child_by_receiver[receiver]["family_common_functional_id"]
                    ),
                )
                for receiver, raw_binding in zip(
                    receiver_ids, raw_bindings, strict=True
                )
            ]
            if (
                len(bindings) != len(receiver_ids)
                or [str(binding["receiver"]) for binding in bindings] != receiver_ids
            ):
                raise ValueError(
                    "gain calibration bindings lack exact receiver coverage"
                )
            all_calibrated = all(
                binding["gain_calibration_status"] == "observed" for binding in bindings
            )
            if (
                type(application["all_receivers_gain_calibrated"]) is not bool
                or application["all_receivers_gain_calibrated"] is not all_calibrated
                or type(application["cross_receiver_percentile_rank_eligible"])
                is not bool
                or application["cross_receiver_percentile_rank_eligible"]
                is not all_calibrated
                or application["training_only_receiver_calibration"] is not True
                or application["receiver_scale_amplification"] is not False
                or application["common_functional_across_receivers"] is not False
                or application["receiver_balanced_descriptive_collection"] is not True
            ):
                raise ValueError(
                    "receiver gain calibration eligibility flags are invalid"
                )
            sender_lineages = application["sender_lineages"]
            if not isinstance(sender_lineages, list) or not sender_lineages:
                raise ValueError("contrast-common sender lineage must be non-empty")
            lineage_keys: list[tuple[str, str, str]] = []
            for lineage in sender_lineages:
                if (
                    not isinstance(lineage, dict)
                    or set(lineage) != _CONTRAST_COMMON_SENDER_LINEAGE_FIELDS
                ):
                    raise ValueError("contrast-common sender lineage is invalid")
                for field_name in _CONTRAST_COMMON_SENDER_LINEAGE_FIELDS:
                    _manifest_string(lineage[field_name], field_name=field_name)
                if (
                    lineage["sender_functional_id"]
                    != application["sender_functional_id"]
                ):
                    raise ValueError(
                        "contrast-common sender functional lineage is inconsistent"
                    )
                lineage_keys.append(
                    (
                        str(lineage["receiver"]),
                        str(lineage["sender_functional_id"]),
                        str(lineage["sender_application_id"]),
                    )
                )
            if lineage_keys != sorted(set(lineage_keys)) or {
                key[0] for key in lineage_keys
            } != set(receiver_ids):
                raise ValueError(
                    "contrast-common sender lineage is noncanonical or incomplete"
                )
            sender_digests = application["sender_application_digests"]
            if not isinstance(sender_digests, list):
                raise ValueError("contrast-common sender digests must be an array")
            digest_receivers: list[str] = []
            for sender_digest in sender_digests:
                if (
                    not isinstance(sender_digest, dict)
                    or set(sender_digest) != _CONTRAST_COMMON_SENDER_DIGEST_FIELDS
                ):
                    raise ValueError("contrast-common sender digest is invalid")
                for field_name in _CONTRAST_COMMON_SENDER_DIGEST_FIELDS:
                    _manifest_string(sender_digest[field_name], field_name=field_name)
                digest_receivers.append(str(sender_digest["receiver"]))
            if digest_receivers != receiver_ids:
                raise ValueError(
                    "contrast-common sender digests lack exact receiver coverage"
                )
            for count_field in ("n_lr_rows", "n_sender_rows"):
                count = application[count_field]
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise ValueError(
                        "contrast-common application row counts must be positive"
                    )
        if fold_ids != sorted(set(fold_ids)):
            raise ValueError(
                "contrast-common fold applications must be canonically ordered"
            )
        payload = {
            key: value
            for key, value in collection.items()
            if key != "contrast_common_collection_id"
        }
        if collection_id != stable_id(
            "contrast_common_oof_score_collection", payload, schema_version="1"
        ):
            raise ValueError("contrast-common collection identity is inconsistent")
        collection_ids.append(collection_id)
        collection_scopes.append((contrast_id, contrast))
        canonical_collection_order.append((contrast, contrast_id))
    if (
        len(collection_ids) != len(set(collection_ids))
        or len(collection_scopes) != len(set(collection_scopes))
        or len(application_ids) != len(set(application_ids))
        or len(functional_ids) != len(set(functional_ids))
        or canonical_collection_order != sorted(canonical_collection_order)
    ):
        raise ValueError("contrast-common collection registry is not unique/canonical")
    if manifest["contrast_common_stage_connected"] is not bool(raw_collections):
        raise ValueError(
            "contrast-common stage flag disagrees with its collection registry"
        )


def _manifest_payload(manifest: dict[str, object]) -> dict[str, object]:
    return {
        key: value for key, value in manifest.items() if key != "crossfit_result_id"
    }


_LATENT_ARTIFACT_FIELDS = frozenset(
    {
        "algorithm_contract",
        "artifact_id",
        "receiver",
        "contrast_name",
        "fold_id",
        "status",
        "reason_code",
        "spec_id",
        "feature_ids",
        "control_feature_ids",
        "training_sample_ids",
        "training_sample_subject_ids",
        "training_sample_context_ids",
        "training_subject_ids",
        "training_row_manifest_id",
        "program_ids",
        "static_coordinate_basis_id",
        "static_program_ids",
        "response_digest",
        "family_basis_digest",
        "nuisance_digest",
        "training_weights_digest",
        "feature_center_digest",
        "feature_scale_digest",
        "precision_weights_digest",
        "static_coordinate_basis_digest",
        "training_input_digest",
        "control_score_input_digest",
        "control_residual_digest",
        "coordinate_basis_digest",
        "factor_scores_digest",
        "control_score_basis_digest",
        "singular_values_digest",
        "explained_variance_fractions_digest",
        "retained_explained_fraction",
        "control_numerical_rank",
        "static_numerical_rank",
        "static_control_numerical_rank",
        "static_orthogonality_max_abs",
        "family_count",
        "nuisance_column_count",
    }
)


def _validated_latent_nuisance_spec(raw: object) -> FrozenLatentNuisanceSpec:
    if not isinstance(raw, Mapping):
        raise ValueError("source latent nuisance specification is invalid")
    expected = {
        "algorithm_contract",
        "max_components",
        "min_control_features",
        "min_training_subjects",
        "minimum_explained_fraction",
        "spec_id",
        "svd_rcond",
    }
    if set(raw) != expected:
        raise ValueError("source latent nuisance specification fields are invalid")
    try:
        spec = FrozenLatentNuisanceSpec(
            max_components=raw["max_components"],
            min_control_features=raw["min_control_features"],
            min_training_subjects=raw["min_training_subjects"],
            svd_rcond=raw["svd_rcond"],
            minimum_explained_fraction=raw["minimum_explained_fraction"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError("source latent nuisance specification is invalid") from error
    if spec.to_dict() != dict(raw):
        raise ValueError("source latent nuisance specification identity is invalid")
    return spec


def _validated_manifest_string_list(
    raw: object,
    *,
    field_name: str,
    unique: bool = False,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be an array")
    values = [_manifest_string(item, field_name=field_name) for item in raw]
    if not allow_empty and not values:
        raise ValueError(f"{field_name} must not be empty")
    if unique and len(values) != len(set(values)):
        raise ValueError(f"{field_name} must contain unique values")
    return values


def _validate_latent_nuisance_artifact(
    raw: object,
    *,
    spec: FrozenLatentNuisanceSpec,
    fold_id: str,
    receiver: str,
    contrast_name: str,
    training_subject_ids: list[str],
    static_resource_id: str | None,
    static_program_ids: list[str],
) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != _LATENT_ARTIFACT_FIELDS:
        raise ValueError("source latent nuisance artifact fields are invalid")
    artifact = cast(dict[str, Any], raw)
    artifact_id = _manifest_string(
        artifact["artifact_id"], field_name="latent_nuisance_artifact_id"
    )
    observed_scope = (
        _manifest_string(artifact["fold_id"], field_name="latent.fold_id"),
        _manifest_string(artifact["receiver"], field_name="latent.receiver"),
        _manifest_string(artifact["contrast_name"], field_name="latent.contrast_name"),
        _manifest_string(artifact["spec_id"], field_name="latent.spec_id"),
    )
    if observed_scope != (fold_id, receiver, contrast_name, spec.spec_id):
        raise ValueError("source latent nuisance artifact scope is inconsistent")
    features = _validated_manifest_string_list(
        artifact["feature_ids"], field_name="latent.feature_ids", unique=True
    )
    controls = _validated_manifest_string_list(
        artifact["control_feature_ids"],
        field_name="latent.control_feature_ids",
        unique=True,
        allow_empty=True,
    )
    if not set(controls).issubset(features):
        raise ValueError("latent nuisance controls are outside the feature universe")
    sample_ids = _validated_manifest_string_list(
        artifact["training_sample_ids"],
        field_name="latent.training_sample_ids",
        unique=True,
    )
    sample_subjects = _validated_manifest_string_list(
        artifact["training_sample_subject_ids"],
        field_name="latent.training_sample_subject_ids",
    )
    sample_contexts = _validated_manifest_string_list(
        artifact["training_sample_context_ids"],
        field_name="latent.training_sample_context_ids",
    )
    subjects = _validated_manifest_string_list(
        artifact["training_subject_ids"],
        field_name="latent.training_subject_ids",
        unique=True,
    )
    if (
        not sample_ids
        or len(sample_subjects) != len(sample_ids)
        or len(sample_contexts) != len(sample_ids)
        or subjects != sorted(set(sample_subjects))
        or subjects != training_subject_ids
    ):
        raise ValueError("latent nuisance training rows are inconsistent")
    expected_row_manifest_id = stable_id(
        "latent_nuisance_training_rows",
        {
            "rows": [
                {
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                    "context_id": context_id,
                }
                for sample_id, subject_id, context_id in zip(
                    sample_ids,
                    sample_subjects,
                    sample_contexts,
                    strict=True,
                )
            ]
        },
        schema_version="1",
    )
    if artifact["training_row_manifest_id"] != expected_row_manifest_id:
        raise ValueError("latent nuisance training-row identity is invalid")
    programs = _validated_manifest_string_list(
        artifact["program_ids"],
        field_name="latent.program_ids",
        unique=True,
        allow_empty=True,
    )
    artifact_static_programs = _validated_manifest_string_list(
        artifact["static_program_ids"],
        field_name="latent.static_program_ids",
        unique=True,
        allow_empty=True,
    )
    static_basis_id = _manifest_optional_string(
        artifact["static_coordinate_basis_id"],
        field_name="latent.static_coordinate_basis_id",
    )
    if artifact_static_programs != static_program_ids or static_basis_id != (
        static_resource_id
    ):
        raise ValueError("latent nuisance static parent lineage is inconsistent")
    status = _manifest_string(artifact["status"], field_name="latent.status")
    reason = _manifest_optional_string(
        artifact["reason_code"], field_name="latent.reason_code"
    )
    if (status == "observed" and (reason is not None or not programs)) or (
        status == "not_estimable" and (reason is None or programs)
    ):
        raise ValueError("latent nuisance artifact status is inconsistent")
    if status not in {"observed", "not_estimable"}:
        raise ValueError("latent nuisance artifact status is unsupported")
    if status == "observed":
        expected_program_ids = [
            stable_id(
                "latent_nuisance_program",
                {
                    "component_index": component,
                    "control_score_input_digest": artifact[
                        "control_score_input_digest"
                    ],
                    "contrast_name": contrast_name,
                    "fold_id": fold_id,
                    "receiver": receiver,
                    "spec_id": spec.spec_id,
                },
                schema_version="1",
            )
            for component in range(len(programs))
        ]
        if programs != expected_program_ids or len(programs) > spec.max_components:
            raise ValueError("latent nuisance program identities are invalid")
    for field_name in (
        "family_count",
        "nuisance_column_count",
        "control_numerical_rank",
        "static_numerical_rank",
        "static_control_numerical_rank",
    ):
        _manifest_nonnegative_int(
            artifact[field_name], field_name=f"latent.{field_name}"
        )
    retained = _manifest_float(
        artifact["retained_explained_fraction"],
        field_name="latent.retained_explained_fraction",
    )
    orthogonality = _manifest_float(
        artifact["static_orthogonality_max_abs"],
        field_name="latent.static_orthogonality_max_abs",
    )
    if (
        retained is None
        or not 0.0 <= retained <= 1.0
        or orthogonality is None
        or orthogonality < 0.0
    ):
        raise ValueError("latent nuisance numerical diagnostics are invalid")
    expected_training_input_digest = stable_id(
        "latent_nuisance_training_input",
        {
            "row_manifest_id": expected_row_manifest_id,
            "feature_ids": features,
            "response_digest": artifact["response_digest"],
            "family_basis_digest": artifact["family_basis_digest"],
            "nuisance_digest": artifact["nuisance_digest"],
            "training_weights_digest": artifact["training_weights_digest"],
            "feature_center_digest": artifact["feature_center_digest"],
            "feature_scale_digest": artifact["feature_scale_digest"],
            "precision_weights_digest": artifact["precision_weights_digest"],
            "static_coordinate_basis_digest": artifact[
                "static_coordinate_basis_digest"
            ],
            "static_coordinate_basis_id": static_basis_id,
            "static_program_ids": artifact_static_programs,
            "family_count": artifact["family_count"],
            "nuisance_column_count": artifact["nuisance_column_count"],
        },
        schema_version="1",
        digest_length=64,
    )
    if artifact["training_input_digest"] != expected_training_input_digest:
        raise ValueError("latent nuisance training-input identity is invalid")
    payload = {key: value for key, value in artifact.items() if key != "artifact_id"}
    if artifact_id != stable_id(
        "fold_latent_nuisance_artifact", payload, schema_version="1"
    ):
        raise ValueError("latent nuisance artifact identity is invalid")
    return artifact


def _validate_source_latent_nuisance_lineage(
    source: dict[str, Any],
    *,
    expected_crossfit_spec_id: str,
) -> None:
    raw_crossfit_spec = source.get("spec")
    if not isinstance(raw_crossfit_spec, Mapping):
        raise ValueError("source cross-fit specification is invalid")
    raw_latent_spec = raw_crossfit_spec.get("latent_nuisance_spec")
    if raw_latent_spec is None:
        return
    latent_spec = _validated_latent_nuisance_spec(raw_latent_spec)
    if raw_crossfit_spec.get("spec_id") != expected_crossfit_spec_id or (
        latent_spec.spec_id != raw_latent_spec.get("spec_id")
    ):
        raise ValueError("source latent nuisance specification linkage is invalid")
    raw_resource = raw_crossfit_spec.get("autonomous_program_resource")
    if raw_resource is None:
        static_resource_id = None
        static_verification_status = None
        static_program_ids: list[str] = []
    else:
        if not isinstance(raw_resource, Mapping):
            raise ValueError("source autonomous program resource is invalid")
        static_resource_id = _manifest_string(
            raw_resource.get("artifact_id"),
            field_name="autonomous_program_resource.artifact_id",
        )
        static_verification_status = _manifest_string(
            raw_resource.get("verification_status"),
            field_name="autonomous_program_resource.verification_status",
        )
        static_program_ids = _validated_manifest_string_list(
            raw_resource.get("program_ids"),
            field_name="autonomous_program_resource.program_ids",
            unique=True,
        )
    folds = source.get("fold_artifacts")
    if not isinstance(folds, list) or not folds:
        raise ValueError("source cross-fit fold registry is invalid")
    for raw_fold in folds:
        if not isinstance(raw_fold, Mapping):
            raise ValueError("source cross-fit fold record is invalid")
        fold_id = _manifest_string(raw_fold.get("fold_id"), field_name="fold_id")
        training_subject_ids = _validated_manifest_string_list(
            raw_fold.get("training_subject_ids"),
            field_name="fold.training_subject_ids",
            unique=True,
        )
        records = raw_fold.get("receiver_incremental_artifacts")
        if not isinstance(records, list):
            raise ValueError("source receiver incremental registry is invalid")
        for raw_record in records:
            if not isinstance(raw_record, Mapping):
                raise ValueError("source receiver incremental record is invalid")
            record = cast(Mapping[str, object], raw_record)
            receiver = _manifest_string(record.get("receiver"), field_name="receiver")
            contrast_name = _manifest_string(
                record.get("contrast_name"), field_name="contrast_name"
            )
            if record.get("latent_nuisance_spec_id") != latent_spec.spec_id or (
                record.get("autonomous_program_resource_id") != static_resource_id
            ):
                raise ValueError("receiver latent nuisance parent linkage is invalid")
            raw_artifact = record.get("latent_nuisance_artifact")
            if raw_artifact is None:
                artifact = None
                artifact_id = None
            else:
                artifact = _validate_latent_nuisance_artifact(
                    raw_artifact,
                    spec=latent_spec,
                    fold_id=fold_id,
                    receiver=receiver,
                    contrast_name=contrast_name,
                    training_subject_ids=training_subject_ids,
                    static_resource_id=static_resource_id,
                    static_program_ids=static_program_ids,
                )
                artifact_id = artifact["artifact_id"]
            if record.get("latent_nuisance_artifact_id") != artifact_id:
                raise ValueError("receiver latent nuisance artifact linkage is invalid")
            observed_latent = artifact is not None and artifact["status"] == "observed"
            expected_source_id: object
            expected_verification: str | None
            if observed_latent and static_resource_id is None:
                expected_source_id = artifact_id
                expected_verification = (
                    "outer_training_control_feature_latent_nested_oof_v1"
                )
            elif observed_latent:
                expected_source_id = record.get("autonomous_program_source_id")
                expected_verification = (
                    "manifest_verified_static_plus_control_feature_latent_nested_oof_v1"
                    if static_verification_status
                    == "manifest_verified_static_trusted_v1"
                    else (
                        "caller_declared_static_plus_control_feature_latent_"
                        "unverified_v1"
                    )
                )
            else:
                expected_source_id = static_resource_id
                expected_verification = static_verification_status
            if (
                record.get("autonomous_program_source_id") != expected_source_id
                or record.get("autonomous_program_verification_status")
                != expected_verification
            ):
                raise ValueError("receiver latent nuisance source linkage is invalid")
            if observed_latent and not isinstance(expected_source_id, str):
                raise ValueError("observed latent nuisance source identity is absent")


def _validate_response_backend_manifest(
    raw: object,
    *,
    fold_id: str,
    receiver: str,
    contrast_name: str,
    response_artifact_id: str,
) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != _RESPONSE_BACKEND_MANIFEST_FIELDS:
        raise ValueError("response backend manifest fields are invalid")
    backend = cast(dict[str, Any], raw)
    if backend["schema_version"] != "1.0.0" or (
        backend["formal_inference_allowed"] is not False
    ):
        raise ValueError("response backend manifest policy is invalid")
    for field_name, expected in (
        ("fold_id", fold_id),
        ("receiver", receiver),
        ("contrast_name", contrast_name),
        ("response_artifact_id", response_artifact_id),
    ):
        if _manifest_string(backend[field_name], field_name=field_name) != expected:
            raise ValueError("response backend manifest scope is inconsistent")
    artifact_kind = _manifest_string(
        backend["artifact_kind"], field_name="artifact_kind"
    )
    method = _manifest_string(backend["method"], field_name="response.method")
    status = _manifest_string(backend["status"], field_name="response.status")
    if status not in {"ok", "exploratory", "not_estimable"}:
        raise ValueError("response backend status is unsupported")
    reason = backend["reason_code"]
    if reason is not None:
        _manifest_string(reason, field_name="response.reason_code")
    feature_ids = _validated_manifest_string_list(
        backend["feature_ids"],
        field_name="response.feature_ids",
        unique=True,
    )
    for field_name in (
        "n_features",
        "n_backend_eligible_features",
        "n_precision_supported_features",
    ):
        value = backend[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"response backend {field_name} is invalid")
    if backend["n_features"] != len(feature_ids):
        raise ValueError("response backend feature count is inconsistent")
    _manifest_string(
        backend["precision_parent_raw_digest"],
        field_name="response.precision_parent_raw_digest",
    )
    raw_supported = backend["precision_supported_feature_indices"]
    if not isinstance(raw_supported, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw_supported
    ):
        raise ValueError("response precision-support indices are invalid")
    supported = cast(list[int], raw_supported)
    if (
        supported != sorted(set(supported))
        or any(index < 0 or index >= len(feature_ids) for index in supported)
        or backend["n_precision_supported_features"] != len(supported)
    ):
        raise ValueError("response precision-support registry is inconsistent")

    identity = backend["response_identity"]
    if not isinstance(identity, dict):
        raise ValueError("response identity payload is invalid")
    if (
        identity.get("method") != method
        or identity.get("status") != status
        or identity.get("reason_code") != reason
    ):
        raise ValueError("response identity payload disagrees with backend fields")
    if artifact_kind == "fold_gene_response_v2":
        artifact_namespace = "fold_gene_response"
        artifact_schema_version = "2"
        expected_precision_method = _FOLD_RESPONSE_PRECISION_METHOD
        expected_precision_lineage = _FOLD_RESPONSE_PRECISION_LINEAGE
        if backend["repeated_design_manifest"] is not None:
            raise ValueError("ordinary fold response cannot carry a repeated design")
    elif artifact_kind in {
        "repeated_cr1_fold_response_v1",
        "repeated_cr2_fold_response_v1",
    }:
        artifact_namespace = "repeated_measures_fold_response"
        artifact_schema_version = "1"
        expected_method = (
            _CR1_REPEATED_RESPONSE_METHOD
            if artifact_kind == "repeated_cr1_fold_response_v1"
            else _CR2_REPEATED_RESPONSE_METHOD
        )
        expected_precision_method = (
            _CR1_REPEATED_PRECISION_METHOD
            if artifact_kind == "repeated_cr1_fold_response_v1"
            else _CR2_REPEATED_PRECISION_METHOD
        )
        expected_precision_lineage = (
            _CR1_REPEATED_PRECISION_LINEAGE
            if artifact_kind == "repeated_cr1_fold_response_v1"
            else _CR2_REPEATED_PRECISION_LINEAGE
        )
        if method != expected_method:
            raise ValueError("repeated response method disagrees with artifact kind")
        repeated_effect_id = _manifest_string(
            identity.get("repeated_effect_artifact_id"),
            field_name="repeated_effect_artifact_id",
        )
        expected_effect_prefix = (
            "repeated_measures_receiver_effect_"
            if artifact_kind == "repeated_cr1_fold_response_v1"
            else "repeated_measures_cr2_receiver_result_"
        )
        if not repeated_effect_id.startswith(expected_effect_prefix):
            raise ValueError(
                "repeated response effect backend identity is inconsistent"
            )
        design = backend["repeated_design_manifest"]
        if not isinstance(design, dict) or design.get("design_id") != identity.get(
            "repeated_design_id"
        ):
            raise ValueError("repeated response design lineage is inconsistent")
        if not isinstance(design.get("spec"), dict):
            raise ValueError("repeated response design policy is absent")
    else:
        raise ValueError("response artifact kind is unsupported")
    if response_artifact_id != stable_id(
        artifact_namespace,
        identity,
        schema_version=artifact_schema_version,
    ):
        raise ValueError("response artifact identity replay failed")

    diagnostics = backend["feature_diagnostics"]
    if not isinstance(diagnostics, list):
        raise ValueError("response feature diagnostics must be an array")
    repeated = artifact_kind != "fold_gene_response_v2"
    if (not repeated and diagnostics) or (
        repeated and len(diagnostics) != len(feature_ids)
    ):
        raise ValueError("response feature diagnostic coverage is inconsistent")
    supported_set = set(supported)
    n_backend_eligible = 0
    for index, raw_diagnostic in enumerate(diagnostics):
        if (
            not isinstance(raw_diagnostic, dict)
            or set(raw_diagnostic) != _RESPONSE_FEATURE_DIAGNOSTIC_FIELDS
        ):
            raise ValueError("response feature diagnostic fields are invalid")
        diagnostic = cast(dict[str, Any], raw_diagnostic)
        if (
            diagnostic["feature_index"] != index
            or diagnostic["feature_id"] != feature_ids[index]
            or diagnostic["precision_supported"] is not (index in supported_set)
        ):
            raise ValueError("response feature diagnostic order is inconsistent")
        for field_name in ("effect_id", "backend", "status"):
            _manifest_string(diagnostic[field_name], field_name=field_name)
        diagnostic_reason = diagnostic["reason_code"]
        if diagnostic_reason is not None:
            _manifest_string(diagnostic_reason, field_name="feature.reason_code")
        formal_eligible = diagnostic["formal_backend_eligible"]
        if type(formal_eligible) is not bool:
            raise ValueError("feature backend eligibility must be boolean")
        n_effective = diagnostic["n_effective_clusters"]
        cluster_df = diagnostic["cluster_df"]
        if (
            isinstance(n_effective, bool)
            or not isinstance(n_effective, int)
            or n_effective < 0
            or (
                cluster_df is not None
                and (
                    isinstance(cluster_df, bool)
                    or not isinstance(cluster_df, int)
                    or cluster_df < 0
                )
            )
        ):
            raise ValueError("feature cluster diagnostics are invalid")
        if artifact_kind == "repeated_cr2_fold_response_v1":
            if (
                diagnostic["status"] not in {"eligible", "not_estimable"}
                or not str(diagnostic["effect_id"]).startswith(
                    "repeated_measures_cr2_feature_effect_"
                )
                or (diagnostic["status"] == "eligible") is not formal_eligible
                or formal_eligible is not diagnostic["precision_supported"]
                or (
                    diagnostic["status"] == "eligible" and diagnostic_reason is not None
                )
                or (
                    diagnostic["status"] == "not_estimable"
                    and diagnostic_reason is None
                )
            ):
                raise ValueError("CR2 feature status is inconsistent")
        elif (
            formal_eligible
            or not str(diagnostic["effect_id"]).startswith(
                "repeated_measures_feature_effect_"
            )
            or diagnostic["status"] not in {"exploratory", "not_estimable"}
            or (diagnostic["status"] == "exploratory")
            is not diagnostic["precision_supported"]
            or (diagnostic["status"] == "exploratory" and diagnostic_reason is not None)
            or (diagnostic["status"] == "not_estimable" and diagnostic_reason is None)
        ):
            raise ValueError("CR1 feature status is inconsistent")
        n_backend_eligible += int(formal_eligible)
    expected_backend_eligible = (
        len(supported)
        if artifact_kind == "fold_gene_response_v2"
        else n_backend_eligible
    )
    if backend["n_backend_eligible_features"] != expected_backend_eligible:
        raise ValueError("response backend eligibility count is inconsistent")
    if artifact_kind == "repeated_cr2_fold_response_v1":
        expected_observed = n_backend_eligible > 0
        if (
            status != ("ok" if expected_observed else "not_estimable")
            or (reason is None) is not expected_observed
        ):
            raise ValueError("CR2 aggregate response status is inconsistent")
    if artifact_kind == "repeated_cr1_fold_response_v1":
        expected_exploratory = any(
            item["status"] == "exploratory" for item in diagnostics
        )
        if (
            status != ("exploratory" if expected_exploratory else "not_estimable")
            or (reason is None) is not expected_exploratory
        ):
            raise ValueError("CR1 aggregate response status is inconsistent")
    if artifact_kind == "fold_gene_response_v2" and (
        status not in {"ok", "not_estimable"}
        or (status == "ok") is not (reason is None)
    ):
        raise ValueError("fold response status is inconsistent")
    expected_diagnostic_digest = stable_id(
        "crossfit_response_feature_diagnostics",
        {
            "feature_diagnostics": diagnostics,
            "feature_ids": feature_ids,
            "response_artifact_id": response_artifact_id,
        },
        schema_version="1",
        digest_length=64,
    )
    if backend["feature_diagnostics_digest"] != expected_diagnostic_digest:
        raise ValueError("response feature diagnostic digest is inconsistent")
    backend_payload = {
        key: value
        for key, value in backend.items()
        if key != "response_backend_manifest_id"
    }
    if backend["response_backend_manifest_id"] != stable_id(
        "crossfit_response_backend_manifest",
        backend_payload,
        schema_version="1",
    ):
        raise ValueError("response backend manifest identity is inconsistent")
    validated = dict(backend)
    validated["expected_precision_method"] = expected_precision_method
    validated["expected_precision_lineage"] = expected_precision_lineage
    return validated


def _validate_precision_audit_manifest(
    raw: object,
    *,
    backend: Mapping[str, Any],
    precision_transform_id: str,
) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != _PRECISION_AUDIT_MANIFEST_FIELDS:
        raise ValueError("precision audit manifest fields are invalid")
    audit = cast(dict[str, Any], raw)
    if (
        audit["schema_version"] != "1.0.0"
        or audit["numeric_values_persisted"] is not False
        or audit["replay_scope"] != "lineage_and_ordered_value_digest_v1"
    ):
        raise ValueError("precision audit replay policy is invalid")
    provenance = audit["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != (
        _PRECISION_PROVENANCE_FIELDS
    ):
        raise ValueError("precision provenance fields are invalid")
    provenance = cast(dict[str, Any], provenance)
    if (
        provenance["precision_transform_id"] != precision_transform_id
        or provenance["response_artifact_id"] != backend["response_artifact_id"]
        or provenance["receiver"] != backend["receiver"]
        or provenance["contrast_name"] != backend["contrast_name"]
        or provenance["fold_id"] != backend["fold_id"]
        or provenance["feature_ids"] != backend["feature_ids"]
        or provenance["raw_precision_digest"] != backend["precision_parent_raw_digest"]
        or provenance["method"] != backend["expected_precision_method"]
        or provenance["lineage_mode"] != backend["expected_precision_lineage"]
        or audit["ordered_values_digest"] != provenance["transformed_precision_digest"]
    ):
        raise ValueError("precision provenance disagrees with its response backend")
    _validated_manifest_string_list(
        provenance["feature_ids"],
        field_name="precision.feature_ids",
        unique=True,
    )
    training_subjects = _validated_manifest_string_list(
        provenance["training_subject_ids"],
        field_name="precision.training_subject_ids",
        unique=True,
    )
    if training_subjects != sorted(training_subjects):
        raise ValueError("precision training subjects are not canonical")
    n_positive = provenance["n_positive_features"]
    minimum_positive = provenance["min_positive_features"]
    if (
        isinstance(n_positive, bool)
        or not isinstance(n_positive, int)
        or n_positive < 0
        or isinstance(minimum_positive, bool)
        or not isinstance(minimum_positive, int)
        or minimum_positive < 1
    ):
        raise ValueError("precision support counts are invalid")
    estimable = provenance["estimable"]
    if type(estimable) is not bool:
        raise ValueError("precision estimability flag is invalid")
    expected_estimable = n_positive >= minimum_positive
    expected_reason = (
        None if expected_estimable else ("insufficient_response_precision_support")
    )
    if (
        estimable is not expected_estimable
        or provenance["reason_code"] != expected_reason
        or provenance["n_positive_features"]
        != backend["n_precision_supported_features"]
    ):
        raise ValueError("precision support status is inconsistent")
    for field_name in (
        "precision_transform_id",
        "method",
        "receiver",
        "contrast_name",
        "fold_id",
        "raw_precision_digest",
        "working_precision_digest",
        "transformed_precision_digest",
        "weight_mode",
        "lineage_mode",
        "response_artifact_id",
        "training_row_manifest_id",
        "encoder_id",
    ):
        _manifest_string(provenance[field_name], field_name=f"precision.{field_name}")
    precision_payload = {
        key: value
        for key, value in provenance.items()
        if key
        not in {
            "precision_transform_id",
            "n_positive_features",
            "estimable",
            "reason_code",
        }
    }
    if precision_transform_id != stable_id(
        "precision_transform",
        precision_payload,
        schema_version="3",
    ):
        raise ValueError("precision transform identity replay failed")
    audit_payload = {
        key: value
        for key, value in audit.items()
        if key != "precision_audit_manifest_id"
    }
    if audit["precision_audit_manifest_id"] != stable_id(
        "crossfit_precision_audit_manifest",
        audit_payload,
        schema_version="1",
    ):
        raise ValueError("precision audit manifest identity is inconsistent")
    return provenance


def _validate_source_response_precision_lineage(source: Mapping[str, object]) -> None:
    folds = source.get("fold_artifacts")
    if not isinstance(folds, list) or not folds:
        raise ValueError("source cross-fit fold registry is invalid")
    for raw_fold in folds:
        if not isinstance(raw_fold, Mapping):
            raise ValueError("source cross-fit fold record is invalid")
        fold_id = _manifest_string(raw_fold.get("fold_id"), field_name="fold_id")
        records = raw_fold.get("receiver_incremental_artifacts")
        if not isinstance(records, list):
            raise ValueError("source receiver incremental registry is invalid")
        for raw_record in records:
            if not isinstance(raw_record, Mapping):
                raise ValueError("source receiver incremental record is invalid")
            record = cast(Mapping[str, Any], raw_record)
            receiver = _manifest_string(record.get("receiver"), field_name="receiver")
            contrast_name = _manifest_string(
                record.get("contrast_name"), field_name="contrast_name"
            )
            response_artifact_id = _manifest_string(
                record.get("response_artifact_id"),
                field_name="response_artifact_id",
            )
            precision_transform_id = _manifest_string(
                record.get("precision_transform_id"),
                field_name="precision_transform_id",
            )
            backend = _validate_response_backend_manifest(
                record.get("response_backend_manifest"),
                fold_id=fold_id,
                receiver=receiver,
                contrast_name=contrast_name,
                response_artifact_id=response_artifact_id,
            )
            precision = _validate_precision_audit_manifest(
                record.get("precision_audit_manifest"),
                backend=backend,
                precision_transform_id=precision_transform_id,
            )
            official_status = record.get("official_incremental_status")
            reason = record.get("reason_code")
            if official_status not in {"observed", "not_estimable"}:
                raise ValueError("receiver incremental official status is invalid")
            if backend["artifact_kind"] == "repeated_cr1_fold_response_v1":
                if (
                    official_status != "not_estimable"
                    or reason != _CR1_REPEATED_OFFICIAL_REASON
                ):
                    raise ValueError("CR1 response cannot support official OOF rows")
            elif official_status == "observed" and (
                backend["status"] != "ok"
                or precision["estimable"] is not True
                or reason is not None
            ):
                raise ValueError("official OOF row lacks an eligible response backend")


def _validate_semantic_collection_manifest(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    raw = manifest.get("semantic_score_collection")
    if not isinstance(raw, dict) or set(raw) != _SEMANTIC_COLLECTION_FIELDS:
        raise ValueError("semantic score collection manifest is invalid")
    collection = cast(dict[str, object], raw)
    if (
        collection.get("crossfit_id") != manifest.get("crossfit_id")
        or collection.get("crossfit_spec_id") != manifest.get("spec_id")
        or collection.get("repeat_id") != manifest.get("repeat_id")
        or collection.get("formal_inference_allowed") is not False
    ):
        raise ValueError("semantic score collection lineage is inconsistent")

    raw_digests = collection.get("table_digests")
    raw_counts = collection.get("table_row_counts")
    raw_statuses = collection.get("view_statuses")
    if not all(
        isinstance(value, list) for value in (raw_digests, raw_counts, raw_statuses)
    ):
        raise ValueError("semantic score collection registries must be arrays")
    digests = cast(list[object], raw_digests)
    counts = cast(list[object], raw_counts)
    statuses = cast(list[object], raw_statuses)
    if not (
        len(digests) == len(counts) == len(statuses) == len(SEMANTIC_SCORE_OUTPUTS)
    ):
        raise ValueError("semantic score collection coverage is incomplete")

    digest_by_name: dict[str, str] = {}
    count_by_name: dict[str, int] = {}
    status_by_name: dict[str, tuple[str, str | None]] = {}
    for index, semantic_output in enumerate(SEMANTIC_SCORE_OUTPUTS):
        digest_item = digests[index]
        count_item = counts[index]
        status_item = statuses[index]
        if (
            not isinstance(digest_item, list)
            or len(digest_item) != 2
            or digest_item[0] != semantic_output
            or not isinstance(digest_item[1], str)
            or not digest_item[1].startswith("semantic_score_table_")
            or len(digest_item[1].removeprefix("semantic_score_table_")) != 64
        ):
            raise ValueError("semantic score table digest registry is invalid")
        if (
            not isinstance(count_item, list)
            or len(count_item) != 2
            or count_item[0] != semantic_output
            or isinstance(count_item[1], bool)
            or not isinstance(count_item[1], int)
            or count_item[1] < 0
        ):
            raise ValueError("semantic score table count registry is invalid")
        if (
            not isinstance(status_item, list)
            or len(status_item) != 3
            or status_item[0] != semantic_output
            or status_item[1] not in _SEMANTIC_VIEW_STATUSES
            or (
                status_item[2] is not None
                and (
                    not isinstance(status_item[2], str)
                    or not status_item[2]
                    or status_item[2] != status_item[2].strip()
                )
            )
        ):
            raise ValueError("semantic score view status registry is invalid")
        status = str(status_item[1])
        reason: str | None = status_item[2]
        if (
            status == "produced" and reason not in {None, _DIRECTIONAL_EXCLUDED_REASON}
        ) or (status != "produced" and reason is None):
            raise ValueError("semantic score view status/reason is inconsistent")
        digest_by_name[semantic_output] = str(digest_item[1])
        count_by_name[semantic_output] = int(count_item[1])
        status_by_name[semantic_output] = (status, reason)

    source = manifest.get("source_crossfit_manifest")
    source_spec = source.get("spec") if isinstance(source, dict) else None
    has_directional_pair = bool(
        source_spec.get("directional_pairs") if isinstance(source_spec, dict) else False
    )
    has_penalty_tuning = bool(
        isinstance(source_spec, dict)
        and source_spec.get("penalty_tuning_spec") is not None
    )
    for semantic_output in SEMANTIC_SCORE_OUTPUTS:
        row_count = count_by_name[semantic_output]
        if row_count:
            expected_status = "produced"
            expected_reason = (
                _DIRECTIONAL_EXCLUDED_REASON
                if has_directional_pair
                and semantic_output in {"integrated_lr_score", "differential_effect"}
                else None
            )
        elif has_directional_pair and semantic_output in {
            "integrated_lr_score",
            "differential_effect",
        }:
            expected_status = "not_produced"
            expected_reason = _DIRECTIONAL_DEDICATED_REASON
        elif not has_penalty_tuning and semantic_output in {
            "integrated_lr_score",
            "differential_effect",
        }:
            expected_status = "not_produced"
            expected_reason = _NO_COMMON_SCORING_REASON
        else:
            expected_status = "not_estimable"
            expected_reason = f"no_{semantic_output}_rows"
        if status_by_name[semantic_output] != (expected_status, expected_reason):
            raise ValueError(
                "semantic score view status does not match its source spec"
            )
    raw_views = collection.get("views")
    if not isinstance(raw_views, list) or len(raw_views) != len(SEMANTIC_SCORE_OUTPUTS):
        raise ValueError("semantic score view manifest is incomplete")
    for index, semantic_output in enumerate(SEMANTIC_SCORE_OUTPUTS):
        view = raw_views[index]
        status, reason = status_by_name[semantic_output]
        expected_view = {
            "semantic_output": semantic_output,
            "grain": _SEMANTIC_GRAINS[semantic_output],
            "status": status,
            "reason_code": reason,
            "row_count": count_by_name[semantic_output],
            "table_digest": digest_by_name[semantic_output],
            "dedicated_view": (
                has_directional_pair
                and semantic_output in {"integrated_lr_score", "differential_effect"}
            ),
            "formal_inference_allowed": False,
        }
        if (
            not isinstance(view, dict)
            or set(view) != _SEMANTIC_VIEW_FIELDS
            or view != expected_view
        ):
            raise ValueError("semantic score view manifest is inconsistent")
    if collection.get("excluded_output_kinds") != list(_SEMANTIC_EXCLUDED_OUTPUT_KINDS):
        raise ValueError("semantic score excluded outputs are inconsistent")
    identity_payload = {
        "crossfit_id": collection["crossfit_id"],
        "crossfit_spec_id": collection["crossfit_spec_id"],
        "formal_inference_allowed": False,
        "repeat_id": collection["repeat_id"],
        "table_digests": digests,
        "table_row_counts": counts,
        "view_statuses": statuses,
    }
    if collection.get("collection_id") != stable_id(
        "crossfit_semantic_score_collection",
        identity_payload,
        schema_version="1",
    ):
        raise ValueError("semantic score collection identity is inconsistent")
    return collection


def _validate_manifest(manifest: dict[str, Any]) -> dict[str, object]:
    schema_version = manifest.get("schema_version")
    if schema_version not in _SUPPORTED_CROSSFIT_RESULT_SCHEMA_VERSIONS:
        raise ValueError("cross-fit result schema version is unsupported")
    expected = {
        "schema_version",
        "crossfit_result_id",
        "status",
        "crossfit_id",
        "spec_id",
        "repeat_id",
        "root_input_digest",
        "certification_status",
        "complete_pipeline_oof_certified",
        "formal_inference_status",
        "claim_scope",
        "source_crossfit_manifest_digest",
        "source_crossfit_manifest",
        "family_common_stage_connected",
        "applications",
        "tables",
    }
    if schema_version in {
        CROSSFIT_RESULT_SCHEMA_VERSION,
        _V8_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V7_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V6_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V5_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V4_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V3_CROSSFIT_RESULT_SCHEMA_VERSION,
    }:
        expected.update(
            {
                "contrast_common_stage_connected",
                "contrast_common_collections",
            }
        )
    if schema_version in _V7_PLUS_CROSSFIT_RESULT_SCHEMA_VERSIONS:
        expected.update({"receiver_universe_id", "receiver_axis_id"})
    if schema_version in _V8_PLUS_CROSSFIT_RESULT_SCHEMA_VERSIONS:
        expected.add("semantic_score_collection")
    if schema_version == CROSSFIT_RESULT_SCHEMA_VERSION:
        expected.update(
            {
                "receiver_family_opportunity_universe_id",
                "family_axis_id",
                "receiver_family_opportunity_axis_id",
                "receiver_family_opportunity_universe",
            }
        )
    if set(manifest) != expected:
        raise ValueError("cross-fit result manifest fields do not match its schema")
    legacy_schema = schema_version == _V1_CROSSFIT_RESULT_SCHEMA_VERSION
    if manifest["status"] != _COMPLETE_STATUS:
        raise ValueError("cross-fit result manifest is not complete")
    if type(manifest["complete_pipeline_oof_certified"]) is not bool:
        raise ValueError("cross-fit result OOF certification flag must be boolean")
    is_oof_certified = bool(manifest["complete_pipeline_oof_certified"])
    if manifest["formal_inference_status"] != _FORMAL_INFERENCE_STATUS:
        raise ValueError("cross-fit result cannot claim formal inference")
    expected_claim_scope = (
        _UNCERTIFIED_CLAIM_SCOPE
        if legacy_schema
        else _expected_claim_scope(is_oof_certified)
    )
    if manifest["claim_scope"] != expected_claim_scope:
        raise ValueError("cross-fit result claim scope is invalid")
    for field_name in (
        "crossfit_result_id",
        "crossfit_id",
        "spec_id",
        "repeat_id",
        "root_input_digest",
        "certification_status",
        "source_crossfit_manifest_digest",
    ):
        value = manifest[field_name]
        if not isinstance(value, str) or not value:
            raise ValueError(f"manifest {field_name} must be a non-empty string")
    if schema_version in _V7_PLUS_CROSSFIT_RESULT_SCHEMA_VERSIONS:
        for field_name in ("receiver_universe_id", "receiver_axis_id"):
            _manifest_string(manifest[field_name], field_name=field_name)
    if schema_version == CROSSFIT_RESULT_SCHEMA_VERSION:
        for field_name in (
            "receiver_family_opportunity_universe_id",
            "family_axis_id",
            "receiver_family_opportunity_axis_id",
        ):
            _manifest_string(manifest[field_name], field_name=field_name)
    source = manifest["source_crossfit_manifest"]
    if not isinstance(source, dict):
        raise ValueError("source_crossfit_manifest must be an object")
    if source.get("crossfit_id") != manifest["crossfit_id"]:
        raise ValueError("source cross-fit identity does not match the result")
    if canonical_digest(source) != manifest["source_crossfit_manifest_digest"]:
        raise ValueError("source cross-fit manifest digest does not match")
    if schema_version in _V7_PLUS_CROSSFIT_RESULT_SCHEMA_VERSIONS:
        _validate_source_receiver_universe(cast(dict[str, object], manifest))
        _validate_source_directional_lr_hypothesis_universe(
            cast(dict[str, object], manifest)
        )
    if schema_version == CROSSFIT_RESULT_SCHEMA_VERSION:
        _validate_source_receiver_family_opportunity_universe(
            cast(dict[str, object], manifest)
        )
    if schema_version in _V8_PLUS_CROSSFIT_RESULT_SCHEMA_VERSIONS:
        _validate_semantic_collection_manifest(manifest)
        _validate_source_response_precision_lineage(source)
    _validate_source_latent_nuisance_lineage(
        source,
        expected_crossfit_spec_id=str(manifest["spec_id"]),
    )
    if legacy_schema:
        if (
            is_oof_certified
            or source.get("complete_pipeline_oof_certified") is not False
            or "oof_certification_audit" in source
            or "oof_certification_audit_id" in source
        ):
            raise ValueError("legacy cross-fit result cannot claim complete OOF")
    else:
        if source.get("complete_pipeline_oof_certified") is not is_oof_certified:
            raise ValueError("source and result OOF certification flags disagree")
    if not legacy_schema:
        raw_audit = source.get("oof_certification_audit")
        if not isinstance(raw_audit, Mapping):
            raise ValueError("source OOF certification audit is invalid")
        try:
            audit = CrossFitOOFCertificationAudit.from_dict(raw_audit)
        except (TypeError, ValueError) as error:
            raise ValueError("source OOF certification audit is invalid") from error
        if audit.to_dict() != dict(raw_audit):
            raise ValueError("source OOF certification audit payload is noncanonical")
        if (
            audit.source_crossfit_id != manifest["crossfit_id"]
            or audit.source_registry_id != source.get("receiver_scoring_registry_id")
            or audit.complete is not is_oof_certified
            or source.get("oof_certification_audit_id") != audit.audit_id
        ):
            raise ValueError("source OOF certification audit lineage is invalid")
        if manifest["certification_status"] != (
            CROSSFIT_OOF_DESCRIPTIVE_SCOPE
            if is_oof_certified
            else source.get("certification_status")
        ):
            raise ValueError("cross-fit result certification status is inconsistent")
    if manifest["crossfit_result_id"] != stable_id(
        "crossfit_result",
        _manifest_payload(cast(dict[str, object], manifest)),
        schema_version="1",
    ):
        raise ValueError("cross-fit result identity does not match its manifest")
    expected_table_names = _bundle_table_names(schema_version)
    tables = manifest["tables"]
    if not isinstance(tables, dict) or set(tables) != set(expected_table_names):
        raise ValueError("cross-fit result tables do not match the released set")
    for name in expected_table_names:
        record = tables[name]
        if not isinstance(record, dict) or set(record) != {
            "filename",
            "rows",
            "sha256",
            "schema_version",
            "columns",
        }:
            raise ValueError(f"manifest table record {name!r} is invalid")
        if (
            record["filename"] != _TABLE_FILENAMES[name]
            or record["schema_version"] != _table_schema_version(schema_version, name)
            or tuple(record["columns"]) != _table_columns(schema_version, name)
            or isinstance(record["rows"], bool)
            or not isinstance(record["rows"], int)
            or record["rows"] < 0
            or not isinstance(record["sha256"], str)
            or len(record["sha256"]) != 64
        ):
            raise ValueError(f"manifest table record {name!r} is invalid")
    applications = manifest["applications"]
    if not isinstance(applications, list):
        raise ValueError("manifest applications must be an array")
    required_application_fields = {
        "fold_id",
        "contrast_id",
        "contrast",
        "receiver",
        "family_common_functional_id",
        "family_common_application_id",
        "family_common_binding_id",
        "sender_functional_id",
        "score_version",
        "certification_status",
        "is_oof_certified",
        "source_table_digests",
        "source_table_row_counts",
        "persisted_component_rows",
        "persisted_differential_rows",
    }
    application_ids: list[str] = []
    for application in applications:
        if (
            not isinstance(application, dict)
            or set(application) != required_application_fields
        ):
            raise ValueError("manifest application registry entry is invalid")
        if application["is_oof_certified"] is not (
            False if legacy_schema else is_oof_certified
        ):
            raise ValueError("application registry OOF certification is inconsistent")
        if application["certification_status"] != manifest["certification_status"]:
            raise ValueError(
                "application registry certification status is inconsistent"
            )
        application_ids.append(str(application["family_common_application_id"]))
    if len(application_ids) != len(set(application_ids)):
        raise ValueError("manifest application IDs must be unique")
    connected = bool(applications)
    if manifest["family_common_stage_connected"] is not connected:
        raise ValueError("family-common stage flag disagrees with application registry")
    if schema_version in {
        CROSSFIT_RESULT_SCHEMA_VERSION,
        _V8_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V7_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V6_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V5_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V4_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V3_CROSSFIT_RESULT_SCHEMA_VERSION,
    }:
        _validate_contrast_common_collections_manifest(manifest)
    return cast(dict[str, object], manifest)


def _load_table(
    root: Path,
    name: str,
    record: dict[str, Any],
    *,
    schema_version: str,
) -> pd.DataFrame:
    path = root / _TABLE_FILENAMES[name]
    try:
        digest = _sha256_file(path)
    except Exception as error:
        raise ResultValidationError(
            f"Cross-fit table {name!r} is missing or corrupted",
            code="corrupted_crossfit_result_table",
            field=name,
            remediation="Reject the bundle and regenerate it from intact artifacts",
        ) from error
    if digest != record["sha256"]:
        raise ResultValidationError(
            f"Cross-fit table {name!r} does not match its manifest",
            code="crossfit_result_digest_mismatch",
            field=name,
            remediation="Reject the bundle and regenerate it from intact artifacts",
        )
    try:
        frame = pd.read_parquet(path)
    except Exception as error:
        raise ResultValidationError(
            f"Cross-fit table {name!r} is corrupted",
            code="corrupted_crossfit_result_table",
            field=name,
            remediation="Reject the bundle and regenerate it from intact artifacts",
        ) from error
    if len(frame) != record["rows"] or tuple(frame.columns) != tuple(record["columns"]):
        raise ResultValidationError(
            f"Cross-fit table {name!r} does not match its manifest",
            code="crossfit_result_digest_mismatch",
            field=name,
            remediation="Reject the bundle and regenerate it from intact artifacts",
        )
    try:
        return _validate_table(name, frame, schema_version=schema_version)
    except (TypeError, ValueError) as error:
        raise ResultValidationError(
            f"Cross-fit table {name!r} violates its semantic contract",
            code="invalid_crossfit_result_table",
            field=name,
            remediation="Reject the bundle and rerun its producer",
        ) from error


def _contrast_common_registries(
    manifest: dict[str, object],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, tuple[dict[str, Any], dict[str, Any]]],
]:
    collections = cast(
        list[dict[str, Any]], manifest.get("contrast_common_collections", [])
    )
    collections_by_id = {
        str(collection["contrast_common_collection_id"]): collection
        for collection in collections
    }
    applications_by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for collection in collections:
        for application in cast(list[dict[str, Any]], collection["fold_applications"]):
            applications_by_id[str(application["global_common_application_id"])] = (
                collection,
                application,
            )
    return collections_by_id, applications_by_id


def _validate_v3_source_contrast_common_registry(
    manifest: dict[str, object],
    applications_by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    source = cast(dict[str, Any], manifest["source_crossfit_manifest"])
    fold_artifacts = source.get("fold_artifacts")
    if not isinstance(fold_artifacts, list):
        raise ResultValidationError(
            "Source cross-fit fold artifacts are unavailable for global validation",
            code="invalid_contrast_common_source_registry",
            field="source_crossfit_manifest.fold_artifacts",
            remediation="Reject the bundle and rerun its producer",
        )
    source_by_id: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {}
    for fold in fold_artifacts:
        if not isinstance(fold, dict):
            raise ResultValidationError(
                "Source cross-fit fold registry is invalid",
                code="invalid_contrast_common_source_registry",
                field="source_crossfit_manifest.fold_artifacts",
                remediation="Reject the bundle and rerun its producer",
            )
        fold_id = str(fold.get("fold_id", ""))
        global_artifacts = fold.get("cross_receiver_common_scoring_artifacts")
        family_artifacts = fold.get("family_common_scoring_artifacts")
        if not isinstance(global_artifacts, list) or not isinstance(
            family_artifacts, list
        ):
            raise ResultValidationError(
                "Source global/common child registries are invalid",
                code="invalid_contrast_common_source_registry",
                field="source_crossfit_manifest.fold_artifacts",
                remediation="Reject the bundle and rerun its producer",
            )
        family_by_functional = {
            str(item["functional_id"]): item
            for item in family_artifacts
            if isinstance(item, dict) and "functional_id" in item
        }
        for item in global_artifacts:
            if not isinstance(item, dict):
                raise ResultValidationError(
                    "Source contrast-common artifact is invalid",
                    code="invalid_contrast_common_source_registry",
                    field="cross_receiver_common_scoring_artifacts",
                    remediation="Reject the bundle and rerun its producer",
                )
            application_id = str(item.get("application_id", ""))
            if not application_id or application_id in source_by_id:
                raise ResultValidationError(
                    "Source contrast-common application identity is invalid",
                    code="invalid_contrast_common_source_registry",
                    field="application_id",
                    remediation="Reject the bundle and rerun its producer",
                )
            source_by_id[application_id] = (fold_id, item, family_by_functional)
    if set(source_by_id) != set(applications_by_id):
        raise ResultValidationError(
            "Persisted global applications do not exactly cover their source registry",
            code="contrast_common_source_coverage_mismatch",
            field="contrast_common_collections",
            remediation="Reject the bundle and rerun its producer",
        )
    for application_id, (collection, application) in applications_by_id.items():
        fold_id, source_record, family_by_functional = source_by_id[application_id]
        expected_source_projection = {
            "fold_id": fold_id,
            "contrast": source_record.get("contrast_name"),
            "contrast_id": source_record.get("contrast_manifest_id"),
            "global_common_functional_id": source_record.get("functional_id"),
            "global_common_application_id": source_record.get("application_id"),
            "receiver_ids": source_record.get("receiver_ids"),
            "score_version": source_record.get("score_version"),
            "estimand": source_record.get("estimand"),
            "functional_spec_id": source_record.get("functional_spec_id"),
            "functional_schema_version": source_record.get("functional_schema_version"),
            "calibration_policy": source_record.get("calibration_policy"),
            "calibration_quantile": source_record.get("calibration_quantile"),
            "global_training_anchor": source_record.get("global_training_anchor"),
            "receiver_training_anchors": source_record.get("receiver_training_anchors"),
            "receiver_scale_factors": source_record.get("receiver_scale_factors"),
            "common_functional_across_receivers": source_record.get(
                "common_functional_across_receivers"
            ),
            "receiver_balanced_descriptive_collection": source_record.get(
                "receiver_balanced_descriptive_collection"
            ),
            "lr_scores_digest": source_record.get("lr_scores_digest"),
            "sender_scores_digest": source_record.get("sender_scores_digest"),
            "n_lr_rows": source_record.get("n_lr_rows"),
            "n_sender_rows": source_record.get("n_sender_rows"),
        }
        observed_projection = {
            "fold_id": application["fold_id"],
            "contrast": collection["contrast"],
            "contrast_id": collection["contrast_id"],
            "global_common_functional_id": application["global_common_functional_id"],
            "global_common_application_id": application["global_common_application_id"],
            "receiver_ids": application["receiver_ids"],
            "score_version": collection["score_version"],
            "estimand": collection["estimand"],
            "functional_spec_id": application["functional_spec_id"],
            "functional_schema_version": application["functional_schema_version"],
            "calibration_policy": application["calibration_policy"],
            "calibration_quantile": application["calibration_quantile"],
            "global_training_anchor": application["global_training_anchor"],
            "receiver_training_anchors": application["receiver_training_anchors"],
            "receiver_scale_factors": application["receiver_scale_factors"],
            "common_functional_across_receivers": application[
                "common_functional_across_receivers"
            ],
            "receiver_balanced_descriptive_collection": application[
                "receiver_balanced_descriptive_collection"
            ],
            "lr_scores_digest": application["lr_scores_digest"],
            "sender_scores_digest": application["sender_scores_digest"],
            "n_lr_rows": application["n_lr_rows"],
            "n_sender_rows": application["n_sender_rows"],
        }
        if (
            observed_projection != expected_source_projection
            or source_record.get("common_functional_across_receivers") is not False
            or source_record.get("receiver_balanced_descriptive_collection") is not True
            or source_record.get("training_only_receiver_calibration") is not True
            or source_record.get("receiver_scale_amplification") is not False
            or source_record.get("formal_inference_allowed") is not False
        ):
            raise ResultValidationError(
                "Persisted global application disagrees with source lineage",
                code="contrast_common_source_lineage_mismatch",
                field="contrast_common_collections",
                remediation="Reject the bundle and rerun its producer",
            )
        child_ids = source_record.get("child_functional_ids")
        if not isinstance(child_ids, list):
            raise ResultValidationError(
                "Source global child lineage is invalid",
                code="invalid_contrast_common_source_registry",
                field="child_functional_ids",
                remediation="Reject the bundle and rerun its producer",
            )
        expected_children = []
        for functional_id in child_ids:
            child = family_by_functional.get(str(functional_id))
            if child is None:
                raise ResultValidationError(
                    "Source global child is absent from family-common lineage",
                    code="invalid_contrast_common_source_registry",
                    field="child_functional_ids",
                    remediation="Reject the bundle and rerun its producer",
                )
            expected_children.append(
                {
                    "receiver": child.get("receiver"),
                    "family_common_functional_id": child.get("functional_id"),
                    "family_common_application_id": child.get("application_id"),
                }
            )
        expected_children.sort(key=lambda item: str(item["receiver"]))
        if application["receiver_children"] != expected_children:
            raise ResultValidationError(
                "Persisted receiver children disagree with source lineage",
                code="contrast_common_source_lineage_mismatch",
                field="receiver_children",
                remediation="Reject the bundle and rerun its producer",
            )


def _validate_v3_contrast_common_table_lineage(
    table_name: str,
    table: pd.DataFrame,
    applications_by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    source_columns = (
        _V3_GLOBAL_COMMON_LR_SCORE_COLUMNS
        if table_name == CROSSFIT_CONTRAST_COMMON_LR_TABLE
        else _V3_GLOBAL_COMMON_SENDER_SCORE_COLUMNS
    )
    digest_name = (
        "cross_receiver_common_lr_scores"
        if table_name == CROSSFIT_CONTRAST_COMMON_LR_TABLE
        else "cross_receiver_common_sender_scores"
    )
    expected_count_field = (
        "n_lr_rows"
        if table_name == CROSSFIT_CONTRAST_COMMON_LR_TABLE
        else "n_sender_rows"
    )
    expected_digest_field = (
        "lr_scores_digest"
        if table_name == CROSSFIT_CONTRAST_COMMON_LR_TABLE
        else "sender_scores_digest"
    )
    observed_counts = _application_registry_counts(
        table, id_column="global_common_application_id"
    )
    expected_counts = {
        application_id: int(application[expected_count_field])
        for application_id, (_, application) in applications_by_id.items()
    }
    normalized_counts = {
        application_id: observed_counts.get(application_id, 0)
        for application_id in expected_counts
    }
    if (
        set(observed_counts).difference(expected_counts)
        or normalized_counts != expected_counts
    ):
        raise ResultValidationError(
            "Contrast-common rows do not match the collection registry",
            code="contrast_common_application_coverage_mismatch",
            field=table_name,
            remediation="Reject the bundle and rerun its producer",
        )
    grouped = table.groupby(
        "global_common_application_id",
        observed=True,
        sort=False,
    )
    for raw_application_id, rows in grouped:
        lineage = applications_by_id.get(str(raw_application_id))
        if lineage is None:
            continue
        collection, application = lineage
        expected_projection: dict[str, object] = {
            "contrast_common_collection_id": collection[
                "contrast_common_collection_id"
            ],
            "fold_id": application["fold_id"],
            "contrast_id": collection["contrast_id"],
            "contrast": collection["contrast"],
            "global_common_functional_id": application["global_common_functional_id"],
            "score_version": collection["score_version"],
            "certification_status": collection["certification_status"],
            "is_oof_certified": collection["is_oof_certified"],
            "claim_scope": collection["claim_scope"],
        }
        for field_name, expected in expected_projection.items():
            observed = rows[field_name].tolist()
            if type(expected) is bool:
                matched = all(
                    type(value) is bool and value is expected for value in observed
                )
            else:
                matched = all(value == expected for value in observed)
            if not matched:
                raise ResultValidationError(
                    "Contrast-common rows mix or misstate collection lineage",
                    code="contrast_common_application_lineage_mismatch",
                    field=f"{table_name}.{field_name}",
                    remediation="Reject the bundle and rerun its producer",
                )
        child_by_receiver = {
            str(child["receiver"]): child
            for child in cast(list[dict[str, Any]], application["receiver_children"])
        }
        scale_by_receiver = {
            str(receiver): float(cast(Any, scale))
            for receiver, scale in cast(
                list[list[object]], application["receiver_scale_factors"]
            )
        }
        for receiver, receiver_rows in rows.groupby("receiver", sort=False):
            child = child_by_receiver.get(str(receiver))
            if (
                child is None
                or set(receiver_rows["source_family_common_functional_id"].astype(str))
                != {str(child["family_common_functional_id"])}
                or set(receiver_rows["source_family_common_application_id"].astype(str))
                != {str(child["family_common_application_id"])}
            ):
                raise ResultValidationError(
                    "Contrast-common rows misstate receiver-child lineage",
                    code="contrast_common_child_lineage_mismatch",
                    field=table_name,
                    remediation="Reject the bundle and rerun its producer",
                )
            expected_scale = scale_by_receiver.get(str(receiver))
            observed_scales = receiver_rows["training_receiver_scale_factor"].tolist()
            if expected_scale is None or not all(
                _optional_float(
                    value,
                    field_name="training_receiver_scale_factor",
                )
                is not None
                and math.isclose(
                    float(value),
                    expected_scale,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for value in observed_scales
            ):
                raise ResultValidationError(
                    "Contrast-common rows misstate train-only receiver calibration",
                    code="contrast_common_calibration_lineage_mismatch",
                    field=f"{table_name}.training_receiver_scale_factor",
                    remediation="Reject the bundle and rerun its producer",
                )
        if set(rows["receiver"].astype(str)) != set(child_by_receiver):
            raise ResultValidationError(
                "Contrast-common rows lack exact receiver coverage",
                code="contrast_common_application_coverage_mismatch",
                field=table_name,
                remediation="Reject the bundle and rerun its producer",
            )
        if table_name == CROSSFIT_CONTRAST_COMMON_LR_TABLE:
            if set(rows["subject_id"].astype(str)) != set(
                application["heldout_subject_ids"]
            ) or set(rows["context_id"].astype(str)) != set(application["context_ids"]):
                raise ResultValidationError(
                    "Contrast-common LR rows lack exact held-out scope",
                    code="contrast_common_application_coverage_mismatch",
                    field=table_name,
                    remediation="Reject the bundle and rerun its producer",
                )
        else:
            observed_sender_lineages = [
                {
                    "receiver": str(row.receiver),
                    "sender_functional_id": str(row.source_sender_functional_id),
                    "sender_application_id": str(row.source_sender_application_id),
                }
                for row in rows.loc[
                    :,
                    [
                        "receiver",
                        "source_sender_functional_id",
                        "source_sender_application_id",
                    ],
                ]
                .drop_duplicates()
                .sort_values(
                    [
                        "receiver",
                        "source_sender_functional_id",
                        "source_sender_application_id",
                    ],
                    kind="stable",
                )
                .itertuples(index=False)
            ]
            if observed_sender_lineages != application["sender_lineages"]:
                raise ResultValidationError(
                    "Contrast-common sender rows misstate source lineage",
                    code="contrast_common_sender_lineage_mismatch",
                    field=table_name,
                    remediation="Reject the bundle and rerun its producer",
                )
        source_projection = rows.loc[:, list(source_columns)].copy(deep=True)
        observed_digest = _global_source_table_digest(
            digest_name,
            source_projection,
            source_columns,
        )
        if observed_digest != application[expected_digest_field]:
            raise ResultValidationError(
                "Contrast-common source table digest is inconsistent",
                code="contrast_common_source_digest_mismatch",
                field=table_name,
                remediation="Reject the bundle and rerun its producer",
            )


def _source_contrast_common_records(
    manifest: dict[str, object],
) -> dict[str, tuple[str, dict[str, Any], dict[str, Any]]]:
    source = cast(dict[str, Any], manifest["source_crossfit_manifest"])
    fold_artifacts = source.get("fold_artifacts")
    if not isinstance(fold_artifacts, list):
        raise ResultValidationError(
            "Source cross-fit fold artifacts are unavailable for global validation",
            code="invalid_contrast_common_source_registry",
            field="source_crossfit_manifest.fold_artifacts",
            remediation="Reject the bundle and rerun its producer",
        )
    source_by_id: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {}
    for fold in fold_artifacts:
        if not isinstance(fold, dict):
            raise ResultValidationError(
                "Source cross-fit fold registry is invalid",
                code="invalid_contrast_common_source_registry",
                field="source_crossfit_manifest.fold_artifacts",
                remediation="Reject the bundle and rerun its producer",
            )
        fold_id = str(fold.get("fold_id", ""))
        global_artifacts = fold.get("cross_receiver_common_scoring_artifacts")
        family_artifacts = fold.get("family_common_scoring_artifacts")
        if not isinstance(global_artifacts, list) or not isinstance(
            family_artifacts, list
        ):
            raise ResultValidationError(
                "Source global/common child registries are invalid",
                code="invalid_contrast_common_source_registry",
                field="source_crossfit_manifest.fold_artifacts",
                remediation="Reject the bundle and rerun its producer",
            )
        family_by_functional = {
            str(item["functional_id"]): item
            for item in family_artifacts
            if isinstance(item, dict) and "functional_id" in item
        }
        for item in global_artifacts:
            if not isinstance(item, dict):
                raise ResultValidationError(
                    "Source contrast-common artifact is invalid",
                    code="invalid_contrast_common_source_registry",
                    field="cross_receiver_common_scoring_artifacts",
                    remediation="Reject the bundle and rerun its producer",
                )
            application_id = str(item.get("application_id", ""))
            if not application_id or application_id in source_by_id:
                raise ResultValidationError(
                    "Source contrast-common application identity is invalid",
                    code="invalid_contrast_common_source_registry",
                    field="application_id",
                    remediation="Reject the bundle and rerun its producer",
                )
            source_by_id[application_id] = (fold_id, item, family_by_functional)
    return source_by_id


def _validate_source_contrast_common_registry(
    manifest: dict[str, object],
    applications_by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    if manifest.get("schema_version") == _V3_CROSSFIT_RESULT_SCHEMA_VERSION:
        _validate_v3_source_contrast_common_registry(manifest, applications_by_id)
        return
    source_by_id = _source_contrast_common_records(manifest)
    if set(source_by_id) != set(applications_by_id):
        raise ResultValidationError(
            "Persisted global applications do not exactly cover their source registry",
            code="contrast_common_source_coverage_mismatch",
            field="contrast_common_collections",
            remediation="Reject the bundle and rerun its producer",
        )
    for application_id, (collection, application) in applications_by_id.items():
        fold_id, source_record, family_by_functional = source_by_id[application_id]
        source_projection = {
            "fold_id": fold_id,
            "contrast": source_record.get("contrast_name"),
            "contrast_id": source_record.get("contrast_manifest_id"),
            "global_common_functional_id": source_record.get("functional_id"),
            "global_common_application_id": source_record.get("application_id"),
            "receiver_ids": source_record.get("receiver_ids"),
            "score_version": source_record.get("score_version"),
            "estimand": source_record.get("estimand"),
            "functional_spec_id": source_record.get("functional_spec_id"),
            "functional_schema_version": source_record.get("functional_schema_version"),
            "calibration_policy": source_record.get("calibration_policy"),
            "scale_policy": source_record.get("scale_policy"),
            "sender_policy": source_record.get("sender_policy"),
            "softmin_power": source_record.get("softmin_power"),
            "epsilon": source_record.get("epsilon"),
            "receiver_gain_calibration_bindings": source_record.get(
                "receiver_gain_calibration_bindings"
            ),
            "all_receivers_gain_calibrated": source_record.get(
                "all_receivers_gain_calibrated"
            ),
            "cross_receiver_percentile_rank_eligible": source_record.get(
                "cross_receiver_percentile_rank_eligible"
            ),
            "common_functional_across_receivers": source_record.get(
                "common_functional_across_receivers"
            ),
            "receiver_balanced_descriptive_collection": source_record.get(
                "receiver_balanced_descriptive_collection"
            ),
            "lr_scores_digest": source_record.get("lr_scores_digest"),
            "sender_scores_digest": source_record.get("sender_scores_digest"),
            "n_lr_rows": source_record.get("n_lr_rows"),
            "n_sender_rows": source_record.get("n_sender_rows"),
        }
        persisted_projection = {
            "fold_id": application["fold_id"],
            "contrast": collection["contrast"],
            "contrast_id": collection["contrast_id"],
            "global_common_functional_id": application["global_common_functional_id"],
            "global_common_application_id": application["global_common_application_id"],
            "receiver_ids": application["receiver_ids"],
            "score_version": collection["score_version"],
            "estimand": collection["estimand"],
            "functional_spec_id": application["functional_spec_id"],
            "functional_schema_version": application["functional_schema_version"],
            "calibration_policy": application["calibration_policy"],
            "scale_policy": application["scale_policy"],
            "sender_policy": application["sender_policy"],
            "softmin_power": application["softmin_power"],
            "epsilon": application["epsilon"],
            "receiver_gain_calibration_bindings": application[
                "receiver_gain_calibration_bindings"
            ],
            "all_receivers_gain_calibrated": application[
                "all_receivers_gain_calibrated"
            ],
            "cross_receiver_percentile_rank_eligible": application[
                "cross_receiver_percentile_rank_eligible"
            ],
            "common_functional_across_receivers": application[
                "common_functional_across_receivers"
            ],
            "receiver_balanced_descriptive_collection": application[
                "receiver_balanced_descriptive_collection"
            ],
            "lr_scores_digest": application["lr_scores_digest"],
            "sender_scores_digest": application["sender_scores_digest"],
            "n_lr_rows": application["n_lr_rows"],
            "n_sender_rows": application["n_sender_rows"],
        }
        if (
            persisted_projection != source_projection
            or source_record.get("common_functional_across_receivers") is not False
            or source_record.get("receiver_balanced_descriptive_collection") is not True
            or source_record.get("training_only_receiver_calibration") is not True
            or source_record.get("receiver_scale_amplification") is not False
            or source_record.get("formal_inference_allowed") is not False
        ):
            raise ResultValidationError(
                "Persisted global application disagrees with source lineage",
                code="contrast_common_source_lineage_mismatch",
                field="contrast_common_collections",
                remediation="Reject the bundle and rerun its producer",
            )
        child_ids = source_record.get("child_functional_ids")
        if not isinstance(child_ids, list):
            raise ResultValidationError(
                "Source global child lineage is invalid",
                code="invalid_contrast_common_source_registry",
                field="child_functional_ids",
                remediation="Reject the bundle and rerun its producer",
            )
        expected_children: list[dict[str, object]] = []
        for functional_id in child_ids:
            child = family_by_functional.get(str(functional_id))
            if child is None:
                raise ResultValidationError(
                    "Source global child is absent from family-common lineage",
                    code="invalid_contrast_common_source_registry",
                    field="child_functional_ids",
                    remediation="Reject the bundle and rerun its producer",
                )
            expected_children.append(
                {
                    "receiver": child.get("receiver"),
                    "family_common_functional_id": child.get("functional_id"),
                    "family_common_application_id": child.get("application_id"),
                }
            )
        expected_children.sort(key=lambda item: str(item["receiver"]))
        if application["receiver_children"] != expected_children:
            raise ResultValidationError(
                "Persisted receiver children disagree with source lineage",
                code="contrast_common_source_lineage_mismatch",
                field="receiver_children",
                remediation="Reject the bundle and rerun its producer",
            )


def _gain_percentile(
    raw_gain: float | None,
    binding: dict[str, Any],
) -> float | None:
    if raw_gain is None or binding["gain_calibration_artifact_id"] is None:
        return None
    if raw_gain == 0.0:
        return 0.0
    if binding["gain_calibration_status"] != "observed":
        return None
    source = [float(value) for value in binding["positive_gain_source_knots"]]
    percentiles = [float(value) for value in binding["positive_gain_percentile_knots"]]
    if raw_gain >= source[-1]:
        return 1.0
    index = bisect_right(source, raw_gain)
    left_x, right_x = source[index - 1], source[index]
    left_y, right_y = percentiles[index - 1], percentiles[index]
    return left_y + (raw_gain - left_x) * (right_y - left_y) / (right_x - left_x)


def _pair_softmin(
    availability: float,
    calibrated_gain: float,
    *,
    power: float,
    epsilon: float,
) -> float:
    if availability == 0.0 or calibrated_gain == 0.0:
        return 0.0
    shifted_inverse = 0.5 * (
        (availability + epsilon) ** (-power) + (calibrated_gain + epsilon) ** (-power)
    )
    value = float(shifted_inverse ** (-1.0 / power) - epsilon)
    return min(max(value, 0.0), 1.0)


def _raise_formula_mismatch(field: str) -> None:
    raise ResultValidationError(
        "Contrast-common calibrated score violates its frozen formula",
        code="contrast_common_score_formula_mismatch",
        field=field,
        remediation="Reject the bundle and rerun its producer",
    )


def _validate_contrast_common_table_lineage(
    table_name: str,
    table: pd.DataFrame,
    applications_by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    *,
    schema_version: str = CROSSFIT_RESULT_SCHEMA_VERSION,
) -> None:
    source_columns = (
        GLOBAL_COMMON_LR_SCORE_COLUMNS
        if table_name == CROSSFIT_CONTRAST_COMMON_LR_TABLE
        else (
            _V5_GLOBAL_COMMON_SENDER_SCORE_COLUMNS
            if schema_version
            in {
                _V4_CROSSFIT_RESULT_SCHEMA_VERSION,
                _V5_CROSSFIT_RESULT_SCHEMA_VERSION,
            }
            else GLOBAL_COMMON_SENDER_SCORE_COLUMNS
        )
    )
    digest_name = (
        "cross_receiver_common_lr_scores"
        if table_name == CROSSFIT_CONTRAST_COMMON_LR_TABLE
        else "cross_receiver_common_sender_scores"
    )
    count_field = (
        "n_lr_rows"
        if table_name == CROSSFIT_CONTRAST_COMMON_LR_TABLE
        else "n_sender_rows"
    )
    digest_field = (
        "lr_scores_digest"
        if table_name == CROSSFIT_CONTRAST_COMMON_LR_TABLE
        else "sender_scores_digest"
    )
    observed_counts = _application_registry_counts(
        table, id_column="global_common_application_id"
    )
    expected_counts = {
        application_id: int(application[count_field])
        for application_id, (_, application) in applications_by_id.items()
    }
    if (
        set(observed_counts).difference(expected_counts)
        or {
            application_id: observed_counts.get(application_id, 0)
            for application_id in expected_counts
        }
        != expected_counts
    ):
        raise ResultValidationError(
            "Contrast-common rows do not match the collection registry",
            code="contrast_common_application_coverage_mismatch",
            field=table_name,
            remediation="Reject the bundle and rerun its producer",
        )
    grouped = table.groupby(
        "global_common_application_id",
        observed=True,
        sort=False,
    )
    for raw_application_id, rows in grouped:
        lineage = applications_by_id.get(str(raw_application_id))
        if lineage is None:
            continue
        collection, application = lineage
        expected_projection: dict[str, object] = {
            "contrast_common_collection_id": collection[
                "contrast_common_collection_id"
            ],
            "fold_id": application["fold_id"],
            "contrast_id": collection["contrast_id"],
            "contrast": collection["contrast"],
            "global_common_functional_id": application["global_common_functional_id"],
            "score_version": collection["score_version"],
            "certification_status": collection["certification_status"],
            "is_oof_certified": collection["is_oof_certified"],
            "claim_scope": collection["claim_scope"],
        }
        for field_name, expected in expected_projection.items():
            observed = rows[field_name].tolist()
            matched = (
                all(type(value) is bool and value is expected for value in observed)
                if type(expected) is bool
                else all(value == expected for value in observed)
            )
            if not matched:
                raise ResultValidationError(
                    "Contrast-common rows mix or misstate collection lineage",
                    code="contrast_common_application_lineage_mismatch",
                    field=f"{table_name}.{field_name}",
                    remediation="Reject the bundle and rerun its producer",
                )
        child_by_receiver = {
            str(child["receiver"]): child
            for child in cast(list[dict[str, Any]], application["receiver_children"])
        }
        binding_by_receiver = {
            str(binding["receiver"]): binding
            for binding in cast(
                list[dict[str, Any]],
                application["receiver_gain_calibration_bindings"],
            )
        }
        for receiver, receiver_rows in rows.groupby("receiver", sort=False):
            receiver_name = str(receiver)
            child = child_by_receiver.get(receiver_name)
            binding = binding_by_receiver.get(receiver_name)
            if (
                child is None
                or binding is None
                or set(receiver_rows["source_family_common_functional_id"].astype(str))
                != {str(child["family_common_functional_id"])}
                or set(receiver_rows["source_family_common_application_id"].astype(str))
                != {str(child["family_common_application_id"])}
            ):
                raise ResultValidationError(
                    "Contrast-common rows misstate receiver-child lineage",
                    code="contrast_common_child_lineage_mismatch",
                    field=table_name,
                    remediation="Reject the bundle and rerun its producer",
                )
            expected_binding_projection = {
                "gain_calibration_binding_id": binding["gain_calibration_binding_id"],
                "gain_calibration_artifact_id": binding["gain_calibration_artifact_id"],
                "gain_calibration_status": binding["gain_calibration_status"],
                "gain_calibration_reason_code": binding["gain_calibration_reason_code"],
            }
            for field_name, expected in expected_binding_projection.items():
                observed = [
                    _optional_string(value)
                    for value in receiver_rows[field_name].tolist()
                ]
                if any(value != expected for value in observed):
                    raise ResultValidationError(
                        "Contrast-common rows misstate gain calibration binding",
                        code="contrast_common_calibration_lineage_mismatch",
                        field=f"{table_name}.{field_name}",
                        remediation="Reject the bundle and rerun its producer",
                    )
            if table_name == CROSSFIT_CONTRAST_COMMON_LR_TABLE:
                power = float(application["softmin_power"])
                epsilon = float(application["epsilon"])
                for row in receiver_rows.itertuples(index=False):
                    raw_gain = _optional_unit_interval(
                        row.receiver_relative_family_gain,
                        field_name="receiver_relative_family_gain",
                    )
                    calibrated = _optional_unit_interval(
                        row.calibrated_family_gain_percentile,
                        field_name="calibrated_family_gain_percentile",
                    )
                    expected_calibrated = _gain_percentile(raw_gain, binding)
                    if not _same_optional_float(calibrated, expected_calibrated):
                        _raise_formula_mismatch(
                            f"{table_name}.calibrated_family_gain_percentile"
                        )
                    availability = _optional_unit_interval(
                        row.availability, field_name="availability"
                    )
                    prior = _optional_unit_interval(
                        row.prior_quality, field_name="prior_quality"
                    )
                    core = _optional_unit_interval(
                        row.global_lr_core_strength,
                        field_name="global_lr_core_strength",
                    )
                    score = _optional_unit_interval(
                        row.global_lr_score, field_name="global_lr_score"
                    )
                    status = str(row.status)
                    reason = _optional_string(row.reason_code)
                    formula_core = (
                        None
                        if availability is None or calibrated is None
                        else _pair_softmin(
                            availability,
                            calibrated,
                            power=power,
                            epsilon=epsilon,
                        )
                    )
                    if status == "observed":
                        if (
                            formula_core is None
                            or formula_core <= 0.0
                            or prior is None
                            or prior <= 0.0
                            or not _same_optional_float(core, formula_core)
                            or not _same_optional_float(score, formula_core * prior)
                        ):
                            _raise_formula_mismatch(f"{table_name}.global_lr_score")
                    elif reason == "prior_quality_zero":
                        if (
                            formula_core is None
                            or prior != 0.0
                            or not _same_optional_float(core, formula_core)
                            or score != 0.0
                        ):
                            _raise_formula_mismatch(f"{table_name}.prior_quality")
                    elif reason == "prior_quality_missing":
                        if (
                            formula_core is None
                            or prior is not None
                            or not _same_optional_float(core, formula_core)
                            or score is not None
                        ):
                            _raise_formula_mismatch(f"{table_name}.prior_quality")
                    elif core is not None and core != 0.0:
                        if formula_core is None or not _same_optional_float(
                            core, formula_core
                        ):
                            _raise_formula_mismatch(
                                f"{table_name}.global_lr_core_strength"
                            )
        if set(rows["receiver"].astype(str)) != set(child_by_receiver):
            raise ResultValidationError(
                "Contrast-common rows lack exact receiver coverage",
                code="contrast_common_application_coverage_mismatch",
                field=table_name,
                remediation="Reject the bundle and rerun its producer",
            )
        if table_name == CROSSFIT_CONTRAST_COMMON_LR_TABLE:
            if set(rows["subject_id"].astype(str)) != set(
                application["heldout_subject_ids"]
            ) or set(rows["context_id"].astype(str)) != set(application["context_ids"]):
                raise ResultValidationError(
                    "Contrast-common LR rows lack exact held-out scope",
                    code="contrast_common_application_coverage_mismatch",
                    field=table_name,
                    remediation="Reject the bundle and rerun its producer",
                )
        else:
            observed_sender_lineages = [
                {
                    "receiver": str(row.receiver),
                    "sender_functional_id": str(row.source_sender_functional_id),
                    "sender_application_id": str(row.source_sender_application_id),
                }
                for row in rows.loc[
                    :,
                    [
                        "receiver",
                        "source_sender_functional_id",
                        "source_sender_application_id",
                    ],
                ]
                .drop_duplicates()
                .sort_values(
                    [
                        "receiver",
                        "source_sender_functional_id",
                        "source_sender_application_id",
                    ],
                    kind="stable",
                )
                .itertuples(index=False)
            ]
            if observed_sender_lineages != application["sender_lineages"]:
                raise ResultValidationError(
                    "Contrast-common sender rows misstate source lineage",
                    code="contrast_common_sender_lineage_mismatch",
                    field=table_name,
                    remediation="Reject the bundle and rerun its producer",
                )
        source_projection = rows.loc[:, list(source_columns)].copy(deep=True)
        observed_digest = _global_source_table_digest(
            digest_name, source_projection, source_columns
        )
        if observed_digest != application[digest_field]:
            raise ResultValidationError(
                "Contrast-common source table digest is inconsistent",
                code="contrast_common_source_digest_mismatch",
                field=table_name,
                remediation="Reject the bundle and rerun its producer",
            )


def _validate_sender_lr_cross_table_lineage_legacy(
    lr_table: pd.DataFrame,
    sender_table: pd.DataFrame,
    *,
    conserved_sender: bool = True,
    validate_conserved_groups: bool = True,
) -> None:
    keys = (
        "global_common_application_id",
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "family_id",
        "driver_id",
        "interaction_id",
        "mode",
    )
    lr_by_key = {
        tuple(str(getattr(row, key)) for key in keys): row
        for row in lr_table.itertuples(index=False)
    }
    for sender in sender_table.itertuples(index=False):
        key = tuple(str(getattr(sender, name)) for name in keys)
        lr = lr_by_key.get(key)
        if lr is None:
            raise ResultValidationError(
                "Sender-LR row has no corresponding LR parent",
                code="contrast_common_sender_lr_parent_mismatch",
                field=CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
                remediation="Reject the bundle and rerun its producer",
            )
        pairs = (
            (sender.global_lr_score, lr.global_lr_score, "global_lr_score"),
            (
                sender.gain_calibration_binding_id,
                lr.gain_calibration_binding_id,
                "gain_calibration_binding_id",
            ),
            (
                sender.gain_calibration_artifact_id,
                lr.gain_calibration_artifact_id,
                "gain_calibration_artifact_id",
            ),
            (
                sender.gain_calibration_status,
                lr.gain_calibration_status,
                "gain_calibration_status",
            ),
            (
                sender.gain_calibration_reason_code,
                lr.gain_calibration_reason_code,
                "gain_calibration_reason_code",
            ),
        )
        for observed, expected, field_name in pairs:
            if field_name == "global_lr_score":
                matched = _same_optional_float(
                    _optional_float(observed, field_name=field_name),
                    _optional_float(expected, field_name=field_name),
                )
            else:
                matched = _optional_string(observed) == _optional_string(expected)
            if not matched:
                raise ResultValidationError(
                    "Sender-LR row misstates its LR/calibration parent",
                    code="contrast_common_sender_lr_parent_mismatch",
                    field=f"{CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE}.{field_name}",
                    remediation="Reject the bundle and rerun its producer",
                )
        lr_status = str(lr.status)
        sender_status = str(sender.status)
        lr_score = _optional_unit_interval(
            lr.global_lr_score, field_name="global_lr_score"
        )
        raw_sender = _optional_unit_interval(
            sender.raw_sender_evidence, field_name="raw_sender_evidence"
        )
        assignment_weight = (
            _optional_unit_interval(
                sender.assignment_weight, field_name="assignment_weight"
            )
            if conserved_sender
            else None
        )
        sender_score = _optional_unit_interval(
            sender.global_sender_lr_score, field_name="global_sender_lr_score"
        )
        if lr_status == "structural_zero":
            valid = sender_status == "structural_zero" and sender_score == 0.0
        elif lr_status == "not_estimable":
            valid = sender_status == "not_estimable" and sender_score is None
        elif conserved_sender and assignment_weight is None:
            valid = sender_status == "not_estimable" and sender_score is None
        elif conserved_sender and assignment_weight == 0.0:
            valid = sender_status == "structural_zero" and sender_score == 0.0
        elif not conserved_sender and raw_sender is None:
            valid = sender_status == "not_estimable" and sender_score is None
        elif not conserved_sender and raw_sender == 0.0:
            valid = sender_status == "structural_zero" and sender_score == 0.0
        else:
            multiplier = assignment_weight if conserved_sender else raw_sender
            valid = (
                lr_score is not None
                and multiplier is not None
                and sender_status == "observed"
                and _same_optional_float(sender_score, lr_score * multiplier)
            )
        if not valid:
            _raise_formula_mismatch(
                f"{CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE}.global_sender_lr_score"
            )
    if conserved_sender and validate_conserved_groups:
        try:
            _validate_conserved_sender_groups(sender_table)
        except ValueError as error:
            raise ResultValidationError(
                "Conserved sender groups violate LR allocation semantics",
                code="contrast_common_sender_conservation_mismatch",
                field=CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
                remediation="Reject the bundle and rerun its producer",
            ) from error


def _validate_sender_lr_cross_table_lineage(
    lr_table: pd.DataFrame,
    sender_table: pd.DataFrame,
    *,
    conserved_sender: bool = True,
    validate_conserved_groups: bool = True,
) -> None:
    keys = (
        "global_common_application_id",
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "family_id",
        "driver_id",
        "interaction_id",
        "mode",
    )

    def standard_key(series: pd.Series) -> bool:
        return _standard_required_string_array(series) is not None

    numeric_columns = (
        "global_lr_score",
        "raw_sender_evidence",
        *(("assignment_weight",) if conserved_sender else ()),
        "global_sender_lr_score",
    )
    standard_numeric = all(
        pd.api.types.is_numeric_dtype(sender_table[column].dtype)
        and not pd.api.types.is_complex_dtype(sender_table[column].dtype)
        and not pd.api.types.is_datetime64_any_dtype(sender_table[column].dtype)
        and not pd.api.types.is_timedelta64_dtype(sender_table[column].dtype)
        for column in numeric_columns
    ) and (
        pd.api.types.is_numeric_dtype(lr_table["global_lr_score"].dtype)
        and not pd.api.types.is_complex_dtype(lr_table["global_lr_score"].dtype)
    )
    if (
        not standard_numeric
        or not all(standard_key(lr_table[key]) for key in keys)
        or not all(standard_key(sender_table[key]) for key in keys)
    ):
        _validate_sender_lr_cross_table_lineage_legacy(
            lr_table,
            sender_table,
            conserved_sender=conserved_sender,
            validate_conserved_groups=validate_conserved_groups,
        )
        return

    parent_fields = (
        "global_lr_score",
        "gain_calibration_binding_id",
        "gain_calibration_artifact_id",
        "gain_calibration_status",
        "gain_calibration_reason_code",
        "status",
    )
    parent = lr_table.loc[:, [*keys, *parent_fields]].rename(
        columns={field: f"parent_{field}" for field in parent_fields}
    )
    sender_fields = (
        "global_lr_score",
        "gain_calibration_binding_id",
        "gain_calibration_artifact_id",
        "gain_calibration_status",
        "gain_calibration_reason_code",
        "raw_sender_evidence",
        *(("assignment_weight",) if conserved_sender else ()),
        "global_sender_lr_score",
        "status",
    )
    merged = sender_table.loc[:, [*keys, *sender_fields]].merge(
        parent,
        on=list(keys),
        how="left",
        sort=False,
        validate="many_to_one",
        indicator=True,
    )
    if not bool(merged["_merge"].eq("both").all()):
        raise ResultValidationError(
            "Sender-LR row has no corresponding LR parent",
            code="contrast_common_sender_lr_parent_mismatch",
            field=CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
            remediation="Reject the bundle and rerun its producer",
        )

    numeric_pairs = _finite_float_matrix(
        merged,
        (
            "global_lr_score",
            "parent_global_lr_score",
            "raw_sender_evidence",
            *(("assignment_weight",) if conserved_sender else ()),
            "global_sender_lr_score",
        ),
    )
    for index, field_name in enumerate(
        (
            "global_lr_score",
            "parent_global_lr_score",
            "raw_sender_evidence",
            *(("assignment_weight",) if conserved_sender else ()),
            "global_sender_lr_score",
        )
    ):
        values = numeric_pairs[:, index]
        present = ~np.isnan(values)
        if bool(np.any((values[present] < 0.0) | (values[present] > 1.0))):
            public_name = field_name.removeprefix("parent_")
            raise ValueError(f"{public_name} must lie in [0, 1] or be missing")
    sender_lr_score = numeric_pairs[:, 0]
    parent_lr_score = numeric_pairs[:, 1]
    sender_lr_present = ~np.isnan(sender_lr_score)
    parent_lr_present = ~np.isnan(parent_lr_score)
    lr_matches = sender_lr_present == parent_lr_present
    both_lr_present = sender_lr_present & parent_lr_present
    lr_matches[both_lr_present] &= _math_isclose_array(
        sender_lr_score[both_lr_present],
        parent_lr_score[both_lr_present],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    if not bool(lr_matches.all()):
        raise ResultValidationError(
            "Sender-LR row misstates its LR/calibration parent",
            code="contrast_common_sender_lr_parent_mismatch",
            field=(
                f"{CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE}.global_lr_score"
            ),
            remediation="Reject the bundle and rerun its producer",
        )

    for field_name in (
        "gain_calibration_binding_id",
        "gain_calibration_artifact_id",
        "gain_calibration_status",
        "gain_calibration_reason_code",
    ):
        observed = merged[field_name]
        expected = merged[f"parent_{field_name}"]
        observed_present = _standard_optional_string_presence(observed)
        expected_present = _standard_optional_string_presence(expected)
        if observed_present is None or expected_present is None:
            _validate_sender_lr_cross_table_lineage_legacy(
                lr_table,
                sender_table,
                conserved_sender=conserved_sender,
                validate_conserved_groups=validate_conserved_groups,
            )
            return
        presence_matches = np.equal(observed_present, expected_present)
        both_present = observed_present & expected_present
        value_matches = observed.astype(str).eq(expected.astype(str)).to_numpy(
            dtype=bool, copy=False
        )
        if not bool((presence_matches & (~both_present | value_matches)).all()):
            raise ResultValidationError(
                "Sender-LR row misstates its LR/calibration parent",
                code="contrast_common_sender_lr_parent_mismatch",
                field=f"{CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE}.{field_name}",
                remediation="Reject the bundle and rerun its producer",
            )

    raw_sender_index = 2
    assignment_index = 3 if conserved_sender else None
    sender_score_index = 4 if conserved_sender else 3
    raw_sender = numeric_pairs[:, raw_sender_index]
    assignment_weight = (
        numeric_pairs[:, assignment_index]
        if assignment_index is not None
        else np.full(len(merged), np.nan, dtype=float)
    )
    sender_score = numeric_pairs[:, sender_score_index]
    sender_score_present = ~np.isnan(sender_score)
    sender_status = _required_string_array(merged["status"])
    lr_status = _required_string_array(merged["parent_status"])
    valid: np.ndarray = np.zeros(len(merged), dtype=bool)
    parent_zero = lr_status == "structural_zero"
    valid[parent_zero] = (
        (sender_status[parent_zero] == "structural_zero")
        & sender_score_present[parent_zero]
        & (sender_score[parent_zero] == 0.0)
    )
    parent_not_estimable = lr_status == "not_estimable"
    valid[parent_not_estimable] = (
        (sender_status[parent_not_estimable] == "not_estimable")
        & ~sender_score_present[parent_not_estimable]
    )
    remaining = ~(parent_zero | parent_not_estimable)
    multiplier = assignment_weight if conserved_sender else raw_sender
    multiplier_missing = remaining & np.isnan(multiplier)
    valid[multiplier_missing] = (
        (sender_status[multiplier_missing] == "not_estimable")
        & ~sender_score_present[multiplier_missing]
    )
    multiplier_zero = remaining & ~np.isnan(multiplier) & (multiplier == 0.0)
    valid[multiplier_zero] = (
        (sender_status[multiplier_zero] == "structural_zero")
        & sender_score_present[multiplier_zero]
        & (sender_score[multiplier_zero] == 0.0)
    )
    formula_rows = remaining & ~np.isnan(multiplier) & (multiplier > 0.0)
    valid[formula_rows] = (
        parent_lr_present[formula_rows]
        & (sender_status[formula_rows] == "observed")
        & sender_score_present[formula_rows]
        & _math_isclose_array(
            sender_score[formula_rows],
            parent_lr_score[formula_rows] * multiplier[formula_rows],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    )
    if not bool(valid.all()):
        _raise_formula_mismatch(
            f"{CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE}.global_sender_lr_score"
        )
    if conserved_sender and validate_conserved_groups:
        try:
            _validate_conserved_sender_groups(sender_table)
        except ValueError as error:
            raise ResultValidationError(
                "Conserved sender groups violate LR allocation semantics",
                code="contrast_common_sender_conservation_mismatch",
                field=CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
                remediation="Reject the bundle and rerun its producer",
            ) from error


def _validate_contrast_common_registry_links(
    manifest: dict[str, object],
    tables: dict[str, pd.DataFrame],
    *,
    tables_prevalidated: bool = False,
) -> None:
    schema_version = str(manifest.get("schema_version"))
    _, applications_by_id = _contrast_common_registries(manifest)
    if manifest.get("schema_version") == _V3_CROSSFIT_RESULT_SCHEMA_VERSION:
        _validate_v3_source_contrast_common_registry(manifest, applications_by_id)
        _validate_v3_contrast_common_table_lineage(
            CROSSFIT_CONTRAST_COMMON_LR_TABLE,
            tables[CROSSFIT_CONTRAST_COMMON_LR_TABLE],
            applications_by_id,
        )
        _validate_v3_contrast_common_table_lineage(
            CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
            tables[CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE],
            applications_by_id,
        )
        return
    _validate_source_contrast_common_registry(manifest, applications_by_id)
    _validate_contrast_common_table_lineage(
        CROSSFIT_CONTRAST_COMMON_LR_TABLE,
        tables[CROSSFIT_CONTRAST_COMMON_LR_TABLE],
        applications_by_id,
        schema_version=schema_version,
    )
    _validate_contrast_common_table_lineage(
        CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
        tables[CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE],
        applications_by_id,
        schema_version=schema_version,
    )
    _validate_sender_lr_cross_table_lineage(
        tables[CROSSFIT_CONTRAST_COMMON_LR_TABLE],
        tables[CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE],
        conserved_sender=schema_version
        in {
            CROSSFIT_RESULT_SCHEMA_VERSION,
            _V8_CROSSFIT_RESULT_SCHEMA_VERSION,
            _V7_CROSSFIT_RESULT_SCHEMA_VERSION,
            _V6_CROSSFIT_RESULT_SCHEMA_VERSION,
        },
        validate_conserved_groups=not tables_prevalidated,
    )


def _validate_directional_registry_links(
    manifest: dict[str, object],
    table: pd.DataFrame,
) -> None:
    schema_version = manifest.get("schema_version")
    is_v7_plus = schema_version in _V7_PLUS_CROSSFIT_RESULT_SCHEMA_VERSIONS
    source = manifest.get("source_crossfit_manifest")
    if not isinstance(source, Mapping):
        raise ResultValidationError(
            "Source cross-fit manifest is unavailable for directional validation",
            code="invalid_crossfit_directional_channel_registry",
            field="source_crossfit_manifest",
            remediation="Reject the bundle and rerun its producer",
        )
    raw_folds = source.get("fold_artifacts")
    if not isinstance(raw_folds, list):
        raise ResultValidationError(
            "Source directional fold registry is missing or invalid",
            code="invalid_crossfit_directional_channel_registry",
            field="fold_artifacts",
            remediation="Reject the bundle and rerun its producer",
        )
    expected_rows: list[dict[str, object]] = []
    binding_ids: list[str] = []
    statuses: list[str] = []
    fold_ids: list[str] = []
    try:
        supported_receivers_by_fold: dict[str, set[str]] = {}
        receiver_ids: tuple[str, ...] = ()
        if is_v7_plus:
            receiver_ids, receiver_support = _validate_source_receiver_universe(
                manifest
            )
            supported_receivers_by_fold = {
                str(fold_id): set(
                    rows.loc[
                        rows["receiver_training_support_status"].eq("observed"),
                        "receiver",
                    ].astype(str)
                )
                for fold_id, rows in receiver_support.groupby(
                    "fold_id", sort=False, dropna=False
                )
            }
        spec = _directional_mapping(source.get("spec"), field_name="source.spec")
        raw_pair_specs = spec.get("directional_pairs")
        stage_connected = bool(raw_pair_specs)
        pair_specs_by_id: dict[str, dict[str, Any]] = {}
        pair_contrast_names: dict[str, tuple[str, str]] = {}
        if stage_connected:
            if not isinstance(raw_pair_specs, list):
                raise ValueError("source.spec.directional_pairs must be an array")
            for raw_pair_spec in raw_pair_specs:
                pair_spec, contrasts = _directional_pair_spec_projection(raw_pair_spec)
                pair_spec_id = str(pair_spec["pair_spec_id"])
                if pair_spec_id in pair_specs_by_id:
                    raise ValueError("source directional pair specs must be unique")
                pair_specs_by_id[pair_spec_id] = pair_spec
                pair_contrast_names[pair_spec_id] = (
                    str(contrasts["forward"]["name"]),
                    str(contrasts["reverse"]["name"]),
                )
        for raw_fold in raw_folds:
            fold = _directional_mapping(raw_fold, field_name="fold_artifacts")
            fold_id = _manifest_string(
                fold.get("fold_id"), field_name="fold_artifacts.fold_id"
            )
            fold_ids.append(fold_id)
            bindings = fold.get("directional_response_bindings")
            if bindings is None and not stage_connected:
                bindings = []
            if not isinstance(bindings, list):
                raise ValueError(
                    "fold_artifacts.directional_response_bindings must be an array"
                )
            observed_fold_keys: set[tuple[str, str]] = set()
            for binding in bindings:
                binding_record = _directional_mapping(
                    binding, field_name="directional binding"
                )
                pair_spec_id = str(binding_record.get("pair_spec_id"))
                if (
                    pair_spec_id not in pair_specs_by_id
                    or binding_record.get("pair_spec") != pair_specs_by_id[pair_spec_id]
                ):
                    raise ValueError(
                        "directional binding pair spec disagrees with source spec"
                    )
                rows = _directional_binding_rows(
                    binding_record,
                    crossfit_id=str(manifest["crossfit_id"]),
                    spec_id=str(manifest["spec_id"]),
                    repeat_id=str(manifest["repeat_id"]),
                    fold_id=fold_id,
                    certification_status=str(manifest["certification_status"]),
                    is_oof_certified=bool(manifest["complete_pipeline_oof_certified"]),
                    claim_scope=str(manifest["claim_scope"]),
                )
                expected_rows.extend(rows)
                binding_ids.append(str(rows[0]["binding_id"]))
                statuses.append(str(rows[0]["status"]))
                observed_fold_keys.add(
                    (str(rows[0]["pair_spec_id"]), str(rows[0]["receiver"]))
                )
            if stage_connected:
                incremental_records = fold.get("receiver_incremental_artifacts")
                if not isinstance(incremental_records, list):
                    raise ValueError(
                        "fold receiver incremental registry must be an array"
                    )
                contrast_receivers: dict[str, set[str]] = {}
                for raw_record in incremental_records:
                    record = _directional_mapping(
                        raw_record,
                        field_name="receiver_incremental_artifacts",
                    )
                    contrast_name = _manifest_string(
                        record.get("contrast_name"),
                        field_name="receiver_incremental_artifacts.contrast_name",
                    )
                    receiver = _manifest_string(
                        record.get("receiver"),
                        field_name="receiver_incremental_artifacts.receiver",
                    )
                    contrast_receivers.setdefault(contrast_name, set()).add(receiver)
                expected_fold_keys: set[tuple[str, str]] = set()
                for pair_spec_id, (
                    forward_contrast,
                    reverse_contrast,
                ) in pair_contrast_names.items():
                    forward_receivers = contrast_receivers.get(forward_contrast, set())
                    reverse_receivers = contrast_receivers.get(reverse_contrast, set())
                    if forward_receivers != reverse_receivers or (
                        not is_v7_plus and not forward_receivers
                    ):
                        raise ValueError(
                            "directional pair lacks symmetric receiver model coverage"
                        )
                    if (
                        is_v7_plus
                        and forward_receivers
                        != supported_receivers_by_fold.get(fold_id)
                    ):
                        raise ValueError(
                            "directional pair receiver models disagree with "
                            "outer-training support"
                        )
                    expected_fold_keys.update(
                        (pair_spec_id, receiver) for receiver in forward_receivers
                    )
                if observed_fold_keys != expected_fold_keys:
                    raise ValueError(
                        "directional bindings do not cover planned pair/receiver scopes"
                    )
        if len(fold_ids) != len(set(fold_ids)):
            raise ValueError("source directional fold IDs must be unique")
        if len(binding_ids) != len(set(binding_ids)):
            raise ValueError("source directional binding IDs must be unique")
        directional_summary_fields = {
            "directional_pair_stage_connected",
            "n_directional_response_bindings",
            "directional_binding_status_counts",
            "directional_registry_complete",
            "directional_supported_binding_registry_complete",
            "n_directional_receiver_pair_opportunities",
            "n_directional_receiver_pair_absent_training",
            "directional_receiver_universe_id",
            "directional_receiver_axis_id",
            "directional_diagnostic_complete",
            "directional_formal_inference_allowed",
        }
        v7_directional_summary_fields = {
            "directional_supported_binding_registry_complete",
            "n_directional_receiver_pair_opportunities",
            "n_directional_receiver_pair_absent_training",
            "directional_receiver_universe_id",
            "directional_receiver_axis_id",
        }
        if stage_connected:
            if source.get("directional_formal_inference_allowed") is not False:
                raise ValueError("directional formal inference must remain disabled")
            if source.get("n_directional_response_bindings") != len(binding_ids):
                raise ValueError("directional binding count is inconsistent")
            expected_status_counts = {
                status: statuses.count(status) for status in sorted(set(statuses))
            }
            if source.get("directional_binding_status_counts") != (
                expected_status_counts
            ):
                raise ValueError("directional binding status counts are inconsistent")
            if source.get("directional_pair_stage_connected") is not True:
                raise ValueError("directional stage connection flag is inconsistent")
            if source.get("directional_diagnostic_complete") is not all(
                status == "observed" for status in statuses
            ):
                raise ValueError(
                    "directional diagnostic completion flag is inconsistent"
                )
            if is_v7_plus:
                n_pairs = len(pair_specs_by_id)
                n_opportunities = n_pairs * len(receiver_ids) * len(fold_ids)
                n_supported_opportunities = n_pairs * sum(
                    len(supported_receivers_by_fold.get(fold_id, set()))
                    for fold_id in fold_ids
                )
                n_absent_training = n_opportunities - n_supported_opportunities
                if (
                    source.get("n_directional_receiver_pair_opportunities")
                    != n_opportunities
                    or source.get("n_directional_receiver_pair_absent_training")
                    != n_absent_training
                    or source.get("directional_receiver_universe_id")
                    != manifest.get("receiver_universe_id")
                    or source.get("directional_receiver_axis_id")
                    != manifest.get("receiver_axis_id")
                ):
                    raise ValueError(
                        "directional receiver opportunity lineage is inconsistent"
                    )
                full_registry_complete = len(binding_ids) == n_opportunities
                supported_registry_complete = (
                    len(binding_ids) == n_supported_opportunities
                )
                if source.get("directional_registry_complete") is not (
                    full_registry_complete
                ):
                    raise ValueError(
                        "directional full-axis registry completion flag is invalid"
                    )
                if (
                    source.get("directional_supported_binding_registry_complete")
                    is not supported_registry_complete
                ):
                    raise ValueError(
                        "directional supported registry completion flag is invalid"
                    )
            else:
                if v7_directional_summary_fields.intersection(source):
                    raise ValueError(
                        "legacy directional registry claims v7 receiver lineage"
                    )
                if source.get("directional_registry_complete") is not True:
                    raise ValueError("directional registry completion flag is invalid")
        elif binding_ids or directional_summary_fields.intersection(source):
            raise ValueError("unplanned directional registry metadata is present")
    except (KeyError, TypeError, ValueError) as error:
        raise ResultValidationError(
            "Source directional channel registry violates its contract",
            code="invalid_crossfit_directional_channel_registry",
            field="directional_response_bindings",
            remediation="Reject the bundle and rerun its producer",
        ) from error
    expected = pd.DataFrame(
        expected_rows,
        columns=CROSSFIT_DIRECTIONAL_CHANNEL_COLUMNS,
    )
    if not expected.empty:
        expected = expected.sort_values(
            ["pair_spec_id", "fold_id", "receiver", "channel_role"],
            kind="stable",
            ignore_index=True,
        )
    observed = table.sort_values(
        ["pair_spec_id", "fold_id", "receiver", "channel_role"],
        kind="stable",
        ignore_index=True,
    )
    expected_records = [
        [_canonical_table_scalar(value) for value in row]
        for row in expected.itertuples(index=False, name=None)
    ]
    observed_records = [
        [_canonical_table_scalar(value) for value in row]
        for row in observed.itertuples(index=False, name=None)
    ]
    if expected_records != observed_records:
        raise ResultValidationError(
            "Directional channel rows do not exactly match source bindings",
            code="crossfit_result_directional_registry_mismatch",
            field=CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE,
            remediation="Reject the bundle and rerun its producer",
        )


def _validate_receiver_training_support_links(
    manifest: dict[str, object],
    table: pd.DataFrame,
) -> None:
    _, expected = _validate_source_receiver_universe(manifest)
    try:
        observed = _validate_receiver_training_support_table(table)
    except (TypeError, ValueError) as error:
        raise ResultValidationError(
            "Receiver training-support rows violate their table contract",
            code="invalid_crossfit_receiver_training_support",
            field=CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE,
            remediation="Reject the bundle and rerun its producer",
        ) from error
    expected = expected.sort_values(
        ["fold_id", "receiver"], kind="stable", ignore_index=True
    )
    observed = observed.sort_values(
        ["fold_id", "receiver"], kind="stable", ignore_index=True
    )
    expected_records = [
        [_canonical_table_scalar(value) for value in row]
        for row in expected.itertuples(index=False, name=None)
    ]
    observed_records = [
        [_canonical_table_scalar(value) for value in row]
        for row in observed.itertuples(index=False, name=None)
    ]
    if expected_records != observed_records:
        raise ResultValidationError(
            "Receiver training-support rows do not exactly match source lineage",
            code="crossfit_result_receiver_training_support_mismatch",
            field=CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE,
            remediation="Reject the bundle and rerun its producer",
        )


def _semantic_expected_integrated_lr(
    manifest: Mapping[str, object],
    tables: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    components = tables[CROSSFIT_COMPONENT_TABLE]
    selected = components.loc[
        components["component_scope"].eq("lr_member")
        & components["component"].eq("sender_unresolved_strength")
    ].copy(deep=True)
    directional = tables[CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE]
    if not directional.empty:
        directional_ids = set(directional["contrast_id"].astype(str))
        directional_names = set(directional["contrast"].astype(str))
        selected = selected.loc[
            ~selected["contrast_id"].astype(str).isin(directional_ids)
            & ~selected["contrast"].astype(str).isin(directional_names)
        ].copy(deep=True)
    source_digests = {
        str(application["family_common_application_id"]): str(
            cast(dict[str, object], application["source_table_digests"])[
                "member_scores"
            ]
        )
        for application in cast(list[dict[str, object]], manifest["applications"])
    }
    expected = pd.DataFrame(columns=SEMANTIC_INTEGRATED_LR_COLUMNS)
    if selected.empty:
        return expected
    direct_columns = (
        "crossfit_id",
        "repeat_id",
        "fold_id",
        "contrast_id",
        "contrast",
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "family_id",
        "driver_id",
        "interaction_id",
        "mode",
        "status",
        "reason_code",
        "family_common_functional_id",
        "family_common_application_id",
        "family_common_binding_id",
        "score_version",
    )
    expected = selected.loc[:, list(direct_columns)].copy(deep=True)
    expected["crossfit_spec_id"] = selected["spec_id"]
    expected["integrated_lr_score"] = selected["component_value"]
    expected["source_component"] = "sender_unresolved_strength"
    expected["source_table_digest"] = (
        selected["family_common_application_id"].astype(str).map(source_digests)
    )
    if expected["source_table_digest"].isna().any():
        raise ResultValidationError(
            "Semantic integrated-LR rows lack source table lineage",
            code="crossfit_semantic_source_lineage_mismatch",
            field=CROSSFIT_SEMANTIC_INTEGRATED_LR_TABLE,
            remediation="Reject the bundle and rerun its producer",
        )
    expected["formal_inference_allowed"] = False
    return expected.loc[:, list(SEMANTIC_INTEGRATED_LR_COLUMNS)].sort_values(
        [
            "fold_id",
            "contrast_id",
            "sample_id",
            "receiver",
            "family_id",
            "driver_id",
            "interaction_id",
            "mode",
        ],
        kind="stable",
        ignore_index=True,
    )


def _semantic_expected_differential(
    manifest: Mapping[str, object],
    tables: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    selected = tables[CROSSFIT_DIFFERENTIAL_TABLE].copy(deep=True)
    directional = tables[CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE]
    if not directional.empty:
        directional_ids = set(directional["contrast_id"].astype(str))
        directional_names = set(directional["contrast"].astype(str))
        selected = selected.loc[
            ~selected["contrast_id"].astype(str).isin(directional_ids)
            & ~selected["contrast"].astype(str).isin(directional_names)
        ].copy(deep=True)
    source_digests = {
        str(application["family_common_application_id"]): str(
            cast(dict[str, object], application["source_table_digests"])[
                "subject_differential"
            ]
        )
        for application in cast(list[dict[str, object]], manifest["applications"])
    }
    expected = pd.DataFrame(columns=SEMANTIC_DIFFERENTIAL_EFFECT_COLUMNS)
    if selected.empty:
        return expected
    direct_columns = (
        "crossfit_id",
        "repeat_id",
        "fold_id",
        "contrast_id",
        "contrast",
        "receiver",
        "subject_id",
        "family_id",
        "differential_effect",
        "status",
        "reason_code",
        "effect_semantics",
        "family_common_functional_id",
        "family_common_application_id",
        "family_common_binding_id",
        "score_version",
    )
    expected = selected.loc[:, list(direct_columns)].copy(deep=True)
    expected["crossfit_spec_id"] = selected["spec_id"]
    expected["source_table_digest"] = (
        selected["family_common_application_id"].astype(str).map(source_digests)
    )
    if expected["source_table_digest"].isna().any():
        raise ResultValidationError(
            "Semantic differential rows lack source table lineage",
            code="crossfit_semantic_source_lineage_mismatch",
            field=CROSSFIT_SEMANTIC_DIFFERENTIAL_EFFECT_TABLE,
            remediation="Reject the bundle and rerun its producer",
        )
    expected["formal_inference_allowed"] = False
    return expected.loc[:, list(SEMANTIC_DIFFERENTIAL_EFFECT_COLUMNS)].sort_values(
        ["fold_id", "contrast_id", "receiver", "subject_id", "family_id"],
        kind="stable",
        ignore_index=True,
    )


def _validate_semantic_source_lineage(
    manifest: Mapping[str, object],
    tables: Mapping[str, pd.DataFrame],
) -> None:
    source = cast(dict[str, object], manifest["source_crossfit_manifest"])
    folds = cast(list[dict[str, object]], source["fold_artifacts"])
    fold_by_id = {str(fold["fold_id"]): fold for fold in folds}
    availability = tables[CROSSFIT_SEMANTIC_AVAILABILITY_TABLE]
    for fold_id, group in availability.groupby("fold_id", observed=True, sort=False):
        source_fold = fold_by_id.get(str(fold_id))
        if source_fold is None or set(group["training_artifact_id"].astype(str)) != {
            str(source_fold["training_artifact_id"])
        }:
            raise ResultValidationError(
                "Semantic availability training lineage is inconsistent",
                code="crossfit_semantic_source_lineage_mismatch",
                field=CROSSFIT_SEMANTIC_AVAILABILITY_TABLE,
                remediation="Reject the bundle and rerun its producer",
            )
        for field_name in (
            "availability_application_id",
            "filter_universe_id",
            "source_table_digest",
        ):
            values = set(group[field_name].astype(str))
            if len(values) != 1 or any(not value for value in values):
                raise ResultValidationError(
                    "Semantic availability source lineage is not fold-unique",
                    code="crossfit_semantic_source_lineage_mismatch",
                    field=f"{CROSSFIT_SEMANTIC_AVAILABILITY_TABLE}.{field_name}",
                    remediation="Reject the bundle and rerun its producer",
                )

    program_records: dict[tuple[str, str, str], dict[str, object]] = {}
    for fold in folds:
        fold_id = str(fold["fold_id"])
        for raw_record in cast(
            list[dict[str, object]], fold["receiver_program_artifacts"]
        ):
            key = (
                fold_id,
                str(raw_record["contrast_name"]),
                str(raw_record["receiver"]),
            )
            program_records[key] = raw_record
    programs = tables[CROSSFIT_SEMANTIC_RECEIVER_PROGRAM_TABLE]
    for application_id, group in programs.groupby(
        "receiver_program_application_id", observed=True, sort=False
    ):
        first = group.iloc[0]
        key = (str(first["fold_id"]), str(first["contrast"]), str(first["receiver"]))
        record = program_records.get(key)
        if (
            record is None
            or str(application_id) != str(record["application_id"])
            or set(group["receiver_program_training_artifact_id"].astype(str))
            != {str(record["training_artifact_id"])}
        ):
            raise ResultValidationError(
                "Semantic receiver-program lineage is inconsistent",
                code="crossfit_semantic_source_lineage_mismatch",
                field=CROSSFIT_SEMANTIC_RECEIVER_PROGRAM_TABLE,
                remediation="Reject the bundle and rerun its producer",
            )
        source_table = group.loc[:, list(_RECEIVER_PROGRAM_SOURCE_COLUMNS)].copy(
            deep=True
        )
        expected_digest = _semantic_table_digest("receiver_program_score", source_table)
        if set(group["source_table_digest"].astype(str)) != {expected_digest}:
            raise ResultValidationError(
                "Semantic receiver-program source digest is inconsistent",
                code="crossfit_semantic_source_lineage_mismatch",
                field=(
                    f"{CROSSFIT_SEMANTIC_RECEIVER_PROGRAM_TABLE}.source_table_digest"
                ),
                remediation="Reject the bundle and rerun its producer",
            )


def _validate_semantic_registry_links(
    manifest: Mapping[str, object],
    tables: Mapping[str, pd.DataFrame],
    *,
    tables_prevalidated: bool = False,
) -> None:
    collection = _validate_semantic_collection_manifest(manifest)
    digests: dict[str, str] = {
        str(item[0]): str(item[1])
        for item in cast(list[list[object]], collection["table_digests"])
    }
    counts: dict[str, int] = {
        str(item[0]): int(cast(Any, item[1]))
        for item in cast(list[list[object]], collection["table_row_counts"])
    }
    for semantic_output, table_name in _SEMANTIC_OUTPUT_TO_TABLE.items():
        table = tables[table_name]
        if not tables_prevalidated:
            _validate_semantic_table_contract(
                semantic_output,
                table,
                crossfit_id=str(manifest["crossfit_id"]),
                crossfit_spec_id=str(manifest["spec_id"]),
                repeat_id=str(manifest["repeat_id"]),
            )
        mismatched = len(table) != int(counts[semantic_output])
        if not tables_prevalidated:
            mismatched = mismatched or _semantic_table_digest(
                semantic_output, table
            ) != str(digests[semantic_output])
        if mismatched:
            raise ResultValidationError(
                "Semantic score table disagrees with its collection manifest",
                code="crossfit_semantic_collection_mismatch",
                field=table_name,
                remediation="Reject the bundle and rerun its producer",
            )
    expected_integrated = _semantic_expected_integrated_lr(manifest, tables)
    observed_integrated = tables[CROSSFIT_SEMANTIC_INTEGRATED_LR_TABLE]
    expected_differential = _semantic_expected_differential(manifest, tables)
    observed_differential = tables[CROSSFIT_SEMANTIC_DIFFERENTIAL_EFFECT_TABLE]
    for semantic_output, table_name, expected, observed in (
        (
            "integrated_lr_score",
            CROSSFIT_SEMANTIC_INTEGRATED_LR_TABLE,
            expected_integrated,
            observed_integrated,
        ),
        (
            "differential_effect",
            CROSSFIT_SEMANTIC_DIFFERENTIAL_EFFECT_TABLE,
            expected_differential,
            observed_differential,
        ),
    ):
        if _semantic_table_digest(semantic_output, expected) != _semantic_table_digest(
            semantic_output, observed
        ):
            raise ResultValidationError(
                "Semantic score rows do not replay their canonical ledger producer",
                code="crossfit_semantic_replay_mismatch",
                field=table_name,
                remediation="Reject the bundle and rerun its producer",
            )
    _validate_semantic_source_lineage(manifest, tables)


def _validate_receiver_family_opportunity_links(
    manifest: dict[str, object],
    tables: Mapping[str, pd.DataFrame],
) -> None:
    """Constrain every persisted receiver-family row to the v9 root grid."""

    contract = _validate_source_receiver_family_opportunity_universe(manifest)
    universe = cast(dict[str, object], contract["universe"])
    allowed_pairs = {
        (str(item["receiver"]), str(item["family_id"]))
        for item in cast(list[dict[str, object]], universe["opportunities"])
    }
    family_by_driver = cast(dict[str, str], contract["family_by_driver"])
    receiver_ids = set(cast(tuple[str, ...], contract["receiver_ids"]))
    for application in cast(list[dict[str, object]], manifest["applications"]):
        if str(application["receiver"]) not in receiver_ids:
            raise ResultValidationError(
                "Application receiver lies outside the frozen opportunity universe",
                code="crossfit_result_receiver_family_universe_mismatch",
                field="applications.receiver",
                remediation="Reject the bundle and rerun its producer",
            )
    for table_name, table in tables.items():
        if table.empty or not {"receiver", "family_id"}.issubset(table.columns):
            continue
        if table[["receiver", "family_id"]].isna().any(axis=None):
            raise ResultValidationError(
                "Persisted receiver-family rows contain null opportunity keys",
                code="crossfit_result_receiver_family_universe_mismatch",
                field=table_name,
                remediation="Reject the bundle and rerun its producer",
            )
        observed_pairs = set(
            table.loc[:, ["receiver", "family_id"]]
            .drop_duplicates()
            .astype(str)
            .itertuples(index=False, name=None)
        )
        if not observed_pairs.issubset(allowed_pairs):
            raise ResultValidationError(
                "Persisted receiver-family rows exceed the frozen opportunity universe",
                code="crossfit_result_receiver_family_universe_mismatch",
                field=table_name,
                remediation="Reject the bundle and rerun its producer",
            )
        if "driver_id" not in table.columns:
            continue
        driver_rows = table.loc[
            table["driver_id"].notna(), ["driver_id", "family_id"]
        ].drop_duplicates()
        if any(
            family_by_driver.get(str(row.driver_id)) != str(row.family_id)
            for row in driver_rows.itertuples(index=False)
        ):
            raise ResultValidationError(
                "Persisted driver-family rows disagree with the frozen family axis",
                code="crossfit_result_receiver_family_universe_mismatch",
                field=f"{table_name}.driver_id",
                remediation="Reject the bundle and rerun its producer",
            )


def _validate_registry_links(
    manifest: dict[str, object],
    tables: dict[str, pd.DataFrame],
    *,
    tables_prevalidated: bool = False,
) -> None:
    applications = cast(list[dict[str, Any]], manifest["applications"])
    applications_by_id = {
        str(application["family_common_application_id"]): application
        for application in applications
    }
    is_oof_certified = bool(manifest["complete_pipeline_oof_certified"])
    claim_scope = _expected_claim_scope(is_oof_certified)
    expected_components = {
        str(row["family_common_application_id"]): int(row["persisted_component_rows"])
        for row in applications
    }
    expected_differential = {
        str(row["family_common_application_id"]): int(
            row["persisted_differential_rows"]
        )
        for row in applications
    }
    observed_components = _application_registry_counts(
        tables[CROSSFIT_COMPONENT_TABLE],
        id_column="family_common_application_id",
    )
    observed_differential = _application_registry_counts(
        tables[CROSSFIT_DIFFERENTIAL_TABLE],
        id_column="family_common_application_id",
    )
    unexpected_components = set(observed_components).difference(expected_components)
    normalized_components = {
        application_id: observed_components.get(application_id, 0)
        for application_id in expected_components
    }
    unexpected_differential = set(observed_differential).difference(
        expected_differential
    )
    normalized_differential = {
        application_id: observed_differential.get(application_id, 0)
        for application_id in expected_differential
    }
    if unexpected_components or normalized_components != expected_components:
        raise ResultValidationError(
            "Component rows do not match the application registry",
            code="crossfit_result_application_coverage_mismatch",
            field=CROSSFIT_COMPONENT_TABLE,
            remediation="Reject the bundle and rerun its producer",
        )
    if unexpected_differential or normalized_differential != expected_differential:
        raise ResultValidationError(
            "Differential rows do not match the application registry",
            code="crossfit_result_application_coverage_mismatch",
            field=CROSSFIT_DIFFERENTIAL_TABLE,
            remediation="Reject the bundle and rerun its producer",
        )
    _validate_application_table_lineage(
        CROSSFIT_COMPONENT_TABLE,
        tables[CROSSFIT_COMPONENT_TABLE],
        applications_by_id,
    )
    _validate_application_table_lineage(
        CROSSFIT_DIFFERENTIAL_TABLE,
        tables[CROSSFIT_DIFFERENTIAL_TABLE],
        applications_by_id,
    )
    _validate_source_scoring_registry(manifest, applications_by_id)
    if manifest["schema_version"] in {
        CROSSFIT_RESULT_SCHEMA_VERSION,
        _V8_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V7_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V6_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V5_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V4_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V3_CROSSFIT_RESULT_SCHEMA_VERSION,
    }:
        _validate_contrast_common_registry_links(
            manifest,
            tables,
            tables_prevalidated=tables_prevalidated,
        )
    if manifest["schema_version"] in {
        CROSSFIT_RESULT_SCHEMA_VERSION,
        _V8_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V7_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V6_CROSSFIT_RESULT_SCHEMA_VERSION,
        _V5_CROSSFIT_RESULT_SCHEMA_VERSION,
    }:
        _validate_directional_registry_links(
            manifest,
            tables[CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE],
        )
    if manifest["schema_version"] in _V7_PLUS_CROSSFIT_RESULT_SCHEMA_VERSIONS:
        _validate_receiver_training_support_links(
            manifest,
            tables[CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE],
        )
    if manifest["schema_version"] == CROSSFIT_RESULT_SCHEMA_VERSION:
        _validate_receiver_family_opportunity_links(manifest, tables)
    if manifest["schema_version"] in _V8_PLUS_CROSSFIT_RESULT_SCHEMA_VERSIONS:
        _validate_semantic_registry_links(
            manifest,
            tables,
            tables_prevalidated=tables_prevalidated,
        )
    for table_name, table in tables.items():
        if not table.empty and set(table["crossfit_id"].astype(str)) != {
            str(manifest["crossfit_id"])
        }:
            raise ResultValidationError(
                "Persisted rows do not match the manifest cross-fit identity",
                code="crossfit_result_identity_mismatch",
                field="crossfit_id",
                remediation="Reject the bundle and rerun its producer",
            )
        if table_name == CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE or table_name in (
            CROSSFIT_SEMANTIC_TABLE_NAMES
        ):
            continue
        if not table.empty and set(table["is_oof_certified"].tolist()) != {
            is_oof_certified
        }:
            raise ResultValidationError(
                "Persisted rows disagree with manifest OOF certification",
                code="crossfit_result_certification_mismatch",
                field="is_oof_certified",
                remediation="Reject the bundle and rerun its producer",
            )
        expected_table_claim_scope = (
            _expected_contrast_common_claim_scope(is_oof_certified)
            if table_name
            in {
                CROSSFIT_CONTRAST_COMMON_LR_TABLE,
                CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
            }
            else claim_scope
        )
        if not table.empty and set(table["claim_scope"].astype(str)) != {
            expected_table_claim_scope
        }:
            raise ResultValidationError(
                "Persisted rows disagree with manifest claim scope",
                code="crossfit_result_certification_mismatch",
                field="claim_scope",
                remediation="Reject the bundle and rerun its producer",
            )
        if not table.empty and set(table["certification_status"].astype(str)) != {
            str(manifest["certification_status"])
        }:
            raise ResultValidationError(
                "Persisted rows disagree with manifest certification status",
                code="crossfit_result_certification_mismatch",
                field="certification_status",
                remediation="Reject the bundle and rerun its producer",
            )


def _directional_opportunity_table(
    manifest: dict[str, object],
    receiver_support: pd.DataFrame,
    directional_registry: pd.DataFrame,
) -> pd.DataFrame:
    """Replay the complete directional channel grid from authenticated parents."""

    if manifest.get("schema_version") not in _V7_PLUS_CROSSFIT_RESULT_SCHEMA_VERSIONS:
        raise KeyError(
            "directional opportunities require a schema v7 or newer cross-fit result"
        )
    _validate_receiver_training_support_links(manifest, receiver_support)
    _validate_directional_registry_links(manifest, directional_registry)
    try:
        source = _directional_mapping(
            manifest.get("source_crossfit_manifest"),
            field_name="source_crossfit_manifest",
        )
        spec = _directional_mapping(source.get("spec"), field_name="source.spec")
        raw_pairs = spec.get("directional_pairs")
        if raw_pairs is None:
            return pd.DataFrame(columns=CROSSFIT_DIRECTIONAL_OPPORTUNITY_COLUMNS)
        if not isinstance(raw_pairs, list) or not raw_pairs:
            raise ValueError("source.spec.directional_pairs must be non-empty")
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        pair_ids: set[str] = set()
        for raw_pair in raw_pairs:
            pair_spec, contrasts = _directional_pair_spec_projection(raw_pair)
            pair_spec_id = str(pair_spec["pair_spec_id"])
            if pair_spec_id in pair_ids:
                raise ValueError("directional pair specifications are duplicated")
            pair_ids.add(pair_spec_id)
            pairs.append((pair_spec, contrasts))
        pairs.sort(key=lambda item: str(item[0]["pair_spec_id"]))

        support = _validate_receiver_training_support_table(receiver_support)
        support = support.sort_values(
            ["fold_id", "receiver"], kind="stable", ignore_index=True
        )
        registry = _validate_directional_channel_registry_table(directional_registry)
        registry_by_key: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
        for raw_record in registry.to_dict(orient="records"):
            record = cast(dict[str, object], raw_record)
            key = (
                str(record["pair_spec_id"]),
                str(record["fold_id"]),
                str(record["receiver"]),
            )
            role = str(record["channel_role"])
            role_records = registry_by_key.setdefault(key, {})
            if role in role_records:
                raise ValueError("directional binding channel role is duplicated")
            role_records[role] = record

        rows: list[dict[str, object]] = []
        consumed_registry_keys: set[tuple[str, str, str]] = set()
        for pair_spec, contrasts in pairs:
            pair_spec_id = str(pair_spec["pair_spec_id"])
            for support_record in support.to_dict(orient="records"):
                fold_id = str(support_record["fold_id"])
                receiver = str(support_record["receiver"])
                support_status = str(support_record["receiver_training_support_status"])
                support_reason = _optional_string(
                    support_record["receiver_training_support_reason_code"]
                )
                key = (pair_spec_id, fold_id, receiver)
                role_records = registry_by_key.get(key, {})
                if support_status == "observed":
                    if set(role_records) != {"forward", "reverse"}:
                        raise ValueError(
                            "supported directional opportunity lacks both channels"
                        )
                    consumed_registry_keys.add(key)
                elif (
                    support_status != "not_estimable"
                    or support_reason != _RECEIVER_ABSENT_REASON
                    or role_records
                ):
                    raise ValueError(
                        "training-absent directional opportunity has fitted lineage"
                    )

                pair_opportunity_id = str(
                    stable_id(
                        "directional_pair_opportunity",
                        {
                            "crossfit_id": support_record["crossfit_id"],
                            "fold_id": fold_id,
                            "pair_spec_id": pair_spec_id,
                            "receiver": receiver,
                            "receiver_axis_id": support_record["receiver_axis_id"],
                            "receiver_training_support_id": support_record[
                                "receiver_training_support_id"
                            ],
                        },
                        schema_version="1",
                    )
                )
                for role in ("forward", "reverse"):
                    binding_row = role_records.get(role)
                    channel = str(pair_spec[f"{role}_channel_name"])
                    contrast = cast(dict[str, Any], contrasts[role])
                    channel_opportunity_id = str(
                        stable_id(
                            "directional_channel_opportunity",
                            {
                                "channel_role": role,
                                "pair_opportunity_id": pair_opportunity_id,
                            },
                            schema_version="1",
                        )
                    )
                    if binding_row is None:
                        binding_id = None
                        response_id = None
                        training_artifact_id = None
                        application_id = None
                        response_pair_id = None
                        response_channel_id = None
                        status = "not_estimable"
                        reason_code = support_reason
                    else:
                        if (
                            str(binding_row["channel"]) != channel
                            or str(binding_row["contrast_id"])
                            != str(pair_spec[f"{role}_contrast_id"])
                            or str(binding_row["contrast"]) != str(contrast["name"])
                        ):
                            raise ValueError(
                                "directional binding changed channel attribution"
                            )
                        binding_id = str(binding_row["binding_id"])
                        response_id = str(binding_row["response_id"])
                        training_artifact_id = str(binding_row["training_artifact_id"])
                        application_id = str(binding_row["application_id"])
                        response_pair_id = _optional_string(
                            binding_row["response_pair_id"]
                        )
                        response_channel_id = _optional_string(
                            binding_row["response_channel_id"]
                        )
                        status = str(binding_row["status"])
                        reason_code = _optional_string(binding_row["reason_code"])
                    rows.append(
                        {
                            "crossfit_id": support_record["crossfit_id"],
                            "spec_id": support_record["spec_id"],
                            "repeat_id": support_record["repeat_id"],
                            "pair_opportunity_id": pair_opportunity_id,
                            "channel_opportunity_id": channel_opportunity_id,
                            "receiver_universe_id": support_record[
                                "receiver_universe_id"
                            ],
                            "receiver_axis_id": support_record["receiver_axis_id"],
                            "fold_id": fold_id,
                            "receiver": receiver,
                            "receiver_training_support_id": support_record[
                                "receiver_training_support_id"
                            ],
                            "receiver_training_support_status": support_status,
                            "receiver_training_support_reason_code": support_reason,
                            "pair_spec_id": pair_spec_id,
                            "binding_id": binding_id,
                            "channel_role": role,
                            "channel": channel,
                            "contrast_id": pair_spec[f"{role}_contrast_id"],
                            "contrast": contrast["name"],
                            "response_id": response_id,
                            "training_artifact_id": training_artifact_id,
                            "application_id": application_id,
                            "response_pair_id": response_pair_id,
                            "response_channel_id": response_channel_id,
                            "status": status,
                            "reason_code": reason_code,
                            "combination_rule": _DIRECTIONAL_COMBINATION_RULE,
                            "active_inhibition_allowed": False,
                            "supports_active_inhibition_claim": False,
                            "paired_score_comparison_allowed": False,
                            "formal_inference_allowed": False,
                            "certification_status": manifest["certification_status"],
                            "is_oof_certified": manifest[
                                "complete_pipeline_oof_certified"
                            ],
                            "formal_inference_status": manifest[
                                "formal_inference_status"
                            ],
                            "claim_scope": manifest["claim_scope"],
                        }
                    )
        if consumed_registry_keys != set(registry_by_key):
            raise ValueError("directional registry contains an unplanned opportunity")
        result = pd.DataFrame(
            rows,
            columns=CROSSFIT_DIRECTIONAL_OPPORTUNITY_COLUMNS,
        ).sort_values(
            ["pair_spec_id", "fold_id", "receiver", "channel_role"],
            kind="stable",
            ignore_index=True,
        )
        expected_count = len(pairs) * len(support) * 2
        if (
            len(result) != expected_count
            or result["channel_opportunity_id"].astype(str).duplicated().any()
            or result.duplicated(
                ["pair_spec_id", "fold_id", "receiver", "channel_role"]
            ).any()
            or not set(result["status"].astype(str)).issubset(
                _DIRECTIONAL_CHANNEL_STATUSES
            )
        ):
            raise ValueError("directional opportunity grid is incomplete")
        return result
    except (KeyError, TypeError, ValueError) as error:
        raise ResultValidationError(
            "Directional opportunity replay violates its frozen receiver grid",
            code="invalid_crossfit_directional_opportunity_registry",
            field="directional_opportunities",
            remediation="Reject the bundle and rerun its producer",
        ) from error


@dataclass(frozen=True, slots=True, init=False)
class CrossFitResult:
    """Validated receiver-local and receiver-balanced descriptive OOF rows."""

    path: Path
    _manifest: dict[str, object] = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "CrossFitResult is producer-owned; use write_crossfit_result() or load()"
        )

    @classmethod
    def _from_validated(
        cls,
        path: Path,
        manifest: dict[str, object],
    ) -> CrossFitResult:
        self = object.__new__(cls)
        object.__setattr__(self, "path", path.resolve())
        object.__setattr__(self, "_manifest", copy.deepcopy(manifest))
        return self

    @classmethod
    def load(cls, path: str | Path) -> CrossFitResult:
        """Load a complete bundle after hashes, schemas, and semantics validate."""

        root = Path(path)
        try:
            marker = _read_json(root / _STATUS_FILENAME)
        except Exception as error:
            raise ResultValidationError(
                "Cross-fit result status marker is missing or corrupted",
                code="invalid_crossfit_result_status",
                field="path",
                remediation="Reject the bundle and regenerate it",
            ) from error
        if marker.get("status") != _COMPLETE_STATUS:
            raise IncompleteResultError(
                "Cross-fit result is incomplete",
                code="incomplete_crossfit_result",
                field="path",
                remediation="Inspect producer diagnostics and rerun to a new directory",
            )
        if (
            set(marker)
            != {
                "schema_version",
                "status",
                "crossfit_result_id",
            }
            or marker.get("schema_version")
            not in _SUPPORTED_CROSSFIT_RESULT_SCHEMA_VERSIONS
        ):
            raise ResultValidationError(
                "Cross-fit result status marker violates its schema",
                code="invalid_crossfit_result_status",
                field="path",
                remediation="Reject the bundle and regenerate it",
            )
        try:
            manifest = _validate_manifest(_read_json(root / _MANIFEST_FILENAME))
        except Exception as error:
            raise ResultValidationError(
                "Cross-fit result manifest is missing, corrupted, or incompatible",
                code="invalid_crossfit_result_manifest",
                field="path",
                remediation="Reject the bundle and regenerate it",
            ) from error
        if (
            marker["schema_version"] != manifest["schema_version"]
            or marker["crossfit_result_id"] != manifest["crossfit_result_id"]
        ):
            raise ResultValidationError(
                "Cross-fit result status marker disagrees with its manifest",
                code="crossfit_result_status_manifest_mismatch",
                field="path",
                remediation="Reject the bundle and regenerate it",
            )
        raw_records = cast(dict[str, dict[str, Any]], manifest["tables"])
        schema_version = str(manifest["schema_version"])
        table_names = _bundle_table_names(schema_version)
        tables = {
            name: _load_table(
                root,
                name,
                raw_records[name],
                schema_version=schema_version,
            )
            for name in table_names
        }
        _validate_registry_links(manifest, tables)
        return cls._from_validated(root, manifest)

    @property
    def manifest(self) -> dict[str, object]:
        """Return a defensive copy of the validated manifest."""

        return copy.deepcopy(self._manifest)

    def read_table(self, name: str) -> pd.DataFrame:
        """Read one released table and revalidate its current bytes."""

        records = cast(dict[str, dict[str, Any]], self._manifest["tables"])
        if name not in records:
            raise KeyError(name)
        return _load_table(
            self.path,
            name,
            records[name],
            schema_version=str(self._manifest["schema_version"]),
        )

    def read_components(self) -> pd.DataFrame:
        """Return the grain-aware family/LR/sender component ledger."""

        return self.read_table(CROSSFIT_COMPONENT_TABLE)

    def read_descriptive_differential(self) -> pd.DataFrame:
        """Return descriptive subject-family effects without inferential fields."""

        return self.read_table(CROSSFIT_DIFFERENTIAL_TABLE)

    def read_contrast_common_lr_scores(self) -> pd.DataFrame:
        """Return held-out LR rows from a receiver-balanced descriptive set."""

        return self.read_table(CROSSFIT_CONTRAST_COMMON_LR_TABLE)

    def read_contrast_common_sender_lr_scores(self) -> pd.DataFrame:
        """Return held-out sender-LR rows from the descriptive collection."""

        return self.read_table(CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE)

    def read_directional_channel_registry(self) -> pd.DataFrame:
        """Return explicit forward/reverse lineage without combining channels."""

        return self.read_table(CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE)

    def read_receiver_training_support(self) -> pd.DataFrame:
        """Return the exact fold-by-receiver outer-training support grid."""

        return self.read_table(CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE)

    def read_directional_opportunities(self) -> pd.DataFrame:
        """Return the complete pair-by-fold-by-receiver-by-channel grid.

        Unlike the binding registry, this derived view retains receivers absent
        from an outer-training fold as typed not-estimable opportunities.
        """

        return _directional_opportunity_table(
            self._manifest,
            self.read_receiver_training_support(),
            self.read_directional_channel_registry(),
        )

    @property
    def semantic_score_manifest(self) -> dict[str, object]:
        """Return the v8 semantic collection manifest as a defensive copy."""

        raw = self._manifest.get("semantic_score_collection")
        if not isinstance(raw, dict):
            raise KeyError("semantic score views require a schema v8 result")
        return copy.deepcopy(cast(dict[str, object], raw))

    def read_semantic_score(self, semantic_output: str) -> pd.DataFrame:
        """Read one exact persisted semantic view without reconstructing scores."""

        if semantic_output not in _SEMANTIC_OUTPUT_TO_TABLE:
            raise ValueError(
                "semantic_output must be availability_score, receiver_program_score, "
                "integrated_lr_score, or differential_effect"
            )
        table_name = _SEMANTIC_OUTPUT_TO_TABLE[semantic_output]
        if table_name not in cast(dict[str, object], self._manifest["tables"]):
            raise KeyError("semantic score views require a schema v8 result")
        return self.read_table(table_name)

    def read_semantic_availability(self) -> pd.DataFrame:
        """Return held-out sender/receiver/LR availability by semantic mode."""

        return self.read_semantic_score("availability_score")

    def read_state_semantic_availability(self) -> pd.DataFrame:
        """Read the held-out state availability view with column projection.

        Multigroup differential post-processing only needs the sender-specific
        state score and its row identity.  Reading this projection avoids
        materializing the ecosystem rows and the wider provenance payload.
        The persisted bytes are still authenticated against the result
        manifest before the projection is returned.
        """

        table_name = CROSSFIT_SEMANTIC_AVAILABILITY_TABLE
        records = cast(dict[str, dict[str, Any]], self._manifest["tables"])
        if table_name not in records:
            raise KeyError("semantic score views require a schema v8 result")
        record = records[table_name]
        path = self.path / _TABLE_FILENAMES[table_name]
        try:
            digest = _sha256_file(path)
        except Exception as error:
            raise ResultValidationError(
                "State semantic availability is missing or corrupted",
                code="corrupted_crossfit_result_table",
                field=table_name,
                remediation="Reject the bundle and regenerate it from intact artifacts",
            ) from error
        if digest != record["sha256"]:
            raise ResultValidationError(
                "State semantic availability does not match its manifest",
                code="crossfit_result_digest_mismatch",
                field=table_name,
                remediation="Reject the bundle and regenerate it from intact artifacts",
            )

        columns = (
            "crossfit_id",
            "fold_id",
            "sample_id",
            "subject_id",
            "sender",
            "receiver",
            "interaction_id",
            "mode",
            "availability_score",
            "status",
            "reason_code",
        )
        if not set(columns).issubset(record["columns"]):
            raise ResultValidationError(
                "State semantic availability manifest lacks required columns",
                code="crossfit_result_digest_mismatch",
                field=table_name,
                remediation="Reject the bundle and regenerate it from intact artifacts",
            )
        try:
            frame = pd.read_parquet(
                path,
                columns=list(columns),
                filters=[("mode", "==", "state")],
            )
        except Exception as error:
            raise ResultValidationError(
                "State semantic availability is corrupted",
                code="corrupted_crossfit_result_table",
                field=table_name,
                remediation="Reject the bundle and regenerate it from intact artifacts",
            ) from error
        if frame.empty or tuple(frame.columns) != columns:
            raise ResultValidationError(
                "State semantic availability projection is empty or malformed",
                code="invalid_crossfit_result_table",
                field=table_name,
                remediation="Reject the bundle and rerun its producer",
            )
        if not frame["mode"].astype(str).eq("state").all():
            raise ResultValidationError(
                "State semantic availability contains another communication mode",
                code="invalid_crossfit_result_table",
                field=table_name,
                remediation="Reject the bundle and rerun its producer",
            )
        if set(frame["crossfit_id"].astype(str)) != {
            str(self._manifest["crossfit_id"])
        }:
            raise ResultValidationError(
                "State semantic availability has invalid cross-fit identity",
                code="crossfit_result_identity_mismatch",
                field=table_name,
                remediation="Reject the bundle and rerun its producer",
            )
        keys = ("fold_id", "sample_id", "sender", "receiver", "interaction_id")
        if frame.duplicated(list(keys)).any():
            raise ResultValidationError(
                "State semantic availability keys are not unique",
                code="invalid_crossfit_result_table",
                field=table_name,
                remediation="Reject the bundle and rerun its producer",
            )
        status = frame["status"].astype(str)
        if not set(status).issubset({"observed", "not_estimable"}):
            raise ResultValidationError(
                "State semantic availability has unsupported statuses",
                code="invalid_crossfit_result_table",
                field=table_name,
                remediation="Reject the bundle and rerun its producer",
            )
        score = pd.to_numeric(frame["availability_score"], errors="coerce")
        observed = status.eq("observed")
        invalid_observed = (
            score.loc[observed].isna()
            | ~score.loc[observed].between(0.0, 1.0)
            | np.isinf(score.loc[observed])
        )
        if invalid_observed.any() or score.loc[~observed].notna().any():
            raise ResultValidationError(
                "State semantic availability score/status semantics are invalid",
                code="invalid_crossfit_result_table",
                field=table_name,
                remediation="Reject the bundle and rerun its producer",
            )
        frame["availability_score"] = score
        return frame.reset_index(drop=True)

    def read_semantic_receiver_programs(self) -> pd.DataFrame:
        """Return source-agnostic held-out receiver-program scores."""

        return self.read_semantic_score("receiver_program_score")

    def read_semantic_integrated_lr_scores(self) -> pd.DataFrame:
        """Return generic family-allocated LR scores with directionals excluded."""

        return self.read_semantic_score("integrated_lr_score")

    def read_semantic_differential_effects(self) -> pd.DataFrame:
        """Return generic held-out subject-family differential effects."""

        return self.read_semantic_score("differential_effect")

    def query_receiver_training_support(
        self,
        *,
        fold_id: str | None = None,
        receiver: str | None = None,
        status: str | None = None,
    ) -> pd.DataFrame:
        """Query frozen receiver opportunities without implying score production."""

        filters = {
            "fold_id": fold_id,
            "receiver": receiver,
            "receiver_training_support_status": status,
        }
        for field_name, value in filters.items():
            if value is not None and (
                not isinstance(value, str) or not value or value != value.strip()
            ):
                raise ValueError(
                    f"{field_name} must be a canonical non-empty string or None"
                )
        if status is not None and status not in {"observed", "not_estimable"}:
            raise ValueError("status must be 'observed', 'not_estimable', or None")
        frame = self.read_receiver_training_support()
        selected = pd.Series(True, index=frame.index, dtype=bool)
        for column, value in filters.items():
            if value is not None:
                selected &= frame[column].astype(str).eq(value)
        return (
            frame.loc[selected]
            .sort_values(["fold_id", "receiver"], kind="stable")
            .reset_index(drop=True)
            .copy(deep=True)
        )

    def query_directional_opportunities(
        self,
        *,
        pair_opportunity_id: str | None = None,
        channel_opportunity_id: str | None = None,
        pair_spec_id: str | None = None,
        binding_id: str | None = None,
        fold_id: str | None = None,
        receiver: str | None = None,
        channel: str | None = None,
        status: str | None = None,
        receiver_training_support_status: str | None = None,
    ) -> pd.DataFrame:
        """Query the full directional opportunity grid without dropping NE rows."""

        filters = {
            "pair_opportunity_id": pair_opportunity_id,
            "channel_opportunity_id": channel_opportunity_id,
            "pair_spec_id": pair_spec_id,
            "binding_id": binding_id,
            "fold_id": fold_id,
            "receiver": receiver,
            "channel": channel,
            "status": status,
            "receiver_training_support_status": (receiver_training_support_status),
        }
        for field_name, value in filters.items():
            if value is not None and (
                not isinstance(value, str) or not value or value != value.strip()
            ):
                raise ValueError(
                    f"{field_name} must be a canonical non-empty string or None"
                )
        if channel is not None and channel not in set(_DIRECTIONAL_CHANNELS.values()):
            raise ValueError(
                "channel must be 'increased_activation_compatible', "
                "'reduced_activation_compatible', or None"
            )
        if status is not None and status not in _DIRECTIONAL_CHANNEL_STATUSES:
            raise ValueError("status must be 'observed', 'not_estimable', or None")
        if receiver_training_support_status is not None and (
            receiver_training_support_status not in {"observed", "not_estimable"}
        ):
            raise ValueError(
                "receiver_training_support_status must be 'observed', "
                "'not_estimable', or None"
            )
        frame = self.read_directional_opportunities()
        selected = pd.Series(True, index=frame.index, dtype=bool)
        for column, value in filters.items():
            if value is not None:
                selected &= frame[column].astype(str).eq(value)
        return (
            frame.loc[selected]
            .sort_values(
                ["pair_spec_id", "fold_id", "receiver", "channel_role"],
                kind="stable",
            )
            .reset_index(drop=True)
            .copy(deep=True)
        )

    def query_directional_channels(
        self,
        *,
        pair_spec_id: str | None = None,
        binding_id: str | None = None,
        fold_id: str | None = None,
        receiver: str | None = None,
        channel: str | None = None,
        status: str | None = None,
    ) -> pd.DataFrame:
        """Query fitted directional bindings, excluding training-absent scopes.

        Use :meth:`query_directional_opportunities` when the complete frozen
        receiver grid, including typed not-estimable opportunities, is needed.
        """

        filters = {
            "pair_spec_id": pair_spec_id,
            "binding_id": binding_id,
            "fold_id": fold_id,
            "receiver": receiver,
            "channel": channel,
            "status": status,
        }
        for field_name, value in filters.items():
            if value is not None and (
                not isinstance(value, str) or not value or value != value.strip()
            ):
                raise ValueError(
                    f"{field_name} must be a canonical non-empty string or None"
                )
        if channel is not None and channel not in set(_DIRECTIONAL_CHANNELS.values()):
            raise ValueError(
                "channel must be 'increased_activation_compatible', "
                "'reduced_activation_compatible', or None"
            )
        if status is not None and status not in _DIRECTIONAL_CHANNEL_STATUSES:
            raise ValueError("status must be 'observed', 'not_estimable', or None")
        frame = self.read_directional_channel_registry()
        selected = pd.Series(True, index=frame.index, dtype=bool)
        for column, value in filters.items():
            if value is not None:
                selected &= frame[column].astype(str).eq(value)
        return (
            frame.loc[selected]
            .sort_values(
                ["pair_spec_id", "fold_id", "receiver", "channel_role"],
                kind="stable",
            )
            .reset_index(drop=True)
            .copy(deep=True)
        )

    query_directional_channel_registry = query_directional_channels

    def _query_contrast_common_scores(
        self,
        *,
        table_name: str,
        contrast_common_collection_id: str | None,
        contrast: str | None,
        fold_id: str | None,
        sample_id: str | None,
        subject_id: str | None,
        context_id: str | None,
        sender: str | None,
        receiver: str | None,
        family_id: str | None,
        interaction_id: str | None,
        mode: str | None,
        status: str | None,
    ) -> pd.DataFrame:
        if table_name == CROSSFIT_CONTRAST_COMMON_LR_TABLE and sender is not None:
            raise ValueError("sender is unavailable at contrast-common LR grain")
        filters = {
            "contrast_common_collection_id": contrast_common_collection_id,
            "contrast": contrast,
            "fold_id": fold_id,
            "sample_id": sample_id,
            "subject_id": subject_id,
            "context_id": context_id,
            "sender": sender,
            "receiver": receiver,
            "family_id": family_id,
            "interaction_id": interaction_id,
            "mode": mode,
            "status": status,
        }
        for field_name, value in filters.items():
            if value is not None and (
                not isinstance(value, str) or not value or value != value.strip()
            ):
                raise ValueError(
                    f"{field_name} must be a canonical non-empty string or None"
                )
        if mode is not None and mode not in {"state", "ecosystem"}:
            raise ValueError("mode must be 'state', 'ecosystem', or None")
        if status is not None and status not in _CONTRAST_COMMON_STATUSES:
            raise ValueError(
                "status must be 'observed', 'structural_zero', 'not_estimable', or None"
            )
        frame = self.read_table(table_name)
        selected = pd.Series(True, index=frame.index, dtype=bool)
        for column, value in filters.items():
            if value is not None:
                selected &= frame[column].astype(str).eq(value)
        sort_columns = [
            "contrast",
            "fold_id",
            "context_id",
            "receiver",
            "sample_id",
            "family_id",
            "interaction_id",
        ]
        if table_name == CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE:
            sort_columns.insert(3, "sender")
        return (
            frame.loc[selected]
            .sort_values(sort_columns, kind="stable")
            .reset_index(drop=True)
            .copy(deep=True)
        )

    def query_contrast_common_lr_scores(
        self,
        *,
        contrast_common_collection_id: str | None = None,
        contrast: str | None = None,
        fold_id: str | None = None,
        sample_id: str | None = None,
        subject_id: str | None = None,
        context_id: str | None = None,
        receiver: str | None = None,
        family_id: str | None = None,
        interaction_id: str | None = None,
        mode: str | None = None,
        status: str | None = None,
    ) -> pd.DataFrame:
        """Query one receiver-balanced descriptive LR collection."""

        return self._query_contrast_common_scores(
            table_name=CROSSFIT_CONTRAST_COMMON_LR_TABLE,
            contrast_common_collection_id=contrast_common_collection_id,
            contrast=contrast,
            fold_id=fold_id,
            sample_id=sample_id,
            subject_id=subject_id,
            context_id=context_id,
            sender=None,
            receiver=receiver,
            family_id=family_id,
            interaction_id=interaction_id,
            mode=mode,
            status=status,
        )

    def query_contrast_common_sender_lr_scores(
        self,
        *,
        contrast_common_collection_id: str | None = None,
        contrast: str | None = None,
        fold_id: str | None = None,
        sample_id: str | None = None,
        subject_id: str | None = None,
        context_id: str | None = None,
        sender: str | None = None,
        receiver: str | None = None,
        family_id: str | None = None,
        interaction_id: str | None = None,
        mode: str | None = None,
        status: str | None = None,
    ) -> pd.DataFrame:
        """Query one receiver-balanced descriptive sender-LR collection."""

        return self._query_contrast_common_scores(
            table_name=CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
            contrast_common_collection_id=contrast_common_collection_id,
            contrast=contrast,
            fold_id=fold_id,
            sample_id=sample_id,
            subject_id=subject_id,
            context_id=context_id,
            sender=sender,
            receiver=receiver,
            family_id=family_id,
            interaction_id=interaction_id,
            mode=mode,
            status=status,
        )

    query_contrast_common_lr_pairs = query_contrast_common_lr_scores
    query_contrast_common_sender_lr_pairs = query_contrast_common_sender_lr_scores

    def _query_component_scope(
        self,
        *,
        component_scope: str,
        component: str,
        contrast: str | None,
        context_id: str | None,
        receiver: str | None,
        family_id: str | None,
        interaction_id: str | None,
        sender: str | None,
        mode: str | None,
        status: str | None,
    ) -> pd.DataFrame:
        if component_scope not in _SCOPE_COMPONENTS:
            raise ValueError("component_scope is unsupported")
        if component not in _SCOPE_COMPONENTS[component_scope]:
            raise ValueError(
                f"component {component!r} is unavailable at {component_scope!r} grain"
            )
        filters = {
            "contrast": contrast,
            "context_id": context_id,
            "receiver": receiver,
            "family_id": family_id,
            "interaction_id": interaction_id,
            "sender": sender,
            "mode": mode,
            "status": status,
        }
        for field_name, value in filters.items():
            if value is not None and (
                not isinstance(value, str) or not value or value != value.strip()
            ):
                raise ValueError(
                    f"{field_name} must be a canonical non-empty string or None"
                )
        if mode is not None and mode not in {"state", "ecosystem"}:
            raise ValueError("mode must be 'state', 'ecosystem', or None")
        if status is not None and status not in _COMPONENT_STATUSES:
            raise ValueError(
                "status must be 'observed', 'structural_zero', 'not_estimable', or None"
            )
        frame = self.read_components()
        selected = frame["component_scope"].eq(component_scope) & frame["component"].eq(
            component
        )
        for column, value in filters.items():
            if value is not None:
                selected &= frame[column].astype(str).eq(value)
        sort_columns = [
            "contrast",
            "context_id",
            "receiver",
            "sample_id",
            "family_id",
            "interaction_id",
            "sender",
        ]
        return (
            frame.loc[selected]
            .sort_values(
                sort_columns,
                kind="stable",
                na_position="first",
            )
            .reset_index(drop=True)
            .copy(deep=True)
        )

    def query_family_scores(
        self,
        *,
        contrast: str | None = None,
        context_id: str | None = None,
        receiver: str | None = None,
        family_id: str | None = None,
        mode: str | None = None,
        status: str | None = None,
        component: str = "integrated_lr_score",
    ) -> pd.DataFrame:
        """Query receiver-family scores without merging receiver functionals."""

        return self._query_component_scope(
            component_scope="family",
            component=component,
            contrast=contrast,
            context_id=context_id,
            receiver=receiver,
            family_id=family_id,
            interaction_id=None,
            sender=None,
            mode=mode,
            status=status,
        )

    def query_lr_pairs(
        self,
        *,
        contrast: str | None = None,
        context_id: str | None = None,
        receiver: str | None = None,
        family_id: str | None = None,
        interaction_id: str | None = None,
        mode: str | None = None,
        status: str | None = None,
        component: str = "sender_unresolved_strength",
    ) -> pd.DataFrame:
        """Query condition-specific LR members before sender allocation."""

        return self._query_component_scope(
            component_scope="lr_member",
            component=component,
            contrast=contrast,
            context_id=context_id,
            receiver=receiver,
            family_id=family_id,
            interaction_id=interaction_id,
            sender=None,
            mode=mode,
            status=status,
        )

    def query_integrated_lr_scores(
        self,
        *,
        contrast: str | None = None,
        context_id: str | None = None,
        receiver: str | None = None,
        family_id: str | None = None,
        interaction_id: str | None = None,
        mode: str | None = None,
        status: str | None = None,
    ) -> pd.DataFrame:
        """Query the canonical generic integrated-LR score projection.

        This fixes the grain and source component to the family-allocated LR
        member score. Registered forward/reverse contrasts are excluded because
        their two channels require the dedicated directional result contract.
        """

        table_records = self._manifest.get("tables")
        has_exact_semantic = (
            isinstance(table_records, dict)
            and CROSSFIT_SEMANTIC_INTEGRATED_LR_TABLE in table_records
        )
        if has_exact_semantic:
            filters = {
                "contrast": contrast,
                "context_id": context_id,
                "receiver": receiver,
                "family_id": family_id,
                "interaction_id": interaction_id,
                "mode": mode,
                "status": status,
            }
            for field_name, value in filters.items():
                if value is not None and (
                    not isinstance(value, str) or not value or value != value.strip()
                ):
                    raise ValueError(
                        f"{field_name} must be a canonical non-empty string or None"
                    )
            if mode is not None and mode not in {"state", "ecosystem"}:
                raise ValueError("mode must be 'state', 'ecosystem', or None")
            if status is not None and status not in _COMPONENT_STATUSES:
                raise ValueError(
                    "status must be 'observed', 'structural_zero', "
                    "'not_estimable', or None"
                )
            frame = self.read_semantic_integrated_lr_scores()
            selected = pd.Series(True, index=frame.index, dtype=bool)
            for column, value in filters.items():
                if value is not None:
                    selected &= frame[column].astype(str).eq(value)
            semantic = frame.loc[selected].copy(deep=True)
        else:
            frame = self._query_component_scope(
                component_scope="lr_member",
                component="sender_unresolved_strength",
                contrast=contrast,
                context_id=context_id,
                receiver=receiver,
                family_id=family_id,
                interaction_id=interaction_id,
                sender=None,
                mode=mode,
                status=status,
            )
            if (
                isinstance(table_records, dict)
                and CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE in table_records
            ):
                directional = self.read_directional_channel_registry()
                directional_ids = set(directional["contrast_id"].astype(str))
                directional_names = set(directional["contrast"].astype(str))
                frame = frame.loc[
                    ~frame["contrast_id"].astype(str).isin(directional_ids)
                    & ~frame["contrast"].astype(str).isin(directional_names)
                ].copy(deep=True)
            direct_columns = (
                "crossfit_id",
                "repeat_id",
                "fold_id",
                "contrast_id",
                "contrast",
                "sample_id",
                "subject_id",
                "context_id",
                "receiver",
                "family_id",
                "driver_id",
                "interaction_id",
                "mode",
                "status",
                "reason_code",
                "family_common_functional_id",
                "family_common_application_id",
                "family_common_binding_id",
                "score_version",
            )
            semantic = frame.loc[:, list(direct_columns)].copy(deep=True)
            semantic["crossfit_spec_id"] = frame["spec_id"]
            semantic["integrated_lr_score"] = frame["component_value"]
            semantic["source_component"] = "sender_unresolved_strength"
            semantic["formal_inference_allowed"] = False
        semantic["spec_id"] = semantic["crossfit_spec_id"]
        semantic["semantic_output"] = "integrated_lr_score"
        semantic["directional_contrasts_excluded"] = True
        return (
            semantic.loc[:, list(CROSSFIT_INTEGRATED_LR_QUERY_COLUMNS)]
            .reset_index(drop=True)
            .copy(deep=True)
        )

    def query_sender_lr_pairs(
        self,
        *,
        contrast: str | None = None,
        context_id: str | None = None,
        sender: str | None = None,
        receiver: str | None = None,
        family_id: str | None = None,
        interaction_id: str | None = None,
        mode: str | None = None,
        status: str | None = None,
        component: str = "sender_resolved_strength",
    ) -> pd.DataFrame:
        """Query condition-specific sender-LR-receiver scores."""

        return self._query_component_scope(
            component_scope="sender_lr_member",
            component=component,
            contrast=contrast,
            context_id=context_id,
            receiver=receiver,
            family_id=family_id,
            interaction_id=interaction_id,
            sender=sender,
            mode=mode,
            status=status,
        )


@validation_scope()
def write_crossfit_result(
    artifacts: CrossFitArtifacts,
    destination: str | Path,
) -> CrossFitResult:
    """Persist verified receiver-local and receiver-balanced descriptive outputs."""

    if type(artifacts) is not CrossFitArtifacts:
        raise TypeError("artifacts must be producer-owned CrossFitArtifacts")
    artifacts._require_intact()
    output = Path(destination)
    if output.exists():
        raise ResultWriteError(
            f"Cross-fit result destination {output.name!r} already exists",
            code="crossfit_result_destination_exists",
            field="destination",
            remediation="Choose a new versioned result directory",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    _mark_incomplete(temporary)
    try:
        tables, applications, common_collections, projection = _result_tables(artifacts)
        semantic_scores = build_crossfit_semantic_scores(artifacts)
        semantic_tables, semantic_manifest_payload = semantic_scores.snapshot()
        tables.update(
            {
                table_name: semantic_tables[semantic_output]
                for semantic_output, table_name in _SEMANTIC_OUTPUT_TO_TABLE.items()
            }
        )
        table_records: dict[str, dict[str, object]] = {}
        for name in CROSSFIT_TABLE_NAMES:
            table = _validate_table(name, tables[name])
            tables[name] = table
            path = temporary / _TABLE_FILENAMES[name]
            table.to_parquet(
                path,
                index=False,
                engine="pyarrow",
                compression="zstd",
                row_group_size=131_072,
            )
            table_records[name] = {
                "filename": _TABLE_FILENAMES[name],
                "rows": len(table),
                "sha256": _sha256_file(path),
                "schema_version": _TABLE_SCHEMA_VERSIONS[name],
                "columns": list(_TABLE_COLUMNS[name]),
            }
        source_manifest = artifacts.to_manifest()
        if source_manifest.get("oof_certification_audit_id") != projection.audit_id:
            raise ValueError("cross-fit OOF certification changed during persistence")
        raw_family_universe = source_manifest.get(
            "receiver_family_opportunity_universe"
        )
        if not isinstance(raw_family_universe, dict):
            raise ValueError(
                "cross-fit source manifest lacks its mandatory receiver-family "
                "opportunity universe"
            )
        semantic_manifest = cast(
            dict[str, object],
            json.loads(canonical_json(semantic_manifest_payload)),
        )
        manifest: dict[str, object] = {
            "schema_version": CROSSFIT_RESULT_SCHEMA_VERSION,
            "status": _COMPLETE_STATUS,
            "crossfit_id": artifacts.crossfit_id,
            "spec_id": artifacts.spec.spec_id,
            "repeat_id": artifacts.spec.repeat_id,
            "receiver_universe_id": artifacts.receiver_universe.universe_id,
            "receiver_axis_id": artifacts.receiver_universe.receiver_axis_id,
            "receiver_family_opportunity_universe_id": raw_family_universe.get(
                "universe_id"
            ),
            "family_axis_id": raw_family_universe.get("family_axis_id"),
            "receiver_family_opportunity_axis_id": raw_family_universe.get(
                "opportunity_axis_id"
            ),
            "receiver_family_opportunity_universe": copy.deepcopy(raw_family_universe),
            "root_input_digest": artifacts.root_input_digest,
            "certification_status": projection.certification_status,
            "complete_pipeline_oof_certified": projection.is_oof_certified,
            "formal_inference_status": _FORMAL_INFERENCE_STATUS,
            "claim_scope": projection.claim_scope,
            "source_crossfit_manifest_digest": canonical_digest(source_manifest),
            "source_crossfit_manifest": source_manifest,
            "family_common_stage_connected": bool(applications),
            "applications": applications,
            "contrast_common_stage_connected": bool(common_collections),
            "contrast_common_collections": common_collections,
            "semantic_score_collection": semantic_manifest,
            "tables": table_records,
        }
        manifest["crossfit_result_id"] = stable_id(
            "crossfit_result",
            manifest,
            schema_version="1",
        )
        validated_manifest = _validate_manifest(cast(dict[str, Any], manifest))
        _validate_registry_links(
            validated_manifest,
            tables,
            tables_prevalidated=True,
        )
        _write_json(temporary / _MANIFEST_FILENAME, validated_manifest)
        _write_json(
            temporary / _STATUS_FILENAME,
            {
                "schema_version": CROSSFIT_RESULT_SCHEMA_VERSION,
                "status": _COMPLETE_STATUS,
                "crossfit_result_id": validated_manifest["crossfit_result_id"],
            },
        )
        os.replace(temporary, output)
        return CrossFitResult._from_validated(output, validated_manifest)
    except Exception as error:
        if temporary.exists():
            _mark_incomplete(temporary, type(error).__name__)
            if not output.exists():
                os.replace(temporary, output)
        raise ResultWriteError(
            "Cross-fit result write for "
            f"{output.name!r} failed and was marked incomplete",
            code="crossfit_result_write_failed",
            field="destination",
            remediation="Inspect producer diagnostics and write to a new directory",
        ) from error


__all__ = [
    "CROSSFIT_COMPONENT_COLUMNS",
    "CROSSFIT_COMPONENT_TABLE",
    "CROSSFIT_CONTRAST_COMMON_LR_COLUMNS",
    "CROSSFIT_CONTRAST_COMMON_LR_TABLE",
    "CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS",
    "CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE",
    "CROSSFIT_DIFFERENTIAL_COLUMNS",
    "CROSSFIT_DIFFERENTIAL_TABLE",
    "CROSSFIT_DIRECTIONAL_CHANNEL_COLUMNS",
    "CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE",
    "CROSSFIT_DIRECTIONAL_OPPORTUNITY_COLUMNS",
    "CROSSFIT_RECEIVER_TABLE_NAMES",
    "CROSSFIT_RECEIVER_TRAINING_SUPPORT_COLUMNS",
    "CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE",
    "CROSSFIT_RESULT_SCHEMA_VERSION",
    "CROSSFIT_TABLE_NAMES",
    "CrossFitResult",
    "write_crossfit_result",
]
