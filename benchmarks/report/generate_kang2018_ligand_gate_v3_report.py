"""Generate the deterministic Kang 2018 ligand-gate v3 diagnostic report.

The presentation layer compares only fold-level ligand contrast support and
receiver-specific receptor eligibility. Receiver-duplicated ligand rows are
validated but never counted as independent evidence. Downstream family,
member, and sender scores are outside the comparison contract.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

REPORT_SCHEMA_VERSION = "crychic-kang2018-ligand-gate-v3-report-v1"
ARTIFACT_SCHEMA_VERSION = "crychic-kang2018-ligand-gate-v3-benchmark-v1"
ARTIFACT_SCOPE = "kang2018_real_data_small_ligand_gate_v3_development_diagnostic"
DEFAULT_GENERATED_AT = "2026-07-15T00:00:00+00:00"
SVG_HASHSALT = "crychic-kang2018-ligand-gate-v3-report-v1"

EXPECTED_CONFIG_SHA256 = (
    "31f0001e8c5db4b2b68c17bc0953a310c684f9f596be500db72bb669fbbb48c5"
)
EXPECTED_INPUT_SHA256 = (
    "5ee14c9df0a58e385828f9339dc93eff52fae7614218e82ed9c9a022804b30ac"
)
EXPECTED_PARTITION_MANIFEST_ID = (
    "7634429d1fa9f6a617cf695b4e4654e71064ed6576da41596fb3a38bab41626f"
)
PARTITION_IDS = (
    "4b00e98c92ca10b29a08a2f2b34877533f8dcaee214bbd455e2f7cc51eb2cfe4",
    "fa2ce0063d1dbde58046287595f7b116ec74357f755a01223701fbeafc38d266",
)
PARTITION_LABELS = dict(zip(PARTITION_IDS, ("Partition A", "Partition B"), strict=True))

CLAIMS: dict[str, bool] = {
    "biological_validation": False,
    "certifying": False,
    "default_switch_allowed": False,
    "exploratory": True,
    "literature_informed": True,
    "method_superiority": False,
}
SENSITIVITY_POLICY: dict[str, object] = {
    "aligned_by": "subject_partition_manifest_id",
    "contrast_support_comparison_allowed": True,
    "family_member_sender_score_comparison_allowed": False,
    "scope": "contrast_support_only",
}
SUBJECT_PRIVACY_SEMANTICS = (
    "deterministic_nonplaintext_identifier_for_alignment_not_anonymization"
)

DIAGNOSTICS = (
    "CXCL10_CXCR3",
    "IFNB1_IFNAR1_IFNAR2",
    "CCL5_CCR5",
)
RECEIVERS = ("CD14+ Monocytes", "CD8 T cells")
DIAGNOSTIC_META: dict[str, dict[str, str]] = {
    "CXCL10_CXCR3": {
        "label": "CXCL10-CXCR3",
        "ligand": "CXCL10",
        "receptor": "CXCR3",
        "artifact_receptor": "CXCR3",
        "interaction_id": "interaction_804881ff4ef080702b2ed06bbad83067",
        "role": "candidate_ligand_supported_positive",
    },
    "IFNB1_IFNAR1_IFNAR2": {
        "label": "IFNB1-IFNAR1/2",
        "ligand": "IFNB1",
        "receptor": "IFNAR1/2",
        "artifact_receptor": "IFNAR1_IFNAR2",
        "interaction_id": "interaction_82ef1d991adffaa8770cbb6bc7068598",
        "role": "exogenous_ifn_beta_attribution_guard",
    },
    "CCL5_CCR5": {
        "label": "CCL5-CCR5",
        "ligand": "CCL5",
        "receptor": "CCR5",
        "artifact_receptor": "CCR5",
        "interaction_id": "interaction_e7aec3cffff41336c1d75206fd160b0e",
        "role": "stable_or_weak_change_control",
    },
}
ROLE_POLICY = {
    "default": (0.0, 0.1),
    "minimum_effect_0p02": (0.02, 0.1),
    "receptor_threshold_0p01": (0.0, 0.01),
}
ROLE_LABELS = {
    "default": "Default (delta = 0)",
    "minimum_effect_0p02": "Minimum-effect sensitivity (delta = 0.02)",
    "receptor_threshold_0p01": "Receptor sensitivity (threshold = 0.01)",
}
EXPECTED_RESOURCE_SIGNATURE = {
    "cellchat_manifest_digest": (
        "8207af7818c6616c24ebaa84f746a32424291f1f586d181138a67167455f79f7"
    ),
    "cellchat_selected_digest": (
        "ba64b59f9901c816ca1711d1e50ee5faf023a4192a707762be6b5b735a4e8ff8"
    ),
    "cellchat_selected_interactions": 3,
    "nichenet_manifest_digest": (
        "87cc10d544e7d02025b99eab65bd292ee03c195c50daba5f9077c4d5ec7460cc"
    ),
    "nichenet_selected_digest": (
        "399608a30dac12c62171c7cf275d1272c3c4dfc7d19ac67e37ff4dd3f129a097"
    ),
    "nichenet_diagnostic_drivers": 3,
    "nichenet_diagnostic_links": 60,
    "nichenet_diagnostic_targets": 57,
}
GATE_POLICY = {
    "matrix_semantics": "unit_normalized_profile_if_eligible_else_zero",
    "policy": "hard_eligibility_v2",
    "threshold_operator": "receptor_gate >= threshold",
    "version": 2,
}

BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
BLACK = "#222222"
MID_GRAY = "#777777"
LIGHT_GRAY = "#D9D9D9"
VERY_LIGHT_GRAY = "#F4F4F4"


@dataclass(frozen=True)
class Snapshot:
    """One validated real-data diagnostic artifact."""

    role: str
    path: Path
    payload: Mapping[str, Any]
    minimum_effect: float
    receptor_threshold: float
    partitions: Mapping[str, Mapping[str, Any]]
    support_rows: tuple[Mapping[str, Any], ...]
    receptor_rows: tuple[Mapping[str, Any], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, Any], value)


def _rows(value: object, *, field: str, count: int) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not all(
        isinstance(row, Mapping) for row in value
    ):
        raise ValueError(f"{field} must be a list of objects")
    if len(value) != count:
        raise ValueError(f"{field} must contain exactly {count} rows")
    return tuple(cast(Mapping[str, Any], row) for row in value)


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require_equal(label: str, left: object, right: object) -> None:
    if _stable_json(left) != _stable_json(right):
        raise ValueError(f"artifacts disagree on frozen {label}")


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(payload, field=str(path))


def _resource_signature(payload: Mapping[str, Any]) -> dict[str, object]:
    resources = _mapping(payload.get("resources"), field="resources")
    cellchat = _mapping(resources.get("cellchat"), field="resources.cellchat")
    nichenet = _mapping(resources.get("nichenet"), field="resources.nichenet")
    diagnostic = _mapping(
        nichenet.get("diagnostic_target_prior"),
        field="resources.nichenet.diagnostic_target_prior",
    )
    return {
        "cellchat_manifest_digest": cellchat.get("manifest_digest"),
        "cellchat_selected_digest": cellchat.get("selected_interaction_id_digest"),
        "cellchat_selected_interactions": cellchat.get("n_selected_interactions"),
        "nichenet_manifest_digest": nichenet.get("manifest_digest"),
        "nichenet_selected_digest": diagnostic.get("selected_target_link_digest"),
        "nichenet_diagnostic_drivers": diagnostic.get("n_drivers"),
        "nichenet_diagnostic_links": diagnostic.get("n_links"),
        "nichenet_diagnostic_targets": diagnostic.get("n_targets"),
    }


def _partition_signature(payload: Mapping[str, Any]) -> dict[str, object]:
    frozen = _mapping(payload.get("frozen_manifest"), field="frozen_manifest")
    manifest = _mapping(
        frozen.get("subject_partition"), field="frozen_manifest.subject_partition"
    )
    raw_partitions = _rows(manifest.get("partitions"), field="partitions", count=2)
    partitions = [
        {key: value for key, value in partition.items() if key != "fold_id"}
        for partition in raw_partitions
    ]
    partitions.sort(key=lambda row: str(row.get("fold_partition_id")))
    return {
        "subject_partition_manifest_id": manifest.get("subject_partition_manifest_id"),
        "outer_fold_partition_seed": manifest.get("outer_fold_partition_seed"),
        "partition_seed_lineage": manifest.get("partition_seed_lineage"),
        "sensitivity_alignment_scope": manifest.get("sensitivity_alignment_scope"),
        "downstream_comparable": manifest.get(
            "downstream_inner_tuned_scores_comparable_across_sensitivity_runs"
        ),
        "partitions": partitions,
    }


def _crossfit_policy_signature(configuration: Mapping[str, Any]) -> dict[str, Any]:
    spec = dict(_mapping(configuration.get("crossfit_spec"), field="crossfit_spec"))
    if spec.get("autonomous_program_use_scope") != "biological_analysis":
        raise ValueError("real-data crossfit must require biological autonomous programs")
    for field in (
        "receptor_gate_threshold",
        "repeat_id",
        "spec_id",
        "training_spec_id",
    ):
        spec.pop(field, None)
    return spec


def _validate_subject_manifest(value: object, *, field: str, n_subjects: int) -> None:
    manifest = _mapping(value, field=field)
    if manifest.get("n_subjects") != n_subjects:
        raise ValueError(f"{field} has the wrong subject count")
    if manifest.get("privacy_semantics") != SUBJECT_PRIVACY_SEMANTICS:
        raise ValueError(f"{field} must state that its digest is not anonymization")
    digest = manifest.get("subject_set_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{field} lacks a stable subject-set digest")


def _validate_diagnostic_manifest(payload: Mapping[str, Any]) -> None:
    frozen = _mapping(payload.get("frozen_manifest"), field="frozen_manifest")
    rows = _rows(
        frozen.get("diagnostic_interactions"),
        field="frozen_manifest.diagnostic_interactions",
        count=3,
    )
    observed = {str(row.get("diagnostic_id")): row for row in rows}
    if set(observed) != set(DIAGNOSTICS):
        raise ValueError("diagnostic interaction identities changed")
    for diagnostic_id, expected in DIAGNOSTIC_META.items():
        row = observed[diagnostic_id]
        if (
            row.get("harmonized_interaction_id") != expected["interaction_id"]
            or row.get("ligand") != expected["ligand"]
            or row.get("receptor") != expected["artifact_receptor"]
            or row.get("pre_specified_role") != expected["role"]
        ):
            raise ValueError(f"diagnostic molecular identity changed: {diagnostic_id}")


def _validate_support_rows(
    rows: tuple[Mapping[str, Any], ...],
    partitions: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_keys = {
        (partition_id, diagnostic_id, receiver)
        for partition_id in PARTITION_IDS
        for diagnostic_id in DIAGNOSTICS
        for receiver in RECEIVERS
    }
    observed_keys: list[tuple[str, str, str]] = []
    for row in rows:
        partition_id = str(row.get("fold_partition_id"))
        diagnostic_id = str(row.get("diagnostic_id"))
        receiver = str(row.get("receiver"))
        key = (partition_id, diagnostic_id, receiver)
        observed_keys.append(key)
        if key not in expected_keys:
            raise ValueError(f"unexpected support key: {key}")
        partition = partitions[partition_id]
        if row.get("fold_id") != partition.get("fold_id"):
            raise ValueError("support row is not bound to its partition-local fold")
        _require_equal(
            "support training subject manifest",
            row.get("training_subject_manifest"),
            partition.get("training_subject_manifest"),
        )
        _require_equal(
            "support heldout subject manifest",
            row.get("heldout_subject_manifest"),
            partition.get("heldout_subject_manifest"),
        )
        expected_interaction = DIAGNOSTIC_META[diagnostic_id]["interaction_id"]
        if row.get("interaction_id") != expected_interaction:
            raise ValueError(f"support interaction changed: {diagnostic_id}")
        if row.get("n_complete") != 4 or row.get("minimum_complete_subjects") != 4:
            raise ValueError("support row must use four complete training subjects")
        if row.get("multiplicity_family_size") != 3:
            raise ValueError(
                "support row must retain the three-interaction Holm family"
            )
        _finite_float(row.get("mean_effect"), field="mean_effect")
        for field in ("raw_one_sided_p_value", "holm_adjusted_p_value"):
            value = _finite_float(row.get(field), field=field)
            if not 0 <= value <= 1:
                raise ValueError(f"{field} must lie in [0, 1]")
        status = row.get("status")
        gate_status = row.get("ligand_contrast_gate_status")
        gate = _finite_float(row.get("ligand_contrast_gate"), field="gate")
        if status not in {"supported", "unsupported"} or gate_status != status:
            raise ValueError("support and ligand-gate statuses disagree")
        if gate != (1.0 if status == "supported" else 0.0):
            raise ValueError("ligand gate does not match support status")
        if status == "supported" and (
            row.get("reason_code") is not None
            or row.get("ligand_contrast_gate_reason_code") is not None
        ):
            raise ValueError("supported rows must not carry a failure reason")
    if (
        len(observed_keys) != len(set(observed_keys))
        or set(observed_keys) != expected_keys
    ):
        raise ValueError("support rows lack exact 12-key stable coverage")

    duplicate_fields = (
        "interaction_id",
        "n_complete",
        "minimum_complete_subjects",
        "mean_effect",
        "raw_one_sided_p_value",
        "holm_adjusted_p_value",
        "holm_rank",
        "multiplicity_family_size",
        "status",
        "reason_code",
        "ligand_contrast_gate",
        "ligand_contrast_gate_status",
        "ligand_contrast_gate_reason_code",
    )
    by_fold: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        fold_key = (str(row["fold_partition_id"]), str(row["diagnostic_id"]))
        by_fold.setdefault(fold_key, []).append(row)
    for fold_key, duplicates in by_fold.items():
        if len(duplicates) != len(RECEIVERS):
            raise ValueError(f"support audit receiver replication changed: {fold_key}")
        reference = {field: duplicates[0].get(field) for field in duplicate_fields}
        if any(
            {field: row.get(field) for field in duplicate_fields} != reference
            for row in duplicates[1:]
        ):
            raise ValueError(
                "receiver-duplicated support rows are not identical fold evidence"
            )


def _validate_receptor_rows(
    rows: tuple[Mapping[str, Any], ...],
    partitions: Mapping[str, Mapping[str, Any]],
    *,
    threshold: float,
) -> None:
    expected_keys = {
        (partition_id, diagnostic_id, receiver)
        for partition_id in PARTITION_IDS
        for diagnostic_id in DIAGNOSTICS
        for receiver in RECEIVERS
    }
    observed_keys: list[tuple[str, str, str]] = []
    for row in rows:
        partition_id = str(row.get("fold_partition_id"))
        diagnostic_id = str(row.get("diagnostic_id"))
        receiver = str(row.get("receiver"))
        key = (partition_id, diagnostic_id, receiver)
        observed_keys.append(key)
        if key not in expected_keys:
            raise ValueError(f"unexpected receptor key: {key}")
        partition = partitions[partition_id]
        if row.get("fold_id") != partition.get("fold_id"):
            raise ValueError("receptor row is not bound to its partition-local fold")
        _require_equal(
            "receptor training subject manifest",
            row.get("training_subject_manifest"),
            partition.get("training_subject_manifest"),
        )
        _require_equal(
            "receptor heldout subject manifest",
            row.get("heldout_subject_manifest"),
            partition.get("heldout_subject_manifest"),
        )
        expected = DIAGNOSTIC_META[diagnostic_id]
        if (
            row.get("interaction_id") != expected["interaction_id"]
            or row.get("driver_id") != expected["ligand"]
        ):
            raise ValueError(f"receptor interaction identity changed: {diagnostic_id}")
        gate = _finite_float(row.get("receptor_gate"), field="receptor_gate")
        observed_threshold = _finite_float(
            row.get("receptor_gate_threshold"), field="receptor_gate_threshold"
        )
        if observed_threshold != threshold or not 0 <= gate <= 1:
            raise ValueError("receptor gate threshold or value changed")
        if row.get("receptor_eligible") is not (gate >= threshold):
            raise ValueError("receptor eligibility does not implement its threshold")
        if row.get("receptor_gate_policy") != GATE_POLICY:
            raise ValueError("receptor gate policy changed")
    if (
        len(observed_keys) != len(set(observed_keys))
        or set(observed_keys) != expected_keys
    ):
        raise ValueError("receptor rows lack exact 12-key stable coverage")


def _load_snapshot(role: str, path: Path) -> Snapshot:
    if role not in ROLE_POLICY:
        raise ValueError(f"unsupported artifact role: {role}")
    payload = _read_json(path)
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"{role} artifact schema changed")
    if payload.get("scope") != ARTIFACT_SCOPE:
        raise ValueError(f"{role} artifact scope changed")
    if payload.get("claims") != CLAIMS:
        raise ValueError(f"{role} artifact claim boundary changed")
    policy = _mapping(
        payload.get("sensitivity_comparison_policy"),
        field="sensitivity_comparison_policy",
    )
    for key, expected in SENSITIVITY_POLICY.items():
        if policy.get(key) != expected:
            raise ValueError(f"{role} sensitivity policy changed: {key}")

    configuration = _mapping(payload.get("configuration"), field="configuration")
    if configuration.get("sha256") != EXPECTED_CONFIG_SHA256:
        raise ValueError(f"{role} config checksum changed")
    minimum_effect = _finite_float(
        configuration.get("ligand_contrast_minimum_effect"),
        field="ligand_contrast_minimum_effect",
    )
    receptor_threshold = _finite_float(
        configuration.get("receptor_gate_threshold"),
        field="receptor_gate_threshold",
    )
    if (minimum_effect, receptor_threshold) != ROLE_POLICY[role]:
        raise ValueError(f"{role} does not implement its frozen sensitivity policy")

    input_manifest = _mapping(payload.get("input"), field="input")
    if (
        input_manifest.get("expected_sha256") != EXPECTED_INPUT_SHA256
        or input_manifest.get("observed_sha256") != EXPECTED_INPUT_SHA256
        or input_manifest.get("checksum_verified") is not True
        or input_manifest.get("conversion_lineage_verified") is not True
        or input_manifest.get("analysis_shape") != [7427, 32938]
    ):
        raise ValueError(f"{role} input identity or full-transcriptome scope changed")
    _validate_subject_manifest(
        input_manifest.get("subject_manifest"),
        field="input.subject_manifest",
        n_subjects=8,
    )
    if _resource_signature(payload) != EXPECTED_RESOURCE_SIGNATURE:
        raise ValueError(f"{role} resource identity changed")
    _validate_diagnostic_manifest(payload)

    provenance = _mapping(payload.get("source_provenance"), field="source_provenance")
    commit = provenance.get("git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or provenance.get("git_worktree_dirty") is not False
        or provenance.get("git_status_porcelain") != []
    ):
        raise ValueError(f"{role} lacks clean, commit-bound source provenance")

    frozen = _mapping(payload.get("frozen_manifest"), field="frozen_manifest")
    partition_manifest = _mapping(
        frozen.get("subject_partition"), field="subject_partition"
    )
    if (
        partition_manifest.get("subject_partition_manifest_id")
        != EXPECTED_PARTITION_MANIFEST_ID
        or partition_manifest.get("sensitivity_alignment_scope")
        != "contrast_support_only"
        or partition_manifest.get(
            "downstream_inner_tuned_scores_comparable_across_sensitivity_runs"
        )
        is not False
    ):
        raise ValueError(f"{role} partition comparison boundary changed")
    raw_partitions = _rows(
        partition_manifest.get("partitions"), field="subject partitions", count=2
    )
    partitions = {str(row.get("fold_partition_id")): row for row in raw_partitions}
    if set(partitions) != set(PARTITION_IDS):
        raise ValueError(f"{role} stable partition identities changed")
    for partition_id, row in partitions.items():
        if not isinstance(row.get("fold_id"), str):
            raise ValueError(f"{role} partition {partition_id} lacks a local fold ID")
        _validate_subject_manifest(
            row.get("training_subject_manifest"),
            field="training_subject_manifest",
            n_subjects=4,
        )
        _validate_subject_manifest(
            row.get("heldout_subject_manifest"),
            field="heldout_subject_manifest",
            n_subjects=4,
        )

    paired = _mapping(payload.get("paired_summaries"), field="paired_summaries")
    if (
        paired.get("within_run_descriptive_only") is not True
        or paired.get("comparable_across_threshold_sensitivity_runs") is not False
    ):
        raise ValueError(f"{role} downstream score comparison boundary changed")
    audit = _mapping(payload.get("crossfit_audit"), field="crossfit_audit")
    if (
        audit.get("complete_pipeline_oof_certified") is not False
        or audit.get("receiver_autonomous_nuisance_intentionally_unresolved")
        is not True
    ):
        raise ValueError(f"{role} unexpectedly claims complete-pipeline certification")

    support_rows = _rows(
        payload.get("fold_receiver_interaction_supports"),
        field="fold_receiver_interaction_supports",
        count=12,
    )
    receptor_rows = _rows(
        payload.get("fold_receiver_interaction_receptor_gates"),
        field="fold_receiver_interaction_receptor_gates",
        count=12,
    )
    _validate_support_rows(support_rows, partitions)
    _validate_receptor_rows(receptor_rows, partitions, threshold=receptor_threshold)
    return Snapshot(
        role=role,
        path=path,
        payload=payload,
        minimum_effect=minimum_effect,
        receptor_threshold=receptor_threshold,
        partitions=partitions,
        support_rows=support_rows,
        receptor_rows=receptor_rows,
    )


def _support_fold_index(
    snapshot: Snapshot,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in snapshot.support_rows:
        key = (str(row["fold_partition_id"]), str(row["diagnostic_id"]))
        index.setdefault(key, row)
    if len(index) != len(PARTITION_IDS) * len(DIAGNOSTICS):
        raise ValueError(
            "support rows did not collapse to exactly six fold diagnostics"
        )
    return index


def _receptor_index(
    snapshot: Snapshot,
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {
        (
            str(row["fold_partition_id"]),
            str(row["diagnostic_id"]),
            str(row["receiver"]),
        ): row
        for row in snapshot.receptor_rows
    }


def _validate_snapshot_set(snapshots: Sequence[Snapshot]) -> None:
    if tuple(snapshot.role for snapshot in snapshots) != tuple(ROLE_POLICY):
        raise ValueError("report requires the three frozen artifact roles in order")
    reference = snapshots[0]
    for snapshot in snapshots[1:]:
        _require_equal(
            "schema",
            reference.payload["schema_version"],
            snapshot.payload["schema_version"],
        )
        _require_equal("input", reference.payload["input"], snapshot.payload["input"])
        _require_equal(
            "resources", reference.payload["resources"], snapshot.payload["resources"]
        )
        _require_equal(
            "source provenance",
            reference.payload["source_provenance"],
            snapshot.payload["source_provenance"],
        )
        _require_equal(
            "diagnostic manifest",
            _mapping(reference.payload["frozen_manifest"], field="frozen_manifest")[
                "diagnostic_interactions"
            ],
            _mapping(snapshot.payload["frozen_manifest"], field="frozen_manifest")[
                "diagnostic_interactions"
            ],
        )
        _require_equal(
            "subject partition manifest",
            _partition_signature(reference.payload),
            _partition_signature(snapshot.payload),
        )
        reference_configuration = _mapping(
            reference.payload["configuration"], field="configuration"
        )
        configuration = _mapping(
            snapshot.payload["configuration"], field="configuration"
        )
        _require_equal(
            "CRYCHIC config",
            reference_configuration.get("crychic_config"),
            configuration.get("crychic_config"),
        )
        _require_equal(
            "crossfit policy excluding sensitivity-bound IDs",
            _crossfit_policy_signature(reference_configuration),
            _crossfit_policy_signature(configuration),
        )

    support_indexes = {
        snapshot.role: _support_fold_index(snapshot) for snapshot in snapshots
    }
    reference_support = support_indexes["default"]
    sensitivity_support = support_indexes["minimum_effect_0p02"]
    receptor_support = support_indexes["receptor_threshold_0p01"]
    for key in sorted(reference_support):
        if (
            reference_support[key]["mean_effect"]
            != sensitivity_support[key]["mean_effect"]
        ):
            raise ValueError(
                "minimum-effect artifacts are not aligned by fold_partition_id"
            )
        comparable_fields = (
            "interaction_id",
            "n_complete",
            "mean_effect",
            "raw_one_sided_p_value",
            "holm_adjusted_p_value",
            "holm_rank",
            "status",
            "reason_code",
            "ligand_contrast_gate",
        )
        if any(
            reference_support[key].get(field) != receptor_support[key].get(field)
            for field in comparable_fields
        ):
            raise ValueError(
                "receptor-threshold run changed fold-level ligand support evidence"
            )

    receptor_indexes = {
        snapshot.role: _receptor_index(snapshot) for snapshot in snapshots
    }
    reference_receptors = receptor_indexes["default"]
    for role, index in receptor_indexes.items():
        for receptor_key in sorted(reference_receptors):
            for field in (
                "interaction_id",
                "driver_id",
                "receptor_gate",
                "receptor_gate_policy",
                "receptor_evidence_digest",
            ):
                if reference_receptors[receptor_key].get(field) != index[
                    receptor_key
                ].get(field):
                    raise ValueError(
                        f"{role} changed partition-aligned receptor evidence"
                    )


def _timestamp(value: str | None) -> str:
    raw = DEFAULT_GENERATED_AT if value is None else value
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("generated_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("generated_at must include an explicit timezone")
    return parsed.astimezone(UTC).isoformat()


def _support_audit_table(snapshots: Sequence[Snapshot]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for snapshot in snapshots:
        for row in snapshot.support_rows:
            diagnostic_id = str(row["diagnostic_id"])
            records.append(
                {
                    "artifact_role": snapshot.role,
                    "minimum_effect": snapshot.minimum_effect,
                    "receptor_threshold": snapshot.receptor_threshold,
                    "fold_partition_id": row["fold_partition_id"],
                    "partition_label": PARTITION_LABELS[str(row["fold_partition_id"])],
                    "fold_id_audit_only_not_alignment_key": row["fold_id"],
                    "receiver": row["receiver"],
                    "diagnostic_id": diagnostic_id,
                    "interaction_label": DIAGNOSTIC_META[diagnostic_id]["label"],
                    "mean_effect": row["mean_effect"],
                    "raw_one_sided_p_value_training_gate_only": row[
                        "raw_one_sided_p_value"
                    ],
                    "holm_adjusted_p_value_training_gate_only": row[
                        "holm_adjusted_p_value"
                    ],
                    "holm_rank": row["holm_rank"],
                    "n_complete_training_subjects": row["n_complete"],
                    "support_status": row["status"],
                    "ligand_contrast_gate": row["ligand_contrast_gate"],
                    "receiver_rows_repeat_same_fold_contrast": True,
                }
            )
    return pd.DataFrame(records).sort_values(
        ["artifact_role", "fold_partition_id", "diagnostic_id", "receiver"],
        kind="stable",
        ignore_index=True,
    )


def _fold_support_table(snapshots: Mapping[str, Snapshot]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for role in ("default", "minimum_effect_0p02"):
        snapshot = snapshots[role]
        for (partition_id, diagnostic_id), row in sorted(
            _support_fold_index(snapshot).items()
        ):
            meta = DIAGNOSTIC_META[diagnostic_id]
            records.append(
                {
                    "artifact_role": role,
                    "artifact_label": ROLE_LABELS[role],
                    "minimum_effect": snapshot.minimum_effect,
                    "fold_partition_id": partition_id,
                    "partition_label": PARTITION_LABELS[partition_id],
                    "diagnostic_id": diagnostic_id,
                    "interaction_label": meta["label"],
                    "ligand": meta["ligand"],
                    "receptor": meta["receptor"],
                    "mean_effect": row["mean_effect"],
                    "raw_one_sided_p_value_training_gate_only": row[
                        "raw_one_sided_p_value"
                    ],
                    "holm_adjusted_p_value_training_gate_only": row[
                        "holm_adjusted_p_value"
                    ],
                    "holm_rank": row["holm_rank"],
                    "n_complete_training_subjects": row["n_complete"],
                    "support_status": row["status"],
                    "ligand_contrast_gate": row["ligand_contrast_gate"],
                    "evidence_unit": "training_fold_partition",
                }
            )
    return pd.DataFrame(records).sort_values(
        ["artifact_role", "diagnostic_id", "fold_partition_id"],
        kind="stable",
        ignore_index=True,
    )


def _receptor_audit_table(snapshots: Mapping[str, Snapshot]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for role in ("default", "receptor_threshold_0p01"):
        snapshot = snapshots[role]
        for row in snapshot.receptor_rows:
            diagnostic_id = str(row["diagnostic_id"])
            records.append(
                {
                    "artifact_role": role,
                    "receptor_threshold": snapshot.receptor_threshold,
                    "fold_partition_id": row["fold_partition_id"],
                    "partition_label": PARTITION_LABELS[str(row["fold_partition_id"])],
                    "fold_id_audit_only_not_alignment_key": row["fold_id"],
                    "receiver": row["receiver"],
                    "diagnostic_id": diagnostic_id,
                    "interaction_label": DIAGNOSTIC_META[diagnostic_id]["label"],
                    "receptor_gate": row["receptor_gate"],
                    "receptor_eligible": row["receptor_eligible"],
                    "gate_policy": row["receptor_gate_policy"]["policy"],
                }
            )
    return pd.DataFrame(records).sort_values(
        ["artifact_role", "diagnostic_id", "receiver", "fold_partition_id"],
        kind="stable",
        ignore_index=True,
    )


def _support_receptor_matrix(
    snapshots: Mapping[str, Snapshot],
) -> pd.DataFrame:
    support = _support_fold_index(snapshots["default"])
    records: list[dict[str, object]] = []
    for role in ("default", "receptor_threshold_0p01"):
        snapshot = snapshots[role]
        receptors = _receptor_index(snapshot)
        for diagnostic_id in DIAGNOSTICS:
            for receiver in RECEIVERS:
                support_values = [
                    support[(partition_id, diagnostic_id)]["status"] == "supported"
                    for partition_id in PARTITION_IDS
                ]
                receptor_values = [
                    bool(
                        receptors[(partition_id, diagnostic_id, receiver)][
                            "receptor_eligible"
                        ]
                    )
                    for partition_id in PARTITION_IDS
                ]
                joint = [
                    ligand and receptor
                    for ligand, receptor in zip(
                        support_values, receptor_values, strict=True
                    )
                ]
                records.append(
                    {
                        "artifact_role": role,
                        "receptor_threshold": snapshot.receptor_threshold,
                        "diagnostic_id": diagnostic_id,
                        "interaction_label": DIAGNOSTIC_META[diagnostic_id]["label"],
                        "receiver": receiver,
                        "n_fold_partitions": len(PARTITION_IDS),
                        "ligand_supported_folds": sum(support_values),
                        "receptor_eligible_folds": sum(receptor_values),
                        "joint_gate_eligible_folds": sum(joint),
                        "ligand_support_unit": "fold_partition_not_receiver_row",
                    }
                )
    return pd.DataFrame(records)


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "savefig.transparent": False,
            "svg.fonttype": "none",
            "svg.hashsalt": SVG_HASHSALT,
            "pdf.fonttype": 42,
        }
    )


def _save_figure(
    figure: Figure,
    figures_dir: Path,
    stem: str,
    generated_at: str,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fixed_datetime = datetime.fromisoformat(generated_at).astimezone(UTC)
    figure.savefig(
        figures_dir / f"{stem}.svg",
        facecolor="white",
        metadata={
            "Creator": "CRYCHIC Kang 2018 ligand-gate v3 report generator",
            "Date": generated_at,
            "Title": stem,
            "Description": "Exploratory training-fold diagnostic; noncertifying.",
        },
    )
    figure.savefig(
        figures_dir / f"{stem}.pdf",
        facecolor="white",
        metadata={
            "Title": stem,
            "Author": "CRYCHIC developers",
            "Subject": "Exploratory training-fold diagnostic; noncertifying.",
            "Creator": "CRYCHIC Kang 2018 ligand-gate v3 report generator",
            "Producer": "CRYCHIC Kang 2018 ligand-gate v3 report generator",
            "CreationDate": fixed_datetime,
            "ModDate": fixed_datetime,
        },
    )
    figure.savefig(
        figures_dir / f"{stem}.png",
        dpi=300,
        facecolor="white",
        metadata={
            "Software": "CRYCHIC Kang 2018 ligand-gate v3 report generator",
            "Creation Time": generated_at,
        },
    )
    plt.close(figure)


def _plot_fold_support(
    table: pd.DataFrame, figures_dir: Path, generated_at: str
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 4.1), sharey=True)
    figure.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.25, wspace=0.12)
    positions = {diagnostic: index for index, diagnostic in enumerate(DIAGNOSTICS)}
    offsets = {PARTITION_IDS[0]: -0.09, PARTITION_IDS[1]: 0.09}
    markers = {PARTITION_IDS[0]: "o", PARTITION_IDS[1]: "s"}
    for axis, role in zip(axes, ("default", "minimum_effect_0p02"), strict=True):
        subset = table.loc[table["artifact_role"] == role]
        for row in subset.itertuples(index=False):
            supported = row.support_status == "supported"
            diagnostic_id = str(row.diagnostic_id)
            partition_id = str(row.fold_partition_id)
            axis.scatter(
                positions[diagnostic_id] + offsets[partition_id],
                row.mean_effect,
                marker=markers[partition_id],
                s=45,
                facecolor=BLUE if supported else "white",
                edgecolor=BLUE if supported else MID_GRAY,
                linewidth=1.0,
                zorder=3,
            )
        threshold = float(subset["minimum_effect"].iloc[0])
        axis.axhline(
            threshold,
            color=ORANGE,
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )
        axis.text(
            0.98,
            threshold,
            f" null effect = {threshold:.2f}",
            transform=axis.get_yaxis_transform(),
            color=ORANGE,
            va="bottom",
            ha="right",
            fontsize=7,
        )
        axis.set_xticks(
            range(len(DIAGNOSTICS)),
            [DIAGNOSTIC_META[item]["label"] for item in DIAGNOSTICS],
            rotation=22,
            ha="right",
        )
        axis.set_yscale("symlog", linthresh=0.03, linscale=1.1)
        axis.set_ylim(-0.09, 1.25)
        axis.set_yticks([-0.05, 0.0, 0.02, 0.1, 1.0])
        axis.set_yticklabels(["-0.05", "0", "0.02", "0.1", "1.0"])
        axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, zorder=0)
        axis.set_title(ROLE_LABELS[role])
    axes[0].set_ylabel("Mean paired ligand-availability effect (symlog)")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=BLUE,
            markeredgecolor=BLUE,
            label="Training gate supported",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=MID_GRAY,
            label="Training gate unsupported",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=BLACK,
            label="Partition A",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            color=BLACK,
            label="Partition B",
        ),
    ]
    figure.legend(
        handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.98)
    )
    figure.suptitle(
        "Fold-level ligand contrast effects and training-gate decisions",
        y=0.995,
        fontsize=10,
    )
    figure.text(
        0.5,
        0.035,
        "Each point is one training partition (4 donors). Receiver audit duplicates are not independent. "
        "Raw/Holm p-values serve the training gate only.",
        ha="center",
        fontsize=7,
        color=MID_GRAY,
    )
    _save_figure(figure, figures_dir, "figure01_fold_ligand_contrast", generated_at)


def _plot_support_receptor_matrix(
    table: pd.DataFrame, figures_dir: Path, generated_at: str
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 4.1), sharey=True)
    figure.subplots_adjust(left=0.17, right=0.88, top=0.82, bottom=0.19, wspace=0.18)
    cmap = mpl.colors.ListedColormap((VERY_LIGHT_GRAY, ORANGE, GREEN))
    for axis, role in zip(axes, ("default", "receptor_threshold_0p01"), strict=True):
        subset = table.loc[table["artifact_role"] == role]
        matrix: npt.NDArray[np.float64] = np.zeros(
            (len(DIAGNOSTICS), len(RECEIVERS)), dtype=float
        )
        annotations: dict[tuple[int, int], tuple[int, int, int]] = {}
        for row_index, diagnostic_id in enumerate(DIAGNOSTICS):
            for column_index, receiver in enumerate(RECEIVERS):
                row = subset.loc[
                    (subset["diagnostic_id"] == diagnostic_id)
                    & (subset["receiver"] == receiver)
                ].iloc[0]
                ligand = int(row["ligand_supported_folds"])
                receptor = int(row["receptor_eligible_folds"])
                joint = int(row["joint_gate_eligible_folds"])
                matrix[row_index, column_index] = joint
                annotations[(row_index, column_index)] = (ligand, receptor, joint)
        image = axis.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
        axis.set_xticks(range(len(RECEIVERS)), ("CD14+ monocytes", "CD8 T cells"))
        axis.set_yticks(
            range(len(DIAGNOSTICS)),
            [DIAGNOSTIC_META[item]["label"] for item in DIAGNOSTICS],
        )
        axis.tick_params(axis="x", rotation=18)
        for (row_index, column_index), (ligand, receptor, joint) in annotations.items():
            axis.text(
                column_index,
                row_index,
                f"L {ligand}/2 | R {receptor}/2\nJoint {joint}/2",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if joint == 2 else BLACK,
            )
        threshold = float(subset["receptor_threshold"].iloc[0])
        axis.set_title(f"Receptor threshold = {threshold:.2f}")
    colorbar = figure.colorbar(image, ax=list(axes), fraction=0.035, pad=0.04)
    colorbar.set_ticks((0, 1, 2))
    colorbar.set_ticklabels(("0/2", "1/2", "2/2"))
    colorbar.set_label("Joint gate-eligible partitions")
    figure.suptitle(
        "Ligand support and receptor eligibility across two partitions",
        y=0.97,
        fontsize=10,
    )
    figure.text(
        0.5,
        0.04,
        "L: fold-level ligand support (default delta = 0). R: receiver-specific receptor eligibility. "
        "Joint is descriptive and noncausal.",
        ha="center",
        fontsize=7,
        color=MID_GRAY,
    )
    _save_figure(figure, figures_dir, "figure02_support_receptor_matrix", generated_at)


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        path,
        index=False,
        na_rep="",
        float_format="%.17g",
        lineterminator="\n",
    )


def _support_summary(table: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for role in ("default", "minimum_effect_0p02"):
        for diagnostic_id in DIAGNOSTICS:
            subset = table.loc[
                (table["artifact_role"] == role)
                & (table["diagnostic_id"] == diagnostic_id)
            ]
            records.append(
                {
                    "artifact_role": role,
                    "minimum_effect": float(subset["minimum_effect"].iloc[0]),
                    "diagnostic_id": diagnostic_id,
                    "n_fold_partitions": len(subset),
                    "supported_fold_partitions": int(
                        (subset["support_status"] == "supported").sum()
                    ),
                    "fold_mean_effects": [
                        float(value) for value in subset["mean_effect"].tolist()
                    ],
                    "evidence_unit": "fold_partition_id",
                }
            )
    return records


def _metrics_summary(
    snapshots: Sequence[Snapshot],
    fold_support: pd.DataFrame,
    matrix: pd.DataFrame,
    generated_at: str,
) -> dict[str, object]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "claims": dict(CLAIMS),
        "comparison_contract": {
            "allowed": "contrast_support_and_receptor_eligibility_only",
            "alignment_key": "fold_partition_id",
            "receiver_duplicated_ligand_rows_are_independent": False,
            "raw_and_holm_p_value_semantics": "training_gate_diagnostic_only",
            "result_layer_inferential_p_or_q": False,
            "family_member_sender_score_comparison_allowed": False,
        },
        "integrity": {
            "artifact_roles": [snapshot.role for snapshot in snapshots],
            "shared_schema_verified": True,
            "shared_input_verified": True,
            "shared_config_verified": True,
            "shared_resources_verified": True,
            "shared_git_commit_verified": True,
            "shared_partition_manifest_verified": True,
            "support_rows_per_artifact": 12,
            "receptor_rows_per_artifact": 12,
            "independent_fold_support_rows_per_artifact": 6,
            "fold_partitions": 2,
        },
        "ligand_support_by_fold": _support_summary(fold_support),
        "support_receptor_matrix": json.loads(matrix.to_json(orient="records")),
        "bounded_observations": {
            "cxcl10": (
                "CXCL10 ligand support occurred in 2/2 partitions at both minimum-"
                "effect settings; CXCR3 was eligible in CD8 T cells in 2/2 and in "
                "CD14+ monocytes in 0/2 at receptor threshold 0.10. This is plausible, "
                "not causal communication evidence."
            ),
            "ifnb1": (
                "IFNB1 was unsupported in 0/2 partitions. Because the experiment used "
                "exogenous IFN-beta, this interaction is an attribution guard rather "
                "than evidence about an endogenous sender source."
            ),
            "conclusion_boundary": (
                "No biological validation, causal claim, complete-pipeline "
                "certification, default switch, or method-superiority conclusion."
            ),
        },
    }


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(clean(value) for value in headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(clean(value) for value in row) + " |" for row in rows]
    return "\n".join((header, divider, *body))


def _report_markdown(
    snapshots: Sequence[Snapshot],
    fold_support: pd.DataFrame,
    matrix: pd.DataFrame,
    generated_at: str,
) -> str:
    support_lookup = {
        (row["artifact_role"], row["diagnostic_id"]): (
            cast(int, row["supported_fold_partitions"]),
            cast(int, row["n_fold_partitions"]),
        )
        for row in _support_summary(fold_support)
    }

    def receptor_count(role: str, diagnostic_id: str, receiver: str) -> int:
        row = matrix.loc[
            (matrix["artifact_role"] == role)
            & (matrix["diagnostic_id"] == diagnostic_id)
            & (matrix["receiver"] == receiver)
        ].iloc[0]
        return int(row["receptor_eligible_folds"])

    result_rows = []
    for diagnostic_id in DIAGNOSTICS:
        default_count = support_lookup[("default", diagnostic_id)]
        sensitivity_count = support_lookup[("minimum_effect_0p02", diagnostic_id)]
        result_rows.append(
            (
                DIAGNOSTIC_META[diagnostic_id]["label"],
                f"{default_count[0]}/{default_count[1]}",
                f"{sensitivity_count[0]}/{sensitivity_count[1]}",
                f"{receptor_count('default', diagnostic_id, RECEIVERS[0])}/2",
                f"{receptor_count('default', diagnostic_id, RECEIVERS[1])}/2",
            )
        )
    input_rows = [
        (
            snapshot.role,
            snapshot.path.name,
            f"{snapshot.minimum_effect:.2f}",
            f"{snapshot.receptor_threshold:.2f}",
            _sha256(snapshot.path),
        )
        for snapshot in snapshots
    ]
    command = (
        "uv run python benchmarks/report/generate_kang2018_ligand_gate_v3_report.py "
        "--default-artifact benchmark_work/kang2018_ligand_gate_v3_default.json "
        "--minimum-effect-artifact "
        "benchmark_work/kang2018_ligand_gate_v3_min_effect_0p02.json "
        "--receptor-artifact benchmark_work/kang2018_ligand_gate_v3_receptor_0p01.json "
        "--output reports/kang2018_ligand_gate_v3 "
        f"--generated-at {generated_at}"
    )
    return f"""# Kang 2018 ligand-gate v3 development diagnostic

Generated at: `{generated_at}`

> Evidence boundary: this is an exploratory, literature-informed training-fold diagnostic. It does not establish biological validation, causal cell-cell communication, complete-pipeline certification, a default-method switch, or method superiority.

## Result

{
        _markdown_table(
            (
                "Interaction",
                "Ligand support, delta=0",
                "Ligand support, delta=0.02",
                "Receptor eligible, CD14 (threshold=0.10)",
                "Receptor eligible, CD8 (threshold=0.10)",
            ),
            result_rows,
        )
    }

The ligand contrast is one fold-level quantity repeated across the two receiver audit rows. Therefore CXCL10 is supported in **2/2 training partitions, not 4/4 independent observations**. IFNB1 and CCL5 are each unsupported in 0/2 partitions at both ligand minimum-effect settings.

CXCL10-CXCR3 shows the only joint pattern at the default receptor threshold: ligand support in 2/2 partitions and CXCR3 eligibility in CD8 T cells in 2/2 partitions, while CXCR3 is ineligible in CD14+ monocytes in 0/2. This is biologically plausible in the IFN-beta-stimulated PBMC setting, but it is not evidence of causal communication or external biological validation.

IFNB1-IFNAR1/2 is an **exogenous IFN-beta attribution guard**. The experiment applies IFN-beta externally, so endogenous IFNB1 expression cannot identify the exposure source. Its unsupported ligand contrast must not be interpreted as evidence for or against the experimental IFN response. Lowering the receptor threshold changes receptor eligibility for IFNAR and CCR5 but does not create ligand support.

CCL5-CCR5 remains ligand-unsupported despite CCR5 eligibility in monocytes at the default threshold. This illustrates that ligand support and receptor eligibility are separate necessary gates in this diagnostic.

## Figures

![Fold-level ligand contrast](figures/figure01_fold_ligand_contrast.png)

Figure 1 shows mean paired ligand-availability effects for the two partition-aligned training folds under the default and delta=0.02 null effects. Raw and Holm-adjusted p-values are used only to operate the **training gate**. They are not result-layer inferential p-values or q-values, and the report makes no statistical-significance claim.

![Support and receptor matrix](figures/figure02_support_receptor_matrix.png)

Figure 2 combines the default fold-level ligand gate with receiver-specific receptor eligibility at thresholds 0.10 and 0.01. The joint count is descriptive and noncausal. Receptor-threshold sensitivity is separate from the minimum-effect comparison; no downstream family, member, or sender score is compared.

## Integrity contract

All artifacts share the same schema, H5AD identity, complete config checksum, resources, clean git commit, and subject partition manifest. Cross-artifact alignment uses `fold_partition_id`; threshold-sensitive `fold_id` values are audit-only. Donor-set digests are deterministic nonplaintext identifiers for alignment and are explicitly **not anonymization**.

{
        _markdown_table(
            (
                "Role",
                "Artifact",
                "Ligand minimum effect",
                "Receptor threshold",
                "SHA-256",
            ),
            input_rows,
        )
    }

The complete 12-row support and 12-row receptor audit from each input is retained in source CSV files. Main ligand results are deduplicated to 3 interactions x 2 fold partitions. Downstream family/member/sender summaries are intentionally excluded because separately tuned downstream fits are not comparable across thresholds and the receiver-autonomous nuisance stage remains unresolved.

## Reproduction

```bash
{command}
```
"""


def _artifact_hashes(
    output: Path, relative_paths: Sequence[str]
) -> list[dict[str, object]]:
    records = []
    for relative in sorted(relative_paths):
        path = output / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def generate_report(
    *,
    default_artifact: Path,
    minimum_effect_artifact: Path,
    receptor_artifact: Path,
    output_dir: Path,
    generated_at: str | None = None,
    summary_output: Path | None = None,
) -> dict[str, object]:
    """Validate frozen artifacts and generate a deterministic report bundle."""

    snapshots = (
        _load_snapshot("default", default_artifact),
        _load_snapshot("minimum_effect_0p02", minimum_effect_artifact),
        _load_snapshot("receptor_threshold_0p01", receptor_artifact),
    )
    _validate_snapshot_set(snapshots)
    by_role = {snapshot.role: snapshot for snapshot in snapshots}
    timestamp = _timestamp(generated_at)
    output = output_dir.expanduser().resolve()
    figures_dir = output / "figures"
    source_dir = output / "source_data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    support_audit = _support_audit_table(snapshots)
    fold_support = _fold_support_table(by_role)
    receptor_audit = _receptor_audit_table(by_role)
    matrix = _support_receptor_matrix(by_role)
    _write_csv(fold_support, source_dir / "figure01_fold_ligand_contrast.csv")
    _write_csv(matrix, source_dir / "figure02_support_receptor_matrix.csv")
    _write_csv(support_audit, source_dir / "support_audit_rows.csv")
    _write_csv(receptor_audit, source_dir / "receptor_gate_audit_rows.csv")

    _plot_fold_support(fold_support, figures_dir, timestamp)
    _plot_support_receptor_matrix(matrix, figures_dir, timestamp)

    metrics = _metrics_summary(snapshots, fold_support, matrix, timestamp)
    (output / "metrics_summary.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = _report_markdown(snapshots, fold_support, matrix, timestamp)
    (output / "REPORT.md").write_text(report, encoding="utf-8")

    generated_paths = [
        "REPORT.md",
        "metrics_summary.json",
        "source_data/figure01_fold_ligand_contrast.csv",
        "source_data/figure02_support_receptor_matrix.csv",
        "source_data/receptor_gate_audit_rows.csv",
        "source_data/support_audit_rows.csv",
        *[
            f"figures/{stem}.{suffix}"
            for stem in (
                "figure01_fold_ligand_contrast",
                "figure02_support_receptor_matrix",
            )
            for suffix in ("png", "pdf", "svg")
        ],
    ]
    artifact_records = _artifact_hashes(output, generated_paths)
    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": timestamp,
        "evidence_class": "real_data_development_training_gate_noncertifying",
        "claims": dict(CLAIMS),
        "comparison_contract": metrics["comparison_contract"],
        "generator": {
            "path": "benchmarks/report/generate_kang2018_ligand_gate_v3_report.py",
            "sha256": _sha256(Path(__file__)),
            "svg_hashsalt": SVG_HASHSALT,
            "figure_metadata_timestamp": timestamp,
        },
        "inputs": [
            {
                "role": snapshot.role,
                "filename": snapshot.path.name,
                "sha256": _sha256(snapshot.path),
                "minimum_effect": snapshot.minimum_effect,
                "receptor_threshold": snapshot.receptor_threshold,
                "git_commit": _mapping(
                    snapshot.payload["source_provenance"], field="source_provenance"
                )["git_commit"],
                "subject_partition_manifest_id": EXPECTED_PARTITION_MANIFEST_ID,
            }
            for snapshot in snapshots
        ],
        "artifacts_excluding_this_manifest": artifact_records,
        "artifact_count_including_manifest": len(generated_paths) + 1,
    }
    (output / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_path: str | None = None
    if summary_output is not None:
        source_commit = _mapping(
            snapshots[0].payload["source_provenance"], field="source_provenance"
        )["git_commit"]
        compact_summary = {
            "schema_version": "crychic-kang2018-ligand-gate-v3-summary-v1",
            "generated_on": timestamp,
            "claims": dict(CLAIMS),
            "comparison_contract": metrics["comparison_contract"],
            "source_commit": source_commit,
            "frozen_identities": {
                "config_sha256": EXPECTED_CONFIG_SHA256,
                "input_h5ad_sha256": EXPECTED_INPUT_SHA256,
                "resources": dict(EXPECTED_RESOURCE_SIGNATURE),
                "subject_partition_manifest_id": EXPECTED_PARTITION_MANIFEST_ID,
                "fold_partition_ids": list(PARTITION_IDS),
            },
            "input_artifacts": [
                {
                    "role": snapshot.role,
                    "relative_path": f"benchmark_work/{snapshot.path.name}",
                    "sha256": _sha256(snapshot.path),
                    "minimum_effect": snapshot.minimum_effect,
                    "receptor_threshold": snapshot.receptor_threshold,
                }
                for snapshot in snapshots
            ],
            "fold_deduplicated_ligand_support": _support_summary(fold_support),
            "receptor_sensitivity": json.loads(matrix.to_json(orient="records")),
            "report_artifacts": [
                *artifact_records,
                {
                    "path": "report_manifest.json",
                    "bytes": (output / "report_manifest.json").stat().st_size,
                    "sha256": _sha256(output / "report_manifest.json"),
                },
            ],
        }
        summary = summary_output.expanduser().resolve()
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(
            json.dumps(compact_summary, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        summary_path = str(summary)
    return {
        "output_dir": str(output),
        "report": str(output / "REPORT.md"),
        "summary": summary_path,
        "artifact_count": len(generated_paths) + 1,
        "claims": dict(CLAIMS),
        "input_sha256": {
            snapshot.role: _sha256(snapshot.path) for snapshot in snapshots
        },
    }


def _parser(repo_root: Path) -> argparse.ArgumentParser:
    workspace = repo_root.parent
    benchmark_work = workspace / "benchmark_work"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--default-artifact",
        type=Path,
        default=benchmark_work / "kang2018_ligand_gate_v3_default.json",
    )
    parser.add_argument(
        "--minimum-effect-artifact",
        type=Path,
        default=benchmark_work / "kang2018_ligand_gate_v3_min_effect_0p02.json",
    )
    parser.add_argument(
        "--receptor-artifact",
        type=Path,
        default=benchmark_work / "kang2018_ligand_gate_v3_receptor_0p01.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "reports/kang2018_ligand_gate_v3",
    )
    parser.add_argument(
        "--generated-at",
        default=DEFAULT_GENERATED_AT,
        help="fixed ISO-8601 timestamp used in all report and figure metadata",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=repo_root / "benchmarks/results/kang2018_ligand_gate_v3_summary.json",
        help="tracked compact summary JSON; omit only through the Python API",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    args = _parser(repo_root).parse_args(argv)
    result = generate_report(
        default_artifact=args.default_artifact,
        minimum_effect_artifact=args.minimum_effect_artifact,
        receptor_artifact=args.receptor_artifact,
        output_dir=args.output,
        generated_at=args.generated_at,
        summary_output=args.summary_output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
