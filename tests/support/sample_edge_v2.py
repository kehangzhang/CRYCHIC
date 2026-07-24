from __future__ import annotations

import numpy as np
import pandas as pd

from crychic.core import canonical_json
from crychic.scoring import (
    SAMPLE_EDGE_SCORE_V2_COLUMNS,
    SampleEdgeScoreV2,
    SampleEdgeScoreV2Provenance,
)


def sample_edge_provenance() -> SampleEdgeScoreV2Provenance:
    return SampleEdgeScoreV2Provenance(
        fold_id="fold-1",
        repeat_id="repeat-1",
        training_subject_ids=("subject-1", "subject-2"),
        application_subject_ids=("subject-3", "subject-4"),
        training_input_digest="training-digest",
        application_input_digest="application-digest",
        config_digest="config-digest",
        resource_id="resource-id",
        resource_version="resource-v1",
        resource_manifest_digest="resource-digest",
        interaction_universe_id="universe-id",
        transform_manifest_id="transform-id",
        seed_lineage_id="seed-lineage-id",
        score_version="sample-comparable-activity-test-v2",
        package_version="0.0.test",
    )


def sample_edge_scores() -> SampleEdgeScoreV2:
    provenance = sample_edge_provenance()
    lineage = canonical_json(provenance.to_dict())
    common = {
        "sample_id": "sample-3",
        "subject_id": "subject-3",
        "condition": "treated",
        "context_id": "context-treated",
        "fold_id": provenance.fold_id,
        "receiver": "receiver",
        "interaction_id": "interaction-1",
        "ligand": "L1",
        "receptor": "R1",
        "parent_peak_raw": 2.0,
        "parent_total_raw": 3.0,
        "parent_mean_raw": 1.5,
        "program_signed": np.nan,
        "program_status": "not_computed",
        "program_reason_code": "program_head_not_computed",
        "program_functional_id": None,
        "coupling_prior": np.nan,
        "coupling_status": "not_computed",
        "coupling_reason_code": "coupling_head_not_computed",
        "coupling_functional_id": None,
        "active_probability": np.nan,
        "occurrence_status": "not_computed",
        "occurrence_reason_code": "occurrence_head_not_computed",
        "occurrence_functional_id": None,
        "null_sender_attribution": 0.1,
        "attribution_status": "observed",
        "attribution_reason_code": None,
        "attribution_functional_id": "attribution-functional-test-v2",
        "cell_count_reliability": 0.8,
        "reliability_weight": 0.8,
        "candidate_sender_count": 2,
        "effective_candidate_count": 2,
        "coverage_status": "measured",
        "structural_impossibility": False,
        "selection_stability": np.nan,
        "status": "observed",
        "reason_code": None,
        "score_version": provenance.score_version,
        "activity_functional_id": provenance.provenance_id,
        "out_of_fold": True,
        "provenance": lineage,
    }
    rows = [
        {
            **common,
            "sender": "sender-a",
            "ligand_activity_raw": 2.5,
            "receptor_activity_raw": 1.5,
            "sender_detection_raw": 2.0,
            "mechanism_support": 0.7,
            "sender_attribution": 0.6,
        },
        {
            **common,
            "sender": "sender-b",
            "ligand_activity_raw": 0.5,
            "receptor_activity_raw": 1.5,
            "sender_detection_raw": 1.0,
            "mechanism_support": 0.4,
            "sender_attribution": 0.3,
        },
    ]
    table = pd.DataFrame(rows, columns=SAMPLE_EDGE_SCORE_V2_COLUMNS)
    return SampleEdgeScoreV2(table=table, provenance=provenance)
