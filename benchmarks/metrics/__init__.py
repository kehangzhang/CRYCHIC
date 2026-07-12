"""Benchmark metrics with explicit real-data and simulation semantics."""

from .exploratory import (
    paired_descriptive_effects,
    paired_proportion_stability,
    rank_concordance,
    top_k_jaccard,
)
from .multicondition import (
    aggregate_loso_primary_endpoint,
    cross_method_concordance,
    external_long_to_score_table,
    paired_differential_loso_reproducibility,
    paired_edge_effects,
    score_coverage_summary,
    summarize_loso_primary_endpoint,
    summarize_run_performance,
    synthetic_edge_truth_metrics,
    unpaired_differential_split_half_reproducibility,
    unpaired_edge_effects,
    unpaired_leave_one_subject_influence,
    validate_score_table,
    within_context_reproducibility,
)

__all__ = [
    "aggregate_loso_primary_endpoint",
    "cross_method_concordance",
    "external_long_to_score_table",
    "paired_descriptive_effects",
    "paired_differential_loso_reproducibility",
    "paired_edge_effects",
    "paired_proportion_stability",
    "rank_concordance",
    "score_coverage_summary",
    "summarize_loso_primary_endpoint",
    "summarize_run_performance",
    "synthetic_edge_truth_metrics",
    "top_k_jaccard",
    "unpaired_differential_split_half_reproducibility",
    "unpaired_edge_effects",
    "unpaired_leave_one_subject_influence",
    "validate_score_table",
    "within_context_reproducibility",
]
