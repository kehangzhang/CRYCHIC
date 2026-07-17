"""Adapters for literature-defined real-data benchmark truth."""

from .citeseq import (
    DEFAULT_HUMAN_ADT_RECEPTOR_ALIASES,
    build_citeseq_receptor_truth,
    label_citeseq_method_scores,
)
from .ipf import build_ipf_truth_universe, load_ipf_gold_standard

__all__ = [
    "DEFAULT_HUMAN_ADT_RECEPTOR_ALIASES",
    "build_citeseq_receptor_truth",
    "build_ipf_truth_universe",
    "label_citeseq_method_scores",
    "load_ipf_gold_standard",
]
