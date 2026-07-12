"""Workflow orchestration, including partial train/apply leakage barriers."""

from .application import TrainingArtifactApplication, apply_training_artifacts
from .baseline import dry_run_baseline, fit_baseline
from .contracts import (
    BaselineArtifacts,
    BaselineAttributionRun,
    BaselineDryRunPlan,
    BaselineMode,
    BaselineScoreRun,
    EdgeEvidenceLedger,
    PlanStatus,
    RunStatus,
    StagePlan,
)
from .persistence import baseline_result_tables, write_baseline_result
from .training import FoldTrainingSpec, TrainingArtifacts, fit_training_artifacts

__all__ = [
    "BaselineArtifacts",
    "BaselineAttributionRun",
    "BaselineDryRunPlan",
    "BaselineMode",
    "BaselineScoreRun",
    "EdgeEvidenceLedger",
    "FoldTrainingSpec",
    "PlanStatus",
    "RunStatus",
    "StagePlan",
    "TrainingArtifactApplication",
    "TrainingArtifacts",
    "apply_training_artifacts",
    "baseline_result_tables",
    "dry_run_baseline",
    "fit_baseline",
    "fit_training_artifacts",
    "write_baseline_result",
]
