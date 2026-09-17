"""Error Analysis Pipeline (shared evaluation core).

This package answers one question that an aggregate metric cannot:

    "where exactly is the model wrong?"

It turns a list of prediction-level records into a confusion matrix, an
operating point, a threshold sweep, slice diagnostics and a structured
evaluation report. Everything in here is pure: no database, no HTTP, no
model loading, no GPU. That is deliberate, because the same code must be
callable from an offline benchmark script, from a test fixture, and from the
server-side quality gate, and must produce byte-identical numbers in all
three.

Start with :mod:`app.evaluation.semantics` -- it defines what "positive"
means, and every other module in the package is written against it.
"""

from .metrics import (
    ConfusionMatrix,
    binary_metrics,
    confusion_matrix,
    latency_stats,
    operating_point,
    score_distributions,
)
from .records import (
    ERROR_TYPES,
    EvaluationError,
    PredictionRecord,
    partition_by_error_type,
)
from .report import (
    EVALUATION_SCHEMA_VERSION,
    build_evaluation_report,
    evaluation_fingerprint,
)
from .semantics import TASK_SEMANTICS, TaskSemantics, semantics_for
from .slices import slice_analysis
from .threshold import derive_threshold_grid, recommend_thresholds, sweep_thresholds

__all__ = [
    "ConfusionMatrix",
    "ERROR_TYPES",
    "EVALUATION_SCHEMA_VERSION",
    "EvaluationError",
    "PredictionRecord",
    "TASK_SEMANTICS",
    "TaskSemantics",
    "binary_metrics",
    "build_evaluation_report",
    "confusion_matrix",
    "derive_threshold_grid",
    "evaluation_fingerprint",
    "latency_stats",
    "operating_point",
    "partition_by_error_type",
    "recommend_thresholds",
    "score_distributions",
    "semantics_for",
    "slice_analysis",
    "sweep_thresholds",
]
