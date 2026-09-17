"""Task semantics: the single definition of "positive" in this repository.

The most damaging failure mode of an industrial evaluation pipeline is a
meaning that drifts between the metric code, the gate policy, the tests and
the dashboard. Four names in particular get mixed up constantly:

    False Positive  /  False Negative      (statistical notation)
    False Accept    /  False Reject        (manufacturing notation)

They are NOT synonyms, and this module exists so that nobody has to guess.

Field semantics
---------------
Both task families in this repository answer the same shop-floor question:
does this unit go to the customer, or to the reject bin?

    predicted positive -> the unit is REJECTED (held / scrapped / reworked)
    predicted negative -> the unit is ACCEPTED (released)

The positive class is therefore *the thing that must not be released*:

    anomaly_detection (PatchCore): positive = ANOMALY   (score = anomaly score)
    defect_detection  (YOLO):      positive = DEFECT    (score = detector confidence)

In both families a sample is predicted positive when ``score >= threshold``.
The threshold is a per-deployment constant, never a fitted quantity, so a
prediction is always reproducible from (score, threshold).

The resulting error identities
------------------------------
    False Accept  (FA) = a DEFECTIVE unit was ACCEPTED
                        = ground truth positive, predicted negative
                        = FALSE NEGATIVE in statistical notation
    False Reject  (FR) = a GOOD unit was REJECTED
                        = ground truth negative, predicted positive
                        = FALSE POSITIVE in statistical notation

Under this failure model:

    false_accept_rate (FAR) = fn / (tp + fn) = 1 - recall            miss rate
    false_reject_rate (FRR) = fp / (fp + tn) = false_positive_rate   alarm rate

Both identities are asserted by the test suite. A gate that raises a recall
floor and a gate that caps FAR are therefore the same constraint expressed in
two vocabularies; the policy declares both names so that a reviewer reading
either convention reaches the same verdict.

This is the consumer's-risk / producer's-risk pairing used in acceptance
sampling (a false accept is consumer's risk, a false reject is producer's
risk). It is a convention, and it is stated here once.
"""

from __future__ import annotations


class UnknownTaskError(ValueError):
    """Raised when a task family has no declared semantics.

    Failing loudly is the point: falling back to a guessed positive class is
    exactly how FP/FN silently swap places between two reports.
    """


class TaskSemantics:
    """Frozen description of one task family's label semantics."""

    __slots__ = (
        "task",
        "positive_label",
        "negative_label",
        "score_field",
        "positive_decision",
        "negative_decision",
        "score_direction",
    )

    def __init__(
        self,
        *,
        task: str,
        positive_label: str,
        negative_label: str,
        score_field: str,
        positive_decision: str = "REJECT",
        negative_decision: str = "ACCEPT",
        score_direction: str = "higher_is_more_positive",
    ) -> None:
        self.task = task
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.score_field = score_field
        self.positive_decision = positive_decision
        self.negative_decision = negative_decision
        self.score_direction = score_direction

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "positive_label": self.positive_label,
            "negative_label": self.negative_label,
            "score_field": self.score_field,
            "positive_decision": self.positive_decision,
            "negative_decision": self.negative_decision,
            "score_direction": self.score_direction,
        }

    def is_positive(self, ground_truth: int) -> bool:
        return int(ground_truth) == 1

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TaskSemantics(task={self.task!r}, positive={self.positive_label!r})"


ANOMALY_DETECTION = TaskSemantics(
    task="anomaly_detection",
    positive_label="anomaly",
    negative_label="normal",
    score_field="anomaly_score",
)

DEFECT_DETECTION = TaskSemantics(
    task="defect_detection",
    positive_label="defect",
    negative_label="defect_free",
    score_field="confidence",
)

TASK_SEMANTICS: dict[str, TaskSemantics] = {
    ANOMALY_DETECTION.task: ANOMALY_DETECTION,
    DEFECT_DETECTION.task: DEFECT_DETECTION,
}

# Registry / manifest model types are deployment vocabulary; the evaluation
# core speaks task vocabulary. The mapping is explicit rather than a fallback.
TASK_ALIASES: dict[str, str] = {
    "patchcore": "anomaly_detection",
    "anomaly": "anomaly_detection",
    "anomaly_detection": "anomaly_detection",
    "yolo": "defect_detection",
    "defect": "defect_detection",
    "defect_detection": "defect_detection",
}


def semantics_for(task: str) -> TaskSemantics:
    """Resolve a task or model_type name to its declared semantics."""
    key = str(task or "").strip().lower()
    resolved = TASK_ALIASES.get(key)
    if resolved is None:
        raise UnknownTaskError(
            f"no declared label semantics for task {task!r}; "
            f"known: {sorted(TASK_SEMANTICS)}"
        )
    return TASK_SEMANTICS[resolved]


def decision_for(prediction: int, semantics: TaskSemantics) -> str:
    """Translate a binary prediction into the field-layer decision."""
    return semantics.positive_decision if int(prediction) == 1 else semantics.negative_decision


__all__ = [
    "ANOMALY_DETECTION",
    "DEFECT_DETECTION",
    "TASK_ALIASES",
    "TASK_SEMANTICS",
    "TaskSemantics",
    "UnknownTaskError",
    "decision_for",
    "semantics_for",
]
