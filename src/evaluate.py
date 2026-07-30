"""Classification evaluation helpers for a heavily imbalanced problem."""

from __future__ import annotations

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def fraud_metrics(y_true, y_pred, y_scores) -> dict[str, float]:
    """Return metrics that stay meaningful when 99.8% of rows are one class.

    Accuracy is deliberately excluded: a model that predicts "never fraud"
    scores 99.8% accuracy while catching zero fraud. Precision, recall, and the
    threshold-free ROC-AUC / PR-AUC (average precision) tell the real story.
    """
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_scores),
        "PR_AUC": average_precision_score(y_true, y_scores),
    }
