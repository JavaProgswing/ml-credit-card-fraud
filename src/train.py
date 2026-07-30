"""Train and evaluate the class-weighted logistic-regression fraud baseline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.evaluate import fraud_metrics
from src.preprocess import (
    PROJECT_ROOT,
    build_preprocessor,
    load_clean_data,
    split_features_target,
)

FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "precision_recall_curve.png"


def build_model() -> Pipeline:
    """Create an unfitted preprocessing and logistic-regression pipeline.

    ``class_weight="balanced"`` re-weights the rare fraud class so the model
    does not simply learn to always predict the majority (legitimate) class.
    """
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )


def save_precision_recall_curve(
    y_true, y_scores, output_path: Path = FIGURE_PATH
) -> None:
    """Save the precision-recall curve as a portfolio figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PrecisionRecallDisplay.from_predictions(y_true, y_scores)
    plt.title("Credit-Card Fraud: Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Run the complete, reproducible baseline experiment."""
    data, duplicate_count = load_clean_data()
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)
    test_scores = model.predict_proba(X_test)[:, 1]

    metrics = pd.DataFrame(
        [fraud_metrics(y_test, test_predictions, test_scores)],
        index=["Test"],
    )

    save_precision_recall_curve(y_test, test_scores)

    fraud_share = y.mean()
    print(f"Rows after cleaning: {len(data):,}")
    print(f"Duplicate rows removed: {duplicate_count:,}")
    print(f"Fraud share of transactions: {fraud_share:.3%}")
    print(f"Fraud cases in test set: {int(y_test.sum())} of {len(y_test):,}")
    print("\nClass-weighted logistic regression metrics (test)")
    print(metrics.round(3).to_string())
    print(f"\nPrecision-recall curve saved to: {FIGURE_PATH}")
