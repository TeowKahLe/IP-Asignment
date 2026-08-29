"""Reusable metrics and selection rules for fruit classification models."""

from __future__ import annotations

from typing import Any

import numpy as np


def evaluate_classification(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    training_time_seconds: float,
    prediction_time_seconds: float,
) -> dict[str, Any]:
    """Calculate the shared classification metrics for any classifier/task."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    actual = np.asarray(true_labels)
    predicted = np.asarray(predicted_labels)
    if actual.ndim != 1 or predicted.ndim != 1 or len(actual) != len(predicted) or not len(actual):
        raise ValueError("true_labels and predicted_labels must be matching non-empty vectors")
    if training_time_seconds < 0 or prediction_time_seconds < 0:
        raise ValueError("timings must be non-negative")
    labels = np.unique(np.concatenate([actual, predicted]))
    precision, recall, f1_score, support = precision_recall_fscore_support(
        actual, predicted, labels=labels, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        actual, predicted, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1_score": float(macro_f1),
        "training_time_seconds": float(training_time_seconds),
        "prediction_time_ms_per_roi": float(prediction_time_seconds * 1000 / len(actual)),
        "samples_evaluated": int(len(actual)),
        "classes_evaluated": int(len(labels)),
        "class_labels": [str(label) for label in labels],
        "per_class_precision": {str(label): float(value) for label, value in zip(labels, precision)},
        "per_class_recall": {str(label): float(value) for label, value in zip(labels, recall)},
        "per_class_f1_score": {str(label): float(value) for label, value in zip(labels, f1_score)},
        "per_class_support": {str(label): int(value) for label, value in zip(labels, support)},
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
    }


def select_best_classification_configuration(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Select by macro F1, then accuracy, then lower training/prediction time."""
    if not results:
        raise ValueError("results must contain at least one configuration")
    return min(
        results,
        key=lambda result: (
            -float(result["macro_f1_score"]),
            -float(result["accuracy"]),
            float(result["training_time_seconds"]),
            float(result["prediction_time_ms_per_roi"]),
        ),
    )


def plot_confusion_matrix(metrics: dict[str, Any], title: str):
    """Display a labelled confusion-matrix diagram from shared metric output."""
    import matplotlib.pyplot as plt

    labels = metrics["class_labels"]
    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) * 0.8)))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="ROI count")
    axis.set(
        title=title, xlabel="Predicted label", ylabel="True label",
        xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
        xticklabels=labels, yticklabels=labels,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row, column in np.ndindex(matrix.shape):
        axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                  color="white" if matrix[row, column] > threshold else "black")
    figure.tight_layout()
    return figure, axis
