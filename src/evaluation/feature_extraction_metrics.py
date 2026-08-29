"""Reusable, method-agnostic feature-extraction metric utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def _validate_features_and_labels(
    features: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must be a non-empty two-dimensional matrix")
    if targets.ndim != 1 or targets.shape[0] != matrix.shape[0]:
        raise ValueError("labels must be one-dimensional and match the number of feature rows")
    if not np.isfinite(matrix).all():
        raise ValueError("features must contain only finite values")
    return matrix, targets


def calculate_fisher_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return one between/within-class Fisher ratio per feature dimension.

    This definition is representation-agnostic: it accepts any numeric feature
    matrix and its class labels, including the later Pipeline B features.
    """
    matrix, targets = _validate_features_and_labels(features, labels)
    classes = np.unique(targets)
    if classes.size < 2:
        raise ValueError("Fisher Score requires at least two classes")

    global_mean = matrix.mean(axis=0)
    between_class = np.zeros(matrix.shape[1], dtype=np.float64)
    within_class = np.zeros(matrix.shape[1], dtype=np.float64)
    for class_label in classes:
        class_features = matrix[targets == class_label]
        class_count = class_features.shape[0]
        class_mean = class_features.mean(axis=0)
        between_class += class_count * np.square(class_mean - global_mean)
        within_class += np.square(class_features - class_mean).sum(axis=0)

    valid_features = within_class > np.finfo(np.float64).eps
    scores = np.zeros(matrix.shape[1], dtype=np.float64)
    scores[valid_features] = between_class[valid_features] / within_class[valid_features]
    return scores


def calculate_fisher_score(features: np.ndarray, labels: np.ndarray) -> float:
    """Return the mean per-feature Fisher ratio for comparison consistency."""
    return float(np.mean(calculate_fisher_scores(features, labels)))


def calculate_fisher_summary(
    features: np.ndarray, labels: np.ndarray, top_k: int = 10
) -> dict[str, float]:
    """Summarize per-dimension Fisher ratios using mean, maximum, and top-k mean."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    scores = calculate_fisher_scores(features, labels)
    selected = np.partition(scores, -min(top_k, scores.size))[-min(top_k, scores.size):]
    return {
        "fisher_score": float(np.mean(scores)),
        "maximum_fisher_score": float(np.max(scores)),
        "top_k_mean_fisher_score": float(np.mean(selected)),
    }


def calculate_silhouette_score(features: np.ndarray, labels: np.ndarray) -> float:
    """Return the Euclidean silhouette score for a generic feature matrix."""
    try:
        from sklearn.metrics import silhouette_score
    except ImportError as error:
        raise ImportError(
            "Silhouette Score requires scikit-learn. Install the project dependencies "
            "with `py -m pip install -r requirements.txt`."
        ) from error
    matrix, targets = _validate_features_and_labels(features, labels)
    classes, class_counts = np.unique(targets, return_counts=True)
    if classes.size < 2 or matrix.shape[0] <= classes.size or np.any(class_counts < 2):
        raise ValueError("Silhouette Score requires two or more classes with at least two samples each")
    return float(silhouette_score(matrix, targets, metric="euclidean"))


def evaluate_feature_representation(
    features: np.ndarray, labels: np.ndarray, extraction_time_seconds: float
) -> dict[str, Any]:
    """Evaluate any feature representation with shared comparison metrics."""
    matrix, targets = _validate_features_and_labels(features, labels)
    if extraction_time_seconds < 0:
        raise ValueError("extraction_time_seconds must be non-negative")
    fisher_summary = calculate_fisher_summary(matrix, targets)
    return {
        **fisher_summary,
        "silhouette_score": calculate_silhouette_score(matrix, targets),
        "feature_extraction_time_ms_per_roi": extraction_time_seconds * 1000 / matrix.shape[0],
        "feature_vector_size": int(matrix.shape[1]),
        "samples_evaluated": int(matrix.shape[0]),
        "classes_evaluated": int(np.unique(targets).size),
    }


def evaluate_feature_representation_for_label_sets(
    features: np.ndarray,
    label_sets: dict[str, np.ndarray],
    extraction_time_seconds: float,
) -> dict[str, Any]:
    """Evaluate one feature matrix against multiple class-label definitions.

    The returned mean Fisher and Silhouette scores provide a balanced selection
    metric, while the individual label-set scores remain visible for reporting.
    This is reusable for any feature method and any named label sets.
    """
    if not label_sets:
        raise ValueError("label_sets must contain at least one named label array")
    shared_metrics: dict[str, Any] | None = None
    output: dict[str, Any] = {}
    fisher_scores: list[float] = []
    silhouette_scores: list[float] = []
    for label_name, labels in label_sets.items():
        if not label_name or not label_name.replace("_", "").isalnum():
            raise ValueError("label-set names must contain only letters, numbers, and underscores")
        metrics = evaluate_feature_representation(features, labels, extraction_time_seconds)
        shared_metrics = metrics if shared_metrics is None else shared_metrics
        output[f"{label_name}_fisher_score"] = metrics["fisher_score"]
        output[f"{label_name}_silhouette_score"] = metrics["silhouette_score"]
        fisher_scores.append(metrics["fisher_score"])
        silhouette_scores.append(metrics["silhouette_score"])

    assert shared_metrics is not None
    output.update({
        "fisher_score": float(np.mean(fisher_scores)),
        "silhouette_score": float(np.mean(silhouette_scores)),
        "feature_extraction_time_ms_per_roi": shared_metrics["feature_extraction_time_ms_per_roi"],
        "feature_vector_size": shared_metrics["feature_vector_size"],
        "samples_evaluated": shared_metrics["samples_evaluated"],
        "label_sets_evaluated": list(label_sets),
    })
    return output


def evaluate_feature_representation_by_group(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    extraction_time_seconds: float,
) -> dict[str, Any]:
    """Evaluate labels within each group and return macro averages.

    It is useful when one label's meaning differs across groups, such as fruit
    ripeness colours across fruit types. Groups without enough samples/classes
    for a valid Silhouette Score are omitted and reported as skipped.
    """
    matrix, targets = _validate_features_and_labels(features, labels)
    group_values = np.asarray(groups)
    if group_values.ndim != 1 or group_values.shape[0] != matrix.shape[0]:
        raise ValueError("groups must be one-dimensional and match the number of feature rows")

    per_group: dict[str, dict[str, Any]] = {}
    for group in np.unique(group_values):
        group_indices = group_values == group
        try:
            per_group[str(group)] = evaluate_feature_representation(
                matrix[group_indices], targets[group_indices], extraction_time_seconds
            )
        except ValueError as error:
            per_group[str(group)] = {"skipped": True, "reason": str(error)}

    valid = [metrics for metrics in per_group.values() if not metrics.get("skipped")]
    if not valid:
        raise ValueError("No groups contain enough labelled samples for grouped evaluation")
    return {
        "per_group": per_group,
        "macro_fisher_score": float(np.mean([metrics["fisher_score"] for metrics in valid])),
        "macro_maximum_fisher_score": float(np.mean([metrics["maximum_fisher_score"] for metrics in valid])),
        "macro_top_k_mean_fisher_score": float(np.mean([metrics["top_k_mean_fisher_score"] for metrics in valid])),
        "macro_silhouette_score": float(np.mean([metrics["silhouette_score"] for metrics in valid])),
        "groups_evaluated": len(valid),
    }


def select_best_feature_configuration(
    results: list[dict[str, Any]], fisher_relative_tolerance: float = 0.01
) -> dict[str, Any]:
    """Select a representation using quality first, then separation and cost.

    Configurations within one percent of the best Fisher Score are treated as
    practically similar; Silhouette Score, extraction time, and vector size
    then break that near-tie in the requested order.
    """
    if not results:
        raise ValueError("results must contain at least one configuration")
    if fisher_relative_tolerance < 0:
        raise ValueError("fisher_relative_tolerance must be non-negative")
    best_fisher = max(float(result["fisher_score"]) for result in results)
    threshold = best_fisher - abs(best_fisher) * fisher_relative_tolerance
    comparable = [result for result in results if float(result["fisher_score"]) >= threshold]
    return min(
        comparable,
        key=lambda result: (
            -float(result["silhouette_score"]),
            float(result["feature_extraction_time_ms_per_roi"]),
            int(result["feature_vector_size"]),
            -float(result["fisher_score"]),
        ),
    )
