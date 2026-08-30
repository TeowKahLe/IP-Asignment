"""K-Nearest Neighbors (KNN) fruit-type classification for Pipeline B."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np


KNN_PARAMETER_GRID = {
    "n_neighbors": (3, 5, 7),
}

KNN_METRIC = "euclidean"
KNN_WEIGHTS = "uniform"


def train_knn(features: np.ndarray, labels: np.ndarray, parameters: dict[str, Any]):
    """Train a KNN fruit-type classifier and return timing."""
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(
        n_neighbors=parameters["n_neighbors"],
        metric=KNN_METRIC,
        weights=KNN_WEIGHTS
    )
    started_at = perf_counter(); model.fit(features, labels)
    return model, perf_counter() - started_at


def predict_knn(model: Any, features: np.ndarray) -> tuple[np.ndarray, float]:
    started_at = perf_counter(); return model.predict(features), perf_counter() - started_at