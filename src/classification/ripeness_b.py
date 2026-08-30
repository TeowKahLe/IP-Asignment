"""Decision Tree ripeness-level classification for Pipeline B."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np


DECISION_TREE_PARAMETERS = {
    "criterion": "gini", "max_depth": 5,
    "min_samples_split": 2, "min_samples_leaf": 1,
}


def train_decision_tree(features: np.ndarray, labels: np.ndarray):
    """Train a deterministic Decision Tree multiclass classifier and return timing."""
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(**DECISION_TREE_PARAMETERS, random_state=42)
    started_at = perf_counter(); model.fit(features, labels)
    return model, perf_counter() - started_at


def predict_decision_tree(model: Any, features: np.ndarray) -> tuple[np.ndarray, float]:
    started_at = perf_counter(); return model.predict(features), perf_counter() - started_at