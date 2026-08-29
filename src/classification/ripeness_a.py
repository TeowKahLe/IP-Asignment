"""CatBoost ripeness-level classification for Pipeline A."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np


CATBOOST_PARAMETER_GRID = {
    "iterations": (100, 200, 300), "learning_rate": (0.03, 0.05, 0.10),
    "depth": (4, 6, 8), "l2_leaf_reg": (1, 3, 5),
}


def train_catboost(features: np.ndarray, labels: np.ndarray, parameters: dict[str, Any]):
    """Train a deterministic CatBoost multiclass classifier and return timing."""
    try:
        from catboost import CatBoostClassifier
    except ImportError as error:
        raise ImportError("CatBoost is required. Install the project requirements.") from error
    model = CatBoostClassifier(**parameters, random_seed=42, verbose=False, allow_writing_files=False)
    started_at = perf_counter(); model.fit(features, labels)
    return model, perf_counter() - started_at


def predict_catboost(model: Any, features: np.ndarray) -> tuple[np.ndarray, float]:
    started_at = perf_counter(); return model.predict(features).reshape(-1), perf_counter() - started_at
