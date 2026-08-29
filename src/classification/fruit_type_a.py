"""LightGBM fruit-type classification for Pipeline A."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np


LIGHTGBM_PARAMETER_GRID = {
    "n_estimators": (100, 200, 300), "learning_rate": (0.03, 0.05, 0.10),
    "num_leaves": (15, 31, 63), "max_depth": (5, 10, -1),
}


def train_lightgbm(features: np.ndarray, labels: np.ndarray, parameters: dict[str, Any]):
    """Train a deterministic LightGBM multiclass classifier and return timing."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:
        raise ImportError("LightGBM is required. Install the project requirements.") from error
    model = LGBMClassifier(**parameters, random_state=42, n_jobs=-1, verbosity=-1)
    started_at = perf_counter(); model.fit(features, labels)
    return model, perf_counter() - started_at


def predict_lightgbm(model: Any, features: np.ndarray) -> tuple[np.ndarray, float]:
    started_at = perf_counter(); return model.predict(features), perf_counter() - started_at
