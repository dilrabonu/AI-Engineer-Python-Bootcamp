from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.utils.config import ModelConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

@dataclass
class ModelWrapper:
    """Thin wrapper around sklearn models with a unified interface."""
    config: ModelConfig
    model: Any = field(init=False)

    def __post_init__(self) -> None:
        self.model = LogisticRegression(
            max_iter=self.config.max_iter,
            C=self.config.C,
            n_jobs=-1,
            multi_class="auto",
        )
        logger.info("Initialized model: %s", self.model)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("Fitting model on %d samples", len(X))
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate model on given data and return metrics dict. """
        y_pred = self.predict(X)
        acc = accuracy_score(y_true, y_pred)

        metrics: Dict[str, float] = {"accuracy": float(acc)}
        logger.info("Evaluation metrics: %s", metrics)
        return metrics
