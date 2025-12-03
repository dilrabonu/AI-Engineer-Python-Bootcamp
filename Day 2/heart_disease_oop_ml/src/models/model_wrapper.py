from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.utils.config import ModelConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

@dataclass 
class ModelWrapper:
    """Thin wrapper around sklearn models with a unified interface."""
    config: ModelConfig
    model: Any = field(init=False)

    def __post_init__(self) -> None:
        if self.config.model_type == "log_reg":
            self.model = LogisticRegression(
                max_iter=self.config.max_iter,
                C=self.config.C,
                n_jobs=-1,
            )
        elif self.config.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=42,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

        logger.info("Initialized model: %s", self.model)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("Fitting model on %d samples", len(X))
        self.model.fit(X,y)

    def predict(self, X:np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability for the positive class (1)."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]

        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return self.predict(X)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray, metric: str = "roc_auc") -> Dict[str, float]:
        """Evaluate model with accuracy and optionally ROC-AUC.

        Args:
            X: Features for evaluation.
            y_true: Ground-truth labels.
            metric:  Which metric to compute("roc_auc" or "none").

        Returns:
            Dictionary with evaluation metrics.
        """
        y_pred = self.predict(X)
        results: Dict[str, float] = {
            "accuracy": accuracy_score(y_true, y_pred),
        }

        if metric == "roc_auc":
            y_score = self.predict_proba(X)
            try:
                results["roc_auc"] = roc_auc_score(y_true, y_score)
            except ValueError:
                #in case only one class present in y_true
                results["roc_auc"] = float("nan")

        logger.info("Evaluation metrics: %s", results)
        return results

            