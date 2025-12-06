from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class IrisPreprocessor:
    """Preprocess numeric Iris features using StandardScaler.

    Responsibilities:
    - fit scaler on training features
    - transform train/test featutres to standardized numpy arrays
    """
    scaler: StandardScaler = field(default_factory=StandardScaler)
    _is_fitted: bool = False

    def fit(self, X: pd.DataFrame) -> "IrisPreprocessor":
        logger.info("Fitting scaler on %d samples", len(X))
        self.scaler.fit(X.values)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform().")

        logger.info("Transforming %d samples", len(X))
        return self.scaler.transform(X.values)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.fit(X)
        return self.transform(X)