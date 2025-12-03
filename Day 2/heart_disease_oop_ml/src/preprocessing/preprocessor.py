from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np 
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class TabularPreprocessor:
    """Preprocess numeric tabular features : impute + scale.
    
    This class is intentionally simple, but structured like a
     real-world sklearn-style transformer.
     """

    numeric_cols: List[str] = field(default_factory=list)
    imputer: SimpleImputer = field(
        default_factory=lambda: SimpleImputer(strategy="median")
    ) 
    scaler: StandardScaler = field(default_factory=StandardScaler)
    fitted: bool = False


    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        self.numeric_cols_ = list(X.select_dtypes(include=["number"]).columns)
        if not self.numeric_cols_:
            raise ValueError("No numeric columns found for preprocessing.")

        numeric_data = X[self.numeric_cols_].values
        numeric_data = self.imputer.fit_transform(numeric_data)
        self.scaler.fit(numeric_data)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        numeric_data = X[self.numeric_cols_].values
        numeric_data = self.imputer.transform(numeric_data)
        scaled = self.scaler.transform(numeric_data)
        return scaled


    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)