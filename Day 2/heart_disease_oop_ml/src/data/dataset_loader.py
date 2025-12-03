from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd 
from sklearn.model_selection import train_test_split

from src.utils.config import DataConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    def __init__(self, config: DataConfig) -> None:
        self.config = config

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load the full raw dataset as a pandas DataFrame."""
        raise NotImplementedError

    def load_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load and split dataset into train/test sets."""
        df = self.load()

        logger.info("Loaded dataset with shape %s", df.shape)

        target_col = self.config.target_column
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        logger.info(
            "Split intpo train (%d samples) and test (%d samples)",
            len(X_train),
            len(X_test),
        )

        return X_train, X_test, y_train, y_test

class HeartDatasetLoader(DatasetLoader):
    """Concrete dataset loader for the heart disease CSV."""

    def load(self) -> pd.DataFrame:
        logger.info("Loading heart disease data from %s", self.config.path)
        df = pd.read_csv(self.config.path)
        logger.info("Columns: %s", list(df.columns))
        return df