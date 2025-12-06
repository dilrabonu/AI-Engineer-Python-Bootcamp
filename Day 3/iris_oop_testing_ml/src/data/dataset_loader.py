from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils.config import DataConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass 
class CSVDatasetLoader:
    """Load the Iris dataset from a CSV file and create train/test splits.

    This class is responsible only for:
    - reading the CSV
    - validating columns
    - encoding the target labels
    - splitting into train/test sets

    It does NOT do any scaling or modeling.
    """
    config: DataConfig
    _label_encoder: Optional[LabelEncoder] = None

    def _load_dataframe(self) -> pd.DataFrame:
        logger.info("Loading data from %s", self.config.path)
        df = pd.read_csv(self.config.path)

        if self.config.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.config.target_column}' "
                f"not found in columns: {list(df.columns)}"
            )

        logger.info("Loaded dataset with shape %s", df.shape)
        return df

    def load_full(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load full dataset and return (X, y).

        X: pandas DataFrame of features
        y: numpy array of encoded labels
        """
        df = self._load_dataframe()

        X = df.drop(columns=[self.config.target_column])
        y_raw = df[self.config.target_column]

        if self._label_encoder is None:
            self._label_encoder = LabelEncoder().fit(y_raw)
            logger.info("Fitted label encoder with classes: %s",
                         list(self._label_encoder.classes_))

        y = self._label_encoder.transform(y_raw)
        return X, y

    def load_train_test(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Return train/test split for features and label."""
        X, y = self.load_full()

        class_counts = pd.Series(y).value_counts()
        min_count = class_counts.min()

        if min_count < 2:
            logger.warning(
                "Skipping stratify: at least one class has only %d sample(s).",
                int(min_count),
            )
            stratify_arg = None
        else:
            stratify_arg = y

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=stratify_arg,
        )

        logger.info(
            "Split into train (%d samples) and test (%d samples)",
            len(X_train),
            len(X_test),
        )

        return X_train, X_test, y_train, y_test