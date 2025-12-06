from __future__ import annotations

import pandas as pd 
import numpy as np
import pytest 

from src.utils.config import DataConfig, ModelConfig, TrainingConfig
from src.models.model_wrapper import ModelWrapper


@pytest.fixture
def iris_small_df() -> pd.DataFrame:
    """Small in-memory Iris-like dataframe for fast tests. """
    return pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 6.2, 5.9],
            "sepal_width": [3.5, 3.0, 2.8, 3.0],
            "petal_length": [1.4, 1.4, 4.8, 5.1],
            "petal_width": [0.2, 0.2, 1.8, 1.8],
            "species": ["setosa", "setosa", "versicolor", "virginica"],
        }
    )

@pytest.fixture
def data_config(tmp_path, iris_small_df: pd.DataFrame) -> DataConfig:
    """ Write the small DF to a temporary CSV and return DataConfig for it. """
    csv_path = tmp_path / "iris_test.csv"
    iris_small_df.to_csv(csv_path, index=False)
    return DataConfig(
        path=str(csv_path),
        target_column="species",
        test_size=0.2,
        random_state=42,
    )

@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(max_iter=200, C=1.0)

@pytest.fixture
def training_config() -> TrainingConfig:
    return TrainingConfig()

@pytest.fixture
def model_wrapper(model_config: ModelConfig) -> ModelWrapper:
    return ModelWrapper(model_config)