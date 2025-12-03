from dataclasses import dataclass
from typing import Literal 

@dataclass
class DataConfig:
    """Configuration for loading and splitting the heart disease dataset. """
    path: str = "data/heart_cleveland_upload.csv"
    target_column: str = "condition"
    test_size: float = 0.2
    random_state: int = 42

@dataclass
class ModelConfig:
    """Configuration for the ML model. """
    model_type: Literal["log_reg", "random_forest"] = "log_reg"
    max_iter: int = 1000         # for Logistic regression
    C: float = 1.0               # regularization strength
    n_estimators: int = 200      # for random forest
    max_depth: int | None = None # for random forest

@dataclass
class TrainingConfig:
    """Configuration for training and evaluation."""
    metric: Literal["accuracy", "roc_auc"] = "roc_auc"
    verbose: bool = True 

@dataclass 
class AppConfig:
    """Top-level configuration holdin all sub-configs."""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig
    training: TrainingConfig = TrainingConfig
