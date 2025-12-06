from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class DataConfig:
    """Configuration for loading and splitting the Iris dataset."""
    path: str = "data/iris.csv"
    target_column: str = "species"
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelConfig:
    """Configuration for the ML model."""
    model_type: Literal["log_reg"] = "log_reg"
    max_iter: int = 500
    C: float = 1.0


@dataclass
class TrainingConfig:
    """Configuration for training and evaluation."""
    metric: Literal["accuracy"] = "accuracy"


@dataclass
class AppConfig:
    """Top-level application configuration object."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig