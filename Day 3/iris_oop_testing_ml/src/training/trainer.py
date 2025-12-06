from __future__ import annotations

from dataclasses import dataclass

from src.data.dataset_loader import CSVDatasetLoader
from src.preprocessing.preprocessor import IrisPreprocessor
from src.models.model_wrapper import ModelWrapper
from src.utils.config import TrainingConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

@dataclass
class Trainer:
    """Orchestrates the end-to-end Iris ML training pipeline."""
    data_loader: CSVDatasetLoader
    preprocessor: IrisPreprocessor
    model: ModelWrapper 
    config: TrainingConfig

    def run(self) -> None:
        logger.info("starting training pipeline ...")

        # 1) Load data
        X_train, X_test, y_train, y_test = self.data_loader.load_train_test()

        # 2) Fit preprocessor on train, transform both
        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)

        # 3) Fit model
        self.model.fit(X_train_proc, y_train)

        # 4) Evaluate
        metrics = self.model.evaluate(X_test_proc, y_test)

        logger.info("Final evaluation metrics: %s", metrics)
        logger.info("Training pipeline finished.")