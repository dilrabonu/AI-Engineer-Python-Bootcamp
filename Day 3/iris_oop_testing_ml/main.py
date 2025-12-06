from __future__ import annotations

from src.data.dataset_loader import CSVDatasetLoader
from src.preprocessing.preprocessor  import IrisPreprocessor
from src.models.model_wrapper import ModelWrapper 
from src.training.trainer import Trainer 
from src.utils.config import AppConfig, DataConfig, ModelConfig, TrainingConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def build_config() -> AppConfig:
    """Create the full application configuration."""
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    training_cfg = TrainingConfig()
    return AppConfig(data=data_cfg, model=model_cfg, training=training_cfg)

def run() -> None:
    """Main entry point for the Iris ML pipeline."""
    config = build_config()
    logger.info("App configuration: %s", config)

    # 1) Data Loader
    data_loader = CSVDatasetLoader(config.data)

    # 2) Preprocessor
    preprocessor = IrisPreprocessor()

    # 3) Model wrapper
    model = ModelWrapper(config.model)

    # 4) Trainer
    trainer = Trainer(
        data_loader=data_loader,
        preprocessor=preprocessor,
        model=model,
        config=config.training,
    )

    # 5) Run full pipeline
    trainer.run()


if __name__ == "__main__":
    run()