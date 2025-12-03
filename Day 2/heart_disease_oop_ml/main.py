from src.utils.config import AppConfig
from src.utils.logging_utils import get_logger
from src.data.dataset_loader import HeartDatasetLoader
from src.preprocessing.preprocessor import TabularPreprocessor
from src.models.model_wrapper import ModelWrapper
from src.training.trainer import Trainer

logger = get_logger(__name__)

def run() -> None:
    """Application entry point for the Day 2 OOP ML Pipeline."""
    cfg = AppConfig()

    logger.info("App configuration: %s", cfg)

    # Build components using composition

    data_loader = HeartDatasetLoader(cfg.data)
    preprocessor = TabularPreprocessor()
    model = ModelWrapper(cfg.model)
    trainer = Trainer(
        data_loader=data_loader,
        preprocessor=preprocessor,
        model=model,
        config=cfg.training,
    )

    #Run full training pipeline
    trainer.run()
    

if __name__ == "__main__":
    run()
