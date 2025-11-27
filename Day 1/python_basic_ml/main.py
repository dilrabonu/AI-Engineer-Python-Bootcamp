from src.utils import load_config, get_logger
from src.data.fake_data import generate_sample
from src.models.fake_model import predict

def run() -> None:
    """ Run the demo ML pipeline using logger, config, and fakle model."""
    

    # Load configuration file.
    config = load_config("config.yaml")

    # Create a logger for this application.
    logger = get_logger(name=config.get("app_name", "python_basic_ml"))
    
    logger.info("Application started")
    logger.info("Environment: %s", config.get("environment"))

    # STEP 1 - generate fake data
    image = generate_sample()
    logger.info("Generate fake image with shape:n %s", image.shape)

    # STEP 2 make fake prediction
    output = predict(image)
    logger.info("Prediction result: %s", output)

    logger.info("Pipeline finished successfully.")

if __name__ == "__main__":
    run()