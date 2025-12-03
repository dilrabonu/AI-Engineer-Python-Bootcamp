import logging 
from typing import Optional

def get_logger(name: str = "heart_disease_oop_ml",
               level: int = logging.INFO) -> logging.Logger:
    """Create and configure an application-level logger.
    
    Args:
        name: Logger name (usually project or module name).
        level: Minimum log level to handle.
        
        
    Returns:
        Configured 'logging.Logger' instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Logger already configured - avoid duplicate handlers in notebooks/ reLoads.
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    #Example log: 2025-12-02 21:00:00, 123 - heart_disease_oop_ml - INFO - Message
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.propagate = False
    return logger

