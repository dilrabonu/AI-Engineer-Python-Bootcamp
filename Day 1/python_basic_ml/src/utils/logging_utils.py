import logging
from typing import Optional 


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """ Create and return a configured logger instance.

    This function creates a logger with a given name, sets its log level
    to INFO, and attaches a console handler if it does not already exist.


    Arg:
        name (Optional[str]): Name of the logger. If None. the root logger
        is returned.


    Returns: 
        logging.Logger: Configured logger instance.
    """

    # Create or get an existing Logger with the given namse.
    logger = logging.getLogger(name)

    # Only configure the Logger if it has no handlers yet.
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create a handler that outputs Logs to the console (stdout)
        handler = logging.StreamHandler()

        # Define how Log message will look.
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s -  %(levelname)s - %(message)s"
        )

        # Attach the formatter to the handler.
        handler.setFormatter(formatter)

        # Attach the handler to the logger.
        logger.addHandler(handler)

    return logger
