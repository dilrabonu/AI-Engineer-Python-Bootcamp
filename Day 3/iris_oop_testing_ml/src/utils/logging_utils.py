from __future__ import annotations

import logging
from typing import Optional

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Create and configure a module-level logger.

    Args:
        name: Logger name, usually __name__ of the module.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name if name else __name__)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger