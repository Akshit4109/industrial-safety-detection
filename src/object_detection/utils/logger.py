"""Logger configuration utility."""
import logging
import sys
from pathlib import Path


def setup_logger(name: str = "object_detection", log_file: Path = None, level: int = logging.INFO) -> logging.Logger:
    """Sets up a structured logger.

    Args:
        name: Logger name.
        log_file: Optional path to save log output.
        level: Logging level.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
