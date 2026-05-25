"""Structured logging configuration for Stablecoin StressBench."""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with a structured format.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level; defaults to INFO.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
