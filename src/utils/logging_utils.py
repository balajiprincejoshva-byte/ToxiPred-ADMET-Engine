"""
ToxiPred — Logging Utilities

Provides a consistent, configurable logging setup across all modules.
Uses Python's built-in logging library for structured output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    fmt: str = "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
) -> logging.Logger:
    """
    Create and configure a logger with consistent formatting.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (default: INFO).
        log_file: Optional path to a log file for persistent logging.
        fmt: Log message format string.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default settings.

    This is the preferred entry point for modules that just need a logger.

    Args:
        name: Logger name (use __name__).

    Returns:
        Configured logging.Logger instance.
    """
    return setup_logger(name)
