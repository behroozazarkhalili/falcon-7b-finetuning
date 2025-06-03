"""Logging utilities for structured logging across the project."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
    use_rich: bool = True,
) -> None:
    """
    Setup structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        log_format: Custom log format string
        use_rich: Whether to use rich formatting for console output
    """
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set root logger level
    root_logger.setLevel(level)

    # Console handler
    if use_rich:
        console = Console(stderr=True)
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
        )
        console_handler.setLevel(level)
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)

        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)

        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class MLLogger:
    """Enhanced logger for ML experiments with structured logging."""

    def __init__(self, name: str, experiment_id: Optional[str] = None):
        """
        Initialize ML logger.

        Args:
            name: Logger name
            experiment_id: Optional experiment ID for tracking
        """
        self.logger = get_logger(name)
        self.experiment_id = experiment_id

    def log_experiment_start(self, config: dict) -> None:
        """Log experiment start with configuration."""
        self.logger.info(
            f"Starting experiment {self.experiment_id or 'unknown'} with config: {config}"
        )

    def log_experiment_end(self, metrics: dict) -> None:
        """Log experiment end with final metrics."""
        self.logger.info(
            f"Experiment {self.experiment_id or 'unknown'} completed with metrics: {metrics}"
        )

    def log_epoch(self, epoch: int, metrics: dict) -> None:
        """Log epoch metrics."""
        self.logger.info(f"Epoch {epoch}: {metrics}")

    def log_model_info(self, model_info: dict) -> None:
        """Log model information."""
        self.logger.info(f"Model info: {model_info}")

    def log_data_info(self, data_info: dict) -> None:
        """Log dataset information."""
        self.logger.info(f"Data info: {data_info}")

    def log_checkpoint(self, checkpoint_path: str, metrics: dict) -> None:
        """Log checkpoint save."""
        self.logger.info(
            f"Checkpoint saved to {checkpoint_path} with metrics: {metrics}"
        )

    def log_error(self, error: Exception, context: Optional[str] = None) -> None:
        """Log error with context."""
        context_str = f" in {context}" if context else ""
        self.logger.error(f"Error{context_str}: {str(error)}", exc_info=True)

    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def log_debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
