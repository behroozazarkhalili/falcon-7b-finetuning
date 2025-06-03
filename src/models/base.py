"""Base model class with common functionality."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..utils.logging import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, config: DictConfig):
        """
        Initialize base model.

        Args:
            config: Model configuration
        """
        self.config = config
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.logger = logger

    @abstractmethod
    def load_model(self) -> PreTrainedModel:
        """
        Load the model.

        Returns:
            Loaded model
        """
        pass

    @abstractmethod
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load the tokenizer.

        Returns:
            Loaded tokenizer
        """
        pass

    def get_model_info(self) -> Dict[str, any]:
        """
        Get model information.

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        info = {
            "model_name": self.config.model.name,
            "model_type": type(self.model).__name__,
            "num_parameters": self.get_num_parameters(),
            "trainable_parameters": self.get_trainable_parameters(),
            "device": next(self.model.parameters()).device.type,
            "dtype": next(self.model.parameters()).dtype,
        }

        return info

    def get_num_parameters(self) -> int:
        """
        Get total number of parameters.

        Returns:
            Total number of parameters
        """
        if self.model is None:
            return 0

        return sum(p.numel() for p in self.model.parameters())

    def get_trainable_parameters(self) -> int:
        """
        Get number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        if self.model is None:
            return 0

        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage information.

        Returns:
            Dictionary with memory usage in GB
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        memory_info = {
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "cached": torch.cuda.memory_reserved() / 1e9,
            "max_allocated": torch.cuda.max_memory_allocated() / 1e9,
            "max_cached": torch.cuda.max_memory_reserved() / 1e9,
        }

        return memory_info

    def save_model(self, output_path: Union[str, Path]) -> None:
        """
        Save model and tokenizer.

        Args:
            output_path: Path to save the model
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(output_path)

        # Save tokenizer
        self.tokenizer.save_pretrained(output_path)

        self.logger.info(f"Model and tokenizer saved to {output_path}")

    def load_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.logger.info(f"Loading model from checkpoint: {checkpoint_path}")

        # This will be implemented by subclasses
        raise NotImplementedError("Subclasses must implement load_from_checkpoint")

    def validate_model(self) -> bool:
        """
        Validate model configuration and state.

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")

        # Check if model is on correct device
        if torch.cuda.is_available():
            model_device = next(self.model.parameters()).device
            if model_device.type != "cuda":
                self.logger.warning(
                    f"Model is on {model_device}, but CUDA is available"
                )

        self.logger.info("Model validation passed")
        return True

    def print_model_summary(self) -> None:
        """Print a summary of the model."""
        if self.model is None:
            self.logger.error("Model not loaded")
            return

        info = self.get_model_info()
        memory = self.get_memory_usage()

        self.logger.info("=== Model Summary ===")
        self.logger.info(f"Model: {info['model_name']}")
        self.logger.info(f"Type: {info['model_type']}")
        self.logger.info(f"Total parameters: {info['num_parameters']:,}")
        self.logger.info(f"Trainable parameters: {info['trainable_parameters']:,}")
        self.logger.info(f"Device: {info['device']}")
        self.logger.info(f"Data type: {info['dtype']}")

        if "error" not in memory:
            self.logger.info(f"GPU memory allocated: {memory['allocated']:.2f} GB")
            self.logger.info(f"GPU memory cached: {memory['cached']:.2f} GB")

        self.logger.info("====================")

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to save memory."""
        if self.model is None:
            raise ValueError("Model not loaded")

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled")
        else:
            self.logger.warning("Model does not support gradient checkpointing")

    def disable_cache(self) -> None:
        """Disable model cache for training."""
        if self.model is None:
            raise ValueError("Model not loaded")

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
            self.logger.info("Model cache disabled")
        else:
            self.logger.warning("Model does not have cache configuration")
