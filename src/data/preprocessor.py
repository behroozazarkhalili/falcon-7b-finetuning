"""Data preprocessing utilities for tokenization and data preparation."""

from typing import Dict, List, Optional, Union

from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizer

from ..utils.logging import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Data preprocessor for tokenization and formatting."""

    def __init__(self, tokenizer: PreTrainedTokenizer, config: DictConfig):
        """
        Initialize data preprocessor.

        Args:
            tokenizer: Tokenizer for text processing
            config: Configuration for preprocessing
        """
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger

        # Get preprocessing parameters
        self.max_length = config.dataset.get("max_length", 512)
        self.text_field = config.dataset.get("text_field", "text")
        self.truncation = config.dataset.get("truncation", True)
        self.padding = config.dataset.get("padding", "max_length")

    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Preprocess function for tokenizing text data.

        Args:
            examples: Batch of examples from dataset

        Returns:
            Tokenized examples
        """
        # Get texts
        texts = examples[self.text_field]

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=None,  # Return lists, not tensors
        )

        return tokenized

    def preprocess_dataset(self, dataset: Dataset, batch_size: int = 1000) -> Dataset:
        """
        Preprocess entire dataset with tokenization.

        Args:
            dataset: Input dataset
            batch_size: Batch size for processing

        Returns:
            Preprocessed dataset
        """
        self.logger.info(f"Preprocessing dataset with {len(dataset)} samples")

        # Apply preprocessing function
        processed_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=batch_size,
            remove_columns=dataset.column_names,  # Remove original columns
            desc="Tokenizing dataset",
        )

        self.logger.info(
            f"Dataset preprocessing completed. New columns: {processed_dataset.column_names}"
        )

        return processed_dataset

    def format_instruction_dataset(self, dataset: Dataset) -> Dataset:
        """
        Format dataset for instruction tuning.

        Args:
            dataset: Input dataset

        Returns:
            Formatted dataset
        """

        def format_function(examples):
            """Format examples for instruction tuning."""
            formatted_texts = []

            for text in examples[self.text_field]:
                # Add instruction formatting if needed
                # This can be customized based on your instruction format
                formatted_text = text
                formatted_texts.append(formatted_text)

            return {self.text_field: formatted_texts}

        formatted_dataset = dataset.map(
            format_function, batched=True, desc="Formatting for instruction tuning"
        )

        return formatted_dataset

    def split_dataset(
        self, dataset: Dataset, train_ratio: float = 0.9
    ) -> Dict[str, Dataset]:
        """
        Split dataset into train and validation sets.

        Args:
            dataset: Input dataset
            train_ratio: Ratio of data for training

        Returns:
            Dictionary with 'train' and 'eval' datasets
        """
        self.logger.info(f"Splitting dataset with train ratio: {train_ratio}")

        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)

        # Split dataset
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, total_size))

        self.logger.info(
            f"Dataset split - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}"
        )

        return {"train": train_dataset, "eval": eval_dataset}

    def get_preprocessing_stats(self, dataset: Dataset) -> Dict[str, any]:
        """
        Get statistics about preprocessed dataset.

        Args:
            dataset: Preprocessed dataset

        Returns:
            Dictionary with preprocessing statistics
        """
        # Get token lengths
        input_ids = dataset["input_ids"]
        token_lengths = [len(ids) for ids in input_ids]

        stats = {
            "total_samples": len(dataset),
            "avg_token_length": sum(token_lengths) / len(token_lengths),
            "min_token_length": min(token_lengths),
            "max_token_length": max(token_lengths),
            "vocab_size": self.tokenizer.vocab_size,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        return stats

    def validate_tokenization(self, dataset: Dataset) -> bool:
        """
        Validate tokenization results.

        Args:
            dataset: Tokenized dataset

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        required_columns = ["input_ids", "attention_mask"]

        # Check required columns
        for col in required_columns:
            if col not in dataset.column_names:
                raise ValueError(
                    f"Required column '{col}' not found after tokenization"
                )

        # Check token lengths
        input_ids = dataset["input_ids"]
        for i, ids in enumerate(input_ids[:10]):  # Check first 10 samples
            if len(ids) > self.max_length:
                raise ValueError(
                    f"Sample {i} exceeds max_length: {len(ids)} > {self.max_length}"
                )

        self.logger.info("Tokenization validation passed")
        return True


def create_tokenizer(model_name: str, config: DictConfig) -> PreTrainedTokenizer:
    """
    Create and configure tokenizer.

    Args:
        model_name: Name of the model for tokenizer
        config: Tokenizer configuration

    Returns:
        Configured tokenizer
    """
    logger.info(f"Creating tokenizer for model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=config.tokenizer.get("trust_remote_code", True)
    )

    # Set pad token if not exists
    pad_token_strategy = config.tokenizer.get("pad_token_strategy", "eos")

    if tokenizer.pad_token is None:
        if pad_token_strategy == "eos":
            tokenizer.pad_token = tokenizer.eos_token
        elif pad_token_strategy == "unk":
            tokenizer.pad_token = tokenizer.unk_token
        else:
            # Add new pad token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    logger.info(f"Tokenizer configured. Vocab size: {tokenizer.vocab_size}")
    logger.info(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    return tokenizer


def preprocess_dataset(
    dataset: Dataset, tokenizer: PreTrainedTokenizer, config: DictConfig
) -> Dataset:
    """
    Convenience function to preprocess dataset.

    Args:
        dataset: Input dataset
        tokenizer: Tokenizer for preprocessing
        config: Configuration

    Returns:
        Preprocessed dataset
    """
    preprocessor = DataPreprocessor(tokenizer, config)
    return preprocessor.preprocess_dataset(dataset)
