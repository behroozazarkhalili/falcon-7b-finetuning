"""Data loading utilities for handling datasets."""

from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from pydantic import BaseModel, ValidationError

from ..utils.logging import get_logger

logger = get_logger(__name__)


class DatasetSchema(BaseModel):
    """Schema validation for dataset configuration."""

    name: str
    split: str = "train"
    text_field: str = "text"
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    min_length: int = 10
    max_samples: Optional[int] = None


class DataLoader:
    """Enhanced data loader with validation and preprocessing capabilities."""

    def __init__(self, config: DictConfig):
        """
        Initialize data loader with configuration.

        Args:
            config: Dataset configuration
        """
        self.config = config
        self.dataset_config = DatasetSchema(**config.dataset)
        self.logger = logger

    def load_dataset(self) -> Dataset:
        """
        Load dataset from Hugging Face Hub or local files.

        Returns:
            Loaded dataset

        Raises:
            ValueError: If dataset cannot be loaded
        """
        try:
            self.logger.info(f"Loading dataset: {self.dataset_config.name}")

            # Load from Hugging Face Hub
            dataset = load_dataset(
                self.dataset_config.name, split=self.dataset_config.split
            )

            self.logger.info(f"Dataset loaded successfully. Size: {len(dataset)}")

            # Validate dataset
            self._validate_dataset(dataset)

            # Apply quality checks
            dataset = self._apply_quality_checks(dataset)

            # Limit samples if specified
            if self.dataset_config.max_samples:
                dataset = dataset.select(
                    range(min(self.dataset_config.max_samples, len(dataset)))
                )
                self.logger.info(f"Limited dataset to {len(dataset)} samples")

            return dataset

        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise ValueError(f"Dataset loading failed: {e}")

    def _validate_dataset(self, dataset: Dataset) -> None:
        """
        Validate dataset schema and required fields.

        Args:
            dataset: Dataset to validate

        Raises:
            ValueError: If validation fails
        """
        # Check required fields
        required_fields = self.config.get("schema", {}).get("required_fields", [])

        for field in required_fields:
            if field not in dataset.column_names:
                raise ValueError(f"Required field '{field}' not found in dataset")

        # Check text field exists
        if self.dataset_config.text_field not in dataset.column_names:
            raise ValueError(
                f"Text field '{self.dataset_config.text_field}' not found in dataset"
            )

        self.logger.info("Dataset validation passed")

    def _apply_quality_checks(self, dataset: Dataset) -> Dataset:
        """
        Apply data quality checks and filtering.

        Args:
            dataset: Input dataset

        Returns:
            Filtered dataset
        """
        quality_config = self.config.get("quality_checks", {})
        original_size = len(dataset)

        # Check for empty text
        if quality_config.get("check_empty_text", True):
            dataset = dataset.filter(
                lambda x: x[self.dataset_config.text_field]
                and len(x[self.dataset_config.text_field].strip()) > 0
            )
            self.logger.info(f"Filtered empty text: {original_size} -> {len(dataset)}")

        # Check text length
        min_length = quality_config.get(
            "min_text_length", self.dataset_config.min_length
        )
        max_length = quality_config.get("max_text_length", 2048)

        dataset = dataset.filter(
            lambda x: min_length <= len(x[self.dataset_config.text_field]) <= max_length
        )
        self.logger.info(
            f"Filtered by length ({min_length}-{max_length}): {len(dataset)} samples"
        )

        # Remove duplicates
        if quality_config.get("check_duplicates", True):
            original_size = len(dataset)
            # Convert to pandas for duplicate removal
            df = dataset.to_pandas()
            df = df.drop_duplicates(subset=[self.dataset_config.text_field])
            dataset = Dataset.from_pandas(df)
            self.logger.info(f"Removed duplicates: {original_size} -> {len(dataset)}")

        return dataset

    def get_dataset_info(self, dataset: Dataset) -> Dict[str, any]:
        """
        Get comprehensive dataset information.

        Args:
            dataset: Dataset to analyze

        Returns:
            Dictionary with dataset statistics
        """
        text_field = self.dataset_config.text_field
        texts = dataset[text_field]

        info = {
            "total_samples": len(dataset),
            "columns": dataset.column_names,
            "text_field": text_field,
            "avg_text_length": sum(len(text) for text in texts) / len(texts),
            "min_text_length": min(len(text) for text in texts),
            "max_text_length": max(len(text) for text in texts),
            "total_characters": sum(len(text) for text in texts),
        }

        return info

    def save_dataset(self, dataset: Dataset, output_path: Union[str, Path]) -> None:
        """
        Save dataset to disk.

        Args:
            dataset: Dataset to save
            output_path: Output directory path
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        dataset.save_to_disk(str(output_path))
        self.logger.info(f"Dataset saved to {output_path}")

    def load_local_dataset(self, data_path: Union[str, Path]) -> Dataset:
        """
        Load dataset from local files.

        Args:
            data_path: Path to local dataset

        Returns:
            Loaded dataset
        """
        data_path = Path(data_path)

        if data_path.is_dir():
            # Load from disk format
            dataset = Dataset.load_from_disk(str(data_path))
        elif data_path.suffix == ".json":
            # Load from JSON
            dataset = Dataset.from_json(str(data_path))
        elif data_path.suffix == ".csv":
            # Load from CSV
            df = pd.read_csv(data_path)
            dataset = Dataset.from_pandas(df)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        self.logger.info(f"Local dataset loaded from {data_path}")
        return dataset


def load_dataset_from_config(config: DictConfig) -> Dataset:
    """
    Convenience function to load dataset from configuration.

    Args:
        config: Dataset configuration

    Returns:
        Loaded dataset
    """
    loader = DataLoader(config)
    return loader.load_dataset()
