#!/usr/bin/env python3
"""
Training script for Falcon-7B fine-tuning with QLoRA.

This script provides a complete training pipeline with configuration management,
logging, and experiment tracking.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from omegaconf import DictConfig

from src.data import DataPreprocessor, load_dataset_from_config
from src.models import create_falcon_model
from src.training import FalconTrainer
from src.utils import (
    get_logger,
    load_config,
    log_reproducibility_info,
    log_system_info,
    merge_configs,
    set_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Falcon-7B with QLoRA")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/default.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model/model.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data/data.yaml",
        help="Path to data configuration file",
    )
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--experiment-name", type=str, help="Override experiment name")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--wandb-project", type=str, help="Override wandb project name")
    parser.add_argument("--wandb-entity", type=str, help="Override wandb entity")
    parser.add_argument(
        "--resume-wandb", type=str, help="Resume wandb run (provide run ID or 'allow')"
    )

    return parser.parse_args()


def load_configurations(args: argparse.Namespace) -> DictConfig:
    """
    Load and merge all configuration files.

    Args:
        args: Command line arguments

    Returns:
        Merged configuration
    """
    # Load individual configs
    training_config = load_config(args.config)
    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config)

    # Merge configurations
    config = merge_configs(training_config, model_config, data_config)

    # Apply command line overrides
    if args.output_dir:
        config.training.output_dir = args.output_dir

    if args.experiment_name:
        config.experiment.name = args.experiment_name

    # Apply wandb overrides
    if args.wandb_project:
        if "wandb" not in config:
            config.wandb = {}
        config.wandb.project = args.wandb_project

    if args.wandb_entity:
        if "wandb" not in config:
            config.wandb = {}
        config.wandb.entity = args.wandb_entity

    return config


def setup_experiment(config: DictConfig, debug: bool = False) -> None:
    """
    Setup experiment environment.

    Args:
        config: Configuration object
        debug: Whether to enable debug mode
    """
    # Setup logging
    log_level = "DEBUG" if debug else "INFO"
    log_file = Path(config.training.get("logging_dir", "./logs")) / "training.log"

    setup_logging(level=log_level, log_file=log_file, use_rich=True)

    logger = get_logger(__name__)
    logger.info("Starting Falcon-7B fine-tuning experiment")

    # Set random seeds for reproducibility
    seed = config.training.get("seed", 42)
    set_seed(seed)

    # Log reproducibility information
    log_reproducibility_info(seed)

    # Log system information to wandb if enabled
    try:
        log_system_info()
    except Exception as e:
        logger.warning(f"Failed to log system info to wandb: {e}")

    # Log configuration
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Output directory: {config.training.output_dir}")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load configurations
    config = load_configurations(args)

    # Setup experiment
    setup_experiment(config, args.debug)

    logger = get_logger(__name__)
    trainer = None

    try:
        # Load and prepare data
        logger.info("Loading dataset...")
        dataset = load_dataset_from_config(config)

        # Create model
        logger.info("Creating model...")
        model = create_falcon_model(config)

        # Create trainer
        logger.info("Setting up trainer...")
        trainer = FalconTrainer(
            model=model.model, tokenizer=model.tokenizer, config=config
        )

        # Log wandb URL if available
        wandb_url = trainer.get_wandb_url()
        if wandb_url:
            logger.info(f"Wandb run URL: {wandb_url}")

        # Prepare data for training
        logger.info("Preparing data for training...")
        preprocessor = DataPreprocessor(model.tokenizer, config)

        # Split dataset
        dataset_splits = preprocessor.split_dataset(
            dataset, train_ratio=config.data.get("train_split", 0.9)
        )

        # Start training
        logger.info("Starting training...")
        trainer.train(
            train_dataset=dataset_splits["train"], eval_dataset=dataset_splits["eval"]
        )

        # Save final model
        logger.info("Saving final model...")
        final_model_path = Path(config.training.output_dir) / "final_model"
        model.save_model(final_model_path)

        logger.info("Training completed successfully!")

        # Log final wandb URL
        if wandb_url:
            logger.info(f"View results at: {wandb_url}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)

    finally:
        # Always finish wandb run
        if trainer:
            trainer.finish_wandb()


if __name__ == "__main__":
    main()
