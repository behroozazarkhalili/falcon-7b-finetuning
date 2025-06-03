"""Falcon trainer implementation using TRL's SFTTrainer."""

from pathlib import Path
from typing import Dict, Optional, List, Union

from datasets import Dataset
from omegaconf import DictConfig, ListConfig
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig

from ..utils.logging import get_logger
from ..utils.wandb_utils import WandbManager, init_wandb_from_config

logger = get_logger(__name__)


class FalconTrainer:
    """Falcon trainer using TRL's SFTTrainer with configuration management."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig
    ):
        """
        Initialize Falcon trainer.
        
        Args:
            model: Pre-trained model
            tokenizer: Model tokenizer
            config: Training configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.sft_config = self._create_sft_config()
        self.wandb_manager = None
        
        logger.info(f"Created SFT config with output_dir: {self.sft_config.output_dir}")
        
        # Initialize wandb if enabled
        self._init_wandb()
        
        # Disable cache for training
        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = False
    
    def _init_wandb(self) -> None:
        """Initialize wandb if enabled in configuration."""
        try:
            self.wandb_manager = init_wandb_from_config(self.config)
            if self.wandb_manager:
                # Log model and system info
                self.wandb_manager.log_model_info(self.model, self.tokenizer)
                
                # Watch model for gradient/parameter tracking
                self.wandb_manager.watch_model(self.model)
                
                logger.info(f"Wandb initialized. Run URL: {self.wandb_manager.get_run_url()}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self.wandb_manager = None
    
    def _convert_omegaconf_to_native(self, obj):
        """Convert OmegaConf objects to native Python types."""
        if isinstance(obj, DictConfig):
            return {k: self._convert_omegaconf_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, ListConfig):
            return [self._convert_omegaconf_to_native(item) for item in obj]
        else:
            return obj
    
    def _create_sft_config(self) -> SFTConfig:
        """Create SFT config from training configuration."""
        training_config = self.config.training
        
        # Convert report_to to list if it's a string
        report_to = training_config.report_to
        if isinstance(report_to, str):
            report_to = [report_to]
        elif hasattr(report_to, '__iter__'):
            report_to = list(report_to)
        
        return SFTConfig(
            output_dir=str(training_config.output_dir),
            per_device_train_batch_size=int(training_config.per_device_train_batch_size),
            per_device_eval_batch_size=int(training_config.per_device_eval_batch_size),
            gradient_accumulation_steps=int(training_config.gradient_accumulation_steps),
            optim=str(training_config.optim),
            save_steps=int(training_config.save_steps),
            logging_steps=int(training_config.logging_steps),
            learning_rate=float(training_config.learning_rate),
            fp16=bool(training_config.fp16),
            max_grad_norm=float(training_config.max_grad_norm),
            num_train_epochs=int(training_config.num_train_epochs),
            warmup_ratio=float(training_config.warmup_ratio),
            lr_scheduler_type=str(training_config.lr_scheduler_type),
            save_total_limit=int(training_config.save_total_limit),
            load_best_model_at_end=bool(training_config.load_best_model_at_end),
            metric_for_best_model=str(training_config.metric_for_best_model),
            greater_is_better=bool(training_config.greater_is_better),
            eval_strategy=str(training_config.eval_strategy),
            eval_steps=int(training_config.eval_steps),
            report_to=report_to,
            logging_dir=str(training_config.logging_dir),
            seed=int(training_config.seed),
            data_seed=int(training_config.data_seed),
            packing=bool(training_config.get("packing", True)),
            max_length=int(training_config.get("max_seq_length", 512)),
        )
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            
        Returns:
            Training metrics
        """
        logger.info("Setting up SFTTrainer...")
        
        # Log dataset info to wandb
        if self.wandb_manager:
            dataset_info = {
                "train_size": len(train_dataset),
                "eval_size": len(eval_dataset) if eval_dataset else 0,
                "total_size": len(train_dataset) + (len(eval_dataset) if eval_dataset else 0),
                "columns": train_dataset.column_names,
            }
            self.wandb_manager.log_dataset_info(dataset_info)
        
        # Create SFTTrainer
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=self.sft_config,
        )
        
        # Start training
        logger.info("Starting model training...")
        train_result = self.trainer.train()
        
        # Save the model
        logger.info(f"Saving model to {self.sft_config.output_dir}")
        self.trainer.save_model()
        
        # Log final model as wandb artifact
        if self.wandb_manager:
            final_model_path = Path(self.sft_config.output_dir)
            self.wandb_manager.log_final_model(final_model_path)
        
        # Log training metrics
        metrics = train_result.metrics
        logger.info(f"Training completed. Final metrics: {metrics}")
        
        # Log final metrics to wandb
        if self.wandb_manager:
            self.wandb_manager.log_metrics({"final_" + k: v for k, v in metrics.items()})
        
        return metrics
    
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict:
        """
        Evaluate the model.
        
        Args:
            eval_dataset: Evaluation dataset (optional)
            
        Returns:
            Evaluation metrics
        """
        if not hasattr(self, 'trainer') or self.trainer is None:
            raise ValueError("Trainer not initialized. Call train() first.")
        
        logger.info("Evaluating model...")
        
        if eval_dataset is not None:
            # Update eval dataset if provided
            self.trainer.eval_dataset = eval_dataset
        
        eval_result = self.trainer.evaluate()
        
        # Log evaluation metrics to wandb
        if self.wandb_manager:
            self.wandb_manager.log_metrics({"eval_" + k: v for k, v in eval_result.items()})
        
        logger.info(f"Evaluation completed. Metrics: {eval_result}")
        return eval_result
    
    def save_model(self, output_path: Optional[str] = None) -> None:
        """
        Save the trained model.
        
        Args:
            output_path: Optional path to save the model
        """
        if not hasattr(self, 'trainer') or self.trainer is None:
            raise ValueError("Trainer not initialized. Call train() first.")
        
        if output_path:
            self.trainer.save_model(output_path)
            logger.info(f"Model saved to {output_path}")
            
            # Log model as wandb artifact
            if self.wandb_manager:
                self.wandb_manager.log_artifact(
                    artifact_path=output_path,
                    artifact_name="saved-model",
                    artifact_type="model",
                    description="Manually saved model"
                )
        else:
            self.trainer.save_model()
            logger.info("Model saved to default location")
    
    def get_training_state(self) -> Dict[str, any]:
        """
        Get current training state information.
        
        Returns:
            Dictionary with training state
        """
        if self.trainer is None:
            return {"error": "Trainer not initialized"}
        
        state = {
            "global_step": self.trainer.state.global_step,
            "epoch": self.trainer.state.epoch,
            "max_steps": self.trainer.state.max_steps,
            "num_train_epochs": self.trainer.state.num_train_epochs,
            "log_history": self.trainer.state.log_history[-5:] if self.trainer.state.log_history else [],
        }
        
        return state
    
    def finish_wandb(self) -> None:
        """Finish wandb run."""
        if self.wandb_manager:
            self.wandb_manager.finish()
            logger.info("Wandb run finished")
    
    def get_wandb_url(self) -> Optional[str]:
        """Get wandb run URL."""
        if self.wandb_manager:
            return self.wandb_manager.get_run_url()
        return None 