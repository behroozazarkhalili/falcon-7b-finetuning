"""Weights & Biases utilities for experiment tracking and logging."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedModel, PreTrainedTokenizer

from .logging import get_logger

logger = get_logger(__name__)


class WandbManager:
    """Manager class for Weights & Biases integration."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize wandb manager.
        
        Args:
            config: Configuration containing wandb settings
        """
        self.config = config
        self.wandb_config = config.get("wandb", {})
        self.experiment_config = config.get("experiment", {})
        self.run = None
        self.logger = logger
    
    def init_wandb(self, resume: Optional[str] = None) -> Any:
        """
        Initialize wandb run.
        
        Args:
            resume: Resume mode ("allow", "must", "never", or run_id)
            
        Returns:
            Wandb run object
        """
        # Prepare wandb init arguments
        init_args = {
            "project": self.wandb_config.get("project", "falcon-7b-finetuning"),
            "name": self.wandb_config.get("name") or self.experiment_config.get("name"),
            "tags": self.wandb_config.get("tags") or self.experiment_config.get("tags"),
            "notes": self.wandb_config.get("notes") or self.experiment_config.get("description"),
            "group": self.wandb_config.get("group"),
            "job_type": self.wandb_config.get("job_type", "training"),
            "config": self._prepare_config_for_wandb(),
        }
        
        # Add entity if specified
        if self.wandb_config.get("entity"):
            init_args["entity"] = self.wandb_config["entity"]
        
        # Add resume if specified
        if resume:
            init_args["resume"] = resume
        
        # Filter out None values
        init_args = {k: v for k, v in init_args.items() if v is not None}
        
        self.logger.info(f"Initializing wandb with project: {init_args['project']}")
        
        try:
            self.run = wandb.init(**init_args)
            self.logger.info(f"Wandb run initialized: {self.run.name} ({self.run.id})")
            
            # Apply wandb settings
            self._apply_wandb_settings()
            
            return self.run
            
        except Exception as e:
            self.logger.error(f"Failed to initialize wandb: {e}")
            raise
    
    def _prepare_config_for_wandb(self) -> Dict[str, Any]:
        """
        Prepare configuration for wandb logging.
        
        Returns:
            Dictionary suitable for wandb config
        """
        # Convert OmegaConf to regular dict for wandb
        config_dict = OmegaConf.to_container(self.config, resolve=True)
        
        # Remove wandb config from the logged config to avoid recursion
        if "wandb" in config_dict:
            del config_dict["wandb"]
        
        return config_dict
    
    def _apply_wandb_settings(self) -> None:
        """Apply wandb-specific settings."""
        settings = self.wandb_config.get("settings", {})
        
        # Set wandb settings
        if settings.get("save_code", True):
            wandb.run.log_code(".")
        
        self.logger.info("Wandb settings applied")
    
    def watch_model(self, model: PreTrainedModel) -> None:
        """
        Watch model for gradient and parameter logging.
        
        Args:
            model: Model to watch
        """
        if not self.run:
            self.logger.warning("Wandb not initialized, cannot watch model")
            return
        
        settings = self.wandb_config.get("settings", {})
        log_config = self.wandb_config.get("log", {})
        
        if settings.get("watch_model", True):
            log_freq = settings.get("watch_freq", 100)
            log_gradients = log_config.get("gradients", False)
            log_parameters = log_config.get("parameters", True)
            
            # Determine what to log
            log_type = []
            if log_gradients:
                log_type.append("gradients")
            if log_parameters:
                log_type.append("parameters")
            
            if log_type:
                wandb.watch(model, log=log_type, log_freq=log_freq)
                self.logger.info(f"Watching model with log_type: {log_type}, freq: {log_freq}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if not self.run:
            self.logger.warning("Wandb not initialized, cannot log metrics")
            return
        
        wandb.log(metrics, step=step)
    
    def log_model_info(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
        """
        Log model information as wandb config.
        
        Args:
            model: Model to log info about
            tokenizer: Tokenizer to log info about
        """
        if not self.run:
            self.logger.warning("Wandb not initialized, cannot log model info")
            return
        
        model_info = {
            "model_name": self.config.model.name,
            "model_type": type(model).__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "vocab_size": tokenizer.vocab_size,
            "max_length": getattr(tokenizer, "model_max_length", "unknown"),
        }
        
        wandb.config.update({"model_info": model_info})
        self.logger.info("Model info logged to wandb")
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]) -> None:
        """
        Log dataset information.
        
        Args:
            dataset_info: Dictionary containing dataset information
        """
        if not self.run:
            self.logger.warning("Wandb not initialized, cannot log dataset info")
            return
        
        wandb.config.update({"dataset_info": dataset_info})
        self.logger.info("Dataset info logged to wandb")
    
    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        artifact_name: str,
        artifact_type: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an artifact to wandb.
        
        Args:
            artifact_path: Path to the artifact
            artifact_name: Name of the artifact
            artifact_type: Type of the artifact (e.g., "model", "dataset")
            description: Optional description
            metadata: Optional metadata dictionary
        """
        if not self.run:
            self.logger.warning("Wandb not initialized, cannot log artifact")
            return
        
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description=description,
            metadata=metadata
        )
        
        artifact.add_file(str(artifact_path))
        wandb.log_artifact(artifact)
        
        self.logger.info(f"Artifact '{artifact_name}' logged to wandb")
    
    def log_model_checkpoint(self, checkpoint_path: Union[str, Path], step: int) -> None:
        """
        Log model checkpoint as wandb artifact.
        
        Args:
            checkpoint_path: Path to the checkpoint
            step: Training step
        """
        artifacts_config = self.wandb_config.get("artifacts", {})
        
        if not artifacts_config.get("log_model_checkpoints", True):
            return
        
        self.log_artifact(
            artifact_path=checkpoint_path,
            artifact_name=f"model-checkpoint-step-{step}",
            artifact_type="model",
            description=f"Model checkpoint at step {step}",
            metadata={"step": step, "type": "checkpoint"}
        )
    
    def log_final_model(self, model_path: Union[str, Path]) -> None:
        """
        Log final trained model as wandb artifact.
        
        Args:
            model_path: Path to the final model
        """
        artifacts_config = self.wandb_config.get("artifacts", {})
        
        if not artifacts_config.get("log_final_model", True):
            return
        
        self.log_artifact(
            artifact_path=model_path,
            artifact_name="final-model",
            artifact_type="model",
            description="Final trained model",
            metadata={"type": "final_model"}
        )
    
    def finish(self) -> None:
        """Finish the wandb run."""
        if self.run:
            wandb.finish()
            self.logger.info("Wandb run finished")
    
    def get_run_url(self) -> Optional[str]:
        """
        Get the URL of the current wandb run.
        
        Returns:
            URL string or None if no run is active
        """
        if self.run:
            return self.run.get_url()
        return None
    
    def get_run_id(self) -> Optional[str]:
        """
        Get the ID of the current wandb run.
        
        Returns:
            Run ID string or None if no run is active
        """
        if self.run:
            return self.run.id
        return None


def init_wandb_from_config(config: DictConfig, resume: Optional[str] = None) -> Optional[WandbManager]:
    """
    Initialize wandb from configuration.
    
    Args:
        config: Configuration containing wandb settings
        resume: Resume mode for wandb
        
    Returns:
        WandbManager instance or None if wandb is disabled
    """
    # Check if wandb is enabled in report_to
    report_to = config.training.get("report_to", [])
    if isinstance(report_to, str):
        report_to = [report_to]
    
    if "wandb" not in report_to:
        logger.info("Wandb not enabled in report_to, skipping initialization")
        return None
    
    # Check if wandb is available
    try:
        import wandb
    except ImportError:
        logger.warning("Wandb not installed, skipping initialization")
        return None
    
    # Initialize wandb manager
    wandb_manager = WandbManager(config)
    wandb_manager.init_wandb(resume=resume)
    
    return wandb_manager


def log_system_info() -> None:
    """Log system information to wandb."""
    if wandb.run is None:
        return
    
    import platform
    import psutil
    import torch
    
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        system_info.update({
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        })
    
    wandb.config.update({"system_info": system_info})
    logger.info("System info logged to wandb") 