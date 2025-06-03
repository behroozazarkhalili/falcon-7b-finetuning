"""Falcon model implementation with QLoRA and PEFT support."""

from typing import Optional

import torch
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ..utils.logging import get_logger
from .base import BaseModel

logger = get_logger(__name__)


class FalconModel(BaseModel):
    """Falcon model with QLoRA and PEFT configuration."""

    def __init__(self, config: DictConfig):
        """
        Initialize Falcon model.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.peft_config: Optional[LoraConfig] = None
        self.quantization_config: Optional[BitsAndBytesConfig] = None

    def _create_quantization_config(self) -> BitsAndBytesConfig:
        """
        Create quantization configuration for QLoRA.

        Returns:
            BitsAndBytesConfig for quantization
        """
        quant_config = self.config.quantization

        # Get compute dtype
        compute_dtype = getattr(
            torch, quant_config.get("bnb_4bit_compute_dtype", "float16")
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config.get("load_in_4bit", True),
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_config.get(
                "bnb_4bit_use_double_quant", True
            ),
        )

        self.logger.info(f"Created quantization config: {bnb_config}")
        return bnb_config

    def _create_lora_config(self) -> LoraConfig:
        """
        Create LoRA configuration for PEFT.

        Returns:
            LoraConfig for PEFT
        """
        lora_config = self.config.lora

        # Convert target_modules to native Python list to avoid JSON serialization issues
        target_modules = lora_config.get("target_modules", ["query_key_value"])
        if hasattr(target_modules, "__iter__") and not isinstance(target_modules, str):
            target_modules = list(target_modules)

        peft_config = LoraConfig(
            lora_alpha=lora_config.get("lora_alpha", 16),
            lora_dropout=lora_config.get("lora_dropout", 0.1),
            r=lora_config.get("r", 64),
            bias=lora_config.get("bias", "none"),
            task_type=lora_config.get("task_type", "CAUSAL_LM"),
            target_modules=target_modules,
        )

        self.logger.info(f"Created LoRA config: {peft_config}")
        return peft_config

    def load_model(self) -> PreTrainedModel:
        """
        Load Falcon model with quantization and PEFT.

        Returns:
            Loaded and configured model
        """
        self.logger.info(f"Loading Falcon model: {self.config.model.name}")

        # Create quantization config
        self.quantization_config = self._create_quantization_config()

        # Load base model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            quantization_config=self.quantization_config,
            device_map=self.config.model.get("device_map", "auto"),
            trust_remote_code=self.config.model.get("trust_remote_code", True),
            torch_dtype=torch.float16,
        )

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Create and apply LoRA config
        self.peft_config = self._create_lora_config()
        model = get_peft_model(model, self.peft_config)

        # Enable training mode
        model.train()

        self.model = model
        self.logger.info("Falcon model loaded successfully")

        return model

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load Falcon tokenizer.

        Returns:
            Loaded tokenizer
        """
        self.logger.info(f"Loading tokenizer: {self.config.tokenizer.name}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer.name,
            trust_remote_code=self.config.tokenizer.get("trust_remote_code", True),
        )

        # Set pad token
        pad_token_strategy = self.config.tokenizer.get("pad_token_strategy", "eos")

        if tokenizer.pad_token is None:
            if pad_token_strategy == "eos":
                tokenizer.pad_token = tokenizer.eos_token
            elif pad_token_strategy == "unk":
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.tokenizer = tokenizer
        self.logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

        return tokenizer

    def get_peft_model_info(self) -> dict:
        """
        Get PEFT-specific model information.

        Returns:
            Dictionary with PEFT model information
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        if not hasattr(self.model, "print_trainable_parameters"):
            return {"error": "Model is not a PEFT model"}

        # Get trainable parameters info
        trainable_params = 0
        all_params = 0

        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        info = {
            "total_parameters": all_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100 * trainable_params / all_params,
            "lora_config": self.peft_config.__dict__ if self.peft_config else None,
            "quantization_config": (
                self.quantization_config.__dict__ if self.quantization_config else None
            ),
        }

        return info

    def print_trainable_parameters(self) -> None:
        """Print trainable parameters information."""
        if self.model is None:
            self.logger.error("Model not loaded")
            return

        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()
        else:
            info = self.get_peft_model_info()
            if "error" not in info:
                self.logger.info(
                    f"Trainable parameters: {info['trainable_parameters']:,}"
                )
                self.logger.info(f"Total parameters: {info['total_parameters']:,}")
                self.logger.info(
                    f"Trainable percentage: {info['trainable_percentage']:.2f}%"
                )

    def merge_and_unload(self) -> PreTrainedModel:
        """
        Merge LoRA weights and unload PEFT model.

        Returns:
            Base model with merged weights
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        if not hasattr(self.model, "merge_and_unload"):
            raise ValueError("Model is not a PEFT model")

        self.logger.info("Merging LoRA weights and unloading PEFT model")
        merged_model = self.model.merge_and_unload()

        return merged_model

    def save_peft_model(self, output_path: str) -> None:
        """
        Save only the PEFT adapter weights.

        Args:
            output_path: Path to save the adapter
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        if not hasattr(self.model, "save_pretrained"):
            raise ValueError("Model is not a PEFT model")

        self.logger.info(f"Saving PEFT adapter to {output_path}")
        self.model.save_pretrained(output_path)

        if self.tokenizer:
            self.tokenizer.save_pretrained(output_path)

    def load_peft_model(self, adapter_path: str) -> None:
        """
        Load PEFT adapter weights.

        Args:
            adapter_path: Path to the adapter weights
        """
        from peft import PeftModel

        if self.model is None:
            # Load base model first
            self.load_model()

        self.logger.info(f"Loading PEFT adapter from {adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)


def create_falcon_model(config: DictConfig) -> FalconModel:
    """
    Create and initialize Falcon model.

    Args:
        config: Model configuration

    Returns:
        Initialized Falcon model
    """
    model = FalconModel(config)

    # Load model and tokenizer
    model.load_model()
    model.load_tokenizer()

    # Validate model
    model.validate_model()

    # Print model summary
    model.print_model_summary()
    model.print_trainable_parameters()

    return model
