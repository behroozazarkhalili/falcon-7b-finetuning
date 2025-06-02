"""Reproducibility utilities for ensuring deterministic behavior."""

import os
import random
from typing import Dict, Optional

import numpy as np
import torch
from transformers import set_seed as transformers_set_seed

from .logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value
    """
    logger.info(f"Setting random seed to {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Transformers
    transformers_set_seed(seed)
    
    # Environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # PyTorch deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info("Random seeds set successfully")


def get_device_info() -> Dict[str, any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary containing device information
    """
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": None,
        "device_memory": None,
        "torch_version": torch.__version__,
    }
    
    if torch.cuda.is_available():
        device_info["device_name"] = torch.cuda.get_device_name()
        device_info["device_memory"] = torch.cuda.get_device_properties(0).total_memory
        
        # Log GPU information
        logger.info(f"CUDA available: {device_info['cuda_available']}")
        logger.info(f"GPU count: {device_info['cuda_device_count']}")
        logger.info(f"Current GPU: {device_info['device_name']}")
        logger.info(f"GPU memory: {device_info['device_memory'] / 1e9:.1f} GB")
    else:
        logger.warning("CUDA not available, using CPU")
    
    return device_info


def get_reproducibility_info(seed: Optional[int] = None) -> Dict[str, any]:
    """
    Get comprehensive reproducibility information.
    
    Args:
        seed: Random seed used (if any)
        
    Returns:
        Dictionary containing reproducibility information
    """
    info = {
        "seed": seed,
        "python_version": os.sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    
    # Add device information
    info.update(get_device_info())
    
    return info


def log_reproducibility_info(seed: Optional[int] = None) -> None:
    """
    Log comprehensive reproducibility information.
    
    Args:
        seed: Random seed used (if any)
    """
    info = get_reproducibility_info(seed)
    
    logger.info("=== Reproducibility Information ===")
    for key, value in info.items():
        logger.info(f"{key}: {value}")
    logger.info("===================================")


def ensure_deterministic_behavior() -> None:
    """
    Ensure deterministic behavior for reproducible results.
    Note: This may impact performance.
    """
    logger.info("Enabling deterministic behavior")
    
    # PyTorch deterministic algorithms
    torch.use_deterministic_algorithms(True)
    
    # Additional environment variables
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    logger.info("Deterministic behavior enabled")


def disable_deterministic_behavior() -> None:
    """
    Disable deterministic behavior for better performance.
    """
    logger.info("Disabling deterministic behavior for performance")
    
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)
    
    logger.info("Deterministic behavior disabled") 