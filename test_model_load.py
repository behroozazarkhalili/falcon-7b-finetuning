#!/usr/bin/env python3
"""
Simple test script to verify model loading without OOM.
"""

import os
import sys
from pathlib import Path

# Set memory optimization environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from src.utils import load_config, merge_configs
from src.models import create_falcon_model

def test_model_loading():
    """Test if the model can be loaded without OOM."""
    print("🧪 Testing model loading...")
    
    # Load configurations
    model_config = load_config("configs/model/model.yaml")
    
    print(f"📊 GPU Memory before loading:")
    if torch.cuda.is_available():
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    try:
        # Create model
        print("🔄 Loading model...")
        model = create_falcon_model(model_config)
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 GPU Memory after loading:")
        if torch.cuda.is_available():
            print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"  Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
        # Print model info
        info = model.get_peft_model_info()
        print(f"📈 Model Info:")
        print(f"  Total parameters: {info['total_parameters']:,}")
        print(f"  Trainable parameters: {info['trainable_parameters']:,}")
        print(f"  Trainable percentage: {info['trainable_percentage']:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("🎉 Model loading test passed!")
    else:
        print("💥 Model loading test failed!")
        sys.exit(1) 