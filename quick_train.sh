#!/bin/bash

# Quick Training Script for Falcon-7B Fine-tuning
# Simple and concise execution

echo "🚀 Starting Falcon-7B Fine-tuning..."

# Initialize conda and activate environment
eval "$(conda shell.bash hook)" && conda activate behrooz && pip install -e . 
python scripts/train.py --config configs/training/default.yaml --model-config configs/model/model.yaml --data-config configs/data/data.yaml

echo "✅ Training completed!"