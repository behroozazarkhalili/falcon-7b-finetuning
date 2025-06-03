#!/bin/bash

# Falcon-7B Fine-tuning Training Script
# This script runs the complete training pipeline with proper environment setup

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to initialize conda
init_conda() {
    # Initialize conda for bash
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
        return 0
    else
        return 1
    fi
}

# Function to check if conda environment exists
check_conda_env() {
    if conda env list | grep -q "^behrooz "; then
        return 0
    else
        return 1
    fi
}

# Main execution
main() {
    print_status "Starting Falcon-7B Fine-tuning Pipeline"
    print_status "========================================"
    
    # Check if we're in the right directory
    if [[ ! -f "scripts/train.py" ]]; then
        print_error "Training script not found. Please run this from the project root directory."
        exit 1
    fi
    
    # Initialize conda
    print_status "Initializing conda..."
    if init_conda; then
        print_success "Conda initialized successfully"
    else
        print_error "Conda not found. Please install conda first."
        exit 1
    fi
    
    # Activate conda environment
    print_status "Activating conda environment 'behrooz'..."
    if check_conda_env; then
        conda activate behrooz
        print_success "Conda environment activated"
    else
        print_error "Conda environment 'behrooz' not found. Please create it first."
        exit 1
    fi
    
    # Check if required packages are installed
    print_status "Checking package installation..."
    if ! python -c "import torch, transformers, peft, trl, datasets" 2>/dev/null; then
        print_warning "Some required packages may be missing. Installing from requirements.txt..."
        pip install -r requirements.txt
    fi
    
    # Install the project in development mode
    print_status "Installing project in development mode..."
    pip install -e .
    
    # Check GPU availability
    print_status "Checking GPU availability..."
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU\"}')"
    
    # Create necessary directories
    print_status "Creating output directories..."
    mkdir -p results logs models/checkpoints models/final
    
    # Display configuration
    print_status "Training Configuration:"
    echo "  - Model config: configs/model/model.yaml"
    echo "  - Data config: configs/data/data.yaml"
    echo "  - Training config: configs/training/default.yaml"
    echo "  - Output directory: ./results"
    
    # Start training
    print_status "Starting training..."
    echo "========================================"
    
    python scripts/train.py --config configs/training/default.yaml --model-config configs/model/model.yaml --data-config configs/data/data.yaml
    
    # Check if training completed successfully
    if [[ $? -eq 0 ]]; then
        print_success "Training completed successfully!"
        print_status "Model saved to: ./results"
        print_status "Logs available in: ./logs"
    else
        print_error "Training failed. Check the logs for details."
        exit 1
    fi
}

# Handle script interruption
trap 'print_warning "Training interrupted by user"; exit 130' INT

# Run main function
main "$@" 