#!/usr/bin/env python3
"""
Training script for Falcon-7B fine-tuning with Wandb integration.

This script demonstrates how to run training with Wandb experiment tracking enabled.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def main():
    """Main function to run training with wandb."""
    parser = argparse.ArgumentParser(description="Train Falcon-7B with Wandb tracking")
    
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="falcon-7b-finetuning",
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        help="Wandb entity (username or team)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume wandb run (run ID or 'allow')"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Build command for the main training script
    cmd_parts = [
        "python", "scripts/train.py",
        "--config", "configs/training/wandb.yaml",  # Use wandb-specific config
        "--model-config", "configs/model/model.yaml",
        "--data-config", "configs/data/data.yaml"
    ]
    
    # Add wandb-specific arguments
    if args.wandb_project:
        cmd_parts.extend(["--wandb-project", args.wandb_project])
    
    if args.wandb_entity:
        cmd_parts.extend(["--wandb-entity", args.wandb_entity])
    
    if args.experiment_name:
        cmd_parts.extend(["--experiment-name", args.experiment_name])
    
    if args.resume:
        cmd_parts.extend(["--resume-wandb", args.resume])
    
    if args.debug:
        cmd_parts.append("--debug")
    
    # Set environment variables for wandb
    env = os.environ.copy()
    
    # Disable wandb prompts in non-interactive environments
    env["WANDB_SILENT"] = "true"
    
    # Print the command being executed
    print("🚀 Starting Falcon-7B training with Wandb tracking...")
    print(f"Command: {' '.join(cmd_parts)}")
    print(f"Wandb project: {args.wandb_project}")
    if args.wandb_entity:
        print(f"Wandb entity: {args.wandb_entity}")
    print("=" * 50)
    
    # Execute the training script
    import subprocess
    result = subprocess.run(cmd_parts, env=env)
    
    if result.returncode == 0:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed!")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main() 