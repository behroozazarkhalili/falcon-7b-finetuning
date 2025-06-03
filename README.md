# Falcon-7B Fine-tuning Project

A production-ready ML engineering project for fine-tuning Falcon-7B using QLoRA and TRL, following industry best practices.

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with reusable components
- **Reproducible Experiments**: Seed management and experiment tracking
- **Configuration Management**: YAML-based configuration system
- **Comprehensive Testing**: Unit tests for all components
- **CI/CD Pipeline**: Automated testing and deployment
- **Data Versioning**: DVC integration for dataset management
- **Monitoring & Logging**: MLflow and Wandb integration for experiment tracking
- **Type Safety**: Full type hints and validation

## 📁 Project Structure

```
falcon-7b-finetuning/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── cd.yml
│       └── model-training.yml
├── configs/
│   ├── model/
│   │   └── model.yaml
│   ├── training/
│   │   ├── default.yaml
│   │   └── wandb.yaml
│   └── data/
│       └── data.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── falcon.py
│   │   └── base.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── callbacks.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logging.py
│       ├── reproducibility.py
│       └── wandb_utils.py
├── tests/
│   ├── __init__.py
│   ├── test_data/
│   ├── test_models/
│   ├── test_training/
│   └── test_utils/
├── scripts/
│   ├── train.py
│   ├── train_with_wandb.py
│   ├── evaluate.py
│   ├── inference.py
│   └── setup_environment.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── .gitkeep
├── models/
│   ├── checkpoints/
│   ├── final/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
├── notebooks/
│   └── exploration.ipynb
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .gitignore
├── .pre-commit-config.yaml
├── requirements.txt
├── environment.yml
├── setup.py
├── pyproject.toml
├── dvc.yaml
└── README.md
```

## 🛠️ Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git
- DVC (for data versioning)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/behroozazarkhalili/falcon-7b-finetuning.git
   cd falcon-7b-finetuning
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # or using conda
   conda env create -f environment.yml
   ```

4. **Setup pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Initialize DVC**
   ```bash
   dvc init
   ```

6. **Setup MLflow tracking**
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
   ```

## 🚀 Usage

### Training

#### Basic Training
```bash
python scripts/train.py --config configs/training/default.yaml
```

#### Training with Wandb
```bash
# Using the dedicated wandb script
python scripts/train_with_wandb.py --wandb-project "my-falcon-project" --wandb-entity "my-username"

# Or using the main script with wandb config
python scripts/train.py --config configs/training/wandb.yaml --wandb-project "my-falcon-project"
```

#### Training with Custom Configuration
```bash
python scripts/train.py \
    --config configs/training/wandb.yaml \
    --model-config configs/model/model.yaml \
    --data-config configs/data/data.yaml \
    --experiment-name "my-experiment" \
    --wandb-project "my-project"
```

### Evaluation

```bash
python scripts/evaluate.py --model-path models/final/falcon-7b-finetuned --config configs/training/default.yaml
```

### Inference

```bash
python scripts/inference.py --model-path models/final/falcon-7b-finetuned --prompt "Your prompt here"
```

## 📊 Experiment Tracking

### Wandb Integration

The project includes comprehensive Wandb integration for experiment tracking:

#### Features:
- **Automatic experiment logging**: Model info, system info, and hyperparameters
- **Real-time metrics tracking**: Training and validation metrics
- **Model artifact logging**: Checkpoints and final models
- **Code versioning**: Automatic code snapshot
- **System monitoring**: GPU usage, memory, and system metrics

#### Configuration:
```yaml
# configs/training/wandb.yaml
wandb:
  project: "falcon-7b-finetuning"
  entity: "your-username"  # Set to your wandb username/team
  group: "falcon-7b-experiments"
  job_type: "fine-tuning"
  
  settings:
    save_code: true
    log_model: true
    watch_model: true
    watch_freq: 100
    
  artifacts:
    log_model_checkpoints: true
    log_final_model: true
    log_dataset_info: true
```

#### Setup Wandb:
1. **Install wandb** (already included in requirements):
   ```bash
   pip install wandb
   ```

2. **Login to wandb**:
   ```bash
   wandb login
   ```

3. **Run training with wandb**:
   ```bash
   python scripts/train_with_wandb.py --wandb-project "my-project"
   ```

#### Environment Variables:
```bash
# Optional: Set wandb environment variables
export WANDB_PROJECT="falcon-7b-finetuning"
export WANDB_ENTITY="your-username"
export WANDB_SILENT="true"  # Disable prompts in non-interactive environments
```

### Configuration

All configurations are stored in YAML files under the `configs/` directory. You can modify hyperparameters, model settings, and data configurations without changing the code.

Example training configuration:
```yaml
model:
  name: "tiiuae/falcon-7b"
  quantization:
    load_in_4bit: true
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_compute_dtype: "float16"
    bnb_4bit_use_double_quant: true

training:
  output_dir: "./results"
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  num_train_epochs: 5
  max_seq_length: 512
  report_to: ["wandb", "tensorboard"]  # Enable both wandb and tensorboard
```

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test categories:
```bash
pytest tests/test_models/ -v
pytest tests/test_data/ -v
pytest tests/test_utils/test_wandb_utils.py -v  # Test wandb integration
```

## 📊 Monitoring

- **Wandb**: Track experiments at your wandb dashboard
- **MLflow**: Track experiments at `http://localhost:5000`
- **TensorBoard**: View training logs with `tensorboard --logdir logs/`
- **Logs**: Check `logs/` directory for detailed training logs

## 🔄 CI/CD

The project includes GitHub Actions workflows for:
- **Continuous Integration**: Automated testing on pull requests
- **Continuous Deployment**: Model deployment on main branch
- **Model Training**: Scheduled retraining pipeline

## 📈 Data Management

- Raw data is stored in `data/raw/`
- Processed data is stored in `data/processed/`
- Data versioning is handled by DVC
- Dataset schemas are documented in `configs/data/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- TRL team for the training framework
- Falcon team for the base model
- Wandb for experiment tracking platform

## Quick Start

### Option 1: Using the Training Scripts (Recommended)

We provide multiple convenient scripts to run the training:

#### Full Training Script (with checks and logging)
```bash
./run_training.sh
```

#### Quick Training Script (minimal)
```bash
./quick_train.sh
```

#### Wandb-enabled Training
```bash
python scripts/train_with_wandb.py --wandb-project "my-project" --wandb-entity "my-username"
```

### Option 2: Manual Execution

1. **Activate the environment:**
```bash
conda activate behrooz  # or your environment name
```

2. **Install the project:**
```bash
pip install -e .
```

3. **Run training:**
```bash
# Basic training
python scripts/train.py \
    --config configs/training/default.yaml \
    --model-config configs/model/model.yaml \
    --data-config configs/data/data.yaml

# Training with wandb
python scripts/train.py \
    --config configs/training/wandb.yaml \
    --model-config configs/model/model.yaml \
    --data-config configs/data/data.yaml \
    --wandb-project "my-project"
``` 