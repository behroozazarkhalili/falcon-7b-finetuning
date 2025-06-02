# Falcon-7B Fine-tuning Project

A production-ready ML engineering project for fine-tuning Falcon-7B using QLoRA and TRL, following industry best practices.

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with reusable components
- **Reproducible Experiments**: Seed management and experiment tracking
- **Configuration Management**: YAML-based configuration system
- **Comprehensive Testing**: Unit tests for all components
- **CI/CD Pipeline**: Automated testing and deployment
- **Data Versioning**: DVC integration for dataset management
- **Monitoring & Logging**: MLflow integration for experiment tracking
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
│   │   └── falcon-7b.yaml
│   ├── training/
│   │   └── default.yaml
│   └── data/
│       └── guanaco.yaml
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
│       └── reproducibility.py
├── tests/
│   ├── __init__.py
│   ├── test_data/
│   ├── test_models/
│   ├── test_training/
│   └── test_utils/
├── scripts/
│   ├── train.py
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

```bash
python scripts/train.py --config configs/training/default.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --model-path models/final/falcon-7b-finetuned --config configs/training/default.yaml
```

### Inference

```bash
python scripts/inference.py --model-path models/final/falcon-7b-finetuned --prompt "Your prompt here"
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
```

## 📊 Monitoring

- **MLflow**: Track experiments at `http://localhost:5000`
- **Logs**: Check `logs/` directory for detailed training logs
- **Metrics**: Model performance metrics are logged automatically

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