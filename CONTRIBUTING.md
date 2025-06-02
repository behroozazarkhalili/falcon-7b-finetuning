# Contributing to Falcon-7B Fine-tuning Project

Thank you for your interest in contributing to this project! This guide will help you get started.

## 🚀 Quick Start

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/your-username/falcon-7b-finetuning.git
   cd falcon-7b-finetuning
   ```

3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 🛠️ Development Workflow

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Automated checks

Run all checks:
```bash
# Format code
black .
isort .

# Lint code
flake8 .

# Type check
mypy src/

# Run all pre-commit hooks
pre-commit run --all-files
```

### Testing

We use pytest for testing. Write tests for all new functionality.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m "not slow"
```

### Documentation

- Use clear, descriptive docstrings for all functions and classes
- Follow Google-style docstring format
- Update README.md for significant changes
- Add type hints to all function signatures

Example docstring:
```python
def train_model(config: DictConfig, dataset: Dataset) -> Dict[str, float]:
    """
    Train a model with the given configuration and dataset.
    
    Args:
        config: Training configuration containing hyperparameters
        dataset: Training dataset
        
    Returns:
        Dictionary containing training metrics
        
    Raises:
        ValueError: If configuration is invalid
    """
```

## 📝 Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   pytest
   pre-commit run --all-files
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

   Use conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for test additions/changes
   - `refactor:` for code refactoring
   - `style:` for formatting changes

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Use a clear, descriptive title
   - Provide detailed description of changes
   - Link any related issues
   - Ensure all CI checks pass

## 🏗️ Project Structure

```
src/
├── data/           # Data loading and preprocessing
├── models/         # Model definitions and utilities
├── training/       # Training logic and callbacks
├── evaluation/     # Evaluation metrics and utilities
├── inference/      # Inference and prediction utilities
└── utils/          # Common utilities

tests/
├── test_data/      # Data module tests
├── test_models/    # Model module tests
├── test_training/  # Training module tests
└── test_utils/     # Utility module tests

configs/
├── data/           # Dataset configurations
├── model/          # Model configurations
└── training/       # Training configurations
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment information**
   - Python version
   - Package versions (`pip freeze`)
   - Operating system
   - GPU information (if applicable)

2. **Steps to reproduce**
   - Minimal code example
   - Configuration files used
   - Command line arguments

3. **Expected vs actual behavior**
   - What you expected to happen
   - What actually happened
   - Error messages and stack traces

## 💡 Feature Requests

For feature requests, please:

1. **Check existing issues** to avoid duplicates
2. **Describe the use case** and motivation
3. **Provide examples** of how the feature would be used
4. **Consider implementation** if you have ideas

## 🔍 Code Review Guidelines

When reviewing code:

- **Be constructive** and respectful
- **Focus on the code**, not the person
- **Suggest improvements** with examples
- **Test the changes** locally if possible
- **Check for**:
  - Code style and formatting
  - Test coverage
  - Documentation updates
  - Performance implications
  - Security considerations

## 📚 Resources

- [Python Style Guide (PEP 8)](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)

## 🤝 Community

- Be respectful and inclusive
- Help others learn and grow
- Share knowledge and best practices
- Follow the project's code of conduct

## 📞 Getting Help

If you need help:

1. Check the documentation and README
2. Search existing issues
3. Ask questions in discussions
4. Reach out to maintainers

Thank you for contributing! 🎉 