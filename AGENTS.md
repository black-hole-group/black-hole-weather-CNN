# Black Hole Weather Forecasting - AGENT Guidelines

## Project Overview
This repository contains a deep learning model for Black Hole Weather Forecasting using a modified U-Net architecture. The codebase is written in Python using TensorFlow/Keras, with multi-GPU training capabilities.

## Build/Lint/Test Commands

### Training
- Train model: `python3 train.py --batch_size=16 --results_path=Test/`
- Use training script: `bash run_train.sh`
- For memory-efficient training with generator: `python3 train_II.py`

### Testing
- No formal test framework currently exists
- To add tests, run: `pip install pytest` then create test files in `tests/` directory
- Run specific tests with: `python -m pytest tests/test_name.py::test_function_name`

### Linting and Type Checking
- Install linting tools: `pip install black flake8 mypy`
- Format code: `black .`
- Lint code: `flake8 .`
- Type check: `mypy .`

## Code Style Guidelines

### Imports
- Standardize to modern imports using `tensorflow.keras` instead of `tensorflow.python.keras`
- Group imports by:
  1. Standard library imports
  2. Third-party imports  
  3. Local application imports

### Formatting
- Use 4-space indentation
- Follow PEP 8 style guide
- Maximum line length: 88 characters
- Use black for automatic code formatting

### Naming Conventions
- Functions and variables: `snake_case`
- Constants: `UPPER_CASE`
- Classes: `PascalCase`

### Types
- Add type hints for all function parameters and return values
- Use Python type annotations consistently

### Error Handling
- Handle exceptions appropriately with try/except blocks
- Include proper logging for debugging and error tracking

### Documentation
- Add docstrings to all functions following numpy style
- Include module-level docstrings for each file

## Development Setup
1. Create virtual environment: `python -m venv venv`
2. Activate environment: `source venv/bin/activate`
3. Install dependencies: `pip install tensorflow numpy matplotlib scipy`

## Configuration
- Hyperparameters in `params.py`
- Model architecture in `models.py`
- Training logic in `train.py` and `train_II.py`
- Inference in `inference.py`

## Directory Structure
- `models.py`: Model architecture definition
- `train.py`: Training without generator
- `train_II.py`: Training with generator (memory efficient)
- `inference.py`: Prediction/evaluation scripts
- `params.py`: Hyperparameter configuration
- `run_train.sh`: Training bash script

## Version Management
- Currently uses TensorFlow 1.8.0 as noted in README
- No specific version pinning for other dependencies