# Black Hole Weather Forecasting - AGENT Guidelines

## Project Overview
This repository contains a deep learning model for Black Hole Weather Forecasting using a modified U-Net architecture. The codebase is written in Python using TensorFlow/Keras, with multi-GPU training capabilities.

## Build/Lint/Test Commands

### Training
- Train model: `python3 train.py --batch_size=16 --results_path=Test/`
- Use training script: `bash run_train.sh`
- For memory-efficient training with generator: `python3 train_II.py`

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
3. Install dependencies: `pip install -r requirements.txt`

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
- Target: TensorFlow ≥ 2.4.0 (see `requirements.txt`). The original paper implementation used Keras 2.1 + TensorFlow 1.8, which explains the mixed import styles across files.
- Deprecated APIs present in the codebase that must not be propagated to new code:
  - `multi_gpu_model` → replace with `tf.distribute.MirroredStrategy`
  - `model.fit_generator` → replace with `model.fit` + a `tf.data` pipeline
  - `tensorflow.python.keras` (private) → use `tensorflow.keras`

## Domain Context

Critical facts for working with this codebase:

- **The 5 channels are temporal, not physical.** Input/output tensors `(N, 256, 192, 5)` stack 5 consecutive density snapshots. The model advances the simulation by 5 frames per call (Δt = 197.97 M each, ≈990 M total).
- **All data paths are hardcoded** in `train.py`, `train_II.py`, and `inference.py` (pointing to `/DL/dl_coding/DL_code/`). Update these before running.
- **`CustomLoss` in `train.py`** encodes physics via fixed pixel regions on the 256×192 grid: the slice weights (8×, 5×, 10×, 4×) correspond to the high-density, inner/accretion disk, torus, and atmosphere regions respectively — do not modify them without consulting the paper (arXiv:2102.06242, Table 1).