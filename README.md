# Black Hole Weather Forecasting with Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

A deep learning model for forecasting black hole weather patterns using a modified U-Net architecture. This is the implementation from the paper ["Black Hole Weather Forecasting with Deep Learning: A Pilot Study"](https://arxiv.org/abs/2102.06242) by Duarte, Nemmen & Navarro (MNRAS, in press).

## Overview

This repository implements a U-Net-based deep learning model for predicting the evolution of gas dynamics around black holes. The model takes spatiotemporal data as input and forecasts the next time step of the evolution.

### Key Features

- Modified U-Net architecture accepting temporal + 2D spatial dimensions
- Multi-GPU training support
- Memory-efficient generator-based training option
- Custom loss functions for physics-informed training
- Support for TensorFlow 2.x

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ GPU memory (for multi-GPU training with batch size 64)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/black-hole-weather-CNN.git
cd black-hole-weather-CNN
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

To train the model:

```bash
# Using the provided shell script
bash run_train.sh

# Or with custom parameters
python3 train.py --batch_size=16 --results_path=Test/
```

For memory-efficient training with a data generator:

```bash
python3 train_II.py --batch_size=32 --results_path=Test/
```

### Input Data Format

The model expects input data with the following specifications:

- **Shape**: `(N, 256, 192, 5)` where:
  - `N`: Number of samples
  - `256`: Height (spatial dimension)
  - `192`: Width (spatial dimension)
  - `5`: Number of channels/features

- **Data files**: Training data should be saved as NumPy arrays:
  - `x.npy`: Input features (shape: `(N, 256, 192, 5)`)
  - `y.npy`: Target labels (shape: `(N, 256, 192, 5)`)

### Configuration

Edit `params.py` to customize training hyperparameters:

```python
--epochs: Number of training epochs (default: 100)
--batch_size: Mini-batch size (default: 64)
--filters: Number of filters in first layer (default: 32)
--results_path: Path to save training results
--alpha, --beta, --delta, --gamma: Loss function parameters (default: 0.1)
```

### Inference

To make predictions using a trained model:

```python
python3 inference.py
```

Note: Update the hardcoded paths in `inference.py` to point to your model weights and test data.

## Model Architecture

The model is based on a U-Net architecture with the following modifications:

- **Input**: Accepts 5-channel spatiotemporal data `(256, 192, 5)`
- **Output**: Predicts next timestep `(256, 192, 5)`
- **Encoder**: 5 convolutional blocks with max pooling
- **Bottleneck**: 2 convolutional blocks
- **Decoder**: 4 upsampling blocks with skip connections
- **Final Layer**: 1x1 convolution with linear activation

The architecture preserves spatial information through skip connections between encoder and decoder layers.

## Data Files

Pre-trained model weights and sample data are available in our [Figshare repository](https://doi.org/10.6084/m9.figshare.19412147.v1).

The model weights are saved in HDF5 format (`.h5` files) following TensorFlow standards. Load them using:

```python
from tensorflow.keras.models import load_model
model = load_model("/path/to/dl_fluids.h5")
```

## Project Structure

```
.
├── models.py          # U-Net model architecture
├── train.py           # Standard training script
├── train_II.py        # Generator-based training (memory efficient)
├── inference.py       # Prediction/inference script
├── params.py          # Hyperparameter configuration
├── run_train.sh       # Shell script for training
├── requirements.txt   # Python dependencies
├── LICENSE           # MIT License
└── README.md         # This file
```

## Hardware Requirements

- **Training**: Multi-GPU setup recommended (tested on P6000 and GP100)
- **Inference**: Single GPU or CPU (slower)
- **Memory**: Sufficient RAM to load training data or use generator-based training

## Troubleshooting

### Common Issues

1. **Path Errors**: Update hardcoded paths in `train.py`, `train_II.py`, and `inference.py` to match your system
2. **Memory Errors**: Reduce `batch_size` or use `train_II.py` with the generator
3. **GPU Not Found**: Ensure CUDA and cuDNN are properly installed

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{duarte2021black,
  title={Black Hole Weather Forecasting with Deep Learning: A Pilot Study},
  author={Duarte, Mayron and Nemmen, Rodrigo and Navarro, Joao},
  journal={Monthly Notices of the Royal Astronomical Society},
  year={2021},
  note={In press},
  url={https://arxiv.org/abs/2102.06242}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see [AGENTS.md](AGENTS.md) for development guidelines and code style requirements.

## Contact

For questions or issues, please open a GitHub issue or contact the authors.

## Acknowledgments

This work was supported by FAPESP and CNPq. We gratefully acknowledge GPU donations by NVIDIA.