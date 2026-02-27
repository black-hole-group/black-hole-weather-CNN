# Black Hole Weather Forecasting with Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

A deep learning model for forecasting black hole weather patterns using a modified U-Net architecture. This is the implementation from the paper ["Black Hole Weather Forecasting with Deep Learning: A Pilot Study"](https://arxiv.org/abs/2102.06242) by Duarte, Nemmen & Navarro (MNRAS, 2022). Accepted 2022 March 3.

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

#### Data Preprocessing

The raw simulation grids (400×200 cells in polar coordinates) are preprocessed before creating `x.npy`/`y.npy`:

1. **Log-normalize** density to [0, 1]:
   ```
   ρ_NORM = (log(ρ) − log(min(ρ))) / (log(max(ρ)) − log(min(ρ)))
   ```
   Raw density spans 10⁻⁵ to 10¹.

2. **Crop** from 400×200 → 256×192: remove 144 outer radial cells (low-density atmosphere with ρ < 10⁻³) and 8 polar cells from the poles.

3. **Stack** 5 consecutive density snapshots per sample (Δt = 197.97 M between frames). **The 5 channels are temporal, not physical variables.** The target `y` contains the 5 frames immediately following the input block, so each prediction advances the simulation by ≈990 M.

### Configuration

Edit `params.py` to customize training hyperparameters:

```python
--epochs: Number of training epochs (default: 100)
--batch_size: Mini-batch size (default: 64)
--filters: Number of filters in first layer (default: 32)
--results_path: Path to save training results
--alpha, --beta, --delta, --gamma: Loss region weights (default: 0.1)
```

**To reproduce paper results**, use the optimal hyperparameters from Table 1 of the paper:

| Parameter | Value | Region |
|-----------|-------|--------|
| α | 8 | High-density region |
| β | 5 | Inner/accretion disk |
| γ | 10 | Torus |
| δ | 4 | Atmosphere |
| Learning rate | 5×10⁻⁴ | (hardcoded in `train.py`) |

The defaults in `params.py` (0.1) are placeholders; pass the values above via CLI flags to reproduce the paper.

### Training Experiments

The paper describes two experimental setups:

- **one-sim** (`train.py`): Trained on a single long simulation (PNSS3, 2,678 frames) with a 70/10/20 temporal split. Uses `CustomLoss` — a physics-informed hierarchical loss with 5 regional components: `L = L_total + 8·L_HD + 5·L_IN + 10·L_torus + 4·L_atm`.

- **multi-sim** (`train_II.py`): Trained on 8 simulations (5,015 frames total), withholding PL0SS3 for out-of-distribution generalization testing. Uses the simpler 2-term `LossCustom(alpha, beta)`.

Simulation naming convention: `[PN/PL][0/2][SS/ST][1/3]` — angular momentum profile (PN=Penna, PL=power-law), exponent, viscosity prescription (SS=Shakura-Sunyaev, ST=Stone), and α×10.

### Inference

To make predictions using a trained model:

```python
python3 inference.py
```

Note: Update the hardcoded paths in `inference.py` to point to your model weights and test data.

The model supports two forecast modes:

- **Direct (nowcasting)**: Feed one density block, predict the immediately following 5 frames. Used for one-step accuracy evaluation.
- **Iterative (forecasting)**: Feed predictions back as inputs for autoregressive multi-step rollout. `inference.py` runs 100 iterations, advancing the simulation by ~10⁵ GM/c³. Known limitation: the one-sim model drifts after ~8×10⁴ GM/c³ due to artificial mass injection.

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
@article{duarte2022black,
  title={Black Hole Weather Forecasting with Deep Learning: A Pilot Study},
  author={Duarte, Roberta and Nemmen, Rodrigo and Navarro, Joao},
  journal={Monthly Notices of the Royal Astronomical Society},
  year={2022},
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