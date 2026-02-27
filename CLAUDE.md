# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep learning model for forecasting black hole gas dynamics using a modified U-Net. A CNN trained on 2D hydrodynamical simulations of radiatively inefficient accretion flows (RIAFs) around a Schwarzschild black hole, predicting the future density field evolution. Published in Duarte, Nemmen & Navarro (MNRAS, 2022) — [arXiv:2102.06242](https://arxiv.org/abs/2102.06242).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train (loads all data into RAM)
python3 src/train.py --batch_size=16 --results_path=Test/
bash run_train.sh  # equivalent shortcut

# Train with generator (memory-efficient, samples 500 random frames per epoch)
python3 src/train_II.py --batch_size=32 --results_path=Test/

# Inference (update hardcoded paths first)
python3 src/inference.py
```

No test suite exists. For linting: `pip install black flake8 && black . && flake8 .`

## Data and Preprocessing

**The 5 channels are temporal, not physical variables.** Each sample stacks 5 consecutive density snapshots into a `(256, 192, 5)` tensor; the model predicts the 5 frames that immediately follow. This is how temporal information is encoded in the 4th tensor dimension.

**Raw simulation grids** are 400×200 cells (polar coordinates, non-uniform, higher resolution at small radii). Before training, the data is:
1. **Log-normalized** to [0, 1]: `ρ_NORM = (log(ρ) − log(min(ρ))) / (log(max(ρ)) − log(min(ρ)))`. Raw density ranges from 10⁻⁵ to 10¹.
2. **Cropped** to 256×192: 144 radial cells removed from the outer boundary (low-density atmosphere with ρ < 10⁻³), 8 polar cells removed from the poles.

**Data files**: `x.npy` (input, shape `(N, 256, 192, 5)`) and `y.npy` (target, same shape). Paths are **hardcoded** in `src/train.py`, `src/train_II.py`, and `src/inference.py` pointing to the original training server (`/DL/dl_coding/DL_code/`) and must be updated for each environment.

**Time step**: Δt = 197.97 M between consecutive frames; each model prediction advances the simulation by 5 frames (≈990 M).

## Architecture

**`src/models.py`** — defines `create_auto_encoder(filters)`: a U-Net with 4-level encoder (doubling filters each level starting from `2×filters`, default 32 → 64/128/256/512), bottleneck, 4-level decoder with skip connections via `concatenate`, and a final `Conv2D(5, 1×1, linear)` output. Uses LeakyReLU, no BatchNorm. Input/output: `(256, 192, 5)`. Originally implemented with Keras 2.1 + TensorFlow 1.8.

**`src/train.py`** — standard training: loads full dataset, 80/10/10 split, multi-GPU via `multi_gpu_model(model, gpus=2)`, Adam(lr=5×10⁻⁴), saves best weights to `dl_fluids.h5`. Uses `CustomLoss`: a physics-informed hierarchical loss `L = L_total + 8·L_HD + 5·L_IN + 10·L_torus + 4·L_atm` where each term is MAE over a fixed pixel region of the 256×192 grid (except L_total which is MSE). These region weights (α=8, β=5, γ=10, δ=4) are the best hyperparameters from the paper's grid search (Table 1).

**`src/train_II.py`** — generator-based training (memory-efficient): randomly samples 500 frames per epoch. Uses the simpler 2-term `LossCustom(alpha, beta)` designed for multi-simulation training. Uses deprecated `fit_generator`.

**`src/params.py`** — `argparse` config imported as `args`. Key params: `--epochs` (100), `--batch_size` (64), `--filters` (32), `--results_path`, `--alpha/beta/delta/gamma` (loss region weights). Also provides `write_results()`.

**`src/inference.py`** — autoregressive prediction loop (100 iterations), saving each `(1, 256, 192, 5)` prediction as a `.npy` file. Uses standalone `keras` imports — needs updating for TF2.

## Training Experiments (from paper)

Two experimental setups described in the paper:
- **one-sim**: trained on a single long simulation (PNSS3, 2678 frames, 70/10/20 split). `src/train.py` + `CustomLoss`.
- **multi-sim**: trained on 8 simulations (5015 frames total), withholding PL0SS3 for out-of-distribution testing. `src/train_II.py` + `LossCustom`.

Simulation naming convention: `[PN/PL][0/2][SS/ST][1/3]` — angular momentum profile (PN=Penna, PL=power-law), exponent, viscosity prescription (SS=Shakura-Sunyaev, ST=Stone), and α×10 value.

Two forecast modes: **direct** (one-step nowcasting) and **iterative** (autoregressive, feeds predictions back as inputs).

## Known Issues / Technical Debt

- `src/models.py` uses `tensorflow.python.keras` (private API); `src/train.py`/`src/train_II.py` use `tensorflow.keras`; `src/inference.py` uses standalone `keras` — inconsistent, should be unified to `tensorflow.keras`.
- `multi_gpu_model` is deprecated in TF2; replace with `tf.distribute.MirroredStrategy`.
- `fit_generator` is deprecated in TF2; replace with `fit` + a `tf.data` pipeline.
- Pre-trained weights and sample data: [Figshare](https://doi.org/10.6084/m9.figshare.19412147.v1).
