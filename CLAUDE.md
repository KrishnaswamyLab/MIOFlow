# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MIOFlow is a Python research package implementing **Manifold Interpolating Optimal-Transport Flows** for trajectory inference in single-cell RNA-seq (scRNA-seq) data. It uses PyTorch-based Neural ODEs combined with optimal transport to model cell developmental trajectories. (Citation: Huguet et al., arXiv:2206.14928)

## Commands

### Installation

```bash
pip install -e .                        # Development install
conda env create -f environment.yml     # Conda environment (recommended for GPU)
```

### Build & Release

```bash
make dist    # Build sdist + wheel into dist/
make clean   # Remove dist/
make pypi    # Upload to PyPI via twine
```

### Running Tutorials

```bash
jupyter notebook tutorials/
```

There are no automated tests beyond CI installation checks (`pip install -e .`). Functionality is validated through the Jupyter notebooks in `tutorials/`.

## Architecture

### Package Structure (`MIOFlow/`)

The package follows a layered architecture:

```
User API (mioflow.py: MIOFlow class)
  ├── Embedding: phate_autoencoder.py + phate_decoder.py
  ├── Training: train.py (train, train_ae, training_regimen)
  ├── Models: models.py (ToyODE, Autoencoder, GrowthRateModel, ConditionalODE)
  ├── Losses: losses.py (MMD_loss, OT_loss, Density_loss)
  ├── Evaluation: eval.py (generate_points, generate_trajectories)
  └── Visualization: plots.py
```

**Supporting modules:**
- `ode.py` — Neural ODE wrappers around `torchdiffeq`
- `geo.py` — Geometric/distance computations for the manifold
- `utils.py` — Sampling, torch↔numpy conversion, group extraction
- `datasets.py` — Synthetic data generators + real scRNA-seq loaders (worm, EB, DynGen)
- `exp.py` — Experiment configuration via Hydra/OmegaConf
- `constants.py` — Dataset file paths
- `preprocessing.py` — Preprocessing utilities (early stage)

### Training Pipeline

MIOFlow trains in two phases within `training_regimen()`:
1. **Local phase** — predicts t+1 from t (short-range temporal consistency)
2. **Global phase** — full trajectory optimization

The `MIOFlow` class in [MIOFlow/mioflow.py](MIOFlow/mioflow.py) is the primary user-facing API and orchestrates the full workflow: dimensionality reduction via PHATE autoencoder → ODE training → trajectory generation.

### Key Neural Components

- **ToyODE** (`models.py`) — Core derivative network with optional augmentation, momentum, and noise scales
- **PhateAutoencoder** (`phate_autoencoder.py`) — PHATE-based encoder for non-linear dimensionality reduction
- **Loss functions** (`losses.py`) — MMD (Maximum Mean Discrepancy), OT (Optimal Transport via EMD/Sinkhorn using `pot`), and Density losses

### Dependencies

Core: `torch`, `torchdiffeq`, `torchsde`, `pytorch-lightning`, `scanpy`, `phate`, `graphtools`, `pot` (Python Optimal Transport), `hydra-core`

## Version Note

The version in [pyproject.toml](pyproject.toml) and [MIOFlow/__init__.py](MIOFlow/__init__.py) may be out of sync — update both when releasing.
