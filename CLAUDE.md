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
User API (mioflow.py: MIOFlow class, TimeSeriesDataset, train_mioflow)
  ├── Embedding: gaga.py (Autoencoder, train_gaga_two_phase, PointCloudDataset, RowStochasticDataset)
  └── Core: core/
        ├── models.py (ODEFunc)
        └── losses.py (ot_loss, density_loss, energy_loss)
```

### Training Pipeline

MIOFlow training is a single global phase managed by `train_mioflow()` in `mioflow.py`. The `MIOFlow` class orchestrates the full workflow:

1. **Encoding** — optionally encodes `adata.obsm[gaga_input_key]` into a latent space using a pre-trained GAGA `Autoencoder` (from `gaga.py`). If no GAGA model is supplied, raw PCA coordinates are used directly.
2. **Normalisation** — Z-normalises the embedding per-dimension (mean/std stored for inversion).
3. **ODE training** — trains `ODEFunc` interval-by-interval using OT loss, optional density loss, and optional energy regularisation.
4. **Trajectory generation** — integrates sampled initial conditions across all time bins with `torchdiffeq.odeint`.
5. **Gene-space decoding** — `decode_to_gene_space()` inverts the GAGA decoder and the PCA projection to recover gene expression.

### Key Neural Components

- **ODEFunc** (`core/models.py`) — Time-conditioned derivative network (Linear → SiLU stack) with optional momentum smoothing. Kaiming initialisation.
- **Autoencoder / GAGA** (`gaga.py`) — Geometric autoencoder trained in two phases: (1) encoder trained for PHATE-distance preservation, (2) decoder trained for reconstruction. Helper classes `PointCloudDataset` and `RowStochasticDataset` are provided for data preparation.
- **Loss functions** (`core/losses.py`) — OT loss (Earth Mover's Distance via `pot`), density loss (kNN hinge), and energy loss (vector-field magnitude regularisation).

### Dependencies

Core: `torch`, `torchdiffeq`, `scanpy`, `phate`, `pot` (Python Optimal Transport)

## Version Note

The version in [pyproject.toml](pyproject.toml) and [MIOFlow/__init__.py](MIOFlow/__init__.py) may be out of sync — update both when releasing.
