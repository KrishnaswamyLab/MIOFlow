<a id="readme-top"></a>

<!-- PROJECT LOGO -->

<div align="center">

  <h1 align="center">MIOFlow</h1>

<!-- PROJECT SHIELDS -->
[![arXiv](https://img.shields.io/badge/arXiv-2206.14928-b31b1b.svg)](https://arxiv.org/abs/2206.14928)
[![Latest PyPI version](https://img.shields.io/pypi/v/mioflow.svg)](https://pypi.org/project/mioflow/)
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![GitHub Stars](https://img.shields.io/github/stars/KrishnaswamyLab/ImmunoStruct.svg?style=social\&label=Stars)](https://github.com/KrishnaswamyLab/ImmunoStruct)
</div>



**MIOFlow** is a Python package for modeling and analyzing single-cell RNA-seq data using **optimal flows**. It leverages **neural ordinary differential equations (neural ODEs)** and **optimal transport** to reconstruct cell developmental trajectories from time-series scRNA-seq data.

## Features

- **Trajectory inference** using optimal transport and neural ODEs
- **GAGA embedding** — geometric autoencoder that preserves PHATE distances in latent space
- **Gene-space decoding** — map trajectories back to full gene expression via PCA inverse projection
- **Flexible I/O** for AnnData and standard scRNA-seq formats

## Installation

### Install from PyPI

```bash
pip install MIOFlow
```

### Install from GitHub (Development Version)

```bash
pip install git+https://github.com/yourusername/MIOFlow.git
```

## Usage

### Basic Workflow

```python
from MIOFlow.gaga import Autoencoder, train_gaga_two_phase, train_valid_loader_from_pc
from MIOFlow.mioflow import MIOFlow

# 1. Train a GAGA autoencoder on PCA coordinates + PHATE distances
gaga_model = Autoencoder(input_dim=50, latent_dim=10)
train_loader, val_loader = train_valid_loader_from_pc(X_pca, D_phate, batch_size=64)
train_gaga_two_phase(gaga_model, train_loader, encoder_epochs=50, decoder_epochs=50)

# 2. Fit MIOFlow
mf = MIOFlow(
    adata,
    gaga_model=gaga_model,
    gaga_input_scaler=scaler,   # StandardScaler fitted on X_pca
    obs_time_key='day',
    n_epochs=200,
)
mf.fit()

# 3. Inspect results
print(mf.trajectories.shape)          # (n_bins, n_trajectories, latent_dim)
gene_traj = mf.decode_to_gene_space() # (n_bins, n_trajectories, n_genes)
```

Full worked examples are in the [tutorials/](tutorials/) directory.


## Citation

If you use **MIOFlow** in your research, please cite:

```
@misc{https://doi.org/10.48550/arxiv.2206.14928,
  doi = {10.48550/ARXIV.2206.14928},
  url = {https://arxiv.org/abs/2206.14928},
  author = {Huguet,  Guillaume and Magruder,  D. S. and Tong,  Alexander and Fasina,  Oluwadamilola and Kuchroo,  Manik and Wolf,  Guy and Krishnaswamy,  Smita},
  keywords = {Machine Learning (cs.LG),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {Manifold Interpolating Optimal-Transport Flows for Trajectory Inference},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual,  non-exclusive license}
}
```

## License

MIOFlow is distributed under the terms of the Yale License.

## Support

- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/MIOFlow/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/MIOFlow/discussions)

## Acknowledgments

- Built with PyTorch and [torchdiffeq](https://github.com/rtqichen/torchdiffeq) for neural ODE integration
- Integrates with the scanpy / AnnData ecosystem for single-cell analysis
- Optimal transport via the [POT](https://pythonot.github.io/) library
- Geometric embedding via [PHATE](https://github.com/KrishnaswamyLab/PHATE)