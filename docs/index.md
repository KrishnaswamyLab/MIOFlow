# MIOFlow

[![arXiv](https://img.shields.io/badge/arXiv-2206.14928-b31b1b.svg)](https://arxiv.org/abs/2206.14928)
[![Latest PyPI version](https://img.shields.io/pypi/v/mioflow.svg)](https://pypi.org/project/mioflow/)

**MIOFlow** is a Python package for modeling and analyzing single-cell RNA-seq data using **optimal flows**. It leverages **neural ordinary differential equations (neural ODEs)** and **optimal transport** to reconstruct trajectories, compare cell populations, and study dynamic biological processes.

## Features

- **Trajectory inference** using optimal transport and neural ODEs
- **Comparison across conditions** (e.g., control vs. perturbation)
- **Visualization utilities** for single-cell dynamics
- **Flexible I/O** for AnnData and standard scRNA-seq formats

## Quick Start

```bash
pip install MIOFlow
```

See the [Installation](installation.md) page for full setup instructions, or jump straight into the [Tutorials](tutorials.md).

## Citation

If you use MIOFlow in your research, please cite:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2206.14928,
  doi = {10.48550/ARXIV.2206.14928},
  url = {https://arxiv.org/abs/2206.14928},
  author = {Huguet, Guillaume and Magruder, D. S. and Tong, Alexander and Fasina, Oluwadamilola and Kuchroo, Manik and Wolf, Guy and Krishnaswamy, Smita},
  title = {Manifold Interpolating Optimal-Transport Flows for Trajectory Inference},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Acknowledgments

- Built with PyTorch for neural ODE implementations
- Integrates with the scanpy ecosystem for single-cell analysis
- Optimal transport based on the [POT](https://pythonot.github.io/) library
