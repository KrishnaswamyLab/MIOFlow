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



**MIOFlow** is a Python package for modeling and analyzing single-cell RNA-seq data using **optimal flows**. It leverages **neural ordinary differential equations (neural ODEs)** and **optimal transport** to reconstruct trajectories, compare cell populations, and study dynamic biological processes.

## Features

- **Trajectory inference** using optimal transport and neural ODEs
- **Comparison across conditions** (e.g., control vs. perturbation)  
- **Visualization utilities** for single-cell dynamics
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

A basic workflow can be found on the tutorials. There is a Google Colab option as well.

tutorials/1_MIOFlow_Example.ipynb or tutorials/2_Colab_Training_MIOFlow


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

- Built with PyTorch for neural ODE implementations
- Integrates with scanpy ecosystem for single-cell analysis
- Optimal transport implementations based on POT library