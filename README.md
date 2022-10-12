# MIOFlow
> MIOFlow Project


## Currently TOC is broken, temp fix:
- [`MIOFlow.ode`](https://KrishnaswamyLab.github.io/MIOFlow/ode)
- [`MIOFlow.losses`](https://KrishnaswamyLab.github.io/MIOFlow/losses)
- [`MIOFlow.utils`](https://KrishnaswamyLab.github.io/MIOFlow/utils)
- [`MIOFlow.models`](https://KrishnaswamyLab.github.io/MIOFlow/models)
- [`MIOFlow.plots`](https://KrishnaswamyLab.github.io/MIOFlow/plots)
- [`MIOFlow.train`](https://KrishnaswamyLab.github.io/MIOFlow/train)
- [`MIOFlow.constants`](https://KrishnaswamyLab.github.io/MIOFlow/constants)
- [`MIOFlow.datasets`](https://KrishnaswamyLab.github.io/MIOFlow/datasets)
- [`MIOFlow.exp`](https://KrishnaswamyLab.github.io/MIOFlow/exp)
- [`MIOFlow.geo`](https://KrishnaswamyLab.github.io/MIOFlow/geo)
- [`MIOFlow.eval`](https://KrishnaswamyLab.github.io/MIOFlow/eval)

## Setup

To get all the pagackes required, run the following command:

```bash
$ conda env create -f environment.yml
```

This will create a new conda environment `sklab-mioflow`, which can be activated via:

```
conda activate sklab-mioflow
```

### Add kernel to Jupyter Notebook

#### automatic conda kernels
For greater detail see the official docs for [`nb_conda_kernels`][nb_conda_kernels].
In short, install `nb_conda_kernels` in the environment from which you launch JupyterLab / Jupyter Notebooks from (e.g. `base`) via:

```bash
$ conda install -n <notebook_env> nb_conda_kernels
```

to add a new or exist conda environment to Jupyter simply install `ipykernel` into that conda environment e.g.

```bash
$ conda install -n <python_env> ipykernel
```


#### manual ipykernel
add to your Jupyter Notebook kernels via

```bash
$ python -m ipykernel install --user --name sklab-mioflow
```

It can be removed via:

```bash
$ jupyter kernelspec uninstall sklab-mioflow
```

#### list kernels found by Jupyter

kernels recognized by conda
```bash
$ python -m nb_conda_kernels list
```

check which kernels are discovered by Jupyter:
```bash
$ jupyter kernelspec list
```

[nb_conda_kernels]: https://github.com/Anaconda-Platform/nb_conda_kernels

## Install

### For developers and internal use:
```
cd path/to/this/repository
pip install -e MIOFlow
```

### For production use:
`pip install MIOFlow`

## How to use

This repository consists of our python library `MIOFlow` as well as a directory of scripts for running and using it. 

### Scripts

To recreate our results with MMD loss and density regulariazation you can run the following command:

```bash
python scripts/run.py -d petals -c mmd -n petal-mmd
```

This will generate the directory `results/petals-mmd` and save everything there.

For a full list of parameters try running:

```bash
python scripts/run.py --help
```

### Python Package
One could simply import everything and use it piecemeal:

```python
from MIOFlow.ode import *
from MIOFlow.losses import *
from MIOFlow.utils import *
from MIOFlow.models import *
from MIOFlow.plots import *
from MIOFlow.train import *
from MIOFlow.constants import *
from MIOFlow.datasets import *
from MIOFlow.exp import *
from MIOFlow.geo import *
from MIOFlow.eval import *
```

### Tutorials
One can also consult or modify the tutorial notebooks for their uses:
- [EB Bodies tutorial][ebbodies]
- [Dyngen tutorial][dyngen]
- [Petals tutorial][petals]

[ebbodies]: https://github.com/dsm-72/MIOFlow/blob/main/notebooks/EB-Bodies.ipynb
[dyngen]: https://github.com/dsm-72/MIOFlow/blob/main/notebooks/Dyngen.ipynb
[petals]: https://github.com/dsm-72/MIOFlow/blob/main/notebooks/%5BMMD%5D%5BTEST%5D%5BDiamond%20Petals%5D%20Trajectory%20Net.ipynb
