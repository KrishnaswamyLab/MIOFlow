# Spatial feature extraction for MIOFlow 2.0.

from typing import Optional, List
import numpy as np
import pandas as pd
import phate
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import matrix_power as sparse_matpow

# TO DO: add spatial features
# robustness for 3D data

def compute_spatial_features(
    adata,
    coords_key: str = 'spatial',
    k: int = 5,
    n_hops: int = 3,
    d_max: Optional[float] = None, # 500?
    lr_pairs: Optional[str] = None,
    lr_expr_key: Optional[str] = None, # add option for dataframe of imputed LR expression values rather than arr?
    celltype_key: Optional[str] = None,
    n_pca_niche: int = 50,
    n_spatial_pca: int = 100,
    batch_key: Optional[str] = None,
    store_key: str = 'X_spatial',
) -> np.ndarray:
    """
    Extract spatial neighbourhood features for each cell.
    
    Computes up to three feature types and concatenates them:

    1. Local expression niche — mean PCA embedding of spatial neighbours.
    2. Neighbourhood composition — cell-type frequency vector of neighbours.
    3. Ligand-receptor signalling — for each known LR pair, the product of
       the target cell's receptor expression and the mean ligand expression of
       its neighbours.

    The concatenated features are z-normalised and dimenionally reduced with PCA
    to produce the final spatial embedding.

    Parameters
    ----------
    adata : AnnData
    coords_key : str
        Key in ``adata.obsm`` holding (x, y) spatial coordinates.
    k : int
        Number of nearest spatial neighbours for the base kNN graph.
    n_hops : int
        Neighbourhood radius.
    d_max : float, optional
        Maximum allowed spatial distance.
    lr_pairs : str, optional
        Path to a CSV file with columns ``'Ligand'`` and ``'Receptor'``.
        Gene names must match the column names of ``adata.obsm[lr_expr_key]``.
    lr_expr_key : str, optional
        Key in ``adata.obsm`` holding a (n_cells, n_genes) array.
        Column names must match the values in ``lr_pairs``.
    celltype_key : str, optional
        Column in ``adata.obs`` with cell-type labels.
    n_pca_niche : int
        PCA components used to summarise gene expression before neighbourhood averaging.
    n_spatial_pca : int
        Output dimensionality after final PCA reduction.
    batch_key : str, optional
        Column in ``adata.obs``. If provided, graph is built independently
        per batch so cells from different batches aren't connected.
    store_key : str
        Key where result is stored in ``adata.obsm``.

    Returns
    -------
    np.ndarray of shape (n_cells, n_spatial_pca) and stored in ``adata.obsm[store_key]``.
    """
    