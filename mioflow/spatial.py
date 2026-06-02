# Spatial feature extraction for MIOFlow 2.0

from typing import Optional, List
import numpy as np
import pandas as pd
import phate
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph as knn
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import matrix_power as sparse_matpow

# TO DO: add spatial features
# robustness for 3D data
# add batch correction for multi-sample datasets?

def compute_spatial_features(
    adata,
    coords_key: str = 'spatial',
    k: int = 5,
    n_hops: int = 3,
    d_max: Optional[float] = None, # 500?
    lr_pairs: Optional[str] = None,
    lr_expr_key: Optional[str] = None,
    celltype_key: Optional[str] = None,
    n_pca_niche: int = 50, # consider tuning these
    n_spatial_pca: int = 100,
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

    The concatenated features are z-normalised and dimensionally reduced with PCA
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
        Key in ``adata.obsm`` holding a (n_cells, n_genes) array or DataFrame.
        Column names must match the values in ``lr_pairs``.
    celltype_key : str, optional
        Column in ``adata.obs`` with cell-type labels.
    n_pca_niche : int
        PCA components used to summarise gene expression before neighbourhood averaging.
    n_spatial_pca : int
        Output dimensionality after final PCA reduction.
    store_key : str
        Key where result is stored in ``adata.obsm``.

    Returns
    -------
    np.ndarray of shape (n_cells, n_spatial_pca), also stored in ``adata.obsm[store_key]``.
    """

    if coords_key not in adata.obsm:
        raise ValueError(f"coords_key='{coords_key}' not found in adata.obsm")

    n_cells = adata.n_obs
    coords = np.array(adata.obsm[coords_key])
    edge_index = _build_graph(coords, k, n_hops, d_max)

# HELPERS

# builds a symmetric kNN graph, returns edge_index of shape (n_edges, 2)
# could be more robust (more bounds checking ie min(k, n-1), n<=1)
def _build_graph(
    coords: np.ndarray,
    k: int,
    n_hops: int,
    d_max: Optional[float],
) -> np.ndarray:

    # build weighted kNN graph based on spatial coordinates
    G = knn(coords, k, mode='distance')

    # eliminate edges exceeding d_max if specified
    if d_max is not None:
        G = G.multiply(G <= d_max)

    # binarise and symmetrise the graph
    G = (G + G.T > 0)
    G.setdiag(0)
    G.eliminate_zeros()

    # expand neighbourhood
    if n_hops > 1:
        G = sparse_matpow(G, n_hops)
        G = (G > 0)
        G.setdiag(0)
        G.eliminate_zeros()

    # returns edge_index of shape (n_edges, 2)
    rows, cols = G.nonzero()
    return np.stack([rows, cols], axis=1)

# mean-aggregate X over neighbours defined by edge_index
# could add more robust checks
def _mean_aggregate(X: np.ndarray, edge_index: np.ndarray) -> np.ndarray:

    n = X.shape[0]

    src, dst = edge_index[:, 0], edge_index[:, 1]

    out = np.zeros_like(X)
    counts = np.zeros(n)

    np.add.at(out, src, X[dst])
    np.add.at(counts, src, 1)

    counts = np.maximum(counts, 1)  # prevent division by zero

    return out / counts[:, None] # think this works? should be broadcasting counts for each feature dimension