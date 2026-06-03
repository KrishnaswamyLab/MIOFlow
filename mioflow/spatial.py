# Spatial feature extraction for MIOFlow 2.0

from typing import Optional, List
import torch
import numpy as np
import pandas as pd
import phate
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph as knn
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import matrix_power as sparse_matpow
from mioflow.gaga import fit_gaga

# TO DO: add spatial features
# robustness for 3D data, key mismatches, edge cases (and add typecasting)
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

    n_cells = adata.n_obs
    coords = np.array(adata.obsm[coords_key])
    edge_index = _build_graph(coords, k, n_hops, d_max)

    feature_blocks = []

    # FEATURE 1: local expression niche (mean PCA embedding of neighbours)
    X_raw = adata.X
    if sp.issparse(X_raw):
        X_raw = X_raw.toarray()

    # determines max possible PCA components
    n_pca_actual = min(n_pca_niche, X_raw.shape[1], X_raw.shape[0] - 1)
    X_pca_niche = PCA(n_components=n_pca_actual).fit_transform(X_raw)
    niche_feats = _mean_aggregate(X_pca_niche, edge_index)

    feature_blocks.append(niche_feats)

    # FEATURE 2: neighbourhood cell-type composition
    if celltype_key is not None:
        cell_types, type_idx = np.unique(adata.obs[celltype_key], return_inverse=True)
        n_types = len(cell_types)

        one_hot = np.zeros((n_cells, n_types))
        one_hot[np.arange(n_cells), type_idx] = 1

        type_feats = _sum_aggregate(one_hot, edge_index)
        feature_blocks.append(type_feats)

    # TO ADD: feature 3 (ligand-receptor signalling)

    # Concatenate and normalise features
    S_raw = np.concatenate(feature_blocks, axis=1) 
    S_raw = StandardScaler().fit_transform(S_raw)

    # determines max possible PCA components based on data dimensions
    n_out = min(n_spatial_pca, S_raw.shape[1], S_raw.shape[0] - 1) 
    S = PCA(n_components=n_out).fit_transform(S_raw)

    adata.obsm[store_key] = S
    return S

# TO DO: adjust default hyperparameters (chosen somewhat arbitrarily)
def fit_spatial_gaga(
    adata,
    spatial_key: str = 'X_spatial',
    latent_dim: int = 2,
    hidden_dims: List[int] = [64, 32],
    batch_size: int = 512,
    encoder_epochs: int = 100,
    decoder_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Train a PHATE-regularised GAGA autoencoder on spatial features.

    Mirrors ``fit_gaga()`` from ``gaga.py`` but operates on the spatial
    feature matrix produced by ``compute_spatial_features()``.

    Parameters
    ----------
    adata : AnnData
        Must have ``adata.obsm[spatial_key]``.
    spatial_key : str
        Key in ``adata.obsm`` holding the spatial feature matrix.
    latent_dim : int
        GAGA latent space dimensionality.
    hidden_dims : list of int, optional
        Hidden layer sizes.
    batch_size : int
        Number of cells per training batch.
    encoder_epochs : int
        Phase 1 epochs (distance preservation).
    decoder_epochs : int
        Phase 2 epochs (reconstruction).
    learning_rate : float
        Adam learning rate.
    device : str
        Torch device string.

    Returns
    -------
    Autoencoder with ``model.input_scaler``.
    """

    if spatial_key not in adata.obsm:
        raise ValueError(f"'{spatial_key}' not found in adata.obsm. Run compute_spatial_features().")

    X_spatial = np.array(adata.obsm[spatial_key])

    print("Computing PHATE on spatial features...")
    phate_op = phate.PHATE(n_components=latent_dim, verbose=True)
    X_phate = phate_op.fit_transform(X_spatial)

    return fit_gaga(
        X_pca=X_spatial,
        X_phate=X_phate,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        batch_size=batch_size,
        encoder_epochs=encoder_epochs,
        decoder_epochs=decoder_epochs,
        learning_rate=learning_rate,
        device=device,
    )

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

    # edge_index is (n_edges, 2) with rows [source, target]
    src, dst = edge_index[:, 0], edge_index[:, 1]

    out = np.zeros_like(X)
    counts = np.zeros(n)

    # for each edge, add the source node's features to the target node's output and increment the count for the target node
    np.add.at(out, src, X[dst])
    np.add.at(counts, src, 1)

    counts = np.maximum(counts, 1)  # prevent division by zero

    return out / counts[:, None] # think this works? should be broadcasting counts for each feature dimension

# sum-aggregate X over neighbours defined by edge_index
def _sum_aggregate(X: np.ndarray, edge_index: np.ndarray) -> np.ndarray:

    src, dst = edge_index[:, 0], edge_index[:, 1]

    # for each edge, add the source node's features to the target node's output
    out = np.zeros_like(X)
    np.add.at(out, src, X[dst])

    return out