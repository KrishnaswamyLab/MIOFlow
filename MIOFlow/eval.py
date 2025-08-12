
__all__ = ['generate_points', 'generate_trajectories', 'generate_plot_data', 'get_points_from_trajectories', 'calculate_nn',
           'generate_tjnet_trajectories', 'get_cell_indexes']

import torch
import logging
import sklearn
import pandas as pd
import numpy as np
from typing import Union
from typing import Literal

from MIOFlow.utils import (
    generate_steps, sample, to_np,
    to_np, get_groups_from_df, get_cell_types_from_df, 
    get_sample_n_from_df, get_times_from_groups
)
import seaborn as sns

def generate_points(
    model, df, n_points=100, 
    sample_with_replacement=False, use_cuda=False, 
    samples_key='samples', sample_time=None, autoencoder=None, recon=False
):
    '''
    Arguments:
    ----------
        model (torch.nn.Module): Trained network with the property `ode` corresponding to a `NeuralODE(ODEF())`.
            See `MIOFlow.ode` for more.
        df (pd.DataFrame): DataFrame containing a column for the timepoint samples and the rest of the data.
        n_points (int): Number of points to generate.
        sample_with_replacement (bool): Defaults to `False`. Whether or not to use replacement when sampling
            initial timepoint.
        use_cuda (bool): Defaults to `False`. Whether or not to use cuda.
        samples_key (str): Defaults to `'samples'`. The column in the `df` which has the timepoint groups.
        sample_time (list | None): Defaults to `None`. If `None` uses the group numbers in order as the 
            timepoints as specified in the column `df[samples_key]`.
        autoencoder (nn.Module|NoneType): Default to None, the trained autoencoder.
        recon (bool): Default to 'False', whether to use the autoencoder for reconstruction.
    Returns:
    ----------
        generated (float[float[]]): a list with shape `(len(sample_time), n_points, len(df.columns) - 1)`
            of the generated points.
    '''
    to_torch = True #if use_cuda else False

    groups = sorted(df[samples_key].unique())
    if sample_time is None:
        sample_time = groups
    data_t0 = sample(
        df, np.min(groups), size=(n_points, ), 
        replace=sample_with_replacement, to_torch=to_torch, use_cuda=use_cuda
    )
    if autoencoder is not None and recon:
        data_t0 = torch.Tensor(data_t0).float()
        data_t0 = autoencoder.encoder(data_t0)
        
    time =  torch.Tensor(sample_time).cuda() if use_cuda else torch.Tensor(sample_time)
    generated = model(data_t0, time, return_whole_sequence=True)
    if autoencoder is not None and recon:
        generated = autoencoder.decoder(generated)
    return to_np(generated)

def generate_trajectories(
    model, df, n_trajectories=30, n_bins=100, 
    sample_with_replacement=False, use_cuda=False, samples_key='samples',autoencoder=None, recon=False
):
    '''
    Arguments:
    ----------
        model (torch.nn.Module): Trained network with the property `ode` corresponding to a `NeuralODE(ODEF())`.
            See `MIOFlow.ode` for more.
        df (pd.DataFrame): DataFrame containing a column for the timepoint samples and the rest of the data.
        n_trajectories (int): Number of trajectories to generate.
        n_bins (int): Number of bins to use for the trajectories. More makes it smoother. Defaults to `100`.
        sample_with_replacement (bool): Defaults to `False`. Whether or not to use replacement when sampling
            initial timepoint.
        use_cuda (bool): Defaults to `False`. Whether or not to use cuda.
        samples_key (str): Defaults to `'samples'`. The column in the `df` which has the timepoint groups.
        autoencoder (nn.Module|NoneType): Default to None, the trained autoencoder.
        recon (bool): Default to 'False', whether to use the autoencoder for reconstruction.
    Returns:
    ----------
        trajectories (float[float[]]): a list with shape `(n_bins, n_points, len(df.columns) - 1)`
            of the generated trajectories.
    '''
    groups = sorted(df[samples_key].unique())
    sample_time = np.linspace(np.min(groups), np.max(groups), n_bins)
    trajectories = generate_points(model, df, n_trajectories, sample_with_replacement, use_cuda, samples_key, sample_time,autoencoder=autoencoder, recon=recon)
    return trajectories
    
def generate_plot_data(
    model, df, n_points, n_trajectories, n_bins, 
    sample_with_replacement=False, use_cuda=False, samples_key='samples',
    logger=None, autoencoder=None, recon=False
):
    '''
    Arguments:
    ----------
        model (torch.nn.Module): Trained network with the property `ode` corresponding to a `NeuralODE(ODEF())`.
            See `MIOFlow.ode` for more.
        df (pd.DataFrame): DataFrame containing a column for the timepoint samples and the rest of the data.
        n_points (int): Number of points to generate.
        n_trajectories (int): Number of trajectories to generate.
        n_bins (int): Number of bins to use for the trajectories. More makes it smoother. Defaults to `100`.
        sample_with_replacement (bool): Defaults to `False`. Whether or not to use replacement when sampling
            initial timepoint.
        use_cuda (bool): Defaults to `False`. Whether or not to use cuda.
        samples_key (str): Defaults to `'samples'`. The column in the `df` which has the timepoint groups.
        autoencoder (nn.Module|NoneType): Default to None, the trained autoencoder.
        recon (bool): Default to 'False', whether to use the autoencoder for reconstruction.
    Returns:
    ----------
        points (float[float[]]): a list with shape `(len(df[sample_key].unique()), n_points, len(df.columns) - 1)`
            of the generated points.
        trajectories (float[float[]]): a list with shape `(n_bins, n_points, len(df.columns) - 1)`
            of the generated trajectories.
    '''
    # TODO: Correct the logger if logger: logger.info(f'Generating points')
    points = generate_points(model, df, n_points, sample_with_replacement, use_cuda, samples_key, None, autoencoder=autoencoder, recon=recon)
    # TODO: Correct the logger if logger: logger.info(f'Generating trajectories')
    trajectories = generate_trajectories(model, df, n_trajectories, n_bins, sample_with_replacement, use_cuda, samples_key, autoencoder=autoencoder, recon=recon)
    return points, trajectories


def get_points_from_trajectories(
    n_groups:int, 
    trajectories:Union[np.ndarray, list], 
    how:Union[Literal['start'], Literal['middle'], Literal['end']]='start', 
    logger:logging.Logger=None
) -> np.ndarray:
    '''
    Arguments:
        n_groups (int): how many time points there are total. 
        trajectories (np.ndarray | list): A list with three dimensions that correspond to:
            `(n_bins, n_points, n_dims)`, where `n_points` are the number of points / trajectories there are,
            `n_bins` correponds to how the trajectories were smoothed via binning (e.g. if there were 5 total
            time points one might have 100 bins to draw a smoother line when plotting), and `n_dims` are the
            number of dimensions the points are in (e.g. 2).
        how (str): Defaults to `'start'`. How to extract the point for the binned trajectories.
            If `'start'` takes the first point in the time window. If `'middle'` takes the floored averaged.
            If `'end'` takes the the last point in the time window.
        logger (logging.Logger): Defaults to `None`.
    Returns:
        points (np.ndarray): Points at the corresponding indicices. If `trajectories` has shape:
            `(n_bins, n_points, n_dims)`, this will have shape `(n_groups, n_points, n_dims)`.
    '''
    # Value handling
    _valid_how = 'start middle end'.split()
    if how not in _valid_how:
        raise ValueError(f'Unknown option specified for `how`. Must be in {_valid_how}')    
        
    if not isinstance(trajectories, np.ndarray):
        trajectories = np.array(trajectories)
        
    trajectories = np.transpose(trajectories, axes=(1, 0, 2))
    (n_points, n_bins, n_dims) = trajectories.shape
    if logger: 
        logger.info(
            f'Given trajectories object with {n_points} points of {n_bins} '
            f'bins in {n_dims} dimensions.'
        )
    
    parts = np.linspace(0, n_bins, n_groups + 1).astype(int).tolist()
    steps = generate_steps(parts)
    results = []
    for step in steps:
        time_window = trajectories[:, slice(*step)] 
        if how == 'start':
            results.append(time_window[:, 0, :])
        elif how == 'middle':
            idx = int(np.floor(n_bins / n_groups / 2))
            results.append(time_window[:, idx, :])
        elif how == 'end':
            results.append(time_window[:, -1, :])
        else:
            raise NotImplementedError(f'how={how} not implemented')        
    return np.array(results)


def calculate_nn(
    df:pd.DataFrame,
    generated:Union[np.ndarray, list]=None,
    trajectories:Union[np.ndarray, list]=None,
    compare_to:Union[Literal['time'], Literal['any']]='time',
    how:Union[Literal['start'], Literal['middle'], Literal['end']]='start',     
    k:int=1,
    groups:Union[np.ndarray, list]=None,
    sample_key:str='samples',
    method:str='mean',
    logger:logging.Logger=None
) -> float:
    '''
    Arguments:
        df (pd.DataFrame): DataFrame containing all points and time sample specified by `sample_key`
        generated (np.ndarray | list): A list of the generate points with shape 
            `(n_groups, n_points, n_dims)`, where `n_groups` is the total number of time indicies
            as specified in `groups`.
        trajectories (np.ndarray | list): A list with three dimensions that correspond to:
            `(n_bins, n_points, n_dims)`, where `n_points` are the number of points / trajectories there are,
            `n_bins` correponds to how the trajectories were smoothed via binning (e.g. if there were 5 total
            time points one might have 100 bins to draw a smoother line when plotting), and `n_dims` are the
            number of dimensions the points are in (e.g. 2).
        compare_to (str): Defaults to `'time'`. Determines points to use for KNN. If `'time'` will only 
            consider points at the same time index. If `'any'` will search for nearest points regardless
            of time.
        how (str): Defaults to `'start'`. How to extract the point for the binned trajectories.
            If `'start'` takes the first point in the time window. If `'middle'` takes the floored averaged.
            If `'end'` takes the the last point in the time window.
        k (int): Defaults to `1`. Number of points to compare predicted points in `generated` 
            or `trajectories` to.
        groups (np.ndarray | list): Defaults to `None` and will be extracted from `df` is not provided. 
            The sorted unique values of time samples from `df` as specified by `sample_key`.
        sample_key (str): Defaults to `'samples'`. The column in `df` which corresponds to the time index.
        method (str): Defaults to `'mean'`. If `'mean'` returns the mean of the knn distances. If `'quartile'`
            returns the mean of the worst (highest distances) quartile.
        logger (logging.Logger): Defaults to `None`.
    Returns:
        mean_dist (float): mean distance of predicted points to the `n` nearest-neighbor points.
    '''
    _valid_compare_to = 'time any'.split()
    if compare_to not in _valid_compare_to:
        raise ValueError(f'Unknown option specified for `compare_to`. Must be in {_valid_compare_to}') 
        
    _valid_how = 'start middle end'.split()
    if how not in _valid_how:
        raise ValueError(f'Unknown option specified for `how`. Must be in {_valid_how}')

    _valid_method = 'mean quartile'.split()
    if method not in _valid_method:
        raise ValueError(f'Unknown option specified for `method`. Must be in {_valid_method}')
        
    if trajectories is None and generated is None:
        raise ValueError(f'Either generated or trajectories must not be None!')
        
    if groups is None:
        groups = sorted(df[sample_key].unique())
    
    if generated is None:
        generated = get_points_from_trajectories(len(groups), trajectories, how, logger)
    
    distances = []    
    for idx, time_sample in enumerate(sorted(groups)):            
        pred_points = generated[idx]
        
        # NOTE: compare to points only at same time index
        if compare_to == 'time':
            true_points = df.groupby(sample_key).get_group(time_sample).drop(columns=sample_key).values
        # NOTE: compare to any point
        elif compare_to == 'any':
            true_points = df.drop(columns=sample_key).values
        else:            
            raise NotImplementedError(f'compare_to={compare_to} not implemented')
        true_points = true_points[:, :pred_points.shape[1]]
        neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=k)
        neigh.fit(true_points)
        dists, indicies = neigh.kneighbors(pred_points, return_distance=True)
        distances.extend(dists.flatten().tolist())
    
    distances = np.array(distances)
    if method == 'mean':      
        return distances.mean()
    elif method == 'quartile':
        q1 = np.quantile(distances, 0.25)
        q2 = np.quantile(distances, 0.50)
        q3 = np.quantile(distances, 0.75)
        
        b1 = distances[np.where(distances < q1)]
        b2 = distances[np.where((distances < q2) & (distances >= q1))]
        b3 = distances[np.where((distances < q3) & (distances >= q2))]
        b4 = distances[np.where(distances >= q3)]

        return np.max([np.mean(b) for b in [b1, b2, b3, b4]])
    else:
        raise NotImplementedError(f'method={method} not implemented')


def generate_tjnet_trajectories(
    model, df, n_bins=10, use_cuda=False, samples_key='samples', 
    autoencoder=None, recon=False, where='end', start=0
):
    '''
    Arguments:
    -----------
        model (nn.Module): Trained MIOFlow model.

        df (pd.DataFrame): DataFrame of shape (n_cells, dimensions + 1), where the extra column
            stems from a samples column (column indicating the timepoint of the cell). 
            By default the samples column is assumed to be `"samples"`.
        
        n_bins (int): For each time point split it into `n_bins` for smoother trajectories.
            If there are `t` time points then there will be `t * n_bins` total points.
        
        use_cuda (bool): Whether or not to use cuda for the model and autoencoder.
        
        samples_key (str): The name of the column in the `df` that corresponds to the time
            samples. Defaults to `"samples"`. 
        
        autoencoder (nn.Module) Trained Geodesic Autoencoder.
        
        recon (bool): Whether or not to use the `autoencoder` to reconstruct the output
            space from the `model`.
        
        where (str): Choices are `"start"`, and `"end"`. Defaults to `"end"`. Whether or not
            to start the trajectories at `t_0` (`"start"`) or `t_n` (`"end"`). 
    
        start (int): Defaults to `0`. Where in `generate_tjnet_trajectories` the trajectories started.
            This is used if attempting to generate outside of `t0`. Note this works relative to `where`.
            E.g. if `where="end"` and `start=0` then this is the same as `groups[-1]`.

    Returns:
    -----------
        trajectories (np.ndarray): Trajectories with shape (time, cells, dimensions)
    '''
    
    _valid_where = 'start end'.split()
    if where not in _valid_where:
        raise ValueError(f'{where} not known. Should be one of {_valid_where}')
    
    groups = sorted(df[samples_key].unique())
    
    # times = groups
    # if where == 'end':
    #     times = times[::-1]
    # times = times[start:]
    
    times = get_times_from_groups(groups, where, start)

    # a, b = (np.max(groups), np.min(groups)) if where == 'end' else (np.min(groups), np.max(groups))    
    a, b = (np.max(times), np.min(times)) if where == 'end' else (np.min(times), np.max(times))    
    # n = -1 if where == 'end' else 0
    n = 0 # because reversed if needed and prunned
    
    # sample_time = np.linspace(a, b, len(groups) * n_bins)
    sample_time = np.linspace(a, b, len(times) * n_bins)
    sample_time = torch.Tensor(sample_time)
    
    # data_tn = df[df.samples == groups[n]].drop(columns=samples_key).values    
    data_tn = df[df.samples == times[n]].drop(columns=samples_key).values    
    data_tn = torch.Tensor(data_tn).float()
    
    if use_cuda:
        data_tn = data_tn.cuda()
        sample_time = sample_time.cuda()
            
    if autoencoder is not None and recon:
        data_tn = autoencoder.encoder(data_tn)

    generated = model(data_tn, sample_time, return_whole_sequence=True)
    if autoencoder is not None and recon:
        generated = autoencoder.decoder(generated)
    generated = to_np(generated)
    return generated


def get_cell_indexes(
    df, genes, trajectories, principal_components,
    top_n=10, where='end', start=0, palette = 'viridis', 
    samples_key='samples',  samples=None,
    cell_type_key=None, cell_types=None, use_cell_types=True
):
    '''
    Notes:
    -----------
        - `samples` refers to the timepoint sample e.g. `samples == 1` should be Boolean array
            corresponding to which rows in `df` that are of `t_1`.
            
        - `use_cell_types` determines the output shape of `top_idxs`. 
        
            + `use_cell_types=True`: `top_idxs` is a nested dictionary with structure
                ```
                    {
                        cell_type_0: {
                            gene_0: [id_0, id_1, ... id_n]
                        },
                        ...
                        cell_type_m: {...},
                    }
                ```
                Where each id is a cell of the outer cell type with the highest expression of
                the specified gene either at `t_0` (`where="start"`) or `t_n` (`where="end"`)
                e.g. cell_type_0[gene_0][0] is the id of the top cell of cell type `cell_type_0`
                expressing gene `gene_0`.
                
            + `use_cell_types=False`: `top_idxs` is a dictionary with structure
                ```
                    {                       
                        gene_0: [id_0, id_1, ... id_n],
                        gene_1: [id_0, id_1, ... id_n],
                        ...
                        gene_m: [id_0, id_1, ... id_n],                        
                    }
                ```
                Where each id is a cell (of any cell type) that has the highest expression of the
                specified gene either at `t_0` (`where="start"`) or `t_n` (`where="end"`).
    
    Arguments:
    -----------
        model (nn.Module): Trained MIOFlow model.

        df (pd.DataFrame): DataFrame of shape (n_cells, n_genes), where the ordering of 
            the columns `n_genes` corresponds to the columns of `principle_components`.
            It is assumed that the index of `df` are the cell types (but this need not be the case. 
            See `cell_types`). If there are additional columns (e.g. `samples_key`, `cell_type_key`)
            should be after the gene columns.
            
        genes (np.ndarray | list): Genes of interest to determine which cell indexes to find.
        
        trajectories (np.ndarray): Trajectories with shape (time, cells, dimensions)
        
        principal_components (np.ndarray): The principle components with shape (dimensions, n_genes).
            If used phate, can be obtained from `phate_operator.graph.data_pca.components_`
        
        top_n (int): Defaults to `10`. The number of cells to use per condition. If 
            `use_cell_types = False` this (conditions) will be the number of genes (`len(genes)`)
            otherwise it will be the number of cell types.
        
        where (str): Choices are `"start"`, and `"end"`. Defaults to `"end"`. Whether or not
            the trajectories start at `t_0` (`"start"`) or `t_n` (`"end"`). 

        start (int): Defaults to `0`. Where in `generate_tjnet_trajectories` the trajectories started.
            This is used if attempting to generate outside of `t0`. Note this works relative to `where`.
            E.g. if `where="end"` and `start=0` then this is the same as `groups[-1]`.
        
        palette (str): A Matplotlib colormap. Defaults to `"viridis"`. 
        
        samples_key (str): The name of the column in the `df` that corresponds to the time
            samples. Defaults to `"samples"`. If `df[samples_key]` throws a `KeyError` 
            either because the `df` doesnt have this column in it or typo, will resort to
            `samples` to determine this.
                        
        samples (np.ndarray | list): List of timepoints where each value corresponds to the 
            timepoint of the same row in `df`. Defaults to `None`.
        
        cell_type_key (str): The column name in the provided DataFrame `df` the corresponds to the 
            cell's cell types. Defaults to `None` which assumes the cell type is the index of the 
            `df i.e. `df.index`
        
        cell_types (np.ndarray | list): List of cell types to use from the provided DataFrame `df`.
            Defaults to `None`. If `use_cell_types = True` will attempt to figure this out from
            `cell_type_key`.
        
        use_cell_types (bool): Whether or not to use cell types.
    
    Returns:
    -----------
        genes (np.ndarray): List of genes similar to those the user passed into this function except
            in order of the columns of the provided `df`. Any genes not found in the `df` put passed in
            by the user will be removed.
            
        top_idxs (dict | dict[dict]): See notes. Dictionary or nested dictionary where leaf values are
            indicies of cells corresponding to those expressing the highest amount of specified genes.
            
        inverse (np.nddary): Reconstructed gene space from `trajectories` and `principal_components`.
            It has the shape (n_time * n_bins, n_cells, n_genes). See `generate_tjnet_trajectories`.
        
        colors (dict): Dictionary of either `{gene: color}` or `{cell_type: color}` depending on `use_cell_types`.
    '''
    # Test for valid start location
    _valid_where = 'start end'.split()
    if where not in _valid_where:
        raise ValueError(f'{where} not known. Should be one of {_valid_where}')
        
    groups = get_groups_from_df(df, samples_key, samples)

    # times = groups
    # if where == 'end':
    #     times = times[::-1]
    # times = times[start:]

    times = get_times_from_groups(groups, where, start)
    
    # Extract all of the cells at the specified either the start / end
    n = -1 if where == 'end' else 0
    # counts_n = get_sample_n_from_df(df, n, samples_key, samples, groups, drop_index=False)
    counts_n = get_sample_n_from_df(df, 0, samples_key, samples, groups=times, drop_index=False)
              
    # Filter for only known genes
    genes_mask = df.columns.isin(genes)
    # Get genes in order
    genes = df.columns[genes_mask]
        
    # Reconstruct full gene space (of just the genes we care about) 
    # from trajectories and principal components
    inverse =  np.dot(trajectories, principal_components[:, genes_mask])
                        
    if use_cell_types:
        # Try to correct for missing cell types if they are required
        cell_types = get_cell_types_from_df(df, cell_type_key, cell_types)
                
        # Get name of cell_type_key column              
        index = counts_n.columns[0] if cell_type_key is None else cell_type_key
            
        # Create colors for each cell type
        cmap = sns.color_palette(palette, n_colors=len(cell_types))
        colors = dict(zip(*[
            cell_types,
            [cmap[i] for i in range(len(cell_types))]
        ]))
        
        # For each cell type and each gene get top_n cells of that cell type expressing that gene
        top_idxs = {}
        for cell_type in cell_types:
            cells = counts_n[counts_n[index] == cell_type] 
            top_idxs[cell_type] = {}
            for gene in genes:
                top_idx = cells[gene].values.flatten().argsort()[-(top_n):]
                top_idxs[cell_type][gene] = top_idx
        
        
    else:
        # Create colors for each gene
        cmap = sns.color_palette(palette, n_colors=len(genes))
        colors = dict(zip(*[
            genes,
            [cmap[i] for i in range(len(genes))]
        ]))    
            
        # For each gene, get top_n cells expressing that gene    
        top_idxs = {}
        for gene in genes:
            top_idx = counts_n[gene].values.flatten().argsort()[-(top_n):]
            top_idxs[gene] = top_idx
            
        
    return genes, top_idxs, inverse, colors
