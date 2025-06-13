__all__ = ['plot_losses', 'plot_comparision', 'new_plot_comparisions', 'plot_gene_trends']

# Standard library imports
import os
import math

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Local imports
from .utils import to_np, get_times_from_groups
from .constants import IMGS_DIR

# Set seaborn color palette
sns.color_palette("bright")

def plot_losses(local_losses=None, batch_losses=None, globe_losses=None, save=False, path=IMGS_DIR, file='losses.png'):
    if not os.path.isdir(path):
        os.makedirs(path)

    s = 1
    fig = plt.figure(figsize=(14/s, 6/s), dpi=300)
    cur = 0
    sub = sum([1 for e in [local_losses, batch_losses, globe_losses] if e is not None])
    
    if local_losses is not None:
        cur += 1
        ax = fig.add_subplot(1,sub,cur)

        df_lloss = pd.DataFrame(local_losses)

        ax.set_title('Loss per step')
        sns.lineplot(data=df_lloss, ax=ax)
        
    
    if batch_losses is not None:
        cur += 1        
        ax = fig.add_subplot(1,sub,cur)
        
        df_bloss = pd.DataFrame(batch_losses, columns=['Batch Loss'])        
        
        ax.set_title('Batch loss per batch')        
        sns.lineplot(data=df_bloss, lw=1, ax=ax)
        
    if globe_losses is not None:
        cur += 1
        ax = fig.add_subplot(1,sub,cur)
        
        df_gloss = pd.DataFrame(globe_losses, columns=['Global Loss'])        

        ax.set_title('Global Loss per batch');
        sns.lineplot(data=df_gloss, ax=ax)
    
    if save:
        # NOTE: savefig complains image is too large but saves it anyway. 
        try:
            fig.savefig(os.path.expanduser(os.path.join(path, file)))
        except ValueError:
            pass
    plt.close()
    return fig

def plot_comparision(
    df, generated, trajectories,
    palette = 'viridis', df_time_key='samples',
    save=False, path=IMGS_DIR, file='comparision.png',
    x='d1', y='d2', z='d3', is_3d=False
):
    if not os.path.isdir(path):
        os.makedirs(path)

    if not is_3d:
        return new_plot_comparisions(
            df, generated, trajectories,
            palette=palette, df_time_key=df_time_key,
            x=x, y=y, z=z, is_3d=is_3d,            
            groups=None,
            save=save, path=path, file=file,
        )

    s = 1
    fig = plt.figure(figsize=(12/s, 8/s), dpi=300)
    if is_3d:
        ax = fig.add_subplot(1,1,1,projection='3d')
    else:
        ax = fig.add_subplot(1,1,1)
    
    states = sorted(df[df_time_key].unique())
    
    if is_3d:
        ax.scatter(
            df[x], df[y], df[z],
            cmap=palette, alpha=0.3,
            c=df[df_time_key], 
            s=df[df_time_key], 
            marker='X',
        )
    else:
        sns.scatterplot(
            data=df, x=x, y=y, palette=palette, alpha=0.3,
            hue=df_time_key, style=df_time_key, size=df_time_key,
            markers={g: 'X' for g in states},
            sizes={g: 100 for g in states}, 
            ax=ax, legend=False
        )
    

    if not isinstance(generated, np.ndarray):
        generated = to_np(generated)
    points = np.concatenate(generated, axis=0)
    n_gen = int(points.shape[0] / len(states))
    colors = [state for state in states for i in range(n_gen)]
    
    if is_3d:
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            cmap=palette,
            c=colors, 
        )
    else:
        sns.scatterplot(
            x=points[:, 0], y=points[:, 1], palette=palette,
            hue=colors, 
            ax=ax, legend=False
        )
    
    ax.legend(title='Timepoint', loc='upper left', labels=['Ground Truth', 'Predicted'])
    ax.set_title('ODE Points compared to Ground Truth')

    if is_3d:
        for trajectory in np.transpose(trajectories, axes=(1,0,2)):
            plt.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], alpha=0.1, color='Black');
    else:
        for trajectory in np.transpose(trajectories, axes=(1,0,2)):
            plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.1, color='Black');
        
    if save:
        # NOTE: savefig complains image is too large but saves it anyway. 
        try:
            fig.savefig(os.path.expanduser(os.path.join(path, file)))
        except ValueError:
            pass 
    plt.close()
    return fig

def new_plot_comparisions(
    df, generated, trajectories,
    palette = 'viridis',
    df_time_key='samples',
    x='d1', y='d2', z='d3', 
    groups=None,
    save=False, path=IMGS_DIR, file='comparision.png',
    is_3d=False
):
    if groups is None:
        groups = sorted(df[df_time_key].unique())
    cmap = plt.cm.viridis
    sns.set_palette(palette)
    plt.rcParams.update({
        'axes.prop_cycle': plt.cycler(color=cmap(np.linspace(0, 1, len(groups) + 1))),
        'axes.axisbelow': False,
        'axes.edgecolor': 'lightgrey',
        'axes.facecolor': 'None',
        'axes.grid': False,
        'axes.labelcolor': 'dimgrey',
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.facecolor': 'white',
        'lines.solid_capstyle': 'round',
        'patch.edgecolor': 'w',
        'patch.force_edgecolor': True,
        'text.color': 'dimgrey',
        'xtick.bottom': False,
        'xtick.color': 'dimgrey',
        'xtick.direction': 'out',
        'xtick.top': False,
        'ytick.color': 'dimgrey',
        'ytick.direction': 'out',
        'ytick.left': False,
        'ytick.right': False, 
        'font.size':12, 
        'axes.titlesize':10,
        'axes.labelsize':12
    })

    n_cols = 1
    n_rols = 1

    grid_figsize = [12, 8]
    dpi = 300
    grid_figsize = (grid_figsize[0] * n_cols, grid_figsize[1] * n_rols)
    fig = plt.figure(None, grid_figsize, dpi=dpi)

    hspace = 0.3
    wspace = None
    gspec = plt.GridSpec(n_rols, n_cols, fig, hspace=hspace, wspace=wspace)

    outline_width = (0.3, 0.05)
    size = 300
    bg_width, gap_width = outline_width
    point = np.sqrt(size)

    gap_size = (point + (point * gap_width) * 2) ** 2
    bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2

    plt.legend(frameon=False)

    is_3d = False
    
    # if is_3d:        
    #     ax = fig.add_subplot(1,1,1,projection='3d')
    # else:
    #     ax = fig.add_subplot(1,1,1)
    
    axs = []
    for i, gs in enumerate(gspec):        
        ax = plt.subplot(gs)
        
        
        n = 0.3   
        ax.scatter(
                df[x], df[y],
                c=df[df_time_key],
                s=size,
                alpha=0.7 * n,
                marker='X',
                linewidths=0,
                edgecolors=None,
                cmap=cmap
            )
        
        for trajectory in np.transpose(trajectories, axes=(1,0,2)):
                plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.3, color='Black');
        
        states = sorted(df[df_time_key].unique())
        points = np.concatenate(generated, axis=0)
        n_gen = int(points.shape[0] / len(states))
        colors = [state for state in states for i in range(n_gen)]
        n = 1
        o = '.'
        ax.scatter(
                points[:, 0], points[:, 1],
                c='black',
                s=bg_size,
                alpha=1 * n,
                marker=o,
                linewidths=0,
                edgecolors=None
            )
        ax.scatter(
                points[:, 0], points[:, 1],
                c='white',
                s=gap_size,
                alpha=1 * n,
                marker=o,
                linewidths=0,
                edgecolors=None
            )
        pnts = ax.scatter(
                points[:, 0], points[:, 1],
                c=colors,
                s=size,
                alpha=0.7 * n,
                marker=o,
                linewidths=0,
                edgecolors=None,
                cmap=cmap
            )
                
        legend_elements = [        
            Line2D(
                [0], [0], marker='o', 
                color=cmap((i) / (len(states)-1)), label=f'T{state}', 
                markerfacecolor=cmap((i) / (len(states)-1)), markersize=15,
            )
            for i, state in enumerate(states)
        ]
        
        leg = plt.legend(handles=legend_elements, loc='upper left')
        ax.add_artist(leg)
        
        legend_elements = [        
            Line2D(
                [0], [0], marker='X', color='w', 
                label='Ground Truth', markerfacecolor=cmap(0), markersize=15, alpha=0.3
            ),
            Line2D([0], [0], marker='o', color='w', label='Predicted', markerfacecolor=cmap(.999), markersize=15),
            Line2D([0], [0], color='black', lw=2, label='Trajectory')
            
        ]
        leg = plt.legend(handles=legend_elements, loc='upper right')
        ax.add_artist(leg)
        
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        kwargs = dict(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.tick_params(which="both", **kwargs)
        ax.set_frame_on(False)
        ax.patch.set_alpha(0)
        

        axs.append(ax)

    if save:
        # NOTE: savefig complains image is too large but saves it anyway. 
        try:
            fig.savefig(os.path.expanduser(os.path.join(path, file)))
        except ValueError:
            pass 
    plt.close()
    return fig


def plot_gene_trends(
    genes, top_idxs, inverse, colors, 
    samples=None, groups=None,
    n_cols=None, n_rows=None, where='end', start=0, top_n=None,
    cell_types=None, use_cell_types=True, 
    save:bool=False, path:str=IMGS_DIR, file:str='gene_trends.png'
):
    '''
    Notes:
        - first four arguments `genes`, `top_idxs`, `inverse`, and `colors` are output from `get_cell_indexes`
            see that function's docstring for more information.
    
    Arguments:
    ----------
        genes (np.ndarray): List of genes similar to those the user passed into this function except
            in order of the columns of the provided `df`. Any genes not found in the `df` put passed in
            by the user will be removed.
            
        top_idxs (dict | dict[dict]): See notes. Dictionary or nested dictionary where leaf values are
            indicies of cells corresponding to those expressing the highest amount of specified genes.
            
        inverse (np.nddary): Reconstructed gene space from `trajectories` and `principal_components`.
            It has the shape (n_time * n_bins, n_cells, n_genes). See `generate_tjnet_trajectories`.
        
        colors (dict): Dictionary of either `{gene: color}` or `{cell_type: color}` depending on `use_cell_types`.
    
        samples (np.ndarray | list): List of timepoints where each value corresponds to the 
            timepoint of the same row in `df`. Defaults to `None`.
        
        groups (np.ndarray | list): List of time groups in order (e.g. `[0, 1, 2, 3, 4, 5, 6, 7]`).
            Defaults to `None`.
            
        n_cols (int): How many columns to use. Defaults to `None`. If `n_cols == n_rows == None`
            then `n_cols = 4` and `n_rows` is calculated dynamically.
        
        n_rows (int): How many rows to use. Defaults to `None`. If `n_cols == n_rows == None`
            then `n_cols = 4` and `n_rows` is calculated dynamically.
        
        where (str): Choices are `"start"`, and `"end"`. Defaults to `"end"`. Whether or not
            the trajectories used to calculate `top_idxs` start at 
            `t_0` (`"start"`) or `t_n` (`"end"`). 

        start (int): Defaults to `0`. Where in `generate_tjnet_trajectories` the trajectories started.
            This is used if attempting to generate outside of `t0`. Note this works relative to `where`.
            E.g. if `where="end"` and `start=0` then this is the same as `groups[-1]`.
            
        top_n (int): Defaults to `None`. Used for legend. If not provided will be wrangled from `top_idxs`.
        
        cell_types (np.ndarray | list): List of cell types to plot. Should be outer keys from `top_idxs`.
            This can be a subset of those keys to plot less.
        
        use_cell_types (bool): Whether or not to use cell types.

        save (bool): Whether or not to save the file. Defaults to `False`.

        path (str): Path to the directory in which to save the file. Defaults to `IMGS_DIR`.

        file (str): Name of the file to be saved in directory specified by `path`. Defaults
            to `"gene_trends.png"`
        
    Returns:
    ----------
        fig (matplotlib.pyplot.figure)
    '''
    
    # Test for valid start location    
    _valid_where = 'start end'.split()
    if where not in _valid_where:
        raise ValueError(f'{where} not known. Should be one of {_valid_where}')
        
    if groups is None and samples is None:
        raise ValueError(f'`groups == samples == None`. `groups` or `samples` must not be `None`.')
        
    # Try to figure out top_n if not provided
    if top_n is None:
        top_n = int(np.round(np.mean(list(map(len, top_idxs.values())))))
    
    # Get groups from samples
    if groups is None:
        groups = sorted(np.unique(samples))

    # NOTE: update to handle generation other than t0
    # times = groups
    # if where == 'end':
    #     times = times[::-1]
    # times = times[start:]
    # groups = times
    groups = get_times_from_groups(groups, where, start)

    # Figure out columns / rows from number of genes
    n_genes = len(genes)
    if n_cols is None and n_rows is None:
        n_cols = 4
        n_rows = np.ceil(n_genes / n_cols).astype(int)
    elif n_cols is None:
        n_cols = np.ceil(n_genes / n_rows).astype(int)
    elif n_rows is None:
        n_rows = np.ceil(n_genes / n_cols).astype(int)
    else:
        pass
    
    # BEGIN: plot options
    grid_figsize = [4, 3]
    
    dpi = 300
    grid_figsize = (grid_figsize[0] * n_cols, grid_figsize[1] * n_rows)
    fig = plt.figure(None, grid_figsize, dpi=dpi)
    
    hspace = 0.3
    wspace = None
    gspec = plt.GridSpec(n_rows, n_cols, fig, hspace=hspace, wspace=wspace)
        
    plt.legend(frameon=False)
    # END: plot options
    
    
    axs = []
    for i, gs in enumerate(gspec):
        # Grid may have more subplots than genes to plot
        if i >= len(genes):
            break
                 
        ax = plt.subplot(gs)
        
        # current gene subplot
        gene = genes[i]
                
        ax.set_title(gene)
        ax.set_yticks([])
        ax.set_xticks(range(np.min(groups), np.max(groups)))
        
        # x-axis 0 ---> n or n ---> 0 depending on `where`
        x_axis = np.linspace(np.min(groups), np.max(groups), inverse.shape[0])
        x_axis = x_axis if where == 'start' else x_axis[::-1]
                    
        if use_cell_types:
            for j, (ctype, gexpr) in enumerate(top_idxs.items()):
                if ctype not in cell_types:
                    continue                
                for k, cell_idx in enumerate(gexpr[gene]):    
                    # Plot top cells of given cell type expressing given gene
                    ax.plot(x_axis, inverse[:, cell_idx, i], c=colors[ctype], label=f'{ctype}_{gene}')    
                    
        else:
            for j, eg in enumerate(genes):
                for cell in top_idxs[eg]:     
                    # Plot top cells expressing any specified genes, expressing given gene
                    ax.plot(x_axis, inverse[:, cell, i], c=colors[eg], label=gene)

    
    if use_cell_types:
        fig.legend(
            handles=[
                mpatches.Patch(color=colors[ctype], label=ctype)
                for ctype in cell_types
            ], 
            title=f'Cell Types'
        )        
    else:
        fig.legend(
            handles=[
                mpatches.Patch(color=colors[gene], label=gene)
                for gene in genes
            ], 
            title=f'Top {top_n} Cells Expressing Genes', 
            bbox_to_anchor=(0, 0, 0, 0.5)
        )

    if save:
        # NOTE: savefig complains image is too large but saves it anyway. 
        try:
            fig.savefig(os.path.expanduser(os.path.join(path, file)))
        except ValueError:
            pass         
        
    plt.close()
    return fig
