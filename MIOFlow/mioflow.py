__all__ = ['MIOFlow']

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from typing import Optional, Union, Dict, Any, List, Tuple
import logging
from pathlib import Path

# Import your training function (adjust import path as needed)
# from your_training_module import training_regimen, config_criterion

from MIOFlow.utils import generate_steps, set_seeds, config_criterion
from MIOFlow.models import make_model, Autoencoder
from MIOFlow.plots import plot_comparision, plot_losses
from MIOFlow.train import train_ae, training_regimen

from MIOFlow.geo import setup_distance
from MIOFlow.exp import setup_exp
from MIOFlow.eval import generate_points, generate_trajectories

from MIOFlow.phate_autoencoder import PhateAutoencoder

class MIOFlow:
    """
    MIOFlow: Manifold Interpolating Optimal-Transport Flows for Trajectory Inference.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing single-cell expression data
    obsm_key : str, optional (default: "X_phate")
        Key in adata.obsm containing embedding coordinates (e.g., UMAP, PCA)
    start_node : str or int, optional
        Starting cell type or cluster for trajectory inference
    debug_level : str, optional (default: 'info')
        Logging level for debugging ('verbose', 'info', 'warning', 'error')
    
    Training Configuration Parameters
    --------------------------------
    n_local_epochs : int, optional (default: 10)
        Number of local training epochs
    n_epochs : int, optional (default: 100)
        Number of main training epochs  
    n_post_local_epochs : int, optional (default: 10)
        Number of post-local training epochs
    exp_dir : str, optional
        Experiment output directory
    criterion_type : str, optional (default: 'mse')
        Type of loss criterion
    use_cuda : bool, optional (default: True)
        Whether to use CUDA if available
    hold_one_out : bool, optional (default: False)
        Whether to use hold-one-out validation
    sample_size : int, optional (default: 1000)
        Sample size for training
    reverse_schema : bool, optional (default: False)
        Whether to use reverse schema
    reverse_n : int, optional (default: 0)
        Number of reverse samples
    use_density_loss : bool, optional (default: False)
        Whether to use density loss
    lambda_density : float, optional (default: 1.0)
        Lambda parameter for density loss
    autoencoder : bool, optional (default: False)
        Whether to use autoencoder
    use_emb : bool, optional (default: True)
        Whether to use embeddings
    use_gae : bool, optional (default: False)
        Whether to use graph autoencoder
    plot_every : int, optional (default: 10)
        Plotting frequency during training
    n_points : int, optional (default: 1000)
        Number of points for visualization
    n_trajectories : int, optional (default: 100)
        Number of trajectories to generate
    n_bins : int, optional (default: 50)
        Number of bins for binning
    **kwargs : dict
        Additional parameters
    """
    
    def __init__(
        self,
        adata,
        input_df = None,
        obsm_key: str = "X_phate",
        start_node: Optional[Union[str, int]] = None,
        debug_level: str = 'info',
        
        # Training structure parameters
        n_local_epochs: int = 10,
        n_epochs: int = 100,
        n_post_local_epochs: int = 10,
        
        # Output configuration
        exp_dir: Optional[str] = None,
        
        # Optimization parameters
        criterion_type: str = 'mse',
        use_cuda: bool = True,
        
        # Data handling parameters
        hold_one_out: bool = False,
        sample_size: int = 1000,
        reverse_schema: bool = False,
        reverse_n: int = 0,
        
        # Loss configuration
        use_density_loss: bool = False,
        lambda_density: float = 1.0,
        
        # Advanced features
        autoencoder: bool = False,
        use_emb: bool = True,
        use_gae: bool = False,
        
        # Visualization and output
        plot_every: int = 10,
        n_points: int = 1000,
        n_trajectories: int = 100,
        n_bins: int = 50,

        **kwargs
    ):
        # Store input parameters
        self.adata = adata.copy() if hasattr(adata, 'copy') else adata

        if input_df is None:
            self._prepare_data()
        else:
            self.df = self._prepare_df(input_df)

        self.obsm_key = obsm_key
        self.start_node = start_node
        self.debug_level = debug_level
        
        # Configure training structure
        self.training_structure = {
            'n_local_epochs': n_local_epochs,
            'n_epochs': n_epochs,
            'n_post_local_epochs': n_post_local_epochs
        }
        
        # Configure output settings
        self.output_config = {
            'exp_dir': exp_dir or f"mioflow_output_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            'plot_every': plot_every,
            'n_points': n_points,
            'n_trajectories': n_trajectories,
            'n_bins': n_bins
        }
        
        # Configure model settings
        self.model_config = {
            'layers': [16, 32, 16],
            'activation': 'CELU', 
            'scales': None,
            'use_cuda': use_cuda and torch.cuda.is_available()
        }
        
        # Configure optimization
        self.optimization_config = {
            'criterion_type': criterion_type,
            'use_density_loss': use_density_loss,
            'lambda_density': lambda_density
        }
        
        # Configure data handling
        self.data_config = {
            'hold_one_out': hold_one_out,
            'sample_size': sample_size,
            'reverse_schema': reverse_schema,
            'reverse_n': reverse_n
        }
        
        # Configure advanced features
        self.advanced_config = {
            'autoencoder': autoencoder,
            'use_emb': use_emb,
            'use_gae': use_gae
        }
        
        # Store additional parameters
        self.kwargs = kwargs
        
        # Initialize state variables
        self.is_fitted = False
        self.model = None
        self.trajectories = {}
        self.pseudotime = None
        self.min_count = None
        
        # Set up logging
        self._setup_logging()
        
        # Validate inputs
        self._validate_inputs()
        
        self.logger.info(f"MIOFlow initialized with {self.adata.n_obs} cells and {self.adata.n_vars} genes")
        self.logger.info(f"Output directory: {self.output_config['exp_dir']}")
    
    def _setup_logging(self):
        """Set up logging based on debug level."""
        level_map = {
            'verbose': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR
        }
        
        logging.basicConfig(
            level=level_map.get(self.debug_level, logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('MIOFlow')
    
    def _validate_inputs(self):
        """Validate input parameters and data."""
        
        # Check if obsm_key exists in adata.obsm
        if self.obsm_key not in self.adata.obsm.keys():
            raise ValueError(f"Embedding key '{self.obsm_key}' not found in adata.obsm")
        
        # Create output directory
        Path(self.output_config['exp_dir']).mkdir(parents=True, exist_ok=True)
        
        self.logger.debug("Input validation completed successfully")

    def _prepare_df(self, input_df):
        """Normalize the dataframe, save the variables nencessary for decoding"""
        # Get embedding columns dynamically
        embed_cols = [col for col in input_df.columns if col.startswith('d') and col[1:].isdigit()] # Retrieve the columns that start with a d
        embedding = input_df[embed_cols].values

        self.mean_vals = np.mean(embedding, axis=0)

        std_vals = np.std(embedding, axis=0)
        self.std_vals = np.where(std_vals == 0, 1, std_vals)  # Prevent division by zero
        
        normalized = (embedding - self.mean_vals) / std_vals
        # We create the dataframe with the normalized data and the samples
        output_df = pd.DataFrame(normalized, columns=[f'd{i+1}' for i in range(normalized.shape[1])])
        output_df['samples'] = input_df['samples']
        return output_df


    def _prepare_data(self):
        """Prepare data in the format expected by training_regimen."""
        self.logger.debug("Preparing data for training")
        
        # Create a numerical bin for each unique value
        #TODO: change day to be an external variable
        self.adata.obs['discrete_time'], _ = pd.factorize(self.adata.obs['day'])
        
        # Now lets create the data structure that mioflows works on top
        self.df = pd.DataFrame(
            self.adata.obsm['X_phate'], 
            columns=[f'd{i}' for i in range(1, self.adata.obsm['X_phate'].shape[1]+1)]
        )
        
        # Add the time labels to the dataframe as a column called 'samples' (this is expected by MIOFlow)
        self.df['samples'] = self.adata.obs['discrete_time'].values
        
        # Calculate min_count for sample size constraints
        sample_counts = self.df['samples'].value_counts()
        self.min_count = int(sample_counts.min())
        
        # Update configs with min_count constraints
        self.data_config['sample_size'] = min(self.min_count, self.data_config['sample_size'])
        self.output_config['n_points'] = min(self.min_count, self.output_config['n_points'])
        self.output_config['n_trajectories'] = min(self.min_count, self.output_config['n_trajectories'])
        
        self.logger.debug(f"Data prepared: {self.df.shape}, min_count: {self.min_count}")

    def _initialize_model(self):
        """Initialize the model for training."""
    
        
        self.model = make_model(
            feature_dims=(len(self.df.columns) - 1),  # Input dimensions (excluding 'samples' column)
            layers=self.model_config['layers'],
            activation=self.model_config['activation'],
            scales=self.model_config['scales'],
            use_cuda=self.model_config['use_cuda'],
        )
    
    def fit(
        self,
        debug_axes: Optional[Any] = None,
        **fit_kwargs
    ):
        """
        Fit the MIOFlow trajectory inference model.
        
        Parameters
        ----------
        debug_axes : matplotlib.axes.Axes, optional
            Axes for plotting debug information
        **fit_kwargs : dict
            Additional fitting parameters that override initialization settings
        
        Returns
        -------
        self : MIOFlow
            Returns self for method chaining
        """
        self.logger.info("Starting MIOFlow fitting")
        
        # #TODO: Verify this code. Update configs with any fit_kwargs
        # for config_dict in [self.training_structure, self.output_config, self.model_config, 
        #                    self.optimization_config, self.data_config, self.advanced_config]:
        #     for key, value in fit_kwargs.items():
        #         if key in config_dict:
        #             config_dict[key] = value
        
        # Initialize model
        self._initialize_model()
        
        # Prepare optimizer and criterion
        optimizer = torch.optim.AdamW(self.model.parameters())
        criterion = config_criterion(self.optimization_config['criterion_type'])
        
        self.logger.info(f"Training with structure: {self.training_structure}")
        self.logger.info(f"Using CUDA: {self.model_config['use_cuda']}")

        try:
            # Compute the phate autoencoder
            print(f"Training PHATE Autoencoder with {self.adata.obsm['X_phate'].shape[0]} cells and {self.adata.obsm['X_phate'].shape[1]} dimensions")

            self.phate_autoencoder = PhateAutoencoder.train(self.adata.obsm['X_phate'], 
                                                            self.adata.obsm['X_pca'], 
                                                            None, 
                                                            self.adata.uns['pca']['variance_ratio'], 
                                                            save_dir=self.output_config['exp_dir'],
                                                            train_reducer=False)
        except Exception as e:
            self.logger.error(f"Phate Autoencoder Failed: {str(e)}")
            raise

        try:
            print(f"Training MIOFlow trajectory inference model")

            # Call the training_regimen function
            self.local_losses, self.batch_losses, self.globe_losses = training_regimen(
                # Training structure
                n_local_epochs=self.training_structure['n_local_epochs'],
                n_epochs=self.training_structure['n_epochs'],
                n_post_local_epochs=self.training_structure['n_post_local_epochs'],
                
                # Output
                exp_dir=self.output_config['exp_dir'],
                
                # Core training parameters
                model=self.model,
                df=self.df,
                groups=sorted(self.df.samples.unique()),
                
                # Optimization
                optimizer=optimizer,
                criterion=criterion,
                use_cuda=self.model_config['use_cuda'],
                
                # Data handling
                hold_one_out=self.data_config['hold_one_out'],
                sample_size=(self.data_config['sample_size'],),
                reverse_schema=self.data_config['reverse_schema'],
                reverse_n=self.data_config['reverse_n'],
                
                # Loss configuration
                use_density_loss=self.optimization_config['use_density_loss'],
                lambda_density=self.optimization_config['lambda_density'],
                
                # Advanced features
                autoencoder=self.advanced_config['autoencoder'],
                use_emb=self.advanced_config['use_emb'],
                use_gae=self.advanced_config['use_gae'],
                
                # Visualization and output
                plot_every=self.output_config['plot_every'],
                n_points=self.output_config['n_points'],
                n_trajectories=self.output_config['n_trajectories'],
                n_bins=self.output_config['n_bins'],
                
                # Logger
                logger=self.logger,
            )
            
            # # After training, extract results
            self._extract_results()
            
            # Mark as fitted
            self.is_fitted = True
            
            self.logger.info("MIOFlow fitting completed successfully")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
        
        return self
    
    def _extract_results(self):
        """Extract results from the training process."""
        self.logger.debug("Extracting training results")
        
        # TODO: Correct the logger if logger: logger.info(f'Generating points')
        #TODO: verify the parameters passed to each of this functions
        self.points = generate_points(self.model, 
                                      self.df,
                                      n_points=self.output_config['n_points'],
                                      use_cuda=self.model_config['use_cuda'],
                                      sample_with_replacement=False,
                                      samples_key='samples',
                                      sample_time=None, 
                                      autoencoder=None, 
                                      recon=False
                                      )
        
        # TODO: Correct the logger if logger: logger.info(f'Generating trajectories')
        self.trajectories = generate_trajectories(self.model, 
                                                  self.df,
                                                  n_trajectories=self.output_config['n_trajectories'],
                                                  n_bins=self.output_config['n_bins'],
                                                  sample_with_replacement=False,
                                                  use_cuda=self.model_config['use_cuda'],
                                                  samples_key='samples',
                                                  autoencoder=self.advanced_config['autoencoder'],
                                                  recon=False)
    
    def decode_to_gene_space(self) -> np.ndarray:
        """
        Decode trajectory points to gene space.
        
        Parameters
        ----------
        
        Returns
        -------
        np.ndarray
            Decoded trajectory points in gene space
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before decoding. Call fit() first.")

        # We need to account for the std and mean in the trajectories.
        denormalized_trajectories = self.trajectories * self.std_vals + self.mean_vals

        traj_shapes = denormalized_trajectories.shape
        traj_flat = denormalized_trajectories.reshape(-1, denormalized_trajectories.shape[-1])

        traj_pca = self.phate_autoencoder.phate2pca(traj_flat)

        # # Here traj_pca is a 2D array with shape (n_time_points,n_trajectories, n_components)
        traj_pca = traj_pca.reshape(traj_shapes[0], traj_shapes[1], -1)
        # Here we compute the decoded trajectories in the original space
        # traj_pca is a 3D array with shape (n_time_points, n_trajectories, n_components)
        # We reshape it to have shape (n_time_points * n_trajectories, n_components)

        trajectories_all_points = traj_pca.reshape(-1, traj_pca.shape[-1])
        X_reconstructed = np.array((trajectories_all_points @ self.adata.varm['PCs'].T) + self.adata.X.mean(axis=0))

        #X_reconstructed have every point in our trajectory back in gene space
        self.trajectories_gene_space = X_reconstructed.reshape(denormalized_trajectories.shape[0], denormalized_trajectories.shape[1], -1)

        return self.trajectories_gene_space
    
    def plot_trajectories(self, **plot_kwargs):
        """
        Plot the inferred trajectories.
        
        Parameters
        ----------
        **plot_kwargs : dict
            Additional plotting parameters
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting. Call fit() first.")
        
        # TODO: Implement trajectory plotting
        self.logger.info("Plotting trajectories")
        
        pass
    
    def get_pseudotime(self) -> np.ndarray:
        """
        Get pseudotime values for all cells.
        
        Returns
        -------
        np.ndarray
            Pseudotime values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting pseudotime. Call fit() first.")
        
        return self.pseudotime
    
    def get_trajectories(self) -> Dict[str, Any]:
        """
        Get fitted trajectory information.
        
        Returns
        -------
        dict
            Dictionary containing trajectory information
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting trajectories. Call fit() first.")
        
        return self.trajectories
    
    def get_config(self) -> Dict[str, Dict]:
        """
        Get all configuration parameters.
        
        Returns
        -------
        dict
            Dictionary containing all configuration dictionaries
        """
        return {
            'training_structure': self.training_structure,
            'output_config': self.output_config,
            'model_config': self.model_config,
            'optimization_config': self.optimization_config,
            'data_config': self.data_config,
            'advanced_config': self.advanced_config
        }
    
    def update_config(self, config_dict: Dict[str, Any]):
        """
        Update configuration parameters.
        
        Parameters
        ----------
        config_dict : dict
            Dictionary of configuration updates
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                getattr(self, key).update(value)
            else:
                self.logger.warning(f"Unknown config key: {key}")
    
    def save(self, filepath: str):
        """
        Save the fitted MIOFlow model.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving.")
        
        # TODO: Implement model saving
        self.logger.info(f"Saving model to {filepath}")
        pass
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load a saved MIOFlow model.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model
        
        Returns
        -------
        MIOFlow
            Loaded MIOFlow instance
        """
        # TODO: Implement model loading
        pass
    
    def __repr__(self):
        """String representation of MIOFlow object."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        return (f"MIOFlow(n_obs={self.adata.n_obs}, n_vars={self.adata.n_vars}, "
                f"obsm_key='{self.obsm_key}', "
                f"n_epochs={self.training_structure['n_epochs']}, status={fitted_status})")