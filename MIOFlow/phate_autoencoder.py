import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from omegaconf import DictConfig
import shutil
import copy
import pathlib

import torch.nn as nn
import matplotlib.pyplot as plt

from .phate_decoder import Decoder


def train(x: np.ndarray, y: np.ndarray, weights: np.ndarray, cfg: DictConfig, 
          metric_prefix: str = "", noise_std: float = 0.0) -> str:
    """
    Train the decoder model with the given data and configuration.
    
    Args:
        x: Input data, already scaled
        y: Target data
        weights: Optional weights for weighted loss (can be None)
        metric_prefix: Prefix for logged metrics
        noise_std: Standard deviation of noise to add to input
        
    Returns:
        str: Path to the best checkpoint
    """
    # Convert to tensors
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    
    if weights is not None:
        weights = torch.FloatTensor(weights)
        dataset = TensorDataset(x, y, weights.expand(len(x), -1))
    else:
        dataset = TensorDataset(x, y)

    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Create dataset with noise
    dataset = NoisyDataset(x, y, weights, noise_std)

    # Calculate split sizes using config values
    train_size = int(cfg.data.train_ratio * len(dataset))
    val_size = int(cfg.data.val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.training.batch_size,
        shuffle=True, 
        num_workers=cfg.training.num_workers,
        persistent_workers=True,
        drop_last=cfg.training.drop_last
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.training.batch_size,
        shuffle=False, 
        persistent_workers=True,
        num_workers=cfg.training.num_workers,
        drop_last=cfg.training.drop_last
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=cfg.training.batch_size,
        shuffle=False, 
        persistent_workers=True,
        num_workers=cfg.training.num_workers,
        drop_last=cfg.training.drop_last
    )

    # Setup logger
    if cfg.logging.logger == "tensorboard":
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(
            save_dir=cfg.logging.tensorboard_dir,
            name=cfg.logging.run_name if hasattr(cfg.logging, 'run_name') else None,
            default_hp_metric=False
        )
    else:  # "none"
        logger = False

    # Setup callbacks with prefixed monitor metric
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.logging.checkpoint_dir,
        save_top_k=cfg.logging.save_top_k,
        monitor=f"{metric_prefix}val_loss" if metric_prefix else "val_loss",
        mode=cfg.logging.monitor_mode
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping = pl.callbacks.EarlyStopping(
        monitor=f"{metric_prefix}val_loss" if metric_prefix else "val_loss",
        mode=cfg.logging.monitor_mode,
        patience=cfg.training.early_stopping_patience,
        min_delta=cfg.training.early_stopping_min_delta,
        verbose=False
    )
    callbacks.append(early_stopping)

    # Update scheduler monitor metric if using prefix
    if hasattr(cfg.model, 'scheduler'):
        if metric_prefix:
            cfg.model.scheduler.monitor = f"{metric_prefix}val_loss"

    # Create model with metric prefix for logging
    model = Decoder(
        input_dim=x.shape[1],
        output_dim=y.shape[1],
        layer_widths=cfg.model.layer_widths,
        dropout=cfg.model.dropout,
        batchnorm=cfg.model.batchnorm,
        weight_decay=cfg.model.weight_decay,
        lr=cfg.model.lr,
        weighted_loss=weights is not None,
        scheduler=cfg.model.scheduler if hasattr(cfg.model, 'scheduler') else None,
        metric_prefix=metric_prefix
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        deterministic=True,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices
    )

    trainer.fit(model, train_loader, val_loader)
    
    # Create scatter plots after training only if using a logger
    if logger:
        model._make_scatter_plot(train_loader, "Train")
        model._make_scatter_plot(val_loader, "Validation")
        model._make_scatter_plot(test_loader, "Test")

    # Return the checkpoint path
    return checkpoint_callback.best_model_path

class NoisyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, weights=None, noise_std=0.0):
        """
        Dataset that adds Gaussian noise to input data.
        
        Args:
            x: Input data
            y: Target data
            weights: Optional weights for weighted loss
            noise_std: Standard deviation of Gaussian noise to add
        """
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
        self.weights = torch.FloatTensor(weights) if weights is not None else None
        self.noise_std = noise_std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Add noise to input
        if self.noise_std > 0:
            noise = torch.randn_like(self.x[idx]) * self.noise_std
            noisy_x = self.x[idx] + noise
        else:
            noisy_x = self.x[idx]

        if self.weights is not None:
            return noisy_x, self.y[idx], self.weights.expand(len(self.x), -1)[idx]
        return noisy_x, self.y[idx]

class PhateAutoencoder:
    def __init__(self, decoder, dim_reducer, phate_scaler, phate_vis_scaler):
        """Initialize PhateAutoencoder with trained models and scalers"""
        self.decoder = decoder
        self.dim_reducer = dim_reducer
        self.phate_scaler = phate_scaler
        self.phate_vis_scaler = phate_vis_scaler
        
        # Ensure models are in eval mode
        if self.decoder is not None:
            self.decoder.eval()
        if self.dim_reducer is not None:
            self.dim_reducer.eval()

    @classmethod
    def train(cls, 
              x_phate: np.ndarray, 
              x_pca: np.ndarray = None,
              x_phate_vis: np.ndarray = None,
              weights: np.ndarray = None,
              save_dir: str = None,
              train_decoder: bool = True,
              train_reducer: bool = False,
              # Shared parameters
              device: str = None,  # Will auto-select best available
              batch_size: int = 32,
              num_workers: int = 4,
              train_ratio: float = 0.7,
              val_ratio: float = 0.15,
              logger: str = "tensorboard",
              project_name: str = "PhateAutoencoder",
              run_name: str = None,
              log_every_n_steps: int = 0,
              # Decoder parameters (larger network, longer training)
              decoder_params: dict = None,
              # Dim reducer parameters (smaller network, shorter training, stronger regularization)
              reducer_params: dict = None,
              # Noise parameters
              decoder_noise_std: float = 0.1,  # Standard deviation of noise for decoder
              reducer_noise_std: float = 0.2,  # Standard deviation of noise for reducer
              ) -> "PhateAutoencoder":
        """
        Train a new PhateAutoencoder model with optional partial training.
        
        Args:
            x_phate: PHATE embeddings (required)
            x_pca: PCA embeddings (required if train_decoder=True)
            x_phate_vis: PHATE visualization embeddings (required if train_reducer=True)
            weights: Optional weights for weighted loss
            save_dir: Directory to save/load model files (required if not training both models)
            train_decoder: Whether to train the decoder model
            train_reducer: Whether to train the dim_reducer model
            
            decoder_params: Optional dict with decoder-specific parameters:
                hidden_layers: List[int] = [32, 32]
                dropout: float = 0.1
                batchnorm: bool = True
                weight_decay: float = 0.0001
                learning_rate: float = 0.001
                max_epochs: int = 25
                early_stopping_patience: int = 10
                early_stopping_min_delta: float = 1e-4
                scheduler_type: str = "reduce_on_plateau"
                scheduler_patience: int = 5
                scheduler_factor: float = 0.5
                scheduler_min_lr: float = 1e-6
                
            reducer_params: Optional dict with dim_reducer-specific parameters:
                hidden_layers: List[int] = [8, 8]
                dropout: float = 0.5
                batchnorm: bool = True
                weight_decay: float = 0.001
                learning_rate: float = 0.001
                max_epochs: int = 15
                early_stopping_patience: int = 5
                early_stopping_min_delta: float = 1e-4
                scheduler_type: str = "reduce_on_plateau"
                scheduler_patience: int = 3
                scheduler_factor: float = 0.5
                scheduler_min_lr: float = 1e-6
            
        Returns:
            PhateAutoencoder: Model with trained/loaded components
        """
        # Input validation
        if train_decoder and x_pca is None:
            raise ValueError("x_pca is required when train_decoder=True")
        if train_reducer and x_phate_vis is None:
            raise ValueError("x_phate_vis is required when train_reducer=True")
        if not (train_decoder or train_reducer):
            raise ValueError("At least one of train_decoder or train_reducer must be True")
        if not train_decoder and not train_reducer and save_dir is None:
            raise ValueError("save_dir is required when loading pre-trained models")

        if save_dir:
            pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Initialize scalers and scale PHATE data
        phate_scaler = StandardScaler()
        x_phate_scaled = phate_scaler.fit_transform(x_phate)

        # Initialize models as None
        decoder = None
        dim_reducer = None
        phate_vis_scaler = None

        # Setup configs if needed
        if train_decoder or train_reducer:
            decoder_config, reducer_config = cls._create_configs(
                decoder_params, reducer_params, device, batch_size, num_workers,
                train_ratio, val_ratio, logger, project_name, run_name,
                log_every_n_steps, save_dir
            )

        # Train or load decoder
        if train_decoder:
            if weights is not None:
                assert weights.shape[0] == x_pca.shape[1], "Weights dimension must match PCA dimension"
                weights = weights / weights.sum()
                weights = torch.FloatTensor(weights)
            decoder_checkpoint_path = train(x_phate_scaled, x_pca, weights, decoder_config, 
                                         metric_prefix="decoder/", 
                                         noise_std=decoder_noise_std)
            decoder = Decoder.load_from_checkpoint(decoder_checkpoint_path)
            if save_dir:
                shutil.copy2(decoder_checkpoint_path, os.path.join(save_dir, 'decoder.ckpt'))
        elif save_dir:  # Load pre-trained decoder
            decoder_path = os.path.join(save_dir, 'decoder.ckpt')
            if os.path.exists(decoder_path):
                decoder = Decoder.load_from_checkpoint(decoder_path)
            else:
                print(f"Warning: Decoder checkpoint not found at {decoder_path}, setting decoder to None")
                decoder = None

        # Train or load dim_reducer
        if train_reducer:
            phate_vis_scaler = StandardScaler()
            x_phate_vis_scaled = phate_vis_scaler.fit_transform(x_phate_vis)
            dim_reducer_checkpoint_path = train(x_phate_scaled, x_phate_vis_scaled, None, reducer_config, 
                                             metric_prefix="dim_reducer/",
                                             noise_std=reducer_noise_std)
            dim_reducer = Decoder.load_from_checkpoint(dim_reducer_checkpoint_path)
            if save_dir:
                shutil.copy2(dim_reducer_checkpoint_path, os.path.join(save_dir, 'dim_reducer.ckpt'))
                joblib.dump(phate_vis_scaler, os.path.join(save_dir, 'phate_vis_scaler.pkl'))
        elif save_dir:  # Load pre-trained dim_reducer
            dim_reducer_path = os.path.join(save_dir, 'dim_reducer.ckpt')
            if os.path.exists(dim_reducer_path):
                dim_reducer = Decoder.load_from_checkpoint(dim_reducer_path)
            else:
                print(f"Warning: Dim reducer checkpoint not found at {dim_reducer_path}, setting dim_reducer to None")
                dim_reducer = None
            phate_vis_scaler_path = os.path.join(save_dir, 'phate_vis_scaler.pkl')
            if os.path.exists(phate_vis_scaler_path):
                phate_vis_scaler = joblib.load(phate_vis_scaler_path)
            else:
                print(f"Warning: PHATE visualization scaler not found at {phate_vis_scaler_path}, setting phate_vis_scaler to None")
                phate_vis_scaler = None

        # Save PHATE scaler if saving anything
        if save_dir:
            joblib.dump(phate_scaler, os.path.join(save_dir, 'phate_scaler.pkl'))

        return cls(decoder, dim_reducer, phate_scaler, phate_vis_scaler)

    @classmethod
    def load(cls, model_dir: str) -> "PhateAutoencoder":
        """
        Load a trained PhateAutoencoder model from directory.
        
        Args:
            model_dir: Directory containing all model files
            
        Returns:
            PhateAutoencoder: Loaded model
        """
        # Load models and scalers from the same directory
        if os.path.exists(os.path.join(model_dir, "decoder.ckpt")):
            decoder = Decoder.load_from_checkpoint(os.path.join(model_dir, "decoder.ckpt"))
        else:
            print(f"Warning: Decoder checkpoint not found at {os.path.join(model_dir, 'decoder.ckpt')}, setting decoder to None")
            decoder = None
        if os.path.exists(os.path.join(model_dir, "dim_reducer.ckpt")):
            dim_reducer = Decoder.load_from_checkpoint(os.path.join(model_dir, "dim_reducer.ckpt"))
        else:
            print(f"Warning: Dim reducer checkpoint not found at {os.path.join(model_dir, 'dim_reducer.ckpt')}, setting dim_reducer to None")
            dim_reducer = None
        if os.path.exists(os.path.join(model_dir, "phate_scaler.pkl")):
            phate_scaler = joblib.load(os.path.join(model_dir, "phate_scaler.pkl"))
        else:
            print(f"Warning: PHATE scaler not found at {os.path.join(model_dir, 'phate_scaler.pkl')}, setting phate_scaler to None")
            phate_scaler = None
        if os.path.exists(os.path.join(model_dir, "phate_vis_scaler.pkl")):
            phate_vis_scaler = joblib.load(os.path.join(model_dir, "phate_vis_scaler.pkl"))
        else:
            print(f"Warning: PHATE visualization scaler not found at {os.path.join(model_dir, 'phate_vis_scaler.pkl')}, setting phate_vis_scaler to None")
            phate_vis_scaler = None
        
        return cls(decoder, dim_reducer, phate_scaler, phate_vis_scaler)

    def phate2pca(self, x: np.ndarray) -> np.ndarray:
        """Convert PHATE embeddings to PCA embeddings"""
        if self.decoder is None:
            raise ValueError("Decoder is not loaded, cannot perform transformation")
        x_scaled = self.phate_scaler.transform(x)
        x_tensor = torch.FloatTensor(x_scaled).to(self.decoder.device)
        
        # Set model to eval mode and ensure no gradient computation
        self.decoder.eval()
        with torch.no_grad():
            x_pca = self.decoder(x_tensor)
        return x_pca.cpu().numpy()

    def phate2vis(self, x: np.ndarray) -> np.ndarray:
        """Convert PHATE embeddings to visualization embeddings"""
        if self.dim_reducer is None:
            raise ValueError("Dim reducer is not loaded, cannot perform transformation")
        x_scaled = self.phate_scaler.transform(x)
        x_tensor = torch.FloatTensor(x_scaled).to(self.dim_reducer.device)
        
        # Set model to eval mode and ensure no gradient computation
        self.dim_reducer.eval()
        with torch.no_grad():
            x_vis = self.dim_reducer(x_tensor)
        x_vis = x_vis.cpu().numpy()
        x_vis = self.phate_vis_scaler.inverse_transform(x_vis)
        return x_vis

    @staticmethod
    def _create_configs(decoder_params, reducer_params, device, batch_size, num_workers,
                       train_ratio, val_ratio, logger, project_name, run_name,
                       log_every_n_steps, save_dir):
        """Create configuration for decoder and reducer"""
        # Default parameters for decoder
        default_decoder_params = {
            'hidden_layers': [32, 32],
            'dropout': 0.1,
            'batchnorm': True,
            'weight_decay': 0.0001,
            'learning_rate': 0.001,
            'max_epochs': 25,
            'early_stopping_patience': 10,
            'early_stopping_min_delta': 1e-4,
            'scheduler_type': "reduce_on_plateau",
            'scheduler_patience': 5,
            'scheduler_factor': 0.5,
            'scheduler_min_lr': 1e-6
        }

        # Default parameters for dim_reducer (smaller, shorter training, stronger regularization)
        default_reducer_params = {
            'hidden_layers': [32, 32],
            'dropout': 0.1,
            'batchnorm': True,
            'weight_decay': 0.0001,
            'learning_rate': 0.001,
            'max_epochs': 25,
            'early_stopping_patience': 10,
            'early_stopping_min_delta': 1e-4,
            'scheduler_type': "reduce_on_plateau",
            'scheduler_patience': 5,
            'scheduler_factor': 0.5,
            'scheduler_min_lr': 1e-6
        }

        # Update with user-provided parameters
        decoder_params = {**default_decoder_params, **(decoder_params or {})}
        reducer_params = {**default_reducer_params, **(reducer_params or {})}

        # Auto-select device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Create base config structure
        base_config = {
            'training': {
                'batch_size': batch_size,
                'accelerator': device,
                'devices': 1,
                'num_workers': num_workers,
                'drop_last': True
            },
            'logging': {
                'logger': logger,
                'wandb_project': project_name,
                'tensorboard_dir': "tensorboard_logs",
                'checkpoint_dir': "checkpoints",
                'save_top_k': 1,
                'monitor_metric': "val_loss",
                'monitor_mode': "min",
                'log_every_n_steps': log_every_n_steps,
                'run_name': run_name
            },
            'data': {
                'train_ratio': train_ratio,
                'val_ratio': val_ratio
            },
            'save': {
                'enabled': save_dir is not None,
                'model_dir': save_dir if save_dir is not None else "saved_model"
            }
        }

        # Create decoder config
        decoder_config = copy.deepcopy(base_config)
        decoder_config['model'] = {
            'layer_widths': decoder_params['hidden_layers'],
            'dropout': decoder_params['dropout'],
            'batchnorm': decoder_params['batchnorm'],
            'weight_decay': decoder_params['weight_decay'],
            'lr': decoder_params['learning_rate'],
            'scheduler': {
                'name': decoder_params['scheduler_type'],
                'patience': decoder_params['scheduler_patience'],
                'factor': decoder_params['scheduler_factor'],
                'min_lr': decoder_params['scheduler_min_lr'],
                'monitor': "val_loss"
            }
        }
        decoder_config['training'].update({
            'max_epochs': decoder_params['max_epochs'],
            'early_stopping_patience': decoder_params['early_stopping_patience'],
            'early_stopping_min_delta': decoder_params['early_stopping_min_delta']
        })

        # Create reducer config
        reducer_config = copy.deepcopy(base_config)
        reducer_config['model'] = {
            'layer_widths': reducer_params['hidden_layers'],
            'dropout': reducer_params['dropout'],
            'batchnorm': reducer_params['batchnorm'],
            'weight_decay': reducer_params['weight_decay'],
            'lr': reducer_params['learning_rate'],
            'scheduler': {
                'name': reducer_params['scheduler_type'],
                'patience': reducer_params['scheduler_patience'],
                'factor': reducer_params['scheduler_factor'],
                'min_lr': reducer_params['scheduler_min_lr'],
                'monitor': "val_loss"
            }
        }
        reducer_config['training'].update({
            'max_epochs': reducer_params['max_epochs'],
            'early_stopping_patience': reducer_params['early_stopping_patience'],
            'early_stopping_min_delta': reducer_params['early_stopping_min_delta']
        })

        return DictConfig(decoder_config), DictConfig(reducer_config)