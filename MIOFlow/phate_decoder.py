import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from omegaconf import DictConfig

import torch.nn as nn
import matplotlib.pyplot as plt

class Decoder(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        output_dim,
        layer_widths,
        dropout,
        batchnorm,
        weight_decay,
        lr,
        weighted_loss=False,
        scheduler=None,
        metric_prefix: str = ""
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for width in layer_widths:
            layers.append(nn.Linear(prev_dim, width))
            if batchnorm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = width
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

        self.save_hyperparameters()
        self.weighted_loss = weighted_loss
        self.loss_fn = nn.MSELoss(reduction='none')
        self.metric_prefix = metric_prefix

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        if self.weighted_loss:
            x, y, w = batch
            y_hat = self.forward(x)
            loss = self.loss_fn(y_hat, y)
            loss = loss.mean(dim=0)
            weighted_loss = (loss * w).mean()
            self.log(f"{self.metric_prefix}train_loss", weighted_loss)
            return weighted_loss
        else:
            x, y = batch
            y_hat = self.forward(x)
            loss = self.loss_fn(y_hat, y).mean()
            self.log(f"{self.metric_prefix}train_loss", loss)
            return loss

    def validation_step(self, batch, batch_idx):
        if self.weighted_loss:
            x, y, w = batch
            y_hat = self.forward(x)
            loss = self.loss_fn(y_hat, y)
            loss = loss.mean(dim=0)
            weighted_loss = (loss * w).mean()
            self.log(f"{self.metric_prefix}val_loss", weighted_loss, prog_bar=True)
        else:
            x, y = batch
            y_hat = self.forward(x)
            loss = self.loss_fn(y_hat, y).mean()
            self.log(f"{self.metric_prefix}val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        if self.weighted_loss:
            x, y, w = batch
            y_hat = self.forward(x)
            loss = self.loss_fn(y_hat, y)
            loss = loss.mean(dim=0)
            weighted_loss = (loss * w).mean()
            self.log(f"{self.metric_prefix}test_loss", weighted_loss)
        else:
            x, y = batch
            y_hat = self.forward(x)
            loss = self.loss_fn(y_hat, y).mean()
            self.log(f"{self.metric_prefix}test_loss", loss)

    def _make_scatter_plot(self, loader, set_name):
        """Create scatter plot of predictions vs targets"""
        all_preds = []
        all_targets = []
        
        # Collect predictions
        self.eval()
        with torch.no_grad():
            for batch in loader:
                if self.weighted_loss:
                    x, y, w = batch
                else:
                    x, y = batch
                y_hat = self.forward(x)
                all_preds.append(y_hat.detach())
                all_targets.append(y.detach())
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        if all_preds.shape[1] >= 2 and all_targets.shape[1] >= 2:
            fig = plt.figure(figsize=(10, 10))
            plt.scatter(all_targets[:, 0].cpu(), all_targets[:, 1].cpu(), 
                       alpha=0.5, label='True')
            plt.scatter(all_preds[:, 0].cpu(), all_preds[:, 1].cpu(), 
                       alpha=0.5, label='Predicted')
            plt.legend()
            plt.title(f'{set_name}: First 2 Dimensions - True vs Predicted')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            
            # Check logger type first
            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure(
                    f"{self.metric_prefix}{set_name.lower()}_scatter",
                    fig,
                    global_step=self.current_epoch
            )
            elif self.logger is not None:  # Handle any other logger types
                try:
                    self.logger.experiment.add_figure(
                        f"{self.metric_prefix}{set_name.lower()}_scatter",
                        fig,
                        global_step=self.current_epoch
                    )
                except AttributeError:
                    pass  # Silently fail if logger doesn't support figure logging
            
            plt.close(fig)  # Make sure to close the figure

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        
        if hasattr(self.hparams, 'scheduler'):
            scheduler_config = self.hparams.scheduler
            if scheduler_config.name == "reduce_on_plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    patience=scheduler_config.patience,
                    factor=scheduler_config.factor,
                    min_lr=scheduler_config.min_lr,
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": scheduler_config.monitor,
                    }
                }
            elif scheduler_config.name == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.trainer.max_epochs,
                    eta_min=scheduler_config.min_lr
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                    }
                }
            elif scheduler_config.name == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_config.patience,
                    gamma=scheduler_config.factor
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                    }
                }
        
        return optimizer

"""
latest model: outputs/2024-12-22/15-50-05
"""
def predict(x: np.ndarray, checkpoint_path: str, scaler_path: str, device: str = "cpu") -> np.ndarray:
    """
    Load trained model and scaler to make predictions.
    
    Args:
        x: Input data of shape (n_samples, n_features)
        checkpoint_path: Path to the model checkpoint file
        scaler_path: Path to the saved scaler file
        device: Device to run the model on ("cpu" or "cuda")
        
    Returns:
        Predicted output of shape (n_samples, n_output_dims)
    """
    # Load scaler and transform input
    x_scaler = joblib.load(scaler_path)
    x_scaled = x_scaler.transform(x)
    x_tensor = torch.FloatTensor(x_scaled).to(device)
    
    # Load model
    model = Decoder.load_from_checkpoint(checkpoint_path).to(device)
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        y_pred = model(x_tensor)
    
    return y_pred.cpu().numpy()

def main(cfg: DictConfig):
    # Load data from npy files
    x = np.load(cfg.data.x_path)
    y = np.load(cfg.data.y_path)
    
    # Load weights if specified
    weights = None
    if hasattr(cfg.data, 'weights_path') and cfg.data.weights_path:
        weights = np.load(cfg.data.weights_path)
        assert weights.shape[0] == y.shape[1], "Weights dimension must match output dimension"
        # Normalize weights to sum to 1
        weights = weights / weights.sum()
        weights = torch.FloatTensor(weights)

    # Initialize and fit scalers
    x_scaler = StandardScaler()
    # y_scaler = StandardScaler()
    
    x_scaled = x_scaler.fit_transform(x)
    # y_scaled = y_scaler.fit_transform(y)

    # Save scalers
    os.makedirs(cfg.scalers.save_dir, exist_ok=True)
    joblib.dump(x_scaler, os.path.join(cfg.scalers.save_dir, 'x_scaler.pkl'))
    # joblib.dump(y_scaler, os.path.join(cfg.scalers.save_dir, 'y_scaler.pkl'))

    # Convert to tensors
    x = torch.FloatTensor(x_scaled)
    # y = torch.FloatTensor(y_scaled)
    y = torch.FloatTensor(y)

    # Fix: Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Fix: Calculate split sizes properly
    if weights is not None:
        dataset = TensorDataset(x, y, weights.expand(len(x), -1))
    else:
        dataset = TensorDataset(x, y)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    # Fix: Add num_workers for better data loading performance
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.training.batch_size,
        shuffle=True, 
        num_workers=cfg.training.num_workers,
        drop_last=cfg.training.drop_last
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.training.batch_size,
        shuffle=False, 
        num_workers=cfg.training.num_workers,
        drop_last=cfg.training.drop_last
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=cfg.training.batch_size,
        shuffle=False, 
        num_workers=cfg.training.num_workers,
        drop_last=cfg.training.drop_last
    )

    # wandb_logger = WandbLogger(
    #     project=cfg.logging.wandb_project,
    #     name=cfg.logging.run_name if hasattr(cfg.logging, 'run_name') else None
    # )
    tensorboard_logger = TensorBoardLogger(
        save_dir=cfg.logging.tensorboard_dir,
        name=cfg.logging.run_name if hasattr(cfg.logging, 'run_name') else None
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.logging.checkpoint_dir,
        save_top_k=cfg.logging.save_top_k,
        monitor=cfg.logging.monitor_metric,
        mode=cfg.logging.monitor_mode
    )

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
        metric_prefix=cfg.logging.run_name if hasattr(cfg.logging, 'run_name') else ""
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=tensorboard_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=cfg.logging.log_every_n_steps,
        deterministic=True,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices
    )

    trainer.fit(model, train_loader, val_loader)
    
    # Create scatter plots after training
    model._make_scatter_plot(train_loader, "Train")
    model._make_scatter_plot(val_loader, "Validation")
    
    # Run test and create test scatter plot
    trainer.test(model, test_loader)
    model._make_scatter_plot(test_loader, "Test")

if __name__ == "__main__":
    main()