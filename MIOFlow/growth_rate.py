import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import ot
import os

try:
    from MIOFlow.core.datasets import TimeSeriesDataset
except ImportError:
    from core.datasets import TimeSeriesDataset


class GrowthRateModel(nn.Module):
    def __init__(
        self,
        adata,
        hidden_dim: int = 32,
        use_time: bool = True,
        # Data encoding — same interface as MIOFlow
        gaga_model=None,
        gaga_input_key: str = 'X_pca',
        obs_time_key: str = 'time_bin',
        use_cuda: bool = True,
    ):
        super().__init__()
        self.use_time = use_time
        self.gaga_autoencoder = gaga_model
        self.gaga_input_key = gaga_input_key
        self.obs_time_key = obs_time_key
        self.device = 'cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu'

        self.dataset = self._prepare_data(adata)
        input_dim = self.dataset.time_series_data[0][0].shape[1]

        actual_input_dim = input_dim + 1 if use_time else input_dim
        self.net = nn.Sequential(
            nn.Linear(actual_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.softplus = nn.Softplus()
        self._initialize_weights()

    def _prepare_data(self, adata) -> TimeSeriesDataset:
        X_raw = adata.obsm[self.gaga_input_key].astype(np.float32)

        if self.gaga_autoencoder is not None:
            X_scaled = (self.gaga_autoencoder.input_scaler.transform(X_raw)
                        if self.gaga_autoencoder.input_scaler is not None else X_raw)
            self.gaga_autoencoder.eval()
            with torch.no_grad():
                embedding = self.gaga_autoencoder.encode(
                    torch.tensor(X_scaled)
                ).cpu().numpy().astype(np.float64)
        else:
            embedding = X_raw.astype(np.float64)

        mean = embedding.mean(axis=0)
        std = embedding.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        normed = ((embedding - mean) / std).astype(np.float32)

        time_labels, _ = pd.factorize(adata.obs[self.obs_time_key], sort=True)
        groups = sorted(set(time_labels))
        return TimeSeriesDataset([
            (normed[time_labels == g], float(g)) for g in groups
        ])

    def _initialize_weights(self):
        # Initialize last layer bias to 0.5413 so initial mass is approx 1.0 (Softplus(0.5413) ~ 1.0)
        # and weights to 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.5413)

    def forward(self, x, t=None):
        if self.use_time:
            if t is None:
                raise ValueError("t must be provided if use_time is True")
            if t.dim() == 0:
                t_expanded = t.expand(x.size(0), 1)
            elif t.dim() == 1 and t.size(0) == 1:
                t_expanded = t.expand(x.size(0), 1)
            else:
                t_expanded = t.view(-1, 1)
            input_feats = torch.cat([x, t_expanded], dim=-1)
        else:
            input_feats = x

        # Mass = Softplus(net(input_feats)) + 1e-4 to ensure strict positivity
        out = self.net(input_feats)
        return self.softplus(out).squeeze(-1) + 1e-4

    def predict_all(self):
        """Return (points, predicted_mass) for every cell in the internal dataset."""
        all_x = torch.cat([torch.tensor(X, dtype=torch.float32) for X, _ in self.dataset.time_series_data])
        all_t = torch.cat([torch.full((len(X), 1), t) for X, t in self.dataset.time_series_data])
        self.eval()
        with torch.no_grad():
            mass = (self(all_x, all_t) if self.use_time else self(all_x)).cpu().numpy()
        return all_x.cpu().numpy(), mass

    def plot_mass(self, save_path=None, figsize=(10, 8)):
        """Scatter-plot the predicted growth-rate mass over all cells."""
        points, mass = self.predict_all()
        fig, ax = plt.subplots(figsize=figsize)
        sc = ax.scatter(points[:, 0], points[:, 1], c=mass, cmap='viridis', s=20, alpha=0.8)
        fig.colorbar(sc, ax=ax, label='Predicted Growth Rate')
        ax.set_title('Predicted Growth Rate Field (UOT Init)')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path)
            print(f"Saved plot to {save_path}")
        plt.show()
        return fig, ax

    def pretrain(
        self,
        reg_m=[1.0, 100.0],
        div: str = 'l2',
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        scheduler_type: str = None,
        scheduler_step_size: int = 30,
        scheduler_gamma: float = 0.5,
        scheduler_t_max: int = None,
        scheduler_min_lr: float = 0.0,
    ):
        """Compute UOT masses from internal dataset and pretrain. Returns self."""
        uot_masses = compute_uot_growth_rates(self.dataset, reg_m=reg_m, div=div)
        pretrain_growth_model(
            self, self.dataset, uot_masses,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=self.device,
            scheduler_type=scheduler_type,
            scheduler_step_size=scheduler_step_size,
            scheduler_gamma=scheduler_gamma,
            scheduler_t_max=scheduler_t_max,
            scheduler_min_lr=scheduler_min_lr,
        )

def compute_uot_growth_rates(dataset: TimeSeriesDataset, reg_m=[1.0, 100.0], div='l2'):
    """
    Computes Unbalanced Optimal Transport (UOT) mass for each point in the dataset 
    by solving sequential UOT problems between timepoints.
    
    A lower reg_m (e.g., 1.0) allows MASSIVE destruction of points, creating huge variance.
    
    Returns a list of 1D tensors, where each tensor contains the target mass 
    for the points at the corresponding time index.
    """
    growth_rates = []
    
    for i in range(len(dataset)):
        batch = dataset[i]
        x0 = batch['X_start'].cpu().numpy()
        x1 = batch['X_end'].cpu().numpy()
        
        m, n = ot.unif(len(x0)), ot.unif(len(x1))
        M = ot.dist(x0, x1)
        plan = ot.unbalanced.mm_unbalanced(m, n, M, reg_m, div=div)
        
        gr = plan.sum(axis=1) * plan.shape[0]
        growth_rates.append(torch.tensor(gr, dtype=torch.float32))
        
    return growth_rates

def pretrain_growth_model(
    model: GrowthRateModel, 
    dataset: TimeSeriesDataset, 
    uot_masses: List[torch.Tensor],
    num_epochs: int = 50, 
    learning_rate: float = 1e-3, 
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    scheduler_type: str = None,  # 'step', 'exponential', 'cosine', or None
    scheduler_step_size: int = 30,  # For StepLR
    scheduler_gamma: float = 0.5,  # Decay factor for schedulers
    scheduler_t_max: int = None,  # For CosineAnnealingLR, defaults to num_epochs
    scheduler_min_lr: float = 0.0   # Minimum learning rate for cosine scheduler
):
    """
    Pretrains the GrowthRateModel using MSE loss to match the UOT computed masses.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create learning rate scheduler if specified
    scheduler = None
    if scheduler_type == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    elif scheduler_type == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
    elif scheduler_type == 'cosine':
        t_max = scheduler_t_max if scheduler_t_max is not None else num_epochs
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=scheduler_min_lr)

    criterion = nn.MSELoss()
    
    model.train()
    
    all_x = []
    all_t = []
    all_target_mass = []
    for i in range(len(dataset)):
        batch = dataset[i]
        all_x.append(batch['X_start'])
        t_start = batch['t_start']
        all_t.append(torch.full((batch['X_start'].size(0), 1), t_start, dtype=torch.float32))
        all_target_mass.append(uot_masses[i])
        
        # Explicitly anchor the final timepoint!
        if i == len(dataset) - 1:
            all_x.append(batch['X_end'])
            t_end = batch['t_end']
            all_t.append(torch.full((batch['X_end'].size(0), 1), t_end, dtype=torch.float32))
            all_target_mass.append(torch.ones(batch['X_end'].size(0), dtype=torch.float32))
        
    all_x = torch.cat(all_x, dim=0).to(device)
    all_t = torch.cat(all_t, dim=0).to(device)
    all_target_mass = torch.cat(all_target_mass, dim=0).to(device)
    
    pbar = tqdm(range(num_epochs), desc='Pre-training Growth Model')
    for epoch in pbar:
        optimizer.zero_grad()
        
        if getattr(model, 'use_time', False):
            predicted_mass = model(all_x, all_t)
        else:
            predicted_mass = model(all_x)
            
        loss = criterion(predicted_mass, all_target_mass)
        
        loss.backward()
        optimizer.step()
        
        # Step the scheduler if specified
        if scheduler is not None:
            scheduler.step()
        
        postfix_dict = {'MSE': f'{loss.item():.4f}'}
        if scheduler is not None:
            postfix_dict['LR'] = f'{optimizer.param_groups[0]["lr"]:.2e}'
            
        pbar.set_postfix(postfix_dict)