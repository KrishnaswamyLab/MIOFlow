import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import ot
from torchdiffeq import odeint
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple

class ODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim), # additional dim for time t.
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, t, x):
        # t is scalar, x is [batch_size, input_dim]
        # Expand t to [batch_size, 1] to match x's batch dimension
        t_expanded = t.expand(x.size(0), 1)
        input = torch.cat([t_expanded, x], dim=-1)
        return self.model(input)

def ot_loss(source, target):
    mu = torch.tensor(ot.unif(source.size()[0]), dtype=source.dtype, device=source.device)
    nu = torch.tensor(ot.unif(target.size()[0]), dtype=target.dtype, device=target.device)
    M = torch.cdist(source, target)**2
    return ot.emd2(mu, nu, M)

def energy_loss(model, x0, t_seq):
    """
    Compute energy loss by evaluating vector field magnitude along the ODE trajectory.
    This penalizes large vector field values to encourage smoother flows.

    Args:
        model: ODEFunc model
        x0: Initial points [batch_size, input_dim]
        t_seq: Time sequence [num_times]

    Returns:
        Energy loss (mean squared magnitude of vector field along trajectory)
    """
    # Compute the full trajectory
    trajectory = odeint(model, x0, t_seq)  # [time_steps, batch_size, input_dim]

    total_energy = 0.0
    num_evaluations = 0

    # Evaluate vector field at each point along the trajectory
    for i, t_val in enumerate(t_seq):
        # Current points at time t_val: trajectory[i]
        x_t = trajectory[i]  # [batch_size, input_dim]

        # Create time tensor for all batch points
        t_tensor = torch.full((x_t.size(0), 1), t_val, device=x_t.device, dtype=x_t.dtype)

        # Get vector field (dx/dt) at current trajectory points
        dx_dt = model(t_tensor, x_t)

        # Add squared magnitude (L2 norm squared for each point)
        total_energy += torch.sum(dx_dt ** 2)
        num_evaluations += x_t.size(0)

    # Return average energy across all trajectory evaluations
    return total_energy / num_evaluations

def density_loss(source, target, top_k=5, hinge_value=0.01):
    """
    Density loss that encourages points to be close to target distribution.
    Uses hinge loss on k-nearest neighbor distances.
    """
    c_dist = torch.cdist(source, target)
    values, _ = torch.topk(c_dist, top_k, dim=1, largest=False, sorted=False)
    values = torch.clamp(values - hinge_value, min=0.0)
    return torch.mean(values)


def infer(x0, model, t_seq):
    return odeint(model, x0, t_seq)


class TimeSeriesDataset(Dataset):
    """
    Dataset for time series data with variable number of points per time step.
    Data format: list of (X_t, t) tuples, where X_t has shape [n_points, dim]
    """
    def __init__(self, time_series_data: List[Tuple[np.ndarray, float]]):
        """
        Args:
            time_series_data: List of (X_t, t) tuples, where X_t is [n_points, dim] array
        """
        self.time_series_data = time_series_data
        self.times = [t for _, t in time_series_data]

    def __len__(self):
        return len(self.time_series_data) - 1  # Number of intervals

    def __getitem__(self, idx):
        """
        Returns data for training interval idx -> idx+1
        """
        X_t, t_start = self.time_series_data[idx]
        X_t1, t_end = self.time_series_data[idx + 1]

        return {
            'X_start': torch.tensor(X_t, dtype=torch.float32),
            'X_end': torch.tensor(X_t1, dtype=torch.float32),
            't_start': t_start,
            't_end': t_end,
            'interval_idx': idx
        }

    def get_time_sequence(self, start_idx=0, end_idx=None):
        """Get time sequence from start_idx to end_idx"""
        if end_idx is None:
            end_idx = len(self.times)
        return torch.tensor(self.times[start_idx:end_idx], dtype=torch.float32)

    def get_initial_condition(self, start_idx=0):
        """Get initial condition X_0"""
        X_0, _ = self.time_series_data[start_idx]
        return torch.tensor(X_0, dtype=torch.float32)


def train_mioflow(
    model: ODEFunc,
    dataset: TimeSeriesDataset,
    num_epochs: int,
    mode: str = 'local',  # 'local' or 'global'
    batch_size: int = None,  # For local mode, how many points to sample per time step
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    lambda_ot: float = 1.0,
    lambda_density: float = 0.1,
    lambda_energy: float = 0.01,
    energy_time_steps: int = 10
) -> Dict:
    """
    Train MioFlow model.

    Args:
        model: ODEFunc model
        dataset: TimeSeriesDataset
        num_epochs: Number of training epochs
        mode: 'local' (train each interval) or 'global' (train full trajectory)
        batch_size: For local mode, number of points to sample per time step (None = use all)
        learning_rate: Learning rate
        device: Device to train on
        lambda_ot: Weight for OT loss
        lambda_density: Weight for density loss
        lambda_energy: Weight for energy regularization
        energy_time_steps: Number of time steps for energy evaluation

    Returns:
        Training history
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'epoch': [],
        'total_loss': [],
        'ot_loss': [],
        'density_loss': [],
        'energy_loss': []
    }

    epoch_pbar = tqdm(range(num_epochs), desc='Epochs')
    for epoch in epoch_pbar:
        epoch_losses = {'total': 0.0, 'ot': 0.0, 'density': 0.0, 'energy': 0.0}
        num_batches = 0

        if mode == 'local':
            # Local mode: train each time interval separately
            for interval_idx in range(len(dataset)):
                batch = dataset[interval_idx]

                X_start = batch['X_start'].to(device)
                X_end = batch['X_end'].to(device)
                t_start = batch['t_start']
                t_end = batch['t_end']

                # Sample points if batch_size specified
                if batch_size is not None:
                    min_size = min(X_start.size(0), X_end.size(0))
                    effective_batch_size = min(batch_size, min_size)
                    indices = torch.randperm(min_size)[:effective_batch_size]
                    X_start = X_start[indices]
                    X_end = X_end[indices]

                # Integrate from X_start to predict X_end
                t_interval = torch.tensor([t_start, t_end], device=device, dtype=torch.float32)
                X_pred = odeint(model, X_start, t_interval)[1]  # Get final time point

                # Compute losses
                ot_loss_val = ot_loss(X_pred, X_end)
                density_loss_val = density_loss(X_pred, X_end)
                energy_loss_val = energy_loss(model, X_start, t_interval)

                # Total loss
                total_loss = (lambda_ot * ot_loss_val +
                            lambda_density * density_loss_val +
                            lambda_energy * energy_loss_val)

                # Optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Accumulate losses
                epoch_losses['total'] += total_loss.item()
                epoch_losses['ot'] += ot_loss_val.item()
                epoch_losses['density'] += density_loss_val.item()
                epoch_losses['energy'] += energy_loss_val.item()
                num_batches += 1

        elif mode == 'global':
            # Global mode: train full trajectory from X_0
            X_0 = dataset.get_initial_condition().to(device)
            full_t_seq = dataset.get_time_sequence().to(device)

            # Integrate full trajectory
            trajectory = odeint(model, X_0, full_t_seq)

            # Compare with ground truth at each time step
            total_loss = 0.0
            ot_loss_total = 0.0
            density_loss_total = 0.0

            for i in range(1, len(full_t_seq)):  # Skip t=0 (initial condition)
                X_pred = trajectory[i]
                X_true, _ = dataset.time_series_data[i]
                X_true = torch.tensor(X_true, device=device, dtype=torch.float32)

                # Sample if needed
                if batch_size is not None and X_pred.size(0) > batch_size:
                    indices = torch.randperm(X_pred.size(0))[:batch_size]
                    X_pred = X_pred[indices]
                    X_true = X_true[indices]

                ot_loss_val = ot_loss(X_pred, X_true)
                density_loss_val = density_loss(X_pred, X_true)
                total_loss += ot_loss_val + lambda_density * density_loss_val
                ot_loss_total += ot_loss_val.item()
                density_loss_total += density_loss_val.item()

            # Energy loss on full trajectory
            energy_t_seq = torch.linspace(full_t_seq[0], full_t_seq[-1], energy_time_steps, device=device)
            energy_loss_val = energy_loss(model, X_0, energy_t_seq)

            total_loss += lambda_energy * energy_loss_val

            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Record losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['ot'] += ot_loss_total
            epoch_losses['density'] += density_loss_total
            epoch_losses['energy'] += energy_loss_val.item()
            num_batches = 1

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        # Record history
        history['epoch'].append(epoch + 1)
        history['total_loss'].append(epoch_losses['total'])
        history['ot_loss'].append(epoch_losses['ot'])
        history['density_loss'].append(epoch_losses['density'])
        history['energy_loss'].append(epoch_losses['energy'])

        # Update progress bar with loss information
        epoch_pbar.set_postfix({
            'Total': f'{epoch_losses["total"]:.4f}',
            'OT': f'{epoch_losses["ot"]:.4f}',
            'Density': f'{epoch_losses["density"]:.4f}',
            'Energy': f'{epoch_losses["energy"]:.4f}'
        })

    return history

