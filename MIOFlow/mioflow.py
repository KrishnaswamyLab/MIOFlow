"""
TODO:
1. per-time-point loss weights
2. deprecate local training
3. add weight initialization
4. test on gpu
5. test different activation functions
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchdiffeq import odeint
from tqdm import tqdm

try:
    from MIOFlow.core.models import ODEFunc
    from MIOFlow.core.losses import ot_loss, density_loss, energy_loss
except ImportError:
    from core.models import ODEFunc
    from core.losses import ot_loss, density_loss, energy_loss

class TimeSeriesDataset(Dataset):
    """
    Time series data with variable numbers of points per time step.
    Stores a list of (X_t, t) tuples where X_t is [n_points, dim].
    """

    def __init__(self, time_series_data: List[Tuple[np.ndarray, float]]):
        self.time_series_data = time_series_data
        self.times = [t for _, t in time_series_data]

    def __len__(self):
        return len(self.time_series_data) - 1

    def __getitem__(self, idx):
        X_t, t_start = self.time_series_data[idx]
        X_t1, t_end = self.time_series_data[idx + 1]
        return {
            'X_start': torch.tensor(X_t, dtype=torch.float32),
            'X_end': torch.tensor(X_t1, dtype=torch.float32),
            't_start': t_start,
            't_end': t_end,
            'interval_idx': idx,
        }

    def get_time_sequence(self, start_idx=0, end_idx=None):
        if end_idx is None:
            end_idx = len(self.times)
        return torch.tensor(self.times[start_idx:end_idx], dtype=torch.float32)

    def get_initial_condition(self, start_idx=0):
        X_0, _ = self.time_series_data[start_idx]
        return torch.tensor(X_0, dtype=torch.float32)


class MIOFlow:
    """
    Manifold Interpolating Optimal-Transport Flow for trajectory inference.

    Typical workflow::

        # 1. Train GAGA externally (see gaga.py)
        gaga_model = Autoencoder(input_dim, latent_dim)
        train_gaga_two_phase(gaga_model, dataloader, ...)

        # 2. Pass the trained model + its input scaler to MIOFlow
        mf = MIOFlow(
            adata,
            gaga_model=gaga_model,
            gaga_input_scaler=scaler,   # StandardScaler fitted on X_pca
            obs_time_key='time_bin',
            n_epochs=100,
        )
        mf.fit()
        mf.trajectories                   # (n_bins, n_trajectories, latent_dim)
        mf.decode_to_gene_space()         # (n_bins, n_trajectories, n_genes)

    Parameters
    ----------
    adata : AnnData
        Annotated data. Must contain ``obsm['X_pca']`` and ``varm['PCs']``
        for gene-space decoding.
    input_df : pd.DataFrame, optional
        Pre-built DataFrame with columns ``d1, d2, ..., samples`` (embedding
        already in latent space). When provided, GAGA encoding is skipped.
    gaga_model : Autoencoder, optional
        A trained GAGA ``Autoencoder`` from ``gaga.py``. Its encoder is used
        to embed ``adata.obsm[gaga_input_key]`` into the latent space for ODE
        training; its decoder is used in ``decode_to_gene_space()``.
    gaga_input_key : str
        Key in ``adata.obsm`` fed to the GAGA encoder (default: ``'X_pca'``).
    gaga_input_scaler : sklearn-compatible scaler, optional
        Scaler (e.g. ``StandardScaler``) already fitted on the data in
        ``adata.obsm[gaga_input_key]``. Used to normalise inputs before
        encoding and to inverse-transform decoder outputs.
    obs_time_key : str
        Column in ``adata.obs`` holding the time/group label
        (default: ``'day'``).
    model_config : dict, optional
        Overrides for ODEFunc. Recognised keys: ``hidden_dim`` (int),
        ``use_cuda`` (bool).
    n_local_epochs : int
        Epochs of local (per-interval) pre-training.
    n_epochs : int
        Epochs of global (full-trajectory) training.
    n_post_local_epochs : int
        Epochs of local fine-tuning after global training.
    lambda_ot : float
        Weight for the OT loss (default 1.0, set to 0 to disable).
    use_density_loss : bool
        Whether to include the density loss term.
    lambda_density : float
        Weight for the density loss.
    lambda_energy : float
        Weight for the energy regularisation.
    energy_time_steps : int
        Number of sub-steps used when computing the energy loss.
    learning_rate : float
        Adam learning rate.
    sample_size : int, optional
        Batch size (points sampled per time step). ``None`` → use all.
    n_trajectories : int
        Number of trajectories to generate after training.
    n_bins : int
        Number of time bins for trajectory integration.
    scheduler_type : str, optional
        LR scheduler: ``'step'``, ``'exponential'``, ``'cosine'``, or None.
    """

    def __init__(
        self,
        adata,
        input_df: Optional[pd.DataFrame] = None,
        # GAGA autoencoder (trained externally)
        gaga_model=None,
        gaga_input_key: str = 'X_pca',
        gaga_input_scaler=None,
        obs_time_key: str = "time_bin",
        debug_level: str = 'info',
        hidden_dim: float = 64,
        use_cuda: bool = True,
        #Model config
        momentum_beta = 0.0,
        # Training
        n_epochs: int = 100,
        # Loss
        lambda_ot: float = 1.0,
        use_density_loss: bool = False,
        lambda_density: float = 0.1,
        lambda_energy: float = 0.01,
        energy_time_steps: int = 10,
        learning_rate: float = 1e-3,
        # Data
        sample_size: Optional[int] = None,
        # Output
        exp_dir: str = '.',
        n_trajectories: int = 100,
        n_bins: int = 100,
        # Scheduler
        scheduler_type: Optional[str] = None,
        scheduler_step_size: int = 30,
        scheduler_gamma: float = 0.5,
        scheduler_t_max: Optional[int] = None,
        scheduler_min_lr: float = 0.0,
    ):
        self.adata = adata
        self.gaga_autoencoder = gaga_model
        self.gaga_input_key = gaga_input_key
        self.gaga_input_scaler = gaga_input_scaler
        self.obs_time_key = obs_time_key

        # Model config
        self.hidden_dim = hidden_dim
        self.momentum_beta = momentum_beta
        self.device = 'cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu'

        # Training
        self.n_epochs = n_epochs

        # Loss weights
        self.lambda_ot = lambda_ot
        self.lambda_density = lambda_density if use_density_loss else 0.0
        self.lambda_energy = lambda_energy
        self.energy_time_steps = energy_time_steps
        self.learning_rate = learning_rate

        # Data
        self.sample_size = sample_size

        # Output
        self.exp_dir = exp_dir
        self.n_trajectories = n_trajectories
        self.n_bins = n_bins

        # Scheduler
        self.scheduler_kwargs = dict(
            scheduler_type=scheduler_type,
            scheduler_step_size=scheduler_step_size,
            scheduler_gamma=scheduler_gamma,
            scheduler_t_max=scheduler_t_max,
            scheduler_min_lr=scheduler_min_lr,
        )

        # State
        self.is_fitted = False
        self.ode_model: Optional[ODEFunc] = None
        self.trajectories: Optional[np.ndarray] = None
        self.losses = None

        self._setup_logging(debug_level)
        Path(exp_dir).mkdir(parents=True, exist_ok=True)

        # Encode, normalize, and store as TimeSeriesDataset
        self.dataset = self._prepare_data(input_df)

        self.logger.info(
            f"MIOFlow initialised | {adata.n_obs} cells, "
            f"{adata.n_vars} genes | device={self.device}"
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_logging(self, debug_level: str):
        level_map = {'verbose': logging.DEBUG, 'info': logging.INFO,
                     'warning': logging.WARNING, 'error': logging.ERROR}
        logging.basicConfig(
            level=level_map.get(debug_level, logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        self.logger = logging.getLogger('MIOFlow')

    def _encode(self, input_df: Optional[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (embedding, time_labels) as numpy arrays.

        Priority order:
        1. *input_df* — used as-is (GAGA encoding skipped).
        2. ``gaga_model`` — encodes ``adata.obsm[gaga_input_key]`` to latent space.
        3. Fallback — reads the raw array from ``adata.obsm[gaga_input_key]``.

        Time labels always come from ``adata.obs[obs_time_key]``.
        """
        if input_df is not None:
            embed_cols = [c for c in input_df.columns if c != 'samples']
            return input_df[embed_cols].values.astype(np.float64), input_df['samples'].values

        if self.gaga_input_key not in self.adata.obsm:
            raise ValueError(f"Key '{self.gaga_input_key}' not found in adata.obsm")

        X_raw = self.adata.obsm[self.gaga_input_key].astype(np.float32)

        if self.gaga_autoencoder is not None:
            X_scaled = (self.gaga_input_scaler.transform(X_raw)
                        if self.gaga_input_scaler is not None else X_raw)
            self.gaga_autoencoder.eval()
            with torch.no_grad():
                embedding = self.gaga_autoencoder.encode(
                    torch.tensor(X_scaled)
                ).cpu().numpy().astype(np.float64)
        else:
            embedding = X_raw.astype(np.float64)

        time_labels, _ = pd.factorize(self.adata.obs[self.obs_time_key])
        return embedding, time_labels

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Z-normalize embedding, storing mean/std for later denormalization."""
        self.mean_vals = embedding.mean(axis=0)
        std = embedding.std(axis=0)
        self.std_vals = np.where(std == 0, 1.0, std)
        return (embedding - self.mean_vals) / self.std_vals

    def _prepare_data(self, input_df: Optional[pd.DataFrame] = None) -> 'TimeSeriesDataset':
        """Encode, normalize, and package data as a TimeSeriesDataset."""
        embedding, time_labels = self._encode(input_df)
        normed = self._normalize(embedding)
        groups = sorted(set(time_labels))
        time_series_data = [
            (normed[time_labels == g].astype(np.float32), float(g))
            for g in groups
        ]
        return TimeSeriesDataset(time_series_data)

    def fit(self) -> 'MIOFlow':
        """Train the ODE model and generate trajectories. Returns self."""
        dataset = self.dataset
        input_dim = self.dataset.time_series_data[0][0].shape[1]

        self.ode_model = ODEFunc(input_dim=input_dim, hidden_dim=self.hidden_dim,momentum_beta=self.momentum_beta)
        self.logger.info(f"ODEFunc: input_dim={input_dim}, hidden_dim={self.hidden_dim}")

        self.logger.info(f"Global training: {self.n_epochs} epochs")
        history = train_mioflow(
            model=self.ode_model,
            dataset=dataset,
            num_epochs=self.n_epochs,
            batch_size=self.sample_size,
            learning_rate=self.learning_rate,
            device=self.device,
            lambda_ot=self.lambda_ot,
            lambda_density=self.lambda_density,
            lambda_energy=self.lambda_energy,
            energy_time_steps=self.energy_time_steps,
            **self.scheduler_kwargs,
        )
        self.losses = history

        self._generate_trajectories(dataset)

        self.is_fitted = True
        self.logger.info("MIOFlow fitting completed.")
        return self

    def _generate_trajectories(self, dataset: TimeSeriesDataset):
        """Integrate n_trajectories paths over n_bins time steps."""
        X_0_full = dataset.get_initial_condition()
        n = min(self.n_trajectories, X_0_full.size(0))
        idx = torch.randperm(X_0_full.size(0))[:n]
        X_0_sample = X_0_full[idx].to(self.device)

        times = dataset.times
        t_bins = torch.linspace(min(times), max(times), self.n_bins, device=self.device)

        self.ode_model.eval()
        with torch.no_grad():
            traj = odeint(self.ode_model, X_0_sample, t_bins)  # (n_bins, n_traj, n_dims)

        self.trajectories = traj.cpu().numpy()
        self.logger.info(f"Trajectories generated: shape={self.trajectories.shape}")

    # ------------------------------------------------------------------
    # Post-fitting API
    # ------------------------------------------------------------------

    def decode_to_gene_space(self) -> np.ndarray:
        """
        Decode trajectories from GAGA latent space back to gene space.

        Requires:
        - ``gaga_model`` passed at construction (for latent → PCA decoding)
        - ``adata.varm['PCs']`` (for PCA → gene space)

        Returns
        -------
        np.ndarray, shape (n_bins, n_trajectories, n_genes)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before decode_to_gene_space().")
        if self.gaga_autoencoder is None:
            raise RuntimeError(
                "No GAGA model available. Pass a trained Autoencoder via "
                "gaga_model= at construction time."
            )
        if 'PCs' not in self.adata.varm:
            raise RuntimeError("adata.varm['PCs'] is required for gene-space decoding.")

        # Denormalise latent trajectories: (n_bins, n_traj, latent_dim)
        traj = self.trajectories * self.std_vals + self.mean_vals
        traj_shape = traj.shape
        traj_flat = traj.reshape(-1, traj_shape[-1])

        # GAGA latent → scaled PCA space
        self.gaga_autoencoder.eval()
        with torch.no_grad():
            traj_pca_scaled = self.gaga_autoencoder.decode(
                torch.tensor(traj_flat, dtype=torch.float32)
            ).cpu().numpy()

        # Inverse-scale back to PCA space (if a scaler was provided)
        traj_pca = (self.gaga_input_scaler.inverse_transform(traj_pca_scaled)
                    if self.gaga_input_scaler is not None
                    else traj_pca_scaled)

        # PCA → gene space
        X_reconstructed = np.array(
            traj_pca @ self.adata.varm['PCs'].T + np.array(self.adata.X.mean(axis=0))
        )
        self.trajectories_gene_space = X_reconstructed.reshape(
            traj_shape[0], traj_shape[1], -1
        )
        return self.trajectories_gene_space

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        shape = self.trajectories.shape if self.trajectories is not None else None
        gaga = self.gaga_autoencoder.__class__.__name__ if self.gaga_autoencoder is not None else 'None'
        return (
            f"MIOFlow(n_obs={self.adata.n_obs}, gaga={gaga}, "
            f"n_epochs={self.n_epochs}, trajectories={shape}, status={status})"
        )

# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------

def _make_scheduler(optimizer, scheduler_type, step_size, gamma, t_max, num_epochs, min_lr):
    if scheduler_type == 'step':
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if scheduler_type == 'exponential':
        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    if scheduler_type == 'cosine':
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max if t_max is not None else num_epochs, eta_min=min_lr
        )
    return None


def train_mioflow(
    model: ODEFunc,
    dataset: TimeSeriesDataset,
    num_epochs: int,
    batch_size: int = None,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    lambda_ot: float = 1.0,
    lambda_density: float = 0.1,
    lambda_energy: float = 0.01,
    energy_time_steps: int = 10,
    scheduler_type: str = None,
    scheduler_step_size: int = 30,
    scheduler_gamma: float = 0.5,
    scheduler_t_max: int = None,
    scheduler_min_lr: float = 0.0,
) -> Dict:
    """
    Train an ODEFunc on a TimeSeriesDataset using global (full-trajectory) training.

    Returns:
        History dict with keys: epoch, total_loss, ot_loss, density_loss, energy_loss.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = _make_scheduler(
        optimizer, scheduler_type, scheduler_step_size, scheduler_gamma,
        scheduler_t_max, num_epochs, scheduler_min_lr,
    )

    history: Dict[str, list] = {
        'epoch': [], 'total_loss': [], 'ot_loss': [], 'density_loss': [], 'energy_loss': [],
    }

    for epoch in tqdm(range(num_epochs), desc='Training (global)'):
        epoch_losses = {'total': 0.0, 'ot': 0.0, 'density': 0.0, 'energy': 0.0}

        num_intervals = 0
        # Train each time interval separately
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
            model.reset_momentum()
            t_interval = torch.tensor([t_start, t_end], device=device, dtype=torch.float32)
            X_pred = odeint(model, X_start, t_interval)[1]

            # Compute losses
            total_loss = torch.tensor(0.0, device=device)
            ot_loss_val = torch.tensor(0.0, device=device)
            density_loss_val = torch.tensor(0.0, device=device)
            energy_loss_val = torch.tensor(0.0, device=device)

            if lambda_ot > 0:
                ot_loss_val = ot_loss(X_pred, X_end)
                total_loss = total_loss + lambda_ot * ot_loss_val

            if lambda_density > 0:
                density_loss_val = density_loss(X_pred, X_end)
                total_loss = total_loss + lambda_density * density_loss_val

            if lambda_energy > 0:
                energy_t_seq = torch.linspace(t_start, t_end, energy_time_steps, device=device, dtype=torch.float32)
                energy_loss_val = energy_loss(model, X_start, energy_t_seq)
                total_loss = total_loss + lambda_energy * energy_loss_val


            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_losses['total'] += total_loss.item()
            epoch_losses['ot'] += ot_loss_val.item()
            epoch_losses['density'] += density_loss_val.item()
            epoch_losses['energy'] += energy_loss_val.item()
            num_intervals += 1

        # Average losses across intervals
        for key in epoch_losses:
            epoch_losses[key] /= num_intervals

        if scheduler is not None:
            scheduler.step()

        history['epoch'].append(epoch + 1)
        history['total_loss'].append(epoch_losses['total'])
        history['ot_loss'].append(epoch_losses['ot'])
        history['density_loss'].append(epoch_losses['density'])
        history['energy_loss'].append(epoch_losses['energy'])

    return history

