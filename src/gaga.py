"""
Combined GAGA (Geometric Autoencoder) and DAE (Denoising Autoencoder) implementation.
Simplified for educational purposes with two-phase training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import phate
import scipy
from typing import List, Union, Optional, Tuple


class Autoencoder(nn.Module):
    """
    Simple autoencoder for dimensionality reduction with distance preservation.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 64]
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to input space"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass"""
        z = self.encode(x)
        return self.decode(z)


def train_gaga_two_phase(
    model: Autoencoder,
    train_loader: torch.utils.data.DataLoader,
    encoder_epochs: int,
    decoder_epochs: int,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    dist_weight_phase1: float = 1.0,
    recon_weight_phase2: float = 1.0
) -> dict:
    """
    Two-phase training: Phase 1 trains encoder (decoder frozen) for distance preservation,
    Phase 2 trains decoder (encoder frozen) for reconstruction.

    Args:
        model: The GAGA autoencoder model
        train_loader: DataLoader with batches containing 'x' (data) and 'd' (distances)
        encoder_epochs: Number of epochs for phase 1 (encoder training)
        decoder_epochs: Number of epochs for phase 2 (decoder training)
        learning_rate: Learning rate for optimizer
        device: Device to train on
        val_loader: Optional validation DataLoader
        dist_weight_phase1: Weight for distance preservation loss in phase 1
        recon_weight_phase2: Weight for reconstruction loss in phase 2

    Returns:
        Combined training history from both phases
    """
    print("Phase 1: Training encoder (decoder frozen) for distance preservation")
    phase1_history = train_gaga(
        model=model,
        train_loader=train_loader,
        num_epochs=encoder_epochs,
        learning_rate=learning_rate,
        device=device,
        val_loader=val_loader,
        recon_weight=0.0,
        dist_weight=dist_weight_phase1,
        freeze_decoder=True,
        freeze_encoder=False
    )

    print("\nPhase 2: Training decoder (encoder frozen) for reconstruction")
    phase2_history = train_gaga(
        model=model,
        train_loader=train_loader,
        num_epochs=decoder_epochs,
        learning_rate=learning_rate,
        device=device,
        val_loader=val_loader,
        recon_weight=recon_weight_phase2,
        dist_weight=0.0,
        freeze_decoder=False,
        freeze_encoder=True
    )

    combined_history = {
        'phase1': phase1_history,
        'phase2': phase2_history
    }

    return combined_history


def train_gaga(
    model: Autoencoder,
    train_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    recon_weight: float = 1.0,
    dist_weight: float = 1.0,
    freeze_encoder: bool = False,
    freeze_decoder: bool = False
) -> dict:
    """
    Train GAGA model with reconstruction and distance preservation losses.

    Args:
        model: The GAGA autoencoder model
        train_loader: DataLoader with batches containing 'x' (data) and 'd' (distances)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on
        val_loader: Optional validation DataLoader
        recon_weight: Weight for reconstruction loss
        dist_weight: Weight for distance preservation loss
        freeze_encoder: Whether to freeze encoder parameters
        freeze_decoder: Whether to freeze decoder parameters

    Returns:
        Training history dictionary
    """
    model = model.to(device)

    # Freeze specified parts of the model
    for param in model.encoder.parameters():
        param.requires_grad = not freeze_encoder
    for param in model.decoder.parameters():
        param.requires_grad = not freeze_decoder

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    criterion = nn.MSELoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'recon_loss': [],
        'dist_loss': [],
        'val_recon_loss': [],
        'val_dist_loss': []
    }

    print(f'Training GAGA on device: {device}')
    print(f'Encoder frozen: {freeze_encoder}, Decoder frozen: {freeze_decoder}')
    print(f'Reconstruction weight: {recon_weight}, Distance weight: {dist_weight}')

    epoch_pbar = tqdm(range(num_epochs), desc='Epochs')
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0.0
        recon_loss_sum = 0.0
        dist_loss_sum = 0.0

        for batch in train_loader:
            data = batch['x'].to(device)
            distances = batch['d'].to(device)

            optimizer.zero_grad()

            # Forward pass
            embedding = model.encode(data)
            reconstructed = model.decode(embedding)

            # Reconstruction loss
            recon_loss = criterion(reconstructed, data)

            # Distance preservation loss (compare pairwise distances in latent space)
            latent_distances = torch.pdist(embedding)
            dist_loss = criterion(latent_distances, distances)

            # Combined loss
            loss = recon_weight * recon_loss + dist_weight * dist_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            dist_loss_sum += dist_loss.item()

        # Average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_recon_loss = recon_loss_sum / len(train_loader)
        avg_dist_loss = dist_loss_sum / len(train_loader)

        history['train_loss'].append(avg_train_loss)
        history['recon_loss'].append(avg_recon_loss)
        history['dist_loss'].append(avg_dist_loss)

        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_recon_loss_sum = 0.0
            val_dist_loss_sum = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    data = batch['x'].to(device)
                    distances = batch['d'].to(device)

                    embedding = model.encode(data)
                    reconstructed = model.decode(embedding)

                    recon_loss = criterion(reconstructed, data)
                    latent_distances = torch.pdist(embedding)
                    dist_loss = criterion(latent_distances, distances)
                    loss = recon_weight * recon_loss + dist_weight * dist_loss

                    val_loss += loss.item()
                    val_recon_loss_sum += recon_loss.item()
                    val_dist_loss_sum += dist_loss.item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_recon_loss = val_recon_loss_sum / len(val_loader)
            avg_val_dist_loss = val_dist_loss_sum / len(val_loader)

            history['val_loss'].append(avg_val_loss)
            history['val_recon_loss'].append(avg_val_recon_loss)
            history['val_dist_loss'].append(avg_val_dist_loss)

            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{avg_val_loss:.4f}',
                'recon': f'{avg_recon_loss:.4f}',
                'dist': f'{avg_dist_loss:.4f}'
            })
        else:
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'recon': f'{avg_recon_loss:.4f}',
                'dist': f'{avg_dist_loss:.4f}'
            })

    # Unfreeze all parameters after training
    for param in model.parameters():
        param.requires_grad = True

    return history


class PointCloudDataset(torch.utils.data.Dataset):
    """
    Dataset for point cloud data with distance matrices.
    """
    def __init__(self, pointcloud, distances, batch_size=64, shuffle=True):
        self.pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
        self.distances = torch.tensor(distances, dtype=torch.float32)
        self.shuffle = shuffle
        self.batch_size = batch_size

        # Generate random permutation once for all batches if shuffling
        if self.shuffle:
            self.perm = torch.randperm(len(self.pointcloud))
        else:
            self.perm = None

    def __len__(self):
        # Return number of complete batches
        return (len(self.pointcloud) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        '''
        Returns a batch of pointclouds and their distances
        batch['x'] = [B, D]
        batch['d'] = [B, B(B-1)/2] (upper triangular), assuming symmetric distance matrix
        '''
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.pointcloud))

        if self.shuffle:
            batch_idxs = self.perm[start_idx:end_idx]
        else:
            batch_idxs = torch.arange(start_idx, end_idx)

        batch = {}
        batch['x'] = self.pointcloud[batch_idxs]
        dist_mat = self.distances[batch_idxs][:, batch_idxs]
        batch['d'] = dist_mat[np.triu_indices(dist_mat.size(0), k=1)]
        return batch


def dataloader_from_pc(pointcloud, distances, batch_size=64, shuffle=True):
    """Create DataLoader from point cloud and distance matrix."""
    dataset = PointCloudDataset(pointcloud=pointcloud, distances=distances,
                               batch_size=batch_size, shuffle=shuffle)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=shuffle)
    return dataloader


def train_valid_loader_from_pc(pointcloud, distances, batch_size=64,
                              train_valid_split=0.8, shuffle=True, seed=42):
    """Split point cloud data into train/validation sets."""
    X = pointcloud
    D = distances

    np.random.seed(seed)

    if shuffle:
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        D = D[idxs][:, idxs]

    split_idx = int(len(X) * train_valid_split)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    D_train = D[:split_idx, :split_idx]
    D_test = D[split_idx:, split_idx:]

    trainloader = dataloader_from_pc(X_train, D_train, batch_size)
    testloader = dataloader_from_pc(X_test, D_test, batch_size)

    return trainloader, testloader


class RowStochasticDataset(torch.utils.data.Dataset):
    """
    Dataset for point cloud with PHATE-based row stochastic matrix.
    """
    def __init__(self, data_name: str, X: np.ndarray, emb_dim: int = 2,
                 knn: int = 5, n_landmark: int = 5000, shuffle: bool = True):
        super().__init__()
        self.data_name = data_name
        self.X = X.astype(np.float32)
        self.pointcloud = torch.tensor(self.X)
        self.emb_dim = emb_dim
        self.knn = knn
        self.n_landmark = n_landmark
        self.shuffle = shuffle

        # Compute PHATE and row stochastic matrix
        self.row_stochastic_matrix, self.phate_embed = self._compute_phate()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index) -> Tuple[int, np.ndarray]:
        return index, self.X[index]

    def __repr__(self) -> str:
        return f"RowStochasticDataset({self.data_name}, {self.X.shape})"

    def _compute_phate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute PHATE embedding and row stochastic transition matrix."""
        phate_op = phate.PHATE(
            verbose=True,
            n_components=self.emb_dim,
            knn=self.knn,
            n_landmark=self.n_landmark
        ).fit(self.X)

        phate_embed = torch.Tensor(phate_op.transform(self.X))
        diff_potential = phate_op.diff_potential
        diff_op_t = np.exp(-1 * diff_potential)
        row_stochastic_matrix = torch.Tensor(diff_op_t)

        print(f'Row stochastic matrix shape: {row_stochastic_matrix.shape}')
        print(f'Row sums check: {np.allclose(row_stochastic_matrix.sum(axis=1), 1)}')

        return row_stochastic_matrix, phate_embed
