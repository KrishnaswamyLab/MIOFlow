from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch

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
