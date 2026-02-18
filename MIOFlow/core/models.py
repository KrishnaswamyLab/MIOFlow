import torch
import torch.nn as nn


class ODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # additional dim for time t
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, t, x):
        t_expanded = t.expand(x.size(0), 1)
        return self.model(torch.cat([t_expanded, x], dim=-1))