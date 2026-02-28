import torch
import torch.nn as nn


class ODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, momentum_beta=0.0):
        super().__init__()
        self.momentum_beta = momentum_beta
        self.previous_v = None
        
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim), # additional dim for time t.
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
        # Kaiming initialization for SiLU (better than default for ReLU-like activations)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def reset_momentum(self):
        """Reset momentum state before integration"""
        self.previous_v = None

    def forward(self, t, x):
        # t is scalar, x is [batch_size, input_dim]
        # Expand t to [batch_size, 1] to match x's batch dimension
        t_expanded = t.expand(x.size(0), 1)
        input = torch.cat([t_expanded, x], dim=-1)
        dxdt = self.model(input)
        
        # Apply momentum if enabled
        if self.momentum_beta > 0.0:
            if self.previous_v is None or self.previous_v.shape[0] != x.shape[0]:
                self.previous_v = torch.zeros_like(dxdt)
            dxdt = self.momentum_beta * self.previous_v + (1 - self.momentum_beta) * dxdt
            self.previous_v = dxdt.detach()
        
        return dxdt