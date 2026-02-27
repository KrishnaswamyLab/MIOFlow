import torch
import torch.nn as nn

class SDEFunc(nn.Module):
    """
    Neural SDE with diagonal noise.
    Implements both drift (f) and diffusion (g) terms.
    """
    noise_type = 'diagonal'
    sde_type = 'ito'
    
    def __init__(self, input_dim, hidden_dim, diffusion_scale=0.1, diffusion_init_scale=0.1, momentum_beta=0.0):
        super().__init__()
        self.diffusion_scale = diffusion_scale
        self.diffusion_init_scale = diffusion_init_scale
        self.momentum_beta = momentum_beta
        self.previous_v = None
        
        # Drift network (same complexity as ODEFunc)
        self.drift_net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
        # Diffusion network (simpler: 1 hidden layer)
        # Softplus ensures positive diffusion: g ∈ [0, ∞)
        self.diffusion_net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus(),
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Kaiming initialization for both drift and diffusion networks
        for net in [self.drift_net, self.diffusion_net]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # Scale down diffusion network for stability
        # diffusion_init_scale controls initial noise level
        # 0.1 = small noise, 0.0 = pure ODE initially, 1.0 = no scaling
        if self.diffusion_init_scale != 1.0:
            with torch.no_grad():
                for m in self.diffusion_net.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.data *= self.diffusion_init_scale
    
    def reset_momentum(self):
        """Reset momentum state before integration"""
        self.previous_v = None
    
    def f(self, t, x):
        """Drift term"""
        t_expanded = t.expand(x.size(0), 1)
        input = torch.cat([t_expanded, x], dim=-1)
        drift = self.drift_net(input)
        
        # Apply momentum if enabled
        if self.momentum_beta > 0.0:
            if self.previous_v is None or self.previous_v.shape[0] != x.shape[0]:
                self.previous_v = torch.zeros_like(drift)
            drift = self.momentum_beta * self.previous_v + (1 - self.momentum_beta) * drift
            self.previous_v = drift.detach()
        
        return drift
    
    def g(self, t, x):
        """Diffusion term - scaled by diffusion_scale"""
        t_expanded = t.expand(x.size(0), 1)
        input = torch.cat([t_expanded, x], dim=-1)
        return self.diffusion_scale * self.diffusion_net(input)