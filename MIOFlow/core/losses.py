import torch
import ot
from torchdiffeq import odeint

def ot_loss(source, target):
    mu = torch.tensor(ot.unif(source.size(0)), dtype=source.dtype, device=source.device)
    nu = torch.tensor(ot.unif(target.size(0)), dtype=target.dtype, device=target.device)
    M = torch.cdist(source, target) ** 2
    return ot.emd2(mu, nu, M)


def energy_loss(model, x0, t_seq):
    """Penalizes large vector field magnitudes along the ODE trajectory."""
    trajectory = odeint(model, x0, t_seq)  # (T, B, D)
    total_energy = 0.0
    num_evaluations = 0
    for i, t_val in enumerate(t_seq):
        x_t = trajectory[i]
        t_tensor = torch.full((x_t.size(0), 1), t_val, device=x_t.device, dtype=x_t.dtype)
        dx_dt = model(t_tensor, x_t)
        total_energy += torch.sum(dx_dt ** 2)
        num_evaluations += x_t.size(0)
    return total_energy / num_evaluations


def density_loss(source, target, top_k=5, hinge_value=0.01):
    """Hinge loss on k-nearest-neighbor distances to target distribution."""
    c_dist = torch.cdist(source, target)
    values, _ = torch.topk(c_dist, top_k, dim=1, largest=False, sorted=False)
    return torch.mean(torch.clamp(values - hinge_value, min=0.0))