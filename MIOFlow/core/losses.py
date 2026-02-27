import torch
import ot
from torchdiffeq import odeint
import torchsde

def ot_loss(source, target, source_mass=None, target_mass=None, reg=0.1):
    use_sinkhorn = source_mass is not None or target_mass is not None
    
    if use_sinkhorn:
        # Mask out explicitly dead points so they don't get forced into the balanced EMD matching
        mask_threshold = 0.05
        mask = source_mass > mask_threshold
        
        if not mask.any():
            # If all points in the batch are dead, no OT cost to transport them
            return torch.tensor(0.0, device=source.device, requires_grad=True)
            
        active_source = source[mask]
        active_mass = source_mass[mask]
        M = torch.cdist(active_source, target)**2

        # Standardize mass to sum to 1.0 for balanced OT
        mu = active_mass / active_mass.sum()
        nu = torch.tensor(ot.unif(target.size(0)), dtype=target.dtype, device=target.device)
        
        # We detach the mass inputs so the ODE solver only sees the spatial vector field gradients
        loss = ot.emd2(mu.detach(), nu.detach(), M)
        return loss
    else:
        M = torch.cdist(source, target)**2
        # Standard balanced uniform EMD for regular MIOFlow
        mu = torch.tensor(ot.unif(source.size(0)), dtype=source.dtype, device=source.device)
        nu = torch.tensor(ot.unif(target.size(0)), dtype=target.dtype, device=target.device)
        return ot.emd2(mu, nu, M)

def energy_loss(model, x0, t_seq, is_sde=False, dt=0.1, lambda_f=1.0, lambda_g=0.0):
    """
    Compute energy loss by evaluating vector field magnitude along the trajectory.
    For ODE: penalizes ||f||^2
    For SDE: penalizes lambda_f * ||f||^2 + lambda_g * ||g||^2
    
    Recommended settings:
    - lambda_f=1.0, lambda_g=0.0: Only penalize drift (diffusion controlled by scale)
    - lambda_f=1.0, lambda_g=1.0: Penalize both equally
    - lambda_f=1.0, lambda_g=2.0: Penalize diffusion more (keep noise small)

    Args:
        model: ODEFunc or SDEFunc model
        x0: Initial points [batch_size, input_dim]
        t_seq: Time sequence [num_times]
        is_sde: Whether model is SDE (has f and g methods)
        dt: Time step for SDE integration
        lambda_f: Weight for drift penalty (default 1.0)
        lambda_g: Weight for diffusion penalty (default 0.0 - don't penalize)

    Returns:
        Energy loss (mean squared magnitude of vector field along trajectory)
    """
    # Reset momentum before integration
    model.reset_momentum()
    
    # Compute the full trajectory
    if is_sde:
        trajectory = torchsde.sdeint_adjoint(model, x0, t_seq, dt=dt, method='euler')
    else:
        trajectory = odeint(model, x0, t_seq)

    total_energy = 0.0
    num_evaluations = 0

    # Evaluate vector field at each point along the trajectory
    for i, t_val in enumerate(t_seq):
        x_t = trajectory[i]  # [batch_size, input_dim]
        t_tensor = t_val.clone().detach() if torch.is_tensor(t_val) else torch.tensor(t_val, device=x_t.device, dtype=x_t.dtype)

        if is_sde:
            # For SDE: separate penalties for drift and diffusion
            f_val = model.f(t_tensor, x_t)
            g_val = model.g(t_tensor, x_t)
            total_energy += lambda_f * torch.sum(f_val ** 2) + lambda_g * torch.sum(g_val ** 2)
        else:
            # For ODE: energy from vector field
            dx_dt = model(t_tensor, x_t)
            total_energy += torch.sum(dx_dt ** 2)
        
        num_evaluations += x_t.size(0)

    return total_energy / num_evaluations

def density_loss(source, target, source_mass=None, top_k=5, hinge_value=0.01):
    """
    Density loss that encourages points to be close to target distribution.
    Uses hinge loss on k-nearest neighbor distances.
    Masks out points that have mass <= 0.05.
    """
    if source_mass is not None:
        mask = source_mass > 0.05
        if not mask.any():
            return torch.tensor(0.0, device=source.device, requires_grad=True)
        active_source = source[mask]
    else:
        active_source = source
        
    c_dist = torch.cdist(active_source, target)
    values, _ = torch.topk(c_dist, top_k, dim=1, largest=False, sorted=False)
    values = torch.clamp(values - hinge_value, min=0.0)
    return torch.mean(values)