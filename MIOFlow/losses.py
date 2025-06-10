
__all__ = ['MMD_loss', 'OT_loss', 'ot_loss_given_plan', 'covariance_loss', 'Density_loss', 'Local_density_loss']


import numpy as np
import torch
import torch.nn as nn
import ot

# TODO: ToyGeo is not implemented.

class MMD_loss(nn.Module):
    '''
    https://github.com/ZongxianLee/MMD_Loss.Pytorch/blob/master/mmd_loss.py
    '''
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

class OT_loss(nn.Module):
    _valid = 'emd sinkhorn sinkhorn_knopp_unbalanced'.split()

    def __init__(self, which='emd', use_cuda=True, detach_mass=True, detach_dist_for_plan=True, sinkhorn_lambda=2.0, covariance_lambda=0.0):
        if which not in self._valid:
            raise ValueError(f'{which} not known ({self._valid})')
        elif which == 'emd':
            self.fn = lambda m, n, M: ot.emd(m, n, M)
        elif which == 'sinkhorn':
            self.fn = lambda m, n, M : ot.sinkhorn(m, n, M, sinkhorn_lambda)
        elif which == 'sinkhorn_knopp_unbalanced':
            self.fn = lambda m, n, M : ot.unbalanced.sinkhorn_knopp_unbalanced(m, n, M, 1.0, 1.0)
        else:
            pass
        self.use_cuda=use_cuda
        self.detach_dist_for_plan = detach_dist_for_plan
        self.detach_mass = detach_mass
        self.covariance_lambda = covariance_lambda

    def __call__(self, source, target, source_mass=None, target_mass=None, use_cuda=None, return_plan=False):
        """
        DEPRECATING the use_cuda argument. Now inferring from the source and target.
        """
        # if use_cuda is None:
            # use_cuda = self.use_cuda
        if source_mass is None:
            mu = torch.tensor(ot.unif(source.size()[0]), dtype=source.dtype, device=source.device)
        else:
            mu = (source_mass)/(source_mass).sum()
        if target_mass is None:
            nu = torch.tensor(ot.unif(target.size()[0]), dtype=target.dtype, device=target.device)
        else:
            nu = (target_mass)/(target_mass).sum()
        M = torch.cdist(source, target)**2
        if self.detach_dist_for_plan:
            # pi = self.fn(mu, nu, M.detach().cpu())
            pi = self.fn(mu, nu, M.detach())
        else:
            pi = self.fn(mu, nu, M)
        if type(pi) is np.ndarray:
            pi = torch.tensor(pi)
        elif type(pi) is torch.Tensor:
            if self.detach_mass:
                pi = pi.clone().detach()
            # pi = pi.cuda() if use_cuda else pi
        # M = M.to(pi.device)
        loss = torch.sum(pi * M)
        
        if self.covariance_lambda > 0:
            loss += self.covariance_lambda * covariance_loss(source, target)
        
        if return_plan:
            return loss, pi
        else:
            return loss
        
        
def ot_loss_given_plan(plan, source, target):
    M = torch.cdist(source, target)**2
    loss = torch.sum(plan * M)
    return loss

def covariance_loss(source, target):
    # Center the data
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    
    # Compute empirical covariance matrices (using unbiased estimate)
    cov_source = source_centered.t() @ source_centered / (source.size(0) - 1)
    cov_target = target_centered.t() @ target_centered / (target.size(0) - 1)
    
    # Compute Frobenius norm of the difference
    loss = torch.norm(cov_source - cov_target, p='fro')
    return loss


class Density_loss(nn.Module):
    def __init__(self, hinge_value=0.01):
        self.hinge_value = hinge_value
        pass

    def __call__(self, source, target, groups = None, to_ignore = None, top_k = 5):
        if groups is not None:
            # for global loss
            c_dist = torch.stack([
                torch.cdist(source[i], target[i]) 
                # NOTE: check if this should be 1 indexed
                for i in range(1,len(groups))
                if groups[i] != to_ignore
            ])
        else:
            # for local loss
             c_dist = torch.stack([
                torch.cdist(source, target)                 
            ])
        values, _ = torch.topk(c_dist, top_k, dim=2, largest=False, sorted=False)
        values -= self.hinge_value
        values[values<0] = 0
        loss = torch.mean(values)
        return loss

class Local_density_loss(nn.Module):
    def __init__(self):
        pass

    def __call__(self, sources, targets, groups, to_ignore, top_k = 5):
        # print(source, target)
        # c_dist = torch.cdist(source, target) 
        c_dist = torch.stack([
            torch.cdist(sources[i], targets[i]) 
            # NOTE: check if should be from range 1 or not.
            for i in range(1, len(groups))
            if groups[i] != to_ignore
        ])
        vals, inds = torch.topk(c_dist, top_k, dim=2, largest=False, sorted=False)
        values = vals[inds[inds]]
        loss = torch.mean(values)
        return loss