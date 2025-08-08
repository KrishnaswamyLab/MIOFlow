
__all__ = ['ToyODE', 'GrowthRateModel', 'make_model', 'Autoencoder', 'ToyModel', 'ToySDEModel', 'ConditionalODE',
           'ConditionalModel', 'ConditionalSDEModel']


import torch
import torch.nn as nn
from torch.nn  import functional as F 
from torchdiffeq import odeint_adjoint as odeint
import torchsde
import itertools

# TODO: ToyGeo is not implemented.

class ToyODE(nn.Module):
    """ 
    ODE derivative network
    
    feature_dims (int) default '5': dimension of the inputs, either in ambient space or embedded space.
    layer (list of int) defaulf ''[64]'': the hidden layers of the network.
    activation (torch.nn) default '"ReLU"': activation function applied in between layers.
    scales (NoneType|list of float) default 'None': the initial scale for the noise in the trajectories. One scale per bin, add more if using an adaptative ODE solver.
    n_aug (int) default '1': number of added dimensions to the input of the network. Total dimensions are features_dim + 1 (time) + n_aug. 
    
    Method
    forward (Callable)
        forward pass of the ODE derivative network.
        Parameters:
        t (torch.tensor): time of the evaluation.
        x (torch.tensor): position of the evalutation.
        Return:
        derivative at time t and position x.   
    """
    def __init__(
        self, 
        feature_dims=5,
        layers=[64],
        activation='ReLU',
        scales=None,
        n_aug=2,
        momentum_beta = 0.0,
        condition_dims=0, # TODO: make it a dict instead of assuming first!
    ):
        super(ToyODE, self).__init__()
        steps = [feature_dims+1+n_aug, *layers, feature_dims]
        pairs = zip(steps, steps[1:])
        self.condition_dim = condition_dims # TODO: make it a dict instead of assuming first!
        chain = list(itertools.chain(*list(zip(
            map(lambda e: nn.Linear(*e), pairs), 
            itertools.repeat(getattr(nn, activation)())
        ))))[:-1]

        self.chain = chain
        self.seq = (nn.Sequential(*chain))
        # self.initialize_final_layer()

        self.alpha = nn.Parameter(torch.tensor(scales, requires_grad=True).float()) if scales is not None else None
        self.n_aug = n_aug  

        self.momentum_beta = momentum_beta
        self.previous_v = None

    # def initialize_final_layer(self):
    #     # For the final layer of fc_g, use Xavier uniform initialization for weights
    #     # and set the bias to a small positive constant (e.g., 0.1)
    #     final_layer = self.seq[-1]
    #     if isinstance(final_layer, nn.Linear):
    #         nn.init.xavier_uniform_(final_layer.weight)
    #         nn.init.constant_(final_layer.bias, 0.1)    
        
    def forward(self, t, x): #NOTE the forward pass when we use torchdiffeq must be forward(self,t,x)
        zero = torch.tensor([0]).cuda() if x.is_cuda else torch.tensor([0])
        zeros = zero.repeat(x.size()[0],self.n_aug)
        time = t.repeat(x.size()[0],1)
        aug = torch.cat((x,time,zeros),dim=1)
        dxdt = self.seq(aug)
        if self.alpha is not None:
            z = torch.randn(x.size(),requires_grad=False).cuda() if x.is_cuda else torch.randn(x.size(),requires_grad=False)
            dxdt = dxdt + z*self.alpha[int(t-1)]

        if self.momentum_beta > 0.0:
            if self.previous_v is None or self.previous_v.shape[0] != x.shape[0]:
                self.previous_v = torch.zeros_like(dxdt)

            dxdt = self.momentum_beta * self.previous_v + (1 - self.momentum_beta) * dxdt
            self.previous_v = dxdt.detach()
        return dxdt



class GrowthRateModel(nn.Module):
    def __init__(self, feature_dims, condition_dims, layers=[64], activation='ReLU', use_time=True):
        super(GrowthRateModel, self).__init__()
        self.feature_dims = feature_dims
        self.condition_dims = condition_dims
        self.layers = layers
        self.activation = activation
        self.use_time = use_time
        self.mlp = nn.Sequential()
        self.mlp.append(nn.Linear(feature_dims + condition_dims + (1 if use_time else 0), layers[0]))
        self.mlp.append(getattr(nn, activation)())
        for i in range(len(layers)-1):
            self.mlp.append(nn.Linear(layers[i], layers[i+1]))
            self.mlp.append(getattr(nn, activation)())
        self.mlp.append(nn.Linear(layers[-1], 1)) # output the growth rate

    def forward(self, x, c=None, t=None):
        if self.condition_dims > 0:
            assert c is not None, "Condition must be provided"
            x = torch.cat([x, c], dim=1)
        if self.use_time:
            t = t.unsqueeze(1)
            x = torch.cat([x, t], dim=1)
        return self.mlp(x)

"""TODO: I want to deprecate this function."""
def make_model(
    feature_dims=5,
    layers=[64],
    output_dims=5,
    activation='ReLU',
    which='ode',
    method='rk4',
    rtol=None,
    atol=None,
    scales=None,
    n_aug=2,
    noise_type='diagonal', sde_type='ito',
    adjoint_method=None,
    use_norm=False,
    use_cuda=False,
    in_features=2, out_features=2, gunc=None,
    n_conditions=0,
    momentum_beta = 0.0,
):
    """
    Creates the 'ode' model or 'sde' model or the Geodesic Autoencoder. 
    See the parameters of the respective classes.
    """
    if which == 'ode' and n_conditions == 0:
        ode = ToyODE(feature_dims, layers, activation,scales,n_aug)
        model = ToyModel(ode,method,rtol, atol, use_norm=use_norm)
    elif which == 'ode' and n_conditions > 0:
        ode = ConditionalODE(feature_dims, n_conditions, layers, activation, scales, n_aug, momentum_beta=momentum_beta)
        model = ConditionalModel(ode, method, rtol, atol, use_norm=use_norm)
    elif which == 'sde' and n_conditions == 0:
        ode = ToyODE(feature_dims, layers, activation,scales,n_aug, momentum_beta=momentum_beta)
        gunc = ToyODE(feature_dims, layers, activation,scales,n_aug)
        model = ToySDEModel(
            ode, method, noise_type, sde_type,
            in_features=in_features, out_features=out_features, 
            gunc=gunc, 
            adjoint_method=adjoint_method, 
            use_norm=use_norm
        )
    elif which == 'sde' and n_conditions > 0:
        ode = ConditionalODE(feature_dims, n_conditions, layers, activation, scales, n_aug, momentum_beta=momentum_beta)
        gunc = ToyODE(feature_dims, layers, activation, scales, n_aug)
        model = ConditionalSDEModel(
            ode, method, noise_type, sde_type,
            in_features=in_features, out_features=out_features, 
            gunc=gunc, 
            adjoint_method=adjoint_method, 
            use_norm=use_norm
        )
    else:
        model = ToyGeo(feature_dims, layers, output_dims, activation)
    if use_cuda:
        model.cuda()
    return model 


class Autoencoder(nn.Module):
    """ 
    Geodesic Autoencoder
    
    encoder_layers (list of int) default '[100, 100, 20]': encoder_layers[0] is the feature dimension, and encoder_layers[-1] the embedded dimension.
    decoder_layers (list of int) defaulf '[20, 100, 100]': decoder_layers[0] is the embbeded dim and decoder_layers[-1] the feature dim.
    activation (torch.nn) default '"Tanh"': activation function applied in between layers.
    use_cuda (bool) default to False: Whether to use GPU or CPU.
    
    Method
    encode
        forward pass of the encoder
        x (torch.tensor): observations
        Return:
        the encoded observations
    decode
        forward pass of the decoder
        z (torch.tensor): embedded observations
        Return:
        the decoded observations
    forward (Callable):
        full forward pass, encoder and decoder
        x (torch.tensor): observations
        Return:
        denoised observations
    """

    def __init__(
        self,
        encoder_layers = [100, 100, 20],
        decoder_layers = [20, 100, 100],
        activation = 'Tanh',
        use_cuda = False
    ):        
        super(Autoencoder, self).__init__()
        if decoder_layers is None:
            decoder_layers = [*encoder_layers[::-1]]
        device = 'cuda' if use_cuda else 'cpu'
        
        encoder_shapes = list(zip(encoder_layers, encoder_layers[1:]))
        decoder_shapes = list(zip(decoder_layers, decoder_layers[1:]))
        
        encoder_linear = list(map(lambda a: nn.Linear(*a), encoder_shapes))
        decoder_linear = list(map(lambda a: nn.Linear(*a), decoder_shapes))
        
        encoder_riffle = list(itertools.chain(*zip(encoder_linear, itertools.repeat(getattr(nn, activation)()))))[:-1]
        encoder = nn.Sequential(*encoder_riffle).to(device)
        
        decoder_riffle = list(itertools.chain(*zip(decoder_linear, itertools.repeat(getattr(nn, activation)()))))[:-1]

        decoder = nn.Sequential(*decoder_riffle).to(device)
        self.encoder = encoder
        self.decoder = decoder

        
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class ToyModel(nn.Module):
    """ 
    Neural ODE
        func (nn.Module): The network modeling the derivative.
        method (str) defaulf '"rk4"': any methods from torchdiffeq.
        rtol (NoneType | float): the relative tolerance of the ODE solver.
        atol (NoneType | float): the absolute tolerance. of the ODE solver.
        use_norm (bool): if True keeps the norm of func.
        norm (list of torch.tensor): the norm of the derivative.
        
        Method
        forward (Callable)
            x (torch.tensor): the initial sample
            t (torch.tensor) time points where we suppose x is from t[0]
            return the last sample or the whole seq.      
    """
    
    def __init__(self, func, method='rk4', rtol=None, atol=None, use_norm=False):
        super(ToyModel, self).__init__()        
        self.func = func
        self.method = method
        self.rtol=rtol
        self.atol=atol
        self.use_norm = use_norm
        self.norm=[]

    def forward(self, x, t, return_whole_sequence=False):

        if self.use_norm:
            for time in t: 
                self.norm.append(torch.linalg.norm(self.func(time,x)).pow(2))
        if self.atol is None and self.rtol is None:
            x = odeint(self.func,x ,t, method=self.method)
        elif self.atol is not None and self.rtol is None:
            x = odeint(self.func,x ,t, method=self.method, atol=self.atol)
        elif self.atol is None and self.rtol is not None:
            x = odeint(self.func,x ,t, method=self.method, rtol=self.rtol)
        else: 
            x = odeint(self.func,x ,t, method=self.method, atol=self.atol, rtol=self.rtol)          
       
        x = x[-1] if not return_whole_sequence else x
        return x

    def train(self, mode=True):
        # reset the norm
        super().train(mode)
        self.norm = []

    def eval(self):
        super().eval()
        self.norm = []


class ToySDEModel(nn.Module):
    """ 
    Neural SDE model
        func (nn.Module): drift term.
        genc (nn.Module): diffusion term.
        method (str): method of the SDE solver.
        
        Method
        forward (Callable)
            x (torch.tensor): the initial sample
            t (torch.tensor) time points where we suppose x is from t[0]
            return the last sample or the whole seq.  
    """
    
    def __init__(self, func, method='euler', noise_type='diagonal', sde_type='ito', 
    in_features=2, out_features=2, gunc=None, dt=0.1, adjoint_method=None, use_norm=False):
        super(ToySDEModel, self).__init__()        
        self.func = func
        self.method = method
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.adjoint_method = adjoint_method
        self.use_norm = use_norm
        self.norm = []
        if gunc is None:
            self._gunc_args = 'y'
            self.gunc = nn.Linear(in_features, out_features)
        else:
            self._gunc_args = 't,y'
            self.gunc = gunc

        self.dt = dt
        
    def f(self, t, y):
        return self.func(t, y)

    def g(self, t, y):
        return self.gunc(t, y) if self._gunc_args == 't,y' else self.gunc(y)

    def forward(self, x, t, return_whole_sequence=False, dt=None):
        if self.use_norm:
            for time in t: 
                f_norm = torch.linalg.norm(self.func(time,x)).pow(2)
                g_norm = torch.linalg.norm(self.gunc(time,x)).pow(2)
                self.norm.append(f_norm + g_norm)
            
        dt = self.dt if self.dt is not None else 0.1 if dt is None else dt        
        if self.adjoint_method is not None:
            x = torchsde.sdeint_adjoint(self, x, t, method=self.method, adjoint_method=self.adjoint_method, dt=dt)
        else:
            x = torchsde.sdeint(self, x, t, method=self.method, dt=dt)
       
        x = x[-1] if not return_whole_sequence else x
        return x

    def train(self, mode=True):
        # reset the norm
        super().train(mode)
        self.norm = []

    def eval(self):
        super().eval()
        self.norm = []



class ConditionalODE(nn.Module):
    """ 
    ODE derivative network

    WARNING: Only supports local training!

    
    feature_dims (int) default '5': dimension of the inputs, either in ambient space or embedded space.
    layer (list of int) defaulf ''[64]'': the hidden layers of the network.
    activation (torch.nn) default '"ReLU"': activation function applied in between layers.
    scales (NoneType|list of float) default 'None': the initial scale for the noise in the trajectories. One scale per bin, add more if using an adaptative ODE solver.
    n_aug (int) default '1': number of added dimensions to the input of the network. Total dimensions are features_dim + 1 (time) + n_aug. 
    
    Method
    forward (Callable)
        forward pass of the ODE derivative network.
        Parameters:
        t (torch.tensor): time of the evaluation.
        x (torch.tensor): position of the evalutation.
        Return:
        derivative at time t and position x.   
    """
    def __init__(
        self, 
        feature_dims=5,
        condition_dims=2,
        layers=[64],
        activation='ReLU',
        scales=None,
        n_aug=2,
        time_homogeneous=True,
        momentum_beta = 0.0,
    ):
        super(ConditionalODE, self).__init__()
        self.time_homogeneous = time_homogeneous
        self.time_dim = 1 if 0 else 1
        self.condition_dim = condition_dims # TODO: make it a dict instead of assuming first!
        steps = [feature_dims+self.condition_dim+self.time_dim+n_aug, *layers, feature_dims]
        pairs = zip(steps, steps[1:])

        chain = list(itertools.chain(*list(zip(
            map(lambda e: nn.Linear(*e), pairs), 
            itertools.repeat(getattr(nn, activation)())
        ))))[:-1]

        self.chain = chain
        self.seq = (nn.Sequential(*chain))
        
        self.alpha = nn.Parameter(torch.tensor(scales, requires_grad=True).float()) if scales is not None else None
        self.n_aug = n_aug
        self.momentum_beta = momentum_beta
        self.previous_v = None
        
    def set_condition(self, condition):
        self.condition = condition

    def forward(self, t, x): #NOTE the forward pass when we use torchdiffeq must be forward(self,t,x)
        zero = torch.tensor([0]).cuda() if x.is_cuda else torch.tensor([0])
        zeros = zero.repeat(x.size()[0],self.n_aug)
        time = t.repeat(x.size()[0],self.time_dim)
        aug = torch.cat((x,self.condition,time,zeros),dim=1)
        
        dxdt = self.seq(aug)
        if self.alpha is not None:
            z = torch.randn(x.size(),requires_grad=False).cuda() if x.is_cuda else torch.randn(x.size(),requires_grad=False)
            dxdt = dxdt + z*self.alpha[int(t-1)]

        if self.momentum_beta > 0.0:
            if self.previous_v is None or self.previous_v.shape[0] != x.shape[0]:
                self.previous_v = torch.zeros_like(dxdt)

            dxdt = self.momentum_beta * self.previous_v + (1 - self.momentum_beta) * dxdt
            self.previous_v = dxdt.detach()  # Update for next step 
        return dxdt
    


class ConditionalModel(nn.Module):
    """ 
    Neural ODE
        func (nn.Module): The network modeling the derivative.
        method (str) defaulf '"rk4"': any methods from torchdiffeq.
        rtol (NoneType | float): the relative tolerance of the ODE solver.
        atol (NoneType | float): the absolute tolerance. of the ODE solver.
        use_norm (bool): if True keeps the norm of func.
        norm (list of torch.tensor): the norm of the derivative.
        
        Method
        forward (Callable)
            x (torch.tensor): the initial sample
            t (torch.tensor) time points where we suppose x is from t[0]
            return the last sample or the whole seq.      
    """
    
    def __init__(self, func, method='rk4', rtol=None, atol=None, use_norm=False):
        super(ConditionalModel, self).__init__()        
        self.func = func
        self.method = method
        self.rtol=rtol
        self.atol=atol
        self.use_norm = use_norm
        self.norm=[]

    def forward(self, x, t, return_whole_sequence=False):
        # assume the last `condition_dim` dimensions of x is the condition
        x = x[:,:-self.func.condition_dim]
        c = x[:,-self.func.condition_dim:]
        self.func.set_condition(c)
        if self.use_norm:
            for time in t: 
                self.norm.append(torch.linalg.norm(self.func(time,x)).pow(2))
        if self.atol is None and self.rtol is None:
            x = odeint(self.func,x ,t, method=self.method)
        elif self.atol is not None and self.rtol is None:
            x = odeint(self.func,x ,t, method=self.method, atol=self.atol)
        elif self.atol is None and self.rtol is not None:
            x = odeint(self.func,x ,t, method=self.method, rtol=self.rtol)
        else: 
            x = odeint(self.func,x ,t, method=self.method, atol=self.atol, rtol=self.rtol)          
       
        x = x[-1] if not return_whole_sequence else x
        return x
    
    def train(self, mode=True):
        # reset the norm
        super().train(mode)
        self.norm = []

    def eval(self):
        super().eval()
        self.norm = []


class ConditionalSDEModel(ToySDEModel):
    def __init__(self, func, method='euler', noise_type='diagonal', sde_type='ito', 
    in_features=2, out_features=2, gunc=None, dt=0.1, adjoint_method=None, use_norm=False):
        super(ConditionalSDEModel, self).__init__(func, method, noise_type, sde_type, in_features, out_features, gunc, dt, adjoint_method, use_norm)

    def forward(self, x, t, return_whole_sequence=False, dt=None):
        # assume the last `condition_dim` dimensions of x is the condition
        x = x[:,:-self.func.condition_dim] # TODO: make it a dict instead of assuming first!
        c = x[:,-self.func.condition_dim:]
        self.func.set_condition(c)
        return super().forward(x, t, return_whole_sequence, dt)
