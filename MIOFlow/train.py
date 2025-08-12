
__all__ = ['train', 'train_ae', 'training_regimen']


import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from tqdm.notebook import tqdm

from .utils import sample, generate_steps
from .losses import MMD_loss, OT_loss, Density_loss, Local_density_loss, ot_loss_given_plan, covariance_loss

from .plots import plot_comparision, plot_losses
from .eval import generate_plot_data


def train(
    model, df, groups, optimizer, n_batches=20, 
    criterion=MMD_loss(),
    use_cuda=False,

    sample_size=(100, ),
    sample_with_replacement=False,

    local_loss=True,
    global_loss=False,

    hold_one_out=False,
    hold_out='random',
    apply_losses_in_time=True,

    top_k = 5,
    hinge_value = 0.01,
    use_density_loss=True,
    # use_local_density=False,

    lambda_density = 1.0,

    autoencoder=None, 
    use_emb=True,
    use_gae=False,

    use_gaussian:bool=True, 
    add_noise:bool=False, 
    noise_scale:float=0.1,
    
    logger=None,

    use_penalty=False,
    lambda_energy=1.0,

    reverse:bool = False,
    n_conditions:int = 0,  # TODO deprecated! now will extract from model.func.condition_dim!
    lambda_cond = 1.0,
    growth_rate_model=None,
    ot_lambda_global=None,
    ot_lambda_local=None,
    lambda_energy_local=None,
    lambda_energy_global=None,
    coef_energy_func=1.0,
    coef_energy_gunc=1.0,
):

    '''
    MIOFlow training loop
    
    Notes:
        - The argument `model` must have a method `forward` that accepts two arguments
            in its function signature:
                ```python
                model.forward(x, t)
                ```
            where, `x` is the input tensor and `t` is a `torch.Tensor` of time points (float).
        - The training loop is divided in two parts; local (predict t+1 from t), and global (predict the entire trajectory).
                        
    Arguments:
        model (nn.Module): the initialized pytorch ODE model.
        
        df (pd.DataFrame): the DataFrame from which to extract batch data.
        
        groups (list): the list of the numerical groups in the data, e.g. 
            `[1.0, 2.0, 3.0, 4.0, 5.0]`, if the data has five groups.
    
        optimizer (torch.optim): an optimizer initilized with the model's parameters.
        
        n_batches (int): Default to '20', the number of batches from which to randomly sample each consecutive pair
            of groups.
            
        criterion (Callable | nn.Loss): a loss function.
        
        use_cuda (bool): Defaults to `False`. Whether or not to send the model and data to cuda. 

        sample_size (tuple): Defaults to `(100, )`

        sample_with_replacement (bool): Defaults to `False`. Whether or not to sample data points with replacement.
        
        local_loss (bool): Defaults to `True`. Whether or not to use a local loss in the model.
            See notes for more detail.
            
        global_loss (bool): Defaults to `False`. Whether or not to use a global loss in the model.
        
        hold_one_out (bool): Defaults to `False`. Whether or not to randomly hold one time pair
            e.g. t_1 to t_2 out when computing the global loss.

        hold_out (str | int): Defaults to `"random"`. Which time point to hold out when calculating the
            global loss.
            
        apply_losses_in_time (bool): Defaults to `True`. Applies the losses and does back propegation
            as soon as a loss is calculated. See notes for more detail.

        top_k (int): Default to '5'. The k for the k-NN used in the density loss.

        hinge_value (float): Defaults to `0.01`. The hinge value for density loss.

        use_density_loss (bool): Defaults to `True`. Whether or not to add density regularization.

        lambda_density (float): Defaults to `1.0`. The weight for density loss.

        autoencoder (NoneType|nn.Module): Default to 'None'. The full geodesic Autoencoder.

        use_emb (bool): Defaults to `True`. Whether or not to use the embedding model.
        
        use_gae (bool): Defaults to `False`. Whether or not to use the full Geodesic AutoEncoder.

        use_gaussian (bool): Defaults to `True`. Whether to use random or gaussian noise.

        add_noise (bool): Defaults to `False`. Whether or not to add noise.

        noise_scale (float): Defaults to `0.30`. How much to scale the noise by.
        
        logger (NoneType|Logger): Default to 'None'. The logger to record information.

        use_penalty (bool): Defaults to `False`. Whether or not to use $L_e$ during training (norm of the derivative).
        
        lambda_energy (float): Default to '1.0'. The weight of the energy penalty.

        reverse (bool): Whether to train time backwards.
    '''
    n_conditions = model.func.condition_dim
    if ot_lambda_global is None:
        ot_lambda_global = {g:1.0 for g in groups}
    if ot_lambda_local is None:
        ot_lambda_local = {g:1.0 for g in groups}
    if lambda_energy_local is None:
        lambda_energy_local = {g:1.0 for g in groups}
    if lambda_energy_global is None:
        lambda_energy_global = {g:1.0 for g in groups}

    if autoencoder is None and (use_emb or use_gae):
        use_emb = False
        use_gae = False
        warnings.warn('\'autoencoder\' is \'None\', but \'use_emb\' or \'use_gae\' is True, both will be set to False.')

    noise_fn = torch.randn if use_gaussian else torch.rand
    def noise(data):
        return noise_fn(*data.shape).cuda() if use_cuda else noise_fn(*data.shape)
    # Create the indicies for the steps that should be used
    steps = generate_steps(groups)

    if reverse:
        groups = groups[::-1]
        steps = generate_steps(groups)

    
    # Storage variables for losses
    batch_losses = []
    globe_losses = []
    if hold_one_out and hold_out in groups:
        groups_ho = [g for g in groups if g != hold_out]
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in generate_steps(groups_ho) if hold_out not in [t0, t1]}
    else:
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in steps}
        
    density_fn = Density_loss(hinge_value) # if not use_local_density else Local_density_loss()

    # Send model to cuda and specify it as training mode
    if use_cuda:
        model = model.cuda()
    
    model.train()
    
    for batch in range(n_batches):
        
        # apply local loss
        if local_loss and not global_loss:
            # for storing the local loss with calling `.item()` so `loss.backward()` can still be used
            batch_loss = []
            if hold_one_out:
                groups = [g for g in groups if g != hold_out] # TODO: Currently does not work if hold_out='random'. Do to_ignore before. 
                steps = generate_steps(groups)
            for step_idx, (t0, t1) in enumerate(steps):  
                if hold_out in [t0, t1] and hold_one_out: # TODO: This `if` can be deleted since the groups does not include the ho timepoint anymore
                    continue                              # i.e. it is always False. 
                optimizer.zero_grad()
                
                #sampling, predicting, and evaluating the loss.
                # sample data
                data_t0 = sample(df, t0, size=sample_size, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
                data_t1 = sample(df, t1, size=sample_size, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
                data_t1_cond = data_t1[:,data_t1.shape[1]-n_conditions:]
                if len(data_t1_cond.shape) == 1:
                    data_t1_cond = data_t1_cond[:,None]
                data_t1 = data_t1[:,:data_t1.shape[1]-n_conditions] # remove the conditional features
                time = torch.Tensor([t0, t1]).cuda() if use_cuda else torch.Tensor([t0, t1])

                if add_noise:
                    data_t0 += noise(data_t0) * noise_scale
                    data_t1 += noise(data_t1) * noise_scale
                if autoencoder is not None and use_gae:
                    data_t0 = autoencoder.encoder(data_t0)
                    data_t1 = autoencoder.encoder(data_t1)
                # prediction
                data_tp = model(data_t0, time)
                
                if autoencoder is not None and use_emb:        
                    # WARNING: not tested with conditional features
                    data_tp, data_t1 = autoencoder.encoder(data_tp), autoencoder.encoder(data_t1)
                
                if growth_rate_model is not None:
                    source_mass = growth_rate_model(data_t0).flatten()
                else:
                    source_mass = None
                
                if n_conditions > 0 and lambda_cond > 0:
                    assert isinstance(criterion, OT_loss)
                    loss, plan = criterion(data_tp, data_t1, source_mass=source_mass, return_plan=True)
                    # print(data_t0[:,-n_conditions:].shape)
                    # print(data_t1_cond.shape)
                    loss = loss * ot_lambda_local[t1]
                    loss_cond_change = ot_loss_given_plan(plan, data_t0[:,-n_conditions:], data_t1_cond)
                    loss += lambda_cond * loss_cond_change
                    # print(f'loss_cond_change: {loss_cond_change.item()}')
                else:   
                    # loss between prediction and sample t1
                    loss = criterion(data_tp, data_t1, source_mass=source_mass)

                if use_density_loss:                
                    density_loss = density_fn(data_tp, data_t1, top_k=top_k)
                    density_loss = density_loss.to(loss.device)
                    loss += lambda_density * density_loss

                if use_penalty:
                    # DEPRECATED, this is just one point, so incorrect.
                    penalty = sum(model.norm) # TODO is this correct?
                    # time_points = torch.linspace(t0, t1, norm_sample, dtype=data_t0.dtype, device=data_t0.device)
                    # dxdt = model.func(time_points, data_t0)

                    # loss += lambda_energy * penalty
                    loss += lambda_energy_local[t1] * penalty

                # apply local loss as we calculate it
                if apply_losses_in_time and local_loss:
                    loss.backward()
                    optimizer.step()
                    model.norm=[]
                # save loss in storage variables 
                local_losses[f'{t0}:{t1}'].append(loss.item())
                batch_loss.append(loss)
        
        
            # convert the local losses into a tensor of len(steps)
            batch_loss = torch.Tensor(batch_loss).float()
            if use_cuda:
                batch_loss = batch_loss.cuda()
            
            if not apply_losses_in_time:
                batch_loss.backward()
                optimizer.step()

            # store average / sum of local losses for training
            ave_local_loss = torch.mean(batch_loss)
            sum_local_loss = torch.sum(batch_loss)            
            batch_losses.append(ave_local_loss.item())
        
        # apply global loss
        elif global_loss and not local_loss:
            optimizer.zero_grad()
            #sampling, predicting, and evaluating the loss.
            # sample data
            data_ti = [
                sample(
                    df, group, size=sample_size, replace=sample_with_replacement, 
                    to_torch=True, use_cuda=use_cuda
                )
                for group in groups
            ]
            time = torch.Tensor(groups).cuda() if use_cuda else torch.Tensor(groups)

            if add_noise:
                data_ti = [
                    data + noise(data) * noise_scale for data in data_ti
                ]
            if autoencoder is not None and use_gae:
                data_ti = [autoencoder.encoder(data) for data in data_ti]
            # prediction
            data_tp = model(data_ti[0], time, return_whole_sequence=True)
            if autoencoder is not None and use_emb:        
                data_tp = [autoencoder.encoder(data) for data in data_tp]
                data_ti = [autoencoder.encoder(data) for data in data_ti]

            if growth_rate_model is not None:
                source_masses = [growth_rate_model(data).flatten() for data in data_ti]
            else:
                source_masses = [None for _ in data_ti]

            #ignoring one time point
            to_ignore = None #TODO: This assignment of `to_ingnore`, could be moved at the beginning of the function. 
            if hold_one_out and hold_out == 'random':
                to_ignore = np.random.choice(groups)
            elif hold_one_out and hold_out in groups:
                to_ignore = hold_out
            elif hold_one_out:
                raise ValueError('Unknown group to hold out')
            else:
                pass

            data_ti = [data_ti[i][:,:data_ti[i].shape[1]-n_conditions] for i in range(len(data_ti))] # remove the conditional features

            loss = sum([
                ot_lambda_global[i] * criterion(data_tp[i], data_ti[i], source_mass=source_masses[i-1]) # mass predicted using previous time point
                for i in range(1, len(groups))
                if groups[i] != to_ignore
            ])

            if use_density_loss:                
                density_loss = density_fn(data_tp, data_ti, groups, to_ignore, top_k)
                density_loss = density_loss.to(loss.device)
                loss += lambda_density * density_loss

            if use_penalty:
                penalty = sum([model.norm[-(i+1)] * lambda_energy_global[groups[i]] for i in range(1, len(groups))
                    if groups[i] != to_ignore])
                # loss += lambda_energy * penalty / (len(groups) - int(to_ignore is not None))
                loss += penalty / (len(groups) - int(to_ignore is not None))
                                       
            loss.backward()
            optimizer.step()
            model.norm=[]

            globe_losses.append(loss.item())
        elif local_loss and global_loss:
            # NOTE: weighted local / global loss has been removed to improve runtime
            raise NotImplementedError()
        else:
            raise ValueError('A form of loss must be specified.')
                     
    # print_loss = globe_losses if global_loss else batch_losses 
    # if logger is not None:      
    #     logger.info(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    return local_losses, batch_losses, globe_losses

def train_ae(
    model, df, groups, optimizer,
    n_epochs=60, criterion=nn.MSELoss(), dist=None, recon = True,
    use_cuda=False, sample_size=(100, ),
    sample_with_replacement=False,
    noise_min_scale=0.09,
    noise_max_scale=0.15,
    hold_one_out:bool=False,
    hold_out='random'
    
):
    """
    Geodesic Autoencoder training loop.
    
    Notes:
        - We can train only the encoder the fit the geodesic distance (recon=False), or the full geodesic Autoencoder (recon=True),
            i.e. matching the distance and reconstruction of the inputs.
            
    Arguments:
    
        model (nn.Module): the initialized pytorch Geodesic Autoencoder model.

        df (pd.DataFrame): the DataFrame from which to extract batch data.
        
        groups (list): the list of the numerical groups in the data, e.g. 
            `[1.0, 2.0, 3.0, 4.0, 5.0]`, if the data has five groups.

        optimizer (torch.optim): an optimizer initilized with the model's parameters.

        n_epochs (int): Default to '60'. The number of training epochs.

        criterion (torch.nn). Default to 'nn.MSELoss()'. The criterion to minimize. 

        dist (NoneType|Class). Default to 'None'. The distance Class with a 'fit(X)' method for a dataset 'X'. Computes the pairwise distances in 'X'.

        recon (bool): Default to 'True'. Whether or not the apply the reconstruction loss. 
        
        use_cuda (bool): Defaults to `False`. Whether or not to send the model and data to cuda. 
        
        sample_size (tuple): Defaults to `(100, )`.
        
        sample_with_replacement (bool): Defaults to `False`. Whether or not to sample data points with replacement.
        
        noise_min_scale (float): Default to '0.0'. The minimum noise scale. 
        
        noise_max_scale (float): Default to '1.0'. The maximum noise scale. The true scale is sampled between these two bounds for each epoch. 
        
        hold_one_out (bool): Default to False, whether or not to ignore a timepoint during training.
        
        hold_out (str|int): Default to 'random', the timepoint to hold out, either a specific element of 'groups' or a random one. 
    
    """
    steps = generate_steps(groups)
    losses = []

    model.train()
    for epoch in tqdm(range(n_epochs)):
        
        # ignoring one time point
        to_ignore = None
        if hold_one_out and hold_out == 'random':
            to_ignore = np.random.choice(groups)
        elif hold_one_out and hold_out in groups:
            to_ignore = hold_out
        elif hold_one_out:
            raise ValueError('Unknown group to hold out')
        else:
            pass
        
        # Training
        optimizer.zero_grad()
        noise_scale = torch.FloatTensor(1).uniform_(noise_min_scale, noise_max_scale)
        data_ti = torch.vstack([sample(df, group, size=sample_size, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda) for group in groups if group != to_ignore])
        noise = (noise_scale*torch.randn(data_ti.size())).cuda() if use_cuda else noise_scale*torch.randn(data_ti.size())
        
        encode_dt = model.encoder(data_ti + noise)
        recon_dt = model.decoder(encode_dt) if recon else None
        
        if recon:
            loss_recon = criterion(recon_dt,data_ti)
            loss = loss_recon
            
            if epoch%50==0:
                tqdm.write(f'Train loss recon: {np.round(np.mean(loss_recon.item()), 5)}')
        
        if dist is not None:
            dist_geo = dist.fit(data_ti.cpu().numpy())
            dist_geo = torch.from_numpy(dist_geo).float().cuda() if use_cuda else torch.from_numpy(dist_geo).float()
            dist_emb = torch.cdist(encode_dt,encode_dt)**2
            loss_dist = criterion(dist_emb,dist_geo)
            loss = loss_recon + loss_dist if recon else loss_dist
            
            if epoch%50==0:
                tqdm.write(f'Train loss dist: {np.round(np.mean(loss_dist.item()), 5)}')
                
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    return losses


def training_regimen(
    n_local_epochs, n_epochs, n_post_local_epochs,
    exp_dir, 

    # BEGIN: train params
    model, df, groups, optimizer, n_batches=20, 
    criterion=MMD_loss(), use_cuda=False,


    hold_one_out=False, hold_out='random', 
    hinge_value=0.01, use_density_loss=True, 

    top_k = 5, lambda_density = 1.0, 
    autoencoder=None, use_emb=True, use_gae=False, 
    sample_size=(100, ), 
    sample_with_replacement=False, 
    logger=None, 
    add_noise=False, noise_scale=0.1, use_gaussian=True,  
    use_penalty=False, lambda_energy=1.0,
    # END: train params

    steps=None, plot_every=None,
    n_points=100, n_trajectories=100, n_bins=100, 
    local_losses=None, batch_losses=None, globe_losses=None,
    reverse_schema=True, reverse_n=4,
    n_conditions:int = 0, # TODO deprecated! now will extract from model.func.condition_dim!
    lambda_cond:float = 0.0,
    growth_rate_model=None,
    lrs=None,
    ot_lambda_global=None,
    ot_lambda_local=None,
    lambda_energy_local=None,
    lambda_energy_global=None
):
    if lrs is not None:
        assert len(lrs) == 3
    recon = use_gae and not use_emb
    if steps is None:
        steps = generate_steps(groups)
        
    if local_losses is None:
        if hold_one_out and hold_out in groups:
            groups_ho = [g for g in groups if g != hold_out]
            local_losses = {f'{t0}:{t1}':[] for (t0, t1) in generate_steps(groups_ho) if hold_out not in [t0, t1]}
            if reverse_schema:
                local_losses = {
                    **local_losses, 
                    **{f'{t0}:{t1}':[] for (t0, t1) in generate_steps(groups_ho[::-1]) if hold_out not in [t0, t1]}
                }
        else:
            local_losses = {f'{t0}:{t1}':[] for (t0, t1) in generate_steps(groups)}
            if reverse_schema:
                local_losses = {
                    **local_losses, 
                    **{f'{t0}:{t1}':[] for (t0, t1) in generate_steps(groups[::-1])}
                }
    if batch_losses is None:
        batch_losses = []
    if globe_losses is None:
        globe_losses = []
    
    reverse = False
    for epoch in tqdm(range(n_local_epochs), desc='Pretraining Epoch'):
        if lrs is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lrs[0]

        reverse = True if reverse_schema and epoch % reverse_n == 0 else False
        l_loss, b_loss, g_loss = train(
            model, df, groups, optimizer, n_batches, 
            criterion = criterion, use_cuda = use_cuda,
            local_loss=True, global_loss=False, apply_losses_in_time=True,
            hold_one_out=hold_one_out, hold_out=hold_out, 
            hinge_value=hinge_value,
            use_density_loss = use_density_loss,    
            top_k = top_k, lambda_density = lambda_density, 
            autoencoder = autoencoder, use_emb = use_emb, use_gae = use_gae, sample_size=sample_size, 
            sample_with_replacement=sample_with_replacement, logger=logger,
            add_noise=add_noise, noise_scale=noise_scale, use_gaussian=use_gaussian, 
            use_penalty=use_penalty, lambda_energy=lambda_energy, reverse=reverse,
            n_conditions=n_conditions, lambda_cond=lambda_cond, growth_rate_model=growth_rate_model,
            ot_lambda_global=ot_lambda_global, ot_lambda_local=ot_lambda_local,
            lambda_energy_local=lambda_energy_local, lambda_energy_global=lambda_energy_global
        )
        for k, v in l_loss.items():  
            local_losses[k].extend(v)
        batch_losses.extend(b_loss)
        globe_losses.extend(g_loss)
        if plot_every is not None and epoch % plot_every == 0:
            generated, trajectories = generate_plot_data(
                model, df, n_points, n_trajectories, n_bins, 
                sample_with_replacement=sample_with_replacement, use_cuda=use_cuda, 
                samples_key='samples', logger=logger,
                autoencoder=autoencoder, recon=recon
            )
            plot_comparision(
                df, generated, trajectories,
                palette = 'viridis', df_time_key='samples',
                save=True, path=exp_dir, 
                file=f'2d_comparision_local_{epoch}.png',
                x='d1', y='d2', z='d3', is_3d=False
            )

    for epoch in tqdm(range(n_epochs), desc='Epoch'):
        if lrs is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lrs[1]

        reverse = True if reverse_schema and epoch % reverse_n == 0 else False
        l_loss, b_loss, g_loss = train(
            model, df, groups, optimizer, n_batches, 
            criterion = criterion, use_cuda = use_cuda,
            local_loss=False, global_loss=True, apply_losses_in_time=True,
            hold_one_out=hold_one_out, hold_out=hold_out, 
            hinge_value=hinge_value,
            use_density_loss = use_density_loss,       
            top_k = top_k, lambda_density = lambda_density, 
            autoencoder = autoencoder, use_emb = use_emb, use_gae = use_gae, sample_size=sample_size, 
            sample_with_replacement=sample_with_replacement, logger=logger, 
            add_noise=add_noise, noise_scale=noise_scale, use_gaussian=use_gaussian,
            use_penalty=use_penalty, lambda_energy=lambda_energy, reverse=reverse,
            n_conditions=n_conditions, lambda_cond=lambda_cond, growth_rate_model=growth_rate_model,
            ot_lambda_global=ot_lambda_global, ot_lambda_local=ot_lambda_local,
            lambda_energy_local=lambda_energy_local, lambda_energy_global=lambda_energy_global
        )
        for k, v in l_loss.items():  
            local_losses[k].extend(v)
        batch_losses.extend(b_loss)
        globe_losses.extend(g_loss)
        if plot_every is not None and epoch % plot_every == 0:
            generated, trajectories = generate_plot_data(
                model, df, n_points, n_trajectories, n_bins, 
                sample_with_replacement=sample_with_replacement, use_cuda=use_cuda, 
                samples_key='samples', logger=logger,
                autoencoder=autoencoder, recon=recon
            )
            plot_comparision(
                df, generated, trajectories,
                palette = 'viridis', df_time_key='samples',
                save=True, path=exp_dir, 
                file=f'2d_comparision_local_{n_local_epochs}_global_{epoch}.png',
                x='d1', y='d2', z='d3', is_3d=False
            )
        
    for epoch in tqdm(range(n_post_local_epochs), desc='Posttraining Epoch'):
        if lrs is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lrs[2]
        reverse = True if reverse_schema and epoch % reverse_n == 0 else False

        l_loss, b_loss, g_loss = train(
            model, df, groups, optimizer, n_batches, 
            criterion = criterion, use_cuda = use_cuda,
            local_loss=True, global_loss=False, apply_losses_in_time=True,
            hold_one_out=hold_one_out, hold_out=hold_out, 
            hinge_value=hinge_value,
            use_density_loss = use_density_loss,       
            top_k = top_k, lambda_density = lambda_density,  
            autoencoder = autoencoder, use_emb = use_emb, use_gae = use_gae, sample_size=sample_size, 
            sample_with_replacement=sample_with_replacement, logger=logger, 
            add_noise=add_noise, noise_scale=noise_scale, use_gaussian=use_gaussian,
            use_penalty=use_penalty, lambda_energy=lambda_energy, reverse=reverse,
            n_conditions=n_conditions, lambda_cond=lambda_cond, growth_rate_model=growth_rate_model,
            ot_lambda_global=ot_lambda_global, ot_lambda_local=ot_lambda_local,
            lambda_energy_local=lambda_energy_local, lambda_energy_global=lambda_energy_global
        )
        for k, v in l_loss.items():  
            local_losses[k].extend(v)
        batch_losses.extend(b_loss)
        globe_losses.extend(g_loss)
        if plot_every is not None and epoch % plot_every == 0:
            generated, trajectories = generate_plot_data(
                model, df, n_points, n_trajectories, n_bins, 
                sample_with_replacement=sample_with_replacement, use_cuda=use_cuda, 
                samples_key='samples', logger=logger,
                autoencoder=autoencoder, recon=recon
            )
            plot_comparision(
                df, generated, trajectories,
                palette = 'viridis', df_time_key='samples',
                save=True, path=exp_dir, 
                file=f'2d_comparision_local_{n_local_epochs}_global_{n_epochs}_post_{epoch}.png',
                x='d1', y='d2', z='d3', is_3d=False
            )

    if reverse_schema:
        _temp = {}
        if hold_one_out:
            for (t0, t1) in generate_steps([g for g in groups if g != hold_out]):
                a = f'{t0}:{t1}'
                b = f'{t1}:{t0}'
                _temp[a] = []
                for i, value in enumerate(local_losses[a]):

                    if i % reverse_n == 0:
                        _temp[a].append(local_losses[b].pop(0))
                        _temp[a].append(value)
                    else:
                        _temp[a].append(value)
            local_losses = _temp
        else:
            for (t0, t1) in generate_steps(groups):
                a = f'{t0}:{t1}'
                b = f'{t1}:{t0}'
                _temp[a] = []
                for i, value in enumerate(local_losses[a]):

                    if i % reverse_n == 0:
                        _temp[a].append(local_losses[b].pop(0))
                        _temp[a].append(value)
                    else:
                        _temp[a].append(value)
            local_losses = _temp



    if plot_every is not None:
        plot_losses(
            local_losses, batch_losses, globe_losses, 
            save=True, path=exp_dir, 
            file=f'losses_l{n_local_epochs}_e{n_epochs}_ple{n_post_local_epochs}.png'
        )

    return local_losses, batch_losses, globe_losses
