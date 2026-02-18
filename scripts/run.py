import os, random, argparse, pandas as pd, numpy as np, seaborn as sns
from tqdm import tqdm
import torch, torch.nn as nn


# set random seed
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
SEED = 10
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)


# load package requirments
from MIOFlow.losses import MMD_loss, OT_loss, Density_loss, Local_density_loss
from MIOFlow.utils import group_extract, sample, to_np, generate_steps
from MIOFlow.core.models import ToyModel, make_model
from MIOFlow.plots import plot_comparision, plot_losses
from MIOFlow.train import train, train_emd
from MIOFlow.constants import ROOT_DIR, DATA_DIR, NTBK_DIR, IMGS_DIR, RES_DIR
from MIOFlow.datasets import (
    make_diamonds, make_dyngen_data, make_swiss_roll, make_tree, 
    make_worm_data, make_eb_data
)
from MIOFlow.ode import NeuralODE, ODEF
from MIOFlow.exp import setup_exp
from MIOFlow.geo import GeoEmbedding, DiffusionDistance
from MIOFlow.eval import generate_plot_data


# for geodesic learning
from scipy.spatial import distance_matrix
from sklearn.gaussian_process.kernels import RBF
from sklearn.manifold import MDS

# define lookup variables
_valid_datasets = {
    'petals': make_diamonds,
    'swiss': make_swiss_roll,
    'tree': make_tree,
    'worm': make_worm_data,
    'eb_bodies': make_eb_data,
    'dyngen': make_dyngen_data,
    'file': lambda file: np.load(file),
    'file': lambda file: pd.read_csv(file),
}

_valid_criterions = {
    'mmd': MMD_loss,
    'ot': OT_loss
}

'''
NOTE: at the moment only simple models are supported via the package submodule
MIOFlow.models. However, this script is easily adapted to handle 
one's own model specification for example, to define a simple model one could do the 
following:

    class ToyODE(ODEF):
        def __init__(self, feature_dims=5):
            super(ToyODE, self).__init__()
            self.seq = nn.Sequential(
                nn.Linear(feature_dims, 64),
                nn.ReLU(),
                nn.Linear(64, feature_dims)            
            )

        def forward(self, x, t):
            x = self.seq(x)
            dxdt = x
            return dxdt

Then below, where we call `make_model`, replace it with the following:

    ode = NeuralODE(ToyODE(len(df.columns)-1))
    model = ToyModel(ode)

and your custom model is ready to go :)
'''

# define parser
parser = argparse.ArgumentParser(prog='Trajectory Net Training', description='''
Train Trajectory Net
''')

# NOTE: Dataset specification
parser.add_argument(
    '--dataset', '-d', type=str, choices=list(_valid_datasets.keys()), required=True,
    help=(
        'Dataset of the experiment to use. '
        'If value is fullpath to a file then tries to load file. '
        'Note, if using your own file we assume it is a pandas '
        'dataframe which has a column named `samples` that correspond to '
        'the timepoints.'
    )
)

parser.add_argument(
    '--time-col', '-tc', type=str, choices='simulation_i step_ix sim_time'.split(), required=False,
    help='Time column of the dataset to use.'
)

# NOTE: Experiment specification
parser.add_argument(
    '--name', '-n', type=str, required=True, default=None,
    help='Name of the experiment. If none is provided timestamp is used.'
)

parser.add_argument(
    '--output-dir', '-od', type=str, default=RES_DIR,
    help='Where experiments should be saved. The results directory will automatically be generated here.'
)

# NOTE: Runtime arguments
parser.add_argument(
    '--local-epochs', '-le', type=int, default=5,
    help='Number of epochs to use `local_loss` while training. These epochs occur first. Defaulst to `5`.'
)
parser.add_argument(
    '--epochs', '-e', type=int, default=15,
    help='Number of epochs to use `global_loss` while training. Defaults to `15`.' 
)
parser.add_argument(
    '--local-post-epochs', '-lpe', type=int, default=5,
    help='Number of epochs to use `local_loss` after training. These epochs occur last. Defaulst to `5`.'
)


# NOTE: Train arguments
parser.add_argument(
    '--criterion', '-c', type=str, choices=list(_valid_criterions.keys()), 
    default='mmd', required=True,
    help='a loss function, either `"mmd"` or `"emd"`. Defaults to `"mmd"`.'
)

parser.add_argument(
    '--batches', '-b', type=int, default=100,
    help='the number of batches from which to randomly sample each consecutive pair of groups.'
)

parser.add_argument(
    '--cuda', '--use-gpu', '-g', type=bool, 
    action=argparse.BooleanOptionalAction, default=True,     
    help='Whether or not to use CUDA. Defaults to `True`.'
)


parser.add_argument(
    '--sample-size', '-ss', type=int, default=100,     
    help='Number of points to sample during each batch. Defaults to `100`.'
)

parser.add_argument(
    '--sample-with-replacement', '-swr', type=bool, 
    action=argparse.BooleanOptionalAction, default=False,     
    help='Whether or not to sample with replacement. Defaults to `True`.'
)



parser.add_argument(
    '--hold-one-out', '-hoo', type=bool, 
    action=argparse.BooleanOptionalAction, default=True,
    help=' Defaults to `True`. Whether or not to randomly hold one time pair e.g. t_1 to t_2 out when computing the global loss.'
)

parser.add_argument(
    '--hold-out', '-ho', type=str, default='random',
    help='Defaults to `"random"`. Which time point to hold out when calculating the global loss.'
)

parser.add_argument(
    '--apply-losses-in-time', '-it',type=bool, 
    action=argparse.BooleanOptionalAction, default=True,
    help='Defaults to `True`. Applies the losses and does back propegation as soon as a loss is calculated. See notes for more detail.'
)

parser.add_argument(
    '--top-k', '-k', type=int, default=5,
    help='the k for the k-NN used in the density loss'
)

parser.add_argument(
    '--hinge-value', '-hv', type=float, default=0.01,
    help='hinge value for density loss function. Defaults to `0.01`.'
)

parser.add_argument(
    '--use-density-loss', '-udl', type=bool, 
    action=argparse.BooleanOptionalAction, default=True,
    help='Defaults to `True`. Whether or not to add density regularization.'
)

parser.add_argument(
    '--use-local-density', '-uld', type=bool, 
    action=argparse.BooleanOptionalAction, default=False,
    help='Defaults to `False`. Whether or not to use local density.'
)

parser.add_argument(
    '--lambda-density', '-ld', type=float, default=1.0,
    help='The weight for density loss. Defaults to `1.0`.'
)

parser.add_argument(
    '--lambda-density-local', '-ldl', type=float, default=1.0,
    help='The weight for local density loss. Defaults to `1.0`.'
)

parser.add_argument(
    '--lambda-local', '-ll', type=float, default=0.2,
    help='the weight for average local loss.  Note `lambda_local + lambda_global = 1.0`. Defaults to `0.2`.'
)

parser.add_argument(
    '--lambda-global', '-lg', type=float, default=0.8,
    help='the weight for global loss. Note `lambda_local + lambda_global = 1.0`. Defaults to `0.8`.'
)

parser.add_argument(
    '--model-layers', '-ml', type=int, nargs='+', default=[64],
    help='Layer sizes for ode model'
)

# NOTE: Geo training args
parser.add_argument(
    '--use-geo', '-ug', type=bool, default=False,
    action=argparse.BooleanOptionalAction,
    help='Whether or not to use a geodesic embedding'
)
# TODO: add Geo training stuff
parser.add_argument(
    '--geo-layers', '-gl', type=int, nargs='+', default=[32],
    help='Layer sizes for geodesic embedding model'
)
parser.add_argument(
    '--geo-features', '-gf', type=int, default=5,
    help='Number of features for geodesic model.'
)



# NOTE: eval stuff
parser.add_argument(
    '--n-points', '-np', type=int, default=100,
    help='number of trajectories to generator for plot. Defaults to `100`.'
)

parser.add_argument(
    '--n-trajectories', '-nt', type=int, default=30,
    help='number of trajectories to generator for plot. Defaults to `30`.'
)

parser.add_argument(
    '--n-bins', '-nb', type=int, default=100,
    help='number of bins to use for generating trajectories. Higher make smoother trajectories. Defaulst to `100`.'
)


if __name__ == '__main__':
    args = parser.parse_args()
    opts = vars(args)


    # make output dir
    if not os.path.isdir(opts['output_dir']):
        os.makedirs(opts['output_dir'])
    exp_dir, logger = setup_exp(opts['output_dir'], opts, opts['name'])
    
    # load dataset
    logger.info(f'Loading dataset')
    if opts['dataset'] in _valid_datasets:
        fn = _valid_datasets[opts['dataset']]
        if opts['dataset'] == 'dyngen':
            df = fn(opts['time-col'])
        else:
            df = fn()
    else:
        df = _valid_datasets['file'](opts['dataset'])

    # setup groups
    groups = sorted(df.samples.unique())
    steps = generate_steps(groups)

    # define models
    # TODO: update to match Dyngen notebook
    logger.info(f'Defining model')
    use_geo = opts['use_geo']

    model_layers = opts['model_layers']
    model_features = len(df.columns) - 1

    geo_layers = opts['geo_layers']
    geo_features = opts['geo_features']

    geoemb = make_model(model_features, geo_layers, geo_features, 'ReLU', which='geo')
    model = make_model(model_features, model_layers, 'ReLU')
    
    

    
    logger.info(f'Defining optimizer and criterion')
    optimizer = torch.optim.Adam(model.parameters())
    geo_optimizer = torch.optim.Adam(geoemb.parameters())
    criterion =  _valid_criterions[opts['criterion']]()

    logger.info(f'Extracting parameters')
    use_cuda = torch.cuda.is_available() and opts['cuda']

    sample_size = (opts['sample_size'], )
    sample_with_replacement = opts['sample_with_replacement' ]

    apply_losses_in_time = opts['apply_losses_in_time']

    n_local_epochs = opts['local_epochs']
    n_epochs = opts['epochs']
    n_post_local_epochs = opts['local-post-epochs']
    n_batches = opts['batches']

    hold_one_out = opts['hold_one_out']
    hold_out = opts['hold_out']
    
    hinge_value = opts['hinge_value']
    top_k = opts['top_k']
    lambda_density = opts['lambda_density']
    lambda_density_local = opts['lambda_density_local']
    use_density_loss = opts['use_density_loss']
    use_local_density = opts['use_local_density']
    
    lambda_local = opts['lambda_local']
    lambda_global = opts['lambda_global']

    n_points=opts['n_points']
    n_trajectories=opts['n_trajectories'] 
    n_bins=opts['n_bins']
    
    
    local_losses = {f'{t0}:{t1}':[] for (t0, t1) in steps}
    batch_losses = []
    globe_losses = []

    # TODO: update argparse to include diffusion distance options
    if use_geo:
        logger.info(f'Training geodesic model')
        train_emd(
            geoemb, df, groups, geo_optimizer,
            n_epochs=60, criterion=nn.MSELoss(),             
            geo_dist=DiffusionDistance(),
            use_cuda=True, sample_size=sample_size, sample_with_replacement=sample_with_replacement 
        )
    
    if n_local_epochs > 0:
        logger.info(f'Beginning pretraining')
        for epoch in tqdm(range(n_local_epochs), desc='Pretraining Epoch'):
            l_loss, b_loss, g_loss = train(
                model, df, groups, optimizer, n_batches, 
                criterion = criterion, use_cuda = use_cuda,
                local_loss=True, global_loss=False, apply_losses_in_time=apply_losses_in_time,
                hold_one_out=hold_one_out, hold_out=hold_out, 
                lambda_local = lambda_local, lambda_global = lambda_global, hinge_value=hinge_value,
                use_density_loss = use_density_loss, use_local_density = use_local_density,       
                top_k = top_k, lambda_density = lambda_density, lambda_density_local = lambda_density_local, 
                geo_emb = geoemb, use_geo = use_geo, sample_size = sample_size,
                sample_with_replacement = sample_with_replacement, logger=logger  
            )
            for k, v in l_loss.items():  
                local_losses[k].extend(v)
            batch_losses.extend(b_loss)
            globe_losses.extend(g_loss)

    logger.info(f'Beginning training')    
    for epoch in tqdm(range(n_epochs), desc='Epoch'):
        l_loss, b_loss, g_loss = train(
            model, df, groups, optimizer, n_batches, 
            criterion = criterion, use_cuda = use_cuda,
            local_loss=False, global_loss=True, apply_losses_in_time=apply_losses_in_time,
            hold_one_out=hold_one_out, hold_out=hold_out, 
            lambda_local = lambda_local, lambda_global = lambda_global, hinge_value=hinge_value,
            use_density_loss = use_density_loss, use_local_density = use_local_density,       
            top_k = top_k, lambda_density = lambda_density, lambda_density_local = lambda_density_local, 
            geo_emb = geoemb, use_geo = use_geo, sample_size = sample_size,
            sample_with_replacement = sample_with_replacement, logger=logger  
        )

        for k, v in l_loss.items():  
            local_losses[k].extend(v)
        batch_losses.extend(b_loss)
        globe_losses.extend(g_loss)

    if n_post_local_epochs > 0:
        logger.info(f'Beginning posttraining')
        for epoch in tqdm(range(n_post_local_epochs), desc='Posttraining Epoch'):
            l_loss, b_loss, g_loss = train(
                model, df, groups, optimizer, n_batches, 
                criterion = criterion, use_cuda = use_cuda,
                local_loss=True, global_loss=False, apply_losses_in_time=apply_losses_in_time,
                hold_one_out=hold_one_out, hold_out=hold_out, 
                lambda_local = lambda_local, lambda_global = lambda_global, hinge_value=hinge_value,
                use_density_loss = use_density_loss, use_local_density = use_local_density,       
                top_k = top_k, lambda_density = lambda_density, lambda_density_local = lambda_density_local, 
                geo_emb = geoemb, use_geo = use_geo, sample_size = sample_size,
                sample_with_replacement = sample_with_replacement, logger=logger  
            )
            for k, v in l_loss.items():  
                local_losses[k].extend(v)
            batch_losses.extend(b_loss)
            globe_losses.extend(g_loss)


    # plotting
    plot_losses(
        local_losses, batch_losses, globe_losses, 
        save=True, path=exp_dir, file='losses.png'
    )

    # generate plot data
    generated, trajectories = generate_plot_data(
        model, df, n_points=n_points, n_trajectories=n_trajectories, n_bins=n_bins, 
        sample_with_replacement=sample_with_replacement, use_cuda=use_cuda, samples_key='samples',
        logger=logger
    )

    plot_comparision(
        df, generated, trajectories,
        palette = 'viridis', df_time_key='samples',
        save=True, path=exp_dir, file='comparision.png',
        x='d1', y='d2', z='d3', is_3d=False
    )

    # save stuff
    if use_geo:
        torch.save(geoemb, os.path.join(exp_dir, 'geoemb'))
    torch.save(model, os.path.join(exp_dir, 'model'))
    with open('points.npy', 'wb') as f:
        np.save(f, generated)
    with open('trajectories.npy', 'wb') as f:
        np.save(f, trajectories)