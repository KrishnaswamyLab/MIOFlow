__all__ = ['config_exp_logger', 'exp_log_filename', 'exp_param_filename', 'list_exps', 'gen_exp_name', 'load_exp_params',
           'save_exp_params', 'setup_exp', 'is_config_subset', 'find_exps', 'test_exp']

import os
import datetime
import yaml
import logging

def config_exp_logger(path):
    '''
    Arguments:
    ----------
        path (str): full path to experiment, i.e. 
            `<path-to-experiments-dir>/<experiment-timestamp>`
    Returns:
    ----------
        logger
    '''
    basename = os.path.basename(path)
    logger = logging.getLogger(basename)

    logging.basicConfig(
        filename=exp_log_filename(path), 
        encoding='utf-8',
        level=logging.DEBUG,
        format='%(asctime)s\t%(levelname)s:%(message)s',
        datefmt='%d/%m/%Y %I:%M:%S %p',
        filemode='w'
    )
    logger.info(f'Experiment path created {path}')
    return logger

def exp_log_filename(path):
    '''
    Arguments:
    ----------
        path (str): full path to experiment, i.e. 
            `<path-to-experiments-dir>/<experiment-timestamp>`
    Returns:
    ----------
        log_file (str): full path to provided experiment's log file
    '''
    return os.path.join(path, 'log.txt')

def exp_param_filename(path):
    '''
    Arguments:
    ----------
        path (str): full path to experiment, i.e. 
            `<path-to-experiments-dir>/<experiment-timestamp>`
    Returns:
    ----------
        param_file (str): full path to provided experiment's parameter file
    '''
    return os.path.join(path, 'params.yml')

def list_exps(path):
    '''
    Notes:
    ----------
        - an experiment is defined as a directory containing a `'params.yml'` file.
    Arguments:
    ----------
        path (str): full path to experiments directory, i.e. 
            `<path-to-experiments-dir>`
    Returns:
    ----------
        experiments (str[]): experiments (subdirectories) in the specified 
            `path`.
    '''
    test_fn = lambda el: os.path.isdir(el) and os.path.isfile(exp_param_filename(el))
    return list(filter(test_fn, os.listdir(path)))

def gen_exp_name(name=None):
    '''    
    Returns:
    ----------
        exp_name (str): timestamp to serve as experiment name
    '''
    if name is None:
        now =  datetime.datetime.now()
        return now.strftime("%Y_%m_%d-%I_%M_%S_%p")
    return name
        
def load_exp_params(path):
    '''
    Arguments:
    ----------
        path (str): full path to experiment, i.e. 
            `<path-to-experiments-dir>/<experiment-timestamp>`
    Returns:
    ----------
        params (dict): the loaded parameters
    '''
    with open(exp_param_filename(path)) as f:
        return yaml.safe_load(f)
    

def save_exp_params(path, params, logger=None):
    '''
    Arguments:
    ----------
        path (str): full path to experiment, i.e. 
            `<path-to-experiments-dir>/<experiment-timestamp>`
        params (dict): dictionary of parameters to save
        logger (logging.Logger): Defaults to None.
    '''
    with open(exp_param_filename(path), 'w') as f:
        yaml.dump(params, f, default_flow_style=False)
    if logger: 
        logger.info('Experiment parameters saved.')

def setup_exp(path, params, name=None):
    '''
    Arguments:
    ----------
        path (str): full path to where to create experiments, i.e. 
            `<path-to-experiments-dir>`
        params (dict): dictionary of parameters to save
    Returns:
    ----------
        exp_dir (str): full path to experiment, i.e. 
            `<path-to-experiments-dir>/<experiment-timestamp>`
        logger (logging.Logger)
    '''
    exp_name = gen_exp_name(name)
    exp_dir = os.path.join(path, exp_name)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)

    logger = config_exp_logger(exp_dir)    
    save_exp_params(exp_dir, params, logger)
    return exp_dir, logger

    
def is_config_subset(truth, params):
    '''
    Arguments:
    ----------
        truth (dict): dictionary of parameters to compare to
        params (dict): dictionary of parameters to test
    Returns:
    ----------
        result (bool) whether or not `params` is a subset of `truth`
    '''
    if not type(truth) == type(params): return False
    for key, val in params.items():
        if key not in truth: return False
        if type(val) is dict:
            if not is_config_subset(truth[key], val): 
                return False
        else:            
            if not truth[key] == val: return False
    return True


def find_exps(path, params):
    '''
    Arguments:
    ----------
        path (str): full path to where to create experiments, i.e. 
            `<path-to-experiments-dir>`
        params (dict): dictionary of parameters to test
    Returns:
    ----------
        results (str[]) list of experiment names where their parameters are 
            supersets of the provided `params`
    '''
    exps = list_exps(path)
    results = []
    for exp in exps:
        exp_name = os.path.join(path, exp)
        exp_params = load_exp_params(exp_param_filename(exp_name))
        if is_config_subset(exp_params, params):
            results.append(exp)
    return results


def test_exp():
    exp_dir, logger = setup_exp('.', {'this':'is','a':'test'})
