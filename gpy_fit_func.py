from GPy_ABCD.Models.model import GPModel
from GPy.models import GPRegression
from multiprocessing import Pool, cpu_count
from warnings import warn
import warnings
import numpy as np
from utils.main_utils import find_se_lengthscale_set_constraint
import GPy
import copy

def initialize_mods(X, Y, start_kexs, models, restarts, optimiser, max_retries):
    mods = []
    for kex, model in zip(start_kexs, models):
        curmod = GPModel(X, Y, kex) ### kex without params
        gp_kernel = model ##### INIT by CFG parser
        curmod.model = GPRegression(X, Y, gp_kernel) #### Now only USE THIS.
        curmod.restarts = restarts
        mods.append(curmod)
    return mods

def find_and_set_param(kernel, ret, param_dict):
    with warnings.catch_warnings(): 
        warnings.simplefilter('ignore')
        if isinstance(kernel, GPy.kern.src.add.Add) or isinstance(kernel, GPy.kern.src.prod.Prod):
            for part in kernel.parts:
                ret = find_and_set_param(part, ret, param_dict)
                
        else:
            if kernel.name=='rbf' and hasattr(kernel, 'lengthscale'):
                if 'SE' in param_dict:
                    if 'lengthscale' in param_dict['SE'] and hasattr(kernel, 'lengthscale'):
                        kernel.lengthscale = param_dict['SE']['lengthscale']
                        ret = True
                
            if kernel.name=='pure_std_periodic':
                if 'PER' in param_dict:
                    if 'lengthscale' in param_dict['PER'] and hasattr(kernel, 'lengthscale'):
                        kernel.lengthscale = param_dict['PER']['lengthscale']
                        ret = True
                    if 'period' in param_dict['PER'] and hasattr(kernel, 'period'):
                        kernel.period = param_dict['PER']['period']
                        ret = True
                    
            if kernel.name=='linear_with_offset' or kernel.name=='linear':
                if 'LIN' in param_dict:
                    if 'offset' in param_dict['LIN'] and hasattr(kernel, 'offset'):
                        kernel.offset = param_dict['LIN']['offset']
                        ret = True
        return ret
    
def initialize_mods_already_tuned(X, Y, kexp, ref_mods, param_dicts, restarts, optimiser, max_retries):
    mods = []
    for kex, ref_mod, param_dict in zip(kexp, ref_mods, param_dicts):
        curmod = GPModel(X, Y, kex) 
        gp_kernel = copy.deepcopy(ref_mod.model) 
        _ = find_and_set_param(gp_kernel.kern, False, param_dict)
        curmod.model = GPRegression(X, Y, gp_kernel.kern)
        curmod.restarts = restarts
        mods.append(curmod)
    return mods

def fit_one_model_with_init(X, Y, gpmodel, restarts, optimiser, max_retries, randomize=False, **kwargs):
    retry = 0
    while True:
        try: 
            # gpmodel.model.optimize_restarts(num_restarts = restarts, verbose = False, robust =False, optimizer = optimiser, **kwargs)
            with warnings.catch_warnings(): # Ignore known numerical warnings
                warnings.simplefilter('ignore')
                if randomize: gpmodel.model.randomize()
                # if retry == max_retries: gpmodel.model.randomize()
                gpmodel.model.optimize()
            return gpmodel
        except Exception as e:
            if retry < max_retries:
                retry += 1
                warn(f'Retry #{retry} (max is {max_retries}) for kernel {gpmodel.model} due to error:\n\t{e}')
            else: raise e

##### FROM PARAMZ
def opt_wrapper(args):
    m = args[0]
    kwargs = args[1]
    return m.optimize(**kwargs)

def my_fit_one_model(X, Y, kex, restarts, optimiser, max_retries, randomize=True, **kwargs):
    retry=10
    for _ in range(retry):
        try: 
            gpmodel = GPModel(X, Y, kex)
            gpmodel.model = GPRegression(X, Y, kex.to_kernel())
            
            # if 'se_kernel_const' in kwargs:
            #     find_se_lengthscale_set_constraint(gpmodel.model.kern, ret=False, constraint=kwargs['se_kernel_const'])
            
            if 'se_kernel_const' in kwargs: find_se_lengthscale_set_constraint(gpmodel.model.kern, ret=False, constraint=kwargs['se_kernel_const'])
                
            with warnings.catch_warnings(): # Ignore known numerical warnings
                warnings.simplefilter('ignore')
                
                # initial_length = len(gpmodel.model.optimization_runs)
                # initial_parameters = gpmodel.model.optimizer_array.copy()
                
                # for i in range(restarts):
                #     if randomize: gpmodel.model.randomize() 
                #     gpmodel.model.optimize()
                # print(f'tuning done for {kex}')
                
                # i = np.argmin([o.f_opt for o in gpmodel.model.optimization_runs[initial_length:]])
                # gpmodel.model.optimizer_array = gpmodel.model.optimization_runs[initial_length + i].x_opt  
                
                gpmodel.model.optimize_restarts(num_restarts = restarts, verbose = False, robust = True, optimizer = optimiser, **kwargs)
                
            return gpmodel
        except Exception as e:
            if retry < max_retries:
                retry += 1
                warn(f'Retry #{retry} (max is {max_retries}) for kernel {kex} due to error:\n\t{e}')
            else: raise e

    
def fit_mods_parallel_processes_with_initialize(X, Y, models, restarts, optimiser, max_retries, randomize=False):
    with Pool() as pool: return pool.starmap_async(fit_one_model_with_init, [(X, Y, model, restarts, optimiser, max_retries, randomize) for model in models], int(len(models) / cpu_count()) + 1).get()
    
def my_fit_mods_parallel_processes(X, Y, models, restarts, optimiser, max_retries, **kwargs):
    with Pool() as pool: return pool.starmap_async(my_fit_one_model, [(X, Y, model, restarts, optimiser, max_retries, kwargs) for model in models], int(len(models) / cpu_count()) + 1).get()
    
