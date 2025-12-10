# BASED ON /home/mok/module/icml25/ASMD/ablation/icmlrebuttal/asmd_ablation_final_c.py

import numpy as np


import os
import sys

import pickle
from GPy_ABCD.KernelExpansion.grammar import start_kernels, expand, production_rules, make_simple_kexs
from GPy_ABCD.Models.modelSearch import fit_mods_parallel_processes
from GPy_ABCD.Util.modelUtil import GPy_optimisers

from GPy_ABCD.Kernels.baseKernels import *
from GPy_ABCD.KernelExpansion.grammar import * # make_simple_kexs
from GPy_ABCD.Util.dataAndPlottingUtil import *
from GPy_ABCD.Util.modelUtil import *

from GPy_ABCD.Kernels.baseKernels import base_kerns

from GPy.plotting.gpy_plot.plot_util import x_frame1D

from GPy_ABCD.Models.model import GPModel
from GPy.models import GPRegression

from multiprocessing import Pool, cpu_count
from warnings import warn

import matplotlib.pyplot as plt
import json
import pandas as pd
import logging
import argparse
from glob import glob
import random
import torch
import logging
from colorlog import ColoredFormatter
from datetime import datetime

from plot_func import plot as plotgp
from utils.dataload_utils import load_sr_data
from utils.main_utils import BIC, mse, weighted_sample_without_replacement, encode_image, find_se_lengthscale_set_constraint, save_genetic_pool, make_dirpath, sort_genetic_pools, save_initial_plot, make_logger
from utils.cfg_utils import parse_kernel_name, parse_kernel_model, parse_kernel_params
from utils.analyzer_utils import analyze_then_generate_kernels_first, analyze_then_generate_kernels
from utils.vision_score_utils import evaluate_mean, evaluate_confidence, evaluate_ss, evaluate_vision_overall

from gpy_fit_func import initialize_mods, initialize_mods_already_tuned, fit_one_model_with_init, my_fit_one_model, fit_mods_parallel_processes_with_initialize, my_fit_mods_parallel_processes
from parse_functools import param_init, build_hierarchical_dict, get_param_dict, random_update_hierarchical_dict

############### SEED FIX ##############
seed = 2021
deterministic = True

random.seed(seed)
np.random.seed(seed) ##### BUT SCIPY.OPTIMIZE WONT BE FIXED.
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
 ############### SEED FIX ##############

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="gpr kernel selection")
    parser.add_argument('-d','--data', nargs='+', type=int, help='dataset', default=[1,2,3,4,5,6,7,8,9,10])
    parser.add_argument('-c','--continue_searching', type=bool, default=False)
    # parser.add_argument('-m','--model', type=str, default='gpt-4.1-mini')
    # parser.add_argument('-m','--model', type=str, default='gpt-4.1-nano')
    parser.add_argument('-m','--model', type=str, default='gpt-4o-mini')
    parser.add_argument('-r', '--round', type=int, default=5, help='round')
    parser.add_argument('-re', '--restart', type=int, default=10, help='restarts')
    parser.add_argument('-t', '--top', type=int, default=3, help='restarts')
    parser.add_argument('-n', '--noise', type=float, default=0, help='restarts')
    parser.add_argument('-s', '--set_se_const', type=bool, default=True, help='restarts')
    
    args = parser.parse_args()
    top_sample = args.top
    noise = args.noise
    
    ##### FITTING OPTIONS
    R=args.round ##### self.iteration
    rng = random.Random(R)
    restarts = args.restart
    optimiser = GPy_optimisers[0]
    max_retries = 5
    mymet_hyp = 50
    vlm_model_name = args.model
    datasets_to_test = args.data
    continue_searching = args.continue_searching
    set_se_constraint = args.set_se_const
    
    USE_ANALYZER = True
    USE_INHERIT_AT_NO_ISLAND = True
    USE_VISUAL_SCORE = True

    directory_path = f'./output/sr/fig-R{R}-restarts{restarts}'
    directory_path += '-analyzer' if USE_ANALYZER else '-noanalyzer'
    directory_path += '-visual' if USE_VISUAL_SCORE else '-novisual'
    directory_path += '-inherit' if USE_INHERIT_AT_NO_ISLAND else '-noinherit'
    directory_path += f'-noise{noise}' if noise!=0 else ''
    directory_path += '-se-constraint' if set_se_constraint else ''
    directory_path += f'-{vlm_model_name}'
    
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    
    datasets = [
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/keijzer/keijzer3',
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/keijzer/keijzer4',
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/keijzer/keijzer6',
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/keijzer/keijzer7',
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/keijzer/keijzer8',
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/keijzer/keijzer9',
        
        # '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/constant/constant1',
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/constant/constant2',
        # '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/constant/constant3',
        # '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/constant/constant4',
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/constant/constant5',
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/constant/constant6',
        # '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/constant/constant7',
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/constant/constant8',
        
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/nguyen/nguyen1',
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/nguyen/nguyen2',
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/nguyen/nguyen3',
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/nguyen/nguyen4',
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/nguyen/nguyen5',
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/nguyen/nguyen6',
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/nguyen/nguyen7',
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/nguyen/nguyen8',
        
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/R/R1',
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/R/R2',
        '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data/R/R3'
    ]
    
    datapath = '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data'
    
    for data_idx in datasets_to_test:
        data_name = datasets[data_idx-1].split('/')[-1]
        data_name_no_num = datasets[data_idx-1].split('/')[-2]
        dirpath = os.path.join(directory_path, data_name)
        if not os.path.exists(directory_path): os.makedirs(dirpath)
        
        logger = make_logger(dirpath)
        logger.info(f'DIRECTORY PATH: {directory_path}/{data_name}')
        logger.info(f'STARTING DATA: {data_name}')

        
        if torch.cuda.is_available():
            torch.cuda.init()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        logger.info(f"Using device: {device} with dtype: {dtype}.")
        
        X, Y, valX, valY, testX, testY, data, range_, origX, origY = load_sr_data(f'{datapath}/{data_name_no_num}/{data_name}')
        
        genetic_pool_dict = {}
        tree_dict = {}
        round_selected_model = {}
        
        if not continue_searching:
            ############################ PARAMETER PREPARATION ##############################
            basis_kernel_str = ['WN', 'C', 'LIN', 'SE', 'PER']
            start_kexs = make_simple_kexs(basis_kernel_str)
            basis_exemplars = fit_mods_parallel_processes(X, Y, start_kexs, restarts, optimiser, max_retries)
            basis_param_dict = {}
            not_used = ['input_dim', 'active_dims', 'name', 'useGPU', 'class']
            
            dirpath_i = make_dirpath(directory_path, data_name, -1)
            for i, basis_exemplar in enumerate(basis_exemplars):
                basis_param_dict[basis_kernel_str[i]] = {}
                for k in basis_exemplar.model.kern.to_dict():
                    if k in not_used: continue
                    basis_param_dict[basis_kernel_str[i]][k] = basis_exemplar.model.kern.to_dict()[k]

                conf_vis_path = f'{dirpath_i}/{i} - {basis_kernel_str[i]}.png'
                plotgp(basis_exemplar.model, ymin=np.min(Y)-range_*0.3, ymax=np.max(Y)+range_*0.3)
                plt.savefig(conf_vis_path)
                plt.cla(); plt.clf()
            ##################################################################################
            
            ######################### MODEL GENERATION FOR ROUND 0 ###########################
            ######################### MODEL PARSING AND FITTING FOR ROUND 0 ###########################
            if USE_ANALYZER:
                k_exemplars, params, logs = analyze_then_generate_kernels_first(X, Y, data_name, round=0, model=vlm_model_name, logger=logger) # TODO: 프롬프트 수정
                start_kexs = [parse_kernel_name(each) for each in k_exemplars] #### 랜덤 initialize용 커널 조합
                # start_kex_mods = [parse_kernel_model(each) for each in k_exemplars]
                start_kex_params = [parse_kernel_params(each) for each in k_exemplars]
            else: raise NotImplementedError
            
            ##### FIT THE MODEL WITH THE BASIS KERNEL PARAMETERS
            k_exemplars = [GPModel(X, Y, start_kex) for start_kex in start_kexs]
            for ti, k_exemplar in enumerate(k_exemplars): 
                k_exemplar.model = GPRegression(X, Y, start_kexs[ti].to_kernel()) ##### RANDOM INIT
            k_exemplars = initialize_mods_already_tuned(X=X, Y=Y, kexp=start_kexs, ref_mods=k_exemplars, param_dicts=[basis_param_dict for _ in range(len(start_kexs))], restarts=restarts, optimiser=optimiser, max_retries=max_retries)
            k_exemplars = fit_mods_parallel_processes_with_initialize(X, Y, k_exemplars, restarts, optimiser, max_retries)
            
            # ##### FIT THE MODEL WITH VLM INITIALIZATION - BASED ON PREVIOUS TUNING
            k_exemplars_with_init = initialize_mods_already_tuned(X=X, Y=Y, kexp=start_kexs, ref_mods=k_exemplars, param_dicts=start_kex_params, restarts=restarts, optimiser=optimiser, max_retries=max_retries)
            k_exemplars_with_init = fit_mods_parallel_processes_with_initialize(X, Y, k_exemplars_with_init, restarts, optimiser, max_retries)
            
            k_exemplars_string = ["%s"%(each.kernel_expression) for each in k_exemplars] 
            logger.info('ROUND 0 SUGGESTED KERNELS: %s', k_exemplars_string)
            round_selected_model[0] = k_exemplars_string
            k_exemplars = [[a,b] for a,b in zip(k_exemplars_with_init, k_exemplars)]
            
            ################# SELECT BEST PARAMETER IN EACH KERNELS #######################
            fgk_dict = {}
            for model in (sum(k_exemplars, [])):
                name = "%s"%model.kernel_expression
                bic = BIC(model.model, model.model.log_likelihood(), len(X), model.model._size_transformed())
                val_mse = mse(model.predict(valX)['mean'], valY)[0]
                param_score = bic+100*val_mse
                    
                if name in fgk_dict:
                    # continue ##### ROUND 0: PLEASE ONLY USE VLM
                    prev_param_score = fgk_dict[name]['param_score']
                    if param_score<prev_param_score:
                        fgk_dict[name] = {'model': model, 'param_score': param_score}
                    
                else: fgk_dict[name] = {'model': model, 'param_score': param_score}
            ###############################################################################
                    
            dirpath_i = make_dirpath(directory_path, data_name, 0)        
            _ = save_initial_plot(k_exemplars, genetic_pool_dict, dirpath_i, X, Y, valX, valY, range_) ##### LATER CHANGE THIS
            
            ####################### MODEL VISUAL SCORING #######################
            # plot the data function for one time!!!!
            # model = fgk_dict[name] #### just for random model with all same dataset
            if USE_VISUAL_SCORE:
                for ii, name in enumerate(fgk_dict.keys()):
                    model = fgk_dict[name]['model']
                    
                    likelihood = model.model.log_likelihood()
                    bic = BIC(model.model, likelihood, len(X), model.model._size_transformed())
                    val_mse = mse(model.predict(valX)['mean'], valY)[0]
                    loo = np.sum(model.model.inference_method.LOO(model.model.kern, valX, valY, model.model.likelihood, model.model.posterior))
                    
                    data_vis_path = f'{dirpath_i}/{ii} - {name}-data.png'
                    plotgp(model.model, ymin=np.min(Y)-range_*0.3, ymax=np.max(Y)+range_*0.3, xlim=[np.min(model.X), np.max(model.X)], plot_data=True, plot_samples=False)
                    plt.savefig(data_vis_path)
                    plt.cla(); plt.clf()
                    
                    conf_vis_path = f'{dirpath_i}/{ii} - {name}.png'
                    plotgp(model.model, ymin=np.min(Y)-range_*0.3, ymax=np.max(Y)+range_*0.3)
                    plt.savefig(conf_vis_path)
                    plt.cla(); plt.clf()

                    mean_vis_path = f'{dirpath_i}/{ii} - {name}-mean.png'
                    plotgp(model.model, ymin=np.min(Y)-range_*0.3, ymax=np.max(Y)+range_*0.3, plot_data=False, plot_samples=True, plot_confidence=False)
                    plt.savefig(mean_vis_path)
                    plt.cla(); plt.clf()
                    
                    score_for_criteria = evaluate_vision_overall(name, data_vis_path, mean_vis_path, conf_vis_path, data_name, vlm_model_name, logger=logger)
                    
                    if name in genetic_pool_dict:
                        param_score = genetic_pool_dict[name]['BIC'] + 100 * genetic_pool_dict[name]['val mse']
                        if bic+100*val_mse < param_score:
                            genetic_pool_dict.pop(name)
                    
                    if not name in genetic_pool_dict:
                        genetic_pool_dict[name] = {
                            'score for match': int(score_for_criteria.get('score for match', 0)),
                            'score for confidence': int(score_for_criteria.get('score for confidence', 0)),
                            'score for consistency': int(score_for_criteria.get('score for consistency', 0)),
                            'score': int(score_for_criteria.get('score for match', 0) + score_for_criteria.get('score for confidence', 0) + score_for_criteria.get('score for consistency', 0)),
                            'model': model,
                            'likelihood': likelihood,
                            'BIC': bic,
                            'val mse': val_mse,
                            'loo': loo,
                            'round': 0
                        }
                    
                    logger.info('TOTAL SCORE: %s', genetic_pool_dict[name]['score'])
                    logger.info('BIC: %s', genetic_pool_dict[name]['BIC'])
                    logger.info('val mse: %s', genetic_pool_dict[name]['val mse'])
                    logger.info('LOO: %s', genetic_pool_dict[name]['loo'])
                    logger.info(f'================================================')
                    
            else:
                for ii, name in enumerate(fgk_dict.keys()):
                    if name in genetic_pool_dict:
                        param_score = genetic_pool_dict[name]['BIC'] + 100 * genetic_pool_dict[name]['val mse']
                        if bic+100*val_mse < param_score:
                            genetic_pool_dict.pop(name)
                    
                    if not name in genetic_pool_dict:
                        genetic_pool_dict[name] = {
                            'score for match': 0,
                            'score for confidence': 0,
                            'score for consistency': 0,
                            'score': 0,
                            'model': model,
                            'likelihood': likelihood,
                            'BIC': bic,
                            'val mse': val_mse,
                            'loo': loo,
                            'round': 0
                        }
            ##################################################################################
        else:
            logger.info('CONTINUEING SEARCHING.')
        
        for i in range(1, R):
            if continue_searching:
                ##### prepare the dict
                with open(f'./{dirpath}/genetic_pool_dict.pickle', 'rb') as f:
                    genetic_pool_dict = pickle.load(f)
                logs = []
                if i<4: continue

            ######################### TOP-K SELECTION ###########################
            tmp_sorted_models, tmp_sorted_scores = sort_genetic_pools(genetic_pool_dict, mymet_hyp)
            k_exemplars = weighted_sample_without_replacement(tmp_sorted_models, tmp_sorted_scores, top_sample, rng)
            
            k_exemplars_string = ["%s"%(each.kernel_expression) for each in k_exemplars] 
            logger.info('SELECTED MODEL FOR ROUND %s: %s', i, k_exemplars_string)
            save_genetic_pool(genetic_pool_dict, dirpath, logs=logs, save_model_logs=False)
            round_selected_model[i] = k_exemplars_string
            with open(f'./{dirpath}/selected_model_log.json', 'w') as f:
                json.dump(round_selected_model, f, indent=4)
            #####################################################################

            ##### analyzer llm variables
            response = []
            response_tree_dict = {}
            mother_dict = {}
            tot_tmp_response = []
            tot_tmp_response_cp = []
            cpcount=0
            ##### CURRENT ROUND DIRPATH            
            dirpath_i = make_dirpath(directory_path, data_name, i)
            
            ######################### MODEL GENERATION & PARSING FOR ROUND i ###########################
            llm_selected_possible_expansion = []
            for each_exemplars, each_exemplars_string in zip(k_exemplars, k_exemplars_string):
                #### BIC metric 추가하기
                if isinstance(each_exemplars, list):
                    BIC_list = [BIC(each.model, each.model.log_likelihood(), len(X), each.model._size_transformed()) for each in each_exemplars]
                    best_each_exemplars = each_exemplars[np.argmin(BIC_list)]
                    each_exemplars = best_each_exemplars
                    each_exemplars_string = "%s"%(best_each_exemplars.kernel_expression)
                
                if USE_ANALYZER:
                    tmp_response, model_logs = analyze_then_generate_kernels([each_exemplars], [each_exemplars_string], X=X, Y=Y, data_name=data_name, round=i, use_changepoint=False, model=vlm_model_name, logger=logger)
                    tot_tmp_response.extend(tmp_response)
                    # tmp_response_cp, model_logs_cp = analyze_then_generate_kernels([each_exemplars], [each_exemplars_string], X=X, Y=Y, data_name=data_name, round=i, use_changepoint=True, model=vlm_model_name, logger=logger)
                    # tot_tmp_response_cp.extend(tmp_response_cp)
                    tmp_response_cp, tot_tmp_response_cp = [], []
                    
                else:
                    tmp_response = []
                    tmp_response_cp = []
                    tmp = expand(each_exemplars.kernel_expression, production_rules['Minimal'])
                    for eacheach in tmp:
                        if 'CP' or 'CW' in '%s'%eacheach:
                            tmp_response_cp.append("%s"%eacheach)
                            tot_tmp_response_cp.append("%s"%eacheach)
                        else:
                            tmp_response.append("%s"%eacheach)
                            tot_tmp_response.append("%s"%eacheach)
                    
                response_tree_dict[each_exemplars_string] = tmp_response
                response_tree_dict[each_exemplars_string].extend(tmp_response_cp)
                
                for tr in tmp_response+tmp_response_cp: 
                    try: name = "%s"%parse_kernel_name(tr)
                    except: 
                        logger.info('%s this should be not parsed!', tr)
                        name = None
                    if name is not None:
                        mother_dict[name] = each_exemplars_string
                        llm_selected_possible_expansion.append(name)
                
            logger.info('ANALYZER LLM GENERATED MODELS AT ROUND %s: %s %s', i, str(len(mother_dict)), mother_dict.keys())
            tree_dict[i] = response_tree_dict
            with open(f'./{dirpath}/tree_dict.json', 'w') as f: json.dump(tree_dict, f, indent=4)
            ##################################################################################

            ######################### MODEL FITTING FOR ROUND i ###########################
            fitted_selected_possible_expansion = []
            if not USE_INHERIT_AT_NO_ISLAND:
                fitted_selected_possible_expansion = fit_mods_parallel_processes(X, Y, llm_selected_possible_expansion, restarts, optimiser, max_retries) ### first is random expansion (optimize_restart)
            ##### PLEASE HIERARCHICALLY MANAGE THE ISLANDS
            
            else:
                generated_kernels_model = []
                for ii in range(len(llm_selected_possible_expansion)): 
                    cur_model = parse_kernel_model('%s'%llm_selected_possible_expansion[ii]) #### .model
                    parent_models = mother_dict[llm_selected_possible_expansion[ii]]
                    inherit_params = build_hierarchical_dict(get_param_dict(genetic_pool_dict[parent_models]['model'].model))
                    param_init(cur_model, inherit_params)
                    ##### MAKE GPModel
                    cur_gp_model = GPModel(X, Y, parse_kernel_name(llm_selected_possible_expansion[ii]))
                    cur_gp_model.model = GPRegression(X, Y, cur_model)
                    generated_kernels_model.append(cur_gp_model)
                
                print('tuning start')
                fitted_generated_kernels = fit_mods_parallel_processes_with_initialize(X, Y, generated_kernels_model, restarts, optimiser, max_retries) 
                
                print('tuning done and start again')
                if set_se_constraint: 
                    fitted_generated_kernels.extend(my_fit_mods_parallel_processes(X, Y, [parse_kernel_name(e) for e in llm_selected_possible_expansion], restarts, optimiser, max_retries, se_kernel_const=[1, 10]))
                    
                # else: 
                 
                fitted_generated_kernels.extend(fit_mods_parallel_processes(X, Y, [parse_kernel_name(e) for e in llm_selected_possible_expansion], restarts, optimiser, max_retries))
                # print('tuning done')
            
            ################# SELECT BEST PARAMETER IN EACH KERNELS #######################
            fgk_dict = {}
            for each_fgk in fitted_generated_kernels: 
                name = "%s"%each_fgk.kernel_expression
                bic = BIC(each_fgk.model, each_fgk.model.log_likelihood(), len(X), each_fgk.model._size_transformed())
                val_mse = mse(each_fgk.predict(valX)['mean'], valY)[0]
                param_score = bic+100*val_mse
                
                if name in fgk_dict:
                    prev_param_score = fgk_dict[name]['param_score']
                    if param_score<prev_param_score:
                        fgk_dict[name] = {'model': each_fgk, 'param_score': param_score}
                    
                else:
                    fgk_dict[name] = {'model': each_fgk, 'param_score': param_score}
            ##################################################################################
            
            # TODO: 커널 삭제 구현
            # 1. BIC가 top percentile 몇일때 삭제
            # 2. parameter evolve가 크게 없는데 안조으면 삭제
            
            ####################### MODEL VISUAL SCORING #######################
            ############ plot the data function for one time!!!!
            model = fgk_dict[name] #### just for random model with all same dataset
            if USE_VISUAL_SCORE:
                for ii, name in enumerate(fgk_dict.keys()):
                    model = fgk_dict[name]['model']
                    
                    likelihood = model.model.log_likelihood()
                    bic = BIC(model.model, likelihood, len(X), model.model._size_transformed())
                    val_mse = mse(model.predict(valX)['mean'], valY)[0]
                    loo = np.sum(model.model.inference_method.LOO(model.model.kern, valX, valY, model.model.likelihood, model.model.posterior))
                    
                    data_vis_path = f'{dirpath_i}/{ii} - {name}-data.png'
                    plotgp(model.model, ymin=np.min(Y)-range_*0.3, ymax=np.max(Y)+range_*0.3, xlim=[np.min(model.X), np.max(model.X)], plot_data=True, plot_samples=False)
                    plt.savefig(data_vis_path)
                    plt.cla(); plt.clf()

                    mean_vis_path = f'{dirpath_i}/{ii} - {name}-mean.png'
                    plotgp(model.model, ymin=np.min(Y)-range_*0.3, ymax=np.max(Y)+range_*0.3, xlim=[np.min(model.X), np.max(model.X)], plot_data=False, plot_samples=True, plot_confidence=False)
                    plt.savefig(mean_vis_path)
                    plt.cla(); plt.clf()
                    
                    conf_vis_path = f'{dirpath_i}/{ii} - {name}.png'
                    plotgp(model.model, ymin=np.min(Y)-range_*0.3, ymax=np.max(Y)+range_*0.3)
                    plt.savefig(conf_vis_path)
                    plt.cla(); plt.clf()
                    
                    score_for_criteria = evaluate_vision_overall(name, data_vis_path, mean_vis_path, conf_vis_path, data_name, vlm_model_name, logger=logger)
                    
                    if name in genetic_pool_dict:
                        param_score = genetic_pool_dict[name]['BIC'] + 100 * genetic_pool_dict[name]['val mse']
                        if bic+100*val_mse < param_score:
                            genetic_pool_dict.pop(name)
                    
                    if not name in genetic_pool_dict:
                        genetic_pool_dict[name] = {
                            'score for match': score_for_criteria.get('score for match', 0),
                            'score for confidence': score_for_criteria.get('score for confidence', 0),
                            'score for consistency': score_for_criteria.get('score for consistency', 0),
                            'score': score_for_criteria.get('score for match', 0) + score_for_criteria.get('score for confidence', 0) + score_for_criteria.get('score for consistency', 0),
                            'model': model,
                            'likelihood': likelihood,
                            'BIC': bic,
                            'val mse': val_mse,
                            'loo': loo,
                            'round': i
                        }
                        
                    logger.info('TOTAL SCORE: %s', genetic_pool_dict[name]['score'])
                    logger.info('BIC: %s', genetic_pool_dict[name]['BIC'])
                    logger.info('val mse: %s', genetic_pool_dict[name]['val mse'])
                    logger.info('LOO: %s', genetic_pool_dict[name]['loo'])
                    logger.info(f'================================================')
            
            else:
                for ii, name in enumerate(fgk_dict.keys()):
                    if name in genetic_pool_dict:
                        param_score = genetic_pool_dict[name]['BIC'] + 100 * genetic_pool_dict[name]['val mse']
                        if bic+100*val_mse < param_score:
                            genetic_pool_dict.pop(name)
                    
                    if not name in genetic_pool_dict:
                        genetic_pool_dict[name] = {
                            'score for match': 0,
                            'score for confidence': 0,
                            'score for consistency': 0,
                            'score': 0,
                            'model': model,
                            'likelihood': likelihood,
                            'BIC': bic,
                            'val mse': val_mse,
                            'loo': loo,
                            'round': 0
                        }
        
        
        
        
        ########################################### FINAL ROUND DONE ############################################
        # dirpath = os.path.join(directory_path, data_name)
        tmp_sorted_models, tmp_sorted_scores = sort_genetic_pools(genetic_pool_dict, mymet_hyp)
        k_exemplars = tmp_sorted_models[:top_sample]
        logger.info('')
        logger.info('----------FINAL SELECTED MODELS-----------')

        for ii, model in enumerate(k_exemplars):
            ##### calc BIC and sort by BIC
            if ii!=0:
                origX = np.expand_dims(origX, axis=1)
            plt.figure(figsize=(10,6))
            # plt.plot(origX, origY, 'k.', label='data')
            plt.plot(X, Y, 'k.', label='train data')
            plt.plot(valX, valY, 'k.', label='val data')
            plt.plot(testX, testY, 'c.', label='test data')

            output = model.predict(origX)
            en_mean, en_cov, en_low_quantile, en_high_quantile = output['mean'], output['covariance'], output['low_quantile'], output['high_quantile']
            
            if origX.ndim==2: origX = origX.flatten()
            if en_low_quantile.ndim==2: en_low_quantile = en_low_quantile.flatten()
            if en_high_quantile.ndim==2: en_high_quantile = en_high_quantile.flatten()
            
            plt.plot(origX, en_mean, 'b-', label='mean')
            plt.fill_between(origX, en_low_quantile, en_high_quantile, color='lightblue', alpha=0.5, label='confidence')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('data, mean, and confidence intervals')
            plt.legend()
            range_ = np.max(origY)-np.min(origY)
            plt.ylim(ymin=np.min(origY)-range_*0.3, ymax=np.max(origY)+range_*0.3)
            plt.grid()

            name = "%s"%(model.kernel_expression)
            likelihood = model.model.log_likelihood()
            logger.info(f'=============KERNEL: {name}==============')
            logger.info('SCORE FOR MATCH: %s', genetic_pool_dict[name]['score for match'])
            logger.info('SCORE FOR CONFIDENCE: %s', genetic_pool_dict[name]['score for confidence'])
            logger.info('SCORE FOR CONSISTENCY: %s', genetic_pool_dict[name]['score for consistency'])
            logger.info('TOTAL SCORE: %s', genetic_pool_dict[name]['score'])
            logger.info('BIC: %s', genetic_pool_dict[name]['BIC'])
            logger.info('val mse: %s', genetic_pool_dict[name]['val mse'])
            logger.info('LOO: %s', genetic_pool_dict[name]['loo'])
            logger.info('================================================')
            tmp = "%s"%(model.kernel_expression)
            plt.savefig(f'{dirpath}/final{ii} - {tmp}')
            plt.cla(); plt.clf()
                
        with open(f'./{dirpath}/genetic_pools_{data_name}.json', 'w') as f:
            writable_genetic_pools = {}
            sorted_genetic_pools = dict(sorted(genetic_pool_dict.items(), key=lambda item: item[1]['mymet']))

            for each in sorted_genetic_pools:
                writable_genetic_pools[each] = {
                    'round': genetic_pool_dict[each]['round'], 
                    'vision-score': genetic_pool_dict[each]['score'],
                    'vison-score-for-match': genetic_pool_dict[each]['score for match'],
                    'vision-score-for-confidence': genetic_pool_dict[each]['score for confidence'],
                    'vision-score-for-consistency': genetic_pool_dict[each]['score for consistency'],
                    'likelihood': genetic_pool_dict[each]['likelihood'],
                    'BIC': genetic_pool_dict[each]['BIC'],
                    'mymet': genetic_pool_dict[each]['mymet'],
                    'val mse': genetic_pool_dict[each]['val mse'],
                }
                
            json.dump(writable_genetic_pools, f, indent=4)
            
        save_genetic_pool(genetic_pool_dict, dirpath, logs=[], save_model_logs=False)
            
        with open(f'./{dirpath}/selected_model_log.json', 'w') as f:
            json.dump(round_selected_model, f, indent=4)
            
        with open(f'./{dirpath}/genetic_pools_{data_name}.pickle', 'wb') as f:
            pickle.dump(sorted_genetic_pools, f)
            
        with open(f'./{dirpath}/tree_dict.json', 'w') as f:
            json.dump(tree_dict, f, indent=4)