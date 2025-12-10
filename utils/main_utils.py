from vision_score_gpt2_0105_func import get_structure_sim_response, get_mean_sim_response, get_confidence_sim_response
from plot_func import plot as plotgp
import numpy as np
import base64
import random
import warnings
import GPy
from openai import OpenAI
import os
import shutil
import json
import pickle
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timezone, timedelta

# from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
# from qwen_vl_utils import process_vision_info

# model_name = "Qwen/Qwen2.5-VL-72B-Instruct"
# # default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_name, torch_dtype="auto", device_map="auto"
# )
# # default processor
# processor = AutoProcessor.from_pretrained(model_name)

os.environ["OPENAI_API_KEY"] = ""
client = OpenAI()

def mse(true_values, predicted_values):
    n = len(true_values)
    
    mse = sum((true_val - pred_val) ** 2 for true_val, pred_val in zip(true_values, predicted_values)) / n
    return mse

def BIC(m, ll, n, k): return -2 * ll + k * np.log(n)

def process_structure_sim(encoded_frame, kernel_name, data_name):
    scores = []
    for consistency_i in range(2):
        score_response = get_structure_sim_response(encoded_frame=encoded_frame, kernel_name=kernel_name, data_name=data_name)
        scores.append(score_response['score'])
    return scores
    
def process_mean_sim(encoded_frame1, encoded_frame2, kernel_name, data_name):
    scores = []
    for consistency_i in range(2):
        score_response = get_mean_sim_response(encoded_frame1=encoded_frame1, encoded_frame2=encoded_frame2, kernel_name=kernel_name, data_name=data_name)
        scores.append(score_response['score'])
    return scores
    
def process_confidence_sim(encoded_frame, kernel_name, data_name):
    scores = []
    for consistency_i in range(2):
        score_response = get_confidence_sim_response(encoded_frame=encoded_frame, kernel_name=kernel_name, data_name=data_name)
        scores.append(score_response['score'])
    return scores

def weighted_sample_without_replacement(population, weights, k, rng=random):
    # v = [rng.random()*0.1 ** (1 / w) for w in weights]
    v = [rng.random()*0.1*w for w in weights]
    order = sorted(range(len(population)), key=lambda i: v[i])
    ret = [population[i] for i in order[-k:]]
    if population[0] not in ret:
        ret.append(population[0]) #### alway include the best one
    return ret

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def find_se_lengthscale_set_constraint(kernel, ret, constraint=[10, 10**2]):
    with warnings.catch_warnings(): 
        warnings.simplefilter('ignore')
        if isinstance(kernel, GPy.kern.src.add.Add) or isinstance(kernel, GPy.kern.src.prod.Prod):
            for part in kernel.parts:
                ret = find_se_lengthscale_set_constraint(part, ret, constraint)
                
        else:
            if kernel.name=='rbf' and hasattr(kernel, 'lengthscale'):
                kernel.lengthscale.constrain_bounded(constraint[0], constraint[1])
                kernel.variance.constrain_bounded(1, 10)
                print('SE kernel found!! in function')
                ret = True
        return ret
    
def sort_genetic_pools(genetic_pool_dict, score_hyperparam):
    tmp_reverse_dict = {}
    best_bic = 100000
    for each in genetic_pool_dict:
        if np.min(genetic_pool_dict[each]['BIC'])<best_bic:
            best_bic = np.min(genetic_pool_dict[each]['BIC'])
            
    for each in genetic_pool_dict.copy():
        bic = genetic_pool_dict[each]['BIC'] #### now the list
        score = np.mean(genetic_pool_dict[each]['score']) #### now the list
        funcs_round = genetic_pool_dict[each]['round']
        new_score = []

        if type(bic)==list:
            for b in bic: new_score.append(b-np.log1p(score)*score_hyperparam-np.log1p(funcs_round)*score_hyperparam)
            new_score = np.min(new_score)
        else:
            new_score = bic-np.log1p(score)*score_hyperparam-np.log1p(funcs_round)*score_hyperparam
            
        genetic_pool_dict[each]['mymet'] = new_score
    
    for each in genetic_pool_dict: 
        score = genetic_pool_dict[each]['score']
        bic = genetic_pool_dict[each]['BIC'] ####### later change this to score!!!!
        
        new_score = genetic_pool_dict[each]['mymet']
        model = genetic_pool_dict[each]['model']
        if new_score not in tmp_reverse_dict:
            tmp_reverse_dict[new_score] = []
        tmp_reverse_dict[new_score].append(model)
    
    
    tmp_sorted_models = []
    tmp_sorted_scores = []
    # for each in dict(sorted(tmp_reverse_dict.items(), reverse=True)): #--> only for score!
    for each in dict(sorted(tmp_reverse_dict.items())):
        for eacheach in tmp_reverse_dict[each]:
            tmp_sorted_models.append(eacheach)
            tmp_sorted_scores.append(-1*each) ## BIC: smaller the better
    
    return tmp_sorted_models, tmp_sorted_scores
    
########## FOR SAVING AND LOGGING ###############
def make_dirpath(directory_path, data_name, round):
    dirpath = os.path.join(directory_path, data_name, f'round{round}')
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    else:
        shutil.rmtree(dirpath)
        os.makedirs(dirpath)
        
    return dirpath
    
def save_genetic_pool(genetic_pool_dict, dirpath, logs=None, save_model_logs=False):
    with open(f'./{dirpath}/genetic_pools.json', 'w') as f:
        writable_genetic_pools = {}

        for each in genetic_pool_dict:
            writable_genetic_pools[each] = {
                'round': genetic_pool_dict[each]['round'], 
                'vision-score': genetic_pool_dict[each]['score'],
                'vison-score-for-match': genetic_pool_dict[each]['score for match'],
                'vision-score-for-confidence': genetic_pool_dict[each]['score for confidence'],
                'vision-score-for-consistency': genetic_pool_dict[each]['score for consistency'],
                'likelihood': genetic_pool_dict[each]['likelihood'],
                'BIC': genetic_pool_dict[each]['BIC'],
                'val mse': genetic_pool_dict[each]['val mse']
            }
            
        json.dump(writable_genetic_pools, f, indent=4)
        
    with open(f'./{dirpath}/genetic_pool_dict.pickle', 'wb') as f:
        pickle.dump(genetic_pool_dict, f)
        
    if save_model_logs:
        with open(f'./{dirpath}/model_logs_all.json', 'w') as f:
                json.dump(logs, f, indent=4)
                
def save_initial_plot(k_exemplars, genetic_pool_dict, dirpath, X, Y, valX, valY, range_):
    writable_genetic_pools = {}
    
    for a, b in k_exemplars:
        writable_genetic_pools["%s"%(a.kernel_expression)] = {
            'likelihood': [a.model.log_likelihood(), b.model.log_likelihood()],
            'BIC': [BIC(a.model, a.model.log_likelihood(), len(X), a.model._size_transformed()), BIC(b.model, b.model.log_likelihood(), len(X), b.model._size_transformed())],
            'val mse': [mse(a.predict(valX)['mean'], valY)[0], mse(b.predict(valX)['mean'], valY)[0]],
            'loo': [np.sum(a.model.inference_method.LOO(a.model.kern, valX, valY, a.model.likelihood, a.model.posterior)), np.sum(b.model.inference_method.LOO(b.model.kern, valX, valY, b.model.likelihood, b.model.posterior))],
            'round': -1
        }
        
        plotgp(a.model, ymin=np.min(Y)-range_*0.3, ymax=np.max(Y)+range_*0.3)
        plt.savefig(f'{dirpath}/{a.kernel_expression}-1vlm.png')
        plt.cla()
        plt.clf()
        
        plotgp(b.model, ymin=np.min(Y)-range_*0.3, ymax=np.max(Y)+range_*0.3)
        plt.savefig(f'{dirpath}/{b.kernel_expression}-2random.png')
        
    with open(f'./{dirpath}/genetic_pools_init.json', 'w') as f:
        json.dump(writable_genetic_pools, f, indent=4)
        
    with open(f'./{dirpath}/genetic_pools_init.pickle', 'wb') as f:
        genetic_pool_dict = writable_genetic_pools
        for a, b in k_exemplars:
            genetic_pool_dict["%s"%(a.kernel_expression)]['model'] = [a, b]
        pickle.dump(genetic_pool_dict, f)
        
    return genetic_pool_dict

######### LLM Inference #########
def llm_inference(conversation_history, append_response=True, model_name='gpt-4o-mini'):
    if 'gpt' in model_name:
        print(f'gpt inference. {model_name}')
        output = client.chat.completions.create(
            model=model_name,
            messages=conversation_history,
            temperature=0,
        )
        
        response = output.choices[0].message.content
        
    elif 'qwen' in model_name:
        print('qwen inference.')
        text = processor.apply_chat_template(
            conversation_history, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(conversation_history)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    if append_response:
        conversation_history.append(
            {"role": "assistant", "content":response}
        )
            
    return conversation_history, response

def llm_inference_qwen(conversation_history, append_response=True, model='qwen'):
    # Preparation for inference
    text = processor.apply_chat_template(
        conversation_history, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(conversation_history)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    if append_response:
        conversation_history.append(
            {"role": "assistant", "content":response}
        )
    
    return response
################################

def make_logger(current_dir):
    if not os.path.exists(current_dir): os.makedirs(current_dir)
    filename = datetime.now(timezone(timedelta(hours=9))).strftime(f'{current_dir}/log-%Y%m%d-%H:%M:%S.log')
    
    logger = logging.getLogger("myLogger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(u'[%(levelname)s - %(filename)s:%(lineno)d] %(message)s', '%Y.%m.%d %I:%M:%S')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    ##################################
    return logger