import transformers
import torch
import numpy as np

from glob import glob
import os
import sys
import  natsort
from PIL import Image
import requests
import copy
import torch

from cfgs.asmd_cfg import parse_asmd_cfg

import logging
from .prompts import VISION_SCORE_STRUCT_SIM_SYSTEMPROMPT, VISION_SCORE_MEAN_SIM_SYSTEMPROMPT, VISION_SCORE_CONFIDENCE_SYSTEMPROMPT, VISION_SCORE_STRUCTURE_SIM_NO_IMG, VISION_SCORE_CONFIDENCE_NO_IMG
from .main_utils import llm_inference

from openai import OpenAI
import base64
os.environ["OPENAI_API_KEY"] = ""
client = OpenAI()

DEBUG = False

####### VISION SCORE UTILS #######
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        ret = base64.b64encode(image_file.read()).decode("utf-8")
    return ret

def parse_score_response_ss(score_response):
    if '```python' in score_response:
        dictcode = score_response.split('```python')[1].split('```')[0]
    else:
        dictcode = score_response
    if "'kernel5'" not in dictcode and '"kernel5"' not in dictcode:
        dictcode = dictcode.replace("kernel5", "'kernel5'")
    
    if "'score for structure similarity'" not in dictcode and '"score for structure similarity"' not in dictcode:
        dictcode = dictcode.replace("score for structure similarity", "'score'")
        
    else:
        dictcode = dictcode.replace("score for structure similarity", "score")
    dictcode = eval(dictcode)
    
    # import pdb; pdb.set_trace()
    if 'kernel5' not in dictcode:
        print('parse error!!')
        import pdb; pdb.set_trace()
    return dictcode

def parse_score_response(score_response):
    if '```python' in score_response:
        dictcode = score_response.split('```python')[1].split('```')[0]
    else:
        dictcode = score_response
        
    if "'kernel5'" not in dictcode and '"kernel5"' not in dictcode:
        dictcode = dictcode.replace("kernel5", "'kernel5'")
    
    if "'score'" not in dictcode and '"score"' not in dictcode:
        dictcode = dictcode.replace("score", "'score'")
    
    if dictcode.strip()[0]!='{':
        dictcode = '{' + dictcode.strip() + '}'
    
    executed = True
    try:
        dictcode = eval(dictcode)
    except:
        executed = False
    if not executed:
        import pdb; pdb.set_trace()
        
    
    return dictcode

######### EVALUARION FUNCTIONS ############
def evaluate_ss(encoded_frame, kernel_name, data_name='02-solar', model='gpt-4o'):
    bad_img1 = encode_image('/node_data_2/mok/asmd/ablation/figs2-R5-restarts5-noanalyzer-noisland-visual/01-airline/round1/8 - C + (PER + C) * (PER + C).png') #### 45 45 50
    bad_img2 = encode_image('/node_data_2/mok/asmd/ablation/figs2-R5-restarts5-noanalyzer-noisland-visual/01-airline/round1/6 - C + LIN * (PER + C).png') #### 20 30 50
    bad_img3 = encode_image('/node_data_2/mok/asmd/ablation/figs2-R5-restarts5-noanalyzer-noisland-visual/01-airline/round1/3 - PER + SE + C.png') #### 30 40 50
    bad_img4 = encode_image('/node_data_2/mok/asmd/ablation/figs2-R5-restarts5-noanalyzer-noisland-visual/01-airline/round1/1 - PER + PER + C.png') #### 45 45 10
    bad_img5 = encode_image('/node_data_2/mok/asmd/ablation/figs2-R5-restarts5-noanalyzer-noisland-visual/01-airline/round1/31 - (PER + C) * (SE + WN).png') # 45 45 20
    
    
    #############
    if 'gpt' in model:
        content = [
            """Please generate the response in the form of a Python dictionary string with keys of kernel name. 'score for structure similarity' are in INTEGER, not STRING.""",
            "Check the few shot examples for your scoring. After the step by step reasoning, your final output should look like this: \n",
            *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, [bad_img1]),
            """{kernel1: {score for structure similarity: 50}""",
            *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, [bad_img4]),
            """{kernel2: {score for structure similarity: 10}""",
            *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, [bad_img2]),
            """{kernel3: {score for structure similarity: 50}""",
            *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, [bad_img5]),
            """{kernel4: {score for structure similarity: 20}""",
            
            """Please evaluate the structure similarity of the kernel5 in the image here. Output should be only the score for the kernel 5. kernel5:""",
            *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, [encoded_frame]),
        ]
    
    if 'qwen' in model:
        content = [
            {'type':'text', 'text': "Please generate the response in the form of a Python dictionary string with keys of kernel name. 'score for structure similarity' are in INTEGER, not STRING.\n Check the few shot examples for your scoring. After the step by step reasoning, your final output should look like this: \n"},
            
            ##### kernel 1
            {'type':'image', 'image':'/node_data_2/mok/asmd/ablation/figs2-R5-restarts5-noanalyzer-noisland-visual/01-airline/round1/8 - C + (PER + C) * (PER + C).png'},
            {'type':'text', 'text':'{kernel1: {score for structure similarity: 50}'},
            ##### kernel 2
            {'type':'image', 'image':'/node_data_2/mok/asmd/ablation/figs2-R5-restarts5-noanalyzer-noisland-visual/01-airline/round1/1 - PER + PER + C.png'},
            {'type':'text', 'text':'{kernel2: {score for structure similarity: 10}'},
            ##### kernel 3
            {'type':'image', 'image':'/node_data_2/mok/asmd/ablation/figs2-R5-restarts5-noanalyzer-noisland-visual/01-airline/round1/6 - C + LIN * (PER + C).png'},
            {'type':'text', 'text':'{kernel3: {score for structure similarity: 50}'},
            ##### kernel 4
            {'type':'image', 'image':'/node_data_2/mok/asmd/ablation/figs2-R5-restarts5-noanalyzer-noisland-visual/01-airline/round1/31 - (PER + C) * (SE + WN).png'},
            {'type':'text', 'text':'{kernel4: {score for structure similarity: 20}'},
            {'type':'text', 'text':'Please evaluate the structure similarity of the kernel5 in the image here. Output should be only the score for the kernel 5. kernel5:'}
        ]
    
    chat_history = [  
        {"role": "system", "content": VISION_SCORE_STRUCT_SIM_SYSTEMPROMPT},
        {"role": "user", "content": content}
    ]
    _, score_response = llm_inference(chat_history, append_response=False, model_name=model)
    
    dictcode = parse_score_response_ss(score_response)
    return dictcode['kernel5']

def evaluate_ss_no_img(encoded_frame, kernel_name, data_name='02-solar', model='gpt-4o'):
    content = VISION_SCORE_STRUCTURE_SIM_NO_IMG%(encoded_frame['train_mean_prediction'], encoded_frame['train_real_data'], encoded_frame['train_covariance'], encoded_frame['validation_mean_prediction'], encoded_frame['validation_real_data'], encoded_frame['validation_covariance'])
        
    chat_history = [  
        {"role": "system", "content": VISION_SCORE_STRUCT_SIM_SYSTEMPROMPT},
        {"role": "user", "content": content}
    ]
    _, score_response = llm_inference(chat_history, append_response=False, model_name=model)
    
    dictcode = parse_score_response_ss(score_response)
    return dictcode['kernel5']

def evaluate_mean(encoded_frame1, encoded_frame2, kernel_name, data_name='02-solar', model='gpt-4o'):
    if 'gpt' in model:
        content = [
            """Please generate the response in the form of a Python dictionary string with keys of kernel name. score is in INTEGER, not STRING.""",
            """Please evaluate how similar the two graphs are. First is data graph and second graph is predicted mean graph. Output should be the score for the kernel1. kernel1:""",
            *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, [encoded_frame1, encoded_frame2]),
            "%s:"%"kernel1",
        ]
    
    elif 'qwen' in model:
        content = [
            {"type": "text", "text": "Please generate the response in the form of a Python dictionary string with keys of kernel name. score is in INTEGER, not STRING. Do not print out any other things.\n Please evaluate how similar the two graphs are. First is data graph and second graph is predicted mean graph. Output should be the score for the kernel1. kernel1:"},
            {"type": "image", "image": encoded_frame1},
            {"type": "image", "image": encoded_frame2},
            {"type": "text", "text": "kernel1:"}
        ]

    chat_history = [  
        {"role": "system", "content": VISION_SCORE_MEAN_SIM_SYSTEMPROMPT},
        {"role": "user", "content": content}
    ]
    _, score_response = llm_inference(chat_history, append_response=False, model_name=model)
    
    if '```python' in score_response:
        dictcode = score_response.split('```python')[1].split('```')[0]
    else:
        dictcode = score_response
    
    if dictcode[0]=='`':
        dictcode = dictcode[1:]
    if dictcode[-1]=='`':
        dictcode = dictcode[:-1]
        
    dictcode = eval(dictcode)

    return {'score': dictcode['kernel1']}

def evaluate_mean_no_img(predicted_val, data_val, kernel_name, data_name='02-solar', model='gpt-4o'):
    content = f"""
Please generate the response in the form of a Python dictionary string with keys of kernel name. score is in INTEGER, not STRING.

Please evaluate how similar the data and the prediction. Output should be the score for the kernel1. kernel1:
prediction: {predicted_val}
data: {data_val}
"""
    chat_history = [  
        {"role": "system", "content": VISION_SCORE_MEAN_SIM_SYSTEMPROMPT},
        {"role": "user", "content": content}
    ]
    _, score_response = llm_inference(chat_history, append_response=False, model_name=model)
    
    dictcode = score_response.split('```python')[1].split('```')[0]
    dictcode = eval(dictcode)
    return {'score': dictcode['kernel1']}

def evaluate_confidence(encoded_frame, kernel_name, data_name='02-solar', model='gpt-4o'):
    bad_img1_path = '/node_data_2/mok/asmd/ablation/figs2-R5-restarts5-analyzer-noisland-visual/01-airline/round4/19 - SE + WN.png'
    bad_img2_path = '/node_data_2/mok/asmd/ablation/figs2-R5-restarts5-analyzer-noisland-visual/01-airline/round4/21 - CP(SE + SE * (PER + LIN), PER + C).png'
    bad_img3_path = '/node_data_2/mok/asmd/ablation/figs2-R5-restarts5-analyzer-noisland-visual/01-airline/round4/20 - C + SE * (PER + LIN).png'
    bad_img4_path = '/node_data_2/mok/asmd/ablation/figs2-R5-restarts5-analyzer-noisland-visual/01-airline/round4/17 - SE + LIN * (PER + C).png'
    bad_img1 = encode_image(bad_img1_path)
    bad_img2 = encode_image(bad_img2_path)
    bad_img3 = encode_image(bad_img3_path)
    bad_img4 = encode_image(bad_img4_path)

    if 'gpt' in model:
        content = [] 
        #############
        content.extend(
            [
                """Please generate the response in the form of a Python dictionary string with keys of kernel name. score is in INTEGER, not STRING.""",
                "Check the 4 reference for your scoring. ", # Check whether the confidence area is bigger or smaller than the given example
                *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, [bad_img3]),
                """{kernel3: {score: 45}""",
                *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, [bad_img4]),
                """{kernel4: {score: 35}""",
                *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, [bad_img1]),
                """{kernel1: {score: 30}""",
                *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, [bad_img2]),
                """{kernel2: {score: 20}""",
                # *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "high"}}, [bad_img6]),
                # """{kernel4: {score for confidence: 10}""",

                """Please evaluate how small the confidence interval area is. Think step by step. The final output should be only the score for the kernel5. kernel5:""",
                *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, [encoded_frame]),
                # "{kernel5:",
            ]
        )
        
    elif 'qwen' in model:
        content = [
            {"type": "text", "text": "Please generate the response in the form of a Python dictionary string with keys of kernel name. score is in INTEGER, not STRING.\nCheck the 4 reference for your scoring."},
            {"type": "image", "image": bad_img3_path},
            {"type": "text", "text": "{kernel3: {score: 45}"},
            {"type": "image", "image": bad_img4_path},
            {"type": "text", "text": "{kernel4: {score: 35}"},
            {"type": "image", "image": bad_img1_path},
            {"type": "text", "text": "{kernel1: {score: 30}"},
            {"type": "image", "image": bad_img2_path},
            {"type": "text", "text": "{kernel2: {score: 20}"},
            {"type": "text", "text": "Please evaluate how small the confidence interval area is. Think step by step. The final output should be only the score for the kernel5. kernel5:"},
            {"type": "image", "image": encoded_frame}
        ]

    chat_history = [  
        {"role": "system", "content": VISION_SCORE_CONFIDENCE_SYSTEMPROMPT},
        {"role": "user", "content": content}
    ]
    _, score_response = llm_inference(chat_history, append_response=False, model_name=model)

    dictcode = parse_score_response(score_response)
    
    if type(dictcode)==set:
        lst = [e for e in dictcode]
        return {'score': lst[0]}
    else:
        return dictcode['kernel5']

def evaluate_confidence_no_img(quantiles, kernel_name, data_name='02-solar', model='gpt-4o'):
    content = VISION_SCORE_CONFIDENCE_NO_IMG
    content+= f"""kernel 5 low quantile: {quantiles[0]}"""
    content+= f"""kernel 5 high quantile: {quantiles[1]}"""
    content+= """{kernel5:"""

    chat_history = [  
        {"role": "system", "content": VISION_SCORE_CONFIDENCE_SYSTEMPROMPT},
        {"role": "user", "content": content}
    ]
    _, score_response = llm_inference(chat_history, append_response=False, model_name=model)

    dictcode = parse_score_response(score_response)
    return dictcode['kernel5']


def evaluate_vision_overall(kernel_name, data_vis_path, mean_vis_path, conf_vis_path, data_name, vlm_model_name='gpt-4o-mini', repeat=2, logger=None):
    tmp_scores = []
    tmp_scores_mean = []
    tmp_scores_confidence = []
        
    for _ in range(repeat):
        if 'gpt' in vlm_model_name:
            score_response = evaluate_mean(encoded_frame1=encode_image(data_vis_path), encoded_frame2=encode_image(mean_vis_path), kernel_name=kernel_name, data_name=data_name, model=vlm_model_name)
            tmp_scores_mean.append(score_response['score'])
            
            score_response = evaluate_confidence(encoded_frame=encode_image(conf_vis_path), kernel_name=kernel_name, data_name=data_name, model=vlm_model_name)
            tmp_scores_confidence.append(score_response['score'])
            
            score_response = evaluate_ss(encoded_frame=encode_image(conf_vis_path), kernel_name=kernel_name, data_name=data_name, model=vlm_model_name)
            tmp_scores.append(score_response['score'])
            
        elif 'qwen' in vlm_model_name:
            score_response = evaluate_mean(encoded_frame1=data_vis_path, encoded_frame2=mean_vis_path, kernel_name=kernel_name, data_name=data_name, model=vlm_model_name)
        
            score_response = evaluate_confidence(encoded_frame=conf_vis_path, kernel_name=kernel_name, data_name=data_name, model=vlm_model_name)
            tmp_scores_confidence.append(score_response['score'])

            score_response = evaluate_ss(encoded_frame=conf_vis_path, kernel_name=kernel_name, data_name=data_name, model=vlm_model_name)
            tmp_scores.append(score_response['score'])
    
    kernel_name = kernel_name
    score_for_criteria = {}
    # score_for_criteria['score for match']= np.mean(tmp_scores_mean)
    # score_for_criteria['score for confidence']= np.mean(tmp_scores_confidence)
    # score_for_criteria['score for consistency']= np.mean(tmp_scores)
    score_for_criteria['score for match']= tmp_scores_mean
    score_for_criteria['score for confidence']= tmp_scores_confidence
    score_for_criteria['score for consistency']= tmp_scores
    
    logger.info(f'=============KERNEL: {kernel_name}==============')
    logger.info(f'SCORE FOR MATCH: {tmp_scores_mean}')
    logger.info(f'SCORE FOR CONFIDENCE: {tmp_scores_confidence}')
    logger.info(f'SCORE FOR CONSISTENCY: {tmp_scores}')
    
    return score_for_criteria