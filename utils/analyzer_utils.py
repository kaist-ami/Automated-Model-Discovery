import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import os
from contextlib import redirect_stdout
from io import StringIO
import base64
import pickle
from utils.prompts import SYSTEMPROMPT, PROMPT_ANALYZE_AND_FIND_FIRST_KERNEL, PROMPT_ANALYZE_AGAIN_FIRST
from utils.prompts import PROMPT_ANALYZE_AND_FIND_NEXT_KERNEL_BETTER, PROMPT_ANALYZE_AND_FIND_NEXT_KERNEL_CP, PROMPT_ANALYZE_AGAIN, MODEL_PRINTOUT_EXPLANATION, EXTRA_EXPLANATION, EXTRA_EXPLANATION_CP
from utils.prompts import INIT_CODE, INIT_CODE_XY_LOAD, INIT_CODE_M3, INIT_CODE_MGP, INIT_CODE_POST, INIT_CODE_with_pred, INIT_CODE_M3_with_pred, INIT_CODE_POST_with_pred, INIT_CODE_XY_LOAD_with_pred
from utils.main_utils import llm_inference
from utils.prompts_utils import add_in_context_learning_prompts

from cfgs.asmd_cfg import parse_asmd_cfg as parse_kernel
from plot_func import plot as plot_kernel

from GPy_ABCD import model_printout



################ CODE PARSER ################
def is_valid_python_code(code_block):
    try:
        # Try to compile the code block to check if it's valid Python
        compile(code_block, "<string>", "exec")
        return True
    except SyntaxError:
        return False

def insert_init_code(data_name, code, X=None, Y=None):
    if X is not None and Y is not None: ##### THE CASE OF GETTING THE X, Y from analyzer (ex.noise)
        data = {'X': X, 'Y': Y}
        with open('data_save.pkl', 'wb') as f: pickle.dump(data, f, pickle.HIGHEST_PROTOCOL) ####
        init_code = INIT_CODE_XY_LOAD
        
    else:
        if data_name in ['dugongs_data', 'arK', 'arma', 'eight_schools', 'surgical_data', 'pilots', 'rats_data']:
            init_code = INIT_CODE_POST%(data_name)
        elif data_name in ['exchange_rate']:
            init_code = INIT_CODE_MGP%(data_name, data_name)
        elif data_name[0].isdigit(): init_code = INIT_CODE%(data_name)
        else: init_code = INIT_CODE_M3%(data_name)
        
    filtered_lines = []
    # import pdb; pdb.set_trace()
    if "X, y = data['X'], data['Y']" not in code:
        for ith_line, line in enumerate(code.splitlines()):
            if ith_line == 0:
                filtered_lines.append(init_code)
                filtered_lines.append(line)
            else:
                filtered_lines.append(line)
        
    else:
        for line in code.splitlines():
            if "X, y = data['X'], data['Y']" not in line:
                filtered_lines.append(line)
            else:
                filtered_lines.append(init_code)
    
    code = "\n".join(filtered_lines)
    return code

def insert_init_code_v2(data_name, code, X=None, Y=None):
    if X is not None and Y is not None: ##### THE CASE OF GETTING THE X, Y from analyzer (ex.noise)
        data = {'X': X, 'Y': Y}
        with open('data_save.pkl', 'wb') as f: pickle.dump(data, f, pickle.HIGHEST_PROTOCOL) ####
        init_code = INIT_CODE_XY_LOAD
    else:
        if data_name in ['dugongs_data', 'arK', 'arma', 'eight_schools', 'surgical_data', 'pilots', 'rats_data']: init_code = INIT_CODE_POST%(data_name)
        elif data_name in ['exchange_rate']: init_code = INIT_CODE_MGP%(data_name, data_name)
        elif data_name[0].isdigit(): init_code = INIT_CODE%(data_name)
        else: init_code = INIT_CODE_M3%(data_name)
    
    filtered_lines = []
    for ith_line, line in enumerate(code.splitlines()):
        if ith_line == 0:
            filtered_lines.append(init_code)
            filtered_lines.append(line)
        else:
            filtered_lines.append(line)
    
    code = "\n".join(filtered_lines)
    return code

def insert_init_code_with_pred(data_name, code, X=None, Y=None):
    if X is not None and Y is not None: ##### THE CASE OF GETTING THE X, Y from analyzer (ex.noise)
        data = {'X': X, 'Y': Y}
        with open('data_save.pkl', 'wb') as f: pickle.dump(data, f, pickle.HIGHEST_PROTOCOL) ####
        init_code = INIT_CODE_XY_LOAD_with_pred
    else:
        if data_name in ['dugongs_data', 'arK', 'arma', 'eight_schools', 'surgical_data', 'pilots', 'rats_data']: init_code = INIT_CODE_POST_with_pred%(data_name)
        elif data_name in ['exchange_rate']: init_code = INIT_CODE_MGP%(data_name, data_name)
        elif data_name[0].isdigit(): init_code = INIT_CODE_with_pred%(data_name)
        else: init_code = INIT_CODE_M3_with_pred%(data_name)
    
    filtered_lines = []
    for line in code.splitlines():
        if "access_data" in line: filtered_lines.append(init_code)
        else: filtered_lines.append(line)
    
    code = "\n".join(filtered_lines)
    return code

def insert_init_code_with_pred_v2(data_name, code, X=None, Y=None):
    if X is not None and Y is not None: ##### THE CASE OF GETTING THE X, Y from analyzer (ex.noise)
        data = {'X': X, 'Y': Y}
        with open('data_save.pkl', 'wb') as f: pickle.dump(data, f, pickle.HIGHEST_PROTOCOL) ####
        init_code = INIT_CODE_XY_LOAD_with_pred
    else:
        if data_name in ['dugongs_data', 'arK', 'arma', 'eight_schools', 'surgical_data', 'pilots', 'rats_data']: init_code = INIT_CODE_POST_with_pred%(data_name)
        elif data_name in ['exchange_rate']: init_code = INIT_CODE_MGP%(data_name, data_name)
        elif data_name[0].isdigit(): init_code = INIT_CODE_with_pred%(data_name)
        else: init_code = INIT_CODE_M3_with_pred%(data_name)
    
    filtered_lines = []
    for ith_line, line in enumerate(code.splitlines()):
        if ith_line == 0:
            filtered_lines.append(init_code)
            filtered_lines.append(line)
        if "model_printout" in line:
            newline = "model = model.model\nprint(model)"
            filtered_lines.append(newline)
        else:
            filtered_lines.append(line)
    
    code = "\n".join(filtered_lines)
    return code
    
def execute_code(code):
    # old_stdout = sys.stdout
    # redirected_output = sys.stdout = StringIO()
    # exec(code)
    # sys.stdout = old_stdout
    # printed_output = redirected_output.getvalue()
    EXEC_ENV = {"__builtins__": __builtins__}
    try:
        f = StringIO()
        with redirect_stdout(f): exec(code, EXEC_ENV, EXEC_ENV)
        printed_output = f.getvalue().strip()
        gen_images = parse_output_image_savefig(code)
    except Exception as e:
        printed_output = 'CODE ERROR OCCCURED DURING EXECUTION. GIVE ME ANOTHER CODE.'
        printed_output += '\nERROR: '+str(e)
        gen_images = []
    return printed_output, gen_images

def parse_python_blocks(text):
    # Define a regex pattern to match Python code blocks
    pattern = r'```python(.*?)```'
    
    # Find all code blocks
    code_blocks = re.findall(pattern, text, re.DOTALL)
    
    # Strip leading and trailing whitespace from each block
    code_blocks = [block.strip() for block in code_blocks if is_valid_python_code(block)]
    
    return '\n'.join(code_blocks)

def parse_output_image_savefig(code):
    lines = code.split('\n')
    gen_images = []
    for l in lines:
        if 'plt.savefig' in l:
            path = l[l.find('(')+1:l.rfind(')')]
            # if os.path.exists(path): #### parse한 Path가 존재할때만 집어넣기.
            #     gen_images.append(path)
            gen_images.append(path)
            # else: print('path does not exists.', path) ### 리부탈
    return gen_images

def parse_output_kernel(response):
    generated_kernels = response
    generated_kernels = generated_kernels[generated_kernels.rfind('next kernels:')+len('next kernels:'):]
    ## remove bracket
    generated_kernels = generated_kernels[generated_kernels.find('[')+1:]
    generated_kernels = generated_kernels[:generated_kernels.find(']')]
    generated_kernels = eval(generated_kernels)
    generated_kernels = generated_kernels.replace("'", '')
    generated_kernels = generated_kernels.replace('"', '')
    generated_kernels = [each.strip() for each in generated_kernels]
    return generated_kernels

def parse_output_kernel_v2(response):
    # return generated_kernels, params
    generated_kernels = response
    generated_kernels = generated_kernels[generated_kernels.rfind('next kernels:')+len('next kernels:'):]
    ## remove bracket
    generated_kernels = generated_kernels[generated_kernels.find('[')+1:]
    generated_kernels = generated_kernels[:generated_kernels.find(']')]
    # 작은 따옴표가 포함되어 있는지 여부 확인
    if "'" in generated_kernels or '"' in generated_kernels:
        generated_kernels = re.sub(r'\s+', ' ', generated_kernels).strip()
        try:
            generated_kernels = eval(generated_kernels)
        except:
            print('error!!!')
            print(generated_kernels)
            exit()
        # generated_kernels = generated_kernels.replace("'", '')
    else:
        output_string = re.sub(r'(\w+[\w\s\+\*\(\),]*)', r'\'\1\'', generated_kernels)
        generated_kernels = eval(output_string)
    generated_kernels = [each.replace("'", '') for each in generated_kernels]
    generated_kernels = [each.replace('"', '') for each in generated_kernels]
    generated_kernels = [each.strip() for each in generated_kernels]
    return generated_kernels

def parse_output_kernel_v2_2(response):
    # return generated_kernels, params
    generated_kernels = response
    generated_kernels = generated_kernels[generated_kernels.rfind('next kernels:')+len('next kernels:'):]
    generated_kernels = generated_kernels[generated_kernels.find('['):]
    generated_kernels = generated_kernels[:generated_kernels.find(']')+1]
    generated_kernels = eval(generated_kernels)
    generated_kernels = [each.strip() for each in generated_kernels]
    # print('gen kernels for v2_2:', generated_kernels)
    return generated_kernels

def parse_output_kernel_v3(response):
    generated_kernels = response
    generated_kernels = generated_kernels[generated_kernels.rfind(' is:')+len(' is:'):]
    ## remove bracket
    generated_kernels = generated_kernels[generated_kernels.find('['):]
    generated_kernels = generated_kernels[:generated_kernels.find(']')+1]
    generated_kernels = eval(generated_kernels) #### 이러면 파이썬 리스트로 뽑히겠지?
    generated_kernels = [each.strip() for each in generated_kernels]
    return generated_kernels

def encode_images(image_paths):
    if type(image_paths) == str:
        image_paths = [image_paths]
    base64Frames = []
    for image_path in image_paths:
        if image_path[0] == "'":
            image_path = image_path.strip("'")
        if not os.path.exists(image_path):
            print(f'{image_path} does not exists')
            continue
        with open(image_path, "rb") as image_file:
            base64Frames.append(base64.b64encode(image_file.read()).decode("utf-8"))
    
    return base64Frames
###########################################

############### KERNEL SAVE CODE ################
def access_data(model):
    with open('fit_model_save.pkl', 'wb') as f:
        pickle.dump(model[0], f, pickle.HIGHEST_PROTOCOL) ####


####################### INFERENCE CODES #######################
def analyze_then_generate_kernels_first(X, Y, data_name, round=0, model='gpt-4o', logger=None):
    repeat_count = 0; y = Y

    ############ INITIAL INFERENCE
    if data_name in ['exchange_rate']: ### multivariate
        global PROMPT_ANALYZE_AND_FIND_FIRST_KERNEL
        PROMPT_ANALYZE_AND_FIND_FIRST_KERNEL = re.sub('1d data', 'multivariate time-series data', PROMPT_ANALYZE_AND_FIND_FIRST_KERNEL)
        PROMPT_ANALYZE_AND_FIND_FIRST_KERNEL += "Please assume that X it NOT 1D array but multivariate, and its shape is (length, 7). For Y, its shape is (length, 1). Make sure that you know it, and don't make mistakes while visualizing X. Please draw figures with the appropriate shape, not so flatten or too small. Please analyze each dim precisely, and also try to find the relation between each dim."
        
    conversation_history = [
        {"role": "system", "content": SYSTEMPROMPT},
        {"role": "user", "content": PROMPT_ANALYZE_AND_FIND_FIRST_KERNEL + 'The output format must be ```python\n ``` or ```plaintext\n next kernels: ["combination1", "combination2", "combination3", "combination4", "combination5" ...]\n```'}
    ]
    conversation_history, response = llm_inference(conversation_history, append_response=True, model_name=model)
    logger.info(f'===========================ANALYZER LLM FOR ROUND 0==============================')
    logger.info(f'---------------------------REPEAT COUNT: {repeat_count}---------------------------')
    
    logger.info(f'-----LLM ANALYSIS & CODE GENERATION----')
    for ch in conversation_history:
        logger.info(f"{ch['role'].upper()}:\n{ch['content']}")
        logger.info('-----------------------------------------')
    logger.info('-----LLM ANALYSIS & CODE GENERATION DONE-----')
    
    if '```python' in response: #### later change detection algorithm    
        code = parse_python_blocks(response)
        code = insert_init_code(data_name, code, X, Y)
        logger.info(f'******CODE INSERTED******')
        logger.info(code)
        logger.info(f'******CODE INSERTED DONE******')
        printed_output, gen_images = execute_code(code)
        printed_output = printed_output.strip()

        logger.info(f'-----CODE EXECUTION-----')
        logger.info(printed_output)
        logger.info(f'-----CODE EXECUTION DONE-----')
    else:
        try:
            # generated_kernels = parse_output_kernel_v2(response)
            generated_kernels = parse_output_kernel(response)
            logger.info(f'-----KERNEL GENERATION-----')
            logger.info(generated_kernels)
            logger.info(f'-----KERNEL GENERATION DONE-----')
            logger.info(f'=========ANALYZER LLM FOR ROUND 0 DONE============')
            return generated_kernels, None, conversation_history
        except:
            printed_output = "Please continue saying what you were doing. Please generate the code or suggest kernels."
        return generated_kernels, '', conversation_history

    ############ INFERENCE AT N>1      
    while repeat_count<15:
        repeat_count += 1
        logger.info(f'---------------------------REPEAT COUNT: {repeat_count}---------------------------')
        ##### THIS CASE PLEASE DEAL WITH ERROR
        if "CODE ERROR OCCCURED DURING EXECUTION. GIVE ME ANOTHER CODE." in printed_output:
            content = printed_output
        
        elif "Please continue saying what you were doing." in printed_output:
            content = printed_output
            
        else:
            if data_name in ['exchange_rate'] and repeat_count==1: ### multivariate
                global PROMPT_ANALYZE_AGAIN_FIRST
                PROMPT_ANALYZE_AGAIN_FIRST += "Please analyze as many possible, and request additional analysis, do not just end by recommending kernels. Analyze each dim precisely with multiple analysis. Do not recommend kernel yet. Try to make the best analysis of the data as many as possible, by trying your best."
        
            base64Frames = encode_images(gen_images)
            imgnames = [each.split('/')[-1].strip("'") for each in gen_images]
            image_analyze = f"\nThe requested code's execution's has made the image {imgnames} which looks like this. Please analyze this image too."
            if len(base64Frames)>0:
                if 'gpt' in model:
                    content = [
                        PROMPT_ANALYZE_AGAIN_FIRST%(printed_output[:1000], image_analyze),
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "high"}}, base64Frames)
                    ]
                elif 'qwen' in model:
                    content = []
                    for gen_image_path in gen_images:
                        if gen_image_path[0]=="'":
                            gen_image_path = gen_image_path.strip("'")
                        content.append({"type": "image", "image": gen_image_path})
                    content.append({"type": "text", "text": PROMPT_ANALYZE_AGAIN_FIRST%(printed_output[:1000], image_analyze)})
            else:
                if 'gpt' in model:
                    content = PROMPT_ANALYZE_AGAIN_FIRST%(printed_output[:1000], '')
                elif 'qwen' in model:
                    content = [{"type": "text", "text": PROMPT_ANALYZE_AGAIN_FIRST%(printed_output[:1000], '')}]

        conversation_history.append({"role": "user", "content": content})
        conversation_history, response = llm_inference(conversation_history, append_response=True, model_name=model)
        logger.info(f'-----LLM ANALYSIS & CODE GENERATION----')
        logger.info(f"USER: \n{conversation_history[-2]['content'][0] if type(conversation_history[-2]['content'])==list else conversation_history[-2]['content']}")
        logger.info(f"LLM RESPONSE: {conversation_history[-1]['content']}")
        logger.info('-----LLM ANALYSIS & CODE GENERATION DONE-----')


        if '```plaintext' in response:
            generated_kernels = parse_output_kernel_v2_2(response)
            logger.info(f'-----KERNEL GENERATION-----')
            logger.info(generated_kernels)
            logger.info(f'-----KERNEL GENERATION DONE-----')
            logger.info(f'=========ANALYZER LLM FOR ROUND 0 DONE============')
            return generated_kernels, None, conversation_history
        
        elif '```python' in response and ('import ' in response or 'print(' in response or "data['X']" in response):
            code = parse_python_blocks(response)
            code = insert_init_code(data_name, code, X, Y)# if "data['X']" in code else insert_init_code_v2(data_name, code)
            logger.info(f'******CODE INSERTED******')
            logger.info(code)
            logger.info(f'******CODE INSERTED DONE******')
            printed_output, gen_images = execute_code(code)
            logger.info(f'-----CODE EXECUTION-----')
            logger.info(printed_output)
            logger.info(f'-----CODE EXECUTION DONE-----')
        else:
            try:
                # generated_kernels = parse_output_kernel_v2(response)
                generated_kernels = parse_output_kernel_v2_2(response)
                logger.info(f'-----KERNEL GENERATION-----')
                logger.info(generated_kernels)
                logger.info(f'-----KERNEL GENERATION DONE-----')
                logger.info(f'=========ANALYZER LLM FOR ROUND 0 DONE============')
                return generated_kernels, None, conversation_history
            except:
                printed_output = "Please continue saying what you were doing. Please generate the code or suggest kernels."

def analyze_then_generate_kernels(old_kernels_model, old_kernels, X, Y, data_name, round, feedback=None, use_changepoint=False, model='gpt-4o', logger=None):
    repeat_count = 0
    models = []

    for each in old_kernels:
        models.append(parse_kernel(each))
    models = models[0]
    access_data(old_kernels_model) ### only first is considered now.
    
    range_ = np.max(Y)-np.min(Y)
    plot_kernel(old_kernels_model[0].model, ymin=np.min(Y)-range_*0.3, ymax=np.max(Y)+range_*0.3)
    path = '/home/mok/module/icml25/ASMD/ztmpimgs/tmp.png'
    plt.savefig(path); plt.close();
    encoded_images = encode_images(path)
    
    ############ INITIAL INFERENCE
    content = []
    if feedback is not None:
        if 'gpt' in model:
            content.append(f"""This is the feedback of previous kernels. Please consider this feedback for the next kernel recommendation.{feedback}\n""")
        elif 'qwen' in model:
            content.append({"type": "text", "text": f"""This is the feedback of previous kernels. Please consider this feedback for the next kernel recommendation.{feedback}\n"""})
    
    if not use_changepoint:
        if 'gpt' in model:
            content.extend([
                PROMPT_ANALYZE_AND_FIND_NEXT_KERNEL_BETTER%(old_kernels) + 'The output format must be ```python\n ``` or ```plaintext\n next kernels: ["combination1", "combination2", "combination3", "combination4", "combination5" ...]\n```',  
                *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, encoded_images)
            ])
        elif 'qwen' in model:
            content.append({"type": "image", "image": path})
            content.append({"type": "text", "text": PROMPT_ANALYZE_AND_FIND_NEXT_KERNEL_BETTER%(old_kernels) + 'The output format must be ```python\n ``` or ```plaintext\n next kernels: ["combination1", "combination2", "combination3", "combination4", "combination5" ...]\n```'})
        
    else:
        if 'gpt' in model:
            content.extend([
                PROMPT_ANALYZE_AND_FIND_NEXT_KERNEL_CP%(old_kernels),  
                *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, encoded_images)]
            )
        elif 'qwen' in model:
            content.append({"type": "image", "image": path})
            content.append({"type": "text", "text": PROMPT_ANALYZE_AND_FIND_NEXT_KERNEL_CP%(old_kernels)})
    
    logger.info(f'===========================ANALYZER LLM FOR ROUND {round}==============================')
    conversation_history = [{"role": "system", "content": SYSTEMPROMPT}]
    conversation_history.append({"role": "user", "content": content})
    conversation_history, response = llm_inference(conversation_history, append_response=True, model_name=model)
    logger.info(f'---------------------------REPEAT COUNT: {repeat_count}---------------------------')
    logger.info(f'-----LLM ANALYSIS & CODE GENERATION----')
    for ch in conversation_history:
        if type(ch['content'])==str:
            logger.info(f"{ch['role'].upper()}:\n{ch['content']}")
            logger.info('-----------------------------------------')
    logger.info('-----LLM ANALYSIS & CODE GENERATION DONE-----')
    
    if '```python' in response: #### later change detection algorithm
        code = parse_python_blocks(response)        
        code = insert_init_code_with_pred(data_name, code, X, Y)
        logger.info(f'******CODE INSERTED******')
        logger.info(code)
        logger.info(f'******CODE INSERTED DONE******')
        printed_output, gen_images = execute_code(code)
        logger.info(f'-----CODE EXECUTION-----')
        logger.info(printed_output)
        logger.info(f'-----CODE EXECUTION DONE-----')
    else:
        generated_kernels = parse_output_kernel_v3(response)
        logger.info(f'-----KERNEL GENERATION-----')
        logger.info(generated_kernels)
        logger.info(f'-----KERNEL GENERATION DONE-----')
        logger.info(f'===========================ANALYZER LLM FOR ROUND {round} DONE==============================')
        return generated_kernels, conversation_history
    
    ############ INFERENCE AT N>1   
    while repeat_count<15:
        repeat_count += 1
        logger.info(f'---------------------------REPEAT COUNT: {repeat_count}---------------------------')
        if repeat_count>5: ### originally 10
            conversation_history = conversation_history[-3:] ### originally 5
            
        ##### THIS CASE PLEASE DEAL WITH ERROR
        if "CODE ERROR OCCCURED DURING EXECUTION. GIVE ME ANOTHER CODE." in printed_output:
            content = printed_output
        
        elif "Please keep saying what you were doing." in printed_output:
            content = printed_output
        
        else:
            encoded_images = encode_images(gen_images)
            imgnames = [each.split('/')[-1].strip("'") for each in gen_images]
            brief_summary = ''
            if len(imgnames)>0:
                brief_summary += f"The requested code's execution contains generating a graph image."
            # if 'model_printout' in code: brief_summary += MODEL_PRINTOUT_EXPLANATION
            
            EXTRA_EXPLANATION_prompt = EXTRA_EXPLANATION if not use_changepoint else EXTRA_EXPLANATION_CP
            if len(encoded_images)>0:
                if 'gpt' in model:
                    content = [
                        PROMPT_ANALYZE_AGAIN%(printed_output[:1000], brief_summary)+f"\nThe analysis is done about this kernels: {old_kernels[0]}."+f"\nThe requested code's execution's saved image {imgnames} looks like this. Please analyze the given graph, and tell me what I should do next, among continue analyzing and recommending me kernel. DO NOT REPEAT what you have already done.\n"+EXTRA_EXPLANATION_prompt,
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, encoded_images)
                    ]
                elif 'qwen' in model:
                    content = []
                    for gen_image_path in gen_images:
                        if gen_image_path[0]=="'":
                            gen_image_path = gen_image_path.strip("'")
                        content.append({"type": "image", "image": gen_image_path})
                    text = PROMPT_ANALYZE_AGAIN%(printed_output[:1000], brief_summary)+f"\nThe analysis is done about this kernels: {old_kernels[0]}."+f"\nThe requested code's execution's saved image {imgnames} looks like this. Please analyze the given graph, and tell me what I should do next, among continue analyzing and recommending me kernel. DO NOT REPEAT what you have already done.\n"+EXTRA_EXPLANATION_prompt
                    content.append({"type": "text", "text": text})
            else:
                if 'gpt' in model:
                    content = PROMPT_ANALYZE_AGAIN%(printed_output[:1000], brief_summary)+f"\nThe analysis is done about this kernels: {old_kernels[0]}."
                elif 'qwen' in model:
                    content = [{"type": "text", "text": PROMPT_ANALYZE_AGAIN%(printed_output[:1000], brief_summary)+f"\nThe analysis is done about this kernels: {old_kernels[0]}."}]
        
        conversation_history.append({"role": "user", "content": content})
        conversation_history, response = llm_inference(conversation_history, append_response=True, model_name=model)
        print(f'=========LLM response for round {round}, repeat count {repeat_count}=========')
        logger.info(f'-----LLM ANALYSIS & CODE GENERATION----')
        logger.info(f"USER: \n{conversation_history[-2]['content'][0] if type(conversation_history[-2]['content'])==list else conversation_history[-2]['content']}")
        logger.info(f"LLM RESPONSE: {conversation_history[-1]['content']}")
        logger.info('-----LLM ANALYSIS & CODE GENERATION DONE-----')
        
        if '```plaintext' in response:
            generated_kernels = parse_output_kernel_v2_2(response)
            logger.info(f'-----KERNEL GENERATION-----')
            logger.info(generated_kernels)
            logger.info(f'-----KERNEL GENERATION DONE-----')
            logger.info(f'===========================ANALYZER LLM FOR ROUND {round} DONE==============================')
            return generated_kernels, conversation_history
        
        elif '```python' in response: 
            code = parse_python_blocks(response) ##### CHECK WHETHER IT IS PYTHON CODE OR KERNEL
            
            if is_valid_python_code(code):
                if 'access_data' in code:
                    access_data(old_kernels_model)                
                    code = insert_init_code_with_pred(data_name, code, X, Y)
                else:
                    access_data(old_kernels_model)
                    code = insert_init_code_with_pred_v2(data_name, code, X, Y)
                
                logger.info(f'******CODE INSERTED******')
                logger.info(code)
                logger.info(f'******CODE INSERTED DONE******')
                printed_output, gen_images = execute_code(code)
                logger.info(f'-----CODE EXECUTION-----')
                logger.info(printed_output)
                logger.info(f'-----CODE EXECUTION DONE-----')
                
            else:
                # generated_kernels = parse_output_kernel_v2(response)
                try:
                    # generated_kernels = parse_output_kernel_v2(response)
                    generated_kernels = parse_output_kernel_v2_2(response)
                    logger.info(f'-----KERNEL GENERATION-----')
                    logger.info(generated_kernels)
                    logger.info(f'-----KERNEL GENERATION DONE-----')
                    logger.info(f'=========ANALYZER LLM FOR ROUND {round} DONE============')
                    return generated_kernels, None, conversation_history
                except:
                    printed_output = "Please continue saying what you were doing. Please generate the code or suggest kernels."
            
        else:
            # generated_kernels = parse_output_kernel_v2(response)
            try:
                generated_kernels = parse_output_kernel_v2_2(response)
                logger.info(f'===========================ANALYZER LLM FOR ROUND {round} DONE==============================')
                return generated_kernels, conversation_history
            except:
                printed_output = "Please continue saying what you were doing. Please generate the code or suggest kernels."
        
        logger.info(f'---------------------------REPEAT COUNT: {repeat_count} DONE---------------------------')
    logger.info(f'===========================ANALYZER LLM FOR ROUND {round} DONE==============================')
        
    return [], conversation_history


########## LLM ONLY MODEL FOR GENERATION ############
def no_analyze_then_generate_kernels_first(X, Y, data_name, round=0, model='gpt-4o', logger=None):
    plt.plot(X, Y)
    plt.savefig(f'./tmp.png')
    
    base64Frames = []
    base64Frames.append(encode_images('./tmp.png')) ##### INITIAL DATA AS PLOT

    conversation_history = [
        {"role": "system", "content": SYSTEMPROMPT}
    ] #### 이렇게 리스트로 넣어도되지만 그냥 이전 히스토리를 프롬프트에 콘캣하는구나
    
    content = [f"""
Given the dataset {data_name}, please provide the kernel compositions for the automated gaussian process kernel discovery.
Please generate the model for the dataset {data_name}. 
Please follow the instructions below to generate the new kernel compositions. Think step by step.
Please reflect the properties of the dataset or plot of the data first. Then sketch a high level modeling approach, and state the hypotheses that it will address before writing a program, and add comments to code that address specific hypotheses.
"""]
    # content = [PROMPT_ANALYZE_AND_FIND_FIRST_KERNEL]
    content = add_in_context_learning_prompts(content)
    content.append("Data visualization:")
    content.append(*map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "high"}}, base64Frames))
    content.append("Maybe you can consider the following kernels: LIN, PER, SE, C, WN")
    content.append("next kernels:")

    #### FIRST ITERATION SHOULD BE DONE LIKE THIS
    conversation_history.append({"role": "user", "content": content})
    
    conversation_history, response = llm_inference(conversation_history, append_response=True, model_name=model)
        
    print('-----------------')
    print(response)
    print('-----------------')

    generated_kernels = response
    generated_kernels = generated_kernels[generated_kernels.rfind('next kernels:')+len('next kernels:'):]
    generated_kernels = generated_kernels[generated_kernels.find('['):]
    generated_kernels = generated_kernels[:generated_kernels.find(']')+1]
    generated_kernels = eval(generated_kernels)
    generated_kernels = [each.strip() for each in generated_kernels]
    
    return generated_kernels, "", conversation_history


########## LLM ONLY MODEL FOR GENERATION ############
def no_analyze_then_generate_kernels(old_kernels_model, old_kernels, X, Y, data_name, round, feedback=None, use_changepoint=False, model='gpt-4o', logger=None):
    plt.plot(X, Y)
    plt.savefig(f'./tmp.png')
    
    base64Frames = []
    base64Frames.append(encode_images('./tmp.png')) ##### INITIAL DATA AS PLOT

    conversation_history = [
        {"role": "system", "content": SYSTEMPROMPT}
    ] #### 이렇게 리스트로 넣어도되지만 그냥 이전 히스토리를 프롬프트에 콘캣하는구나
    
    content = [f"""
Given the dataset {data_name}, please provide the kernel compositions for the automated gaussian process kernel discovery.
Please generate the model for the dataset {data_name}. The current kernel is {old_kernels}.
Please follow the instructions below to generate the new kernel compositions. Think step by step.
Please reflect the properties of the dataset or plot of the data first. Then sketch a high level modeling approach, and state the hypotheses that it will address before writing a program, and add comments to code that address specific hypotheses.
"""]
            
    # content = [PROMPT_ANALYZE_AND_FIND_FIRST_KERNEL]
    content = add_in_context_learning_prompts(content)
    content.append("Data visualization:")
    content.append(*map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "high"}}, base64Frames))
    content.append("Maybe you can consider the following kernels: LIN, PER, SE, C, WN")
    content.append("next kernels:")

    #### FIRST ITERATION SHOULD BE DONE LIKE THIS
    conversation_history.append({"role": "user", "content": content})
    
    conversation_history, response = llm_inference(conversation_history, append_response=True, model_name=model)
        
    print('-----------------')
    print(response)
    print('-----------------')

    generated_kernels = response
    generated_kernels = generated_kernels[generated_kernels.rfind('next kernels:')+len('next kernels:'):]
    generated_kernels = generated_kernels[generated_kernels.find('['):]
    generated_kernels = generated_kernels[:generated_kernels.find(']')+1]
    generated_kernels = eval(generated_kernels)
    generated_kernels = [each.strip() for each in generated_kernels]
    
    return generated_kernels, "", conversation_history