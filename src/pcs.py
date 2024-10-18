import gc
import torch
import numpy as np
from scipy.spatial.distance import cosine
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from utils import get_model_save_path

def release():
    torch.cuda.empty_cache()
    gc.collect()
    
def load_params(model_tag):
    model_path = get_model_save_path(model_tag)
    model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True)
    model = model.eval()
    params = torch.cat([model.state_dict()[key].view(-1) for key in model.state_dict().keys() if 'lm_head' not in key and 'embed' not in key]) 

    return params

def cal_cos(source_params, target_params, align_strategy='truncation'):
    len1, len2 = source_params.size()[0], target_params.size()[0]
    if len1 != len2:
        print(align_strategy)
        if align_strategy == 'random_sampling':
            if len1 > len2:
                indices = torch.randperm(len1)[:len2]
                source_params = source_params[indices]
            elif len2 > len1:
                indices = torch.randperm(len2)[:len1]
                target_params = target_params[indices]
        elif align_strategy == 'truncation':
            if len1 > len2:
                source_params = source_params[:len2]
            elif len2 > len1:
                target_params = target_params[:len1]
        elif align_strategy == 'padding':
            if len1 > len2:
                padding = torch.zeros(len1 - len2)
                target_params = torch.cat([target_params, padding], dim=0)
            elif len2 > len1:
                padding = torch.zeros(len2 - len1)
                source_params = torch.cat([source_params, padding], dim=0)

    
    source_params = source_params.detach().numpy()
    target_params = target_params.detach().numpy()
    cosine_similarity = 1 - cosine(source_params, target_params)

    return cosine_similarity

    

source_params = load_params('llama-2-7b')
tmodel_tags = [
    'llama-2-7b-chat', 'vicuna-7b-v1.5', 'tulu-2-7b','vicuna-backdoored-7b', 'llama-2-coder-7b',
    'chinese-llama-2-7b', 'Sheared-LLaMA-2.7B-ShareGPT', 'wizardmath-7b','llemma-7b','codellama-7b',
    'llama-3-8b','mistral-7b',
    'baichuan-2-7b','internlm2-7b', 'qwen-7b-v1.5',  
    'Sheared-LLaMA-2.7B-ShareGPT','Sheared-LLaMA-2.7B-Pruned','Sheared-LLaMA-2.7B',
    'Sheared-LLaMA-1.3B-ShareGPT','Sheared-LLaMA-1.3B-Pruned','Sheared-LLaMA-1.3B',
    'vicuna-7b-v1.5', 'llama-2-finance-7b'
    # 'shisa-gamma-7b', 'wizardmath-7b-1.1', 'abel-7b-002'
    # 'openllama-2-7b', 'mpt-7b'
    'wandallama-2-7b', 'gblmllama-2-7b', 'sparsellama-2-7b',
]
align_strategys = ['truncation']
for align_strategy in align_strategys:
    for tmodel_tag in tmodel_tags:
        target_params = load_params(tmodel_tag)
        cos_sim = cal_cos(source_params, target_params, align_strategy)
        
        print(f'PCS of llama-2-7b and {tmodel_tag} is: {cos_sim}')
        release()