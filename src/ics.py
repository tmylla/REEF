
import json
import gc
import torch
from tqdm import tqdm
from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F
from utils import get_model_save_path
from scipy.spatial.distance import cosine

torch.manual_seed(42)

with open('sorted_tokens.json', 'r', encoding='utf-8') as f:
    sorted_token_dict = json.load(f)

def attack(X_hat, W_q, W_k, W_v, W_o, W_up, W_down, attack_type):
    print('='*10 + f'{attack_type}-attacking' + '='*10)

    if attack == 'scaling':
        W_q, W_k, W_v, W_o, W_up, W_down = 0.8*W_q, 0.8*W_k, 0.8*W_v, 0.8*W_o, 0.8*W_up, 0.8*W_down
    
    if attack_type == 'att':
        n = W_q.shape[-1]
        C1 = torch.eye(n).cuda()
        perturbation = torch.normal(mean=0.0, std=0.01, size=(n, n)).cuda()
        C1 += perturbation
        C1_inv = torch.inverse(C1).cuda()
        
        W_q = W_q @ C1
        W_k = W_k @ C1_inv
        W_v = W_v @ C1
        W_o = W_o @ C1_inv
        
    if attack_type == 'fnn':
        indices = torch.randperm(W_up.shape[-1])
        P_ffn = torch.eye(W_up.shape[-1])[indices].cuda()
        P_ffn_inv = torch.inverse(P_ffn).cuda()

        W_up = W_up @ P_ffn
        W_down = P_ffn_inv @ W_down
        
    if attack_type == 'embed':
        indices = torch.randperm(W_q.shape[-1])
        P_E = torch.eye(W_q.shape[-1])[indices].cuda()
        P_E_inv = torch.inverse(P_E).cuda()

        W_q = P_E_inv @ W_q
        W_k = P_E_inv @ W_k
        W_v = P_E_inv @ W_v
        W_o = W_o @ P_E
        W_up = P_E_inv @ W_up
        W_down = W_down @ P_E

    if attack_type == 'mix':
        n, m = W_q.shape[-1], W_up.shape[-1]
        
        C1 = torch.eye(n).cuda()
        perturbation = torch.normal(mean=0.0, std=0.01, size=(n, n)).cuda()
        C1 += perturbation.cuda()
        C1_inv = torch.inverse(C1).cuda()

        P_ffn = torch.eye(m)[torch.randperm(m)].cuda()
        P_ffn_inv = torch.inverse(P_ffn).cuda()
        
        P_E = torch.eye(n)[torch.randperm(n)].cuda()
        P_E_inv = torch.inverse(P_E).cuda()

        W_q = P_E_inv @ W_q @ C1
        W_k = P_E_inv @ W_k @ C1_inv
        W_v = P_E_inv @ W_v @ C1
        W_o = C1_inv @ W_o @ P_E
        W_up = P_E_inv @ W_up @ P_ffn
        W_down = P_ffn_inv @ W_down @ P_E
        X_hat = X_hat @ P_E
        

    return X_hat, W_q, W_k, W_v, W_o, W_up, W_down

def invariant_terms(model_tag, layers, do_attack=False, attack_type='att'):
    model_path = get_model_save_path(model_tag)
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True)
    model = model.eval()

    params = model.state_dict()
    if 'internlm2' in model_tag:
        vocab_embedding = params['model.tok_embeddings.weight']
    elif model_tag=='mpt-7b':
        vocab_embedding = params['transformer.wte.weight']
    else:
        vocab_embedding = params['model.embed_tokens.weight'] 
        

    indices = []
    for key in tqdm(sorted_token_dict.keys()):
        try:
            indices.append(tokenizer.get_vocab()[key])
        except:
            indices.append(tokenizer.get_vocab()['unk'])

    indices_tensor = torch.tensor(indices)
    token_embeddings = vocab_embedding[indices_tensor].cuda()

    invariant_terms = []
    for layer in layers:
        if model_tag == 'baichuan-2-7b':
            W_pack = params[f'model.layers.{layer}.self_attn.W_pack.weight'].cuda()
            c_size = 4096
            W_q = W_pack[:c_size, :]
            W_k = W_pack[c_size:2*c_size, :]
            W_v = W_pack[2*c_size:, :]
            
            W_o = params[f'model.layers.{layer}.self_attn.o_proj.weight'].cuda()
            W_up = params[f'model.layers.{layer}.mlp.up_proj.weight'].cuda()
            W_down = params[f'model.layers.{layer}.mlp.down_proj.weight'].cuda()
        elif 'internlm2' in model_tag:
            kqv = params[f'model.layers.{layer}.attention.wqkv.weight'].cuda()
            num_key_value_groups = 4
            head_dim = 128
            
            v = rearrange(
                kqv,
                '(h gs d) dim -> h gs d dim',
                gs=2 + num_key_value_groups,
                d=head_dim,
            )
            wq, wk, wv = torch.split(v, [num_key_value_groups, 1, 1], dim=1)
            wq = rearrange(wq, 'h gs d dim -> (h gs d) dim')
            wk = rearrange(wk, 'h gs d dim -> (h gs d) dim')
            wv = rearrange(wv, 'h gs d dim -> (h gs d) dim')

            W_q = wq
            W_k = torch.repeat_interleave(wk, dim=0, repeats=4)
            W_v = torch.repeat_interleave(wv, dim=0, repeats=4)

            W_o = params[f'model.layers.{layer}.attention.wo.weight'].cuda()
            W_up = params[f'model.layers.{layer}.feed_forward.w3.weight'].cuda()
            W_down = params[f'model.layers.{layer}.feed_forward.w2.weight'].cuda()
        elif model_tag == 'mpt-7b':
            W_qkv = params[f'transformer.blocks.{layer}.attn.Wqkv.weight'].cuda()
            W_q = W_qkv[:4096].cuda()
            W_k = W_qkv[4096:8192].cuda()
            W_v = W_qkv[8192:].cuda()

            W_o = params[f'transformer.blocks.{layer}.attn.out_proj.weight'].cuda()
            W_up = params[f'transformer.blocks.{layer}.ffn.up_proj.weight'].cuda()
            W_down = params[f'transformer.blocks.{layer}.ffn.down_proj.weight'].cuda()
            
        else:
            W_q = params[f'model.layers.{layer}.self_attn.q_proj.weight'].cuda()
            W_k = params[f'model.layers.{layer}.self_attn.k_proj.weight'].cuda()
            W_v = params[f'model.layers.{layer}.self_attn.v_proj.weight'].cuda()
            if model_tag in ['llama-3-8b','mistral-7b','llama-3-8b-it','evollm-jp-7b','shisa-gamma-7b','wizardmath-7b-1.1','abel-7b-002']:
                W_k = torch.repeat_interleave(W_k, dim=0, repeats=4).cuda()
                W_v = torch.repeat_interleave(W_v, dim=0, repeats=4).cuda()
                
            W_o = params[f'model.layers.{layer}.self_attn.o_proj.weight'].cuda()
            W_up = params[f'model.layers.{layer}.mlp.up_proj.weight'].cuda()
            W_down = params[f'model.layers.{layer}.mlp.down_proj.weight'].cuda()

        X_hat = token_embeddings 
        W_q, W_k, W_v, W_o, W_up, W_down = W_q.T, W_k.T, W_v.T, W_o.T, W_up.T, W_down.T

        
        if do_attack:
            if attack_type == 'shuffle':
                X_hat = token_embeddings[torch.randperm(token_embeddings.shape[0])] 
            else:
                X_hat, W_q, W_k, W_v, W_o, W_up, W_down = attack(X_hat, W_q, W_k, W_v, W_o, W_up, W_down, attack_type) 


        M_a = X_hat @ W_q @ W_k.T @ X_hat.T
        M_b = X_hat @ W_v @ W_o @ X_hat.T
        M_f = X_hat @ W_up @ W_down @ X_hat.T

        invariant_terms.append(M_a.view(-1))
        invariant_terms.append(M_b.view(-1))
        invariant_terms.append(M_f.view(-1))

    invariant_terms = torch.cat(invariant_terms)
    return invariant_terms


def cosine_similarity(tensor1, tensor2):
    cosine_similarity = 1 - cosine(tensor1.cpu().numpy(), tensor2.cpu().numpy())
    return cosine_similarity


def release():
    torch.cuda.empty_cache()
    gc.collect()


layers = [30, 31]
smodel_tag = 'llama-2-7b'
s_vector = invariant_terms(smodel_tag, layers)

tmodel_tags = [
    'llama-2-7b-chat', 'vicuna-7b-v1.5', 'tulu-2-7b', 'llama-2-coder-7b', 'llama-2-finance-7b',
    'chinese-llama-2-7b', 'Sheared-LLaMA-2.7B-ShareGPT', 'wizardmath-7b','llemma-7b','codellama-7b',
    'llama-3-8b','mistral-7b', 'baichuan-2-7b','internlm2-7b', 'qwen-7b-v1.5',  
    'Sheared-LLaMA-2.7B-ShareGPT','Sheared-LLaMA-2.7B-Pruned','Sheared-LLaMA-2.7B',
    'Sheared-LLaMA-1.3B-ShareGPT','Sheared-LLaMA-1.3B-Pruned','Sheared-LLaMA-1.3B',
    # 'shisa-gamma-7b', 'wizardmath-7b-1.1', 'abel-7b-002'
    # 'openllama-2-7b', 'mpt-7b'
    'wandallama-2-7b', 'gblmllama-2-7b', 'sparsellama-2-7b',
]

for tmodel_tag in tmodel_tags:
    if '1.3B' in tmodel_tag:
        layers = [22,23]
    t_vector = invariant_terms(tmodel_tag, layers)

    assert s_vector.size()[0] == t_vector.size()[0]
    cos_simi = cosine_similarity(s_vector, t_vector)
    print(f'ICS of {smodel_tag} and {tmodel_tag} is: {cos_simi}')

    release()