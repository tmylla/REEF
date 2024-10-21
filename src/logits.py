
import torch
import os
import numpy as np
from tqdm import tqdm
from glob import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_victim_logits(smodel_tag):
    W = torch.load(f'xxx/your_path/model/{smodel_tag}-logits.pt').to(device)
    return W


def load_suspect_logits(tmodel_tag, dataset_name):
    directory = f'activations-lmhead/{dataset_name}'
    activation_files = glob(os.path.join(directory, f'{tmodel_tag}_*.pt'))
    acts = [
            torch.load(os.path.join(directory, f'{tmodel_tag}_{i}.pt'), 
                    map_location=lambda storage, loc: storage.cuda(device))
            for i in range(0, 400 * len(activation_files), 400)
        ]
    acts = torch.cat(acts, dim=0).to(device)
    acts = acts - torch.mean(acts, dim=0)
    acts = acts / torch.std(acts, dim=0)

    logits = acts[:,:32000] # TODO
    return logits

def compute_x(W, logit):
    x = torch.linalg.lstsq(W, logit.unsqueeze(-1)).solution
    return x.squeeze(-1)


def compute_difference(W, x, logit):
    reconstructed_logit = torch.mm(W, x.unsqueeze(-1)).squeeze(-1)
    d = torch.norm(reconstructed_logit - logit)
    return d


def compute_ratio(differences, e):
    n = torch.sum(differences < e).item()
    return n / len(differences)

def main():
    smodel_tag = 'llama-2-7b'
    dataset_name = 'truthfulqa-300'
    tmodel_tags = [
        "llama-2-7b", 'llama-2-7b-chat', 'vicuna-7b-v1.5', 'vicuna-7b-v1.5-16k','tulu-2-7b', 
        'llama-2-finance-7b','wizardmath-7b',
        'chinese-llama-2-7b','codellama-7b','llemma-7b',
        'Sheared-LLaMA-1.3B-Pruned','Sheared-LLaMA-1.3B', 'Sheared-LLaMA-1.3B-ShareGPT',
        'Sheared-LLaMA-2.7B-Pruned','Sheared-LLaMA-2.7B', 'Sheared-LLaMA-2.7B-ShareGPT',
        'fusellm-7b', 'openllama-2-7b', 
        'llama-3-8b', 'mistral-7b','qwen-7b-v1.5', 'baichuan-2-7b',
        'amberchat','gemma-7b','qwen-7b','falcon-7b','yi-6b','internlm-7b'
        'shisa-gamma-7b', 'wizardmath-7b-1.1', 'abel-7b-002'
        'llama-2-7b', 'openllama-2-7b', 'llama-3-8b', 'mpt-7b'
        'sparsellama-2-7b', 'wandallama-2-7b', 'pruned-50', 'gblmllama-2-7b',
        # 'internlm2-20b-chat', 'mistral-8-7b-it-v0.1', 'qwen-72b-chat-v1.5', 'llama-3-8b',
    ]

    outs = []
    for tmodel_tag in tmodel_tags:
        W = load_victim_logits(smodel_tag)[:32000,:]
        suspect_logits = load_suspect_logits(tmodel_tag, dataset_name)

        # suspect_logits = suspect_logits[:, torch.randperm(suspect_logits.size(-1))] # permutation
        # suspect_logits = 0.8 * suspect_logits # scaling transformation

        print(W.shape, suspect_logits.shape)
        differences = torch.empty(suspect_logits.shape[0], device=device)
        for i, logit in enumerate(tqdm(suspect_logits)):
            x = compute_x(W, logit)
            d = compute_difference(W, x, logit)
            differences[i] = d

        max_difference = torch.max(differences).item()
        min_difference = torch.min(differences).item()
        print(f"distance between {smodel_tag} and {tmodel_tag}: {min_difference:.4f}~{max_difference:.4f}")

        outs.append(f"{min_difference:.4f}~{max_difference:.4f}")

        e_values = [150, 200, 250, 300]
        for e in e_values:
            ratio = compute_ratio(differences, e)
            print(f"Ratio for e={e}: {ratio:.4f}")
    print('\n'.join(outs))

if __name__ == "__main__":
    main()
