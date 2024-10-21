import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
import pandas as pd
from tqdm import tqdm
import os
from glob import glob
from utils import *

ACTS_BATCH_SIZE = 400


class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        self.out = module_outputs 

def load_model(model_size, device, model_tag='llama-2-7b'):
    model_path = get_model_save_path(model_tag)
    print(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True) 
    if args.load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit = True),
            device_map="auto"
            )
    elif args.load_in_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
            device_map="auto"
            )
    elif args.model=='nmer1':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
            ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            ).to(device)
        
    return tokenizer, model

def load_statements(dataset_name):
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(f"datasets/{dataset_name}.csv")
    statements = dataset['statement'].tolist()
    return statements

def get_acts(statements, tokenizer, model, layers, device, token_pos=-1):
    """
    Get given layer activations for the statements. 
    Return dictionary of stacked activations.

    token_pos: default to fetch the last token's activations
    """
    # attach hooks
    hooks, handles = [], []
    for layer in layers:
        hook = Hook()
        
        if model_tag=='mpt-7b':
            handle = model.transformer.blocks[layer].register_forward_hook(hook) # mpt-7b
        elif model_tag=='falcon-7b':
            handle = model.transformer.h[layer].register_forward_hook(hook) # falcon-7b
        else:
            handle = model.model.layers[layer].register_forward_hook(hook)

        hooks.append(hook), handles.append(handle)
    
    # get activations
    acts = {layer : [] for layer in layers}
    for statement in tqdm(statements):
        input_ids = tokenizer.encode(statement, return_tensors="pt").to(device=model.device)
        model(input_ids)
        for layer, hook in zip(layers, hooks):
            acts[layer].append(hook.out[0][0, token_pos])
    
    # stack len(statements)'s activations
    for layer, act in acts.items():
        acts[layer] = torch.stack(act).float()
    
    # remove hooks
    for handle in handles:
        handle.remove()
    
    return acts


def load_acts(dataset_name, model_tag, layer, center=True, scale=False, device='cpu', acts_dir='activations'):
    """
    Collects activations from a dataset of statements, returns as a tensor of shape [n_activations, activation_dimension].
    """

    directory = os.path.join(PROJECT_ROOT, acts_dir, model_tag, dataset_name)
    activation_files = glob(os.path.join(directory, f'layer_{layer}_*.pt'))
    if device=='cpu':
        acts = [torch.load(os.path.join(directory, f'layer_{layer}_{i}.pt'), map_location=torch.device('cpu')) for i in range(0, ACTS_BATCH_SIZE * len(activation_files), ACTS_BATCH_SIZE)]
    else:
        acts = [
            torch.load(os.path.join(directory, f'layer_{layer}_{i}.pt'), 
                    map_location=lambda storage, loc: storage.cuda(device))
            for i in range(0, ACTS_BATCH_SIZE * len(activation_files), ACTS_BATCH_SIZE)
        ]
    acts = torch.cat(acts, dim=0).to(device)
    if center:
        acts = acts - torch.mean(acts, dim=0)
    if scale:
        acts = acts / torch.std(acts, dim=0)
    return acts


if __name__ == "__main__":
    """
    read statements from dataset, record activations in given layers, and save to specified files
    """
    parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
    parser.add_argument("--model", default="llama-2-7b")
    parser.add_argument("--layers", nargs='+',help="Layers to save embeddings from")
    parser.add_argument("--datasets", nargs='+',help="Names of datasets, without .csv extension")
    parser.add_argument("--output_dir", default="activations",help="Directory to save activations to")
    parser.add_argument("--load_in_8bit", action='store_true')
    parser.add_argument("--load_in_4bit", action='store_true')
    parser.add_argument("--downsample", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(args)

    model_tag = args.model

    torch.set_grad_enabled(False)

    ### generate acts 
    tokenizer, model = load_model(args.model, args.device, model_tag=model_tag)
    for dataset in args.datasets:
        if args.downsample == 42:
            statements = load_statements(dataset)
        else:
            statements = load_statements(dataset)[:args.downsample]

        layers = [int(layer) for layer in args.layers]
        if layers == [-1]:
            try:
                layers = list(range(len(model.model.layers)))
            except:
                if '7b' in model_tag or '8b' in model_tag:
                    layers = list(range(32))
                else:
                    layers = list(range(40))

        if args.downsample != 42:
            dataset = f'{dataset}-{args.downsample}'

        if args.load_in_8bit:
            save_dir = f"{args.output_dir}/{model_tag}-8bit/{dataset}/"
        elif args.load_in_4bit:
            save_dir = f"{args.output_dir}/{model_tag}-4bit/{dataset}/"
        else:
            save_dir = f"{args.output_dir}/{model_tag}/{dataset}/"
        os.makedirs(save_dir, exist_ok=True)

        # reduce the load of each file
        for idx in range(0, len(statements), ACTS_BATCH_SIZE):
            acts = get_acts(statements[idx:idx + ACTS_BATCH_SIZE], tokenizer, model, layers, args.device)
            for layer, act in acts.items():
                torch.save(act, f"{save_dir}/layer_{layer}_{idx}.pt")

    