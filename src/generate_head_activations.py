
import os
import torch
import argparse
import pandas as pd
import configparser
from glob import glob
from tqdm import tqdm
from utils import get_model_save_path

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

ACTS_BATCH_SIZE = 400
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        self.out = module_outputs  

def load_model(device, model_tag='llama-2-7b'):
    model_path = get_model_save_path(model_tag)

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
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            ).to(device)


    print(model)
    return tokenizer, model


def load_statements(dataset_name):
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(f"datasets/{dataset_name}.csv")
    statements = dataset['statement'].tolist()
    return statements


def get_acts(statements, tokenizer, model, device, token_pos=-1):
    """
    Get given layer activations for the statements. 
    Return dictionary of stacked activations.

    token_pos: default to fetch the last token's activations
    """
    hook = Hook()
    handle = model.lm_head.register_forward_hook(hook)

    acts = []
    for statement in tqdm(statements):
        input_ids = tokenizer.encode(statement, return_tensors="pt").to(device)
        model(input_ids)
        acts.append(hook.out[0][token_pos])

    acts = torch.stack(acts).float()

    handle.remove()
    
    return acts


def load_acts(dataset_name, model_tag, center=True, scale=False, device='cpu', acts_dir='activations-lmhead'):
    """
    Collects activations from a dataset of statements, returns as a tensor of shape [n_activations, activation_dimension].
    """
    directory = os.path.join(PROJECT_ROOT, acts_dir, dataset_name)
    print(directory)
    activation_files = glob(os.path.join(directory, f'{model_tag}_*.pt'))
    acts = [torch.load(os.path.join(directory, f'{model_tag}_{i}.pt')).to(device) for i in range(0, ACTS_BATCH_SIZE * len(activation_files), ACTS_BATCH_SIZE)]
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
    parser.add_argument("--model", default="llama-2-7b",
                        help="Name of model")
    parser.add_argument("--datasets", nargs='+', default=['truthfulqa'], help="Names of datasets, without .csv extension")
    parser.add_argument("--output_dir", default="activations-lmhead", help="Directory to save activations to")
    parser.add_argument("--load_in_8bit", action='store_true')
    parser.add_argument("--load_in_4bit", action='store_true')
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    model_tag = args.model

    torch.set_grad_enabled(False)

    
    tokenizer, model = load_model(args.device, model_tag=model_tag)

    for dataset in args.datasets:
        statements = load_statements(dataset)
        statements = statements[:300]

        save_dir = f"{args.output_dir}/{dataset}-300/"
        os.makedirs(save_dir, exist_ok=True)

        for idx in range(0, len(statements), ACTS_BATCH_SIZE):
            acts = get_acts(statements[idx:idx + ACTS_BATCH_SIZE], tokenizer, model, args.device)
            print(acts.shape)
            torch.save(acts, f"{save_dir}/{model_tag}_{idx}.pt")