import torch
import argparse
import pandas as pd
import numpy as np
from utils import time_record, IOStream, accuracy, save_obj

from generate_activations import load_acts
import torch.utils.data as Data
import torch.nn as nn
from model import MLP, ConvNet, GCN, Projection, ConvNet_13B
import utils
from torch.utils.data import Dataset


import warnings
warnings.filterwarnings('ignore')


def create_adjacency_matrix(features, threshold):
    distances = euclidean_distances(features.cpu().numpy())
    adjacency_matrix = np.where(distances < threshold, 1, 0)
    np.fill_diagonal(adjacency_matrix, 0)
    return adjacency_matrix

class GraphDataset(Dataset):
    def __init__(self, features, adjacency_matrix, labels):
        self.features = torch.tensor(features, dtype=torch.float32).to(args.device)
        self.adjacency_matrix = adjacency_matrix
        self.labels = torch.tensor(labels, dtype=torch.long).to(args.device)
        self.edge_index = torch.tensor(np.array(np.nonzero(self.adjacency_matrix)), dtype=torch.long).to(args.device)


    def __len__(self):
        return 1
    def __getitem__(self, index):
        return Data(x=self.features, edge_index=self.edge_index, y=self.labels)


parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
parser.add_argument("--model", default="llama-2-7b",
                    help="Size of the model to use. Options are 7B or 30B")
parser.add_argument("--layers", nargs='+', type=int, default=[18],
                    help="Layers to save embeddings from, -1 denotes all layers")
parser.add_argument("--datasets", default = "truthfulqa", help="Names of datasets, without .csv extension")
parser.add_argument("--output_dir", default="activations",
                    help="Directory to save activations to")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--frommodel", default="llama-2-7b")
parser.add_argument("--fromlayer", default=18, type=int)
parser.add_argument("--load_probes", action='store_true')

parser.add_argument("--detector_type", default="linear", help="type of the model to detector")
parser.add_argument('--embed_dim', default=4096, type=int)
parser.add_argument('--path', default='results', type=str)
parser.add_argument('--iterations', default=1500, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--seed', default=2024, type=int)
parser.add_argument('--depth', default=2, type=int)
parser.add_argument('--width', default=256, type=int, help='width of fully connected layers')
parser.add_argument('--hidden_dim', default=64, type=int, help='dimension of hidden_dim')
parser.add_argument('--threshold', default=1, type=float)
parser.add_argument("--eval_freq", default=20, type=int)
parser.add_argument('--save_freq', default=200, type=int)
parser.add_argument('--classes', default=2, type=int)
parser.add_argument('--bias', default=True, type=bool)
parser.add_argument('--attack', default=False, type=bool)
parser.add_argument('--dropout', default=False, type=bool)
parser.add_argument("--pretrain_dir", default="/your_path/",help="Directory to save activations to")

parser.add_argument("--load_in_8bit", action='store_true')
parser.add_argument("--load_in_4bit", action='store_true')

args = parser.parse_args()
print(args)
device = args.device

print(f"{args.frommodel}-{args.detector_type} trained on {args.datasets}")

dataset_name = f'{args.datasets}'

if args.detector_type == "MLP":
    Net = MLP(in_dim=args.embed_dim, hidd_dim=args.width, out_dim=args.classes, n_layer=args.depth, bias = args.bias).to(args.device)
elif args.detector_type == "linear":
    Net = nn.Linear(args.embed_dim, 2, bias=True).to(args.device)
elif args.detector_type == "ConvNet":
    Net = ConvNet(num_class=2).to(args.device)
elif args.detector_type == "ConvNet_13B":
    Net = ConvNet_13B(num_class=2).to(args.device)
elif args.detector_type == "GCN":
    Net = GCN(args).to(args.device)
elif args.detector_type == "Projection":
    Net = Projection(in_dim=args.embed_dim, num_class=2).to(args.device)
else:
    raise Exception(f"Invalid type: {args.detector_type}.")

print(Net)
checkpoint = torch.load(args.pretrain_dir, map_location='cuda')
Net.load_state_dict(checkpoint['model_state_dict'])
Net.eval()



model_list=[
    'llama-2-7b', 'llama-2-7b-chat', 'vicuna-7b-v1.5', 'chinese-llama-2-7b', 'xwinlm-7b', 
    'mistral-7b','baichuan-2-7b', 'qwen-7b-v1.5', 'internlm-7b',
    # 'llama-2-13b', 'llama-2-13b-chat', 'vicuna-13b', 'chinesellama-2-13b', 'xwinlm-13b', 
    # 'plamo-13b', 'baichuan-2-13b', 'qwen-14b-v1.5', #'internlm-20b-chat', 
    ]


for model_tag in model_list:
    if args.load_in_8bit:
        model_tag = model_tag + '-8bit'
    if args.load_in_4bit:
        model_tag = model_tag + '-4bit'

    layers = [int(layer) for layer in args.layers]
    if layers == [-1]:
        layers = list(range(utils.LLM_LAYERS_MAP.get(model_tag, 32)))

    for layer in layers: 
        all_layer_acts = []  
        acts = load_acts(dataset_name, model_tag, layer=layer, center=True, scale=True, device=args.device, acts_dir=args.output_dir)
        all_layer_acts.append(acts)

        all_layer_acts = torch.stack(all_layer_acts).to(args.device) 
        
        num_layers = all_layer_acts.shape[0]
        num_prompts = all_layer_acts.shape[1]
        hidden_dim = all_layer_acts.shape[2]
        train_ratio = 0.8  
        num_train = int(num_prompts * train_ratio)
        num_test = num_prompts - num_train


        targets = pd.read_csv(f"datasets/{args.datasets}.csv", dtype={'label': int})['label'].to_numpy()
        targets = torch.tensor(targets).long().to(args.device)
        targets = targets.unsqueeze(0).repeat(num_layers, 1) 

        test_activations = all_layer_acts[:, num_train:, :]
        test_targets = targets[:, num_train:]

        if args.attack:
            test_activations = test_activations[:, :, torch.randperm(test_activations.size(-1))]

        total_size = 0
        total_acc = 0

        if args.detector_type != "GCN":
            test_set = Data.TensorDataset(test_activations[0], test_targets[0])
            test_loader = Data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

            for image, label in test_loader:
                out = Net(image)
                prec = accuracy(out, label)
                bs = image.size(0)
                total_size += int(bs)
                total_acc += float(prec) * bs

            acc = total_acc / total_size

        else:
            from torch_geometric.data import Data, DataLoader as GeoDataLoader
            from torch_geometric.nn import GCNConv
            from sklearn.metrics.pairwise import euclidean_distances
            from sklearn.model_selection import train_test_split

            X_test, y_test = test_activations[0], test_targets[0]

            adjacency_matrix_test = create_adjacency_matrix(X_test, args.threshold)

            test_dataset = GraphDataset(X_test, adjacency_matrix_test, y_test)

            test_loader = GeoDataLoader([test_dataset[0]], batch_size=1, shuffle=False)

            correct = 0
            with torch.no_grad():
                for data in test_loader:
                    out = Net(data)
                    pred = out.argmax(dim=1)
                    correct += pred.eq(data.y).sum().item()

            acc = correct / len(test_loader.dataset[0].y)

        print(round(acc/100,4))

