import os
import argparse
import pandas as pd
from utils import time_record, IOStream, accuracy, save_obj, seed_torch

from generate_activations import load_acts
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import os.path as osp

import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split



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


class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()

        self.hidden_dim = args.width
        self.output_dim = args.classes
        self.dropout = args.dropout

        self.conv1 = GCNConv(4096, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if self.dropout:
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index)
        if self.dropout:
            x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)



def train(loader):
    model.train()
    total_loss = 0
    correct = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset[0].y)

def validate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            out = model(data)
            loss = F.nll_loss(out, data.y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset[0].y)


parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
parser.add_argument("--model", default="llama-2-7b",
                    help="Size of the model to use. Options are 7B or 30B")
parser.add_argument("--layers", nargs='+', type=int, default=[18],
                    help="Layers to save embeddings from, -1 denotes all layers")
parser.add_argument("--datasets", nargs='+',
                    help="Names of datasets, without .csv extension")
parser.add_argument("--output_dir", default="activations",
                    help="Directory to save activations to")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--frommodel", default="llama-2-7b")
parser.add_argument("--fromlayer", default=18)
parser.add_argument("--load_probes", action='store_true')
parser.add_argument("--embed_dim",default=4096)

parser.add_argument("--detector_type", default="GCN", choices=['linear','MLP','MLP_Conv', 'ConvNet', "GNN", "GCN"],
                    help="type of the model to detector")
parser.add_argument('--path', default='results', type=str)
parser.add_argument('--iterations', default=200, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--seed', default=2024, type=int)
parser.add_argument('--depth', default=2, type=int)
parser.add_argument('--width', default=64, type=int, help='dimension of hidden_dim')
parser.add_argument("--eval_freq", default=10, type=int)
parser.add_argument('--save_freq', default=20, type=int)
parser.add_argument('--classes', default=2, type=int)
parser.add_argument('--bias', default=True, type=bool)
parser.add_argument('--dropout', default=False, type=bool)
parser.add_argument('--kernel_size', default=3, type=int)
parser.add_argument('--stride_size', default=1, type=int)
parser.add_argument('--padding_size', default=1, type=int)
parser.add_argument('--threshold', default=1, type=float)
args = parser.parse_args()


model_tag = args.model
device = args.device
seed_torch(args.seed)

dataset_name = f'{args.datasets[0]}'
save_dir = f"results/{args.model}/{dataset_name}/"
os.makedirs(save_dir, exist_ok=True)

all_layer_acts = []
layers = [int(layer) for layer in args.layers]

for layer in layers:    
    acts = load_acts(dataset_name, model_tag, layer=layer, center=True, scale=True, device=args.device, acts_dir=args.output_dir)
    all_layer_acts.append(acts)

all_layer_acts = torch.stack(all_layer_acts).to(args.device) 

num_layers = all_layer_acts.shape[0]
num_prompts = all_layer_acts.shape[1]
hidden_dim = all_layer_acts.shape[2]
train_ratio = 0.8 
num_train = int(num_prompts * train_ratio)
num_test = num_prompts - num_train

targets = pd.read_csv(f"datasets/{args.datasets[0]}.csv", dtype={'label': int})['label'].to_numpy()
targets = torch.tensor(targets).long().to(args.device)
targets = targets.unsqueeze(0).repeat(num_layers, 1)

train_activations = all_layer_acts[:, :num_train, :]
test_activations = all_layer_acts[:, num_train:, :]
train_targets = targets[:, :num_train]
test_targets = targets[:, num_train:]


for layer in range(len(layers)):
    time_stamp = time_record()
    args.current_layer = layers[layer]

    save_dir_tmp = osp.join(save_dir, "layer-" + str(args.current_layer), f"{args.detector_type}-{args.lr}-{args.width}-{args.threshold}-{args.iterations}-{args.dropout}", str(time_stamp))
    os.makedirs(save_dir_tmp)
    io = IOStream(save_dir_tmp + '/run.log')

    os.system(f"cp {os.path.abspath(__file__)} {save_dir_tmp}/train_backup.backup")
 
    X_train, y_train = train_activations[layer], train_targets[layer]
    X_test, y_test = test_activations[layer], test_targets[layer]

    adjacency_matrix_train = create_adjacency_matrix(X_train, args.threshold)
    adjacency_matrix_test = create_adjacency_matrix(X_test, args.threshold)

    train_dataset = GraphDataset(X_train, adjacency_matrix_train, y_train)
    test_dataset = GraphDataset(X_test, adjacency_matrix_test, y_test)

    train_loader = GeoDataLoader([train_dataset[0]], batch_size=1, shuffle=True)
    test_loader = GeoDataLoader([test_dataset[0]], batch_size=1, shuffle=False)

    model = GCN(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas = (0.9, 0.999))


    for epoch in range(args.iterations):
        train_loss, train_acc = train(train_loader)
        test_loss, test_acc = validate(test_loader)
        outstr = 'Train Epoch %d, train loss: %.6f, train acc: %.6f, test loss: %.6f, test acc: %.6f' % (epoch, train_loss, train_acc, test_loss, test_acc)
        io.cprint(outstr)
        torch.save({'model_state_dict': model.state_dict()}, osp.join(save_dir_tmp, f"model_{epoch}.pt"))