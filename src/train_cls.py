import os
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from utils import time_record, IOStream, accuracy, save_obj

from generate_activations import load_acts
import torch.utils.data as Data
import torch.nn as nn
from model import MLP, MLP_Conv, ConvNet, GNN, Projection, ConvNet_13B
import os.path as osp
from tqdm import tqdm
import utils

import warnings
warnings.filterwarnings('ignore')


class Trainer(nn.Module):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.width = args.width
        self.depth = args.depth
        self.device = args.device
        self.iterations = args.iterations
        self.batch_size = args.batch_size
        self.num_classes = args.classes

        self.train_loader, self.test_loader = train_loader, test_loader

        if args.detector_type == "MLP":
            self.net = MLP(in_dim=args.embed_dim, hidd_dim=args.width, out_dim=self.num_classes, n_layer=args.depth, bias = args.bias).to(
            self.device)
        elif args.detector_type == "linear":
            self.net = nn.Linear(args.embed_dim, 2, bias=True).to(self.device)

        elif args.detector_type == "MLP_Conv":
            self.net = MLP_Conv(num_class=self.num_classes).to(self.device)

        elif args.detector_type == "ConvNet_13B":
            self.net = ConvNet_13B(num_class=self.num_classes).to(self.device)

        elif args.detector_type == "ConvNet":
            self.net = ConvNet(num_class=self.num_classes).to(self.device)

        elif args.detector_type == "GNN":
            self.net = GNN(num_class=self.num_classes, k= 10).to(self.device)
        
        elif args.detector_type == "Projection":
            self.net = Projection(in_dim=args.embed_dim, num_class=self.num_classes).to(self.device)

        else:

            raise Exception(f"Invalid type: {args.detector_type}.")


        
        print(self.net)

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), betas = (0.9, 0.999), lr=args.lr, weight_decay=0)
        self.save_freq = args.save_freq
        self.eval_freq = args.eval_freq
        self.iterations = args.iterations
        self.save_dir = osp.join(save_dir, "layer-" + str(args.current_layer), f"{args.detector_type}-{args.width}-{args.depth}-{args.lr}-{args.batch_size}-{args.iterations}_w_bias", str(time_stamp))
        self.iter = 0
        self.epoch = 0
        

        os.makedirs(self.save_dir)
        self.io = IOStream(self.save_dir + '/run.log')

        os.system(f"cp {os.path.abspath(__file__)} {self.save_dir}/train_backup.backup")


        os.makedirs(osp.join(self.save_dir, "curve"))
        self.generate_plot_dict()

    def generate_plot_dict(self):
        self.plot_dict = {
            "training_loss": [],
            "training_acc": [],
            "eval_test_loss": [],
            "eval_test_acc": [],
            "eval_train_loss": [],
            "eval_train_acc": []
        }

    def run(self):
        while True:
            self.io.cprint(f"EPOCH {self.epoch + 1}")
            self.epoch += 1
            pbar = tqdm(self.train_loader, mininterval=1, ncols=100)
            pbar.set_description("training")
            for i, (image, label) in enumerate(pbar):

                pbar.set_description(f"eval epoch{self.epoch} iter[{self.iter + 1}/{self.iterations}]")
                
                if self.iter % args.eval_freq == 0:
                    test_acc, test_loss = self.eval(test=True)

                    outstr = 'Train iteration %d, test loss: %.6f, test acc: %.6f' % (self.iter, test_loss, test_acc)
                    self.io.cprint(outstr)

                    self.plot_dict["eval_test_loss"].append(test_loss)
                    self.plot_dict["eval_test_acc"].append(test_acc)

                    train_acc, train_loss = self.eval(test=False)
                    outstr = 'Train iteration %d, train loss: %.6f, train acc: %.6f' % (self.iter, train_loss, train_acc)
                    self.io.cprint(outstr)

                    self.plot_dict["eval_train_loss"].append(train_loss)
                    self.plot_dict["eval_train_acc"].append(train_acc)

                if self.iter % self.save_freq == 0:
                    self.save()

                pbar.set_description(f"training epoch{self.epoch} iter[{self.iter + 1}/{self.iterations}]")
                self.net.train()
                image, label = image.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()
                out = self.net(image)
                loss = self.criterion(out, label)
                loss.backward()
                acc = accuracy(out, label)
                self.plot_dict["training_loss"].append(loss.item())
                self.plot_dict["training_acc"].append(acc.item())
                pbar.set_postfix_str("loss={:.4f} acc={:.2f}%".format(loss.item(), acc.item()))
                self.optimizer.step()
                if self.iter > self.iterations:
                    test_acc, test_loss = self.eval(test=True)
                    self.plot_dict["eval_test_loss"].append(test_loss)
                    self.plot_dict["eval_test_acc"].append(test_acc)
                    train_acc, train_loss = self.eval(test=False)
                    self.plot_dict["eval_train_loss"].append(train_loss)
                    self.plot_dict["eval_train_acc"].append(train_acc)
                    torch.save({
                        'model_state_dict': self.net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, osp.join(self.save_dir, f"model.pt"))
                    save_obj(self.plot_dict, osp.join(self.save_dir, "curve", "data.bin"))
                    break
 
                self.iter += 1
                self.plot()

            if self.iter > self.iterations:
                break

    def plot(self):
        plt.figure(figsize=(6, 4))
        plt.subplot(1, 2, 1)
        plt.xlabel("iteration")
        plt.plot(list(range(1, len(self.plot_dict["eval_test_loss"]) * self.eval_freq + 1, self.eval_freq)),
                 self.plot_dict["eval_test_loss"],
                 label="evaluate test loss")
        plt.plot(list(range(1, len(self.plot_dict["eval_train_loss"]) * self.eval_freq + 1, self.eval_freq)),
                 self.plot_dict["eval_train_loss"],
                 label="evaluate train loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.xlabel("iteration")
        plt.plot(list(range(1, len(self.plot_dict["eval_test_acc"]) * self.eval_freq + 1, self.eval_freq)),
                 self.plot_dict["eval_test_acc"],
                 label="evaluate test acc")
        plt.plot(list(range(1, len(self.plot_dict["eval_train_acc"]) * self.eval_freq + 1, self.eval_freq)),
                 self.plot_dict["eval_train_acc"],
                 label="evaluate train acc")
        plt.legend()
        plt.tight_layout()
        plt.savefig(osp.join(self.save_dir, "curve", "curve.png"), dpi=300)
        plt.close("all")

    def eval(self, test=False):
        self.net.eval()
        total_size = 0
        total_loss = 0
        total_acc = 0
        if test:
            dataloader = self.test_loader
        else:
            dataloader = self.train_loader
        for image, label in dataloader:
            # loop over dataset
            image, label = image.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            out = self.net(image)
            loss = self.criterion(out, label)
            prec = accuracy(out, label)
            bs = image.size(0)
            total_size += int(bs)
            total_loss += float(loss) * bs
            total_acc += float(prec) * bs

        loss, acc = total_loss / total_size, total_acc / total_size

        return acc, loss

    def save(self):
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, osp.join(self.save_dir, f"model_{self.iter}.pt"))

        save_obj(self.plot_dict, osp.join(self.save_dir, "curve", "data.bin"))



parser = argparse.ArgumentParser(description="Train probing for activations")
parser.add_argument("--model", default="llama-2-7b",
                    help="Size of the model to use. Options are 7B or 30B")
parser.add_argument("--layers", nargs='+', type=int, default=[18, 20, 22],
                    help="Layers to save embeddings from, -1 denotes all layers")
parser.add_argument("--datasets", nargs='+', default=['truthfulqa'], 
                    help="Names of datasets, without .csv extension")
parser.add_argument("--output_dir", default="activations",
                    help="Directory to save activations to")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--frommodel", default="llama-2-7b-hf")
parser.add_argument("--fromlayer", default=20)
parser.add_argument("--load_probes", action='store_true')

parser.add_argument("--detector_type", default="MLP_Conv", choices=['linear','MLP','MLP_Conv', 'ConvNet', 'ConvNet_13B', "GNN", 'Projection', 'GCN'],
                    help="type of the model to detector")
parser.add_argument('--embed_dim', default=4096, type=int)
parser.add_argument('--path', default='results', type=str)
parser.add_argument('--iterations', default=1500, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--seed', default=2024, type=int)
parser.add_argument('--depth', default=2, type=int)
parser.add_argument('--width', default=256, type=int, help='width of fully connected layers')
parser.add_argument("--eval_freq", default=20, type=int)
parser.add_argument('--save_freq', default=300, type=int)
parser.add_argument('--classes', default=2, type=int)
parser.add_argument('--bias', default=True, type=bool)
parser.add_argument('--kernel_size', default=3, type=int)
parser.add_argument('--stride_size', default=1, type=int)
parser.add_argument('--padding_size', default=1, type=int)
args = parser.parse_args()

model_tag = args.model
device = args.device

dataset_name = f'{args.datasets[0]}'
save_dir = f"results/{args.model}/{dataset_name}/"
os.makedirs(save_dir, exist_ok=True)

all_layer_acts = []
layers = [int(layer) for layer in args.layers]
if layers == [-1]:
    layers = list(range(utils.LLM_LAYERS_MAP.get(model_tag, 32)))

for layer in layers:    
    acts = load_acts(dataset_name, model_tag, layer=layer, center=True, scale=True, device=args.device, acts_dir=args.output_dir)
    all_layer_acts.append(acts)

all_layer_acts = torch.stack(all_layer_acts).to(args.device) 

num_layers = all_layer_acts.shape[0]
num_prompts = all_layer_acts.shape[1]
hidden_dim = all_layer_acts.shape[2]
EPOCH = 1000
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

print(train_activations.shape)


for layer in range(len(layers)):
    time_stamp = time_record()
    train_set = Data.TensorDataset(train_activations[layer], train_targets[layer])
    train_loader = Data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_set = Data.TensorDataset(test_activations[layer], test_targets[layer])
    test_loader = Data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    args.current_layer = layers[layer]

    trainer = Trainer(args)
    trainer.run()




