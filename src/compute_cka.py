import os
import argparse
from utils import seed_torch
from generate_activations import load_acts
import numpy as np
import math
import torch


class CKA(object):
    def __init__(self):
        pass 
    
    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H) 

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX
 
    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)

    
class CudaCKA(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)


if __name__ == "__main__":
    """
    Compute the CKA of the base model (victim model) and the test model (suspect model)
    """
    parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
    parser.add_argument("--base_model", default="llama-2-7b")
    parser.add_argument("--base_layers", nargs='+', type=int, default=[18], help="Layers to load activations; can also set 24/32/40/80 to load all layers' activations according to the model size")
    parser.add_argument("--test_model", default="vicuna-7b-v1.5")
    parser.add_argument("--test_layers", nargs='+', type=int, default=[18], help="Layers to load activations; can also set 24/32/40/80 to load all layers' activations according to the model size")

    parser.add_argument("--datasets", nargs='+', default=['truthfulqa-200'], help="Names of datasets, without .csv extension")
    parser.add_argument("--output_dir", default="activations", help="Directory to save activations to")
    parser.add_argument("--device", default="cuda")

    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--permutation', action='store_true')
    parser.add_argument('--scaling', action='store_true')
    parser.add_argument('--l2', action='store_true')  
    parser.add_argument('--noise', action='store_true') 

    parser.add_argument("--load_in_8bit", action='store_true')
    parser.add_argument("--load_in_4bit", action='store_true')
    args = parser.parse_args()


    device = args.device
    seed_torch(args.seed)
    dataset_name = f'{args.datasets[0]}'

    if args.base_layers[0] in [24,32,40,80]:
        base_layers = list(range(args.base_layers[0]))
    else:
        base_layers = [int(layer) for layer in args.base_layers]

    test_layers = list(range(args.test_layers[0]))

    base_layer_acts = []
    for layer in base_layers:    
        acts = load_acts(dataset_name, args.base_model, layer=layer, center=True, scale=True, device=args.device, acts_dir=args.output_dir)
        base_layer_acts.append(acts)

    base_layer_acts = torch.stack(base_layer_acts).to(args.device) # [layer_num, data_num, feature_dim]


    test_model = args.test_model
    if args.load_in_8bit:
        test_model = test_model + '-8bit'
    if args.load_in_4bit:
        test_model = test_model + '-4bit'

    cka_matrix = np.zeros((len(base_layers), len(test_layers)))
    for base_idx in range(len(base_layers)):
        X = base_layer_acts[base_idx] 
        
        for test_idx, layer in enumerate(test_layers): 
            Y = load_acts(dataset_name, test_model, layer=layer, center=True, scale=True, device=args.device, acts_dir=args.output_dir) 
            if args.permutation:
                print('---------------------------')
                Y = Y[:, torch.randperm(Y.size(-1))]

            if args.scaling:
                print('---------------------------')
                alpha = 0.8
                Y = alpha * Y

            if args.l2:
                norm = torch.norm(Y, p=2, dim=1, keepdim=True)
                Y =  Y / (norm + 1e-8) 
            if args.noise:
                noise = torch.randn_like(Y) * 1. + 0.
                Y =  Y + noise
            

            cuda_cka = CudaCKA(args.device)
            cka_value = cuda_cka.linear_CKA(X, Y)
            # cka_value = cuda_cka.kernel_CKA(X, Y)
            cka_matrix[base_idx, test_idx] = cka_value

            print(f'X: {args.base_model}-{base_layers[base_idx]}layer\tY: {test_model}-{layer}layer')
            print('Linear CKA, between X and Y: {}'.format(cka_value))
            
    save_dir = f'results/cka-matrix/{dataset_name}'
    os.makedirs(save_dir, exist_ok=True)

    if args.permutation:
        np.save(f'{save_dir}/{args.base_model}_{test_model}-permutation.npy', cka_matrix)
    elif args.scaling:
        np.save(f'{save_dir}/{args.base_model}_{test_model}-scaling.npy', cka_matrix)
    else:
        np.save(f'{save_dir}/{args.base_model}_{test_model}.npy', cka_matrix)
                

