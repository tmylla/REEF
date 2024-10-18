import torch
import torch.nn as nn
import torch.nn.functional as F


class Projection(nn.Module):
    def __init__(self, in_dim, num_class):
        super(Projection, self).__init__()
        
        if in_dim==8192:
            turn_dim = 4096
        elif in_dim==4096:
            turn_dim = 8192

        self.fc1 = nn.Linear(in_dim, 5120)
        self.fc2 = nn.Linear(5120, turn_dim)
        self.fc3 = nn.Linear(turn_dim, num_class)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class MLP_Conv(nn.Module):

    def __init__(self, num_class):
        super(MLP_Conv, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.linear = nn.Linear(64, num_class, bias=False)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool1d(x, 1).view(-1, 64) 
        x = self.linear(x)

        return x


class GNN(nn.Module):

    def __init__(self, num_class, k):
        super(GNN, self).__init__()

        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Sequential(nn.Conv2d(1*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.ReLU())
        self.linear = nn.Linear(64*2, num_class, bias=False)

    def knn(self, x, k):
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx


    def get_graph_feature(self, x, k=20, idx=None, dim9=False):

        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            if dim9 == False:
                idx = self.knn(x, k=k).cuda(0) 
            else:
                idx = self.knn(x[:, 6:], k=k)

        device = torch.device('cuda:0')

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)
    
        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous() 
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims) 
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
    
        return feature  

    def forward(self, x):

        x = x.unsqueeze(1)
        batch_size = x.size(0)
        
        x = self.get_graph_feature(x, k=self.k) 
        x = self.conv1(x) 
        x = x.max(dim=-1, keepdim=False)[0] 
       
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)     
        x = torch.cat((x1, x2), 1)  

        
        x = self.linear(x)      

        return x


class ConvNet(nn.Module):

    def __init__(self, num_class):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding = 1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64*64*64, num_class, bias=False)

    def forward(self, x):

        # x: Bx4096
        x = x.view(x.shape[0], 1, 64, 64)

        x = F.relu(self.conv1(x))  
        x = self.flatten(x)
        x = self.linear(x)

        return x

class ConvNet_13B(nn.Module):
    def __init__(self, num_class, input_channels=1):
        super(ConvNet_13B, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64 * 64 * 80, num_class, bias=False)

    def forward(self, x):
        x = x.view(x.shape[0], 1, 64, 80)

        x = F.relu(self.conv1(x)) 
        x = self.flatten(x) 
        x = self.linear(x)  
        return x

class MLP(nn.Module):

    def __init__(self, in_dim, hidd_dim, out_dim, n_layer, bias):
        super(MLP, self).__init__()

        if n_layer < 2:
            raise Exception(f"Invalid #layer: {n_layer}.")
        self.depth = n_layer
        self.layers = self._make_layers(in_dim, hidd_dim, out_dim, self.depth, bias)
        self.in_dim = in_dim

    def _make_layers(self, in_dim, hidd_dim, out_dim, n_layer, bias):
        if bias:
            layers = [nn.Linear(in_dim, hidd_dim), nn.ReLU()]
            for _ in range(n_layer - 2):
                layers.extend([nn.Linear(hidd_dim, hidd_dim), nn.ReLU()])
            layers.append(nn.Linear(hidd_dim, out_dim))
            return nn.Sequential(*layers)
        else:
            layers = [nn.Linear(in_dim, hidd_dim, bias=False), nn.ReLU()]
            for _ in range(n_layer - 2):
                layers.extend([nn.Linear(hidd_dim, hidd_dim, bias=False), nn.ReLU()])
            layers.append(nn.Linear(hidd_dim, out_dim, bias=False))
            return nn.Sequential(*layers)


    def forward(self, x):
        x = x.view(x.size(0), self.in_dim)
        return self.layers(x)

    def get_feature(self, x, layer_num=None):
        '''

        :param x: input tensor
        :param layer_num: the FCL number
        :return: feature after ReLU
        '''
        if layer_num is None:
            out = self.layers(x)
            return out
        elif isinstance(layer_num, str) and layer_num.startswith("FClayer_"):
            layer_num = layer_num.split("_")[-1]
            assert len(layer_num) == 1
            layer = int(layer_num)
            assert layer > 0
            idx = 2 * layer
            out = self.layers[:idx](x)
            return out

    def get_transform(self, x, layer_num):
        '''

        :param x: input tensor
        :param layer_num: the FCL number
        :return: transform
        '''
        if isinstance(layer_num, str) and layer_num.startswith("FClayer_"):
            layer_num = layer_num.split("_")[-1]
            assert len(layer_num) == 1
            layer = int(layer_num)
            assert layer > 0 and layer < self.depth
            idx = 2 * layer
            out = self.layers[:idx - 1](x)
            out = (out > 0).float()
            return out


if __name__ == '__main__':
    net = MLP(in_dim=10, hidd_dim=128, out_dim=2, n_layer=8)
    print(str(net))
    x = torch.randn(10, 10)
    feature = net.get_feature(x, layer_num=f"FClayer_{2}")
