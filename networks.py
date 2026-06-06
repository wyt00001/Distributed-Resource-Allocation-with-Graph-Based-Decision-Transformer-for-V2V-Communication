import torch
import torch_geometric.nn
#Only for example
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class GCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCN, self).__init__()
        self.gcn1 = torch_geometric.nn.GCNConv(input_size, hidden_size, normalize=False)
        self.relu = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        out = self.gcn1(x, edge_index)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, output_size, num_layers=3, heads=4):
        super(GAT, self).__init__()
        self.gat1 = torch_geometric.nn.GAT(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, v2=True, dropout=0, act=torch.nn.LeakyReLU(), edge_dim=1, heads=heads)
        self.relu = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(hidden_channels, output_size)

    def forward(self, x, edge_index, edge_attr=None):
        out = self.gat1(x, edge_index, edge_attr=edge_attr)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Qmix(torch.nn.Module):
    def __init__(self, num_cars):
        super(Qmix, self).__init__()
        self.w = torch.nn.Parameter(torch.ones(num_cars))
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return torch.sum(x * self.relu(self.w))
