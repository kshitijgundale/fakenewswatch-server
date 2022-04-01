import torch
from torch_geometric.nn import global_mean_pool
from torch.nn import ModuleList, Linear

class GNN(torch.nn.Module):

  def __init__(self, gnn_channels, fcnn_channels, conv, mean, std, loss_f=None):
    super(GNN, self).__init__()
    torch.manual_seed(12345)
    self.gnn_layers = ModuleList()
    self.fcnn_layers = ModuleList()
    self.conv = conv
    self.mean = mean
    self.std = std

    self.loss_f = loss_f

    for i in range(len(gnn_channels)-1):
      self.gnn_layers.append(self.conv(gnn_channels[i], gnn_channels[i+1]))
    for i in range(len(fcnn_channels) - 1):
      self.fcnn_layers.append(Linear(fcnn_channels[i], fcnn_channels[i+1]))

  def forward(self, x, edge_index, batch):

    x = (x - self.mean)/self.std

    for n,l in enumerate(self.gnn_layers):
      if n != len(self.gnn_layers) - 1:
        x = l(x, edge_index).relu()
      else:
        x = l(x, edge_index)
    x = global_mean_pool(x, batch)
    
    for n,l in enumerate(self.fcnn_layers):
      if n != len(self.fcnn_layers) - 1:
        x = l(x).relu()
      else:
        x = l(x)

    return x  