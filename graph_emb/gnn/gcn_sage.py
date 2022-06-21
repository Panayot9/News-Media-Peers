import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv
import sys
sys.path.append('../')


class Gcn(torch.nn.Module):
    def __init__(self, num_features, dim=16, num_classes=1, num_layers=2, model_type='gcn'):
        super(Gcn, self).__init__()

        self.conv1 = SAGEConv(num_features, dim) if model_type == 'sage' else GCNConv(num_features, dim)
        self.gcs = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(1, num_layers-1):
            conv = SAGEConv(dim, dim) if model_type == 'sage' else GCNConv(dim, dim)
            self.gcs.append(conv)
        self.conv2 = SAGEConv(dim, num_classes) if model_type == 'sage' else GCNConv(dim, num_classes)

    def forward(self, x, edge_index, data=None, save_embedding=False):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        for i in range(1, self.num_layers-1):
            x = F.relu(self.gcs[i-1](x, edge_index))
            x = F.dropout(x, training=self.training)
        if save_embedding: 
            return x
        x = self.conv2(x, edge_index)

        pred = F.log_softmax(x, dim=-1) 
        #print("[pred]", pred)
        return pred